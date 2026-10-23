// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::codestream::WaveletTransform;
use super::super::rect::IntRect;
use super::filter_common::{
    floor_div_i64, periodic_symmetric_extension_left, periodic_symmetric_extension_right,
};
use crate::math::{self, dispatch, f32x8, Level, Simd, SIMD_WIDTH};
use j2k_codec_math::dwt;

/// One vertical lifting step, applied to every row of one parity from the
/// rows above and below it.
#[derive(Clone, Copy, Debug)]
pub(super) enum VerticalStep {
    /// Equation (F-5).
    Reversible53Predict,
    /// Equation (F-6).
    Reversible53Update,
    /// Steps 3 to 6 of F.3.8.2: `s + coefficient * (above + below)`.
    Irreversible97(f32),
}

/// The steps of [`filter_vertical_codestream`] after any scaling, each with
/// whether it updates the rows at even offsets from the window top. `y0` is
/// the window's first row.
pub(super) fn vertical_steps(transform: WaveletTransform, y0: u32) -> [(VerticalStep, bool); 4] {
    let first_even = y0.is_multiple_of(2);
    match transform {
        WaveletTransform::Reversible53 => [
            (VerticalStep::Reversible53Predict, first_even),
            (VerticalStep::Reversible53Update, !first_even),
            // Unused: the 5/3 filter has two steps.
            (VerticalStep::Reversible53Update, !first_even),
            (VerticalStep::Reversible53Update, !first_even),
        ],
        WaveletTransform::Irreversible97 => [
            (
                VerticalStep::Irreversible97(dwt::IDWT97_NEG_DELTA_F32),
                first_even,
            ),
            (
                VerticalStep::Irreversible97(dwt::IDWT97_NEG_GAMMA_F32),
                !first_even,
            ),
            (
                VerticalStep::Irreversible97(dwt::IDWT97_NEG_BETA_F32),
                first_even,
            ),
            (
                VerticalStep::Irreversible97(dwt::IDWT97_NEG_ALPHA_F32),
                !first_even,
            ),
        ],
    }
}

/// How many of [`vertical_steps`] the transform uses.
pub(super) const fn vertical_step_count(transform: WaveletTransform) -> usize {
    match transform {
        WaveletTransform::Reversible53 => 2,
        WaveletTransform::Irreversible97 => 4,
    }
}

/// The factor [`filter_vertical_codestream`] scales a 9/7 row by before
/// lifting, by whether the row is at an even offset from the window top.
pub(super) fn irreversible_row_scale(y0: u32, even_offset: bool, high_pass: f32) -> f32 {
    if y0.is_multiple_of(2) == even_offset {
        dwt::DWT97_KAPPA_F32
    } else {
        high_pass
    }
}

/// [`lift_row`] at the CPU's SIMD level, for callers outside a dispatch.
pub(super) fn lift_one_row(step: VerticalStep, row: &mut [f32], above: &[f32], below: &[f32]) {
    dispatch!(Level::new(), simd => lift_row(simd, step, row, above, below));
}

/// [`scale_row`] at the CPU's SIMD level, for callers outside a dispatch.
pub(super) fn scale_one_row(row: &mut [f32], factor: f32) {
    dispatch!(Level::new(), simd => scale_row(simd, row, factor));
}

/// Apply `step` to `row` from the rows above and below it.
#[expect(
    clippy::inline_always,
    reason = "the vertical lifting primitive is intentionally inlined with scalar and SIMD operations specialized"
)]
#[inline(always)]
fn lift_row<S: Simd>(simd: S, step: VerticalStep, row: &mut [f32], above: &[f32], below: &[f32]) {
    match step {
        VerticalStep::Reversible53Predict => lift_row_with(
            simd,
            row,
            above,
            below,
            #[inline(always)]
            |s1, s2, s3| s1 - ((s2 + s3 + 2.0) * 0.25).floor(),
            #[inline(always)]
            |s1, s2, s3| s1 - math::floor_f32(math::mul_add(s2 + s3, 0.25, 0.5)),
        ),
        VerticalStep::Reversible53Update => lift_row_with(
            simd,
            row,
            above,
            below,
            #[inline(always)]
            |s1, s2, s3| s1 + ((s2 + s3) * 0.5).floor(),
            #[inline(always)]
            |s1, s2, s3| s1 + math::floor_half_sum_f32(s2, s3),
        ),
        VerticalStep::Irreversible97(coefficient) => {
            let coefficient_simd = f32x8::splat(simd, coefficient);
            lift_row_with(
                simd,
                row,
                above,
                below,
                #[inline(always)]
                |s1, s2, s3| (s2 + s3).mul_add(coefficient_simd, s1),
                #[inline(always)]
                |s1, s2, s3| math::mul_add(s2 + s3, coefficient, s1),
            );
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "the vertical lifting primitive is intentionally inlined with scalar and SIMD operations specialized"
)]
#[inline(always)]
fn lift_row_with<S: Simd>(
    simd: S,
    row: &mut [f32],
    above: &[f32],
    below: &[f32],
    f_simd: impl Fn(f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    f_scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let width = row.len();
    let simd_width = width / SIMD_WIDTH * SIMD_WIDTH;

    // Process SIMD chunks.
    for base_column in (0..simd_width).step_by(SIMD_WIDTH) {
        let s1 = f32x8::from_slice(simd, &row[base_column..][..SIMD_WIDTH]);
        let s2 = f32x8::from_slice(simd, &above[base_column..][..SIMD_WIDTH]);
        let s3 = f32x8::from_slice(simd, &below[base_column..][..SIMD_WIDTH]);

        let result = f_simd(s1, s2, s3);
        result.store(&mut row[base_column..][..SIMD_WIDTH]);
    }

    // Process scalar remainder.
    for col in simd_width..width {
        row[col] = f_scalar(row[col], above[col], below[col]);
    }
}

/// Multiply every sample of `row` by `factor`.
#[expect(
    clippy::inline_always,
    reason = "the vertical scaling primitive is intentionally inlined with scalar and SIMD operations specialized"
)]
#[inline(always)]
fn scale_row<S: Simd>(simd: S, row: &mut [f32], factor: f32) {
    let simd_width = row.len() / SIMD_WIDTH * SIMD_WIDTH;
    let factor_simd = f32x8::splat(simd, factor);
    for base_column in (0..simd_width).step_by(SIMD_WIDTH) {
        let values = f32x8::from_slice(simd, &row[base_column..][..SIMD_WIDTH]) * factor_simd;
        values.store(&mut row[base_column..][..SIMD_WIDTH]);
    }
    for sample in &mut row[simd_width..] {
        *sample *= factor;
    }
}

/// The row at `index` of a `width`-wide `scanline` to update, and the rows
/// at `above` and `below`, which must differ from it.
pub(super) fn split_rows(
    scanline: &mut [f32],
    width: usize,
    index: usize,
    above: usize,
    below: usize,
) -> (&mut [f32], &[f32], &[f32]) {
    let (head, tail) = scanline.split_at_mut(index * width);
    let (row, rest) = tail.split_at_mut(width);
    let other = |other: usize| -> &[f32] {
        if other < index {
            &head[other * width..][..width]
        } else {
            &rest[(other - index - 1) * width..][..width]
        }
    };
    (row, other(above), other(below))
}

#[expect(
    clippy::inline_always,
    reason = "the vertical lifting primitive is intentionally inlined with scalar and SIMD operations specialized"
)]
#[inline(always)]
fn filter_step_vertical<S: Simd>(
    simd: S,
    scanline: &mut [f32],
    height: usize,
    width: usize,
    first: usize,
    step: VerticalStep,
) {
    for row in (first..height).step_by(2) {
        let row_above = periodic_symmetric_extension_left(row, 1);
        let row_below = periodic_symmetric_extension_right(row, 1, height);
        let (target, above, below) = split_rows(scanline, width, row, row_above, row_below);
        lift_row(simd, step, target, above, below);
    }
}

/// The `VER_SR` procedure from F.3.5.
pub(super) fn filter_vertical(
    coefficients: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
) {
    filter_vertical_with_high_pass(coefficients, rect, transform, dwt::DWT97_INV_KAPPA_F32);
}

pub(super) fn filter_vertical_codestream(
    coefficients: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
) {
    filter_vertical_with_high_pass(
        coefficients,
        rect,
        transform,
        super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
    );
}

fn filter_vertical_with_high_pass(
    coefficients: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
    irreversible_high_pass: f32,
) {
    dispatch!(Level::new(), simd => filter_vertical_impl(
        simd,
        coefficients,
        rect,
        transform,
        irreversible_high_pass,
    ));
}

pub(super) fn filter_vertical_i64(coefficients: &mut [i64], rect: IntRect) {
    let width = rect.width() as usize;
    let height = rect.height() as usize;
    let y0 = rect.y0 as usize;

    if height == 1 {
        if !y0.is_multiple_of(2) {
            for sample in coefficients.iter_mut().take(width) {
                *sample = floor_div_i64(*sample, 2);
            }
        }
        return;
    }

    let first_even = y0 % 2;
    let first_odd = 1 - first_even;

    filter_step_vertical_i64(
        coefficients,
        height,
        width,
        first_even,
        |s, above, below| s - floor_div_i64(above + below + 2, 4),
    );
    filter_step_vertical_i64(coefficients, height, width, first_odd, |s, above, below| {
        s + floor_div_i64(above + below, 2)
    });
}

fn filter_step_vertical_i64(
    scanline: &mut [i64],
    height: usize,
    width: usize,
    first: usize,
    f: impl Fn(i64, i64, i64) -> i64,
) {
    for row in (first..height).step_by(2) {
        let row_above = periodic_symmetric_extension_left(row, 1);
        let row_below = periodic_symmetric_extension_right(row, 1, height);

        for col in 0..width {
            let idx = row * width + col;
            scanline[idx] = f(
                scanline[idx],
                scanline[row_above * width + col],
                scanline[row_below * width + col],
            );
        }
    }
}

#[expect(
    clippy::inline_always,
    reason = "the SIMD IDWT implementation is intentionally specialized at the architecture dispatch boundary"
)]
#[inline(always)]
fn filter_vertical_impl<S: Simd>(
    simd: S,
    scanline: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
    irreversible_high_pass: f32,
) {
    let width = rect.width() as usize;
    let height = rect.height() as usize;

    if height == 1 {
        if !rect.y0.is_multiple_of(2) {
            scale_row(simd, &mut scanline[..width], 0.5);
        }
        return;
    }

    if transform == WaveletTransform::Irreversible97 {
        // Step 1 and 2.
        // Originally: for i in (start / 2 - 1)..(end / 2 + 2).
        // Originally: for i in (start / 2 - 2)..(end / 2 + 2).
        for (offset, row) in scanline.chunks_exact_mut(width).take(height).enumerate() {
            let factor = irreversible_row_scale(rect.y0, offset % 2 == 0, irreversible_high_pass);
            scale_row(simd, row, factor);
        }
    }

    // 5/3: equations (F-5) and (F-6). 9/7: steps 3 to 6.
    for (step, even_offsets) in vertical_steps(transform, rect.y0)
        .into_iter()
        .take(vertical_step_count(transform))
    {
        let first = usize::from(!even_offsets);
        filter_step_vertical(simd, scanline, height, width, first, step);
    }
}
