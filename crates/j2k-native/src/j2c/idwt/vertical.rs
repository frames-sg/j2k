// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::codestream::WaveletTransform;
use super::super::rect::IntRect;
use super::filter_common::{
    floor_div_i64, periodic_symmetric_extension_left, periodic_symmetric_extension_right,
};
use crate::math::{self, dispatch, f32x8, Level, Simd, SIMD_WIDTH};
use j2k_codec_math::dwt;

#[inline(always)]
fn filter_step_vertical<S: Simd>(
    simd: S,
    scanline: &mut [f32],
    height: usize,
    width: usize,
    simd_width: usize,
    first: usize,
    f_simd: impl Fn(f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    f_scalar: impl Fn(f32, f32, f32) -> f32,
) {
    for row in (first..height).step_by(2) {
        let row_above = periodic_symmetric_extension_left(row, 1);
        let row_below = periodic_symmetric_extension_right(row, 1, height);

        // Process SIMD chunks.
        for base_column in (0..simd_width).step_by(SIMD_WIDTH) {
            let s1 = f32x8::from_slice(simd, &scanline[row * width + base_column..][..SIMD_WIDTH]);
            let s2 = f32x8::from_slice(
                simd,
                &scanline[row_above * width + base_column..][..SIMD_WIDTH],
            );
            let s3 = f32x8::from_slice(
                simd,
                &scanline[row_below * width + base_column..][..SIMD_WIDTH],
            );

            let result = f_simd(s1, s2, s3);
            result.store(&mut scanline[row * width + base_column..][..SIMD_WIDTH]);
        }

        // Process scalar remainder.
        for col in simd_width..width {
            let s1 = scanline[row * width + col];
            let s2 = scanline[row_above * width + col];
            let s3 = scanline[row_below * width + col];
            scanline[row * width + col] = f_scalar(s1, s2, s3);
        }
    }
}

/// The `VER_SR` procedure from F.3.5.
pub(super) fn filter_vertical(
    coefficients: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
) {
    dispatch!(Level::new(), simd => filter_vertical_impl(simd, coefficients, rect, transform));
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

#[inline(always)]
fn filter_vertical_impl<S: Simd>(
    simd: S,
    scanline: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
) {
    let width = rect.width() as usize;
    let height = rect.height() as usize;
    let y0 = rect.y0 as usize;

    if height == 1 {
        if !y0.is_multiple_of(2) {
            let simd_width = width / SIMD_WIDTH * SIMD_WIDTH;
            for base_column in (0..simd_width).step_by(SIMD_WIDTH) {
                let mut loaded = f32x8::from_slice(simd, &scanline[base_column..][..SIMD_WIDTH]);
                loaded *= 0.5;
                loaded.store(&mut scanline[base_column..][..SIMD_WIDTH]);
            }

            // Scalar remainder.
            #[allow(clippy::needless_range_loop)]
            for col in simd_width..width {
                scanline[col] *= 0.5;
            }
        }
        return;
    }

    match transform {
        WaveletTransform::Reversible53 => {
            reversible_filter_53r_simd(simd, scanline, height, width, y0);
        }
        WaveletTransform::Irreversible97 => {
            irreversible_filter_97i_simd(simd, scanline, height, width, y0);
        }
    }
}

/// The 1D FILTER 5-3R procedure from F.3.8.1.
#[inline(always)]
fn reversible_filter_53r_simd<S: Simd>(
    simd: S,
    scanline: &mut [f32],
    height: usize,
    width: usize,
    y0: usize,
) {
    let first_even = y0 % 2;
    let first_odd = 1 - first_even;
    let simd_width = width / SIMD_WIDTH * SIMD_WIDTH;

    // Equation (F-5).
    // Originally: for i in (start / 2)..(end / 2 + 1).
    filter_step_vertical(
        simd,
        scanline,
        height,
        width,
        simd_width,
        first_even,
        #[inline(always)]
        |s1, s2, s3| s1 - ((s2 + s3 + 2.0) * 0.25).floor(),
        #[inline(always)]
        |s1, s2, s3| s1 - math::floor_f32(math::mul_add(s2 + s3, 0.25, 0.5)),
    );

    // Equation (F-6).
    // Originally: for i in (start / 2)..(end / 2).
    filter_step_vertical(
        simd,
        scanline,
        height,
        width,
        simd_width,
        first_odd,
        #[inline(always)]
        |s1, s2, s3| s1 + ((s2 + s3) * 0.5).floor(),
        #[inline(always)]
        |s1, s2, s3| s1 + math::floor_f32((s2 + s3) * 0.5),
    );
}

/// The 1D Filter 9-7I procedure from F.3.8.2.
#[inline(always)]
fn irreversible_filter_97i_simd<S: Simd>(
    simd: S,
    scanline: &mut [f32],
    height: usize,
    width: usize,
    y0: usize,
) {
    const NEG_ALPHA: f32 = dwt::IDWT97_NEG_ALPHA_F32;
    const NEG_BETA: f32 = dwt::IDWT97_NEG_BETA_F32;
    const NEG_GAMMA: f32 = dwt::IDWT97_NEG_GAMMA_F32;
    const NEG_DELTA: f32 = dwt::IDWT97_NEG_DELTA_F32;
    const KAPPA: f32 = dwt::DWT97_KAPPA_F32;

    const INV_KAPPA: f32 = dwt::DWT97_INV_KAPPA_F32;

    let neg_alpha = f32x8::splat(simd, NEG_ALPHA);
    let neg_beta = f32x8::splat(simd, NEG_BETA);
    let neg_gamma = f32x8::splat(simd, NEG_GAMMA);
    let neg_delta = f32x8::splat(simd, NEG_DELTA);
    let kappa = f32x8::splat(simd, KAPPA);
    let inv_kappa = f32x8::splat(simd, INV_KAPPA);

    // Determine which local row indices correspond to even/odd global positions.
    let first_even = y0 % 2;
    let first_odd = 1 - first_even;
    let simd_width = width / SIMD_WIDTH * SIMD_WIDTH;

    let (k0, k1, k0_simd, k1_simd) = if first_even == 0 {
        (KAPPA, INV_KAPPA, kappa, inv_kappa)
    } else {
        (INV_KAPPA, KAPPA, inv_kappa, kappa)
    };

    // Step 1 and 2.
    // Originally: for i in (start / 2 - 1)..(end / 2 + 2).
    // Originally: for i in (start / 2 - 2)..(end / 2 + 2).
    for row in (0..height.saturating_sub(1)).step_by(2) {
        for base_column in (0..simd_width).step_by(SIMD_WIDTH) {
            let mut vals0 =
                f32x8::from_slice(simd, &scanline[row * width + base_column..][..SIMD_WIDTH]);
            let mut vals1 = f32x8::from_slice(
                simd,
                &scanline[(row + 1) * width + base_column..][..SIMD_WIDTH],
            );
            vals0 = vals0 * k0_simd;
            vals1 = vals1 * k1_simd;
            vals0.store(&mut scanline[row * width + base_column..][..SIMD_WIDTH]);
            vals1.store(&mut scanline[(row + 1) * width + base_column..][..SIMD_WIDTH]);
        }
        for col in simd_width..width {
            scanline[row * width + col] *= k0;
            scanline[(row + 1) * width + col] *= k1;
        }
    }

    if height % 2 == 1 {
        let row = height - 1;
        for base_column in (0..simd_width).step_by(SIMD_WIDTH) {
            let mut vals =
                f32x8::from_slice(simd, &scanline[row * width + base_column..][..SIMD_WIDTH]);
            vals = vals * k0_simd;
            vals.store(&mut scanline[row * width + base_column..][..SIMD_WIDTH]);
        }
        for col in simd_width..width {
            scanline[row * width + col] *= k0;
        }
    }

    // Step 3.
    // Originally: for i in (start / 2 - 1)..(end / 2 + 2).
    filter_step_vertical(
        simd,
        scanline,
        height,
        width,
        simd_width,
        first_even,
        #[inline(always)]
        |s1, s2, s3| (s2 + s3).mul_add(neg_delta, s1),
        #[inline(always)]
        |s1, s2, s3| math::mul_add(s2 + s3, NEG_DELTA, s1),
    );

    // Step 4.
    // Originally: for i in (start / 2 - 1)..(end / 2 + 1).
    filter_step_vertical(
        simd,
        scanline,
        height,
        width,
        simd_width,
        first_odd,
        #[inline(always)]
        |s1, s2, s3| (s2 + s3).mul_add(neg_gamma, s1),
        #[inline(always)]
        |s1, s2, s3| math::mul_add(s2 + s3, NEG_GAMMA, s1),
    );

    // Step 5.
    // Originally: for i in (start / 2)..(end / 2 + 1).
    filter_step_vertical(
        simd,
        scanline,
        height,
        width,
        simd_width,
        first_even,
        #[inline(always)]
        |s1, s2, s3| (s2 + s3).mul_add(neg_beta, s1),
        #[inline(always)]
        |s1, s2, s3| math::mul_add(s2 + s3, NEG_BETA, s1),
    );

    // Step 6.
    // Originally: for i in (start / 2)..(end / 2).
    filter_step_vertical(
        simd,
        scanline,
        height,
        width,
        simd_width,
        first_odd,
        #[inline(always)]
        |s1, s2, s3| (s2 + s3).mul_add(neg_alpha, s1),
        #[inline(always)]
        |s1, s2, s3| math::mul_add(s2 + s3, NEG_ALPHA, s1),
    );
}
