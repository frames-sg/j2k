// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::codestream::WaveletTransform;
use super::super::rect::IntRect;
use super::filter_common::{
    floor_div_i64, periodic_symmetric_extension_left, periodic_symmetric_extension_right,
};
use crate::math;
use j2k_codec_math::dwt;

/// The `HOR_SR` procedure from F.3.4.
pub(super) fn filter_horizontal(
    coefficients: &mut [f32],
    rect: IntRect,
    transform: WaveletTransform,
) {
    let width = rect.width() as usize;

    for scanline in coefficients
        .chunks_exact_mut(width)
        .take(rect.height() as usize)
    {
        filter_row(scanline, width, rect.x0 as usize, transform);
    }
}

/// The `1D_SR` procedure from F.3.6.
fn filter_row(scanline: &mut [f32], width: usize, x0: usize, transform: WaveletTransform) {
    if width == 1 {
        if !x0.is_multiple_of(2) {
            scanline[0] *= 0.5;
        }

        return;
    }

    match transform {
        WaveletTransform::Reversible53 => reversible_filter_53r(scanline, width, x0),
        WaveletTransform::Irreversible97 => irreversible_filter_97i(scanline, width, x0),
    }
}

/// The 1D FILTER 5-3R procedure from F.3.8.1.
fn reversible_filter_53r(scanline: &mut [f32], width: usize, x0: usize) {
    let first_even = x0 % 2;
    let first_odd = 1 - first_even;

    // Equation (F-5).
    // Originally: for i in (start / 2)..(end / 2 + 1).
    filter_step_horizontal(
        scanline,
        width,
        first_even,
        #[inline(always)]
        |s, left, right| s - math::floor_f32(math::mul_add(left + right, 0.25, 0.5)),
    );

    // Equation (F-6).
    // Originally: for i in (start / 2)..(end / 2).
    filter_step_horizontal(
        scanline,
        width,
        first_odd,
        #[inline(always)]
        |s, left, right| s + math::floor_f32((left + right) * 0.5),
    );
}

pub(super) fn filter_horizontal_i64(coefficients: &mut [i64], rect: IntRect) {
    let width = rect.width() as usize;

    for scanline in coefficients
        .chunks_exact_mut(width)
        .take(rect.height() as usize)
    {
        filter_row_i64(scanline, width, rect.x0 as usize);
    }
}

fn filter_row_i64(scanline: &mut [i64], width: usize, x0: usize) {
    if width == 1 {
        if !x0.is_multiple_of(2) {
            scanline[0] = floor_div_i64(scanline[0], 2);
        }
        return;
    }

    reversible_filter_53r_i64(scanline, width, x0);
}

fn reversible_filter_53r_i64(scanline: &mut [i64], width: usize, x0: usize) {
    let first_even = x0 % 2;
    let first_odd = 1 - first_even;

    filter_step_horizontal_i64(scanline, width, first_even, |s, left, right| {
        s - floor_div_i64(left + right + 2, 4)
    });
    filter_step_horizontal_i64(scanline, width, first_odd, |s, left, right| {
        s + floor_div_i64(left + right, 2)
    });
}

/// The 1D Filter 9-7I procedure from F.3.8.2.
fn irreversible_filter_97i(scanline: &mut [f32], width: usize, x0: usize) {
    // Table F.4.
    const NEG_ALPHA: f32 = dwt::IDWT97_NEG_ALPHA_F32;
    const NEG_BETA: f32 = dwt::IDWT97_NEG_BETA_F32;
    const NEG_GAMMA: f32 = dwt::IDWT97_NEG_GAMMA_F32;
    const NEG_DELTA: f32 = dwt::IDWT97_NEG_DELTA_F32;
    const KAPPA: f32 = dwt::DWT97_KAPPA_F32;
    const INV_KAPPA: f32 = dwt::DWT97_INV_KAPPA_F32;

    let first_even = x0 % 2;
    let first_odd = 1 - first_even;

    let (k0, k1) = if first_even == 0 {
        (KAPPA, INV_KAPPA)
    } else {
        (INV_KAPPA, KAPPA)
    };

    // Step 1 and 2.
    // Originally: for i in (start / 2 - 1)..(end / 2 + 2).
    // Originally: for i in (start / 2 - 2)..(end / 2 + 2).
    for i in (0..width.saturating_sub(1)).step_by(2) {
        scanline[i] *= k0;
        scanline[i + 1] *= k1;
    }
    if width % 2 == 1 {
        scanline[width - 1] *= k0;
    }

    // Step 3.
    // Originally: for i in (start / 2 - 1)..(end / 2 + 2).
    filter_step_horizontal(
        scanline,
        width,
        first_even,
        #[inline(always)]
        |s, left, right| math::mul_add(left + right, NEG_DELTA, s),
    );

    // Step 4.
    // Originally: for i in (start / 2 - 1)..((x0 + width) / 2 + 1).
    filter_step_horizontal(
        scanline,
        width,
        first_odd,
        #[inline(always)]
        |s, left, right| math::mul_add(left + right, NEG_GAMMA, s),
    );

    // Step 5.
    // Originally: for i in (start / 2)..(end / 2 + 1).
    filter_step_horizontal(
        scanline,
        width,
        first_even,
        #[inline(always)]
        |s, left, right| math::mul_add(left + right, NEG_BETA, s),
    );

    // Step 6.
    // Originally: for i in (start / 2)..(end / 2).
    filter_step_horizontal(
        scanline,
        width,
        first_odd,
        #[inline(always)]
        |s, left, right| math::mul_add(left + right, NEG_ALPHA, s),
    );
}

#[cfg(test)]
pub(crate) fn test_irreversible_filter_97i(scanline: &mut [f32], width: usize, x0: usize) {
    irreversible_filter_97i(scanline, width, x0);
}

#[expect(
    clippy::inline_always,
    reason = "the horizontal lifting primitive is intentionally inlined with its filter operation specialized"
)]
#[inline(always)]
fn filter_step_horizontal(
    scanline: &mut [f32],
    width: usize,
    first: usize,
    f: impl Fn(f32, f32, f32) -> f32,
) {
    if first == 0 {
        let left = periodic_symmetric_extension_left(0, 1);
        let right = periodic_symmetric_extension_right(0, 1, width);
        scanline[0] = f(scanline[0], scanline[left], scanline[right]);
    }

    let middle_start = if first == 0 { 2 } else { 1 };
    for i in (middle_start..width - 1).step_by(2) {
        scanline[i] = f(scanline[i], scanline[i - 1], scanline[i + 1]);
    }

    if width > 1 && (width - 1) % 2 == first {
        let i = width - 1;
        let left = periodic_symmetric_extension_left(i, 1);
        let right = periodic_symmetric_extension_right(i, 1, width);
        scanline[i] = f(scanline[i], scanline[left], scanline[right]);
    }
}

fn filter_step_horizontal_i64(
    scanline: &mut [i64],
    width: usize,
    first: usize,
    f: impl Fn(i64, i64, i64) -> i64,
) {
    if first == 0 {
        let left = periodic_symmetric_extension_left(0, 1);
        let right = periodic_symmetric_extension_right(0, 1, width);
        scanline[0] = f(scanline[0], scanline[left], scanline[right]);
    }

    let middle_start = if first == 0 { 2 } else { 1 };
    for i in (middle_start..width - 1).step_by(2) {
        scanline[i] = f(scanline[i], scanline[i - 1], scanline[i + 1]);
    }

    if width > 1 && (width - 1) % 2 == first {
        let i = width - 1;
        let left = periodic_symmetric_extension_left(i, 1);
        let right = periodic_symmetric_extension_right(i, 1, width);
        scanline[i] = f(scanline[i], scanline[left], scanline[right]);
    }
}
