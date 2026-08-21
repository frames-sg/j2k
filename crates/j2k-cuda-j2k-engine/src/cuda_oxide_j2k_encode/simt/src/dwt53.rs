// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::helpers::{floor_f32, load_f32};

#[inline(always)]
pub(crate) fn j2k_fdwt53_predict_row(
    src: *const f32,
    row_base: u32,
    width: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let left = load_f32(src, row_base + odd - 1);
    let right = if odd + 1 < width {
        load_f32(src, row_base + odd + 1)
    } else {
        load_f32(src, row_base + last_even)
    };
    load_f32(src, row_base + odd) - floor_f32((left + right) * 0.5)
}

#[inline(always)]
pub(crate) fn j2k_fdwt53_predict_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if height % 2 == 0 {
        height - 2
    } else {
        height - 1
    };
    let top = load_f32(src, (odd - 1) * full_width + x);
    let bottom = if odd + 1 < height {
        load_f32(src, (odd + 1) * full_width + x)
    } else {
        load_f32(src, last_even * full_width + x)
    };
    load_f32(src, odd * full_width + x) - floor_f32((top + bottom) * 0.5)
}
