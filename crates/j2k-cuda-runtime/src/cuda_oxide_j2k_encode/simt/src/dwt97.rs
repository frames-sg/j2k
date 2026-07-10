// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    constants::{J2K_FDWT97_ALPHA, J2K_FDWT97_BETA, J2K_FDWT97_DELTA, J2K_FDWT97_GAMMA},
    helpers::load_f32,
};

#[inline(always)]
pub(crate) fn j2k_fdwt97_high1_row(
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
    load_f32(src, row_base + odd) + J2K_FDWT97_ALPHA * (left + right)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_low1_row(
    src: *const f32,
    row_base: u32,
    width: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let left = if low_index > 0 {
        j2k_fdwt97_high1_row(src, row_base, width, low_index - 1)
    } else {
        j2k_fdwt97_high1_row(src, row_base, width, 0)
    };
    let right = if even + 1 < width {
        j2k_fdwt97_high1_row(src, row_base, width, low_index)
    } else {
        left
    };
    load_f32(src, row_base + even) + J2K_FDWT97_BETA * (left + right)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_high2_row(
    src: *const f32,
    row_base: u32,
    width: u32,
    high_index: u32,
) -> f32 {
    let odd = high_index * 2 + 1;
    let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
    let last_low = last_even / 2;
    let left = j2k_fdwt97_low1_row(src, row_base, width, high_index);
    let right = if odd + 1 < width {
        j2k_fdwt97_low1_row(src, row_base, width, high_index + 1)
    } else {
        j2k_fdwt97_low1_row(src, row_base, width, last_low)
    };
    j2k_fdwt97_high1_row(src, row_base, width, high_index) + J2K_FDWT97_GAMMA * (left + right)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_low2_row(
    src: *const f32,
    row_base: u32,
    width: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let left = if low_index > 0 {
        j2k_fdwt97_high2_row(src, row_base, width, low_index - 1)
    } else {
        j2k_fdwt97_high2_row(src, row_base, width, 0)
    };
    let right = if even + 1 < width {
        j2k_fdwt97_high2_row(src, row_base, width, low_index)
    } else {
        left
    };
    j2k_fdwt97_low1_row(src, row_base, width, low_index) + J2K_FDWT97_DELTA * (left + right)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_high1_col(
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
    load_f32(src, odd * full_width + x) + J2K_FDWT97_ALPHA * (top + bottom)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_low1_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let top = if low_index > 0 {
        j2k_fdwt97_high1_col(src, x, full_width, height, low_index - 1)
    } else {
        j2k_fdwt97_high1_col(src, x, full_width, height, 0)
    };
    let bottom = if even + 1 < height {
        j2k_fdwt97_high1_col(src, x, full_width, height, low_index)
    } else {
        top
    };
    load_f32(src, even * full_width + x) + J2K_FDWT97_BETA * (top + bottom)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_high2_col(
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
    let last_low = last_even / 2;
    let top = j2k_fdwt97_low1_col(src, x, full_width, height, high_index);
    let bottom = if odd + 1 < height {
        j2k_fdwt97_low1_col(src, x, full_width, height, high_index + 1)
    } else {
        j2k_fdwt97_low1_col(src, x, full_width, height, last_low)
    };
    j2k_fdwt97_high1_col(src, x, full_width, height, high_index) + J2K_FDWT97_GAMMA * (top + bottom)
}

#[inline(always)]
pub(crate) fn j2k_fdwt97_low2_col(
    src: *const f32,
    x: u32,
    full_width: u32,
    height: u32,
    low_index: u32,
) -> f32 {
    let even = low_index * 2;
    let top = if low_index > 0 {
        j2k_fdwt97_high2_col(src, x, full_width, height, low_index - 1)
    } else {
        j2k_fdwt97_high2_col(src, x, full_width, height, 0)
    };
    let bottom = if even + 1 < height {
        j2k_fdwt97_high2_col(src, x, full_width, height, low_index)
    } else {
        top
    };
    j2k_fdwt97_low1_col(src, x, full_width, height, low_index) + J2K_FDWT97_DELTA * (top + bottom)
}
