// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    constants::{
        DWT97_ALPHA, DWT97_BETA, DWT97_DELTA, DWT97_GAMMA, DWT97_INV_KAPPA, DWT97_KAPPA,
        DWT97_ROW_LIFT_MAX_WIDTH, IDCT_C0, IDCT_C1, IDCT_C2, IDCT_C3, IDCT_C5, IDCT_C6, IDCT_C7,
    },
    helpers::{load_f32, load_i16, store_f32},
};

#[inline(always)]
pub(crate) fn idct8_basis_0(_sample_idx: i32) -> f32 {
    IDCT_C0
}

#[inline(always)]
pub(crate) fn idct8_basis_1(sample_idx: i32) -> f32 {
    match sample_idx {
        0 => IDCT_C1,
        1 => IDCT_C3,
        2 => IDCT_C5,
        3 => IDCT_C7,
        4 => -IDCT_C7,
        5 => -IDCT_C5,
        6 => -IDCT_C3,
        7 => -IDCT_C1,
        _ => 0.0,
    }
}

#[inline(always)]
pub(crate) fn idct8_basis_2(sample_idx: i32) -> f32 {
    match sample_idx {
        0 => IDCT_C2,
        1 => IDCT_C6,
        2 => -IDCT_C6,
        3 => -IDCT_C2,
        4 => -IDCT_C2,
        5 => -IDCT_C6,
        6 => IDCT_C6,
        7 => IDCT_C2,
        _ => 0.0,
    }
}

#[inline(always)]
pub(crate) fn idct8_basis_3(sample_idx: i32) -> f32 {
    match sample_idx {
        0 => IDCT_C3,
        1 => -IDCT_C7,
        2 => -IDCT_C1,
        3 => -IDCT_C5,
        4 => IDCT_C5,
        5 => IDCT_C1,
        6 => IDCT_C7,
        7 => -IDCT_C3,
        _ => 0.0,
    }
}

#[inline(always)]
pub(crate) fn idct8_basis_4(sample_idx: i32) -> f32 {
    match sample_idx {
        0 | 3 | 4 | 7 => IDCT_C0,
        1 | 2 | 5 | 6 => -IDCT_C0,
        _ => 0.0,
    }
}

#[inline(always)]
pub(crate) fn idct8_basis_5(sample_idx: i32) -> f32 {
    match sample_idx {
        0 => IDCT_C5,
        1 => -IDCT_C1,
        2 => IDCT_C7,
        3 => IDCT_C3,
        4 => -IDCT_C3,
        5 => -IDCT_C7,
        6 => IDCT_C1,
        7 => -IDCT_C5,
        _ => 0.0,
    }
}

#[inline(always)]
pub(crate) fn idct8_basis_6(sample_idx: i32) -> f32 {
    match sample_idx {
        0 => IDCT_C6,
        1 => -IDCT_C2,
        2 => IDCT_C2,
        3 => -IDCT_C6,
        4 => -IDCT_C6,
        5 => IDCT_C2,
        6 => -IDCT_C2,
        7 => IDCT_C6,
        _ => 0.0,
    }
}

#[inline(always)]
pub(crate) fn idct8_basis_7(sample_idx: i32) -> f32 {
    match sample_idx {
        0 => IDCT_C7,
        1 => -IDCT_C5,
        2 => IDCT_C3,
        3 => -IDCT_C1,
        4 => IDCT_C1,
        5 => -IDCT_C3,
        6 => IDCT_C5,
        7 => -IDCT_C7,
        _ => 0.0,
    }
}

macro_rules! accumulate_idct_f32_row {
    ($sample:ident, $block:expr, $row:expr, $y_basis:expr, $x0:expr, $x1:expr, $x2:expr, $x3:expr, $x4:expr, $x5:expr, $x6:expr, $x7:expr) => {{
        let y_basis = $y_basis;
        $sample += load_f32($block, ($row * 8) as u64) * y_basis * $x0;
        $sample += load_f32($block, ($row * 8 + 1) as u64) * y_basis * $x1;
        $sample += load_f32($block, ($row * 8 + 2) as u64) * y_basis * $x2;
        $sample += load_f32($block, ($row * 8 + 3) as u64) * y_basis * $x3;
        $sample += load_f32($block, ($row * 8 + 4) as u64) * y_basis * $x4;
        $sample += load_f32($block, ($row * 8 + 5) as u64) * y_basis * $x5;
        $sample += load_f32($block, ($row * 8 + 6) as u64) * y_basis * $x6;
        $sample += load_f32($block, ($row * 8 + 7) as u64) * y_basis * $x7;
    }};
}

macro_rules! accumulate_idct_i16_row {
    ($sample:ident, $block:expr, $row:expr, $y_basis:expr, $x0:expr, $x1:expr, $x2:expr, $x3:expr, $x4:expr, $x5:expr, $x6:expr, $x7:expr) => {{
        let y_basis = $y_basis;
        $sample += load_i16($block, ($row * 8) as u64) as f32 * y_basis * $x0;
        $sample += load_i16($block, ($row * 8 + 1) as u64) as f32 * y_basis * $x1;
        $sample += load_i16($block, ($row * 8 + 2) as u64) as f32 * y_basis * $x2;
        $sample += load_i16($block, ($row * 8 + 3) as u64) as f32 * y_basis * $x3;
        $sample += load_i16($block, ($row * 8 + 4) as u64) as f32 * y_basis * $x4;
        $sample += load_i16($block, ($row * 8 + 5) as u64) as f32 * y_basis * $x5;
        $sample += load_i16($block, ($row * 8 + 6) as u64) as f32 * y_basis * $x6;
        $sample += load_i16($block, ($row * 8 + 7) as u64) as f32 * y_basis * $x7;
    }};
}

#[inline(always)]
pub(crate) fn idct8x8_sample(block: *const f32, local_x: i32, local_y: i32) -> f32 {
    let x0 = idct8_basis_0(local_x);
    let x1 = idct8_basis_1(local_x);
    let x2 = idct8_basis_2(local_x);
    let x3 = idct8_basis_3(local_x);
    let x4 = idct8_basis_4(local_x);
    let x5 = idct8_basis_5(local_x);
    let x6 = idct8_basis_6(local_x);
    let x7 = idct8_basis_7(local_x);
    let mut sample = 0.0_f32;
    accumulate_idct_f32_row!(
        sample,
        block,
        0,
        idct8_basis_0(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        1,
        idct8_basis_1(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        2,
        idct8_basis_2(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        3,
        idct8_basis_3(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        4,
        idct8_basis_4(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        5,
        idct8_basis_5(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        6,
        idct8_basis_6(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_f32_row!(
        sample,
        block,
        7,
        idct8_basis_7(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    sample
}

#[inline(always)]
pub(crate) fn idct8x8_sample_i16(block: *const i16, local_x: i32, local_y: i32) -> f32 {
    let x0 = idct8_basis_0(local_x);
    let x1 = idct8_basis_1(local_x);
    let x2 = idct8_basis_2(local_x);
    let x3 = idct8_basis_3(local_x);
    let x4 = idct8_basis_4(local_x);
    let x5 = idct8_basis_5(local_x);
    let x6 = idct8_basis_6(local_x);
    let x7 = idct8_basis_7(local_x);
    let mut sample = 0.0_f32;
    accumulate_idct_i16_row!(
        sample,
        block,
        0,
        idct8_basis_0(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        1,
        idct8_basis_1(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        2,
        idct8_basis_2(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        3,
        idct8_basis_3(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        4,
        idct8_basis_4(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        5,
        idct8_basis_5(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        6,
        idct8_basis_6(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    accumulate_idct_i16_row!(
        sample,
        block,
        7,
        idct8_basis_7(local_y),
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7
    );
    sample
}

#[inline(always)]
pub(crate) fn forward_lift_97(data: *mut f32, n: i32, stride: i32) {
    if n < 2 {
        return;
    }
    let last_even = if n % 2 == 0 { n - 2 } else { n - 1 };

    let mut i = 1_i32;
    while i < n {
        let left = load_f32(data.cast_const(), ((i - 1) * stride) as u64);
        let right = if i + 1 < n {
            load_f32(data.cast_const(), ((i + 1) * stride) as u64)
        } else {
            load_f32(data.cast_const(), (last_even * stride) as u64)
        };
        let value = load_f32(data.cast_const(), (i * stride) as u64) + DWT97_ALPHA * (left + right);
        store_f32(data, (i * stride) as u64, value);
        i += 2;
    }

    let mut i = 0_i32;
    while i < n {
        let left = if i > 0 {
            load_f32(data.cast_const(), ((i - 1) * stride) as u64)
        } else {
            load_f32(data.cast_const(), stride as u64)
        };
        let right = if i + 1 < n {
            load_f32(data.cast_const(), ((i + 1) * stride) as u64)
        } else {
            left
        };
        let value = load_f32(data.cast_const(), (i * stride) as u64) + DWT97_BETA * (left + right);
        store_f32(data, (i * stride) as u64, value);
        i += 2;
    }

    let mut i = 1_i32;
    while i < n {
        let left = load_f32(data.cast_const(), ((i - 1) * stride) as u64);
        let right = if i + 1 < n {
            load_f32(data.cast_const(), ((i + 1) * stride) as u64)
        } else {
            load_f32(data.cast_const(), (last_even * stride) as u64)
        };
        let value = load_f32(data.cast_const(), (i * stride) as u64) + DWT97_GAMMA * (left + right);
        store_f32(data, (i * stride) as u64, value);
        i += 2;
    }

    let mut i = 0_i32;
    while i < n {
        let left = if i > 0 {
            load_f32(data.cast_const(), ((i - 1) * stride) as u64)
        } else {
            load_f32(data.cast_const(), stride as u64)
        };
        let right = if i + 1 < n {
            load_f32(data.cast_const(), ((i + 1) * stride) as u64)
        } else {
            left
        };
        let value = load_f32(data.cast_const(), (i * stride) as u64) + DWT97_DELTA * (left + right);
        store_f32(data, (i * stride) as u64, value);
        i += 2;
    }

    let mut i = 0_i32;
    while i < n {
        let value = load_f32(data.cast_const(), (i * stride) as u64) * DWT97_INV_KAPPA;
        store_f32(data, (i * stride) as u64, value);
        i += 2;
    }

    let mut i = 1_i32;
    while i < n {
        let value = load_f32(data.cast_const(), (i * stride) as u64) * DWT97_KAPPA;
        store_f32(data, (i * stride) as u64, value);
        i += 2;
    }
}

#[inline(always)]

pub(crate) fn shared_row_index(row_lane: i32, x: i32) -> u64 {
    row_lane as u64 * DWT97_ROW_LIFT_MAX_WIDTH as u64 + x as u64
}
