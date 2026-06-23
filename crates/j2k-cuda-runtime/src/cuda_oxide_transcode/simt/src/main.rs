#![allow(static_mut_refs)]

use cuda_device::{SharedArray, kernel, thread};
use cuda_host::cuda_module;

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

#[inline(always)]
fn load_i16(ptr: *const i16, index: u64) -> i16 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_i32(ptr: *const i32, index: u64) -> i32 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn load_f32(ptr: *const f32, index: u64) -> f32 {
    unsafe { *ptr.add(index as usize) }
}

#[inline(always)]
fn store_i32(ptr: *mut i32, index: u64, value: i32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn store_f32(ptr: *mut f32, index: u64, value: f32) {
    unsafe {
        *ptr.add(index as usize) = value;
    }
}

#[inline(always)]
fn offset_i32_mut(ptr: *mut i32, index: u64) -> *mut i32 {
    unsafe { ptr.add(index as usize) }
}

#[inline(always)]
fn offset_f32_mut(ptr: *mut f32, index: u64) -> *mut f32 {
    unsafe { ptr.add(index as usize) }
}

#[inline(always)]
fn floor_div_pos(a: i32, d: i32) -> i32 {
    let mut q = a / d;
    let r = a - q * d;
    if r < 0 {
        q -= 1;
    }
    q
}

#[inline(always)]
fn descale(value: i32, shift: i32) -> i32 {
    (value + (1_i32 << (shift - 1))) >> shift
}

#[inline(always)]
fn clamp_level_shift(value: i32) -> i32 {
    let shifted = value + 128;
    if shifted < 0 {
        -128
    } else if shifted > 255 {
        127
    } else {
        shifted - 128
    }
}

#[allow(clippy::too_many_lines)]
#[inline(always)]
fn idct_islow_signed(input: *const i16, output: *mut i32) {
    let mut work = [0_i32; 64];
    let mut col = 0_i32;
    while col < 8 {
        let p0 = load_i16(input, col as u64) as i32;
        let p1 = load_i16(input, (col + 8) as u64) as i32;
        let p2 = load_i16(input, (col + 16) as u64) as i32;
        let p3 = load_i16(input, (col + 24) as u64) as i32;
        let p4 = load_i16(input, (col + 32) as u64) as i32;
        let p5 = load_i16(input, (col + 40) as u64) as i32;
        let p6 = load_i16(input, (col + 48) as u64) as i32;
        let p7 = load_i16(input, (col + 56) as u64) as i32;

        let mut z2 = p2;
        let mut z3 = p6;
        let mut z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;

        z2 = p0;
        z3 = p4;
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let tmp0 = p7;
        let tmp1 = p5;
        let tmp2 = p3;
        let tmp3 = p1;

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        let mut z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let mut tmp0 = tmp0 * FIX_0_298631336;
        let mut tmp1 = tmp1 * FIX_2_053119869;
        let mut tmp2 = tmp2 * FIX_3_072711026;
        let mut tmp3 = tmp3 * FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        let shift = CONST_BITS - PASS1_BITS;
        work[col as usize] = descale(tmp10 + tmp3, shift);
        work[(col + 56) as usize] = descale(tmp10 - tmp3, shift);
        work[(col + 8) as usize] = descale(tmp11 + tmp2, shift);
        work[(col + 48) as usize] = descale(tmp11 - tmp2, shift);
        work[(col + 16) as usize] = descale(tmp12 + tmp1, shift);
        work[(col + 40) as usize] = descale(tmp12 - tmp1, shift);
        work[(col + 24) as usize] = descale(tmp13 + tmp0, shift);
        work[(col + 32) as usize] = descale(tmp13 - tmp0, shift);
        col += 1;
    }

    let mut row = 0_i32;
    while row < 8 {
        let base = row * 8;
        let p0 = work[base as usize];
        let p1 = work[(base + 1) as usize];
        let p2 = work[(base + 2) as usize];
        let p3 = work[(base + 3) as usize];
        let p4 = work[(base + 4) as usize];
        let p5 = work[(base + 5) as usize];
        let p6 = work[(base + 6) as usize];
        let p7 = work[(base + 7) as usize];

        let shift = CONST_BITS + PASS1_BITS + 3;
        let mut z2 = p2;
        let mut z3 = p6;
        let mut z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;

        let tmp0 = (p0 + p4) << CONST_BITS;
        let tmp1 = (p0 - p4) << CONST_BITS;

        let tmp10 = tmp0 + tmp3;
        let tmp13 = tmp0 - tmp3;
        let tmp11 = tmp1 + tmp2;
        let tmp12 = tmp1 - tmp2;

        let tmp0 = p7;
        let tmp1 = p5;
        let tmp2 = p3;
        let tmp3 = p1;

        z1 = tmp0 + tmp3;
        z2 = tmp1 + tmp2;
        z3 = tmp0 + tmp2;
        let mut z4 = tmp1 + tmp3;
        let z5 = (z3 + z4) * FIX_1_175875602;

        let mut tmp0 = tmp0 * FIX_0_298631336;
        let mut tmp1 = tmp1 * FIX_2_053119869;
        let mut tmp2 = tmp2 * FIX_3_072711026;
        let mut tmp3 = tmp3 * FIX_1_501321110;
        z1 *= -FIX_0_899976223;
        z2 *= -FIX_2_562915447;
        z3 *= -FIX_1_961570560;
        z4 *= -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        tmp0 += z1 + z3;
        tmp1 += z2 + z4;
        tmp2 += z2 + z3;
        tmp3 += z1 + z4;

        let values = [
            descale(tmp10 + tmp3, shift),
            descale(tmp11 + tmp2, shift),
            descale(tmp12 + tmp1, shift),
            descale(tmp13 + tmp0, shift),
            descale(tmp13 - tmp0, shift),
            descale(tmp12 - tmp1, shift),
            descale(tmp11 - tmp2, shift),
            descale(tmp10 - tmp3, shift),
        ];
        let mut k = 0_usize;
        while k < 8 {
            store_i32(
                output,
                (base as u64) + k as u64,
                clamp_level_shift(values[k]),
            );
            k += 1;
        }
        row += 1;
    }
}

#[inline(always)]
fn sample_at(samples: *const i32, block_cols: i32, x: i32, y: i32) -> i32 {
    let block_idx = (y >> 3) * block_cols + (x >> 3);
    let local_idx = (y & 7) * 8 + (x & 7);
    load_i32(samples, block_idx as u64 * 64 + local_idx as u64)
}

#[inline(always)]
fn vertical_high(samples: *const i32, block_cols: i32, height: i32, x: i32, high_idx: i32) -> i32 {
    let odd_idx = high_idx * 2 + 1;
    let current = sample_at(samples, block_cols, x, odd_idx);
    let left = sample_at(samples, block_cols, x, odd_idx - 1);
    if height % 2 == 0 && odd_idx + 1 == height {
        return current - left;
    }
    let right_idx = if odd_idx + 1 < height {
        odd_idx + 1
    } else {
        height - 1
    };
    let right = sample_at(samples, block_cols, x, right_idx);
    current - floor_div_pos(left + right, 2)
}

#[inline(always)]
fn vertical_low(samples: *const i32, block_cols: i32, height: i32, x: i32, low_idx: i32) -> i32 {
    let even_idx = low_idx * 2;
    let current = sample_at(samples, block_cols, x, even_idx);
    if height < 2 {
        return current;
    }
    if height % 2 == 0 {
        let right = vertical_high(samples, block_cols, height, x, low_idx);
        if low_idx == 0 {
            return current + floor_div_pos(right + 1, 2);
        }
        let left = vertical_high(samples, block_cols, height, x, low_idx - 1);
        return current + floor_div_pos(left + right + 2, 4);
    }
    let high_len = height / 2;
    if high_len == 0 {
        return current;
    }
    let left = vertical_high(
        samples,
        block_cols,
        height,
        x,
        if low_idx > 0 { low_idx - 1 } else { 0 },
    );
    let right = if low_idx < high_len {
        vertical_high(samples, block_cols, height, x, low_idx)
    } else {
        left
    };
    current + floor_div_pos(left + right + 2, 4)
}

#[inline(always)]
fn reversible_lift_row(row: *mut i32, n: i32) {
    if n < 2 {
        return;
    }
    if n % 2 == 0 {
        let mut i = 1_i32;
        while i < n - 1 {
            let value = load_i32(row.cast_const(), i as u64)
                - floor_div_pos(
                    load_i32(row.cast_const(), (i - 1) as u64)
                        + load_i32(row.cast_const(), (i + 1) as u64),
                    2,
                );
            store_i32(row, i as u64, value);
            i += 2;
        }
        let last =
            load_i32(row.cast_const(), (n - 1) as u64) - load_i32(row.cast_const(), (n - 2) as u64);
        store_i32(row, (n - 1) as u64, last);
        store_i32(
            row,
            0,
            load_i32(row.cast_const(), 0) + floor_div_pos(load_i32(row.cast_const(), 1) + 1, 2),
        );
        let mut i = 2_i32;
        while i < n {
            let value = load_i32(row.cast_const(), i as u64)
                + floor_div_pos(
                    load_i32(row.cast_const(), (i - 1) as u64)
                        + load_i32(row.cast_const(), (i + 1) as u64)
                        + 2,
                    4,
                );
            store_i32(row, i as u64, value);
            i += 2;
        }
        return;
    }

    let last_even = n - 1;
    let mut i = 1_i32;
    while i < n {
        let right = if i + 1 < n {
            load_i32(row.cast_const(), (i + 1) as u64)
        } else {
            load_i32(row.cast_const(), last_even as u64)
        };
        let value = load_i32(row.cast_const(), i as u64)
            - floor_div_pos(load_i32(row.cast_const(), (i - 1) as u64) + right, 2);
        store_i32(row, i as u64, value);
        i += 2;
    }
    let mut i = 0_i32;
    while i < n {
        let left = if i > 0 {
            load_i32(row.cast_const(), (i - 1) as u64)
        } else {
            load_i32(row.cast_const(), 1)
        };
        let right = if i + 1 < n {
            load_i32(row.cast_const(), (i + 1) as u64)
        } else {
            left
        };
        let value = load_i32(row.cast_const(), i as u64) + floor_div_pos(left + right + 2, 4);
        store_i32(row, i as u64, value);
        i += 2;
    }
}

const DWT97_ALPHA: f32 = -1.586_134_3;
const DWT97_BETA: f32 = -0.052_980_117;
const DWT97_GAMMA: f32 = 0.882_911_1;
const DWT97_DELTA: f32 = 0.443_506_87;
const DWT97_KAPPA: f32 = 1.230_174_1;
const DWT97_INV_KAPPA: f32 = 1.0 / DWT97_KAPPA;
const DWT97_ROW_LIFT_MAX_WIDTH: usize = 1024;
const DWT97_ROW_LIFT_ROWS_PER_BLOCK: usize = 4;
const DWT97_ROW_LIFT_SHARED_SAMPLES: usize =
    DWT97_ROW_LIFT_MAX_WIDTH * DWT97_ROW_LIFT_ROWS_PER_BLOCK;

const IDCT_C0: f32 = 0.353_553_38;
const IDCT_C1: f32 = 0.490_392_65;
const IDCT_C2: f32 = 0.461_939_75;
const IDCT_C3: f32 = 0.415_734_8;
const IDCT_C5: f32 = 0.277_785_12;
const IDCT_C6: f32 = 0.191_341_71;
const IDCT_C7: f32 = 0.097_545_16;

#[inline(always)]
fn idct8_basis_0(_sample_idx: i32) -> f32 {
    IDCT_C0
}

#[inline(always)]
fn idct8_basis_1(sample_idx: i32) -> f32 {
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
fn idct8_basis_2(sample_idx: i32) -> f32 {
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
fn idct8_basis_3(sample_idx: i32) -> f32 {
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
fn idct8_basis_4(sample_idx: i32) -> f32 {
    match sample_idx {
        0 | 3 | 4 | 7 => IDCT_C0,
        1 | 2 | 5 | 6 => -IDCT_C0,
        _ => 0.0,
    }
}

#[inline(always)]
fn idct8_basis_5(sample_idx: i32) -> f32 {
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
fn idct8_basis_6(sample_idx: i32) -> f32 {
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
fn idct8_basis_7(sample_idx: i32) -> f32 {
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
fn idct8x8_sample(block: *const f32, local_x: i32, local_y: i32) -> f32 {
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
fn idct8x8_sample_i16(block: *const i16, local_x: i32, local_y: i32) -> f32 {
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
fn forward_lift_97(data: *mut f32, n: i32, stride: i32) {
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
fn floor_f32(value: f32) -> f32 {
    let truncated = value as i32 as f32;
    if truncated > value {
        truncated - 1.0
    } else {
        truncated
    }
}

#[inline(always)]
fn abs_f32(value: f32) -> f32 {
    if value < 0.0 { -value } else { value }
}

#[inline(always)]
fn min_i32(a: i32, b: i32) -> i32 {
    if a < b { a } else { b }
}

#[inline(always)]
fn quantize_dwt97_deadzone(value: f32, inv_delta: f32) -> i32 {
    let sign = if value < 0.0 { -1 } else { 1 };
    sign * floor_f32(abs_f32(value) * inv_delta) as i32
}

#[inline(always)]
fn dwt97_codeblock_major_offset(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    cb_width: i32,
    cb_height: i32,
) -> u64 {
    if cb_width == 64 && cb_height == 64 {
        let cbx = x >> 6;
        let cby = y >> 6;
        let local_x = x & 63;
        let local_y = y & 63;
        let block_width = min_i32(64, width - (cbx << 6));
        let block_height = min_i32(64, height - (cby << 6));
        return (cby as u64) * 64 * width as u64
            + (cbx as u64) * 64 * block_height as u64
            + (local_y as u64) * block_width as u64
            + local_x as u64;
    }
    let cbx = x / cb_width;
    let cby = y / cb_height;
    let local_x = x - cbx * cb_width;
    let local_y = y - cby * cb_height;
    let block_width = min_i32(cb_width, width - cbx * cb_width);
    let block_height = min_i32(cb_height, height - cby * cb_height);
    (cby as u64) * cb_height as u64 * width as u64
        + (cbx as u64) * cb_width as u64 * block_height as u64
        + (local_y as u64) * block_width as u64
        + local_x as u64
}

#[inline(always)]
fn shared_row_index(row_lane: i32, x: i32) -> u64 {
    row_lane as u64 * DWT97_ROW_LIFT_MAX_WIDTH as u64 + x as u64
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn transcode_reversible53_idct(
        blocks: *const i16,
        samples: *mut i32,
        block_count: u32,
    ) {
        let idx = thread::index_1d().get() as u32;
        if idx >= block_count {
            return;
        }
        idct_islow_signed(unsafe { blocks.add(idx as usize * 64) }, unsafe {
            samples.add(idx as usize * 64)
        });
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_vertical_low(
        samples: *const i32,
        block_cols: i32,
        width: i32,
        height: i32,
        v_low: *mut i32,
        low_height: i32,
    ) {
        let x = thread::index_2d_col() as i32;
        let yl = thread::index_2d_row() as i32;
        if x >= width || yl >= low_height {
            return;
        }
        store_i32(
            v_low,
            (yl * width + x) as u64,
            vertical_low(samples, block_cols, height, x, yl),
        );
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_vertical_high(
        samples: *const i32,
        block_cols: i32,
        width: i32,
        height: i32,
        v_high: *mut i32,
        high_height: i32,
    ) {
        let x = thread::index_2d_col() as i32;
        let yh = thread::index_2d_row() as i32;
        if x >= width || yh >= high_height {
            return;
        }
        store_i32(
            v_high,
            (yh * width + x) as u64,
            vertical_high(samples, block_cols, height, x, yh),
        );
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_horizontal_low(
        v_low: *mut i32,
        width: i32,
        low_height: i32,
        low_width: i32,
        high_width: i32,
        ll: *mut i32,
        hl: *mut i32,
    ) {
        let yl = thread::index_1d().get() as i32;
        if yl >= low_height {
            return;
        }
        let row = offset_i32_mut(v_low, (yl * width) as u64);
        reversible_lift_row(row, width);
        let mut i = 0_i32;
        while i < low_width {
            store_i32(
                ll,
                (yl * low_width + i) as u64,
                load_i32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_i32(
                hl,
                (yl * high_width + i) as u64,
                load_i32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_reversible53_horizontal_high(
        v_high: *mut i32,
        width: i32,
        high_height: i32,
        low_width: i32,
        high_width: i32,
        lh: *mut i32,
        hh: *mut i32,
    ) {
        let yh = thread::index_1d().get() as i32;
        if yh >= high_height {
            return;
        }
        let row = offset_i32_mut(v_high, (yh * width) as u64);
        reversible_lift_row(row, width);
        let mut i = 0_i32;
        while i < low_width {
            store_i32(
                lh,
                (yh * low_width + i) as u64,
                load_i32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_i32(
                hh,
                (yh * high_width + i) as u64,
                load_i32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_idct(
        blocks: *const f32,
        block_cols: i32,
        width: i32,
        height: i32,
        spatial: *mut f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        if x >= width || y >= height {
            return;
        }
        let block_idx = (y >> 3) * block_cols + (x >> 3);
        let block = unsafe { blocks.add(block_idx as usize * 64) };
        store_f32(
            spatial,
            (y * width + x) as u64,
            idct8x8_sample(block, x & 7, y & 7),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_row_lift(
        spatial: *mut f32,
        width: i32,
        height: i32,
        low_width: i32,
        high_width: i32,
        row_low: *mut f32,
        row_high: *mut f32,
    ) {
        let y = thread::index_1d().get() as i32;
        if y >= height {
            return;
        }
        let row = offset_f32_mut(spatial, (y * width) as u64);
        forward_lift_97(row, width, 1);
        let mut i = 0_i32;
        while i < low_width {
            store_f32(
                row_low,
                (y * low_width + i) as u64,
                load_f32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_f32(
                row_high,
                (y * high_width + i) as u64,
                load_f32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_column_lift(
        rows: *mut f32,
        band_width: i32,
        height: i32,
        low_out: *mut f32,
        high_out: *mut f32,
    ) {
        let x = thread::index_1d().get() as i32;
        if x >= band_width {
            return;
        }
        forward_lift_97(offset_f32_mut(rows, x as u64), height, band_width);
        let mut i = 0_i32;
        while i < height {
            let value = load_f32(rows.cast_const(), (i * band_width + x) as u64);
            if i & 1 == 0 {
                store_f32(low_out, ((i / 2) * band_width + x) as u64, value);
            } else {
                store_f32(high_out, ((i / 2) * band_width + x) as u64, value);
            }
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_idct_batch(
        blocks: *const f32,
        block_cols: i32,
        width: i32,
        height: i32,
        blocks_per_item: i32,
        spatial: *mut f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        let item = thread::blockIdx_z() as u64;
        if x >= width || y >= height {
            return;
        }
        let item_blocks = unsafe { blocks.add((item * blocks_per_item as u64 * 64) as usize) };
        let block_idx = (y >> 3) * block_cols + (x >> 3);
        let block = unsafe { item_blocks.add(block_idx as usize * 64) };
        store_f32(
            spatial,
            (item * height as u64 + y as u64) * width as u64 + x as u64,
            idct8x8_sample(block, x & 7, y & 7),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_idct_i16_batch(
        blocks: *const i16,
        block_cols: i32,
        width: i32,
        height: i32,
        blocks_per_item: i32,
        spatial: *mut f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        let item = thread::blockIdx_z() as u64;
        if x >= width || y >= height {
            return;
        }
        let item_blocks = unsafe { blocks.add((item * blocks_per_item as u64 * 64) as usize) };
        let block_idx = (y >> 3) * block_cols + (x >> 3);
        let block = unsafe { item_blocks.add(block_idx as usize * 64) };
        store_f32(
            spatial,
            (item * height as u64 + y as u64) * width as u64 + x as u64,
            idct8x8_sample_i16(block, x & 7, y & 7),
        );
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_row_lift_batch(
        spatial: *mut f32,
        width: i32,
        height: i32,
        low_width: i32,
        high_width: i32,
        row_low: *mut f32,
        row_high: *mut f32,
    ) {
        let y = thread::blockIdx_x() as i32 * thread::blockDim_x() as i32
            + thread::threadIdx_x() as i32;
        let item = thread::blockIdx_y() as u64;
        if y >= height {
            return;
        }
        let item_spatial = offset_f32_mut(spatial, item * width as u64 * height as u64);
        let item_row_low = offset_f32_mut(row_low, item * height as u64 * low_width as u64);
        let item_row_high = offset_f32_mut(row_high, item * height as u64 * high_width as u64);
        let row = offset_f32_mut(item_spatial, y as u64 * width as u64);
        forward_lift_97(row, width, 1);
        let mut i = 0_i32;
        while i < low_width {
            store_f32(
                item_row_low,
                (y * low_width + i) as u64,
                load_f32(row.cast_const(), (i * 2) as u64),
            );
            i += 1;
        }
        let mut i = 0_i32;
        while i < high_width {
            store_f32(
                item_row_high,
                (y * high_width + i) as u64,
                load_f32(row.cast_const(), (i * 2 + 1) as u64),
            );
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_row_lift_batch_coop(
        spatial: *const f32,
        width: i32,
        height: i32,
        low_width: i32,
        high_width: i32,
        row_low: *mut f32,
        row_high: *mut f32,
    ) {
        static mut ROWS: SharedArray<f32, DWT97_ROW_LIFT_SHARED_SAMPLES> = SharedArray::UNINIT;

        let rows = unsafe { ROWS.as_mut_ptr() };
        let row_lane = thread::threadIdx_y() as i32;
        let tid = thread::threadIdx_x() as i32;
        let block_step = thread::blockDim_x() as i32;
        let y = thread::blockIdx_x() as i32 * DWT97_ROW_LIFT_ROWS_PER_BLOCK as i32 + row_lane;
        let item = thread::blockIdx_y() as u64;
        let valid = y < height && width <= DWT97_ROW_LIFT_MAX_WIDTH as i32;

        if valid {
            let item_spatial =
                unsafe { spatial.add((item * width as u64 * height as u64) as usize) };
            let source = unsafe { item_spatial.add((y as u64 * width as u64) as usize) };
            let mut i = tid;
            while i < width {
                store_f32(
                    rows,
                    shared_row_index(row_lane, i),
                    load_f32(source, i as u64),
                );
                i += block_step;
            }
        }
        thread::sync_threads();

        if width >= 2 && width <= DWT97_ROW_LIFT_MAX_WIDTH as i32 {
            if valid {
                let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
                let mut i = tid * 2 + 1;
                while i < width {
                    let left = load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1));
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, last_even))
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_ALPHA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let mut i = tid * 2;
                while i < width {
                    let left = if i > 0 {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, 1))
                    };
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        left
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_BETA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let last_even = if width % 2 == 0 { width - 2 } else { width - 1 };
                let mut i = tid * 2 + 1;
                while i < width {
                    let left = load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1));
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, last_even))
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_GAMMA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let mut i = tid * 2;
                while i < width {
                    let left = if i > 0 {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i - 1))
                    } else {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, 1))
                    };
                    let right = if i + 1 < width {
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i + 1))
                    } else {
                        left
                    };
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        + DWT97_DELTA * (left + right);
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();

            if valid {
                let mut i = tid * 2;
                while i < width {
                    let value = load_f32(rows.cast_const(), shared_row_index(row_lane, i))
                        * DWT97_INV_KAPPA;
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
                let mut i = tid * 2 + 1;
                while i < width {
                    let value =
                        load_f32(rows.cast_const(), shared_row_index(row_lane, i)) * DWT97_KAPPA;
                    store_f32(rows, shared_row_index(row_lane, i), value);
                    i += block_step * 2;
                }
            }
            thread::sync_threads();
        }

        if valid {
            let item_row_low = offset_f32_mut(row_low, item * height as u64 * low_width as u64);
            let item_row_high = offset_f32_mut(row_high, item * height as u64 * high_width as u64);
            let mut i = tid;
            while i < low_width {
                store_f32(
                    item_row_low,
                    (y * low_width + i) as u64,
                    load_f32(rows.cast_const(), shared_row_index(row_lane, i * 2)),
                );
                i += block_step;
            }
            let mut i = tid;
            while i < high_width {
                store_f32(
                    item_row_high,
                    (y * high_width + i) as u64,
                    load_f32(rows.cast_const(), shared_row_index(row_lane, i * 2 + 1)),
                );
                i += block_step;
            }
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_column_lift_batch(
        rows: *mut f32,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        low_out: *mut f32,
        high_out: *mut f32,
    ) {
        let x = thread::blockIdx_x() as i32 * thread::blockDim_x() as i32
            + thread::threadIdx_x() as i32;
        let item = thread::blockIdx_y() as u64;
        if x >= band_width {
            return;
        }
        let item_rows = offset_f32_mut(rows, item * height as u64 * band_width as u64);
        let item_low = offset_f32_mut(low_out, item * low_height as u64 * band_width as u64);
        let item_high = offset_f32_mut(high_out, item * high_height as u64 * band_width as u64);
        forward_lift_97(offset_f32_mut(item_rows, x as u64), height, band_width);
        let mut i = 0_i32;
        while i < height {
            let value = load_f32(item_rows.cast_const(), (i * band_width + x) as u64);
            if i & 1 == 0 {
                store_f32(item_low, ((i / 2) * band_width + x) as u64, value);
            } else {
                store_f32(item_high, ((i / 2) * band_width + x) as u64, value);
            }
            i += 1;
        }
    }

    #[kernel]
    pub unsafe fn transcode_dwt97_quantize_codeblocks(
        band: *const f32,
        output: *mut i32,
        width: i32,
        height: i32,
        cb_width: i32,
        cb_height: i32,
        inv_delta: f32,
    ) {
        let x = thread::index_2d_col() as i32;
        let y = thread::index_2d_row() as i32;
        let item = thread::blockIdx_z() as u64;
        if x >= width || y >= height {
            return;
        }
        let item_stride = width as u64 * height as u64;
        let value = load_f32(
            band,
            item * item_stride + y as u64 * width as u64 + x as u64,
        );
        let offset = dwt97_codeblock_major_offset(x, y, width, height, cb_width, cb_height);
        store_i32(
            output,
            item * item_stride + offset,
            quantize_dwt97_deadzone(value, inv_delta),
        );
    }

    #[kernel]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn transcode_dwt97_column_lift_quantize_codeblocks_batch(
        rows: *mut f32,
        band_width: i32,
        height: i32,
        low_height: i32,
        high_height: i32,
        low_out: *mut i32,
        high_out: *mut i32,
        cb_width: i32,
        cb_height: i32,
        inv_delta_low: f32,
        inv_delta_high: f32,
    ) {
        let x = thread::blockIdx_x() as i32 * thread::blockDim_x() as i32
            + thread::threadIdx_x() as i32;
        let item = thread::blockIdx_y() as u64;
        if x >= band_width {
            return;
        }
        let item_rows = offset_f32_mut(rows, item * height as u64 * band_width as u64);
        let item_low = offset_i32_mut(low_out, item * low_height as u64 * band_width as u64);
        let item_high = offset_i32_mut(high_out, item * high_height as u64 * band_width as u64);

        forward_lift_97(offset_f32_mut(item_rows, x as u64), height, band_width);
        let mut i = 0_i32;
        while i < height {
            let value = load_f32(item_rows.cast_const(), (i * band_width + x) as u64);
            if i & 1 == 0 {
                let y = i / 2;
                let offset =
                    dwt97_codeblock_major_offset(x, y, band_width, low_height, cb_width, cb_height);
                store_i32(
                    item_low,
                    offset,
                    quantize_dwt97_deadzone(value, inv_delta_low),
                );
            } else {
                let y = i / 2;
                let offset = dwt97_codeblock_major_offset(
                    x,
                    y,
                    band_width,
                    high_height,
                    cb_width,
                    cb_height,
                );
                store_i32(
                    item_high,
                    offset,
                    quantize_dwt97_deadzone(value, inv_delta_high),
                );
            }
            i += 1;
        }
    }
}

fn main() {}
