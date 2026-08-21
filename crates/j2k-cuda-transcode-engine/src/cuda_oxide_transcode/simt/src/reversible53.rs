// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    constants::{
        CONST_BITS, FIX_0_298631336, FIX_0_390180644, FIX_0_541196100, FIX_0_765366865,
        FIX_0_899976223, FIX_1_175875602, FIX_1_501321110, FIX_1_847759065, FIX_1_961570560,
        FIX_2_053119869, FIX_2_562915447, FIX_3_072711026, PASS1_BITS,
    },
    helpers::{floor_div_pos, load_i16, load_i32, store_i32},
};

#[inline(always)]
pub(crate) fn descale(value: i32, shift: i32) -> i32 {
    (value + (1_i32 << (shift - 1))) >> shift
}

#[inline(always)]
pub(crate) fn clamp_level_shift(value: i32) -> i32 {
    let shifted = value + 128;
    if shifted < 0 {
        -128
    } else if shifted > 255 {
        127
    } else {
        shifted - 128
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "the integer IDCT keeps its two ordered fixed-point passes cohesive for device codegen"
)]
#[inline(always)]
pub(crate) fn idct_islow_signed(input: *const i16, output: *mut i32) {
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
pub(crate) fn sample_at(samples: *const i32, block_cols: i32, x: i32, y: i32) -> i32 {
    let block_idx = (y >> 3) * block_cols + (x >> 3);
    let local_idx = (y & 7) * 8 + (x & 7);
    load_i32(samples, block_idx as u64 * 64 + local_idx as u64)
}

#[inline(always)]
pub(crate) fn vertical_high(
    samples: *const i32,
    block_cols: i32,
    height: i32,
    x: i32,
    high_idx: i32,
) -> i32 {
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
pub(crate) fn vertical_low(
    samples: *const i32,
    block_cols: i32,
    height: i32,
    x: i32,
    low_idx: i32,
) -> i32 {
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
pub(crate) fn reversible_lift_row(row: *mut i32, n: i32) {
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
