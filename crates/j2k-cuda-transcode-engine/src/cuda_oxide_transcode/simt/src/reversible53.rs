// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::helpers::{floor_div_pos, load_i16, load_i32, store_i32};
use j2k_codec_math::jpeg::idct::{islow_column, islow_row, sample_centered};

/// The 8-bit ISLOW of one block to samples centred on zero, with
/// `jidctint.c`'s AC-zero shortcuts, as the CPU oracle
/// (`j2k_codec_math::jpeg::idct::islow_block`) takes them. The row shortcut is
/// not an optimisation only: a row whose DC reaches 2^18 wraps in the full
/// pass, so computing every row in full returns other samples.
#[inline(always)]
pub(crate) fn idct_islow_signed(input: *const i16, output: *mut i32) {
    let mut work = [0_i32; 64];
    let mut col = 0_usize;
    while col < 8 {
        let mut p = [0_i32; 8];
        let mut row = 0_usize;
        while row < 8 {
            p[row] = load_i16(input, (row * 8 + col) as u64) as i32;
            row += 1;
        }
        let out = islow_column(p);
        let mut row = 0_usize;
        while row < 8 {
            work[row * 8 + col] = out[row];
            row += 1;
        }
        col += 1;
    }

    let mut row = 0_usize;
    while row < 8 {
        let base = row * 8;
        let out = islow_row([
            work[base],
            work[base + 1],
            work[base + 2],
            work[base + 3],
            work[base + 4],
            work[base + 5],
            work[base + 6],
            work[base + 7],
        ]);
        let mut k = 0_usize;
        while k < 8 {
            store_i32(output, (base + k) as u64, sample_centered(out[k]));
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
