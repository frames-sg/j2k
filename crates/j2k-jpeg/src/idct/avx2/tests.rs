// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exactness of the AVX2 IDCT against `scalar::idct_islow` on structured
//! inputs: the dispatch everywhere, the 16-bit kernel inside its range bound,
//! the bound's tightness, and the 32-bit kernel outside it. Randomized
//! coverage lives in `tests/idct_parity.rs`.

use super::{
    first_pass_fits_i16, idct_islow, idct_islow_i16, idct_islow_i32, ISLOW_GAINS,
    MAX_COLUMN_ABS_SUM,
};
use crate::idct::scalar::idct_islow as idct_scalar;
use crate::simd::x86::{exact_avx2_kernel, ExactAvx2};

exact_avx2_kernel! {
    fn fits_i16(avx2: ExactAvx2, input: &[i16; 64]) -> bool {
        first_pass_fits_i16(input)
    }
}

exact_avx2_kernel! {
    fn run_i16_unchecked(avx2: ExactAvx2, input: &[i16; 64], output: &mut [u8; 64]) {
        idct_islow_i16(input, output);
    }
}

exact_avx2_kernel! {
    fn run_i32(avx2: ExactAvx2, input: &[i16; 64], output: &mut [u8; 64]) {
        idct_islow_i32(input, output);
    }
}

fn scalar(input: &[i16; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    idct_scalar(input, &mut out);
    out
}

fn dispatch(avx2: ExactAvx2, input: &[i16; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    idct_islow(avx2, input, &mut out);
    out
}

fn i16_kernel(avx2: ExactAvx2, input: &[i16; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    run_i16_unchecked(avx2, input, &mut out);
    out
}

fn i32_kernel(avx2: ExactAvx2, input: &[i16; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    run_i32(avx2, input, &mut out);
    out
}

/// Largest per-column `sum(|x|)`, the quantity the range check bounds.
fn max_column_abs_sum(input: &[i16; 64]) -> i32 {
    (0..8)
        .map(|col| {
            (0..8)
                .map(|row| i32::from(input[row * 8 + col]).abs())
                .sum::<i32>()
        })
        .max()
        .expect("eight columns")
}

/// The dispatch and the 32-bit kernel always equal scalar; the check accepts
/// exactly the blocks within the bound, and on those the 16-bit kernel
/// equals scalar too.
fn assert_exact(avx2: ExactAvx2, input: &[i16; 64]) {
    let expected = scalar(input);
    assert_eq!(dispatch(avx2, input), expected, "dispatch on {input:?}");
    assert_eq!(
        i32_kernel(avx2, input),
        expected,
        "32-bit kernel on {input:?}"
    );
    let within = max_column_abs_sum(input) <= MAX_COLUMN_ABS_SUM;
    assert_eq!(fits_i16(avx2, input), within, "range check on {input:?}");
    if within {
        assert_eq!(
            i16_kernel(avx2, input),
            expected,
            "16-bit kernel on {input:?}"
        );
    }
}

fn set_column(block: &mut [i16; 64], col: usize, values: [i16; 8]) {
    for (row, value) in values.into_iter().enumerate() {
        block[row * 8 + col] = value;
    }
}

/// `magnitude` carrying the sign of `gain`.
fn signed_like(gain: i32, magnitude: i16) -> i16 {
    if gain < 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[test]
fn avx2_matches_scalar_on_all_zero() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    assert_exact(avx2, &[0; 64]);
}

#[test]
fn avx2_matches_scalar_on_dc_only() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let mut input = [0i16; 64];
    input[0] = 8 * 8;
    assert_exact(avx2, &input);
}

#[test]
fn avx2_matches_scalar_on_mixed_coefficients() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let mut input = [0i16; 64];
    input[0] = 64;
    input[1] = 24;
    input[2] = -12;
    input[8] = 18;
    input[9] = -7;
    input[16] = 5;
    assert_exact(avx2, &input);
}

#[test]
fn avx2_matches_scalar_on_saturation() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let mut input = [0i16; 64];
    input[0] = i16::MAX;
    assert_exact(avx2, &input);

    input[0] = i16::MIN;
    assert_exact(avx2, &input);
}

#[test]
fn avx2_matches_scalar_on_horizontal_basis() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let mut input = [0i16; 64];
    input[1] = 400;
    assert_exact(avx2, &input);
}

#[test]
fn every_single_coefficient_block_matches_scalar() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let values = [
        1, -1, 2, -2, 7, -7, 128, -128, 1_023, -1_024, 2_047, -2_048, 4_095, -4_096, 5_904, -5_904,
        5_905, -5_905, 5_906, -5_906, 8_191, -8_192, 16_383, -16_384, 32_767, -32_768,
    ];
    for position in 0..64 {
        for value in values {
            let mut block = [0i16; 64];
            block[position] = value;
            assert_exact(avx2, &block);
        }
    }
}

#[test]
fn worst_case_columns_at_the_bound_match_scalar() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let bound = i16::try_from(MAX_COLUMN_ABS_SUM).expect("fits");
    // 7 * 738 + 739 = 5,905: the bound spread over a whole column.
    let spread = [738, 738, 738, 738, 738, 738, 738, 739];
    assert_eq!(
        spread.iter().map(|&v| i32::from(v)).sum::<i32>(),
        MAX_COLUMN_ABS_SUM
    );
    for (output, gains) in ISLOW_GAINS.iter().enumerate() {
        let strongest = (0..8)
            .max_by_key(|&k| gains[k].abs())
            .expect("eight inputs");
        for col in 0..8 {
            // All weight on the largest gain into this output, both signs.
            for value in [bound, -bound] {
                let mut block = [0i16; 64];
                block[strongest * 8 + col] = value;
                assert_exact(avx2, &block);
            }
            // The bound spread with this output's sign pattern, and negated.
            for flip in [1, -1] {
                let mut block = [0i16; 64];
                let column = core::array::from_fn(|k| flip * signed_like(gains[k], spread[k]));
                set_column(&mut block, col, column);
                assert!(fits_i16(avx2, &block), "output {output} col {col}");
                assert_exact(avx2, &block);
            }
        }
    }
}

#[test]
fn bound_is_tight_one_past_it_the_unchecked_16_bit_kernel_diverges() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    let past = i16::try_from(MAX_COLUMN_ABS_SUM + 1).expect("fits");
    for col in 0..8 {
        // Input row 1 has the largest gain (11,363) into output row 0: its
        // first-pass value is (11,363 * 5,906 + 1,024) >> 11 = 32,769.
        let mut block = [0i16; 64];
        block[8 + col] = past;
        assert!(!fits_i16(avx2, &block), "col {col}");
        assert_exact(avx2, &block);
        assert_ne!(
            i16_kernel(avx2, &block),
            scalar(&block),
            "col {col}: the 16-bit kernel would be wrong past the bound"
        );
    }
}

#[test]
fn column_sums_saturate_instead_of_wrapping() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    // Two i16::MIN magnitudes sum to 65,536, which wraps to 0 in u16; four
    // and eight wrap to 0 as well.
    for rows in [2, 4, 8] {
        let mut block = [0i16; 64];
        for row in 0..rows {
            block[row * 8 + 3] = i16::MIN;
        }
        assert!(!fits_i16(avx2, &block), "{rows} rows");
        assert_exact(avx2, &block);
    }
    let mut block = [0i16; 64];
    block[5] = i16::MIN;
    block[13] = i16::MIN;
    block[21] = 1;
    assert!(!fits_i16(avx2, &block));
    assert_exact(avx2, &block);
}

#[test]
fn rows_reduced_to_dc_match_the_scalar_shortcut() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    // One nonzero column makes every first-pass row DC-only. From 2^18 up
    // the scalar's `(p0 + 16) >> 5` differs from the wrapped full transform,
    // which the 32-bit kernel must not fall back to.
    for col in 0..8 {
        for magnitude in [1, 100, 738, 1_024, 4_096, 16_384, i16::MAX] {
            for gains in &ISLOW_GAINS {
                let mut block = [0i16; 64];
                set_column(
                    &mut block,
                    col,
                    core::array::from_fn(|k| signed_like(gains[k], magnitude)),
                );
                assert_exact(avx2, &block);
            }
            let mut block = [0i16; 64];
            set_column(&mut block, col, [magnitude; 8]);
            assert_exact(avx2, &block);
        }
        let mut block = [0i16; 64];
        set_column(&mut block, col, [i16::MIN; 8]);
        assert_exact(avx2, &block);
    }
}

#[test]
fn extreme_dense_blocks_match_scalar() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    for block in [[i16::MAX; 64], [i16::MIN; 64], [1; 64], [-1; 64]] {
        assert_exact(avx2, &block);
    }
    let alternating: [i16; 64] =
        core::array::from_fn(|index| if index % 2 == 0 { i16::MAX } else { i16::MIN });
    assert_exact(avx2, &alternating);
    let checkerboard: [i16; 64] = core::array::from_fn(|index| {
        if (index / 8 + index % 8) % 2 == 0 {
            738
        } else {
            -738
        }
    });
    assert_exact(avx2, &checkerboard);
}

#[test]
fn every_support_of_the_top_left_quadrant_matches_scalar() {
    let Some(avx2) = ExactAvx2::detect() else {
        return;
    };
    for mask in 0u16..=u16::MAX {
        let block: [i16; 64] = core::array::from_fn(|index| {
            let (row, col) = (index / 8, index % 8);
            let bit = row * 4 + col;
            if row < 4 && col < 4 && mask & (1 << bit) != 0 {
                // Deterministic values in [-1476, 1476]: four rows stay
                // within the column bound.
                let mixed = u32::from(mask)
                    .wrapping_mul(2_654_435_761)
                    .rotate_left(u32::try_from(bit).expect("small"));
                i16::try_from(mixed % 2_953).expect("small") - 1_476
            } else {
                0
            }
        });
        assert_exact(avx2, &block);
    }
}
