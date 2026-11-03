// SPDX-License-Identifier: MIT OR Apache-2.0
//
//! Proptest-driven bit-exact parity between the scalar ISLOW IDCT and every
//! SIMD backend variant.
//!
//! The test only exercises the SIMD variant the host CPU can actually run —
//! we do not cross-compile to the other arch. CI runs on both aarch64 and
//! `x86_64` so each backend is exercised at least once.

#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use proptest::prelude::*;

#[cfg(target_arch = "x86_64")]
use j2k_jpeg::bench_support::BenchAvx2Idct;

fn scalar(input: &[i16; 64]) -> [u8; 64] {
    let mut out = [0u8; 64];
    j2k_jpeg::bench_support::bench_idct_reference_block_with(input, &mut out);
    out
}

#[cfg(target_arch = "aarch64")]
fn assert_neon_matches_scalar(input: &[i16; 64]) {
    let scalar_out = scalar(input);
    let mut neon_out = [0u8; 64];
    j2k_jpeg::bench_support::bench_idct_neon_block(input, &mut neon_out);
    assert_eq!(
        scalar_out, neon_out,
        "NEON IDCT diverged from scalar on input {input:?}"
    );
}

#[cfg(target_arch = "x86_64")]
fn assert_avx2_matches_scalar(input: &[i16; 64]) {
    let scalar_out = scalar(input);
    let mut avx_out = [0u8; 64];
    j2k_jpeg::bench_support::bench_idct_avx2_block(input, &mut avx_out);
    assert_eq!(
        scalar_out, avx_out,
        "AVX2 IDCT diverged from scalar on input {input:?}"
    );
}

/// Coefficients only in natural-order rows `0..rows` and columns `0..cols`,
/// the shapes the sparse IDCT entry points accept.
fn shaped_blocks(
    rows: usize,
    cols: usize,
    values: impl Strategy<Value = i16>,
) -> impl Strategy<Value = [i16; 64]> {
    prop::collection::vec(values, rows * cols).prop_map(move |values| {
        let mut block = [0i16; 64];
        for (index, value) in values.into_iter().enumerate() {
            block[(index / cols) * 8 + index % cols] = value;
        }
        block
    })
}

fn small_coefficients() -> impl Strategy<Value = [i16; 64]> {
    // Typical JPEG space after dequantization: DC up to ~1024, AC small.
    shaped_blocks(8, 8, -512i16..512)
}

fn large_coefficients() -> impl Strategy<Value = [i16; 64]> {
    // Full i16 range — exercises wrapping-then-clamp on saturating inputs.
    shaped_blocks(8, 8, any::<i16>())
}

fn sparse_blocks() -> impl Strategy<Value = [i16; 64]> {
    any::<i16>().prop_map(|dc| {
        let mut block = [0i16; 64];
        block[0] = dc;
        block
    })
}

fn bottom_half_zero_blocks() -> impl Strategy<Value = [i16; 64]> {
    shaped_blocks(4, 8, -512i16..512)
}

fn top_left_2x2_blocks() -> impl Strategy<Value = [i16; 64]> {
    prop_oneof![
        4 => prop::array::uniform4(any::<i16>()),
        2 => (any::<i16>(), any::<i16>()).prop_map(|(a, b)| [a, 0, b, 0]),
        2 => (any::<i16>(), any::<i16>()).prop_map(|(a, b)| [a, b, 0, 0]),
        1 => (any::<i16>(), any::<i16>()).prop_map(|(a, b)| [0, a, 0, b]),
        1 => (any::<i16>(), any::<i16>()).prop_map(|(a, b)| [0, 0, a, b]),
        1 => Just([i16::MAX, 0, i16::MAX, 0]),
        1 => Just([i16::MIN, 0, i16::MIN, 0]),
        1 => Just([i16::MAX, 0, i16::MIN, 0]),
        1 => Just([i16::MIN, 0, i16::MAX, 0]),
    ]
    .prop_map(|coefficients| {
        let mut block = [0i16; 64];
        for (index, coefficient) in [0, 1, 8, 9].into_iter().zip(coefficients) {
            block[index] = coefficient;
        }
        block
    })
}

/// Coefficients small enough that columns of `rows` of them stay within the
/// x86 16-bit kernel's bound, where it runs instead of the 32-bit fallback.
#[cfg(target_arch = "x86_64")]
fn within_16_bit_bound(rows: i16) -> core::ops::RangeInclusive<i16> {
    let max = BenchAvx2Idct::MAX_COLUMN_ABS_SUM / rows;
    -max..=max
}

/// One nonzero column: every first-pass row is DC-only, which the scalar row
/// transform shortcuts.
#[cfg(target_arch = "x86_64")]
fn single_column_blocks() -> impl Strategy<Value = [i16; 64]> {
    (0usize..8, prop::array::uniform8(any::<i16>())).prop_map(|(col, column)| {
        let mut block = [0i16; 64];
        for (row, value) in column.into_iter().enumerate() {
            block[row * 8 + col] = value;
        }
        block
    })
}

/// Columns whose absolute values sum to exactly the 16-bit kernel's bound,
/// with random signs: seven random cut points split each column.
#[cfg(target_arch = "x86_64")]
fn columns_at_the_16_bit_bound() -> impl Strategy<Value = [i16; 64]> {
    let bound = BenchAvx2Idct::MAX_COLUMN_ABS_SUM;
    let column = (
        prop::array::uniform7(0i16..=bound),
        prop::array::uniform8(any::<bool>()),
    )
        .prop_map(move |(mut cuts, negative)| {
            cuts.sort_unstable();
            core::array::from_fn::<i16, 8, _>(|row| {
                let low = if row == 0 { 0 } else { cuts[row - 1] };
                let high = if row == 7 { bound } else { cuts[row] };
                if negative[row] {
                    low - high
                } else {
                    high - low
                }
            })
        });
    prop::array::uniform8(column)
        .prop_map(|columns| core::array::from_fn(|index| columns[index % 8][index / 8]))
}

/// Each x86 sparse entry point and the dispatch equal scalar on a block of
/// that entry point's shape.
#[cfg(target_arch = "x86_64")]
fn assert_avx2_shape_matches_scalar(rows: usize, cols: usize, input: &[i16; 64]) {
    assert_avx2_matches_scalar(input);
    let Some(avx2) = BenchAvx2Idct::try_new() else {
        return;
    };
    let shape = BenchAvx2Idct::SHAPES
        .into_iter()
        .find(|shape| (shape.rows, shape.cols) == (rows, cols))
        .unwrap_or_else(|| panic!("no {rows}x{cols} entry point"));
    let expected = scalar(input);
    let mut actual = [0u8; 64];
    avx2.run_shape(shape, input, &mut actual);
    assert_eq!(
        actual, expected,
        "{} kernel diverged on {input:?}",
        shape.name
    );
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2_000,
        .. ProptestConfig::default()
    })]

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_small_coefficients(block in small_coefficients()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_large_coefficients(block in large_coefficients()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_sparse_blocks(block in sparse_blocks()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_on_bottom_half_zero_blocks(block in bottom_half_zero_blocks()) {
        assert_neon_matches_scalar(&block);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_top_left_2x2_matches_scalar_on_full_i16_coefficients(block in top_left_2x2_blocks()) {
        let expected = scalar(&block);
        let mut actual = [0u8; 64];
        j2k_jpeg::bench_support::bench_idct_neon_top_left_2x2_block(&block, &mut actual);
        prop_assert_eq!(actual, expected);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_small_coefficients(block in small_coefficients()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_large_coefficients(block in large_coefficients()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_sparse_blocks(block in sparse_blocks()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_bottom_half_zero_blocks(block in bottom_half_zero_blocks()) {
        assert_avx2_shape_matches_scalar(4, 8, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_within_the_16_bit_bound(block in shaped_blocks(8, 8, within_16_bit_bound(8))) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_single_column_blocks(block in single_column_blocks()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_columns_at_the_16_bit_bound(block in columns_at_the_16_bit_bound()) {
        assert_avx2_matches_scalar(&block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_full_range_bottom_half_zero_blocks(block in shaped_blocks(4, 8, any::<i16>())) {
        assert_avx2_shape_matches_scalar(4, 8, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_in_range_bottom_half_zero_blocks(block in shaped_blocks(4, 8, within_16_bit_bound(4))) {
        assert_avx2_shape_matches_scalar(4, 8, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_right_half_zero_blocks(block in shaped_blocks(8, 4, any::<i16>())) {
        assert_avx2_shape_matches_scalar(8, 4, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_in_range_right_half_zero_blocks(block in shaped_blocks(8, 4, within_16_bit_bound(8))) {
        assert_avx2_shape_matches_scalar(8, 4, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_top_left_quadrant_blocks(block in shaped_blocks(4, 4, any::<i16>())) {
        assert_avx2_shape_matches_scalar(4, 4, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_in_range_top_left_quadrant_blocks(block in shaped_blocks(4, 4, within_16_bit_bound(4))) {
        assert_avx2_shape_matches_scalar(4, 4, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_full_range_top_left_2x2_blocks(block in shaped_blocks(2, 2, any::<i16>())) {
        assert_avx2_shape_matches_scalar(2, 2, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_in_range_top_left_2x2_blocks(block in shaped_blocks(2, 2, within_16_bit_bound(2))) {
        assert_avx2_shape_matches_scalar(2, 2, &block);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_on_top_left_2x2_blocks(block in top_left_2x2_blocks()) {
        assert_avx2_shape_matches_scalar(2, 2, &block);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_scalar_on_hand_picked_edges() {
    assert_neon_matches_scalar(&[0; 64]);

    let mut block = [0i16; 64];
    block[0] = i16::MAX;
    assert_neon_matches_scalar(&block);
    block[0] = i16::MIN;
    assert_neon_matches_scalar(&block);

    assert_neon_matches_scalar(&[i16::MAX; 64]);
    assert_neon_matches_scalar(&[i16::MIN; 64]);
    assert_neon_matches_scalar(&[1; 64]);
    assert_neon_matches_scalar(&[-1; 64]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn avx2_matches_scalar_on_hand_picked_edges() {
    assert_avx2_matches_scalar(&[0; 64]);

    let mut block = [0i16; 64];
    block[0] = i16::MAX;
    assert_avx2_matches_scalar(&block);
    block[0] = i16::MIN;
    assert_avx2_matches_scalar(&block);

    assert_avx2_matches_scalar(&[i16::MAX; 64]);
    assert_avx2_matches_scalar(&[i16::MIN; 64]);
    assert_avx2_matches_scalar(&[1; 64]);
    assert_avx2_matches_scalar(&[-1; 64]);
}
