// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(dead_code, unreachable_pub)]
#[path = "support/dct53_1d.rs"]
mod dct53_1d;
#[allow(dead_code, unreachable_pub, unused_imports)]
#[path = "../src/dct_grid.rs"]
mod dct_grid;
#[allow(dead_code, unused_imports)]
#[path = "../src/reversible53.rs"]
mod reversible53;

use dct53_1d::{
    dct8_blocks_to_dwt53_float_linear, dct8_blocks_to_dwt53_float_linear_with_len,
    dct8_to_dwt53_float_linear, dct8_to_dwt53_reversible_i16, idct8_blocks_then_dwt53_float,
    idct8_blocks_then_dwt53_float_with_len, idct8_rounded_then_dwt53_reversible,
    idct8_then_dwt53_float, Dwt53OneLevel, Dwt53Row,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn linear_multi_block_mapping_matches_reference_for_generated_coefficients(
        blocks in proptest::collection::vec(proptest::array::uniform8(-256_i16..=256), 1..5)
    ) {
        let blocks: Vec<[f64; 8]> = blocks
            .into_iter()
            .map(|block| block.map(f64::from))
            .collect();

        let direct = dct8_blocks_to_dwt53_float_linear(&blocks);
        let reference = idct8_blocks_then_dwt53_float(&blocks);

        prop_assert_eq!(direct.low.len(), reference.low.len());
        prop_assert_eq!(direct.high.len(), reference.high.len());
        for (actual, expected) in direct.low.iter().zip(reference.low.iter()) {
            prop_assert!((actual - expected).abs() <= 1.0e-9);
        }
        for (actual, expected) in direct.high.iter().zip(reference.high.iter()) {
            prop_assert!((actual - expected).abs() <= 1.0e-9);
        }
    }
}

#[test]
fn linear_single_level_mapping_matches_float_reference_for_synthetic_blocks() {
    for coeffs in synthetic_float_coefficients() {
        let direct = dct8_to_dwt53_float_linear(coeffs);
        let reference = idct8_then_dwt53_float(coeffs);

        assert_dwt53_close(direct, reference, 1.0e-10);
    }
}

#[test]
fn linear_mapping_crosses_dct_block_boundary_for_two_blocks() {
    let blocks = [
        [52.0, 11.0, -4.0, 7.0, 0.0, -3.0, 2.0, 1.0],
        [47.0, -9.0, 5.0, -2.0, 8.0, 0.0, -1.0, 3.0],
    ];

    let direct = dct8_blocks_to_dwt53_float_linear(&blocks);
    let reference = idct8_blocks_then_dwt53_float(&blocks);

    assert_eq!(direct.low.len(), 8);
    assert_eq!(direct.high.len(), 8);
    assert_dwt53_row_close(&direct, &reference, 1.0e-10);
}

#[test]
fn linear_mapping_handles_cropped_even_and_odd_row_lengths() {
    let blocks = [
        [52.0, 11.0, -4.0, 7.0, 0.0, -3.0, 2.0, 1.0],
        [47.0, -9.0, 5.0, -2.0, 8.0, 0.0, -1.0, 3.0],
        [21.0, 3.0, -8.0, 1.0, 2.0, -4.0, 6.0, -5.0],
    ];

    for sample_len in [15_usize, 16, 17] {
        let direct =
            dct8_blocks_to_dwt53_float_linear_with_len(&blocks, sample_len).expect("valid row");
        let reference =
            idct8_blocks_then_dwt53_float_with_len(&blocks, sample_len).expect("valid row");

        assert_eq!(direct.low.len(), sample_len.div_ceil(2));
        assert_eq!(direct.high.len(), sample_len / 2);
        assert_dwt53_row_close(&direct, &reference, 1.0e-10);
    }
}

#[test]
fn linear_mapping_matches_reference_for_dc_high_frequency_and_random_rows() {
    for blocks in multi_block_float_coefficients() {
        let direct = dct8_blocks_to_dwt53_float_linear(&blocks);
        let reference = idct8_blocks_then_dwt53_float(&blocks);

        assert_dwt53_row_close(&direct, &reference, 1.0e-10);
    }
}

#[test]
fn reversible_single_level_mapping_matches_rounded_reference_for_synthetic_blocks() {
    for coeffs in synthetic_i16_coefficients() {
        let direct = dct8_to_dwt53_reversible_i16(coeffs);
        let reference = idct8_rounded_then_dwt53_reversible(coeffs);

        assert_eq!(direct, reference);
    }
}

#[test]
fn cropped_row_length_rejects_missing_dct_coverage() {
    let blocks = [[1.0; 8]];
    let err = dct8_blocks_to_dwt53_float_linear_with_len(&blocks, 9).unwrap_err();

    assert_eq!(err.sample_len(), 9);
    assert_eq!(err.capacity(), 8);
}

fn synthetic_float_coefficients() -> Vec<[f64; 8]> {
    vec![
        [0.0; 8],
        [32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 18.0, -7.0, 0.0, 5.0, 0.0, 0.0, 0.0],
        [91.0, -36.0, 14.0, -9.0, 3.0, 22.0, -11.0, 4.0],
        [-40.0, 12.0, 28.0, -17.0, 6.0, -3.0, 2.0, -1.0],
    ]
}

fn synthetic_i16_coefficients() -> Vec<[i16; 8]> {
    vec![
        [0; 8],
        [32, 0, 0, 0, 0, 0, 0, 0],
        [0, 18, -7, 0, 5, 0, 0, 0],
        [91, -36, 14, -9, 3, 22, -11, 4],
        [-40, 12, 28, -17, 6, -3, 2, -1],
    ]
}

fn multi_block_float_coefficients() -> Vec<Vec<[f64; 8]>> {
    vec![
        vec![
            [64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [32.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.0],
            [0.0, 0.0, 0.0, 0.0, -17.0, 0.0, 13.0, 0.0],
            [0.0, -19.0, 0.0, 11.0, 0.0, -7.0, 0.0, 5.0],
        ],
        pseudo_random_blocks(5),
    ]
}

fn pseudo_random_blocks(block_count: usize) -> Vec<[f64; 8]> {
    let mut state = 0x7a37_4c21_u32;
    (0..block_count)
        .map(|_| {
            let mut block = [0.0; 8];
            for coeff in &mut block {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let bounded = u16::try_from(state % 97).expect("modulo result fits u16");
                *coeff = f64::from(i32::from(bounded) - 48);
            }
            block
        })
        .collect()
}

fn assert_dwt53_close(actual: Dwt53OneLevel<f64>, expected: Dwt53OneLevel<f64>, tolerance: f64) {
    for (idx, (actual, expected)) in actual.low.iter().zip(expected.low.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "low[{idx}] differs: actual={actual}, expected={expected}"
        );
    }
    for (idx, (actual, expected)) in actual.high.iter().zip(expected.high.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "high[{idx}] differs: actual={actual}, expected={expected}"
        );
    }
}

fn assert_dwt53_row_close(actual: &Dwt53Row<f64>, expected: &Dwt53Row<f64>, tolerance: f64) {
    assert_eq!(actual.low.len(), expected.low.len());
    assert_eq!(actual.high.len(), expected.high.len());

    for (idx, (actual, expected)) in actual.low.iter().zip(expected.low.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "low[{idx}] differs: actual={actual}, expected={expected}"
        );
    }
    for (idx, (actual, expected)) in actual.high.iter().zip(expected.high.iter()).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "high[{idx}] differs: actual={actual}, expected={expected}"
        );
    }
}
