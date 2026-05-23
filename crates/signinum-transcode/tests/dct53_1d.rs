// SPDX-License-Identifier: Apache-2.0

use signinum_transcode::dct53_1d::{
    dct8_to_dwt53_float_linear, dct8_to_dwt53_reversible_i16, idct8_rounded_then_dwt53_reversible,
    idct8_then_dwt53_float, Dwt53OneLevel,
};

#[test]
fn linear_single_level_mapping_matches_float_reference_for_synthetic_blocks() {
    for coeffs in synthetic_float_coefficients() {
        let direct = dct8_to_dwt53_float_linear(coeffs);
        let reference = idct8_then_dwt53_float(coeffs);

        assert_dwt53_close(direct, reference, 1.0e-10);
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
