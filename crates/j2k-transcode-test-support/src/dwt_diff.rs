// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::{Dwt53TwoDimensional, Dwt97TwoDimensional};

#[must_use]
pub fn max_abs_diff_53(
    actual: &Dwt53TwoDimensional<f64>,
    expected: &Dwt53TwoDimensional<f64>,
) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    max_abs_diff_bands([
        (
            &actual.ll,
            &expected.ll,
            expected.low_width * expected.low_height,
        ),
        (
            &actual.hl,
            &expected.hl,
            expected.high_width * expected.low_height,
        ),
        (
            &actual.lh,
            &expected.lh,
            expected.low_width * expected.high_height,
        ),
        (
            &actual.hh,
            &expected.hh,
            expected.high_width * expected.high_height,
        ),
    ])
}

#[must_use]
pub fn max_abs_diff_97(
    actual: &Dwt97TwoDimensional<f64>,
    expected: &Dwt97TwoDimensional<f64>,
) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    max_abs_diff_bands([
        (
            &actual.ll,
            &expected.ll,
            expected.low_width * expected.low_height,
        ),
        (
            &actual.hl,
            &expected.hl,
            expected.high_width * expected.low_height,
        ),
        (
            &actual.lh,
            &expected.lh,
            expected.low_width * expected.high_height,
        ),
        (
            &actual.hh,
            &expected.hh,
            expected.high_width * expected.high_height,
        ),
    ])
}

fn max_abs_diff_bands(bands: [(&[f64], &[f64], usize); 4]) -> f64 {
    let mut max_diff = 0.0_f64;
    for (actual, expected, sample_count) in bands {
        assert_eq!(actual.len(), sample_count, "actual DWT band length");
        assert_eq!(expected.len(), sample_count, "reference DWT band length");
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!(actual.is_finite(), "nonfinite actual DWT coefficient");
            assert!(expected.is_finite(), "nonfinite reference DWT coefficient");
            max_diff = max_diff.max((actual - expected).abs());
        }
    }
    max_diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_rejects_missing_and_nonfinite_coefficients() {
        let fixture = || Dwt97TwoDimensional {
            ll: vec![1.0],
            hl: vec![2.0],
            lh: vec![3.0],
            hh: vec![4.0],
            low_width: 1,
            low_height: 1,
            high_width: 1,
            high_height: 1,
        };
        for invalid in [vec![], vec![f64::NAN], vec![f64::INFINITY], vec![4.0, 4.0]] {
            let mut actual = fixture();
            actual.hh = invalid;
            let expected = fixture();
            assert!(
                std::panic::catch_unwind(|| max_abs_diff_97(&actual, &expected)).is_err(),
                "invalid actual coefficients passed parity: {:?}",
                actual.hh
            );
            assert!(
                std::panic::catch_unwind(|| max_abs_diff_97(&expected, &actual)).is_err(),
                "invalid reference coefficients passed parity: {:?}",
                actual.hh
            );
        }
    }
}
