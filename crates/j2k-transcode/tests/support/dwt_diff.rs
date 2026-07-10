// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_transcode::{Dwt53TwoDimensional, Dwt97TwoDimensional};

#[must_use]
pub(crate) fn max_abs_diff_53(
    actual: &Dwt53TwoDimensional<f64>,
    expected: &Dwt53TwoDimensional<f64>,
) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    max_abs_diff_bands([
        (&actual.ll, &expected.ll),
        (&actual.hl, &expected.hl),
        (&actual.lh, &expected.lh),
        (&actual.hh, &expected.hh),
    ])
}

#[must_use]
pub(crate) fn max_abs_diff_97(
    actual: &Dwt97TwoDimensional<f64>,
    expected: &Dwt97TwoDimensional<f64>,
) -> f64 {
    assert_eq!(actual.low_width, expected.low_width);
    assert_eq!(actual.low_height, expected.low_height);
    assert_eq!(actual.high_width, expected.high_width);
    assert_eq!(actual.high_height, expected.high_height);

    max_abs_diff_bands([
        (&actual.ll, &expected.ll),
        (&actual.hl, &expected.hl),
        (&actual.lh, &expected.lh),
        (&actual.hh, &expected.hh),
    ])
}

fn max_abs_diff_bands(bands: [(&[f64], &[f64]); 4]) -> f64 {
    bands
        .into_iter()
        .flat_map(|(actual, expected)| actual.iter().zip(expected.iter()))
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}
