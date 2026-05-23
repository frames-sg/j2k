// SPDX-License-Identifier: Apache-2.0

//! Error metrics for coefficient-domain validation.

use core::fmt;
use std::collections::BTreeMap;

/// Difference summary between two integer coefficient vectors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorMetrics {
    /// Number of compared coefficients.
    pub total: usize,
    /// Number of coefficients with exact equality.
    pub exact_matches: usize,
    /// Maximum absolute coefficient error.
    pub max_abs_error: i64,
    /// Absolute-error histogram keyed by LSB distance.
    pub absolute_error_histogram: BTreeMap<i64, usize>,
}

impl ErrorMetrics {
    /// Fraction of coefficients that match exactly.
    #[must_use]
    pub fn exact_match_rate(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }

        self.exact_matches as f64 / self.total as f64
    }

    /// Number of coefficients at the given absolute error.
    #[must_use]
    pub fn absolute_error_count(&self, absolute_error: i64) -> usize {
        self.absolute_error_histogram
            .get(&absolute_error)
            .copied()
            .unwrap_or(0)
    }

    /// Whether the metrics satisfy a one-LSB-bounded claim at the requested
    /// exact-match threshold.
    #[must_use]
    pub fn is_one_lsb_bounded(&self, exact_match_threshold: f64) -> bool {
        self.max_abs_error <= 1 && self.exact_match_rate() >= exact_match_threshold
    }
}

/// Error returned when metric inputs do not describe the same coefficient set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricsLengthError {
    actual_len: usize,
    expected_len: usize,
}

impl MetricsLengthError {
    /// Length of the actual coefficient slice.
    #[must_use]
    pub const fn actual_len(self) -> usize {
        self.actual_len
    }

    /// Length of the expected coefficient slice.
    #[must_use]
    pub const fn expected_len(self) -> usize {
        self.expected_len
    }
}

impl fmt::Display for MetricsLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "metric input lengths differ: actual {}, expected {}",
            self.actual_len, self.expected_len
        )
    }
}

impl std::error::Error for MetricsLengthError {}

/// Compute exact-match rate, max absolute error, and absolute-LSB histogram for
/// two integer coefficient vectors.
pub fn error_metrics_i32(
    actual: &[i32],
    expected: &[i32],
) -> Result<ErrorMetrics, MetricsLengthError> {
    if actual.len() != expected.len() {
        return Err(MetricsLengthError {
            actual_len: actual.len(),
            expected_len: expected.len(),
        });
    }

    let mut exact_matches = 0;
    let mut max_abs_error = 0;
    let mut absolute_error_histogram = BTreeMap::new();

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        let abs_error = (i64::from(actual) - i64::from(expected)).abs();
        if abs_error == 0 {
            exact_matches += 1;
        }
        max_abs_error = max_abs_error.max(abs_error);
        *absolute_error_histogram.entry(abs_error).or_insert(0) += 1;
    }

    Ok(ErrorMetrics {
        total: actual.len(),
        exact_matches,
        max_abs_error,
        absolute_error_histogram,
    })
}
