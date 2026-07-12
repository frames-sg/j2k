// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error metrics for coefficient-domain validation.

use core::{fmt, mem::size_of};

use j2k_core::{try_host_vec_with_capacity, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

/// One sorted absolute-error histogram bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorHistogramBucket {
    absolute_error: i64,
    count: usize,
}

impl ErrorHistogramBucket {
    /// Absolute coefficient error represented by this bucket.
    #[must_use]
    pub const fn absolute_error(self) -> i64 {
        self.absolute_error
    }

    /// Number of coefficients in this bucket.
    #[must_use]
    pub const fn count(self) -> usize {
        self.count
    }
}

/// Sorted, move-only absolute-error histogram.
///
/// Storage is a single fallibly reserved vector. Construction sorts and
/// coalesces that owner in place, so no second coefficient-sized allocation is
/// needed and its allocator-reported capacity remains inspectable.
#[derive(Debug, PartialEq, Eq)]
pub struct ErrorHistogram {
    buckets: Vec<ErrorHistogramBucket>,
}

impl ErrorHistogram {
    /// Number of distinct absolute-error buckets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buckets.len()
    }

    /// Whether the histogram contains no buckets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Count for one absolute error, or zero when that bucket is absent.
    #[must_use]
    pub fn count(&self, absolute_error: i64) -> usize {
        self.buckets
            .binary_search_by_key(&absolute_error, |bucket| bucket.absolute_error)
            .ok()
            .map_or(0, |index| self.buckets[index].count)
    }

    /// Iterate over sorted histogram buckets.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = ErrorHistogramBucket> + '_ {
        self.buckets.iter().copied()
    }

    /// Allocator-reported bytes retained by the histogram backing vector.
    pub fn retained_bytes(&self) -> Result<usize, MetricsError> {
        checked_histogram_bytes(self.buckets.capacity(), DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }
}

impl IntoIterator for ErrorHistogram {
    type Item = ErrorHistogramBucket;
    type IntoIter = std::vec::IntoIter<ErrorHistogramBucket>;

    fn into_iter(self) -> Self::IntoIter {
        self.buckets.into_iter()
    }
}

/// Difference summary between two integer coefficient vectors.
#[derive(Debug, PartialEq, Eq)]
pub struct ErrorMetrics {
    /// Number of compared coefficients.
    pub total: usize,
    /// Number of coefficients with exact equality.
    pub exact_matches: usize,
    /// Maximum absolute coefficient error.
    pub max_abs_error: i64,
    /// Absolute-error histogram keyed by LSB distance.
    pub absolute_error_histogram: ErrorHistogram,
}

impl ErrorMetrics {
    /// Fraction of coefficients that match exactly.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "validation rates are intentionally reported as approximate f64 ratios"
    )]
    pub fn exact_match_rate(&self) -> f64 {
        if self.total == 0 {
            return 1.0;
        }

        self.exact_matches as f64 / self.total as f64
    }

    /// Number of coefficients at the given absolute error.
    #[must_use]
    pub fn absolute_error_count(&self, absolute_error: i64) -> usize {
        self.absolute_error_histogram.count(absolute_error)
    }

    /// Whether the metrics satisfy a one-LSB-bounded claim at the requested
    /// exact-match threshold.
    #[must_use]
    pub fn is_one_lsb_bounded(&self, exact_match_threshold: f64) -> bool {
        self.max_abs_error <= 1 && self.exact_match_rate() >= exact_match_threshold
    }
}

/// Typed validation-metrics construction failure.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MetricsError {
    /// Actual and expected coefficients describe different sample counts.
    LengthMismatch {
        /// Actual coefficient count.
        actual: usize,
        /// Expected coefficient count.
        expected: usize,
    },
    /// Input owners plus histogram storage exceed the shared host cap.
    MemoryCapExceeded {
        /// Required live bytes, saturated on arithmetic overflow.
        requested: usize,
        /// Maximum accepted live bytes.
        cap: usize,
    },
    /// Histogram storage could not be reserved.
    HostAllocationFailed {
        /// Requested histogram allocation bytes.
        bytes: usize,
    },
}

impl fmt::Display for MetricsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { actual, expected } => {
                write!(
                    f,
                    "metric input lengths differ: actual {actual}, expected {expected}"
                )
            }
            Self::MemoryCapExceeded { requested, cap } => write!(
                f,
                "metrics host workspace requires {requested} bytes, exceeding the {cap}-byte cap"
            ),
            Self::HostAllocationFailed { bytes } => {
                write!(f, "metrics host allocation failed for {bytes} bytes")
            }
        }
    }
}

impl std::error::Error for MetricsError {}

/// Compute exact-match rate, max absolute error, and absolute-LSB histogram for
/// two integer coefficient vectors.
pub fn error_metrics_i32(actual: &[i32], expected: &[i32]) -> Result<ErrorMetrics, MetricsError> {
    error_metrics_i32_with_live_budget(actual, expected, 0, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
}

pub(crate) fn error_metrics_i32_with_live_budget(
    actual: &[i32],
    expected: &[i32],
    external_live_bytes: usize,
    cap: usize,
) -> Result<ErrorMetrics, MetricsError> {
    if actual.len() != expected.len() {
        return Err(MetricsError::LengthMismatch {
            actual: actual.len(),
            expected: expected.len(),
        });
    }

    checked_histogram_live_capacity(external_live_bytes, actual.len(), cap)?;
    let mut buckets = try_host_vec_with_capacity(actual.len()).map_err(|error| {
        MetricsError::HostAllocationFailed {
            bytes: error.requested_bytes(),
        }
    })?;
    checked_histogram_live_capacity(external_live_bytes, buckets.capacity(), cap)?;

    let mut exact_matches = 0;
    let mut max_abs_error = 0;

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        let abs_error = (i64::from(actual) - i64::from(expected)).abs();
        if abs_error == 0 {
            exact_matches += 1;
        }
        max_abs_error = max_abs_error.max(abs_error);
        buckets.push(ErrorHistogramBucket {
            absolute_error: abs_error,
            count: 1,
        });
    }

    buckets.sort_unstable_by_key(|bucket| bucket.absolute_error);
    let mut output_len = 0usize;
    for input_index in 0..buckets.len() {
        let bucket = buckets[input_index];
        if output_len > 0 && buckets[output_len - 1].absolute_error == bucket.absolute_error {
            buckets[output_len - 1].count = buckets[output_len - 1]
                .count
                .checked_add(bucket.count)
                .ok_or_else(cap_overflow)?;
        } else {
            buckets[output_len] = bucket;
            output_len += 1;
        }
    }
    buckets.truncate(output_len);

    Ok(ErrorMetrics {
        total: actual.len(),
        exact_matches,
        max_abs_error,
        absolute_error_histogram: ErrorHistogram { buckets },
    })
}

fn checked_histogram_bytes(capacity: usize, cap: usize) -> Result<usize, MetricsError> {
    capacity
        .checked_mul(size_of::<ErrorHistogramBucket>())
        .ok_or(MetricsError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })
}

fn checked_histogram_live_capacity(
    external_live_bytes: usize,
    histogram_capacity: usize,
    cap: usize,
) -> Result<usize, MetricsError> {
    let histogram_bytes = checked_histogram_bytes(histogram_capacity, cap)?;
    checked_metrics_live_bytes(external_live_bytes, histogram_bytes, cap)
}

fn checked_metrics_live_bytes(
    external_live_bytes: usize,
    histogram_bytes: usize,
    cap: usize,
) -> Result<usize, MetricsError> {
    let requested = external_live_bytes.checked_add(histogram_bytes).ok_or(
        MetricsError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        },
    )?;
    if requested > cap {
        return Err(MetricsError::MemoryCapExceeded { requested, cap });
    }
    Ok(requested)
}

fn cap_overflow() -> MetricsError {
    MetricsError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use core::mem::size_of;

    use super::{
        checked_histogram_live_capacity, error_metrics_i32, ErrorHistogramBucket, MetricsError,
    };

    #[test]
    fn histogram_is_sorted_and_coalesced_in_place() -> Result<(), MetricsError> {
        let metrics = error_metrics_i32(&[10, 3, -4, 9, 8], &[10, 1, -3, 8, 10])?;
        let buckets = metrics
            .absolute_error_histogram
            .iter()
            .map(|bucket| (bucket.absolute_error(), bucket.count()))
            .collect::<Vec<_>>();

        assert_eq!(buckets, [(0, 1), (1, 2), (2, 2)]);
        assert_eq!(metrics.absolute_error_count(0), 1);
        assert_eq!(metrics.absolute_error_count(7), 0);
        assert_eq!(metrics.max_abs_error, 2);
        Ok(())
    }

    #[test]
    fn all_unique_errors_keep_one_sorted_bucket_per_coefficient() -> Result<(), MetricsError> {
        let metrics = error_metrics_i32(&[0, 1, 2, 3], &[0, 0, 0, 0])?;
        let buckets = metrics
            .absolute_error_histogram
            .into_iter()
            .map(|bucket| (bucket.absolute_error(), bucket.count()))
            .collect::<Vec<_>>();

        assert_eq!(buckets, [(0, 1), (1, 1), (2, 1), (3, 1)]);
        Ok(())
    }

    #[test]
    fn length_mismatch_remains_typed() {
        assert_eq!(
            error_metrics_i32(&[1, 2], &[1]),
            Err(MetricsError::LengthMismatch {
                actual: 2,
                expected: 1,
            })
        );
    }

    #[test]
    fn histogram_live_budget_accepts_exact_cap_and_rejects_one_over() {
        let bucket_bytes = size_of::<ErrorHistogramBucket>();
        let cap = bucket_bytes * 3;

        assert_eq!(
            checked_histogram_live_capacity(bucket_bytes, 2, cap),
            Ok(cap)
        );
        assert_eq!(
            checked_histogram_live_capacity(bucket_bytes + 1, 2, cap),
            Err(MetricsError::MemoryCapExceeded {
                requested: cap + 1,
                cap,
            })
        );
    }

    #[test]
    fn histogram_live_budget_uses_allocator_capacity_not_logical_length() {
        let bucket_bytes = size_of::<ErrorHistogramBucket>();
        let planned_len = 2;
        let allocator_capacity = 3;
        let cap = bucket_bytes * planned_len;

        assert_eq!(
            checked_histogram_live_capacity(0, planned_len, cap),
            Ok(cap)
        );
        assert_eq!(
            checked_histogram_live_capacity(0, allocator_capacity, cap),
            Err(MetricsError::MemoryCapExceeded {
                requested: bucket_bytes * allocator_capacity,
                cap,
            })
        );
    }
}
