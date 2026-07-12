// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn validation_metrics_remain_fallible_typed_and_actual_capacity_bounded() {
    let path = repo_root().join("crates/j2k-transcode/src/metrics.rs");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));

    assert_pattern_checks(&[PatternCheck::new("transcode validation metrics", &source)
        .required(&[
            "pub struct ErrorHistogramBucket",
            "pub struct ErrorHistogram",
            "pub enum MetricsError",
            "try_host_vec_with_capacity(actual.len())",
            "checked_histogram_live_capacity(external_live_bytes, buckets.capacity(), cap)",
            "sort_unstable_by_key",
            "buckets.truncate(output_len)",
            "histogram_live_budget_accepts_exact_cap_and_rejects_one_over",
            "histogram_live_budget_uses_allocator_capacity_not_logical_length",
            "all_unique_errors_keep_one_sorted_bucket_per_coefficient",
        ])
        .forbidden(&[
            "BTreeMap",
            "MetricsLengthError",
            "#[derive(Clone, Debug, PartialEq, Eq)]\npub struct ErrorMetrics",
            "#[derive(Clone, Debug, PartialEq, Eq)]\npub struct ErrorHistogram",
        ])]);
}

#[test]
fn transcode_error_retains_the_typed_metrics_source() {
    let path = repo_root().join("crates/j2k-transcode/src/jpeg_to_htj2k/error.rs");
    let source = fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));

    assert_pattern_checks(&[
        PatternCheck::new("transcode metrics error crossing", &source)
            .required(&[
                "Metrics(MetricsError)",
                "Self::Metrics(err) => Some(err)",
                "metrics_failure_retains_its_typed_error_source",
            ])
            .forbidden(&["Metrics(String)", "Metrics(reason.to_string())"]),
    ]);
}

#[test]
fn validation_metrics_policy_stays_focused() {
    assert!(
        include_str!("metrics_policy.rs").lines().count() < 60,
        "validation metrics policy must stay focused"
    );
}
