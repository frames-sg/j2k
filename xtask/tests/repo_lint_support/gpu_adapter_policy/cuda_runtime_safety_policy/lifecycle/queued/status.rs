// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::LifecycleSources;
use super::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_queued_status_contract(sources: &LifecycleSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA queued status and release error selection",
            &sources.htj2k_decode_queued_status,
        )
        .required(&[
            "fn select_status_release_result(",
            "select_resource_release_error(",
        ]),
        PatternCheck::new(
            "CUDA queued compound status regressions",
            &sources.htj2k_decode_queued_status_tests,
        )
        .required(&[
            "kernel_and_release_failures_are_both_preserved",
            "single_failure_and_success_paths_keep_their_original_result",
        ]),
    ]);
}
