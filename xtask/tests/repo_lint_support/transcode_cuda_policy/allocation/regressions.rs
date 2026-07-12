// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed allocation diagnostics and allocator-capacity regression coverage.

use super::super::super::{assert_pattern_checks, PatternCheck};
use super::super::CudaTranscodeSources;

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    let production = sources.combined();
    let full = sources.full_combined();
    assert_pattern_checks(&[
        PatternCheck::new("CUDA transcode allocation errors", &production).required(&[
            "HostAllocationTooLarge {",
            "requested: usize",
            "cap: usize",
            "HostAllocationFailed {",
            "what: &'static str",
            "Self::HostAllocationTooLarge",
            "Self::HostAllocationFailed",
        ]),
        PatternCheck::new("CUDA transcode allocation regressions", &full).required(&[
            "host_staging_rejects_overflow_and_over_cap_before_allocation",
            "allocator_reported_capacity_has_exact_and_one_under_boundaries",
            "phase_budget_reconciles_synthetic_allocator_capacity_and_failure",
            "block_grid_validation_rejects_mismatch_and_overflow_without_allocation",
        ]),
        PatternCheck::new(
            "CUDA transcode allocator-capacity reconciliation",
            &production,
        )
        .required(&[
            "HostAllocationBudget::new(cap)",
            ".check_capacity::<T>(element_count)",
            ".account_vec(values)",
            "transcode_capacity_error",
        ]),
    ]);
}
