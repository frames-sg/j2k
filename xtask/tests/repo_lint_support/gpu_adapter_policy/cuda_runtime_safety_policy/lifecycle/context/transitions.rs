// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::LifecycleSources;
use super::{assert_pattern_checks, PatternCheck};

pub(in super::super) fn assert_completion_and_transition_contract(sources: &LifecycleSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA compound completion diagnostic", &sources.error).required(&[
            "CompletionFailed {",
            "primary: Box<CudaError>",
            "completion: Box<CudaError>",
            "ResourceReleaseFailed {",
            "release: Box<CudaError>",
            "fn select_uncertain_completion_error(",
            "fn select_resource_release_error(",
        ]),
        PatternCheck::new("CUDA completion outcome contract", &sources.completion).required(&[
            "enum CudaSynchronizationOutcome",
            "Completed",
            "CompletionUncertain(CudaError)",
            "synchronization_failure_preserves_both_operation_diagnostics",
        ]),
        PatternCheck::new("CUDA context completion integration", &sources.execution).required(&[
            "fn synchronize_for_resource_release",
            "with_current_completion_operation",
            "select_uncertain_completion_error(error, None)",
        ]),
        PatternCheck::new(
            "CUDA resource transition quarantine",
            &sources.memory_pinned_staging_operations,
        )
        .required(&[
            "with_current_stateful_operation",
            "retain_pinned_upload_staging_after_lock_poison(",
            "retain_pinned_upload_staging_after_release_failure(",
            "primary/release diagnostic",
            "uncertain",
        ]),
        PatternCheck::new(
            "CUDA fallible pinned staging retention",
            &sources.memory_pinned_staging,
        )
        .required(&[
            "fn retain_pinned_upload_staging_after_lock_poison(",
            "fn retain_pinned_upload_staging_after_release_failure(",
            "fn retain_pinned_upload_staging_after_abandoned_checkout(",
            "pool.try_retain_after_uncertain_release(staging)",
            "poisoned_pool_retains_returned_raw_allocation_wrapper",
            "failed_release_retains_wrapper_in_clean_or_poisoned_pool",
            "abandoned_or_unwound_checkout_uses_prepared_quarantine_and_fails_closed",
        ]),
        PatternCheck::new(
            "CUDA module transition classification",
            &sources.kernel_cache,
        )
        .required(&[
            "mod tests;",
            "with_current_stateful_operation",
            "with_current_resource_operation",
            "select_resource_release_error(error, unload_error)",
        ]),
        PatternCheck::new(
            "CUDA module transition regressions",
            &sources.kernel_cache_tests,
        )
        .required(&["failed_function_lookup_retains_rollback_failure_diagnostic"]),
        PatternCheck::new("CUDA event ownership transitions", &sources.events).required(&[
            "with_current_stateful_operation",
            "already either establish context-wide",
            "mut work: impl FnMut()",
            "fn complete_default_stream_work<T>(",
            "Disabling event collection must not weaken",
            "self.synchronize()?;",
        ]),
    ]);
}
