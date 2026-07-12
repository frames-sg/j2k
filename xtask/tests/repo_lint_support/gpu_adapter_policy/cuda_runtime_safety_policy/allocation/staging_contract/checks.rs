// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::super::super::{assert_pattern_checks, PatternCheck};
use super::sources::StagingSources;

pub(super) fn assert_pinned_operation_contracts(sources: &StagingSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA pinned staging retention", &sources.pinned).required(&[
            "pool.try_retain_after_uncertain_release(staging)",
            "select_resource_release_error(",
            "PinnedUploadStaging` intentionally",
        ]),
        PatternCheck::new(
            "bounded CUDA pinned staging operations",
            &sources.pinned_operations,
        )
        .required(&[
            "validate_pinned_upload_staging_len(len, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;",
            "cuMemHostAlloc",
            "pool.prepare_unwind_quarantine_slots()?",
            "pool.take_best_fit(len)?",
            "cached_plus_request_fits_host_cap",
            "PinnedUploadStagingAdmission::Reject",
            "PinnedUploadStagingAdmission::Evict",
            "PinnedUploadStagingAdmission::Admit",
            "release_inactive_pinned_upload_staging(",
            "begin_pinned_upload_operation()?.upload(bytes)",
            "pub fn prepare_upload(",
            "PinnedUploadStaging::from_raw",
        ]),
        PatternCheck::new(
            "CUDA pinned staging recycle transitions",
            &sources.pinned_recycle,
        )
        .required(&[
            "release_active_pinned_upload_staging(",
            "release_inactive_pinned_upload_staging(",
            "finish_active_checkout(",
            "retain_pinned_upload_staging_after_active_release_failure(",
            "retain_pinned_upload_staging_after_release_failure(",
            "select_resource_release_error(",
        ]),
        PatternCheck::new("CUDA pinned staging transaction gate", &sources.pinned_gate).required(
            &[
                "pub struct CudaPinnedUploadOperationGuard",
                "MutexGuard<'a, ()>",
                "PhantomData<Cell<()>>",
                "transaction guard",
            ],
        ),
        PatternCheck::new(
            "CUDA pinned staging checkout ownership",
            &sources.pinned_checkout,
        )
        .required(&[
            "pub struct CudaPinnedUploadStagingCheckout",
            "pub fn allocation_byte_len(",
            "pub fn retained_page_locked_bytes(",
            "pub fn upload(",
            "pub fn recycle(",
            "impl Drop for CudaPinnedUploadStagingCheckout",
            "retain_pinned_upload_staging_after_abandoned_checkout(",
            "select_pinned_upload_result(upload_result, recycle_result)",
        ]),
        PatternCheck::new(
            "CUDA pinned staging public diagnostics",
            &sources.pinned_api,
        )
        .required(&[
            "pinned_upload_staging_pool_diagnostics(",
            "observability rather than a cross-owner admission transaction",
        ]),
        PatternCheck::new(
            "CUDA pinned staging operation policy",
            &sources.pinned_policy,
        )
        .required(&[
            "validate_pinned_upload_staging_len(",
            "HostPhaseBudget::with_cap",
            "lock_pinned_upload_operation(",
            "validate_pinned_upload_operation_context(",
        ]),
        PatternCheck::new(
            "CUDA pooled pinned staging transaction",
            &sources.device_pool,
        )
        .required(&[
            "begin_pinned_upload_operation()?",
            "operation.prepare_upload(bytes.len())?",
            "staging.recycle()",
            "select_pinned_upload_result",
        ]),
    ]);
}

pub(super) fn assert_pinned_pool_contracts(sources: &StagingSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "bounded CUDA pinned staging ledger",
            &sources.pinned_pool_state,
        )
        .required(&[
            "uncertain: Vec<PinnedUploadStaging>",
            "checked_add",
            "checked_sub",
            "min_by_key",
            "evict_largest_oldest(",
            "buffers.try_reserve(1)",
            "uncertain.try_reserve(1)",
            "prepare_unwind_quarantine_slots",
            "accounting_poisoned",
            "active_buffers",
            "active_bytes",
            "try_admit_active",
            "ensure_no_uncertain_release",
        ])
        .forbidden(&["swap_remove("]),
        PatternCheck::new(
            "CUDA pinned staging diagnostics ledger",
            &sources.pinned_pool_diagnostics,
        )
        .required(&[
            "const DEFAULT_MAX_CACHED_BYTES",
            "const DEFAULT_MAX_CACHED_BUFFERS",
            "pub struct CudaPinnedUploadStagingPoolLimits",
            "pub struct CudaPinnedUploadStagingPoolDiagnostics",
            "pub retained_bytes: usize",
            "pub peak_retained_bytes: usize",
            "pub active_bytes: usize",
            "pub peak_active_bytes: usize",
            "two fallible",
        ]),
        PatternCheck::new(
            "CUDA pinned staging boundary regressions",
            &sources.pinned_pool_tests,
        )
        .required(&[
            "exact_byte_and_count_limits_admit_and_one_over_evicts",
            "retained_cache_plus_current_request_honors_exact_host_cap",
            "oversized_candidate_is_rejected_without_displacing_reusable_staging",
            "uncertain_release_quarantine_is_separate_from_bounded_reuse",
            "best_fit_take_and_largest_oldest_eviction_are_deterministic",
            "long_distinct_size_churn_has_stable_bounded_retention",
            "byte_accounting_overflow_is_typed",
            "arc_aliases_observe_one_diagnostics_ledger",
        ]),
        PatternCheck::new(
            "CUDA active staging diagnostics regressions",
            &sources.pinned_pool_active_tests,
        )
        .required(&[
            "best_fit_actual_length_drives_transaction_exact_and_one_over",
            "active_checkout_is_visible_in_current_and_peak_diagnostics",
            "confirmed_new_checkout_updates_actual_retained_high_water",
            "diagnostics.active_bytes",
            "diagnostics.peak_retained_bytes",
        ]),
        PatternCheck::new(
            "CUDA pinned staging allocation cap regression",
            &sources.pinned_policy_tests,
        )
        .required(&[
            "pinned_staging_allocation_accepts_exact_cap_and_rejects_one_over",
            "explicit_empty_staging_checkout_is_rejected_without_driver_work",
            "foreign_pinned_upload_operation_is_rejected",
            "clone_shared_operation_gate_serializes_checkout_through_recycle_window",
            "poisoned_operation_gate_surfaces_typed_error",
            "CudaError::HostAllocationTooLarge",
        ]),
        PatternCheck::new(
            "CUDA pinned staging compound failure regression",
            &sources.pinned_checkout_tests,
        )
        .required(&[
            "upload_and_recycle_failures_preserve_primary_and_release_sources",
            "CudaError::ResourceReleaseFailed",
        ]),
    ]);
}

pub(super) fn assert_other_allocation_contracts(sources: &StagingSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "CUDA pinned token consumes successful free",
            &sources.pinned_token,
        )
        .required(&[
            "validate_non_null_pinned_host_allocation(",
            "NonNull::new(ptr)",
            "pub(crate) fn from_raw(",
            "pub(crate) fn free(&mut self",
            "self.ptr = std::ptr::null_mut();",
            "self.len = 0;",
            "nonzero_null_pinned_allocation_is_rejected_before_safe_slice_construction",
        ]),
        PatternCheck::new(
            "CUDA kernel cache reserves before module creation",
            &sources.kernel_cache,
        )
        .required(&[
            "modules.try_reserve(1)",
            "let compiled = CompiledKernel::load(self, key)?;",
            "modules.insert(key, compiled);",
        ]),
        PatternCheck::new(
            "exact initialized spare-capacity readback",
            &sources.readback,
        )
        .required(&[
            "&mut out.spare_capacity_mut()[..byte_len]",
            "out.set_len(byte_len);",
        ]),
        PatternCheck::new("exact CUDA batch spare-capacity readback", &sources.surface).required(
            &[
                "&mut out.spare_capacity_mut()[..required]",
                "out.set_len(required);",
            ],
        ),
    ]);
}
