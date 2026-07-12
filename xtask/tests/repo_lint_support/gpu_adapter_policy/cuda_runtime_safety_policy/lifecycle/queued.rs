// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::super::{assert_pattern_checks, PatternCheck};
use super::LifecycleSources;

mod status;

pub(super) fn assert_queued_ownership_contract(sources: &LifecycleSources) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA queued execution contract", &sources.queued).required(&[
            "impl Drop for CudaQueuedExecution",
            "#[must_use = \"queued CUDA work must be finished or retained until Drop synchronizes it\"]",
            "pub fn finish(mut self) -> Result<CudaExecutionStats, CudaError>",
            "pub unsafe fn release_pool_reuse_after_completion",
        ]),
        PatternCheck::new("CUDA deferred pool reuse guard", &sources.memory_pool).required(&[
            "reuse_holds: usize",
            "fn defer_reuse(",
            "Pool invariants are unknown after poisoning",
            "nested_pool_reuse_holds_release_only_at_zero",
        ]),
        PatternCheck::new(
            "CUDA deferred pool completion ownership",
            &sources.memory_pool_reuse_guard,
        )
        .required(&[
            "struct CudaBufferPoolReuseGuard",
            "leave the hold active rather than",
            "fn synchronize_then_error<T>(self, error: CudaError)",
            "if let Err(completion_error) = self.synchronize_pool_context().into_result()",
            "match self.release()",
            "fn release_after_recoverable_operation_error<T>(",
            "Do not synchronize a",
        ]),
        PatternCheck::new("CUDA recovered D2H pool ownership", &sources.htj2k_decode).required(&[
            "pool_reuse_guard.release_after_recoverable_operation_error(error)",
        ]),
        PatternCheck::new("CUDA queued D2H error ownership", &sources.htj2k_decode_queued)
            .required(&[
                "fn release_after_recoverable_operation_error(&mut self, primary_error: CudaError)",
                "fn synchronize_release_after_error(&mut self, primary_error: CudaError)",
                "HostPhaseBudget::with_live_bytes(",
                "budget.try_vec_filled(self.status_count, CudaHtj2kStatus::default())",
                "select_uncertain_completion_error(primary_error, Some(completion_error))",
                "return Err(self.release_after_recoverable_operation_error(error));",
            ]),
        PatternCheck::new(
            "CUDA queued D2H drop ownership",
            &sources.htj2k_decode_queued_drop,
        )
        .required(&[
            "impl Drop for CudaQueuedHtj2kCleanup",
            "self.context.synchronize_for_resource_release()",
            "self.abandon_resources()",
        ]),
    ]);
    status::assert_queued_status_contract(sources);

    assert_eq!(
        sources
            .events
            .matches("synchronize_then_error(error)")
            .count(),
        3,
        "completed, event-timed, and explicitly submitted work must synchronize error paths"
    );
}
