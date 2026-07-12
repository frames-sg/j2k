// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::super::{assert_pattern_checks, PatternCheck};
use super::LifecycleSources;

mod transitions;

pub(super) use transitions::assert_completion_and_transition_contract;

pub(super) fn assert_context_lifecycle_contract(sources: &LifecycleSources) {
    assert_pattern_checks(&[
        PatternCheck::new(
            "serialized CUDA context resource lifecycle",
            &sources.lifecycle,
        )
        .required(&[
            "struct ContextResourceLifecycle",
            "gate: Mutex<()>",
            "poisoned: AtomicBool",
            "fn run_recoverable<T>(",
            "fn run_completion<T>(",
            "fn run_stateful<T>(",
            "fn run_gated<T>(",
            "lifecycle.mark_poisoned();",
        ]),
        PatternCheck::new("CUDA lifecycle behavior", &sources.lifecycle_tests).required(&[
            "recovered_operation_failure_keeps_context_available",
            "failed_operation_recovery_poisons_and_blocks_later_work",
            "direct_completion_failure_poisons_context",
            "stateful_operation_failure_quarantines_context_after_successful_completion",
            "stateful_operation_and_completion_failures_are_both_retained",
            "successfully_rolled_back_lookup_failure_keeps_context_available",
            "stateful_failure_publishes_quarantine_before_blocking_completion",
            "failed_context_binding_poisons_without_running_operation",
            "concurrent_operations_are_serialized",
            "panic_while_gated_poisons_later_operations",
            "panic_during_recovery_poisons_before_releasing_gate",
        ]),
        PatternCheck::new("CUDA lifecycle owner", &sources.context_inner)
            .required(&["resource_lifecycle: ContextResourceLifecycle"])
            .forbidden(&["fn with_current_resource_operation<T>("]),
        PatternCheck::new(
            "CUDA context-bound operation classes",
            &sources.context_operations,
        )
        .required(&[
            "fn with_current_resource_operation<T>(",
            "fn with_current_completion_operation<T>(",
            "fn with_current_stateful_operation<T>(",
            ".run_recoverable(",
            ".run_completion(",
            ".run_stateful(",
            "fn synchronize_current_after_operation_error",
            "fn resource_lifetimes_poisoned(&self) -> bool",
        ]),
    ]);
}
