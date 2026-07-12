// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::repo_root;

mod context;
mod queued;

struct LifecycleSources {
    lifecycle: String,
    lifecycle_tests: String,
    context_inner: String,
    context_operations: String,
    kernel_cache: String,
    kernel_cache_tests: String,
    error: String,
    execution: String,
    completion: String,
    events: String,
    queued: String,
    htj2k_decode: String,
    htj2k_decode_queued: String,
    htj2k_decode_queued_drop: String,
    htj2k_decode_queued_status: String,
    htj2k_decode_queued_status_tests: String,
    memory_pinned_staging: String,
    memory_pinned_staging_operations: String,
    memory_pool: String,
    memory_pool_reuse_guard: String,
}

impl LifecycleSources {
    fn read() -> Self {
        let root = repo_root();
        let read = |relative: &str| {
            fs::read_to_string(root.join(relative))
                .unwrap_or_else(|error| panic!("read {relative}: {error}"))
        };
        Self {
            lifecycle: read("crates/j2k-cuda-runtime/src/context/lifecycle.rs"),
            lifecycle_tests: read("crates/j2k-cuda-runtime/src/context/lifecycle/tests.rs"),
            context_inner: read("crates/j2k-cuda-runtime/src/context/inner.rs"),
            context_operations: read("crates/j2k-cuda-runtime/src/context/operations.rs"),
            kernel_cache: read("crates/j2k-cuda-runtime/src/context/kernel_cache.rs"),
            kernel_cache_tests: read("crates/j2k-cuda-runtime/src/context/kernel_cache/tests.rs"),
            error: read("crates/j2k-cuda-runtime/src/error.rs"),
            execution: read("crates/j2k-cuda-runtime/src/execution.rs"),
            completion: read("crates/j2k-cuda-runtime/src/execution/completion.rs"),
            events: read("crates/j2k-cuda-runtime/src/execution/events.rs"),
            queued: read("crates/j2k-cuda-runtime/src/execution/queued.rs"),
            htj2k_decode: read("crates/j2k-cuda-runtime/src/htj2k_decode/completion.rs"),
            htj2k_decode_queued: read("crates/j2k-cuda-runtime/src/htj2k_decode/queued.rs"),
            htj2k_decode_queued_drop: read(
                "crates/j2k-cuda-runtime/src/htj2k_decode/queued/drop_guard.rs",
            ),
            htj2k_decode_queued_status: read("crates/j2k-cuda-runtime/src/htj2k_decode/status.rs"),
            htj2k_decode_queued_status_tests: read(
                "crates/j2k-cuda-runtime/src/htj2k_decode/status/tests.rs",
            ),
            memory_pinned_staging: [
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging.rs"),
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging/tests.rs"),
            ]
            .concat(),
            memory_pinned_staging_operations: [
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging/operations.rs"),
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/recycle.rs"),
                read("crates/j2k-cuda-runtime/src/memory/pinned_staging/operations/checkout.rs"),
            ]
            .concat(),
            memory_pool: read("crates/j2k-cuda-runtime/src/memory/pool.rs"),
            memory_pool_reuse_guard: read("crates/j2k-cuda-runtime/src/memory/pool/reuse_guard.rs"),
        }
    }
}

fn assert_lifecycle_modules_stay_focused(sources: &LifecycleSources) {
    for (relative, source, max_lines) in [
        ("context/lifecycle.rs", sources.lifecycle.as_str(), 175usize),
        (
            "context/operations.rs",
            sources.context_operations.as_str(),
            100,
        ),
    ] {
        let line_count = source.lines().count();
        assert!(
            line_count < max_lines,
            "crates/j2k-cuda-runtime/src/{relative} has {line_count} lines; split it before reaching {max_lines}"
        );
    }
}

#[test]
fn cuda_resource_failures_recover_or_quarantine_by_operation_class() {
    let sources = LifecycleSources::read();
    assert_lifecycle_modules_stay_focused(&sources);
    context::assert_context_lifecycle_contract(&sources);
    context::assert_completion_and_transition_contract(&sources);
    queued::assert_queued_ownership_contract(&sources);
}
