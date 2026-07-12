// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, read_runtime, PatternCheck};

mod ordering;

#[test]
fn queued_decode_resources_require_explicit_completion() {
    let htj2k_decode = read_runtime("htj2k_decode/completion.rs");
    let htj2k_context_validation = read_runtime("htj2k_decode/context_validation.rs");
    let htj2k_queued = read_runtime("htj2k_decode/queued.rs");
    let htj2k_queued_drop = read_runtime("htj2k_decode/queued/drop_guard.rs");
    let idwt = read_runtime("j2k_decode/idwt.rs");
    let idwt_sequence = read_runtime("j2k_decode/idwt/sequence.rs");
    let idwt_all = [idwt.as_str(), idwt_sequence.as_str()].concat();
    let idwt_context_validation = read_runtime("j2k_decode/idwt/context_validation.rs");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA queued cleanup entry point", &htj2k_decode).required(&[
            "pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool",
        ]),
        PatternCheck::new(
            "CUDA queued cleanup metadata retention is allocated before launch",
            &htj2k_decode,
        )
        .required(&[
            "let mut queued_resources = host_budget.try_vec_with_capacity(1)?;",
            "queued_resources.push(jobs_buffer);",
            "pooled_device_buffer(&queued_resources[0])?",
            "resources: queued_resources,",
        ])
        .forbidden(&["resources: vec![jobs_buffer]"]),
        PatternCheck::new("CUDA queued cleanup completion guard", &htj2k_queued).required(&[
            "#[must_use = \"queued HTJ2K cleanup must be finished or retained until Drop synchronizes it\"]",
            "mod drop_guard;",
            "pool_reuse_guard: Option<CudaBufferPoolReuseGuard>",
            "fn synchronize_and_release(&mut self)",
            "fn abandon_resources(&mut self)",
            "pub fn finish(mut self) -> Result<CudaExecutionStats, CudaError>",
        ]),
        PatternCheck::new("CUDA queued cleanup drop guard", &htj2k_queued_drop).required(&[
            "impl Drop for CudaQueuedHtj2kCleanup",
            "self.context.synchronize_for_resource_release()",
            "self.abandon_resources()",
        ]),
        PatternCheck::new("CUDA queued IDWT completion guard", &idwt_all).required(&[
            "pub unsafe fn j2k_inverse_dwt_batch_device_enqueue_with_pool",
            "pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool",
            "let pool_reuse_guard = pool.defer_reuse()?;",
            "pool_reuse_guard: Some(pool_reuse_guard)",
            "let sequence_result = (|| -> Result<(), CudaError>",
            "return pool_reuse_guard.synchronize_then_error(error)",
        ]),
        PatternCheck::new(
            "CUDA queued IDWT metadata retention is allocated before launch",
            &idwt_all,
        )
        .required(&[
            "let mut queued_resources = host_budget.try_vec_with_capacity(1)?;",
            "queued_resources.push(jobs_buffer);",
            "resources: queued_resources,",
        ])
        .forbidden(&["resources: vec![jobs_buffer]"]),
        PatternCheck::new(
            "CUDA HTJ2K decode context and target validation",
            &htj2k_context_validation,
        )
        .required(&[
            "fn validate_cleanup_context(",
            "fn validate_dequantize_context(",
            "fn validate_target_allocations_disjoint",
            "CheckedDeviceBufferRanges::from_same_context",
            "first_self_overlap",
            "resources, targets, and pool must belong to the launch context",
            "target allocations must be pairwise disjoint",
        ]),
        PatternCheck::new(
            "CUDA IDWT context and alias validation",
            &idwt_context_validation,
        )
        .required(&[
            "fn validate_idwt_enqueue_context(",
            "fn validate_idwt_sequence_enqueue_context(",
            "CheckedDeviceBufferRanges::from_same_context",
            "first_cross_overlap",
            "first_self_overlap",
            "overlaps a concurrently read input",
            "outputs must be pairwise disjoint",
        ]),
    ]);
    assert_queued_completion_ordering(&htj2k_decode, &idwt, &idwt_sequence, &idwt_all);
}

fn assert_queued_completion_ordering(
    htj2k_decode: &str,
    idwt: &str,
    idwt_sequence: &str,
    idwt_all: &str,
) {
    assert!(!htj2k_decode.contains("let _ = self.synchronize();"));
    assert!(!idwt_all.contains("let _ = self.synchronize();"));
    assert_eq!(
        idwt_all.matches("resources: queued_resources,").count(),
        2,
        "both queued IDWT launchers must retain metadata fallibly"
    );
    ordering::assert_retention_precedes_launch(
        htj2k_decode,
        "pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool",
        "let launch_result =",
    );
    ordering::assert_retention_precedes_launch(
        idwt,
        "pub unsafe fn j2k_inverse_dwt_batch_device_enqueue_with_pool",
        "let interleave_horizontal_result =",
    );
    ordering::assert_retention_precedes_launch(
        idwt_sequence,
        "pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes",
        "let sequence_result =",
    );
}
