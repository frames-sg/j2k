// SPDX-License-Identifier: MIT OR Apache-2.0

//! Error-source preservation at the resident Metal batch preparation seam.

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn resident_metal_batch_preparation_propagates_typed_errors_directly() {
    let source =
        fs::read_to_string(repo_root().join("crates/j2k-metal/src/encode/resident_submit.rs"))
            .expect("read resident Metal submit owner");

    assert_pattern_checks(&[
        PatternCheck::new("resident Metal batch preparation", &source)
            .required(&["prepare_planned_resident_lossless_tiles_batch(chunk_planned, session)?"])
            .forbidden(&[
                "resident {family_name} batch encode failed",
                "prepare_planned_resident_lossless_tiles_batch(chunk_planned, session).map_err",
            ]),
    ]);
}

#[test]
fn resident_cuda_encode_stage_errors_preserve_sources_and_resource_categories() {
    let source = fs::read_to_string(repo_root().join("crates/j2k-cuda/src/encode/stage_error.rs"))
        .expect("read resident CUDA stage-error mapping");

    assert_pattern_checks(&[
        PatternCheck::new("resident CUDA stage-error mapping", &source)
            .required(&[
                "J2kEncodeStageError::host_allocation_failed(what, bytes)",
                "J2kEncodeStageError::host_allocation_failed(operation, bytes)",
                "J2kEncodeStageError::memory_cap_exceeded(what, requested, cap)",
                "J2kEncodeStageError::backend(\"cuda\", operation, source)",
                "adapter_backend_failure_keeps_concrete_source",
                "adapter_cap_failure_preserves_phase_budget_details",
                "runtime_backend_failure_keeps_concrete_source",
                "runtime_cap_failure_keeps_budget_details",
                "downcast_ref::<crate::Error>()",
                "downcast_ref::<j2k_cuda_runtime::CudaError>()",
            ])
            .forbidden(&[
                "source.to_string()",
                "format!(\"{source}",
                "Result<T, &'static str>",
            ]),
    ]);
}
