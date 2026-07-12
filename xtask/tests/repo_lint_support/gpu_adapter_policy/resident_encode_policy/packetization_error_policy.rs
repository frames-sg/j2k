// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn cuda_packetization_failures_keep_explicit_typed_routing() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let plan_sources = [
        read("crates/j2k-cuda/src/encode/packetization/error.rs"),
        read("crates/j2k-cuda/src/encode/packetization/flatten.rs"),
        read("crates/j2k-cuda/src/encode/packetization/state.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tag_tree.rs"),
        read("crates/j2k-cuda/src/encode/packetization/types.rs"),
    ]
    .concat();
    let stage = read("crates/j2k-cuda/src/encode/stage.rs");
    let behavior_tests = [
        read("crates/j2k-cuda/src/encode/packetization/tests.rs"),
        read("crates/j2k-cuda/src/encode/packetization/tests/ht_segment.rs"),
        read("crates/j2k-cuda/src/encode/tests/routing.rs"),
    ]
    .concat();

    assert_pattern_checks(&[
        PatternCheck::new("typed CUDA packetization plan failures", &plan_sources)
            .required(&[
                "enum CudaHtj2kPacketizationPlanError",
                "Invalid(&'static str)",
                "HostAllocation",
                "return Err(CudaHtj2kPacketizationPlanError::Invalid(",
                ".ok_or(CudaHtj2kPacketizationPlanError::Invalid(",
                ".map_err(cuda_ht_segment_length_error)",
                "packetization_plan_allocation_error",
                "fn cuda_ht_segment_length_error(",
                "ContributionLengthExceedsU32",
                "MultiPassLengthOverflow",
                "EmptyContributionHasSegments",
                "RefinementOnlyLengthMismatch",
                "RefinementLengthOutOfRange",
                "SinglePassHasRefinement",
                "SinglePassLengthMismatch",
                "MultiPassRequiresSegments",
                "MultiPassLengthMismatch",
                "CleanupLengthOutOfRange",
            ])
            .forbidden(&[
                "impl From<&'static str> for CudaHtj2kPacketizationPlanError",
                "return Err(\"CUDA HTJ2K packetization",
                "return Err(\"CUDA packetization",
                ".ok_or(\"CUDA HTJ2K packetization",
                ".map_err(|_| \"CUDA HTJ2K packetization",
                ".into())",
            ]),
        PatternCheck::new("CUDA packetization stage category routing", &stage).required(&[
            "fn cuda_packetization_plan_fallback_reason(",
            "CudaHtj2kPacketizationPlanError::Invalid(reason) => Ok(reason)",
            "CudaHtj2kPacketizationPlanError::ArithmeticOverflow(what)",
            "CudaHtj2kPacketizationPlanError::MemoryCapExceeded",
            "CudaHtj2kPacketizationPlanError::HostAllocation { what, bytes }",
            "CudaHtj2kPacketizationPlanError::Adapter(source)",
            "cuda_packetization_plan_fallback_reason(error)?",
        ]),
        PatternCheck::new("CUDA packetization error behavior tests", &behavior_tests).required(&[
            "packetization_plan_allocation_failure_keeps_its_typed_category",
            "ht_segment_validation_errors_keep_invalid_and_overflow_categories",
            "cuda_invalid_packetization_plan_falls_back_after_classification",
            "cuda_packetization_host_allocation_is_a_hard_stage_error",
        ]),
    ]);
}
