// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::repo_lint_support::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_policy(
    allocation: &str,
    htj2k: &str,
    packetization: &str,
    stage: &str,
    dwt_output: &str,
) {
    assert_pattern_checks(&[
        PatternCheck::new("CUDA adapter incremental allocation helpers", allocation).required(&[
            "pub(crate) struct HostPhaseBudget",
            "HostAllocationBudget::new(cap)",
            "pub(crate) fn with_live_bytes(",
            "pub(crate) const fn live_bytes(",
            "pub(crate) fn account_capacity<T>(",
            "pub(crate) fn account_bytes(",
            ".check_capacity::<T>(capacity)",
            ".account_vec(values)",
            "Error::HostAllocationTooLarge",
            "fn try_collect_exact<",
            "fn try_vec_reserve<",
            "fn try_vec_push<",
            "fn try_vec_extend_from_slice<",
            "incremental_helpers_reserve_before_mutating",
            "phase_budget_reconciles_existing_vector_growth",
        ]),
        PatternCheck::new("CUDA image-derived HTJ2K encode allocation", htj2k)
            .required(&[
                "HostPhaseBudget::new(\"j2k CUDA HTJ2K batch staging\")",
                "HostPhaseBudget::new(\"j2k CUDA HTJ2K component packet graph\")",
                "account_encoded_resolution_owners(",
                ".account_vec(&block.data)",
                "checked_mul(component_iters.len())",
                "j2k CUDA HTJ2K region jobs",
                ".try_vec_with_capacity(subband.code_blocks.len())",
                "j2k CUDA HTJ2K tile packetization",
                "flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes(",
                "packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(",
                "host_budget.live_bytes()",
                "Deliberately codec-bounded",
            ])
            .forbidden(&[".collect()", ".to_vec()"]),
        PatternCheck::new("CUDA packetization plan allocation", packetization)
            .required(&[
                "try_vec_filled(",
                "try_vec_extend_from_slice(",
                "try_vec_push(",
                "try_vec_reserve(",
                "HostPhaseBudget::with_live_bytes(\"j2k CUDA packetization owner graph\"",
                "host_budget: &'a mut crate::allocation::HostPhaseBudget",
                "flatten_cuda_htj2k_packetization_job_classified_with_live_host_bytes(",
                "let mut temporary_budget = HostPhaseBudget::with_live_bytes(",
                "host_budget: &mut HostPhaseBudget",
                "CUDA_HTJ2K_PACKET_MAX_TAG_LEVELS",
                "CUDA_HTJ2K_PACKET_MAX_TAG_NODES",
                "checked_tag_tree_retained_bytes(",
                "tag_tree_oversized_request_is_rejected_before_allocation",
                "tag_tree_actual_capacity_has_exact_and_one_over_boundaries",
                "if state_index >= descriptor_count",
                ".checked_add(1)",
                "sparse_descriptor_state_index_is_rejected_before_state_allocation",
                "descriptor_state_count_addition_is_checked",
                "CudaHtj2kPacketizationPlanError",
                "Self::HostAllocation { what, bytes }",
                "Self::MemoryCapExceeded",
                "packetization_plan_allocation_error",
            ])
            .forbidden(&[".collect()", ".to_vec()"]),
        PatternCheck::new("CUDA packetization allocation failure routing", stage)
            .required(&[
                "flatten_cuda_htj2k_packetization_job_classified(",
                "cuda_packetization_plan_fallback_reason(error)?",
                "CudaHtj2kPacketizationPlanError::HostAllocation { what, bytes }",
                "HostPhaseBudget::new(\"j2k CUDA HTJ2K staged packetization\")",
                "packetize_htj2k_cleanup_packets_with_tag_state_and_live_host_bytes(",
                "host_budget.live_bytes()",
            ])
            .forbidden(&[
                "reason ==",
                "reason.contains(",
                "CUDA_PACKETIZATION_HOST_ALLOCATION_FAILED",
            ]),
        PatternCheck::new("CUDA encode owned-result moves", stage)
            .required(&["output.into_coefficients()", "packetized.into_data()"])
            .forbidden(&[
                "output.coefficients().to_vec()",
                "packetized.data().to_vec()",
            ]),
        PatternCheck::new("CUDA HTJ2K owned-result moves", htj2k)
            .required(&[
                "packetized.into_data()",
                "encoded.into_parts()",
                "encoded.into_code_blocks()",
            ])
            .forbidden(&["packetized.data().to_vec()", "encoded.data().to_vec()"]),
        PatternCheck::new("CUDA DWT conversion allocation", dwt_output)
            .required(&[
                "HostPhaseBudget::new(\"j2k CUDA DWT host output\")",
                ".try_vec_with_capacity(",
            ])
            .forbidden(&["Vec::with_capacity", ".to_vec()"]),
    ]);
}
