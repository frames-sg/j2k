// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate phase-workspace preflight and staging-lifetime relationships.

use super::super::rust_function_policy::FunctionCalls;
use super::CudaTranscodeSources;

fn calls(sources: &CudaTranscodeSources, source_name: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse_many(source_name, &sources.sources(), function_name)
}

fn assert_aggregate_byte_helpers(sources: &CudaTranscodeSources) {
    calls(
        sources,
        "CUDA aggregate allocation helpers",
        "checked_host_bytes",
    )
    .assert_ordered(
        "CUDA typed byte count",
        &["checked_host_element_count", "saturating_mul"],
    );
    calls(
        sources,
        "CUDA aggregate allocation helpers",
        "checked_host_byte_sum",
    )
    .assert_ordered(
        "CUDA aggregate byte sum",
        &["try_fold", "checked_host_byte_add"],
    );
    calls(
        sources,
        "CUDA aggregate allocation helpers",
        "checked_host_byte_add",
    )
    .assert_contains("CUDA overflow-safe byte addition", &["saturating_add"]);
}

fn assert_phase_preflights(sources: &CudaTranscodeSources) {
    calls(
        sources,
        "CUDA staging workspace validation",
        "validate_staging_and_readback_workspace",
    )
    .assert_ordered(
        "CUDA staging plus readback byte budget",
        &[
            "checked_element_product",
            "checked_element_product",
            "checked_host_byte_sum",
            "checked_host_bytes",
            "checked_host_bytes",
        ],
    );
    calls(
        sources,
        "CUDA widening workspace validation",
        "preflight_dwt97_conversion_budget",
    )
    .assert_contains(
        "CUDA f32-to-f64 aggregate budget",
        &[
            "checked_host_byte_add",
            "checked_host_bytes",
            "checked_host_byte_sum",
            "preflight_bytes",
        ],
    );
    calls(
        sources,
        "CUDA component workspace validation",
        "preflight_component_allocation_budget",
    )
    .assert_contains(
        "CUDA nested component metadata budget",
        &[
            "checked_host_bytes",
            "checked_host_byte_sum",
            "checked_element_product",
            "preflight_bytes",
        ],
    );
}

fn assert_staging_is_dropped_between_phases(sources: &CudaTranscodeSources) {
    calls(sources, "CUDA reversible single dispatch", "run_reversible").assert_contains(
        "CUDA reversible caller-live handoff",
        &["j2k_transcode_reversible_dwt53_and_live_host_bytes"],
    );
    calls(sources, "CUDA 9/7 single dispatch", "run_dwt97").assert_ordered(
        "CUDA single staging lifetime",
        &[
            "validate_staging_and_readback_workspace",
            "HostPhaseBudget::with_live_bytes",
            "flatten_f64_blocks_to_f32",
            "j2k_transcode_dwt97_and_live_host_bytes",
            "drop",
            "dwt97_bands_to_f64_with_live_host_bytes",
        ],
    );
    calls(sources, "CUDA 9/7 batch dispatch", "dispatch_dwt97_batch").assert_ordered(
        "CUDA batch staging lifetime",
        &[
            "validate_staging_and_readback_workspace",
            "HostPhaseBudget::new",
            "try_vec_for_product",
            "j2k_transcode_dwt97_batch_with_pool_and_live_host_bytes",
            "drop",
            "dwt97_batch_bands_to_f64",
        ],
    );
    calls(
        sources,
        "CUDA 9/7 code-block dispatch",
        "dispatch_htj2k97_codeblock_batch",
    )
    .assert_ordered(
        "CUDA code-block staging lifetime",
        &[
            "validate_staging_and_readback_workspace",
            "HostPhaseBudget::new",
            "try_vec_for_product",
            "j2k_transcode_htj2k97_codeblock_batch_with_pool_and_live_host_bytes",
            "drop",
            "codeblock_bands_to_components",
        ],
    );
    calls(
        sources,
        "CUDA component materialization",
        "codeblock_bands_to_components",
    )
    .assert_ordered(
        "CUDA component allocation preflight",
        &[
            "account_codeblock_bands",
            "preflight_component_allocation_budget",
            "try_vec_with_capacity",
            "component_from_subbands",
        ],
    );
}

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    assert!(
        include_str!("phase_budget.rs").lines().count() < 175,
        "CUDA aggregate phase-budget policy must remain a focused module"
    );
    assert_aggregate_byte_helpers(sources);
    assert_phase_preflights(sources);
    assert_staging_is_dropped_between_phases(sources);
}
