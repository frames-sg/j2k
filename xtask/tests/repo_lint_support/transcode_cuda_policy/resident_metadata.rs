// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared resident metadata and caller-live staging budget relationships.

use super::super::rust_function_policy::FunctionCalls;
use super::super::{assert_pattern_checks, PatternCheck};
use super::call_arguments::FunctionCallArguments;
use super::CudaTranscodeSources;

mod grouped_budget;

fn calls(sources: &CudaTranscodeSources, label: &str, function: &str) -> FunctionCalls {
    FunctionCalls::parse_many(label, &sources.sources(), function)
}

fn arguments(sources: &CudaTranscodeSources, label: &str, function: &str) -> FunctionCallArguments {
    FunctionCallArguments::parse_many(label, &sources.sources(), function)
}

fn assert_budget_primitives(sources: &CudaTranscodeSources) {
    calls(sources, "CUDA resident live-byte seed", "with_live_bytes").assert_ordered(
        "CUDA resident live-byte seed",
        &["Self::new", "account_bytes"],
    );
    calls(sources, "CUDA resident byte preflight", "preflight_bytes").assert_contains(
        "CUDA resident non-mutating preflight",
        &["account_bytes", "map_err"],
    );
    calls(
        sources,
        "CUDA resident fallible allocation",
        "try_vec_with_capacity_using",
    )
    .assert_ordered(
        "CUDA resident actual-capacity accounting",
        &["preflight_capacity", "allocate", "account_vec"],
    );

    let component_budget = calls(
        sources,
        "CUDA resident component assembly budget",
        "reserve_component_assembly_budget",
    );
    component_budget.assert_count("CUDA component cardinalities", "checked_element_product", 2);
    component_budget.assert_count("CUDA component metadata types", "checked_host_bytes", 3);
    component_budget.assert_contains(
        "CUDA nested component aggregate",
        &["checked_host_byte_sum", "preflight_bytes"],
    );
}

fn assert_plan_and_output_preflights(sources: &CudaTranscodeSources) {
    let build = calls(
        sources,
        "CUDA resident grouped planning",
        "build_resident_subband_group_plans",
    );
    build.assert_ordered(
        "CUDA resident grouped plan budget",
        &[
            "ResidentMetadataBudget::with_live_bytes",
            "try_vec_with_capacity",
            "resident_subband_encode_plan",
        ],
    );
    build.assert_count(
        "CUDA four-subband grouped planning",
        "resident_subband_encode_plan",
        4,
    );

    for function in ["resident_group_targets", "resident_targets"] {
        calls(sources, "CUDA resident target metadata", function).assert_contains(
            "CUDA resident targets use the shared phase budget",
            &["try_vec_with_capacity"],
        );
    }
    calls(
        sources,
        "CUDA resident subband planning",
        "resident_subband_encode_plan",
    )
    .assert_ordered(
        "CUDA resident jobs and shapes",
        &["try_vec_with_capacity", "try_vec_with_capacity"],
    );

    for function in [
        "split_resident_subband_blocks",
        "split_resident_compact_subband_blocks",
    ] {
        calls(sources, "CUDA resident split output", function).assert_ordered(
            "CUDA resident split outer and inner metadata",
            &[
                "preflight_split_metadata",
                "try_vec_with_capacity",
                "try_vec_with_capacity",
            ],
        );
    }
    for function in [
        "assemble_preencoded_components_with_budget",
        "assemble_compact_preencoded_components_with_budget",
    ] {
        let assembly = calls(sources, "CUDA resident component assembly", function);
        assembly.assert_ordered(
            "CUDA resident nested component allocation",
            &[
                "reserve_component_assembly_budget",
                "try_vec_with_capacity",
                "try_vec_from_array",
            ],
        );
        assembly.assert_count(
            "CUDA resident resolution and subband vectors",
            "try_vec_from_array",
            3,
        );
    }
}

fn assert_one_budget_threads_every_resident_phase(sources: &CudaTranscodeSources) {
    let build = arguments(
        sources,
        "CUDA resident grouped planning",
        "build_resident_subband_group_plans",
    );
    build.assert_ident_argument(
        "CUDA resident live-byte seed",
        "with_live_bytes",
        "live_metadata_bytes",
        1,
    );
    build.assert_mut_ident_argument(
        "CUDA four-subband grouped planning",
        "resident_subband_encode_plan",
        "budget",
        4,
    );

    for owner in [
        "encode_resident_subbands",
        "encode_resident_compact_subbands",
    ] {
        let flow = arguments(sources, "CUDA fixed resident orchestration", owner);
        flow.assert_mut_ident_argument(
            "CUDA fixed resident planning",
            "resident_subband_encode_plan",
            "budget",
            4,
        );
        flow.assert_mut_ident_argument(
            "CUDA fixed resident targets",
            "resident_targets",
            "budget",
            1,
        );
    }
    arguments(
        sources,
        "CUDA fixed resident orchestration",
        "encode_resident_subbands",
    )
    .assert_mut_ident_argument(
        "CUDA fixed resident split",
        "split_resident_subband_blocks",
        "budget",
        4,
    );
    arguments(
        sources,
        "CUDA fixed compact resident orchestration",
        "encode_resident_compact_subbands",
    )
    .assert_mut_ident_argument(
        "CUDA fixed compact resident split",
        "split_resident_compact_subband_blocks",
        "budget",
        4,
    );

    assert_group_flow(
        sources,
        "device_band_groups_to_preencoded_components",
        "split_resident_subband_blocks",
        "assemble_preencoded_components_with_budget",
    );
    assert_group_flow(
        sources,
        "device_band_groups_to_compact_preencoded_components",
        "split_resident_compact_subband_blocks",
        "assemble_compact_preencoded_components_with_budget",
    );
}

fn assert_group_flow(sources: &CudaTranscodeSources, owner: &str, split: &str, assembly: &str) {
    let flow = arguments(sources, "CUDA grouped resident orchestration", owner);
    flow.assert_ident_argument(
        "CUDA grouped resident live-byte seed",
        "build_resident_subband_group_plans",
        "live_metadata_bytes",
        1,
    );
    flow.assert_mut_ident_argument(
        "CUDA grouped resident targets",
        "resident_group_targets",
        "budget",
        1,
    );
    flow.assert_mut_ident_argument("CUDA grouped resident split", split, "budget", 4);
    flow.assert_mut_ident_argument("CUDA grouped resident assembly", assembly, "budget", 1);

    calls(sources, "CUDA grouped resident orchestration", owner).assert_ordered(
        "CUDA grouped outputs share the resident budget",
        &[
            "build_resident_subband_group_plans",
            "resident_group_targets",
            "try_vec_with_capacity",
            split,
            assembly,
        ],
    );
}

fn assert_actual_source_owners_are_reconciled(sources: &CudaTranscodeSources) {
    calls(
        sources,
        "CUDA resident preencoded source ownership",
        "account_preencoded_subband_sources",
    )
    .assert_count(
        "CUDA resident outer, block, and payload owners",
        "account_vec",
        3,
    );
    calls(
        sources,
        "CUDA resident compact source ownership",
        "account_compact_subband_sources",
    )
    .assert_count(
        "CUDA compact resident outer and block owners",
        "account_vec",
        2,
    );
    calls(
        sources,
        "CUDA resident compact assembly",
        "assemble_compact_preencoded_components",
    )
    .assert_ordered(
        "CUDA compact payload remains in the assembly phase budget",
        &[
            "ResidentMetadataBudget::new",
            "account_vec",
            "account_compact_subband_sources",
            "assemble_compact_preencoded_components_with_budget",
        ],
    );

    calls(
        sources,
        "CUDA resident output ownership",
        "encode_resident_subbands",
    )
    .assert_ordered(
        "CUDA resident runtime outputs join the phase budget before splitting",
        &[
            "into_code_blocks",
            "account_vec",
            "split_resident_subband_blocks",
        ],
    );
    calls(
        sources,
        "CUDA compact resident output ownership",
        "encode_resident_compact_subbands",
    )
    .assert_ordered(
        "CUDA compact payload and block owners join the phase budget",
        &[
            "into_payload_and_code_blocks",
            "account_vec",
            "account_vec",
            "split_resident_compact_subband_blocks",
        ],
    );
}

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    assert_budget_primitives(sources);
    grouped_budget::assert_policy(sources);
    assert_plan_and_output_preflights(sources);
    assert_one_budget_threads_every_resident_phase(sources);
    assert_actual_source_owners_are_reconciled(sources);
    assert_pattern_checks(&[PatternCheck::new(
        "CUDA resident aggregate allocation regressions",
        &sources.full_combined(),
    )
    .required(&[
        "resident_metadata_budget_rejects_aggregate_subcap_reservations",
        "resident_metadata_budget_counts_live_caller_bytes",
        "nested_component_metadata_is_preflighted_as_one_budget",
        "live_dispatch_metadata_is_counted_with_staging",
    ])]);
    assert!(
        include_str!("resident_metadata.rs").lines().count() < 325,
        "CUDA resident metadata policy must remain focused"
    );
    assert!(
        include_str!("call_arguments.rs").lines().count() < 175,
        "CUDA call-argument policy helper must remain focused"
    );
    assert!(
        include_str!("resident_metadata/grouped_budget.rs")
            .lines()
            .count()
            < 175,
        "CUDA grouped resident budget policy must remain focused"
    );
}
