// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate native-decode workspace and fallible metadata-growth ratchets.

use std::fs;

use super::rust_function_policy::FunctionCalls;
use super::{assert_pattern_checks, repo_root, PatternCheck};

mod move_only;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

#[test]
fn native_decode_allocation_policy_stays_focused() {
    assert!(
        include_str!("native_decode_allocation_policy.rs")
            .lines()
            .count()
            < 225,
        "native decode allocation policy must stay below its focused-module ratchet"
    );
}

#[test]
fn native_decode_preflights_aggregate_tile_workspace_before_roi_or_allocation() {
    let allocation = read("crates/j2k-native/src/j2c/build/allocation.rs");
    let reuse = read("crates/j2k-native/src/j2c/build/allocation/reuse.rs");
    let decode = read("crates/j2k-native/src/j2c/decode.rs");

    let decode_tile = calls("native tile decode", &decode, "decode_tile");
    decode_tile.assert_ordered(
        "native full-tile preflight before ROI planning",
        &[
            "build::build",
            "RoiPlan::build",
            "segment::parse",
            "decode_component_tile_bit_planes_budgeted",
        ],
    );
    decode_tile.assert_propagated(
        "native tile workspace stages",
        &[
            "build::build",
            "segment::parse",
            "decode_component_tile_bit_planes_budgeted",
        ],
    );

    let prepare = calls(
        "native decomposition allocation owner",
        &allocation,
        "prepare_decomposition_storage",
    );
    prepare.assert_ordered(
        "logical plan, stale-capacity normalization, and actual-capacity validation",
        &[
            "plan::build_allocation_plan",
            "reuse::discard_stale_capacity",
            "account_live_workspace",
            "reuse::reserve_decomposition_storage",
            "account_live_workspace",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("native aggregate allocation plan", &allocation).required(&[
            "struct DecompositionAllocationPlan",
            "total_bytes: usize",
            "checked_include_reusable_elements::<TileDecompositions>",
            "checked_include_reusable_elements::<Decomposition>",
            "checked_include_reusable_elements::<SubBand>",
            "checked_include_reusable_elements::<Precinct>",
            "checked_include_reusable_elements::<CodeBlock>",
            "checked_include_reusable_elements::<Layer>",
            "checked_include_reusable_elements::<TagNode>",
            "checked_include_elements::<f32>",
            "checked_include_elements::<i64>",
            "plan::build_allocation_plan",
        ]),
        PatternCheck::new("native transient-safe reusable allocation", &reuse)
            .required(&[
                "discard_stale_capacity",
                "struct ReallocationBudget",
                "transient_bytes",
                "try_reserve_decode_elements",
                "values.capacity()",
                "new_live_bytes > DEFAULT_MAX_DECODE_BYTES",
            ])
            .forbidden(&["Vec::with_capacity(plan."]),
    ]);
}

#[test]
fn native_decode_metadata_growth_remains_fallible_and_inside_the_aggregate_budget() {
    let reuse = read("crates/j2k-native/src/j2c/build/allocation/reuse.rs");
    let segment = read("crates/j2k-native/src/j2c/segment.rs");
    let lib = read("crates/j2k-native/src/lib.rs");
    let tests = read("crates/j2k-native/src/tests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("native decomposition reservation targets", &reuse).normalized_required(
            &[
                "budget.reserve(&mut storage.tile_decompositions, plan.tile_decompositions)?",
                "budget.reserve(&mut storage.decompositions, plan.decompositions)?",
                "budget.reserve(&mut storage.sub_bands, plan.sub_bands)?",
                "budget.reserve(&mut storage.precincts, plan.precincts)?",
                "budget.reserve(&mut storage.code_blocks, plan.code_blocks)?",
                "budget.reserve(&mut storage.layers, plan.layers)?",
                "budget.reserve(&mut storage.tag_tree_nodes, plan.tag_tree_nodes)?",
                "budget.reserve(&mut storage.coefficients, plan.coefficients)?",
            ],
        ),
        PatternCheck::new("native remaining packet workspace", &segment).normalized_required(&[
            "DEFAULT_MAX_DECODE_BYTES",
            ".checked_sub(structural_workspace_bytes)",
            "let max_segment_count = available_bytes / size_of::<Segment<'_>>()",
            "validate_segment_reallocation_peak",
            "try_reserve_decode_elements",
            "segments.capacity()",
            "packet_workspace_error",
            "packet_segment_growth_respects_remaining_workspace_budget",
        ]),
        PatternCheck::new("native allocation failure mapping", &lib).normalized_required(&[
            "checked_decode_byte_len2(target_len, core::mem::size_of::<T>())?",
            ".try_reserve_exact(target_len - values.len())",
            ".map_err(|_| DecodingError::HostAllocationFailed)?",
        ]),
        PatternCheck::new("native large-tile ROI regression", &tests).required(&[
            "decode_region_rejects_large_full_tile_workspace_before_allocation",
            "rewrite_siz_to_single_large_tile(&mut codestream, 60_000)",
        ]),
    ]);
}

#[test]
fn native_decode_downstream_live_owners_share_the_same_budget() {
    let decode = read("crates/j2k-native/src/j2c/decode.rs");
    let tier1 = read("crates/j2k-native/src/j2c/decode/tier1.rs");
    let parallel = read("crates/j2k-native/src/j2c/decode/subband/parallel.rs");
    let schedule = read("crates/j2k-native/src/j2c/bitplane/schedule.rs");
    let direct = read("crates/j2k-native/src/direct_cpu.rs");
    let direct_allocation = read("crates/j2k-native/src/direct_cpu/allocation.rs");
    let recode = read("crates/j2k-native/src/j2c/recode.rs");
    let postprocess = read("crates/j2k-native/src/color/postprocess.rs");

    calls(
        "native budgeted Tier-1 wrapper",
        &decode,
        "decode_component_tile_bit_planes_budgeted",
    )
    .assert_ordered(
        "prepare, decode, and release Tier-1 workspace",
        &[
            "tier1::prepare_tier1_workspace",
            "decode_component_tile_bit_planes",
            "tier1::release_tier1_workspace",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("native serial Tier-1 high-water", &tier1).required(&[
            "DecodeAllocationBudget::for_storage(storage)?",
            "classic_decode_workspace_bytes",
            "ht_decode_workspace_bytes",
            "actual_bytes > planned_bytes",
            "storage.structural_workspace_bytes",
            "release_tier1_allocations",
        ]),
        PatternCheck::new("native parallel Tier-1 workspaces", &parallel).required(&[
            "preallocate_classic_workspaces",
            "preallocate_ht_workspaces",
            "budget.include_bytes(planned_bytes)?",
            "workspace.allocated_bytes()?",
            "par_iter_mut()",
        ]),
        PatternCheck::new("allocation-free classic segment assembly", &schedule)
            .required(&[
                "extend_preallocated",
                "push_preallocated",
                "coding_passes.checked_add",
            ])
            .forbidden(&[
                "assert_eq!(segment.idx",
                "combined_layers.extend(segment.data)",
            ]),
        PatternCheck::new("direct CPU scalar workspace budget", &direct_allocation).required(&[
            "validate_aggregate_plan",
            "DirectWorkspaceBudget",
            "validate_workspace",
            "actual_scalar_workspace_uses_the_remaining_direct_budget",
        ]),
        PatternCheck::new("direct CPU prepared scalar execution", &direct).required(&[
            "decode_j2k_code_block_scalar_with_workspace",
            "decode_ht_code_block_scalar_with_workspace",
            "workspace.allocated_bytes()?",
        ]),
        PatternCheck::new("fallible coefficient recode handoff", &recode)
            .required(&[
                "RecodeOutputPlan",
                "DecodeAllocationBudget::for_storage",
                "try_reserve_decode_elements",
                "include_capacity_overage",
                "ctx.release_reusable_allocations()",
            ])
            .forbidden(&["Vec::with_capacity", ".to_vec()"]),
        PatternCheck::new("fallible palette and channel postprocess", &postprocess)
            .required(&[
                "DecodeOwnerBudget::for_components",
                "try_reserve_decode_elements",
                "try_resize_decode_elements",
                "sort_unstable_by_key",
                "components.swap",
            ])
            .forbidden(&["Vec::with_capacity", "component.clone()", ".collect::<Vec"]),
    ]);
}
