// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, calls, read, PatternCheck};

#[test]
fn native_decode_downstream_live_owners_share_the_same_budget() {
    let decode = read("crates/j2k-native/src/j2c/decode.rs");
    let tier1 = read("crates/j2k-native/src/j2c/decode/tier1.rs");
    let parallel = read("crates/j2k-native/src/j2c/decode/subband/parallel.rs");
    let schedule = read("crates/j2k-native/src/j2c/bitplane/schedule.rs");
    let direct = read("crates/j2k-native/src/direct_cpu.rs");
    let direct_component = read("crates/j2k-native/src/direct_cpu/component.rs");
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
        PatternCheck::new("direct CPU prepared scalar wiring", &direct).required(&[
            "mod component;",
            "use component::{",
            "execute_component_plan",
        ]),
        PatternCheck::new("direct CPU prepared scalar execution", &direct_component).required(&[
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
