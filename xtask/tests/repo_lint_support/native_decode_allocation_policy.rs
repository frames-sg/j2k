// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate native-decode workspace and fallible metadata-growth ratchets.

use std::{collections::BTreeSet, fs};

use super::rust_function_policy::FunctionCalls;
use super::{assert_pattern_checks, repo_root, PatternCheck};

mod downstream;
mod move_only;

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

fn calls(source_name: &str, source: &str, function_name: &str) -> FunctionCalls {
    FunctionCalls::parse(source_name, source, function_name)
}

fn top_level_type_names(source_name: &str, source: &str) -> BTreeSet<String> {
    let syntax = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("parse {source_name} as Rust: {error}"));
    syntax
        .items
        .into_iter()
        .filter_map(|item| match item {
            syn::Item::Enum(item) => Some(item.ident.to_string()),
            syn::Item::Struct(item) => Some(item.ident.to_string()),
            syn::Item::Trait(item) => Some(item.ident.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn ht_decode_statistics_are_owned_separately_from_allocation_state() {
    let module = syn::parse_file(&read("crates/j2k-native/src/j2c/ht_block_decode.rs"))
        .expect("parse HT block decode module");
    let modules = module
        .items
        .into_iter()
        .filter_map(|item| match item {
            syn::Item::Mod(item) if item.content.is_none() => Some(item.ident.to_string()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert!(modules.contains("state"));
    assert!(modules.contains("stats"));

    let state = top_level_type_names(
        "HT allocation state",
        &read("crates/j2k-native/src/j2c/ht_block_decode/state.rs"),
    );
    assert!(state.contains("HtBlockDecodeContext"));
    assert!(state.contains("HtBlockDecodeScratch"));
    assert!(state.is_disjoint(&BTreeSet::from([
        "HtBlockDecodeStats".to_owned(),
        "HtDecodeObserver".to_owned(),
        "NoHtDecodeStats".to_owned(),
        "RecordingHtDecodeStats".to_owned(),
    ])));

    let stats = top_level_type_names(
        "HT decode statistics",
        &read("crates/j2k-native/src/j2c/ht_block_decode/stats.rs"),
    );
    assert!(stats.is_superset(&BTreeSet::from([
        "HtBlockDecodeStats".to_owned(),
        "HtDecodeObserver".to_owned(),
        "NoHtDecodeStats".to_owned(),
        "RecordingHtDecodeStats".to_owned(),
    ])));
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
