// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity tile/tile-part metadata ownership ratchets.

use std::fs;

use super::rust_function_policy::FunctionCalls;
use super::{assert_pattern_checks, repo_root, PatternCheck};

const ACCOUNTING_TESTS: &str = "crates/j2k-native/src/j2c/tile/metadata/accounting_tests.rs";

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn tile_metadata_policy_stays_focused() {
    let policy_lines = include_str!("native_tile_metadata_policy.rs")
        .lines()
        .count();
    assert!(policy_lines < 120, "tile metadata policy exceeds 120 lines");
    for (relative, limit) in [
        ("crates/j2k-native/src/j2c/tile/metadata.rs", 640),
        (ACCOUNTING_TESTS, 100),
        ("crates/j2k-native/src/j2c/tile/tile_part.rs", 540),
        ("crates/j2k-native/src/j2c/tile/tile_part/tests.rs", 140),
    ] {
        assert!(
            read(relative).lines().count() < limit,
            "{relative} must stay below {limit} lines"
        );
    }
}

#[test]
fn tile_metadata_uses_one_transactional_actual_capacity_ledger() {
    let metadata = read("crates/j2k-native/src/j2c/tile/metadata.rs");
    let tile_part = read("crates/j2k-native/src/j2c/tile/tile_part.rs");
    let tile = read("crates/j2k-native/src/j2c/tile.rs");

    assert_pattern_checks(&[
        PatternCheck::new("native tile actual-capacity ledger", &metadata)
            .required(&[
                "struct TileMetadataBudget",
                "struct TileMetadataTransaction",
                "try_reserve_accounted_with",
                "validate_transient_peak(live_before, planned_bytes, self.cap)?",
                "values.capacity()",
                "checked_replacement_bytes",
                "temporary_bytes",
                "impl Drop for TileMetadataTransaction",
                "tile_owner_allocation_bytes",
                "validate_owner_graph",
            ])
            .forbidden(&["fn account_elements", "fn release_elements"]),
        PatternCheck::new("native tile-part ownership transitions", &tile_part)
            .required(&[
                "track_temporary_vec",
                "try_reserve_temporary",
                "replace_coding_parameters",
                "replace_quantization",
                "append_temporary",
                "release_temporary_capacity",
                "retain_tile_part_metadata",
                "ppm_header_count",
            ])
            .forbidden(&[
                "try_reserve_decode_elements",
                ".account_elements",
                ".release_elements",
            ]),
        PatternCheck::new("native tile owner construction", &tile)
            .required(&[
                "metadata_budget.try_reserve_retained(&mut tiles, num_tiles)?",
                "inherit_tile_metadata(tile, main_header, &mut metadata_budget)?",
                "metadata_budget.validate_owner_graph(&tiles)?",
            ])
            .forbidden(&[
                "Tile::try_new",
                "try_clone_component_infos",
                "try_reserve_decode_elements(&mut tiles",
            ]),
    ]);

    FunctionCalls::parse("native tile parser", &tile, "parse").assert_ordered(
        "native tile construction and owner handoff",
        &[
            "TileMetadataBudget::for_image",
            "try_reserve_retained",
            "inherit_tile_metadata",
            "parse_tile_part",
            "validate_owner_graph",
            "ParsedTiles::new",
        ],
    );
}

#[test]
fn tile_metadata_regressions_cover_boundaries_and_failure_rollback() {
    let regressions = format!(
        "{}\n{}\n{}\n{}\n{}",
        read("crates/j2k-native/src/j2c/tile/metadata.rs"),
        read(ACCOUNTING_TESTS),
        read("crates/j2k-native/src/j2c/tile/tile_part/tests.rs"),
        read("crates/j2k-native/src/j2c/tile/tile_part/tests/transaction.rs"),
        read("crates/j2k-native/tests/multitile_tile_parts.rs"),
    );
    assert_pattern_checks(&[
        PatternCheck::new("native tile metadata regressions", &regressions).required(&[
            "transient_peak_accepts_exact_cap_and_rejects_one_over",
            "allocator_overcapacity_reconciles_final_owner_before_failing_peak",
            "failed_reserve_keeps_existing_capacity_and_ledger_in_sync",
            "replacement_transfers_new_claim_and_releases_old_capacity",
            "malformed_ppt_rolls_back_temporary_owner_capacity",
            "multi_tile_direct_packets_preserve_parent_packet_marker_modes",
        ]),
    ]);
}
