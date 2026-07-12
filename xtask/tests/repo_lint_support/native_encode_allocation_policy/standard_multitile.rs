// SPDX-License-Identifier: MIT OR Apache-2.0

//! Direct standard multi-tile packet ownership ratchets.

use super::{production_source, read};
use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

fn assert_in_order(name: &str, source: &str, patterns: &[&str]) {
    let mut cursor = 0usize;
    for pattern in patterns {
        let relative = source[cursor..]
            .find(pattern)
            .unwrap_or_else(|| panic!("{name} missing ordered pattern {pattern:?}"));
        cursor += relative + pattern.len();
    }
}

#[test]
fn standard_multitile_modules_stay_focused_and_child_codestream_free() {
    const MODULES: &[(&str, usize)] = &[
        ("crates/j2k-native/src/j2c/encode/multitile.rs", 80),
        ("crates/j2k-native/src/j2c/encode/multitile/execute.rs", 90),
        ("crates/j2k-native/src/j2c/encode/multitile/tile.rs", 240),
        (
            "crates/j2k-native/src/j2c/encode/multitile/tile/grid.rs",
            120,
        ),
        (
            "crates/j2k-native/src/j2c/encode/multitile/finalize.rs",
            180,
        ),
        ("crates/j2k-native/src/j2c/encode/multitile/input.rs", 230),
        (
            "crates/j2k-native/src/j2c/encode/multitile/input/roi.rs",
            100,
        ),
        (
            "crates/j2k-native/src/j2c/encode/multitile/ownership.rs",
            180,
        ),
        ("crates/j2k-native/src/j2c/encode/single_tile.rs", 410),
        ("crates/j2k-native/src/j2c/encode/tile_parts.rs", 180),
        (
            "crates/j2k-native/src/j2c/encode/tile_parts/consume.rs",
            230,
        ),
        (
            "crates/j2k-native/src/j2c/encode/tile_parts/consume/copy.rs",
            230,
        ),
    ];
    for (relative, ceiling) in MODULES {
        let lines = read(relative).lines().count();
        assert!(
            lines <= *ceiling,
            "{relative} has {lines} lines; direct multi-tile ceiling is {ceiling}"
        );
    }

    let removed_parser =
        repo_root().join("crates/j2k-native/src/j2c/encode/multitile/child_codestream.rs");
    assert!(
        !removed_parser.exists(),
        "standard multi-tile encode must not restore child codestream reparsing"
    );
}

#[test]
fn standard_multitile_execution_is_split_without_lint_exceptions() {
    let facade = read("crates/j2k-native/src/j2c/encode/multitile.rs");
    let execution = production_source("crates/j2k-native/src/j2c/encode/multitile/execute.rs");
    let tile_transition = production_source("crates/j2k-native/src/j2c/encode/multitile/tile.rs");
    assert_pattern_checks(&[
        PatternCheck::new("standard multi-tile facade", &facade)
            .required(&[
                "mod execute;",
                "mod tile;",
                "pub(super) struct MultiTileEncodeRequest",
            ])
            .forbidden(&[
                "clippy::too_many_arguments",
                "clippy::too_many_lines",
                "for row in",
            ]),
        PatternCheck::new("standard multi-tile grid orchestration", &execution)
            .required(&[
                "TileGrid::try_new(request)?",
                "TilePosition::try_new(request, &grid, row, column)?",
                "encode_tile(",
                "loop_plan.into_final_plan(",
                "finalize_multitile_codestream(",
            ])
            .forbidden(&["clippy::too_many_arguments", "clippy::too_many_lines"]),
        PatternCheck::new(
            "standard multi-tile direct packet boundary",
            &tile_transition,
        )
        .required(&[
            "encode_single_tile_packets_impl(",
            "packet_encode::packetized_tile_retained_bytes(&packetized)?",
            "drop(input.pixels);",
            "drop(input.roi_regions);",
            "consume_packetized_tile_into_tile_parts(",
            "append_encoded_tile_parts(",
        ])
        .forbidden(&[
            "clippy::too_many_arguments",
            "clippy::too_many_lines",
            "child_codestream",
            "let tile_codestream",
            "extract_single_tile",
            "split_packetized_tile_into_tile_parts",
            "encode_impl(",
        ]),
    ]);
}

#[test]
fn standard_multitile_uses_direct_packet_ownership_without_child_codestreams() {
    // These facades declare out-of-line test modules before their functions,
    // so reading the complete facade is still a production-only source scan.
    let multitile_facade = read("crates/j2k-native/src/j2c/encode/multitile.rs");
    let execution = production_source("crates/j2k-native/src/j2c/encode/multitile/execute.rs");
    let tile_transition = production_source("crates/j2k-native/src/j2c/encode/multitile/tile.rs");
    let tile_grid = production_source("crates/j2k-native/src/j2c/encode/multitile/tile/grid.rs");
    let single_tile = read("crates/j2k-native/src/j2c/encode/single_tile.rs");
    let packet_path = single_tile
        .split_once("pub(in crate::j2c::encode) fn encode_single_tile_packets_impl(")
        .and_then(|(_, tail)| tail.split_once("fn finalize_prepared_single_tile("))
        .map(|(path, _)| path)
        .expect("direct single-tile packet path must remain distinct from finalization");
    let accumulation = production_source("crates/j2k-native/src/j2c/encode/multitile/ownership.rs");
    let input = [
        production_source("crates/j2k-native/src/j2c/encode/multitile/input.rs"),
        production_source("crates/j2k-native/src/j2c/encode/multitile/input/roi.rs"),
    ]
    .concat();
    let finalization = production_source("crates/j2k-native/src/j2c/encode/multitile/finalize.rs");
    let tile_parts = production_source("crates/j2k-native/src/j2c/encode/tile_parts.rs");
    let consume = production_source("crates/j2k-native/src/j2c/encode/tile_parts/consume.rs");
    let split_copy =
        production_source("crates/j2k-native/src/j2c/encode/tile_parts/consume/copy.rs");
    let handoff_modules = [
        multitile_facade.as_str(),
        execution.as_str(),
        tile_transition.as_str(),
        tile_grid.as_str(),
        accumulation.as_str(),
        input.as_str(),
        finalization.as_str(),
        tile_parts.as_str(),
        consume.as_str(),
        split_copy.as_str(),
    ]
    .concat();

    assert_pattern_checks(&[
        PatternCheck::new("single-tile packet-only path", packet_path)
            .required(&[
                "prepare_validated_single_tile(",
                "Ok(prepared.into_packetized_tile())",
                "packet_encode::PacketizedTileData",
            ])
            .forbidden(&[
                "write_single_tile_packetized_codestream_for_session(",
                "finalize_accelerated_codestream(",
                "finalize_staged_codestream(",
            ]),
        PatternCheck::new("direct multi-tile ownership modules", &handoff_modules).forbidden(&[
            ".to_vec(",
            ".clone(",
            "Vec::with_capacity(",
            ".collect::<",
            ".collect()",
            "vec![",
        ]),
        PatternCheck::new("borrowed parent codestream finalization", &finalization)
            .required(&[
                "write_codestream_tiles_accounted_with_peak_check(",
                "data: &tile.data,",
                "packet_lengths: &tile.packet_lengths,",
                "packet_headers: &tile.packet_headers,",
            ])
            .forbidden(&[".to_vec(", ".clone(", "Vec::with_capacity(", ".collect"]),
    ]);
}

#[test]
fn consuming_tile_parts_preflight_and_reconcile_every_owned_allocation() {
    let consume = production_source("crates/j2k-native/src/j2c/encode/tile_parts/consume.rs");
    let copy = production_source("crates/j2k-native/src/j2c/encode/tile_parts/consume/copy.rs");
    assert_pattern_checks(&[
        PatternCheck::new("move-only unsplit packet ownership", &consume)
            .required(&[
                "validate_packetized_tile(&packetized)?;",
                "data: packetized.data,",
                "packet_lengths: packetized.packet_lengths,",
                "packet_headers: packetized.packet_headers,",
                "reconcile_source_and_outer(",
                "drop(packetized);",
                "EncodeError::InternalInvariant",
            ])
            .forbidden(&[".to_vec(", ".clone(", "Vec::with_capacity(", ".collect"]),
        PatternCheck::new("fallible split packet copies", &copy)
            .required(&[
                "struct PartCopyTracker",
                "tracker.before(",
                "requested_part_bytes(",
                "try_copy_slice(",
                "self.before(requested, what)?;",
                ".try_reserve_exact(count)",
                "self.retain(checked_element_bytes::<T>(values.capacity(), what)?, what)?;",
            ])
            .forbidden(&[
                ".to_vec(",
                ".clone(",
                "Vec::with_capacity(",
                ".collect",
                "vec![",
            ]),
    ]);

    let split_outer = consume
        .split_once("let requested_outer =")
        .and_then(|(_, tail)| tail.split_once("let mut data_offset = 0usize;"))
        .map(|(body, _)| body)
        .expect("split outer-owner transition must remain explicit");
    assert_in_order(
        "split outer-owner transition",
        split_outer,
        &[
            "session.checked_phase(",
            ".try_reserve_exact(part_count)",
            "reconcile_source_and_outer(",
        ],
    );
    let unsplit = consume
        .split_once("fn consume_single_part(")
        .and_then(|(_, tail)| tail.split_once("fn reconcile_source_and_outer("))
        .map(|(body, _)| body)
        .expect("unsplit ownership transition must remain explicit");
    assert_in_order(
        "unsplit owner transition",
        unsplit,
        &[
            "session.checked_phase(",
            ".try_reserve_exact(1)",
            "reconcile_source_and_outer(",
            "data: packetized.data,",
        ],
    );
    assert_in_order(
        "per-allocation split copy",
        &copy,
        &[
            "self.before(requested, what)?;",
            ".try_reserve_exact(count)",
            "self.retain(checked_element_bytes::<T>(values.capacity(), what)?, what)?;",
        ],
    );
}

#[test]
fn parent_tile_input_and_accumulation_reconcile_actual_capacity_before_next_allocation() {
    let tile_transition = production_source("crates/j2k-native/src/j2c/encode/multitile/tile.rs");
    let accumulation = production_source("crates/j2k-native/src/j2c/encode/multitile/ownership.rs");
    assert_in_order(
        "initial parent part-owner reservation",
        &accumulation,
        &[
            "session.checked_phase(",
            "parts.try_reserve_exact(tile_count)",
            "parts.capacity()",
            "session.checked_phase(",
        ],
    );
    let tile_input = tile_transition
        .split_once("fn prepare_input(")
        .and_then(|(_, tail)| tail.split_once("fn packetize("))
        .map(|(body, _)| body)
        .expect("tile input ownership transition must remain explicit");
    assert_in_order(
        "tile pixel and ROI allocation reconciliation",
        tile_input,
        &[
            "session.checked_phase(",
            "extract_interleaved_tile(",
            "pixels.capacity()",
            "session.checked_phase(",
            "roi_regions_for_tile(",
            "roi_regions.capacity()",
            "session.checked_phase(",
        ],
    );
}

#[test]
fn standard_multitile_semantic_and_exact_cap_regressions_stay_present() {
    let direct_tests = read("crates/j2k-native/src/j2c/encode/multitile/tests.rs");
    let consume_tests = read("crates/j2k-native/src/j2c/encode/tile_parts/consume/tests.rs");
    let ownership_tests = read("crates/j2k-native/src/j2c/encode/multitile/ownership/tests.rs");
    let integration = read("crates/j2k-native/tests/multitile_tile_parts.rs");
    assert_pattern_checks(&[
        PatternCheck::new("direct packet semantic regressions", &direct_tests).required(&[
            "isolated_child_returns_direct_packet_owners_with_separated_headers",
            "direct_packet_owners_match_single_tile_marker_serialization",
            "assert_eq!(header.plm_packet_lengths, packetized.packet_lengths)",
            "assert_eq!(&codestream[sod + 2..eoc], packetized.data)",
            "direct_multitile_handoff_accepts_exact_peak_and_rejects_one_byte_less",
            "requested == upper && cap == upper - 1",
        ]),
        PatternCheck::new("consuming ownership regressions", &consume_tests).required(&[
            "unsplit_transition_moves_packetized_owners_without_payload_copy",
            "exact move-only handoff peak",
            "split_transition_accepts_exact_overlap_peak_and_rejects_one_byte_less",
            "malformed_packet_metadata_is_a_typed_internal_invariant",
        ]),
        PatternCheck::new("parent owner reservation regressions", &ownership_tests).required(&[
            "initial_part_reservation_accepts_exact_actual_capacity_and_rejects_one_byte_less",
            "requested == exact_cap && cap == exact_cap - 1",
        ]),
        PatternCheck::new("parent marker and tile-part regressions", &integration).required(&[
            "multi_tile_packet_limit_splits_only_parent_tile_parts_and_round_trips",
            "multi_tile_direct_packets_preserve_parent_packet_marker_modes",
            "(\"PLT\", 0x58",
            "(\"PLM\", 0x57",
            "(\"PPM\", 0x60",
            "(\"PPT\", 0x61",
            "assert_roundtrip(&codestream, &pixels)",
        ]),
    ]);
}
