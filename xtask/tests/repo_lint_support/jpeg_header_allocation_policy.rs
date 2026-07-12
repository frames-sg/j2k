// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact, fallible JPEG header and progressive-script ownership ratchets.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_header_uses_versioned_table_arenas_and_one_parse_budget() {
    let root = repo_root();
    let types = fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/header/types.rs"))
        .expect("read JPEG parsed header types");
    let progressive =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/header/progressive.rs"))
            .expect("read JPEG progressive header walker");
    let script =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/header/progressive/script.rs"))
            .expect("read JPEG progressive script validator");
    let tables = fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/tables/state.rs"))
        .expect("read JPEG table-version state");
    let allocation = fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/allocation.rs"))
        .expect("read JPEG parse allocation ledger");
    let sof = fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/sof.rs"))
        .expect("read JPEG SOF parser");
    let parse_sources = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/parse/header/walk.rs",
            "crates/j2k-jpeg/src/parse/header/progressive.rs",
            "crates/j2k-jpeg/src/parse/header/progressive/application.rs",
            "crates/j2k-jpeg/src/parse/header/progressive/pending.rs",
            "crates/j2k-jpeg/src/parse/header/progressive/script.rs",
            "crates/j2k-jpeg/src/parse/tables/dht.rs",
            "crates/j2k-jpeg/src/parse/tables/dqt.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("move-only compact progressive scan", &types)
            .required(&[
                "pub(crate) struct ParsedProgressiveScan",
                "table_state: ProgressiveTableState",
                "terminal_offset: usize",
                "terminal_code: u8",
                "fn retained_allocation_bytes",
                "self.progressive_scans.capacity()",
                "self.huffman_tables.retained_allocation_bytes()?",
                "self.quant_tables.retained_allocation_bytes()?",
            ])
            .forbidden(&[
                "#[derive(Debug, Clone)]\npub(crate) struct ParsedProgressiveScan",
                "entropy_end:",
            ]),
        PatternCheck::new("versioned raw JPEG table arenas", &tables)
            .required(&[
                "struct RawHuffmanTableId(NonZeroU32)",
                "struct RawQuantTableId(NonZeroU32)",
                "versions: Vec<RawHuffmanTable>",
                "versions: Vec<[u16; 64]>",
                "struct ProgressiveTableState",
                "fn capture(",
                "fn active_dc_version_index(",
                "fn active_ac_version_index(",
                "fn dc_version_index(",
                "fn ac_version_index(",
                "unvalidated DHT class reached table definition",
                "unvalidated DQT slot reached table definition",
            ])
            .forbidden(&["Arc<", "Box<", "unreachable!"]),
        PatternCheck::new("single running parse metadata ledger", &allocation).required(&[
            "struct ParsedMetadataBudget",
            "try_reserve_for_len_with_live_budget(",
            "pub(crate) fn try_push<T>",
            "pub(crate) fn finish(",
            "ensure_retained_metadata_bytes",
        ]),
        PatternCheck::new("progressive table-state capture", &progressive)
            .required(&[
                "mod application;",
                "mod pending;",
                "mod script;",
                "ProgressiveTableState::capture(",
                "script.record_scan(",
                "self.script.finish_terminal(",
                "scan.finish(terminal_offset, terminal_code)",
                "parse_dht(",
                "parse_dqt(",
                "Warning::MissingEoi",
                "marker: MarkerKind::Soi",
            ])
            .forbidden(&[
                "huffman_tables.clone()",
                "quant_tables.clone()",
                "push_progressive_scan",
            ]),
        PatternCheck::new("heap-free progressive scan script", &script)
            .required(&[
                "struct ProgressiveScriptState",
                "approximation: [[u8; COEFFICIENTS]; MAX_COMPONENTS]",
                "fn record_scan(",
                "fn finish_terminal(",
                "ProgressiveScanStateError::DuplicateInitial",
                "ProgressiveScanStateError::RefinementBeforeInitial",
                "ProgressiveScanStateError::RefinementMismatch",
                "ProgressiveScanStateError::MissingInitialDc",
            ])
            .forbidden(&["Vec<", "Box<", "Arc<", "HashMap<"]),
    ]);
    assert_inline_metadata_and_fallible_growth(&sof, &parse_sources);
}

fn assert_inline_metadata_and_fallible_growth(sof: &str, parse_sources: &str) {
    assert_pattern_checks(&[
        PatternCheck::new("inline frame component metadata", sof)
            .required(&[
                "struct FrameComponentValues",
                "entries: [u8; MAX_FRAME_COMPONENTS]",
                "component_ids: FrameComponentValues",
                "quant_table_ids: FrameComponentValues",
            ])
            .forbidden(&[
                "component_ids: Vec<u8>",
                "quant_table_ids: Vec<u8>",
                "try_vec_with_capacity(nf as usize)",
            ]),
        PatternCheck::new("fallible parse production growth", parse_sources).forbidden(&[
            "Vec::with_capacity",
            "try_reserve_exact",
            "huffman_tables.clone()",
            "quant_tables.clone()",
        ]),
    ]);
}

#[test]
fn jpeg_header_modules_and_boundary_tests_stay_focused() {
    let root = repo_root();
    let shell = fs::read_to_string(root.join("crates/j2k-jpeg/src/parse/header.rs"))
        .expect("read JPEG header shell");
    assert!(shell.lines().count() <= 25);
    for declaration in [
        "mod inspect;",
        "mod markers;",
        "mod progressive;",
        "mod types;",
        "mod validation;",
        "mod walk;",
        "mod tests;",
    ] {
        assert!(
            shell.contains(declaration),
            "missing header boundary {declaration}"
        );
    }

    let limits = [
        ("inspect.rs", 110usize),
        ("markers.rs", 125),
        ("progressive.rs", 225),
        ("progressive/application.rs", 45),
        ("progressive/eof_tests.rs", 55),
        ("progressive/pending.rs", 55),
        ("progressive/script.rs", 165),
        ("progressive/script/tests.rs", 155),
        ("progressive/terminal_tests.rs", 70),
        ("types.rs", 100),
        ("validation.rs", 125),
        ("walk.rs", 265),
        ("tests.rs", 270),
    ];
    for (file, limit) in limits {
        let relative = format!("crates/j2k-jpeg/src/parse/header/{file}");
        let source = fs::read_to_string(root.join(&relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() <= limit,
            "{relative} must remain at or below {limit} lines"
        );
        assert!(!source.contains("include!(") && !source.contains("use super::*"));
    }

    let tests = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/parse/allocation.rs",
            "crates/j2k-jpeg/src/parse/header/tests.rs",
            "crates/j2k-jpeg/src/parse/header/progressive/eof_tests.rs",
            "crates/j2k-jpeg/src/parse/header/progressive/script/tests.rs",
            "crates/j2k-jpeg/src/parse/header/progressive/terminal_tests.rs",
            "crates/j2k-jpeg/src/parse/tables/tests.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("JPEG header allocation regressions", &tests).required(&[
            "parsed_growth_counts_old_and_replacement_peak_exactly",
            "progressive_scan_metadata_boundary_shares_the_context_cap",
            "warning_metadata_boundary_shares_the_context_cap",
            "repeated_sos_reuses_versioned_table_state_without_snapshot_growth",
            "dht_redefinition_creates_one_version_and_snapshot_tracks_it",
            "dqt_redefinition_creates_one_version_and_snapshot_tracks_it",
            "table_state_rejects_an_unvalidated_huffman_class_without_panicking",
            "table_state_rejects_an_unvalidated_quant_slot_without_panicking",
            "rejects_zero_quantizers_without_mutating_table_state",
            "valid_multiscan_script_advances_dc_and_partial_ac_state",
            "duplicate_initial_dc_reports_first_coefficient_and_prior_state",
            "overlapping_initial_ac_ranges_report_shared_boundary",
            "refinement_before_initial_reports_coefficient_63_boundary",
            "skipped_refinement_reports_previous_and_requested_levels",
            "eoi_requires_initial_dc_for_each_frame_component_only",
            "eof_without_eoi_is_retained_as_a_missing_eoi_warning",
            "eof_still_requires_initial_dc_for_every_frame_component",
            "trailing_ff_marker_prefix_is_truncated_not_missing_eoi",
            "embedded_second_soi_after_entropy_is_a_duplicate_marker_error",
            "valid_multiscan_parser_records_each_exact_terminal",
        ]),
    ]);
}
