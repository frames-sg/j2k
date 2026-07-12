// SPDX-License-Identifier: MIT OR Apache-2.0

//! Inline, single-owner JPEG prepared-table architecture ratchets.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

#[test]
fn jpeg_prepared_tables_use_checked_ids_and_single_inline_arenas() {
    let root = repo_root();
    let huffman = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/huffman.rs"))
        .expect("read JPEG Huffman source");
    let sequential = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/sequential/plan.rs",
            "crates/j2k-jpeg/src/entropy/sequential/plan/resolved.rs",
        ],
    );
    let progressive = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/entropy/progressive.rs",
            "crates/j2k-jpeg/src/entropy/progressive/model.rs",
            "crates/j2k-jpeg/src/entropy/progressive/terminal.rs",
        ],
    );
    let context = fs::read_to_string(root.join("crates/j2k-jpeg/src/context.rs"))
        .expect("read JPEG context source");
    let bit_reader = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/internal/bit_reader.rs",
            "crates/j2k-jpeg/src/internal/bit_reader/terminal.rs",
        ],
    );

    assert_pattern_checks(&[
        PatternCheck::new("inline compiled Huffman values", &huffman)
            .required(&[
                "fast_dc: [u32; FAST_ENTRIES]",
                "fast_ac: [u32; FAST_ENTRIES]",
                "struct PreparedHuffmanTableId(NonZeroU32)",
                "struct PreparedHuffmanTables",
                "entries: Vec<HuffmanTable>",
                "fn retained_allocation_bytes(&self)",
                "prepared_arena_ids_are_checked_and_capacity_bounded",
            ])
            .forbidden(&[
                "use alloc::boxed::Box",
                "fast_dc: Box<",
                "fast_ac: Box<",
                "Arc<HuffmanTable>",
            ]),
        PatternCheck::new("sequential checked table references", &sequential)
            .required(&[
                "quant: [u16; 64]",
                "dc_table: Option<PreparedHuffmanTableId>",
                "ac_table: Option<PreparedHuffmanTableId>",
                "huffman_tables: PreparedHuffmanTables",
                "struct ResolvedPreparedComponentPlan",
                "fn resolve_component",
                "fn retained_allocation_bytes(&self)",
            ])
            .forbidden(&["Arc<", "Box<"]),
        PatternCheck::new("progressive checked table references", &progressive)
            .required(&[
                "quant: [u16; 64]",
                "dc_table: Option<PreparedHuffmanTableId>",
                "ac_table: Option<PreparedHuffmanTableId>",
                "huffman_tables: PreparedHuffmanTables",
                "terminal_offset: usize",
                "terminal_code: u8",
                "fn finish_progressive_scan(",
                "fn retained_allocation_bytes(&self)",
            ])
            .forbidden(&["Arc<", "Box<"]),
        PatternCheck::new("bounded inline context table cache", &context)
            .required(&[
                "huffman_tables: Vec<Option<CachedHuffmanTable>>",
                "fn ensure_huffman_cache_slots",
                "try_reserve_for_len_with_live_budget(",
                "table: HuffmanTable",
                "cache_hit_clone_shares_one_exact_external_live_budget",
            ])
            .forbidden(&["Arc<HuffmanTable>", "Arc<[u16; 64]>"]),
        PatternCheck::new("progressive terminal bit accounting", &bit_reader).required(&[
            "synthetic_bits: u8",
            "allow_eof_padding: bool",
            "while code_pos < self.bytes.len() && self.bytes[code_pos] == 0xff",
            "fn unread_real_bits(",
            "fn unread_real_bits_are_ones(",
        ]),
    ]);
}

#[test]
fn progressive_decode_owners_stay_focused_and_real() {
    let root = repo_root();
    let facade = fs::read_to_string(root.join("crates/j2k-jpeg/src/entropy/progressive.rs"))
        .expect("read progressive facade");
    for declaration in [
        "mod allocation;",
        "mod model;",
        "mod render;",
        "mod scan;",
        "mod terminal;",
        "mod tests;",
    ] {
        assert!(facade.contains(declaration), "missing {declaration}");
    }
    for (relative, max_lines) in [
        ("crates/j2k-jpeg/src/entropy/progressive.rs", 70usize),
        ("crates/j2k-jpeg/src/entropy/progressive/model.rs", 130),
        ("crates/j2k-jpeg/src/entropy/progressive/allocation.rs", 185),
        ("crates/j2k-jpeg/src/entropy/progressive/scan.rs", 450),
        ("crates/j2k-jpeg/src/entropy/progressive/terminal.rs", 135),
        (
            "crates/j2k-jpeg/src/entropy/progressive/terminal/mismatch_tests.rs",
            50,
        ),
        (
            "crates/j2k-jpeg/src/entropy/progressive/terminal/tests.rs",
            115,
        ),
        ("crates/j2k-jpeg/src/entropy/progressive/render.rs", 230),
        ("crates/j2k-jpeg/src/entropy/progressive/tests.rs", 105),
        ("crates/j2k-jpeg/src/internal/bit_reader/terminal.rs", 45),
        ("crates/j2k-jpeg/tests/progressive_terminal.rs", 40),
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        assert!(
            source.lines().count() <= max_lines,
            "{relative} must remain at or below {max_lines} lines"
        );
        assert!(!source.contains("include!(") && !source.contains("#[path"));
        assert!(!source.contains("use super::*"));
    }

    let terminal_tests = read_source_files(
        root,
        &[
            "crates/j2k-jpeg/src/parse/header/progressive/terminal_tests.rs",
            "crates/j2k-jpeg/src/entropy/progressive/terminal/mismatch_tests.rs",
            "crates/j2k-jpeg/src/entropy/progressive/terminal/tests.rs",
            "crates/j2k-jpeg/tests/progressive_terminal.rs",
        ],
    );
    assert_pattern_checks(&[PatternCheck::new(
        "progressive terminal behavior regressions",
        &terminal_tests,
    )
    .required(&[
        "valid_multiscan_parser_records_each_exact_terminal",
        "valid_multiscan_boundaries_accept_exact_sos_and_eoi_markers",
        "valid_multiscan_and_missing_eoi_decode_identically_with_typed_warning",
        "trailing_ff_marker_prefix_is_truncated_not_missing_eoi",
        "embedded_second_soi_after_entropy_is_a_duplicate_marker_error",
        "residual_eob_run_is_rejected_at_scan_end",
        "complete_excess_entropy_byte_is_rejected",
        "unexpected_intermediate_restart_does_not_match_the_parsed_eoi_boundary",
    ])]);
}

#[test]
fn jpeg_parsed_to_prepared_handoff_is_consuming_and_exactly_budgeted() {
    let sources = read_source_files(
        repo_root(),
        &[
            "crates/j2k-jpeg/src/decoder.rs",
            "crates/j2k-jpeg/src/decoder/plan.rs",
            "crates/j2k-jpeg/src/decoder/plan/construction.rs",
            "crates/j2k-jpeg/src/decoder/plan/progressive_quant.rs",
            "crates/j2k-jpeg/src/decoder/allocation.rs",
        ],
    );
    assert_pattern_checks(&[
        PatternCheck::new("consuming prepared metadata handoff", &sources).required(&[
            "struct PreparedDecoderMetadata",
            "fn prepare_header_with_external_live(",
            "external_live_bytes: usize",
            "header: ParsedHeader",
            "warnings: header.warnings",
            "ensure_prepared_construction_fits(&header, prepared_bytes)?;",
            "retained_allocation_bytes_excluding_cpu_checkpoint_cache",
            "progressive.retained_allocation_bytes()?",
            "retained_baseline_excludes_existing_cpu_checkpoint_cache_capacity",
        ]),
    ]);
}

#[test]
fn jpeg_prepared_construction_uses_one_actual_capacity_ledger() {
    let root = repo_root();
    let plan = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/plan.rs"))
        .expect("read JPEG decoder plan source");
    let construction =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/plan/construction.rs"))
            .expect("read JPEG prepared-construction ledger source");
    let construction_tests =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/plan/construction/tests.rs"))
            .expect("read JPEG prepared-construction regression source");

    assert!(
        construction.lines().count() <= 230,
        "decoder/plan/construction.rs must remain at or below 230 lines"
    );
    assert!(!construction.contains("include!(") && !construction.contains("#[path"));
    assert!(!construction.contains("use super::*"));
    assert!(
        construction_tests.lines().count() <= 120,
        "decoder/plan/construction/tests.rs must remain at or below 120 lines"
    );
    assert!(!construction_tests.contains("include!(") && !construction_tests.contains("#[path"));
    assert!(!construction_tests.contains("use super::*"));
    assert_pattern_checks(&[
        PatternCheck::new("prepared construction lifecycle", &plan).required(&[
            "PreparedConstructionBudget::with_external_live(",
            "construction.rebase_after_plan_cache(",
            "construction.verify_retained(",
            "construction.try_vec(",
            "construction.try_huffman_tables(",
        ]),
        PatternCheck::new("actual-capacity construction ledger", &construction).required(&[
            "try_reserve_for_len_with_live_budget(",
            "resolve_huffman_table_with_live_budget(",
        ]),
        PatternCheck::new(
            "actual-capacity construction regressions",
            &construction_tests,
        )
        .required(&[
            "forced_spare_vector_capacity_is_counted_exactly",
            "prior_actual_capacity_is_used_by_the_next_reserve",
        ]),
        PatternCheck::new("unbudgeted prepared vector exclusion", &plan).forbidden(&[
            "try_vec_with_capacity(",
            "PreparedHuffmanTables::try_with_capacity(",
        ]),
    ]);

    let progressive = plan.find("Self::build_progressive_plan(").unwrap();
    let companion = plan
        .find("Self::build_progressive_host_output_plan(")
        .unwrap();
    assert!(
        progressive < companion,
        "progressive retained capacity must feed the companion plan"
    );
}

#[test]
fn progressive_quant_tables_bind_to_first_component_scan_snapshots() {
    let root = repo_root();
    let plan = fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/plan.rs"))
        .expect("read JPEG decoder plan source");
    let latch =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/decoder/plan/progressive_quant.rs"))
            .expect("read progressive quant-table latch source");
    let tests = fs::read_to_string(
        root.join("crates/j2k-jpeg/src/decoder/plan/progressive_quant/tests.rs"),
    )
    .expect("read progressive quant-table latch tests");
    let error = fs::read_to_string(root.join("crates/j2k-jpeg/src/error.rs"))
        .expect("read JPEG error taxonomy");

    for (relative, source, max_lines) in [
        ("decoder/plan/progressive_quant.rs", latch.as_str(), 90usize),
        (
            "decoder/plan/progressive_quant/tests.rs",
            tests.as_str(),
            170,
        ),
    ] {
        assert!(
            source.lines().count() <= max_lines,
            "crates/j2k-jpeg/src/{relative} must remain at or below {max_lines} lines"
        );
        assert!(!source.contains("include!(") && !source.contains("#[path"));
        assert!(!source.contains("use super::*"));
    }

    assert_pattern_checks(&[
        PatternCheck::new("progressive first-use DQT latch", &latch)
            .required(&[
                "let resolved = *header",
                ".resolve(&parsed.table_state, table_id)",
                "ProgressiveQuantTableChanged",
            ])
            .forbidden(&["quant_tables.entries", "InternalInvariant"]),
        PatternCheck::new("latched progressive and companion plans", &plan).required(&[
            "latch_progressive_quant_tables(header)?",
            "let latched_quant_tables",
            ".table(output_index)",
            "quant: component.quant",
        ]),
        PatternCheck::new("typed progressive DQT input error", &error).required(&[
            "ProgressiveQuantTableChanged",
            "Self::ProgressiveQuantTableChanged",
        ]),
        PatternCheck::new("progressive DQT behavior coverage", &tests).required(&[
            "valid_redefinitions_bind_each_component_at_first_scan",
            "quant_redefinition_after_component_latch_is_rejected",
            "assert_eq!(error.offset(), Some(offending_offset))",
        ]),
    ]);
}
