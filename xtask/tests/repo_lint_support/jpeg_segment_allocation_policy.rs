// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation and ownership integrity for public JPEG segment utilities.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn segment_source() -> String {
    fs::read_to_string(repo_root().join("crates/j2k-jpeg/src/segment.rs"))
        .expect("read JPEG segment utilities")
}

fn table_normalization_source() -> String {
    fs::read_to_string(repo_root().join("crates/j2k-jpeg/src/segment/table_normalization.rs"))
        .expect("read JPEG table normalization helper")
}

fn allocation_tests_source() -> String {
    fs::read_to_string(repo_root().join("crates/j2k-jpeg/src/segment/allocation_tests.rs"))
        .expect("read JPEG segment allocation tests")
}

#[test]
fn jpeg_segment_rewrite_and_tiff_assembly_preflight_fallible_output() {
    let source = segment_source();
    assert_pattern_checks(&[
        PatternCheck::new("move-only prepared JPEG owner", &source)
            .required(&[
                "#[derive(Debug, PartialEq, Eq)]\npub enum PreparedJpeg<'a>",
                "pub fn try_clone(&self) -> Result<Self, JpegError>",
                "Self::Borrowed(bytes) => Ok(Self::Borrowed(bytes))",
                "Self::Owned(bytes) => Ok(Self::Owned(try_copy_bytes(bytes)?))",
            ])
            .forbidden(&["#[derive(Debug, Clone, PartialEq, Eq)]\npub enum PreparedJpeg<'a>"]),
        PatternCheck::new("JPEG SOF rewrite allocation", &source).required(&[
            "fn try_copy_bytes",
            "checked_allocation_bytes::<u8>(input.len())?",
            "try_vec_with_capacity(input.len())?",
        ]),
        PatternCheck::new("TIFF JPEG aggregate output plan", &source).required(&[
            "fn abbreviated_output_len",
            "fn checked_abbreviated_output_len",
            "checked_add_allocation_bytes",
            "let output_len = abbreviated_output_len(",
            "opts.duplicate_table_policy",
            "let mut out = try_vec_with_capacity(output_len)?",
        ]),
    ]);
}

#[test]
fn jpeg_normalized_segments_remain_borrowed_until_the_single_output_copy() {
    let source = segment_source();
    let normalization = table_normalization_source();
    assert_pattern_checks(&[
        PatternCheck::new("JPEG table normalization module boundary", &source)
            .required(&["mod table_normalization;"])
            .forbidden(&["enum TableKey", "fn for_each_dqt_definition("]),
        PatternCheck::new("borrowed JPEG table definitions", &normalization)
            .required(&[
                "pub(super) enum NormalizedSegment<'a>",
                "struct NormalizationState<'a>",
                "fn for_each_dqt_definition<'a>(",
                "fn for_each_dht_definition<'a>(",
                "definitions: [&'a [u8]; MAX_DISTINCT_TABLES]",
            ])
            .forbidden(&[
                "collect_normalized_segments",
                "Vec<(TableKey, Vec<u8>)>",
                "definition.bytes.to_vec()",
                "input.to_vec()",
            ]),
    ]);
}

#[test]
fn jpeg_segment_allocation_boundaries_remain_covered() {
    let source = segment_source();
    let normalization = table_normalization_source();
    let allocation_tests = allocation_tests_source();
    let combined = format!("{source}\n{normalization}\n{allocation_tests}");
    assert_pattern_checks(&[
        PatternCheck::new("JPEG segment allocation regressions", &combined).required(&[
            "abbreviated_output_length_has_an_exact_shared_cap_boundary",
            "normalized_segments_stay_borrowed_and_identical_tables_are_deduplicated",
        ]),
    ]);
}

#[test]
fn jpeg_table_normalization_stays_focused_and_covers_multi_table_markers() {
    let root = repo_root();
    let segment = segment_source();
    let normalization = table_normalization_source();
    let normalization_tests =
        fs::read_to_string(root.join("crates/j2k-jpeg/src/segment/table_normalization/tests.rs"))
            .expect("read JPEG table normalization tests");
    let allocation_tests = allocation_tests_source();
    let integration = fs::read_to_string(root.join("crates/j2k-jpeg/tests/inspect.rs"))
        .expect("read public JPEG segment integration tests");

    assert!(
        segment.lines().count() < 930,
        "JPEG segment facade must remain below 930 lines"
    );
    assert!(
        normalization.lines().count() < 400,
        "JPEG table normalization helper must remain below 400 lines"
    );
    assert!(
        normalization_tests.lines().count() < 260,
        "JPEG table normalization tests must remain below 260 lines"
    );
    assert!(
        allocation_tests.lines().count() < 100,
        "JPEG segment allocation tests must remain below 100 lines"
    );
    assert_pattern_checks(&[
        PatternCheck::new("distinct duplicate table policy", &segment).required(&[
            "Coalesce byte-identical duplicate definitions while rejecting conflicts",
            "Preserve byte-identical redefinitions but reject conflicting definitions",
        ]),
        PatternCheck::new("complete DQT/DHT definition walk", &normalization)
            .required(&[
                "while position < payload.len()",
                "let precision = selector >> 4;",
                "let class = selector >> 4;",
                "let value_count = payload[position + 1..counts_end]",
                "DuplicateTablePolicy::AllowIdentical",
                "DuplicateTablePolicy::RejectConflicting",
            ])
            .forbidden(&["payload.first()"]),
        PatternCheck::new(
            "multi-table and malformed marker regressions",
            &normalization_tests,
        )
        .required(&[
            "multi_table_dqt_conflict_uses_the_later_identifier",
            "multi_table_dht_distinguishes_dc_and_ac_definitions",
            "duplicate_policies_preserve_or_coalesce_identical_definitions",
            "allow_identical_rebuilds_partially_deduplicated_multi_table_markers",
            "malformed_or_truncated_dqt_definitions_fail_closed",
            "malformed_or_truncated_dht_definitions_fail_closed",
            "table_stream_without_duplicates_preserves_marker_bytes",
        ]),
        PatternCheck::new("public TIFF duplicate policy and byte parity", &integration).required(
            &[
                "assert_eq!(prepared.as_bytes(), full.as_slice());",
                "identical_duplicate_jpeg_tables_are_deduplicated_under_allow_identical",
                "reject_conflicting_preserves_identical_table_redefinitions",
                "conflicting_duplicate_jpeg_tables_are_rejected",
                "prepared_jpeg_try_clone_borrows_or_copies_without_changing_bytes",
            ],
        ),
    ]);
}

#[test]
fn jpeg_sof_inspection_keeps_fixed_sampling_on_the_stack_and_owns_ids_fallibly() {
    let source = segment_source();
    assert_pattern_checks(&[
        PatternCheck::new("public JPEG SOF parser allocation", &source)
            .required(&[
                "fn parse_sof_info_at(",
                "fn parse_sof_components(",
                "let mut sampling = [(0u8, 0u8); 4];",
                "let mut ids = try_vec_with_capacity(count)?;",
                "let mut quant_table_ids = try_vec_with_capacity(count)?;",
            ])
            .forbidden(&[
                "let mut sampling = Vec::with_capacity(usize::from(component_count));",
                "let mut component_ids = Vec::with_capacity(usize::from(component_count));",
                "let mut quant_table_ids = Vec::with_capacity(usize::from(component_count));",
            ]),
    ]);
}
