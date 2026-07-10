// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for the fixture-comparison harness.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

fn assert_line_budget(relative_path: &str, source: &str, max_lines: usize) {
    let line_count = source.lines().count();
    assert!(
        line_count < max_lines,
        "{relative_path} has {line_count} lines; expected fewer than {max_lines}"
    );
}

#[test]
fn fixture_compare_stays_split_by_responsibility() {
    let shell = read("crates/j2k-compare/src/fixture_compare.rs");
    let cli = read("crates/j2k-compare/src/fixture_compare/cli.rs");
    let fixtures = read("crates/j2k-compare/src/fixture_compare/fixtures.rs");
    let metadata = read("crates/j2k-compare/src/fixture_compare/metadata.rs");
    let validation = read("crates/j2k-compare/src/fixture_compare/validation.rs");
    let measurement = read("crates/j2k-compare/src/fixture_compare/measurement.rs");
    let decode = read("crates/j2k-compare/src/fixture_compare/decode.rs");
    let comparators = read("crates/j2k-compare/src/fixture_compare/comparators.rs");
    let gates = read("crates/j2k-compare/src/fixture_compare/gates.rs");
    let manifest = read("crates/j2k-compare/src/fixture_compare/manifest.rs");
    let rows = read("crates/j2k-compare/src/fixture_compare/rows.rs");
    let types = read("crates/j2k-compare/src/fixture_compare/types.rs");

    for (path, source, max_lines) in [
        ("fixture_compare.rs", shell.as_str(), 250),
        ("fixture_compare/cli.rs", cli.as_str(), 200),
        ("fixture_compare/fixtures.rs", fixtures.as_str(), 525),
        ("fixture_compare/metadata.rs", metadata.as_str(), 500),
        ("fixture_compare/validation.rs", validation.as_str(), 250),
        ("fixture_compare/measurement.rs", measurement.as_str(), 250),
        ("fixture_compare/decode.rs", decode.as_str(), 650),
        ("fixture_compare/comparators.rs", comparators.as_str(), 350),
        ("fixture_compare/gates.rs", gates.as_str(), 400),
        ("fixture_compare/manifest.rs", manifest.as_str(), 325),
        ("fixture_compare/rows.rs", rows.as_str(), 350),
        ("fixture_compare/types.rs", types.as_str(), 400),
    ] {
        assert_line_budget(path, source, max_lines);
        assert!(
            !source.contains("use super::*"),
            "crates/j2k-compare/src/{path} must keep explicit module imports"
        );
        assert!(
            !source.contains("include!("),
            "crates/j2k-compare/src/{path} must remain a real Rust module"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("fixture_compare coordinator", &shell)
            .required(&[
                "mod cli;",
                "mod fixtures;",
                "mod metadata;",
                "mod validation;",
                "mod measurement;",
                "mod decode;",
                "fn run() -> Result<(), String>",
            ])
            .forbidden(&[
                "struct FixtureCase",
                "fn batch_size_config_from_env(",
                "fn all_fixture_cases(",
                "fn emit_metadata(",
                "fn validate_cases(",
                "fn measure_case_batch_rows(",
                "fn decode_j2k_batch(",
            ]),
        PatternCheck::new("fixture_compare CLI ownership", &cli).required(&[
            "pub(super) fn benchmark_mode_from_env",
            "pub(super) fn batch_size_config_from_env",
            "pub(super) fn validate_comparator_gates",
            "pub(super) fn filter_cases_for_mode",
        ]),
        PatternCheck::new("fixture_compare fixture ownership", &fixtures).required(&[
            "pub(super) fn all_fixture_cases",
            "pub(super) fn fixture_cases",
            "pub(super) fn load_external_fixture_cases",
            "pub(super) fn encode_lossless",
        ]),
        PatternCheck::new("fixture_compare metadata ownership", &metadata).required(&[
            "pub(super) fn emit_metadata",
            "pub(super) fn external_unique_input_count",
            "pub(super) fn skipped_comparators_label",
            "pub(super) fn resolved_workers_label",
        ]),
        PatternCheck::new("fixture_compare validation ownership", &validation).required(&[
            "pub(super) fn validate_cases",
            "pub(super) fn validate_mixed_batches",
            "pub(super) fn skip_reason",
            "pub(super) fn mixed_skip_reason",
        ]),
        PatternCheck::new("fixture_compare measurement ownership", &measurement).required(&[
            "pub(super) fn measure_case_batch_rows",
            "pub(super) fn measure_mixed_batch_rows",
        ]),
        PatternCheck::new("fixture_compare decode ownership", &decode).required(&[
            "pub(super) fn decode_j2k_batch",
            "pub(super) fn decode_external_once",
            "pub(super) fn decode_method_label",
            "pub(super) fn crop_interleaved",
        ]),
        PatternCheck::new("fixture_compare type ownership", &types).required(&[
            "pub(super) struct FixtureCase",
            "pub(super) struct Measurement",
            "pub(super) struct MixedFixtureBatch",
            "pub(super) struct MetadataContext",
        ]),
    ]);
}
