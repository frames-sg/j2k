// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for the fixture-comparison harness.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

mod decode;
mod fixtures;
mod manifest;

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

fn assert_module_boundary(relative_path: &str, source: &str, max_lines: usize) {
    assert_line_budget(relative_path, source, max_lines);
    assert!(
        !source.contains("use super::*"),
        "crates/j2k-compare/src/{relative_path} must keep explicit module imports"
    );
    assert!(
        !source.contains("include!("),
        "crates/j2k-compare/src/{relative_path} must remain a real Rust module"
    );
}

fn assert_comparator_contracts(comparators: &str, comparator_tests: &str) {
    assert_pattern_checks(&[
        PatternCheck::new("fixture comparator cleanup ownership", comparators).required(&[
            "fn cleanup_cli_staging(",
            "let input_cleanup = cleanup_cli_temp(",
            "let output_cleanup = cleanup_cli_temp(",
            "staged input cleanup failed:",
            "staged output cleanup failed:",
        ]),
        PatternCheck::new("fixture comparator command regressions", comparator_tests).required(&[
            "fn staging_cleanup_reports_both_failures_after_attempting_both_paths()",
            "fn comparator_commands_decode_and_report_process_errors()",
            "fn comparator_cleanup_attempts_output_after_input_failure()",
            "output cleanup must still run after input cleanup fails",
            "thread_local!",
        ]),
    ]);
}

#[test]
fn fixture_compare_stays_split_by_responsibility() {
    let shell = read("crates/j2k-compare/src/fixture_compare.rs");
    let cli = read("crates/j2k-compare/src/fixture_compare/cli.rs");
    let metadata = read("crates/j2k-compare/src/fixture_compare/metadata.rs");
    let validation = read("crates/j2k-compare/src/fixture_compare/validation.rs");
    let measurement = read("crates/j2k-compare/src/fixture_compare/measurement.rs");
    let comparators = read("crates/j2k-compare/src/fixture_compare/comparators.rs");
    let comparator_tests = read("crates/j2k-compare/src/fixture_compare/comparators/tests.rs");
    let gates = read("crates/j2k-compare/src/fixture_compare/gates.rs");
    let rows = read("crates/j2k-compare/src/fixture_compare/rows.rs");
    let types = read("crates/j2k-compare/src/fixture_compare/types.rs");

    for (path, source, max_lines) in [
        ("fixture_compare.rs", shell.as_str(), 250),
        ("fixture_compare/cli.rs", cli.as_str(), 200),
        ("fixture_compare/metadata.rs", metadata.as_str(), 500),
        ("fixture_compare/validation.rs", validation.as_str(), 250),
        ("fixture_compare/measurement.rs", measurement.as_str(), 250),
        ("fixture_compare/comparators.rs", comparators.as_str(), 350),
        (
            "fixture_compare/comparators/tests.rs",
            comparator_tests.as_str(),
            350,
        ),
        ("fixture_compare/gates.rs", gates.as_str(), 400),
        ("fixture_compare/rows.rs", rows.as_str(), 350),
        ("fixture_compare/types.rs", types.as_str(), 400),
    ] {
        assert_module_boundary(path, source, max_lines);
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
        PatternCheck::new("fixture_compare type ownership", &types).required(&[
            "pub(super) struct FixtureCase",
            "pub(super) struct Measurement",
            "pub(super) struct MixedFixtureBatch",
            "pub(super) struct MetadataContext",
        ]),
    ]);
    assert_comparator_contracts(&comparators, &comparator_tests);
    decode::assert_contract();
    fixtures::assert_contract();
    manifest::assert_contract();
}
