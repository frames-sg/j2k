// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for changed-line coverage tooling.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the coverage-module ownership and size checks form one cohesive structural policy"
)]
fn coverage_tooling_stays_split_by_responsibility() {
    let coordinator = read("xtask/src/coverage.rs");
    let model = read("xtask/src/coverage/model.rs");
    let lane = read("xtask/src/coverage/lane.rs");
    let parsing = read("xtask/src/coverage/parsing.rs");
    let evaluation = read("xtask/src/coverage/evaluation.rs");
    let summary = read("xtask/src/coverage/summary.rs");
    let exclusions = read("xtask/src/coverage/exclusion_policy.rs");
    let tests = read("xtask/src/coverage/tests.rs");

    for (relative_path, source, max_lines) in [
        ("xtask/src/coverage.rs", coordinator.as_str(), 300),
        ("xtask/src/coverage/model.rs", model.as_str(), 600),
        ("xtask/src/coverage/lane.rs", lane.as_str(), 600),
        ("xtask/src/coverage/parsing.rs", parsing.as_str(), 600),
        ("xtask/src/coverage/evaluation.rs", evaluation.as_str(), 600),
        ("xtask/src/coverage/summary.rs", summary.as_str(), 600),
        (
            "xtask/src/coverage/exclusion_policy.rs",
            exclusions.as_str(),
            600,
        ),
        ("xtask/src/coverage/tests.rs", tests.as_str(), 250),
    ] {
        let line_count = source.lines().count();
        assert!(
            line_count < max_lines,
            "{relative_path} has {line_count} lines; expected fewer than {max_lines}"
        );
        assert!(
            !source.contains("::*"),
            "{relative_path} must keep explicit imports"
        );
        assert!(
            !source.contains("include!("),
            "{relative_path} must remain a real Rust module"
        );
        assert!(
            !source.contains("#[allow("),
            "{relative_path} must not add lint suppressions"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("coverage coordinator wiring", &coordinator)
            .required(&[
                "mod evaluation;",
                "mod exclusion_policy;",
                "mod lane;",
                "mod model;",
                "mod parsing;",
                "mod summary;",
                "pub(crate) fn coverage(",
            ])
            .forbidden(&[
                "enum CoverageLane",
                "fn run_lane(",
                "fn parse_lcov(",
                "fn evaluate_changed_coverage(",
                "fn write_summary(",
                "const COVERAGE_EXCLUSIONS",
            ]),
        PatternCheck::new("coverage model and option ownership", &model).required(&[
            "pub(super) const CHANGED_LINE_THRESHOLD_PERCENT: u64 = 80",
            "pub(super) enum CoverageLane",
            "pub(super) struct CoverageOptions",
            "pub(super) struct ChangedCoverageResult",
            "pub(super) fn parse_options(",
        ]),
        PatternCheck::new("coverage lane execution ownership", &lane).required(&[
            "const METAL_COVERAGE_ENV",
            "const CUDA_COVERAGE_ENV",
            "pub(super) fn run_lane(",
            "fn run_host_coverage(",
            "fn run_metal_coverage(",
            "fn run_cuda_coverage(",
            "fn run_llvm_cov(",
        ]),
        PatternCheck::new("coverage diff and LCOV parser ownership", &parsing).required(&[
            "pub(super) fn resolve_diff_base(",
            "pub(super) fn git_output(",
            "pub(super) fn parse_changed_lines(",
            "pub(super) fn parse_lcov(",
            "fn normalize_lcov_path(",
        ]),
        PatternCheck::new("coverage changed-line evaluation ownership", &evaluation).required(&[
            "pub(super) fn evaluate_changed_coverage(",
            "fn terminal_test_module_start(",
            "fn source_has_instrumentable_function(",
            "pub(super) fn coverage_violations(",
            "fn meets_threshold(",
        ]),
        PatternCheck::new("coverage summary ownership", &summary).required(&[
            "pub(super) fn write_summary(",
            "pub(super) fn print_summary(",
            "j2k-changed-line-coverage-v1",
            "accelerator_host_rust",
            "narrow_exclusions",
        ]),
        PatternCheck::new("coverage exclusion policy ownership", &exclusions).required(&[
            "pub(super) const COVERAGE_EXCLUSIONS",
            "cuda-simt-device-rust",
            "metal-embedded-shader-body",
            "pub(super) fn matching_exclusion(",
            "pub(super) fn validate_exclusion_policy(",
            "fn collect_rust_files(",
        ]),
        PatternCheck::new("coverage regression tests", &tests).required(&[
            "fn parses_added_diff_hunks_without_counting_deletions()",
            "fn lcov_parser_merges_duplicate_line_records_by_max_count()",
            "fn eighty_percent_changed_line_coverage_passes_exactly()",
            "fn exclusion_policy_maps_every_narrow_rule_to_existing_tests()",
            "fn coverage_cli_defaults_to_host_and_accepts_explicit_lanes()",
        ]),
    ]);
}
