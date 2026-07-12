// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the adoption benchmark module map is a single fail-closed ownership policy"
)]
fn adoption_benchmark_lives_in_focused_modules() {
    let root = repo_root();
    let coordinator = fs::read_to_string(root.join("xtask/src/adoption_benchmark.rs"))
        .expect("read adoption benchmark coordinator");
    let options = fs::read_to_string(root.join("xtask/src/adoption_benchmark/options.rs"))
        .expect("read adoption benchmark options");
    let runner = fs::read_to_string(root.join("xtask/src/adoption_benchmark/runner.rs"))
        .expect("read adoption benchmark runner");
    let existing = fs::read_to_string(root.join("xtask/src/adoption_benchmark/existing.rs"))
        .expect("read adoption benchmark existing-result discovery");
    let existing_tests =
        fs::read_to_string(root.join("xtask/src/adoption_benchmark/existing/tests.rs"))
            .expect("read adoption benchmark existing-result tests");
    let parsing = fs::read_to_string(root.join("xtask/src/adoption_benchmark/parsing.rs"))
        .expect("read adoption benchmark parsing");
    let parsing_tests =
        fs::read_to_string(root.join("xtask/src/adoption_benchmark/parsing/tests.rs"))
            .expect("read adoption benchmark parser tests");
    let summary = fs::read_to_string(root.join("xtask/src/adoption_benchmark/summary.rs"))
        .expect("read adoption benchmark summary");
    let readme = fs::read_to_string(root.join("xtask/src/adoption_benchmark/readme.rs"))
        .expect("read adoption benchmark README renderer");
    let support = fs::read_to_string(root.join("xtask/src/adoption_benchmark/support.rs"))
        .expect("read adoption benchmark publication/path support");

    for (path, source, max_lines) in [
        ("adoption_benchmark.rs", coordinator.as_str(), 600),
        ("adoption_benchmark/options.rs", options.as_str(), 250),
        ("adoption_benchmark/runner.rs", runner.as_str(), 700),
        ("adoption_benchmark/existing.rs", existing.as_str(), 200),
        (
            "adoption_benchmark/existing/tests.rs",
            existing_tests.as_str(),
            260,
        ),
        ("adoption_benchmark/parsing.rs", parsing.as_str(), 700),
        (
            "adoption_benchmark/parsing/tests.rs",
            parsing_tests.as_str(),
            360,
        ),
        ("adoption_benchmark/summary.rs", summary.as_str(), 300),
        ("adoption_benchmark/readme.rs", readme.as_str(), 300),
        ("adoption_benchmark/support.rs", support.as_str(), 150),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "xtask/src/{path} must stay below its focused-module line-count ratchet of {max_lines}"
        );
        assert!(
            !source.contains("use super::*") && !source.contains("include!("),
            "xtask/src/{path} must keep explicit real-Rust module boundaries"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("adoption benchmark coordinator wiring", &coordinator).required(&[
            "mod existing;",
            "mod options;",
            "mod parsing;",
            "mod readme;",
            "mod runner;",
            "mod summary;",
            "mod support;",
            "pub(crate) fn adoption_benchmark(",
        ]),
        PatternCheck::new(
            "adoption benchmark coordinator ownership exclusions",
            &coordinator,
        )
        .forbidden(&[
            "const SCRUBBED_BENCH_ENV_VARS",
            "struct AdoptionStep",
            "fn run_cpu_encode_compare(",
            "fn existing_steps(",
            "fn write_summary(",
            "fn criterion_summary_json(",
            "fn write_readme(",
            "fn enforce_publication_gate(",
            "impl AdoptionBenchmarkOptions",
        ]),
        PatternCheck::new("adoption benchmark option ownership", &options).required(&[
            "pub(crate) struct AdoptionBenchmarkOptions",
            "pub(super) fn parse(",
            "pub(super) fn help_text(",
            "pub(super) fn parse_batch_size_list(",
        ]),
        PatternCheck::new("adoption benchmark runner ownership", &runner).required(&[
            "pub(super) const SCRUBBED_BENCH_ENV_VARS",
            "pub(super) fn run_cpu_encode_compare(",
            "pub(super) fn run_cuda_htj2k_decode(",
            "pub(super) fn run_metal_transcode_benchmark(",
            "pub(super) fn run_logged_owned(",
            "pub(super) fn display_command(",
        ]),
        PatternCheck::new("adoption benchmark existing-result ownership", &existing).required(&[
            "mod tests;",
            "pub(super) fn existing_steps(",
            "pub(super) fn existing_ran_step(",
        ]),
        PatternCheck::new(
            "adoption benchmark existing-result regressions",
            &existing_tests,
        )
        .required(&[
            "fn existing_step_requires_nonempty_stdout_and_a_regular_stderr_file()",
            "fn existing_step_preserves_artifact_and_criterion_paths()",
            "fn existing_steps_marks_unrequested_accelerators_skipped()",
            "fn existing_steps_reuses_requested_accelerator_artifacts()",
            "fn existing_steps_propagates_the_exact_missing_artifact()",
        ]),
        PatternCheck::new("adoption benchmark parser ownership", &parsing).required(&[
            "mod tests;",
            "pub(super) fn criterion_summary_json(",
            "pub(super) fn read_metal_decode_summary(",
            "pub(super) fn read_metal_encode_summary(",
            "pub(super) fn read_metal_transcode_summary(",
            "pub(super) fn read_tsv_metadata(",
        ]),
        PatternCheck::new("adoption benchmark parser regressions", &parsing_tests).required(&[
            "fn criterion_summary_reports_only_completed_benchmark_roots()",
            "fn criterion_estimate_json_preserves_confidence_interval()",
            "fn metal_decode_summary_reports_skips_and_unreadable_output()",
            "fn metal_encode_summary_reconciles_all_row_kinds_and_metadata()",
            "fn metal_encode_summary_reports_skips_and_unreadable_output()",
            "fn metal_transcode_summary_fails_closed_for_each_missing_stream()",
            "fn row_parsers_preserve_optional_suffixes_and_reject_wrong_profiles()",
            "fn primitive_parsers_reject_invalid_boolean_and_empty_metadata()",
        ]),
        PatternCheck::new("adoption benchmark summary/model ownership", &summary).required(&[
            "pub(super) struct AdoptionStep",
            "pub(super) enum StepStatus",
            "pub(super) fn write_summary(",
            "pub(super) fn step_json(",
        ]),
        PatternCheck::new("adoption benchmark README ownership", &readme)
            .required(&["pub(super) fn write_readme("]),
        PatternCheck::new("adoption benchmark publication/path ownership", &support).required(&[
            "pub(super) fn enforce_publication_gate(",
            "pub(super) fn benchmark_env_path(",
            "pub(super) fn benchmark_env_path_list(",
            "pub(super) fn canonical_benchmark_path(",
        ]),
    ]);
}
