// SPDX-License-Identifier: MIT OR Apache-2.0

//! Structural ownership and size ratchets for the xtask command dispatcher.

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

mod codegen;
mod lint_policy;
mod release_commands;
mod release_integrity;
mod release_status;

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

struct XtaskSources {
    main: String,
    benchmark: String,
    support: String,
    panic_surface: String,
    quality: String,
    release: String,
    semver: String,
    semver_review: String,
    semver_tests: String,
    stable_api: String,
    release_integrity_policy: String,
    release_integrity_markdown: String,
    release_integrity_html: String,
    release_integrity_policy_tests: String,
    release_integrity_changelog_tests: String,
    release_integrity_metadata_tests: String,
    release_integrity_provenance_tests: String,
    release_integrity_provenance_boundary_tests: String,
}

impl XtaskSources {
    fn read() -> Self {
        Self {
            main: read("xtask/src/main.rs"),
            benchmark: read("xtask/src/benchmark_commands.rs"),
            support: read("xtask/src/command_support.rs"),
            panic_surface: read("xtask/src/panic_surface.rs"),
            quality: read("xtask/src/quality_commands.rs"),
            release: read("xtask/src/release_commands.rs"),
            semver: read("xtask/src/semver.rs"),
            semver_review: read("xtask/src/semver/review.rs"),
            semver_tests: read("xtask/src/semver/tests.rs"),
            stable_api: read("xtask/src/stable_api.rs"),
            release_integrity_policy: read(
                "xtask/src/release_commands/release_integrity_policy.rs",
            ),
            release_integrity_markdown: read(
                "xtask/src/release_commands/release_integrity_policy/markdown.rs",
            ),
            release_integrity_html: read(
                "xtask/src/release_commands/release_integrity_policy/markdown/html.rs",
            ),
            release_integrity_policy_tests: read(
                "xtask/src/release_commands/release_integrity_policy/tests.rs",
            ),
            release_integrity_changelog_tests: read(
                "xtask/src/release_commands/release_integrity_policy/tests/changelog.rs",
            ),
            release_integrity_metadata_tests: read(
                "xtask/src/release_commands/release_integrity_policy/tests/metadata.rs",
            ),
            release_integrity_provenance_tests: read(
                "xtask/src/release_commands/release_integrity_policy/tests/provenance.rs",
            ),
            release_integrity_provenance_boundary_tests: read(
                "xtask/src/release_commands/release_integrity_policy/tests/provenance_boundaries.rs",
            ),
        }
    }
}

fn assert_modules_stay_focused(sources: &XtaskSources) {
    for (relative_path, source, max_lines) in [
        ("xtask/src/main.rs", sources.main.as_str(), 700),
        (
            "xtask/src/benchmark_commands.rs",
            sources.benchmark.as_str(),
            800,
        ),
        (
            "xtask/src/command_support.rs",
            sources.support.as_str(),
            300,
        ),
        (
            "xtask/src/panic_surface.rs",
            sources.panic_surface.as_str(),
            650,
        ),
        (
            "xtask/src/quality_commands.rs",
            sources.quality.as_str(),
            800,
        ),
        (
            "xtask/src/release_commands.rs",
            sources.release.as_str(),
            800,
        ),
        ("xtask/src/semver.rs", sources.semver.as_str(), 900),
        (
            "xtask/src/semver/review.rs",
            sources.semver_review.as_str(),
            350,
        ),
        (
            "xtask/src/semver/tests.rs",
            sources.semver_tests.as_str(),
            300,
        ),
        ("xtask/src/stable_api.rs", sources.stable_api.as_str(), 350),
        (
            "xtask/src/release_commands/release_integrity_policy.rs",
            sources.release_integrity_policy.as_str(),
            300,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/markdown.rs",
            sources.release_integrity_markdown.as_str(),
            100,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/markdown/html.rs",
            sources.release_integrity_html.as_str(),
            100,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/tests.rs",
            sources.release_integrity_policy_tests.as_str(),
            50,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/tests/changelog.rs",
            sources.release_integrity_changelog_tests.as_str(),
            175,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/tests/metadata.rs",
            sources.release_integrity_metadata_tests.as_str(),
            100,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/tests/provenance.rs",
            sources.release_integrity_provenance_tests.as_str(),
            75,
        ),
        (
            "xtask/src/release_commands/release_integrity_policy/tests/provenance_boundaries.rs",
            sources.release_integrity_provenance_boundary_tests.as_str(),
            75,
        ),
    ] {
        assert_module_stays_focused(relative_path, source, max_lines);
    }
}

fn assert_module_stays_focused(relative_path: &str, source: &str, max_lines: usize) {
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

fn assert_dispatcher_and_command_ownership(sources: &XtaskSources) {
    assert_pattern_checks(&[
        PatternCheck::new("xtask root dispatcher", &sources.main)
            .required(&[
                "mod benchmark_commands;",
                "mod clone_audit;",
                "mod codegen_commands;",
                "mod command_support;",
                "mod panic_surface;",
                "mod quality_commands;",
                "mod release_commands;",
                "mod stable_api;",
                "mod source_audit;",
                "\"clone-audit\" => clone_audit(env::args().skip(2))",
                "\"release-integrity\" => release_integrity(env::args().skip(2))",
                "fn run() -> Result<(), String>",
                "fn print_help()",
            ])
            .forbidden(&[
                "fn bench_build()",
                "fn codec_math_codegen(",
                "fn ensure_clean_worktree()",
                "fn panic_surface()",
                "fn release_integrity(",
            ]),
        PatternCheck::new("xtask benchmark command ownership", &sources.benchmark).required(&[
            "pub(super) fn bench_build(args:",
            "pub(super) fn j2k_bench_signoff()",
            "pub(super) fn bench_report(",
            "fn render_benchmark_report(",
        ]),
        PatternCheck::new("xtask stable API collection ownership", &sources.stable_api).required(
            &[
                "pub(super) fn collect_package_apis(",
                "fn collect_package_api(",
                "fn package_public_api(",
                "fn validate_public_api_environment_keys",
            ],
        ),
        PatternCheck::new("xtask semver module boundaries", &sources.semver).required(&[
            "mod review;",
            "mod tests;",
            "review::load_review_config()?",
            "review::validate_reviews(",
        ]),
        PatternCheck::new("xtask semver review ownership", &sources.semver_review).required(&[
            "pub(super) fn parse_review_config(",
            "pub(super) fn validate_reviews(",
            "hidden_fingerprint",
            "hidden_rationale",
        ]),
        PatternCheck::new("xtask process and path helper ownership", &sources.support).required(&[
            "pub(super) fn ensure_clean_worktree()",
            "pub(super) fn run_cargo(",
            "pub(super) fn run_cargo_test_with_pass_floor(",
            "pub(super) fn command_output_os(",
            "pub(super) fn command_output_os_detailed_with_env(",
            "pub(super) fn rust_sources(",
        ]),
        PatternCheck::new(
            "xtask panic-surface command ownership",
            &sources.panic_surface,
        )
        .required(&[
            "mod source_inventory;",
            "pub(super) fn panic_surface()",
            "enforce_panic_macro_inventory",
            "fn parse_panic_surface_selection(",
            "fn parse_panic_surface_output(",
        ]),
        PatternCheck::new("xtask quality command ownership", &sources.quality).required(&[
            "pub(super) fn ci()",
            "pub(super) fn clippy_strict()",
            "pub(super) fn fuzz_build()",
            "pub(super) fn verify_unsafe_audit()",
        ]),
        PatternCheck::new("xtask release command ownership", &sources.release).required(&[
            "mod release_integrity_policy;",
            "const PUBLISHABLE_PACKAGES:",
            "pub(super) const STABLE_SEMVER_PACKAGES:",
            "pub(super) fn release_integrity(",
            "pub(super) fn release_cpu()",
            "pub(super) fn package()",
        ]),
    ]);
}

#[test]
fn xtask_dispatcher_stays_split_by_command_family() {
    let sources = XtaskSources::read();
    assert_modules_stay_focused(&sources);
    assert_dispatcher_and_command_ownership(&sources);
    codegen::assert_ownership_and_focus();
    release_commands::assert_regressions_stay_focused();
    release_integrity::assert_ownership(&sources);
    release_integrity::assert_regressions(&sources);
    release_status::assert_ownership_and_focus();
}
