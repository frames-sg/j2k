// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ownership and fail-closed ratchets for production-source quality audits.

use std::fs;

use super::{assert_pattern_checks, repo_root, workflow_job, PatternCheck};

fn read(relative_path: &str) -> String {
    fs::read_to_string(repo_root().join(relative_path))
        .unwrap_or_else(|error| panic!("read {relative_path}: {error}"))
}

#[test]
fn audit_modules_stay_focused_and_reuse_the_cfg_analyzer() {
    let sources = [
        ("xtask/src/clone_audit.rs", 150),
        ("xtask/src/clone_audit/config.rs", 180),
        ("xtask/src/clone_audit/report.rs", 80),
        ("xtask/src/clone_audit/stage.rs", 120),
        ("xtask/src/clone_audit/test_stage.rs", 120),
        ("xtask/src/source_audit.rs", 50),
        ("xtask/src/source_audit/mask.rs", 225),
        ("xtask/src/source_audit/panic_macros.rs", 160),
        ("xtask/src/source_audit/paths.rs", 200),
        ("xtask/src/coverage/source_analysis/audit.rs", 100),
        ("xtask/src/panic_surface/source_inventory.rs", 250),
    ];
    let combined = sources
        .iter()
        .map(|(path, max_lines)| {
            let source = read(path);
            assert!(
                source.lines().count() < *max_lines,
                "{path} must stay below {max_lines} lines"
            );
            assert!(!source.contains("::*"), "{path} must use explicit imports");
            assert!(
                !source.contains("include!("),
                "{path} must be a real module"
            );
            assert!(
                !source.contains("#[allow("),
                "{path} must not suppress lints"
            );
            source
        })
        .collect::<Vec<_>>()
        .join("\n");
    let coverage = read("xtask/src/coverage.rs");

    assert_pattern_checks(&[
        PatternCheck::new(
            "shared source-audit shell",
            &read("xtask/src/source_audit.rs"),
        )
        .required(&["mod mask;", "mod panic_macros;", "mod paths;"]),
        PatternCheck::new(
            "private coverage analyzer with narrow audit facade",
            &coverage,
        )
        .required(&[
            "mod source_analysis;",
            "analyze_test_only_syntax, SourceAuditSyntax, SourceAuditTestSpan",
        ])
        .forbidden(&["pub(crate) mod source_analysis;"]),
        PatternCheck::new("audit implementations reuse source analysis", &combined)
            .required(&["analyze_test_only_syntax", "mask_test_only_syntax"])
            .forbidden(&["syn::parse_file", "regex::", "Regex::new"]),
    ]);
}

#[test]
fn clone_audit_is_source_aware_pinned_and_ci_required() {
    let command = read("xtask/src/clone_audit.rs");
    let config_policy = read("xtask/src/clone_audit/config.rs");
    let report = read("xtask/src/clone_audit/report.rs");
    let stage = read("xtask/src/clone_audit/stage.rs");
    let test_stage = read("xtask/src/clone_audit/test_stage.rs");
    let tests = read("xtask/src/source_audit/tests.rs");
    let config = read(".jscpd.json");
    let test_config = read(".jscpd-tests.json");
    let workflow = read(".github/workflows/ci.yml");
    let clone_job = workflow_job(&workflow, "clone-audit");
    let aggregate = workflow_job(&workflow, "release-candidate");
    let evidence = read("engineering/clone-audit.md");

    assert_pattern_checks(&[
        PatternCheck::new("pinned clone-audit command", &command).required(&[
            "const JSCPD_PACKAGE: &str = \"jscpd@4.0.5\";",
            "validate_clone_config(&config_path)?",
            "stage_production_sources(&root, &stage_root)?",
            "stage_test_sources(&root, &test_stage_root)?",
            "validate_jscpd_report(",
            "DUPLICATED_LINE_THRESHOLD",
            "TEST_DUPLICATED_LINE_THRESHOLD",
        ]),
        PatternCheck::new("source-preserving production stage", &stage).required(&[
            "production_rust_sources(repository_root",
            "mask_test_only_syntax(repository_root",
            "stage_root.join(&source_path.relative)",
        ]),
        PatternCheck::new("source-aware test stage", &test_stage).required(&[
            "auditable_rust_sources(repository_root",
            "retain_test_only_syntax(repository_root",
            "is_production_rust_path(&source_path.relative)",
            "stage_root.join(&source_path.relative)",
        ]),
        PatternCheck::new("clone config ratchet", &config_policy).normalized_required(&[
            "const DUPLICATED_LINE_THRESHOLD: f64 = 2.01",
            "const TEST_DUPLICATED_LINE_THRESHOLD: f64 = 4.14",
            "require_number(&config, \"threshold\", threshold)",
            "require_number(&config, \"minLines\", 20.0)",
            "require_number(&config, \"maxLines\", 20_000.0)",
            "require_exact_keys(",
            "require_string_array(&config, \"reporters\", &[\"console\", \"json\"])",
            "require_string_array(&config, \"ignore\", ignore)",
        ]),
        PatternCheck::new("independent clone report validation", &report).required(&[
            "require_count(total, \"lines\")",
            "require_percentage(total, \"percentage\")",
            "duplicated_lines > lines",
            "percentage >= threshold",
        ]),
        PatternCheck::new("source-aware clone regression fixtures", &tests).required(&[
            "inline_cfg_test_clones_do_not_reach_production_clone_counts",
            "production_clones_remain_visible_after_source_aware_masking",
            "masking_preserves_source_byte_and_line_positions",
        ]),
        PatternCheck::new("repository jscpd config", &config).required(&[
            "\"threshold\": 2.01",
            "\"output\": \"target/clone-audit/report\"",
            "\"gitignore\": false",
        ]),
        PatternCheck::new("repository test jscpd config", &test_config).required(&[
            "\"threshold\": 4.14",
            "\"output\": \"target/clone-audit/test-report\"",
            "\"ignore\": []",
        ]),
        PatternCheck::new("CI clone audit", clone_job).required(&[
            "cargo xtask clone-audit",
            "j2k-clone-audits",
            "target/clone-audit/report/jscpd-report.json",
            "target/clone-audit/test-report/jscpd-report.json",
            "if-no-files-found: error",
        ]),
        PatternCheck::new("required clone gate", aggregate).required(&["clone-audit"]),
        PatternCheck::new("canonical clone evidence", &evidence).required(&[
            "cargo xtask clone-audit",
            "source-aware",
            "Production baseline: 1.96%",
            "Test/support baseline: 4.09%",
        ]),
    ]);
}

#[test]
fn panic_surface_tracks_every_explicit_panic_macro_family() {
    let panic_surface = read("xtask/src/panic_surface.rs");
    let inventory = read("xtask/src/panic_surface/source_inventory.rs");
    let scanner = read("xtask/src/source_audit/panic_macros.rs");

    assert_pattern_checks(&[
        PatternCheck::new("panic-surface source inventory", &panic_surface).required(&[
            "mod source_inventory;",
            "enforce_panic_macro_inventory",
            "explicit production macros: {macro_inventory}",
        ]),
        PatternCheck::new("reviewed panic-macro ratchets", &inventory).required(&[
            "const PANIC_MACRO_BASELINE",
            "panic: 0",
            "unreachable: 50",
            "assert: 8",
            "assert_eq: 3",
            "debug_assert: 91",
            "debug_assert_eq: 66",
            "mask_test_only_syntax",
            "macro_ratchet_violations",
            "format_exceeded_sites",
            "site.path",
            "site.line",
            "site.column",
        ]),
        PatternCheck::new("token-aware panic family scanner", &scanner).required(&[
            "struct PanicMacroSite",
            "\"panic\" =>",
            "\"unreachable\" =>",
            "\"assert\" =>",
            "\"assert_eq\" =>",
            "\"assert_ne\" =>",
            "\"debug_assert\" =>",
            "\"debug_assert_eq\" =>",
            "\"debug_assert_ne\" =>",
        ]),
    ]);
}
