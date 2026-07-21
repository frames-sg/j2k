// SPDX-License-Identifier: MIT OR Apache-2.0

use super::read;

#[test]
fn coverage_structure_policy_has_focused_real_module_owners() {
    let shell_relative = "xtask/tests/repo_lint_support/coverage_structure_policy.rs";
    let shell = read(shell_relative);
    let include_macro = ["include", "!("].concat();

    for (module, owned_symbols, max_lines) in [
        (
            "line_ratchets",
            &[
                "fn coverage_structure_policy_has_focused_real_module_owners(",
                "fn coverage_source_modules_stay_within_structural_ratchets(",
            ][..],
            300usize,
        ),
        (
            "coordination_parsing",
            &[
                "fn coverage_coordination_model_and_build_outputs_stay_explicit(",
                "fn coverage_lane_and_diff_parsing_ownership_stays_explicit(",
                "fn coverage_compiler_region_ownership_stays_explicit(",
            ][..],
            300usize,
        ),
        (
            "evaluation_exclusion",
            &[
                "fn coverage_evaluation_and_reporting_ownership_stays_explicit(",
                "fn coverage_critical_path_policy_ownership_stays_explicit(",
                "fn coverage_exclusion_policy_ownership_stays_explicit(",
            ][..],
            250usize,
        ),
        (
            "source_analysis_tests",
            &[
                "fn coverage_source_analysis_ast_ownership_stays_explicit(",
                "fn coverage_source_analysis_boundaries_stay_explicit(",
                "fn coverage_workspace_discovery_stays_explicit(",
                "fn coverage_source_analysis_regression_ownership_stays_explicit(",
                "fn coverage_source_roles_and_evaluation_regressions_stay_explicit(",
            ][..],
            350usize,
        ),
    ] {
        assert!(
            shell.contains(&format!("mod {module};")),
            "{shell_relative} must declare the {module} responsibility owner"
        );
        let relative =
            format!("xtask/tests/repo_lint_support/coverage_structure_policy/{module}.rs");
        let source = read(&relative);
        for owned_symbol in owned_symbols {
            let expected_count = usize::from(module == "line_ratchets") + 1;
            assert_eq!(
                source.matches(owned_symbol).count(),
                expected_count,
                "{relative} must own {owned_symbol} exactly once"
            );
        }
        assert!(
            source.lines().count() < max_lines,
            "{relative} exceeded its {max_lines}-line responsibility ratchet"
        );
        assert!(
            !source.lines().any(|line| line.trim() == "use super::*;"),
            "{relative} must keep explicit imports"
        );
        assert!(
            !source.contains(&include_macro),
            "{relative} must remain a real Rust module"
        );
    }

    assert!(
        shell.lines().count() < 20,
        "{shell_relative} must remain a tiny shared-read module shell"
    );
    assert!(!shell.contains("#[test]"));
    assert!(!shell.contains("PatternCheck::new"));
    assert!(!shell.contains(&include_macro));
    assert!(!shell.lines().any(|line| line.trim() == "use super::*;"));
}

#[test]
fn coverage_source_modules_stay_within_structural_ratchets() {
    let include_macro = ["include", "!("].concat();
    for (relative_path, max_lines) in [
        ("xtask/src/coverage.rs", 300),
        ("xtask/src/coverage/accelerator_ownership.rs", 250),
        ("xtask/src/coverage/build_outputs.rs", 350),
        ("xtask/src/coverage/build_outputs/target.rs", 150),
        ("xtask/src/coverage/build_outputs/tests.rs", 200),
        ("xtask/src/coverage/compiler_regions.rs", 200),
        ("xtask/src/coverage/compiler_regions/parsing.rs", 250),
        ("xtask/src/coverage/compiler_regions/tests.rs", 180),
        (
            "xtask/src/coverage/compiler_regions/tests/line_evidence.rs",
            100,
        ),
        ("xtask/src/coverage/critical_path_policy.rs", 350),
        (
            "xtask/src/coverage/critical_path_policy/classification.rs",
            175,
        ),
        ("xtask/src/coverage/model.rs", 600),
        ("xtask/src/coverage/lane.rs", 600),
        ("xtask/src/coverage/parsing.rs", 600),
        ("xtask/src/coverage/evaluation.rs", 600),
        ("xtask/src/coverage/summary.rs", 600),
        ("xtask/src/coverage/exclusion_policy.rs", 600),
        (
            "xtask/src/coverage/exclusion_policy/evidence_modules.rs",
            150,
        ),
        ("xtask/src/coverage/exclusion_policy/tests.rs", 180),
        ("xtask/src/coverage/source_analysis.rs", 300),
        ("xtask/src/coverage/source_analysis/audit.rs", 100),
        ("xtask/src/coverage/source_analysis/ast.rs", 300),
        ("xtask/src/coverage/source_analysis/ast/executable.rs", 180),
        ("xtask/src/coverage/source_analysis/ast/visitor.rs", 300),
        (
            "xtask/src/coverage/source_analysis/ast/visitor/items.rs",
            300,
        ),
        (
            "xtask/src/coverage/source_analysis/ast/visitor/runtime.rs",
            200,
        ),
        ("xtask/src/coverage/source_analysis/cfg_eval.rs", 350),
        ("xtask/src/coverage/source_analysis/graph.rs", 200),
        ("xtask/src/coverage/source_analysis/module_resolver.rs", 180),
        ("xtask/src/coverage/source_analysis/node_attrs.rs", 200),
        (
            "xtask/src/coverage/source_analysis/test_constructors.rs",
            150,
        ),
        ("xtask/src/coverage/source_analysis/test_lines.rs", 150),
        ("xtask/src/coverage/source_analysis/ast/test_spans.rs", 100),
        ("xtask/src/coverage/source_analysis/workspace.rs", 500),
        (
            "xtask/src/coverage/source_analysis/workspace/fuzz_manifests.rs",
            250,
        ),
        ("xtask/src/coverage/tests.rs", 250),
        ("xtask/src/coverage/tests/attributes.rs", 200),
        ("xtask/src/coverage/tests/cfg_provenance.rs", 100),
        ("xtask/src/coverage/tests/critical_path_policy.rs", 125),
        (
            "xtask/src/coverage/tests/critical_path_policy/release_gates.rs",
            150,
        ),
        ("xtask/src/coverage/tests/deferred_bodies.rs", 200),
        ("xtask/src/coverage/tests/evaluation.rs", 250),
        ("xtask/src/coverage/tests/evaluation/non_executable.rs", 100),
        (
            "xtask/src/coverage/tests/evaluation/compiler_line_evidence.rs",
            100,
        ),
        ("xtask/src/coverage/tests/executable_evidence.rs", 180),
        ("xtask/src/coverage/tests/presence.rs", 150),
        ("xtask/src/coverage/tests/source_analysis.rs", 250),
        ("xtask/src/coverage/tests/source_roles.rs", 250),
        ("xtask/src/coverage/tests/support.rs", 100),
    ] {
        let source = read(relative_path);
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
            !source.contains(&include_macro),
            "{relative_path} must remain a real Rust module"
        );
        assert!(
            !source.contains("#[allow("),
            "{relative_path} must not add lint suppressions"
        );
    }
}
