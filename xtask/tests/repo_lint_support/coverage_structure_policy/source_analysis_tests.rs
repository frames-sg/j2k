// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_pattern_checks, PatternCheck};
use super::read;

#[test]
fn coverage_source_analysis_ast_ownership_stays_explicit() {
    let facade = read("xtask/src/coverage/source_analysis.rs");
    let audit = read("xtask/src/coverage/source_analysis/audit.rs");
    let ast = read("xtask/src/coverage/source_analysis/ast.rs");
    let executable = read("xtask/src/coverage/source_analysis/ast/executable.rs");
    let visitor = read("xtask/src/coverage/source_analysis/ast/visitor.rs");
    let visitor_items = read("xtask/src/coverage/source_analysis/ast/visitor/items.rs");
    let visitor_runtime = read("xtask/src/coverage/source_analysis/ast/visitor/runtime.rs");
    let visitors = [
        visitor.as_str(),
        visitor_items.as_str(),
        visitor_runtime.as_str(),
    ]
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new("coverage source-analysis facade", &facade).required(&[
            "mod ast;",
            "mod audit;",
            "mod cfg_eval;",
            "mod graph;",
            "mod module_resolver;",
            "mod node_attrs;",
            "mod test_constructors;",
            "mod workspace;",
            "pub(super) struct FunctionSpan",
            "pub(super) struct ExecutableBodySpan",
            "pub(super) struct OpaqueMacroSpan",
            "pub(super) executable_lines: BTreeSet<usize>",
            "pub(super) body_start: usize",
            "pub(super) struct SourceIndex",
        ]),
        PatternCheck::new("coverage production-audit facade", &audit).required(&[
            "pub(crate) struct SourceAuditTestSpan",
            "pub(crate) struct SourceAuditSyntax",
            "pub(crate) fn analyze_test_only_syntax(",
            "CoverageCfgContext::for_current_target",
            "ReachKind::Production",
        ]),
        PatternCheck::new("coverage syn AST analysis", &ast).required(&[
            "syn::parse_file(source)",
            "mod executable;",
            "mod visitor;",
            "struct AstCollector",
            "fn visit_attributed_node(",
            "self.executable_lines.insert(body_start)",
        ]),
        PatternCheck::new("coverage executable-span collector", &executable).required(&[
            "fn record_executable_span(",
            "fn record_closure(",
            "fn record_opaque_macro(",
            "fn visit_executable_node(",
        ]),
        PatternCheck::new("coverage syn AST visitor", &visitors).required(&[
            "impl<'ast> Visit<'ast> for AstCollector<'_>",
            "mod items;",
            "mod runtime;",
            "function.block.span()",
            "function.default",
            "Expr::Closure(closure)",
            "Expr::Macro(expression_macro)",
            "Item::Verbatim(_)",
            "unclassified cfg/test attribute",
            "Attribute payloads are compile-time metadata",
            "fn visit_fn_arg(",
            "fn visit_pat(",
        ]),
    ]);
}

#[test]
fn coverage_source_analysis_boundaries_stay_explicit() {
    let cfg = read("xtask/src/coverage/source_analysis/cfg_eval.rs");
    let graph = read("xtask/src/coverage/source_analysis/graph.rs");
    let resolver = read("xtask/src/coverage/source_analysis/module_resolver.rs");
    let constructors = read("xtask/src/coverage/source_analysis/test_constructors.rs");
    let node_attrs = read("xtask/src/coverage/source_analysis/node_attrs.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage cfg evaluation", &cfg).required(&[
            "pub(super) struct CoverageCfgContext",
            "enabled_features",
            "custom_flags: Option<BTreeMap<String, bool>>",
            "SymbolicTruth::Unknown",
            "conservatively_active",
            "target_feature",
            "structural cfg_attr",
        ]),
        PatternCheck::new("coverage module graph", &graph).required(&[
            "pub(super) enum ReachKind",
            "pub(super) struct ReachState",
            "pub(super) fn module_reachability(",
        ]),
        PatternCheck::new("coverage module path boundary", &resolver).required(&[
            "pub(in crate::coverage) fn resolve_external_module(",
            "fs::canonicalize(root)",
            "resolves outside repository root",
            "more than one path attribute",
        ]),
        PatternCheck::new("coverage test constructors", &constructors).required(&[
            "impl SourceIndex",
            "pub(in crate::coverage) fn single(",
            "pub(in crate::coverage) fn repository_subset(",
            "pub(in crate::coverage) fn repository_manifest_fuzz_subset(",
        ]),
        PatternCheck::new("coverage attribute-bearing node accessors", &node_attrs).required(&[
            "pub(super) fn expression(",
            "Expr::Struct(value)",
            "pub(super) fn foreign_item(",
            "pub(super) fn function_argument(",
            "pub(super) fn generic_parameter(",
            "pub(super) fn pattern(",
            "unclassified non-exhaustive expression variant",
        ]),
    ]);
}

#[test]
fn coverage_workspace_discovery_stays_explicit() {
    let workspace = read("xtask/src/coverage/source_analysis/workspace.rs");
    let fuzz_manifests = read("xtask/src/coverage/source_analysis/workspace/fuzz_manifests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage workspace discovery", &workspace)
            .required(&[
                "mod fuzz_manifests;",
                "discover_source_roots(",
                "cargo metadata",
                "fuzz_manifests::discover(",
                "capture_cfg_contexts(",
                "has_build_script",
                "current_cfg_flags(&selected_packages, &build_script_packages)",
                "GENERATED_DWT_DISPOSITION",
                "VENDORED_BLOCK_DISPOSITION",
            ])
            .forbidden(&[
                "fresh_cfg_flags",
                "unwrap_or_default()",
                "fn is_test_target(",
                "fn is_example_bench_or_fuzz(",
                "path.starts_with(\"tests/\")",
                "path.starts_with(\"examples/\")",
                "path.starts_with(\"benches/\")",
                "path.starts_with(\"fuzz/\")",
                "path.contains(\"/tests/\")",
                "path.contains(\"/examples/\")",
                "path.contains(\"/benches/\")",
                "path.contains(\"/fuzz/\")",
            ]),
        PatternCheck::new("manifest-backed cargo-fuzz roots", &fuzz_manifests).required(&[
            "pub(super) fn discover(",
            "fn candidate_manifests(",
            "fn parse_manifest(",
            "metadata.get(\"cargo-fuzz\")",
            "must explicitly declare [[bin]] targets",
            "ReachKind::ExampleBenchFuzz",
        ]),
    ]);
}

#[test]
fn coverage_source_analysis_regression_ownership_stays_explicit() {
    let tests = read("xtask/src/coverage/tests.rs");
    let attributes = read("xtask/src/coverage/tests/attributes.rs");
    let cfg_provenance = read("xtask/src/coverage/tests/cfg_provenance.rs");
    let exclusion_policy = read("xtask/src/coverage/tests/exclusion_policy.rs");
    let source = read("xtask/src/coverage/tests/source_analysis.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage regression tests", &tests).required(&[
            "mod attributes;",
            "mod cfg_provenance;",
            "mod deferred_bodies;",
            "mod exclusion_policy;",
            "mod executable_evidence;",
            "fn parses_added_diff_hunks_without_counting_deletions()",
            "fn untracked_rust_sources_fail_the_local_coverage_preflight()",
            "fn lcov_parser_merges_duplicate_line_records_by_max_count()",
            "fn eighty_percent_changed_line_coverage_passes_exactly()",
            "fn coverage_cli_defaults_to_host_and_accepts_explicit_lanes()",
        ]),
        PatternCheck::new("coverage attribute-disposition regressions", &attributes).required(&[
            "fn cfg_test_module_does_not_hide_later_production_items()",
            "fn cfg_test_attributes_on_fields_locals_arms_and_expressions_are_test_only()",
            "fn cfg_test_function_parameters_are_test_only_without_hiding_patterns()",
        ]),
        PatternCheck::new("coverage cfg provenance regressions", &cfg_provenance)
            .required(&["fn cfg_active_changed_source_cannot_evade_coverage_gate()"]),
        PatternCheck::new("coverage exclusion regressions", &exclusion_policy).required(&[
            "fn exclusion_policy_maps_every_narrow_rule_to_existing_tests()",
            "fn vendored_gpu_interop_exclusion_covers_only_pinned_patch_roots()",
        ]),
        PatternCheck::new("coverage source-analysis regressions", &source).required(&[
            "fn body_bearing_function_forms_have_item_and_body_spans()",
            "fn nested_inline_module_uses_its_real_module_directory()",
            "fn nonterminal_external_module_in_named_crate_root_uses_sibling_source()",
            "fn module_path_cannot_escape_the_repository_root()",
            "fn unknown_custom_cfg_is_conservatively_required()",
            "fn unknown_cfg_in_either_polarity_is_conservatively_required()",
        ]),
    ]);
}

#[test]
fn coverage_source_roles_and_evaluation_regressions_stay_explicit() {
    let roles = read("xtask/src/coverage/tests/source_roles.rs");
    let evaluation = read("xtask/src/coverage/tests/evaluation.rs");
    let non_executable = read("xtask/src/coverage/tests/evaluation/non_executable.rs");
    let executable = read("xtask/src/coverage/tests/executable_evidence.rs");
    let deferred = read("xtask/src/coverage/tests/deferred_bodies.rs");
    let presence = read("xtask/src/coverage/tests/presence.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage source-role regressions", &roles).required(&[
            "fn nonterminal_external_test_modules_do_not_truncate_production_files()",
            "crates/j2k-cuda-runtime/src/lib.rs",
            "crates/j2k-jpeg/src/backend/mod.rs",
            "crates/j2k-native/src/j2c/encode/single_tile.rs",
            "fn cfg_test_helper_trees_are_not_production_source()",
            "crates/j2k-cuda-runtime/src/context/test_kernels.rs",
            "fn repository_owned_xtask_rust_fixtures_are_explicitly_test_only()",
            "xtask/tests/fixtures/clone_audit/production_clone_b.rs",
            "fn unreachable_role_named_directories_fail_closed()",
            "crate/src/tests/orphan.rs",
            "crate/src/examples/orphan.rs",
            "crate/src/benches/orphan.rs",
            "crate/src/fuzz/orphan.rs",
            "xtask/tests/fixtures/other/orphan.rs",
            "xtask/tests/fixtures/clone_audit/nested/orphan.rs",
            "fn cargo_target_roots_retain_metadata_roles()",
            "fn cargo_fuzz_manifest_only_grants_reachable_targets_the_fuzz_role()",
        ]),
        PatternCheck::new("coverage evaluation regressions", &evaluation).required(&[
            "mod non_executable;",
            "fn changed_signature_requires_a_positive_da_record_in_the_function_body()",
            "fn changed_function_without_covered_body_is_a_critical_audit_finding()",
            "fn registered_shared_accelerator_sources_reach_both_gpu_denominators()",
            "fn generated_and_vendored_sources_have_reviewed_dispositions()",
        ]),
        PatternCheck::new("coverage non-executable-line regressions", &non_executable).required(&[
            "fn residual_unmeasured_lines_remain_explicit()",
            "fn compiler_mapped_documentation_is_not_an_executable_changed_line()",
        ]),
        PatternCheck::new("coverage executable-evidence regressions", &executable).required(&[
            "fn changed_uncalled_closure_requires_coverage_in_its_own_body()",
            "fn changed_opaque_macro_definition_and_invocation_are_audited_without_blanket_failure()",
            "fn cfg_test_macro_remains_test_only()",
        ]),
        PatternCheck::new("coverage deferred-body regressions", &deferred).required(&[
            "fn executed_one_line_closure_accepts_its_own_compiler_region()",
            "fn unpolled_one_line_async_records_its_zero_count_compiler_region()",
            "fn body_without_a_compiler_region_is_recorded_as_noninstrumentable()",
        ]),
        PatternCheck::new("coverage presence regressions", &presence).required(&[
            "fn partial_file_lcov_does_not_mask_second_changed_function_without_covered_body()",
            "fn shared_accelerator_source_absent_from_metal_lcov_is_a_violation()",
            "fn zero_count_body_record_does_not_prove_changed_signature_coverage()",
            "fn changed_executable_body_line_without_da_is_uncovered()",
        ]),
    ]);
}
