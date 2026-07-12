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
    let accelerator_ownership = read("xtask/src/coverage/accelerator_ownership.rs");
    let build_outputs = read("xtask/src/coverage/build_outputs.rs");
    let build_output_target = read("xtask/src/coverage/build_outputs/target.rs");
    let build_output_tests = read("xtask/src/coverage/build_outputs/tests.rs");
    let model = read("xtask/src/coverage/model.rs");
    let lane = read("xtask/src/coverage/lane.rs");
    let parsing = read("xtask/src/coverage/parsing.rs");
    let evaluation = read("xtask/src/coverage/evaluation.rs");
    let summary = read("xtask/src/coverage/summary.rs");
    let exclusions = read("xtask/src/coverage/exclusion_policy.rs");
    let exclusion_tests = read("xtask/src/coverage/exclusion_policy/tests.rs");
    let source_analysis = read("xtask/src/coverage/source_analysis.rs");
    let source_audit = read("xtask/src/coverage/source_analysis/audit.rs");
    let source_ast = read("xtask/src/coverage/source_analysis/ast.rs");
    let source_ast_executable = read("xtask/src/coverage/source_analysis/ast/executable.rs");
    let source_ast_visitor = read("xtask/src/coverage/source_analysis/ast/visitor.rs");
    let source_ast_visitor_items = read("xtask/src/coverage/source_analysis/ast/visitor/items.rs");
    let source_ast_visitor_runtime =
        read("xtask/src/coverage/source_analysis/ast/visitor/runtime.rs");
    let source_ast_visitors = [
        source_ast_visitor.as_str(),
        source_ast_visitor_items.as_str(),
        source_ast_visitor_runtime.as_str(),
    ]
    .join("\n");
    let source_cfg = read("xtask/src/coverage/source_analysis/cfg_eval.rs");
    let source_graph = read("xtask/src/coverage/source_analysis/graph.rs");
    let module_resolver = read("xtask/src/coverage/source_analysis/module_resolver.rs");
    let node_attrs = read("xtask/src/coverage/source_analysis/node_attrs.rs");
    let source_test_constructors = read("xtask/src/coverage/source_analysis/test_constructors.rs");
    let test_lines = read("xtask/src/coverage/source_analysis/test_lines.rs");
    let test_spans = read("xtask/src/coverage/source_analysis/ast/test_spans.rs");
    let source_workspace = read("xtask/src/coverage/source_analysis/workspace.rs");
    let fuzz_manifests = read("xtask/src/coverage/source_analysis/workspace/fuzz_manifests.rs");
    let tests = read("xtask/src/coverage/tests.rs");
    let attribute_tests = read("xtask/src/coverage/tests/attributes.rs");
    let cfg_provenance_tests = read("xtask/src/coverage/tests/cfg_provenance.rs");
    let deferred_body_tests = read("xtask/src/coverage/tests/deferred_bodies.rs");
    let evaluation_tests = read("xtask/src/coverage/tests/evaluation.rs");
    let executable_evidence_tests = read("xtask/src/coverage/tests/executable_evidence.rs");
    let presence_tests = read("xtask/src/coverage/tests/presence.rs");
    let source_tests = read("xtask/src/coverage/tests/source_analysis.rs");
    let source_role_tests = read("xtask/src/coverage/tests/source_roles.rs");
    let test_support = read("xtask/src/coverage/tests/support.rs");

    for (relative_path, source, max_lines) in [
        ("xtask/src/coverage.rs", coordinator.as_str(), 300),
        (
            "xtask/src/coverage/accelerator_ownership.rs",
            accelerator_ownership.as_str(),
            250,
        ),
        (
            "xtask/src/coverage/build_outputs.rs",
            build_outputs.as_str(),
            350,
        ),
        (
            "xtask/src/coverage/build_outputs/target.rs",
            build_output_target.as_str(),
            150,
        ),
        (
            "xtask/src/coverage/build_outputs/tests.rs",
            build_output_tests.as_str(),
            200,
        ),
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
        (
            "xtask/src/coverage/exclusion_policy/tests.rs",
            exclusion_tests.as_str(),
            180,
        ),
        (
            "xtask/src/coverage/source_analysis.rs",
            source_analysis.as_str(),
            300,
        ),
        (
            "xtask/src/coverage/source_analysis/audit.rs",
            source_audit.as_str(),
            100,
        ),
        (
            "xtask/src/coverage/source_analysis/ast.rs",
            source_ast.as_str(),
            300,
        ),
        (
            "xtask/src/coverage/source_analysis/ast/executable.rs",
            source_ast_executable.as_str(),
            180,
        ),
        (
            "xtask/src/coverage/source_analysis/ast/visitor.rs",
            source_ast_visitor.as_str(),
            300,
        ),
        (
            "xtask/src/coverage/source_analysis/ast/visitor/items.rs",
            source_ast_visitor_items.as_str(),
            300,
        ),
        (
            "xtask/src/coverage/source_analysis/ast/visitor/runtime.rs",
            source_ast_visitor_runtime.as_str(),
            200,
        ),
        (
            "xtask/src/coverage/source_analysis/cfg_eval.rs",
            source_cfg.as_str(),
            350,
        ),
        (
            "xtask/src/coverage/source_analysis/graph.rs",
            source_graph.as_str(),
            200,
        ),
        (
            "xtask/src/coverage/source_analysis/module_resolver.rs",
            module_resolver.as_str(),
            180,
        ),
        (
            "xtask/src/coverage/source_analysis/node_attrs.rs",
            node_attrs.as_str(),
            200,
        ),
        (
            "xtask/src/coverage/source_analysis/test_constructors.rs",
            source_test_constructors.as_str(),
            150,
        ),
        (
            "xtask/src/coverage/source_analysis/test_lines.rs",
            test_lines.as_str(),
            150,
        ),
        (
            "xtask/src/coverage/source_analysis/ast/test_spans.rs",
            test_spans.as_str(),
            100,
        ),
        (
            "xtask/src/coverage/source_analysis/workspace.rs",
            source_workspace.as_str(),
            500,
        ),
        (
            "xtask/src/coverage/source_analysis/workspace/fuzz_manifests.rs",
            fuzz_manifests.as_str(),
            250,
        ),
        ("xtask/src/coverage/tests.rs", tests.as_str(), 250),
        (
            "xtask/src/coverage/tests/attributes.rs",
            attribute_tests.as_str(),
            200,
        ),
        (
            "xtask/src/coverage/tests/cfg_provenance.rs",
            cfg_provenance_tests.as_str(),
            100,
        ),
        (
            "xtask/src/coverage/tests/deferred_bodies.rs",
            deferred_body_tests.as_str(),
            200,
        ),
        (
            "xtask/src/coverage/tests/evaluation.rs",
            evaluation_tests.as_str(),
            250,
        ),
        (
            "xtask/src/coverage/tests/executable_evidence.rs",
            executable_evidence_tests.as_str(),
            180,
        ),
        (
            "xtask/src/coverage/tests/presence.rs",
            presence_tests.as_str(),
            150,
        ),
        (
            "xtask/src/coverage/tests/source_analysis.rs",
            source_tests.as_str(),
            250,
        ),
        (
            "xtask/src/coverage/tests/source_roles.rs",
            source_role_tests.as_str(),
            250,
        ),
        (
            "xtask/src/coverage/tests/support.rs",
            test_support.as_str(),
            100,
        ),
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
                "mod accelerator_ownership;",
                "mod build_outputs;",
                "mod evaluation;",
                "mod exclusion_policy;",
                "mod lane;",
                "mod model;",
                "mod parsing;",
                "mod source_analysis;",
                "mod summary;",
                "pub(crate) fn coverage(",
                "ensure_no_untracked_rust_sources()?;",
                "validate_shared_accelerator_registry(&root)?;",
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
            "struct AcceleratorLaneSpec",
            "struct AcceleratorPackageSpec",
            "const METAL_ACCELERATOR_LANE",
            "const CUDA_ACCELERATOR_LANE",
            "pub(super) enum CoverageLane",
            "pub(super) fn coverage_packages(",
            "pub(super) struct CoverageOptions",
            "pub(super) struct ChangedCoverageResult",
            "pub(super) fn parse_options(",
        ]),
        PatternCheck::new("coverage build-output cfg ownership", &build_outputs).required(&[
            "mod target;",
            "pub(super) use target::CurrentBuildTarget;",
            "pub(super) struct BuildOutputEvidence",
            "pub(super) fn capture(",
            "pub(super) fn current_cfg_flags(",
            "fn scan_outputs(",
            "fn reconcile_cfg_flags(",
            "fn parse_build_cfg_output(",
        ]),
        PatternCheck::new("current coverage build target", &build_output_target).required(&[
            "pub(in crate::coverage) struct CurrentBuildTarget",
            "pub(in crate::coverage) fn create(",
            "CARGO_LLVM_COV_TARGET_DIR",
            "CARGO_TARGET_DIR",
            ".j2k-current-coverage-",
            "fs::create_dir(",
            "fs::remove_dir_all(",
        ]),
        PatternCheck::new("coverage build-output regressions", &build_output_tests).required(&[
            "fn identical_rerun_output_is_current_build_evidence()",
            "fn stale_scope_output_is_outside_current_build_provenance()",
            "fn missing_selected_package_build_output_fails_closed()",
            "fn conflicting_current_scopes_fail_closed()",
            "fn hyphenated_package_name_matches_its_full_build_scope()",
        ]),
        PatternCheck::new("coverage lane execution ownership", &lane)
            .required(&[
                "const METAL_COVERAGE_ENV",
                "const CUDA_COVERAGE_ENV",
                "const REQUIRED_CARGO_LLVM_COV_VERSION: &str = \"0.8.7\"",
                "pub(super) fn run_lane(",
                "CurrentBuildTarget::create(root)",
                "BuildOutputEvidence::capture(current_build_target)",
                "CARGO_LLVM_COV_TARGET_DIR",
                "CARGO_LLVM_COV_BUILD_DIR",
                "fn run_host_coverage(",
                "fn run_metal_coverage(",
                "fn run_cuda_coverage(",
                "fn coverage_tool_version(",
                "fn parse_coverage_tool_version(",
                "fn package_coverage_args(",
                "fn accelerator_lane_package_args_include_every_shared_source_owner()",
                "fn lane_spec_drives_package_args_and_source_ownership()",
                "fn shared_accelerator_source_owners_drive_lane_package_selection()",
                "fn coverage_tool_version_parser_requires_named_record()",
                "fn llvm_cov_commands_share_unique_target_and_build_directory()",
                "--include-build-script",
                "fn run_llvm_cov(",
            ])
            .forbidden(&["\"llvm-cov\", \"clean\""]),
        PatternCheck::new("coverage diff and LCOV parser ownership", &parsing).required(&[
            "pub(super) fn ensure_no_untracked_rust_sources()",
            "pub(super) fn validate_no_untracked_rust_sources(",
            "pub(super) fn resolve_diff_base(",
            "pub(super) fn git_output(",
            "pub(super) fn parse_changed_lines(",
            "pub(super) fn parse_lcov(",
            "fn normalize_lcov_path(",
        ]),
        PatternCheck::new("coverage changed-line evaluation ownership", &evaluation)
            .required(&[
                "pub(super) fn evaluate_changed_coverage(",
                "struct ChangedFileEvidence<'a>",
                "fn evaluate_changed_lines(",
                "fn record_missing_body_evidence(",
                "self.body_is_covered(function.body_start, function.body_end)",
                "changed_functions_without_covered_body",
                "changed_executable_bodies_without_covered_body",
                "changed_deferred_bodies_without_distinct_line_evidence",
                "mixed_test_production_lines",
                "changed_opaque_macros",
                "source_dispositions",
                "pub(super) fn coverage_violations(",
                "fn meets_threshold(",
            ])
            .forbidden(&[
                "fn terminal_test_module_start(",
                "fn source_has_instrumentable_function(",
            ]),
        PatternCheck::new("coverage summary ownership", &summary).required(&[
            "pub(super) fn write_summary(",
            "pub(super) fn print_summary(",
            "j2k-changed-line-coverage-v2",
            "cargo_llvm_cov_version",
            "residual_unmeasured_lines",
            "changed_functions_without_covered_body",
            "changed_executable_bodies_without_covered_body",
            "changed_deferred_bodies_without_distinct_line_evidence",
            "mixed_test_production_lines",
            "changed_opaque_macros",
            "accelerator_host_rust",
            "narrow_exclusions",
        ]),
        PatternCheck::new("coverage exclusion policy ownership", &exclusions).required(&[
            "pub(super) const COVERAGE_EXCLUSIONS",
            "enum EvidenceClass",
            "fn require_primary_evidence(",
            "fn enclosing_cfg_is_conditional(",
            "cuda-simt-device-rust",
            "metal-embedded-shader-body",
            "generated-codec-math-fragment",
            "vendored-block-ffi-binding",
            "pub(super) fn matching_exclusion(",
            "pub(super) fn validate_exclusion_policy(",
            "fn collect_rust_files(",
        ]),
        PatternCheck::new("coverage exclusion evidence regressions", &exclusion_tests).required(&[
            "fn direct_and_inherited_cfg_require_supplemental_classification()",
            "fn exact_enclosing_cfg_test_is_harness_plumbing()",
            "fn supplemental_only_exclusion_evidence_is_rejected()",
        ]),
        PatternCheck::new("coverage source-analysis facade", &source_analysis).required(&[
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
        PatternCheck::new("coverage production-audit facade", &source_audit).required(&[
            "pub(crate) struct SourceAuditTestSpan",
            "pub(crate) struct SourceAuditSyntax",
            "pub(crate) fn analyze_test_only_syntax(",
            "CoverageCfgContext::for_current_target",
            "ReachKind::Production",
        ]),
        PatternCheck::new("coverage syn AST analysis", &source_ast).required(&[
            "syn::parse_file(source)",
            "mod executable;",
            "mod visitor;",
            "struct AstCollector",
            "fn visit_attributed_node(",
        ]),
        PatternCheck::new("coverage executable-span collector", &source_ast_executable).required(
            &[
                "fn record_executable_span(",
                "fn record_closure(",
                "fn record_opaque_macro(",
                "fn visit_executable_node(",
            ],
        ),
        PatternCheck::new("coverage syn AST visitor", &source_ast_visitors).required(&[
            "impl<'ast> Visit<'ast> for AstCollector<'_>",
            "mod items;",
            "mod runtime;",
            "function.block.span()",
            "function.default",
            "Expr::Closure(closure)",
            "Expr::Macro(expression_macro)",
            "Item::Verbatim(_)",
            "unclassified cfg/test attribute",
            "fn visit_fn_arg(",
            "fn visit_pat(",
        ]),
        PatternCheck::new("coverage cfg evaluation", &source_cfg).required(&[
            "pub(super) struct CoverageCfgContext",
            "enabled_features",
            "custom_flags: Option<BTreeMap<String, bool>>",
            "SymbolicTruth::Unknown",
            "conservatively_active",
            "target_feature",
            "structural cfg_attr",
        ]),
        PatternCheck::new("coverage module graph", &source_graph).required(&[
            "pub(super) enum ReachKind",
            "pub(super) struct ReachState",
            "pub(super) fn module_reachability(",
        ]),
        PatternCheck::new("coverage module path boundary", &module_resolver).required(&[
            "pub(super) fn resolve_external_module(",
            "fs::canonicalize(root)",
            "resolves outside repository root",
            "more than one path attribute",
        ]),
        PatternCheck::new("coverage test constructors", &source_test_constructors).required(&[
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
        PatternCheck::new("coverage workspace discovery", &source_workspace)
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
        PatternCheck::new("coverage regression tests", &tests).required(&[
            "mod attributes;",
            "mod cfg_provenance;",
            "mod deferred_bodies;",
            "mod executable_evidence;",
            "fn parses_added_diff_hunks_without_counting_deletions()",
            "fn untracked_rust_sources_fail_the_local_coverage_preflight()",
            "fn lcov_parser_merges_duplicate_line_records_by_max_count()",
            "fn eighty_percent_changed_line_coverage_passes_exactly()",
            "fn exclusion_policy_maps_every_narrow_rule_to_existing_tests()",
            "fn coverage_cli_defaults_to_host_and_accepts_explicit_lanes()",
        ]),
        PatternCheck::new(
            "coverage attribute-disposition regressions",
            &attribute_tests,
        )
        .required(&[
            "fn cfg_test_module_does_not_hide_later_production_items()",
            "fn cfg_test_attributes_on_fields_locals_arms_and_expressions_are_test_only()",
            "fn cfg_test_function_parameters_are_test_only_without_hiding_patterns()",
        ]),
        PatternCheck::new("coverage cfg provenance regressions", &cfg_provenance_tests)
            .required(&["fn cfg_active_changed_source_cannot_evade_coverage_gate()"]),
        PatternCheck::new("coverage source-analysis regressions", &source_tests).required(&[
            "fn body_bearing_function_forms_have_item_and_body_spans()",
            "fn nested_inline_module_uses_its_real_module_directory()",
            "fn nonterminal_external_module_in_named_crate_root_uses_sibling_source()",
            "fn module_path_cannot_escape_the_repository_root()",
            "fn unknown_custom_cfg_is_conservatively_required()",
            "fn unknown_cfg_in_either_polarity_is_conservatively_required()",
        ]),
        PatternCheck::new("coverage source-role regressions", &source_role_tests).required(&[
            "fn nonterminal_external_test_modules_do_not_truncate_production_files()",
            "crates/j2k-cuda-runtime/src/lib.rs",
            "crates/j2k-jpeg/src/backend/mod.rs",
            "crates/j2k-native/src/j2c/encode/single_tile.rs",
            "fn cfg_test_helper_trees_are_not_production_source()",
            "crates/j2k-cuda-runtime/src/context/test_kernels.rs",
            "fn unreachable_role_named_directories_fail_closed()",
            "crate/src/tests/orphan.rs",
            "crate/src/examples/orphan.rs",
            "crate/src/benches/orphan.rs",
            "crate/src/fuzz/orphan.rs",
            "fn cargo_target_roots_retain_metadata_roles()",
            "fn cargo_fuzz_manifest_only_grants_reachable_targets_the_fuzz_role()",
        ]),
        PatternCheck::new("coverage evaluation regressions", &evaluation_tests).required(&[
            "fn changed_signature_requires_a_positive_da_record_in_the_function_body()",
            "fn changed_function_without_covered_body_is_a_host_violation()",
            "fn residual_unmeasured_lines_remain_explicit()",
            "fn registered_shared_accelerator_sources_reach_both_gpu_denominators()",
            "fn generated_and_vendored_sources_have_reviewed_dispositions()",
        ]),
        PatternCheck::new(
            "coverage executable-evidence regressions",
            &executable_evidence_tests,
        )
        .required(&[
            "fn changed_uncalled_closure_requires_coverage_in_its_own_body()",
            "fn changed_opaque_macro_definition_and_invocation_fail_closed()",
            "fn cfg_test_macro_remains_test_only()",
        ]),
        PatternCheck::new("coverage presence regressions", &presence_tests).required(&[
            "fn partial_file_lcov_does_not_mask_second_changed_function_without_covered_body()",
            "fn shared_accelerator_source_absent_from_metal_lcov_is_a_violation()",
            "fn zero_count_body_record_does_not_prove_changed_signature_coverage()",
            "fn changed_executable_body_line_without_da_is_uncovered()",
        ]),
    ]);
}
