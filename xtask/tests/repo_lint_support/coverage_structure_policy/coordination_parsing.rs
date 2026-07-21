// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_pattern_checks, PatternCheck};
use super::read;

#[test]
fn coverage_coordination_model_and_build_outputs_stay_explicit() {
    let coordinator = read("xtask/src/coverage.rs");
    let model = read("xtask/src/coverage/model.rs");
    let build_outputs = read("xtask/src/coverage/build_outputs.rs");
    let build_output_target = read("xtask/src/coverage/build_outputs/target.rs");
    let build_output_tests = read("xtask/src/coverage/build_outputs/tests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage coordinator wiring", &coordinator)
            .required(&[
                "mod accelerator_ownership;",
                "mod build_outputs;",
                "mod compiler_regions;",
                "mod critical_path_policy;",
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
                "parse_compiler_regions(&compiler_regions, &root)?",
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
            "pub(super) critical: CoverageCounts",
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
    ]);
}

#[test]
fn coverage_lane_and_diff_parsing_ownership_stays_explicit() {
    let lane = read("xtask/src/coverage/lane.rs");
    let parsing = read("xtask/src/coverage/parsing.rs");

    assert_pattern_checks(&[
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
                "fn report_compiler_regions(",
                "fn report_compiler_regions_args(",
                "fn coverage_tool_version(",
                "fn parse_coverage_tool_version(",
                "fn package_coverage_args(",
                "fn accelerator_lane_package_args_include_every_shared_source_owner()",
                "fn lane_spec_drives_package_args_and_source_ownership()",
                "fn shared_accelerator_source_owners_drive_lane_package_selection()",
                "fn coverage_tool_version_parser_requires_named_record()",
                "fn llvm_cov_commands_share_unique_target_and_build_directory()",
                "fn lane_orchestrators_execute_complete_hermetic_cargo_plans()",
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
            "pub(super) fn normalize_coverage_path(",
        ]),
    ]);
}

#[test]
fn coverage_compiler_region_ownership_stays_explicit() {
    let compiler_regions = read("xtask/src/coverage/compiler_regions.rs");
    let compiler_region_parsing = read("xtask/src/coverage/compiler_regions/parsing.rs");
    let compiler_region_tests = read("xtask/src/coverage/compiler_regions/tests.rs");
    let compiler_region_line_tests =
        read("xtask/src/coverage/compiler_regions/tests/line_evidence.rs");
    let compiler_line_evaluation_tests =
        read("xtask/src/coverage/tests/evaluation/compiler_line_evidence.rs");

    assert_pattern_checks(&[
        PatternCheck::new("compiler region evidence ownership", &compiler_regions).required(&[
            "mod parsing;",
            "pub(super) use parsing::parse_compiler_regions;",
            "pub(super) struct CompilerRegionReport",
            "pub(super) fn evidence_for(",
            "pub(super) fn evidence_for_line(",
            "CompilerRegionEvidence::NonInstrumentable",
            "mod tests;",
        ]),
        PatternCheck::new(
            "compiler region JSON parsing ownership",
            &compiler_region_parsing,
        )
        .required(&[
            "llvm.coverage.json.export",
            "pub(in crate::coverage) fn parse_compiler_regions(",
            "must contain exactly 8 integer fields",
        ]),
        PatternCheck::new(
            "compiler region evidence regressions",
            &compiler_region_tests,
        )
        .required(&[
            "fn parser_aggregates_code_regions_by_normalized_repository_path()",
            "fn body_without_a_nested_code_region_is_compiler_noninstrumentable()",
            "fn nested_zero_count_code_region_is_uncovered()",
            "fn malformed_or_unrelated_reports_fail_closed()",
            "fn dependency_macro_expansion_regions_are_ignored_without_hiding_repository_regions()",
        ]),
        PatternCheck::new(
            "compiler line evidence regressions",
            &compiler_region_line_tests,
        )
        .required(&["fn line_evidence_uses_the_most_specific_intersecting_region()"]),
        PatternCheck::new(
            "compiler line evaluation regressions",
            &compiler_line_evaluation_tests,
        )
        .required(&[
            "fn covered_compiler_region_owns_multiline_expression_lines_without_da_records()",
            "fn zero_compiler_region_keeps_an_executable_line_without_da_uncovered()",
        ]),
    ]);
}
