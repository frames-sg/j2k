// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{assert_pattern_checks, PatternCheck};
use super::read;

#[test]
fn coverage_evaluation_and_reporting_ownership_stays_explicit() {
    let evaluation = read("xtask/src/coverage/evaluation.rs");
    let summary = read("xtask/src/coverage/summary.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage changed-line evaluation ownership", &evaluation)
            .required(&[
                "pub(super) fn evaluate_changed_coverage(",
                "struct ChangedFileEvidence<'a>",
                "fn evaluate_changed_lines(",
                "evidence_for_line(self.path, line_number)",
                "fn record_missing_body_evidence(",
                "self.body_is_covered(function.body_start, function.body_end)",
                "changed_functions_without_covered_body",
                "changed_executable_bodies_without_covered_body",
                "changed_deferred_bodies_without_covered_compiler_region",
                "compiler_noninstrumentable_deferred_bodies",
                "compiler_noninstrumentable_lines",
                "mixed_test_production_lines",
                "changed_opaque_macros",
                "source_dispositions",
                "pub(super) fn coverage_violations(",
                "audited_zero_body_findings(lane, result)",
                "lane.enforces_overall_changed_lines()",
                "changed critical-path lines",
                "critical executable bodies are absent",
                "fn meets_threshold(",
            ])
            .forbidden(&[
                "fn terminal_test_module_start(",
                "fn source_has_instrumentable_function(",
            ]),
        PatternCheck::new("coverage summary ownership", &summary).required(&[
            "pub(super) fn write_summary(",
            "pub(super) fn print_summary(",
            "j2k-changed-line-coverage-v6",
            "head_sha",
            "lane_scope",
            "cargo_llvm_cov_version",
            "residual_unmeasured_lines",
            "changed_functions_without_covered_body",
            "changed_executable_bodies_without_covered_body",
            "changed_deferred_bodies_without_covered_compiler_region",
            "compiler_noninstrumentable_deferred_bodies",
            "compiler_noninstrumentable_lines",
            "compiler_regions_artifact",
            "mixed_test_production_lines",
            "changed_opaque_macros",
            "critical_paths",
            "zero_body_audit",
            "accelerator_host_rust",
            "narrow_exclusions",
        ]),
    ]);
}

#[test]
fn coverage_critical_path_policy_ownership_stays_explicit() {
    let policy = read("xtask/src/coverage/critical_path_policy.rs");
    let classification = read("xtask/src/coverage/critical_path_policy/classification.rs");
    let policy_modules = [policy.as_str(), classification.as_str()].join("\n");
    let tests = read("xtask/src/coverage/tests/critical_path_policy.rs");
    let release_gate_tests = read("xtask/src/coverage/tests/critical_path_policy/release_gates.rs");
    let test_modules = [tests.as_str(), release_gate_tests.as_str()].join("\n");

    assert_pattern_checks(&[
        PatternCheck::new(
            "coverage critical-path and residual ownership",
            &policy_modules,
        )
        .required(&[
            "pub(super) enum CriticalPathClass",
            "pub(super) enum ResidualDisposition",
            "pub(super) enum ZeroBodyAudit",
            "pub(in crate::coverage) fn classify_path(",
            "pub(super) fn audited_zero_body_findings(",
            "Self::Unreachable => \"unreachable\"",
            "Self::HardwareOnly => \"hardware-only\"",
            "Self::Trivial => \"trivial\"",
            "Self::LowRiskTooling => \"low-risk-tooling\"",
        ]),
        PatternCheck::new("coverage critical-path policy regressions", &test_modules).required(&[
            "fn critical_path_threshold_cannot_be_masked_by_low_risk_tooling_coverage()",
            "fn accelerator_lane_uses_raw_implementation_coverage_as_audit_evidence()",
            "fn host_lane_keeps_the_overall_changed_line_gate()",
            "fn low_risk_tooling_absence_is_an_audited_residual_not_a_critical_failure()",
            "fn critical_path_classification_covers_release_risk_boundaries()",
            "fn accelerator_critical_paths_exclude_broad_compute_and_diagnostic_internals()",
            "fn zero_body_audit_records_each_approved_residual_disposition()",
            "fn critical_zero_body_findings_remain_in_the_audit_without_individual_failure()",
        ]),
    ]);
}

#[test]
fn coverage_exclusion_policy_ownership_stays_explicit() {
    let exclusions = read("xtask/src/coverage/exclusion_policy.rs");
    let evidence_modules = read("xtask/src/coverage/exclusion_policy/evidence_modules.rs");
    let tests = read("xtask/src/coverage/exclusion_policy/tests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("coverage exclusion policy ownership", &exclusions)
            .required(&[
                "mod evidence_modules;",
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
            ])
            .forbidden(&[
                "fn resolve_external_module_path(",
                "fn explicit_module_path(",
            ]),
        PatternCheck::new(
            "coverage exclusion external evidence traversal",
            &evidence_modules,
        )
        .required(&[
            "fn collect_evidence_symbols_from_file(",
            "resolve_external_module(",
            "existing_repository_source(",
        ])
        .forbidden(&[
            "fn resolve_external_module_path(",
            "fn explicit_module_path(",
        ]),
        PatternCheck::new("coverage exclusion evidence regressions", &tests).required(&[
            "fn direct_and_inherited_cfg_require_supplemental_classification()",
            "fn exact_enclosing_cfg_test_is_harness_plumbing()",
            "fn supplemental_only_exclusion_evidence_is_rejected()",
        ]),
    ]);
}
