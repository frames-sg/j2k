// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::coverage::critical_path_policy::{
    audit_zero_body, audited_zero_body_findings, classify_path, CriticalPathClass,
    ResidualDisposition, ZeroBodyAudit, ZeroBodyKind,
};
use crate::coverage::evaluation::coverage_violations;
use crate::coverage::model::{CoverageCounts, CoverageLane};
use crate::coverage::tests::synthetic_result;

#[test]
fn critical_path_threshold_cannot_be_masked_by_low_risk_tooling_coverage() {
    let mut result = synthetic_result(100, 99);
    result.critical = CoverageCounts {
        measurable: 5,
        covered: 3,
    };

    let violations = coverage_violations(CoverageLane::Host, &result);

    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("critical-path lines"));
}

#[test]
fn low_risk_tooling_absence_is_an_audited_residual_not_a_critical_failure() {
    let mut result = synthetic_result(5, 4);
    let finding = "xtask/src/perf_guard.rs::render_benchmark@1";
    result
        .absent_instrumentable_files
        .push("xtask/src/perf_guard.rs".to_string());
    result
        .changed_functions_without_covered_body
        .push(finding.to_string());

    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
    assert_eq!(
        audited_zero_body_findings(CoverageLane::Host, &result)[0].audit,
        ZeroBodyAudit::Residual(ResidualDisposition::LowRiskTooling)
    );
}

#[test]
fn critical_path_classification_covers_release_risk_boundaries() {
    assert_eq!(
        classify_path("crates/j2k-jpeg/src/parse/header.rs"),
        Some(CriticalPathClass::Parser)
    );
    assert_eq!(
        classify_path("crates/j2k-core/src/batch/allocation.rs"),
        Some(CriticalPathClass::Ownership)
    );
    assert_eq!(
        classify_path("crates/j2k/src/lib.rs"),
        Some(CriticalPathClass::PublicApi)
    );
    assert_eq!(
        classify_path("xtask/src/stable_api.rs"),
        Some(CriticalPathClass::PublicApi)
    );
    assert_eq!(
        classify_path("xtask/src/release_commands/release_integrity_policy.rs"),
        Some(CriticalPathClass::Security)
    );
    assert_eq!(
        classify_path("crates/j2k-native/src/unsafe_boundary.rs"),
        Some(CriticalPathClass::Safety)
    );
    assert_eq!(
        classify_path("crates/j2k-native/src/j2c/idwt.rs"),
        Some(CriticalPathClass::Correctness)
    );
    assert_eq!(classify_path("xtask/src/perf_guard.rs"), None);
}

#[test]
fn zero_body_audit_records_each_approved_residual_disposition() {
    let cases = [
        (
            CoverageLane::Host,
            ZeroBodyKind::Function,
            "crates/j2k-test-support/src/cuda.rs::cuda_strict_oxide_gate@31",
            ResidualDisposition::HardwareOnly,
        ),
        (
            CoverageLane::Host,
            ZeroBodyKind::Function,
            "crates/j2k-native/src/error.rs::fmt@389",
            ResidualDisposition::Trivial,
        ),
        (
            CoverageLane::Host,
            ZeroBodyKind::Function,
            "xtask/src/perf_guard.rs::j2k_perf_guard@170",
            ResidualDisposition::LowRiskTooling,
        ),
        (
            CoverageLane::Host,
            ZeroBodyKind::DeferredBody,
            "xtask/src/release_commands/package_gate.rs::closure@72",
            ResidualDisposition::Unreachable,
        ),
    ];

    for (lane, kind, finding, disposition) in cases {
        assert_eq!(
            audit_zero_body(lane, kind, finding),
            ZeroBodyAudit::Residual(disposition),
            "{finding}"
        );
    }
}

#[test]
fn critical_zero_body_findings_remain_in_the_audit_without_individual_failure() {
    assert_eq!(
        audit_zero_body(
            CoverageLane::Host,
            ZeroBodyKind::Function,
            "crates/j2k-jpeg/src/parse/header/progressive.rs::handle_restart_interval@167",
        ),
        ZeroBodyAudit::Critical(CriticalPathClass::Parser)
    );
}
