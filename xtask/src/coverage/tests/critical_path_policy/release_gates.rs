// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::coverage::critical_path_policy::{
    audit_zero_body, classify_path, CriticalPathClass, ResidualDisposition, ZeroBodyAudit,
    ZeroBodyKind,
};
use crate::coverage::evaluation::coverage_violations;
use crate::coverage::model::{CoverageCounts, CoverageLane};
use crate::coverage::tests::synthetic_result;

#[test]
fn accelerator_lane_uses_raw_implementation_coverage_as_audit_evidence() {
    let mut result = synthetic_result(100, 72);
    result.critical = CoverageCounts {
        measurable: 5,
        covered: 4,
    };

    assert!(coverage_violations(CoverageLane::Metal, &result).is_empty());
    assert!(coverage_violations(CoverageLane::Cuda, &result).is_empty());

    result.critical.covered = 3;
    let violations = coverage_violations(CoverageLane::Metal, &result);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("critical-path lines"));
}

#[test]
fn host_lane_keeps_the_overall_changed_line_gate() {
    let mut result = synthetic_result(100, 72);
    result.accelerator = CoverageCounts::default();
    result.critical = CoverageCounts {
        measurable: 5,
        covered: 4,
    };

    let violations = coverage_violations(CoverageLane::Host, &result);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("changed executable Rust lines"));
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
        classify_path("crates/j2k-codec-math/src/dwt.rs"),
        Some(CriticalPathClass::Correctness)
    );
    assert_eq!(classify_path("xtask/src/perf_guard.rs"), None);
}

#[test]
fn accelerator_critical_paths_exclude_broad_compute_and_diagnostic_internals() {
    assert_eq!(
        classify_path("crates/j2k-metal/src/encode/packet_plan.rs"),
        Some(CriticalPathClass::Correctness)
    );
    assert_eq!(
        classify_path("crates/j2k-metal/src/session.rs"),
        Some(CriticalPathClass::Ownership)
    );
    assert_eq!(
        classify_path("crates/j2k-metal/src/compute/tier1_encode.rs"),
        None
    );
    assert_eq!(
        classify_path("crates/j2k-metal/src/compute/resident_tier1/counter_validation/validate.rs"),
        None
    );
    assert_eq!(
        classify_path("crates/j2k-metal/src/compute/resident_codestream/classic_tier1.rs"),
        None
    );
    assert_eq!(
        audit_zero_body(
            CoverageLane::Metal,
            ZeroBodyKind::Function,
            "crates/j2k-metal/src/compute/tier1_encode.rs::encode_codeblock@1",
        ),
        ZeroBodyAudit::Residual(ResidualDisposition::HardwareOnly)
    );
}
