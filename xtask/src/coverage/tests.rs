// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::evaluation::coverage_violations;
use super::exclusion_policy::{
    matching_exclusion, validate_exclusion_policy, ExclusionMatcher, COVERAGE_EXCLUSIONS,
};
use super::model::{parse_options, ChangedCoverageResult, CoverageCounts, CoverageLane};
use super::parsing::{parse_changed_lines, parse_lcov, validate_no_untracked_rust_sources};
use super::source_analysis::SourceRole;

mod attributes;
mod cfg_provenance;
mod critical_path_policy;
mod deferred_bodies;
mod evaluation;
mod executable_evidence;
mod presence;
mod source_analysis;
mod source_roles;
mod support;

pub(super) fn synthetic_result(measurable: usize, covered: usize) -> ChangedCoverageResult {
    ChangedCoverageResult {
        overall: CoverageCounts {
            measurable,
            covered,
        },
        critical: CoverageCounts {
            measurable,
            covered,
        },
        accelerator: CoverageCounts {
            measurable,
            covered,
        },
        changed_files: BTreeSet::new(),
        uncovered: Vec::new(),
        unmeasured: Vec::new(),
        exclusions: BTreeMap::new(),
        source_dispositions: BTreeMap::new(),
        absent_instrumentable_files: Vec::new(),
        changed_functions_without_covered_body: Vec::new(),
        changed_executable_bodies_without_covered_body: Vec::new(),
        changed_deferred_bodies_without_covered_compiler_region: Vec::new(),
        compiler_noninstrumentable_deferred_bodies: Vec::new(),
        compiler_noninstrumentable_lines: Vec::new(),
        mixed_test_production_lines: Vec::new(),
        changed_opaque_macros: Vec::new(),
    }
}

#[test]
fn parses_added_diff_hunks_without_counting_deletions() {
    let diff = "\
diff --git a/crates/a/src/lib.rs b/crates/a/src/lib.rs
--- a/crates/a/src/lib.rs
+++ b/crates/a/src/lib.rs
@@ -2,0 +3,2 @@
+first
+second
@@ -8 +10 @@
-old
+new
";

    let changed = parse_changed_lines(diff).unwrap();

    assert_eq!(changed["crates/a/src/lib.rs"], BTreeSet::from([3, 4, 10]));
}

#[test]
fn untracked_rust_sources_fail_the_local_coverage_preflight() {
    assert!(validate_no_untracked_rust_sources("").is_ok());
    let error = validate_no_untracked_rust_sources(
        "crates/example/src/new_module.rs\nxtask/src/new_gate.rs\n",
    )
    .unwrap_err();

    assert!(error.contains("cannot classify untracked Rust sources"));
    assert!(error.contains("crates/example/src/new_module.rs"));
    assert!(error.contains("xtask/src/new_gate.rs"));
}

#[test]
fn lcov_parser_merges_duplicate_line_records_by_max_count() {
    let root = Path::new("/repo");
    let lcov = "\
SF:/repo/crates/a/src/lib.rs
DA:3,0
DA:4,2
end_of_record
SF:/repo/crates/a/src/lib.rs
DA:3,1
end_of_record
";

    let report = parse_lcov(lcov, root).unwrap();

    assert_eq!(report.lines["crates/a/src/lib.rs"][&3], 1);
    assert_eq!(report.lines["crates/a/src/lib.rs"][&4], 2);
}

#[test]
fn eighty_percent_changed_line_coverage_passes_exactly() {
    let result = synthetic_result(5, 4);
    assert!(coverage_violations(CoverageLane::Cuda, &result).is_empty());
}

#[test]
fn metal_line_percentages_are_audited_without_overriding_hardware_parity() {
    let result = synthetic_result(100, 72);
    assert!(coverage_violations(CoverageLane::Metal, &result).is_empty());
}

#[test]
fn accelerator_threshold_cannot_be_masked_by_cpu_coverage() {
    let mut result = synthetic_result(100, 99);
    result.accelerator = CoverageCounts {
        measurable: 5,
        covered: 3,
    };

    let violations = coverage_violations(CoverageLane::Host, &result);

    assert_eq!(violations.len(), 1);
    assert!(violations[0].contains("accelerator host lines"));
}

#[test]
fn coverage_lanes_partition_host_and_accelerator_production_rust() {
    assert!(
        !CoverageLane::Host.includes_source("crates/j2k-cuda/src/error.rs", SourceRole::Production)
    );
    assert!(!CoverageLane::Host
        .includes_source("crates/j2k-metal/src/error.rs", SourceRole::Production));
    assert!(!CoverageLane::Host
        .includes_source("crates/j2k-core/src/accelerator.rs", SourceRole::Production));
    assert!(
        CoverageLane::Cuda.includes_source("crates/j2k-cuda/src/error.rs", SourceRole::Production)
    );
    assert!(CoverageLane::Metal
        .includes_source("crates/j2k-metal/src/error.rs", SourceRole::Production));
    assert!(CoverageLane::Cuda
        .includes_source("crates/j2k-core/src/accelerator.rs", SourceRole::Production));
    assert!(CoverageLane::Metal
        .includes_source("crates/j2k-core/src/accelerator.rs", SourceRole::Production));
    assert!(CoverageLane::Host.includes_source("crates/j2k/src/error.rs", SourceRole::Production));
    assert!(CoverageLane::Host.includes_source("xtask/src/coverage.rs", SourceRole::Production));
    assert!(
        !CoverageLane::Host.includes_source("crates/j2k/tests/decode.rs", SourceRole::TestTarget)
    );
}

#[test]
fn metal_raw_shader_span_is_narrower_than_the_host_source_file() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let path = "crates/j2k-metal/src/compute/shader_source.rs";
    let source = fs::read_to_string(root.join(path)).unwrap();
    let lines = source.lines().collect::<Vec<_>>();
    let embedded_shader_line = lines
        .iter()
        .position(|line| line.contains("#include <metal_stdlib>"))
        .expect("embedded shader marker")
        + 1;
    let included_shader_line = lines
        .iter()
        .position(|line| line.contains("include_str!(\"../store.metal\")"))
        .expect("included shader marker")
        + 1;

    assert_eq!(
        matching_exclusion(path, embedded_shader_line, &lines)
            .unwrap()
            .map(|rule| rule.id),
        Some("metal-embedded-shader-body")
    );
    assert!(matching_exclusion(path, included_shader_line, &lines)
        .unwrap()
        .is_none());
}

#[test]
fn cuda_simt_exclusion_covers_split_device_modules_only() {
    let split_device_module =
        "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/simt/src/exports.rs";
    assert_eq!(
        matching_exclusion(split_device_module, 1, &[])
            .unwrap()
            .map(|rule| rule.id),
        Some("cuda-simt-device-rust")
    );

    let host_module = "crates/j2k-cuda-runtime/src/cuda_oxide_j2k_encode/src/main.rs";
    assert_eq!(
        matching_exclusion(host_module, 1, &[])
            .unwrap()
            .map(|rule| rule.id),
        Some("cuda-generated-host-scaffold")
    );
    assert!(
        matching_exclusion("crates/j2k-cuda-runtime/src/j2k_encode.rs", 1, &[])
            .unwrap()
            .is_none()
    );
}

#[test]
fn exclusion_policy_maps_every_narrow_rule_to_existing_tests() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    validate_exclusion_policy(&root).unwrap();
    assert!(COVERAGE_EXCLUSIONS
        .iter()
        .all(|rule| !rule.evidence.is_empty()));
    assert!(!COVERAGE_EXCLUSIONS.iter().any(|rule| {
        matches!(
            rule.matcher,
            ExclusionMatcher::WholeFile {
                path: "crates/j2k-cuda/" | "crates/j2k-metal/"
            }
        )
    }));
}

#[test]
fn coverage_cli_defaults_to_host_and_accepts_explicit_lanes() {
    let default = parse_options(std::iter::empty()).unwrap();
    let metal = parse_options(
        [
            "metal".to_string(),
            "--base".to_string(),
            "HEAD^".to_string(),
        ]
        .into_iter(),
    )
    .unwrap();
    let cuda = parse_options(["cuda".to_string()].into_iter()).unwrap();

    assert_eq!(default.lane, CoverageLane::Host);
    assert_eq!(metal.lane, CoverageLane::Metal);
    assert_eq!(metal.base.as_deref(), Some("HEAD^"));
    assert_eq!(cuda.lane, CoverageLane::Cuda);
}
