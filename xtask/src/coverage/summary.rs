// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use serde_json::json;

use super::evaluation::coverage_percent;
use super::exclusion_policy::COVERAGE_EXCLUSIONS;
use super::model::{ChangedCoverageResult, CoverageLane, CHANGED_LINE_THRESHOLD_PERCENT};

pub(super) struct CoverageSummaryInput<'a> {
    pub(super) path: &'a Path,
    pub(super) lane: CoverageLane,
    pub(super) base: &'a str,
    pub(super) merge_base: &'a str,
    pub(super) head_sha: &'a str,
    pub(super) lcov_path: &'a Path,
    pub(super) compiler_regions_path: &'a Path,
    pub(super) cargo_llvm_cov_version: &'a str,
    pub(super) result: &'a ChangedCoverageResult,
    pub(super) violations: &'a [String],
}

pub(super) fn write_summary(input: &CoverageSummaryInput<'_>) -> Result<(), String> {
    let document = summary_document(input);
    let rendered = serde_json::to_string_pretty(&document)
        .map_err(|err| format!("failed to render coverage summary: {err}"))?;
    fs::write(input.path, format!("{rendered}\n"))
        .map_err(|err| format!("failed to write {}: {err}", input.path.display()))
}

fn summary_document(input: &CoverageSummaryInput<'_>) -> serde_json::Value {
    let result = input.result;
    let exclusions = COVERAGE_EXCLUSIONS
        .iter()
        .map(|exclusion| {
            json!({
                "id": exclusion.id,
                "reason": exclusion.reason,
                "changed_lines_excluded": result.exclusions.get(exclusion.id).copied().unwrap_or(0),
                "evidence_tests": exclusion.evidence.iter().map(|evidence| {
                    format!("{}::{}", evidence.path, evidence.name)
                }).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    let source_dispositions = result
        .source_dispositions
        .iter()
        .map(|(id, counts)| {
            (
                *id,
                json!({
                    "changed_lines": counts.changed_lines,
                    "files": counts.files,
                }),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let document = json!({
        "schema": "j2k-changed-line-coverage-v4",
        "lane": input.lane.name(),
        "lane_scope": input.lane.scope_name(),
        "status": if input.violations.is_empty() { "passed" } else { "failed" },
        "base": input.base,
        "merge_base": input.merge_base,
        "head_sha": input.head_sha,
        "threshold_percent": CHANGED_LINE_THRESHOLD_PERCENT,
        "cargo_llvm_cov_version": input.cargo_llvm_cov_version,
        "lcov_artifact": input.lcov_path.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
        "compiler_regions_artifact": input.compiler_regions_path.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
        "changed_files": result.changed_files,
        "overall": {
            "measurable_lines": result.overall.measurable,
            "covered_lines": result.overall.covered,
            "coverage_percent": coverage_percent(&result.overall),
        },
        "accelerator_host_rust": {
            "measurable_lines": result.accelerator.measurable,
            "covered_lines": result.accelerator.covered,
            "coverage_percent": coverage_percent(&result.accelerator),
        },
        "uncovered_lines": result.uncovered.iter().map(|(path, line)| format!("{path}:{line}")).collect::<Vec<_>>(),
        "residual_unmeasured_lines": result.unmeasured.iter().map(|(path, line)| format!("{path}:{line}")).collect::<Vec<_>>(),
        "absent_instrumentable_files": result.absent_instrumentable_files,
        "changed_functions_without_covered_body": result.changed_functions_without_covered_body,
        "changed_executable_bodies_without_covered_body": result.changed_executable_bodies_without_covered_body,
        "changed_deferred_bodies_without_covered_compiler_region": result.changed_deferred_bodies_without_covered_compiler_region,
        "compiler_noninstrumentable_deferred_bodies": result.compiler_noninstrumentable_deferred_bodies,
        "mixed_test_production_lines": result.mixed_test_production_lines,
        "changed_opaque_macros": result.changed_opaque_macros,
        "source_dispositions": source_dispositions,
        "narrow_exclusions": exclusions,
        "violations": input.violations,
    });
    document
}

pub(super) fn print_summary(
    lane: CoverageLane,
    summary_path: &Path,
    result: &ChangedCoverageResult,
) {
    let percent = coverage_percent(&result.overall)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}%"));
    let accelerator_percent = coverage_percent(&result.accelerator)
        .map_or_else(|| "n/a".to_string(), |value| format!("{value:.2}%"));
    eprintln!(
        "{} changed-line coverage: {} ({} / {} measurable lines)",
        lane.name(),
        percent,
        result.overall.covered,
        result.overall.measurable
    );
    eprintln!(
        "{} accelerator host coverage: {} ({} / {} measurable lines)",
        lane.name(),
        accelerator_percent,
        result.accelerator.covered,
        result.accelerator.measurable
    );
    eprintln!("coverage evidence: {}", summary_path.display());
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::Path;

    use super::{summary_document, CoverageSummaryInput};
    use crate::coverage::model::{ChangedCoverageResult, CoverageCounts, CoverageLane};

    #[test]
    fn summary_records_partitioned_schema_and_exact_source_sha() {
        let result = ChangedCoverageResult {
            overall: CoverageCounts {
                measurable: 5,
                covered: 4,
            },
            accelerator: CoverageCounts::default(),
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
            mixed_test_production_lines: Vec::new(),
            changed_opaque_macros: Vec::new(),
        };
        let document = summary_document(&CoverageSummaryInput {
            path: Path::new("coverage-host-summary.json"),
            lane: CoverageLane::Host,
            base: "v0.6.2",
            merge_base: "1111111111111111111111111111111111111111",
            head_sha: "2222222222222222222222222222222222222222",
            lcov_path: Path::new("lcov-host.info"),
            compiler_regions_path: Path::new("coverage-host-regions.json"),
            cargo_llvm_cov_version: "0.8.7",
            result: &result,
            violations: &[],
        });

        assert_eq!(document["schema"], "j2k-changed-line-coverage-v4");
        assert_eq!(
            document["head_sha"],
            "2222222222222222222222222222222222222222"
        );
        assert_eq!(document["lane_scope"], "non-accelerator-production");
    }
}
