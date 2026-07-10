// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::Path;

use serde_json::json;

use super::evaluation::coverage_percent;
use super::exclusion_policy::COVERAGE_EXCLUSIONS;
use super::model::{ChangedCoverageResult, CoverageLane, CHANGED_LINE_THRESHOLD_PERCENT};

pub(super) fn write_summary(
    path: &Path,
    lane: CoverageLane,
    base: &str,
    merge_base: &str,
    lcov_path: &Path,
    result: &ChangedCoverageResult,
    violations: &[String],
) -> Result<(), String> {
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
    let document = json!({
        "schema": "j2k-changed-line-coverage-v1",
        "lane": lane.name(),
        "status": if violations.is_empty() { "passed" } else { "failed" },
        "base": base,
        "merge_base": merge_base,
        "threshold_percent": CHANGED_LINE_THRESHOLD_PERCENT,
        "lcov_artifact": lcov_path.file_name().map_or_else(String::new, |name| name.to_string_lossy().into_owned()),
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
        "non_executable_or_not_instrumented_lines": result.unmeasured.iter().map(|(path, line)| format!("{path}:{line}")).collect::<Vec<_>>(),
        "absent_instrumentable_files": result.absent_instrumentable_files,
        "narrow_exclusions": exclusions,
        "violations": violations,
    });
    let rendered = serde_json::to_string_pretty(&document)
        .map_err(|err| format!("failed to render coverage summary: {err}"))?;
    fs::write(path, format!("{rendered}\n"))
        .map_err(|err| format!("failed to write {}: {err}", path.display()))
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
