// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::exclusion_policy::matching_exclusion;
use super::model::{
    is_accelerator_path, ChangedCoverageResult, CoverageCounts, CoverageLane, LcovReport,
    CHANGED_LINE_THRESHOLD_PERCENT,
};

pub(super) fn evaluate_changed_coverage(
    lane: CoverageLane,
    root: &Path,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    report: &LcovReport,
) -> Result<ChangedCoverageResult, String> {
    let mut result = ChangedCoverageResult {
        overall: CoverageCounts::default(),
        accelerator: CoverageCounts::default(),
        changed_files: BTreeSet::new(),
        uncovered: Vec::new(),
        unmeasured: Vec::new(),
        exclusions: BTreeMap::new(),
        absent_instrumentable_files: Vec::new(),
    };

    for (path, lines) in changed {
        if !lane.includes_path(path) {
            continue;
        }
        let source_path = root.join(path);
        let source = fs::read_to_string(&source_path).map_err(|err| {
            format!(
                "failed to read changed source {}: {err}",
                source_path.display()
            )
        })?;
        let source_lines = source.lines().collect::<Vec<_>>();
        let test_module_start = terminal_test_module_start(&source_lines);
        let file_coverage = report.lines.get(path);
        let mut changed_unexcluded = false;

        result.changed_files.insert(path.clone());
        for &line_number in lines {
            if line_number == 0 || line_number > source_lines.len() {
                continue;
            }
            if test_module_start.is_some_and(|start| line_number >= start) {
                continue;
            }
            if let Some(exclusion) = matching_exclusion(path, line_number, &source_lines)? {
                *result.exclusions.entry(exclusion.id).or_default() += 1;
                continue;
            }
            changed_unexcluded = true;
            let Some(count) = file_coverage.and_then(|coverage| coverage.get(&line_number)) else {
                result.unmeasured.push((path.clone(), line_number));
                continue;
            };

            result.overall.measurable += 1;
            if *count > 0 {
                result.overall.covered += 1;
            } else {
                result.uncovered.push((path.clone(), line_number));
            }
            if is_accelerator_path(path) {
                result.accelerator.measurable += 1;
                if *count > 0 {
                    result.accelerator.covered += 1;
                }
            }
        }

        if lane != CoverageLane::Host
            && changed_unexcluded
            && file_coverage.is_none()
            && source_has_instrumentable_function(path, &source_lines)?
        {
            result.absent_instrumentable_files.push(path.clone());
        }
    }

    Ok(result)
}

fn terminal_test_module_start(source: &[&str]) -> Option<usize> {
    source.windows(3).enumerate().find_map(|(index, lines)| {
        let first = lines[0].trim();
        let second = lines[1].trim();
        let third = lines[2].trim();
        if first == "#[cfg(test)]"
            && (second.starts_with("mod tests") || third.starts_with("mod tests"))
        {
            Some(index + 1)
        } else {
            None
        }
    })
}

fn source_has_instrumentable_function(path: &str, source: &[&str]) -> Result<bool, String> {
    for (index, line) in source.iter().enumerate() {
        let line_number = index + 1;
        if terminal_test_module_start(source).is_some_and(|start| line_number >= start)
            || matching_exclusion(path, line_number, source)?.is_some()
        {
            continue;
        }
        let trimmed = line.trim_start();
        if !trimmed.starts_with("//")
            && !trimmed.starts_with('*')
            && (trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("pub(crate) fn ")
                || trimmed.starts_with("pub(super) fn ")
                || trimmed.starts_with("const fn ")
                || trimmed.starts_with("pub const fn "))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(super) fn coverage_violations(
    lane: CoverageLane,
    result: &ChangedCoverageResult,
) -> Vec<String> {
    let mut violations = Vec::new();
    if !meets_threshold(&result.overall) {
        violations.push(format!(
            "{} changed executable Rust lines are {:.2}% covered ({} / {}), below {}%",
            lane.name(),
            coverage_percent(&result.overall).unwrap_or(0.0),
            result.overall.covered,
            result.overall.measurable,
            CHANGED_LINE_THRESHOLD_PERCENT
        ));
    }
    if result.accelerator.measurable > 0 && !meets_threshold(&result.accelerator) {
        violations.push(format!(
            "{} changed accelerator host lines are {:.2}% covered ({} / {}), below {}%",
            lane.name(),
            coverage_percent(&result.accelerator).unwrap_or(0.0),
            result.accelerator.covered,
            result.accelerator.measurable,
            CHANGED_LINE_THRESHOLD_PERCENT
        ));
    }
    if !result.absent_instrumentable_files.is_empty() {
        violations.push(format!(
            "instrumentable accelerator source files are absent from the {} LCOV artifact: {}",
            lane.name(),
            result.absent_instrumentable_files.join(", ")
        ));
    }
    violations
}

fn meets_threshold(counts: &CoverageCounts) -> bool {
    counts.measurable == 0
        || counts.covered.saturating_mul(100)
            >= counts
                .measurable
                .saturating_mul(CHANGED_LINE_THRESHOLD_PERCENT as usize)
}

pub(super) fn coverage_percent(counts: &CoverageCounts) -> Option<f64> {
    (counts.measurable > 0).then(|| counts.covered as f64 * 100.0 / counts.measurable as f64)
}
