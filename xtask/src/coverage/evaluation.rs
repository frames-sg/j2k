// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::exclusion_policy::matching_exclusion;
use super::model::{
    is_accelerator_path, ChangedCoverageResult, CoverageCounts, CoverageLane, LcovReport,
    CHANGED_LINE_THRESHOLD_PERCENT,
};
use super::source_analysis::{
    DeferredBodyEvidence, SourceFileAnalysis, SourceIndex, SourceRole, TestOnlyLineDisposition,
};

struct EvaluationInputs<'a> {
    lane: CoverageLane,
    root: &'a Path,
    report: &'a LcovReport,
    source_index: &'a SourceIndex,
}

struct ChangedFileEvidence<'a> {
    path: &'a str,
    source_lines: &'a [&'a str],
    coverage: Option<&'a BTreeMap<usize, u64>>,
    analysis: &'a SourceFileAnalysis,
    measurable_source: bool,
}

pub(super) fn evaluate_changed_coverage(
    lane: CoverageLane,
    root: &Path,
    changed: &BTreeMap<String, BTreeSet<usize>>,
    report: &LcovReport,
    source_index: &SourceIndex,
) -> Result<ChangedCoverageResult, String> {
    let mut result = ChangedCoverageResult {
        overall: CoverageCounts::default(),
        accelerator: CoverageCounts::default(),
        changed_files: BTreeSet::new(),
        uncovered: Vec::new(),
        unmeasured: Vec::new(),
        exclusions: BTreeMap::new(),
        source_dispositions: BTreeMap::new(),
        absent_instrumentable_files: Vec::new(),
        changed_functions_without_covered_body: Vec::new(),
        changed_executable_bodies_without_covered_body: Vec::new(),
        changed_deferred_bodies_without_distinct_line_evidence: Vec::new(),
        mixed_test_production_lines: Vec::new(),
        changed_opaque_macros: Vec::new(),
    };
    let mut absent_files = BTreeSet::new();
    let inputs = EvaluationInputs {
        lane,
        root,
        report,
        source_index,
    };

    for (path, lines) in changed {
        inputs.evaluate_file(path, lines, &mut result, &mut absent_files)?;
    }

    result.absent_instrumentable_files = absent_files.into_iter().collect();
    result.changed_functions_without_covered_body.sort();
    result.changed_executable_bodies_without_covered_body.sort();
    result
        .changed_deferred_bodies_without_distinct_line_evidence
        .sort();
    result.mixed_test_production_lines.sort();
    result.changed_opaque_macros.sort();
    Ok(result)
}

impl EvaluationInputs<'_> {
    fn evaluate_file(
        &self,
        path: &str,
        lines: &BTreeSet<usize>,
        result: &mut ChangedCoverageResult,
        absent_files: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        if !self.lane.owns_path(path) {
            return Ok(());
        }
        let analysis = self.source_index.file(path)?;
        let source_path = self.root.join(path);
        let source = fs::read_to_string(&source_path).map_err(|error| {
            format!(
                "failed to read changed source {}: {error}",
                source_path.display()
            )
        })?;
        let source_lines = source.lines().collect::<Vec<_>>();
        let evidence = ChangedFileEvidence {
            path,
            source_lines: &source_lines,
            coverage: self.report.lines.get(path),
            analysis,
            measurable_source: self.lane.includes_source(path, analysis.role),
        };

        result.changed_files.insert(path.to_string());
        let instrumentable_lines = evidence.evaluate_changed_lines(lines, result)?;
        evidence.record_missing_body_evidence(&instrumentable_lines, result, absent_files);
        Ok(())
    }
}

impl ChangedFileEvidence<'_> {
    fn evaluate_changed_lines(
        &self,
        changed_lines: &BTreeSet<usize>,
        result: &mut ChangedCoverageResult,
    ) -> Result<BTreeSet<usize>, String> {
        let mut instrumentable_lines = BTreeSet::new();
        for &line_number in changed_lines {
            if line_number == 0 || line_number > self.source_lines.len() {
                return Err(format!(
                    "changed-line diff references invalid line {}:{line_number}; source has {} lines",
                    self.path,
                    self.source_lines.len()
                ));
            }
            let test_disposition = if self.measurable_source {
                self.analysis
                    .test_only_disposition(line_number, self.source_lines[line_number - 1])
            } else {
                TestOnlyLineDisposition::Production
            };
            let disposition = match test_disposition {
                TestOnlyLineDisposition::TestOnly => SourceRole::TestOnly.disposition(),
                TestOnlyLineDisposition::Mixed => "mixed-test-production",
                TestOnlyLineDisposition::Production => self.analysis.role.disposition(),
            };
            record_disposition(result, disposition, self.path);

            let exclusion = matching_exclusion(self.path, line_number, self.source_lines)?;
            if let Some(exclusion) = exclusion {
                *result.exclusions.entry(exclusion.id).or_default() += 1;
            }
            if !self.measurable_source {
                if reviewed_nonmeasurable_role(self.analysis.role) && exclusion.is_none() {
                    return Err(format!(
                        "reviewed non-instrumentable source `{}` line {line_number} has no matching coverage exclusion",
                        self.path
                    ));
                }
                continue;
            }
            if test_disposition == TestOnlyLineDisposition::Mixed {
                result
                    .mixed_test_production_lines
                    .push(format!("{}:{line_number}", self.path));
                continue;
            }
            if test_disposition == TestOnlyLineDisposition::TestOnly || exclusion.is_some() {
                continue;
            }

            instrumentable_lines.insert(line_number);
            let Some(count) = self
                .coverage
                .and_then(|coverage| coverage.get(&line_number))
            else {
                result.unmeasured.push((self.path.to_string(), line_number));
                if self.analysis.executable_lines.contains(&line_number) {
                    record_measurable_line(result, self.path, line_number, 0);
                }
                continue;
            };
            record_measurable_line(result, self.path, line_number, *count);
        }
        Ok(instrumentable_lines)
    }

    fn record_missing_body_evidence(
        &self,
        instrumentable_lines: &BTreeSet<usize>,
        result: &mut ChangedCoverageResult,
        absent_files: &mut BTreeSet<String>,
    ) {
        for function in &self.analysis.functions {
            if !function.required_on_host
                || !changed_span(instrumentable_lines, function.start, function.end)
                || self.body_is_covered(function.body_start, function.body_end)
            {
                continue;
            }
            self.record_absent_file(absent_files);
            result.changed_functions_without_covered_body.push(format!(
                "{}::{}@{}",
                self.path, function.name, function.start
            ));
        }

        for body in &self.analysis.executable_bodies {
            if !body.required_on_host || !changed_span(instrumentable_lines, body.start, body.end) {
                continue;
            }
            match body.evidence {
                DeferredBodyEvidence::DistinctLines { start, end }
                    if self.body_is_covered(start, end) => {}
                DeferredBodyEvidence::DistinctLines { .. } => {
                    self.record_absent_file(absent_files);
                    result
                        .changed_executable_bodies_without_covered_body
                        .push(format!("{}::{}", self.path, body.label));
                }
                DeferredBodyEvidence::SharedCreationLine => {
                    result
                        .changed_deferred_bodies_without_distinct_line_evidence
                        .push(format!("{}::{}", self.path, body.label));
                }
            }
        }

        for opaque in &self.analysis.opaque_macros {
            if !opaque.required_on_host
                || !changed_span(instrumentable_lines, opaque.start, opaque.end)
            {
                continue;
            }
            self.record_absent_file(absent_files);
            result
                .changed_opaque_macros
                .push(format!("{}::{}", self.path, opaque.label));
        }
    }

    fn body_is_covered(&self, start: usize, end: usize) -> bool {
        self.coverage
            .is_some_and(|coverage| coverage.range(start..=end).any(|(_, count)| *count > 0))
    }

    fn record_absent_file(&self, absent_files: &mut BTreeSet<String>) {
        if self.coverage.is_none() {
            absent_files.insert(self.path.to_string());
        }
    }
}

fn changed_span(lines: &BTreeSet<usize>, start: usize, end: usize) -> bool {
    lines.range(start..=end).next().is_some()
}

fn record_measurable_line(
    result: &mut ChangedCoverageResult,
    path: &str,
    line_number: usize,
    count: u64,
) {
    result.overall.measurable += 1;
    if count > 0 {
        result.overall.covered += 1;
    } else {
        result.uncovered.push((path.to_string(), line_number));
    }
    if is_accelerator_path(path) {
        result.accelerator.measurable += 1;
        if count > 0 {
            result.accelerator.covered += 1;
        }
    }
}

fn record_disposition(result: &mut ChangedCoverageResult, disposition: &'static str, path: &str) {
    let counts = result.source_dispositions.entry(disposition).or_default();
    counts.changed_lines += 1;
    counts.files.insert(path.to_string());
}

const fn reviewed_nonmeasurable_role(role: SourceRole) -> bool {
    matches!(
        role,
        SourceRole::Generated(_) | SourceRole::VendoredReviewed(_)
    )
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
    if !result.changed_functions_without_covered_body.is_empty() {
        violations.push(format!(
            "changed instrumentable functions have no covered body in the {} LCOV artifact: {}",
            lane.name(),
            result.changed_functions_without_covered_body.join(", ")
        ));
    }
    if !result
        .changed_executable_bodies_without_covered_body
        .is_empty()
    {
        violations.push(format!(
            "changed executable bodies have no covered body in the {} LCOV artifact: {}",
            lane.name(),
            result
                .changed_executable_bodies_without_covered_body
                .join(", ")
        ));
    }
    if !result
        .changed_deferred_bodies_without_distinct_line_evidence
        .is_empty()
    {
        violations.push(format!(
            "changed deferred bodies share their only LCOV line with the creation site in the {} lane; line coverage cannot prove that the body ran. Put the deferred body on a distinct source line or move the changed behavior into a covered named function: {}",
            lane.name(),
            result
                .changed_deferred_bodies_without_distinct_line_evidence
                .join(", ")
        ));
    }
    if !result.mixed_test_production_lines.is_empty() {
        violations.push(format!(
            "changed source lines mix cfg(test)-only and production Rust, so line-only LCOV cannot attribute execution safely; split test-only and production syntax onto separate lines: {}",
            result.mixed_test_production_lines.join(", ")
        ));
    }
    if !result.changed_opaque_macros.is_empty() {
        violations.push(format!(
            "changed opaque macros require a narrow reviewed coverage exclusion in the {} lane: {}",
            lane.name(),
            result.changed_opaque_macros.join(", ")
        ));
    }
    violations
}

fn meets_threshold(counts: &CoverageCounts) -> bool {
    let Ok(threshold) = usize::try_from(CHANGED_LINE_THRESHOLD_PERCENT) else {
        return false;
    };
    counts.measurable == 0
        || counts.covered.saturating_mul(100) >= counts.measurable.saturating_mul(threshold)
}

pub(super) fn coverage_percent(counts: &CoverageCounts) -> Option<f64> {
    let covered = u32::try_from(counts.covered).ok()?;
    let measurable = u32::try_from(counts.measurable).ok()?;
    (measurable > 0).then(|| f64::from(covered) * 100.0 / f64::from(measurable))
}
