// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use super::compiler_regions::{CompilerRegionEvidence, CompilerRegionReport};
use super::critical_path_policy::{
    audited_zero_body_findings, classify_path, ZeroBodyAudit, ZeroBodyKind,
};
use super::exclusion_policy::matching_exclusion;
use super::model::{
    is_accelerator_path, ChangedCoverageResult, CoverageCounts, CoverageLane, LcovReport,
    CHANGED_LINE_THRESHOLD_PERCENT,
};
use super::source_analysis::{
    DeferredBodyEvidence, OpaqueMacroKind, SourceFileAnalysis, SourceIndex, SourceRole,
    TestOnlyLineDisposition,
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
    compiler_regions: &'a CompilerRegionReport,
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
        critical: CoverageCounts::default(),
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
        compiler_noninstrumentable_lines: Vec::new(),
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
        .changed_deferred_bodies_without_covered_compiler_region
        .sort();
    result.compiler_noninstrumentable_deferred_bodies.sort();
    result.compiler_noninstrumentable_lines.sort();
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
            compiler_regions: &self.report.compiler_regions,
            analysis,
            measurable_source: self.lane.includes_source(path, analysis.role),
        };

        result.changed_files.insert(path.to_string());
        let instrumentable_lines = evidence.evaluate_changed_lines(lines, result)?;
        evidence.record_missing_body_evidence(&instrumentable_lines, result, absent_files)?;
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
            if !self.analysis.executable_lines.contains(&line_number) {
                if self
                    .coverage
                    .and_then(|coverage| coverage.get(&line_number))
                    .is_none()
                {
                    result.unmeasured.push((self.path.to_string(), line_number));
                }
                continue;
            }
            let Some(count) = self
                .coverage
                .and_then(|coverage| coverage.get(&line_number))
            else {
                match self
                    .compiler_regions
                    .evidence_for_line(self.path, line_number)
                {
                    Some(CompilerRegionEvidence::Covered) => {
                        record_measurable_line(result, self.path, line_number, 1);
                        continue;
                    }
                    Some(CompilerRegionEvidence::Uncovered) => {
                        record_measurable_line(result, self.path, line_number, 0);
                        continue;
                    }
                    Some(CompilerRegionEvidence::NonInstrumentable) => {
                        result
                            .compiler_noninstrumentable_lines
                            .push(format!("{}:{line_number}", self.path));
                        continue;
                    }
                    None => {}
                }
                result.unmeasured.push((self.path.to_string(), line_number));
                record_measurable_line(result, self.path, line_number, 0);
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
    ) -> Result<(), String> {
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
                DeferredBodyEvidence::CompilerRegion(span) => {
                    match self.compiler_regions.evidence_for(self.path, span)? {
                        CompilerRegionEvidence::Covered => {}
                        CompilerRegionEvidence::Uncovered => result
                            .changed_deferred_bodies_without_covered_compiler_region
                            .push(format!("{}::{}", self.path, body.label)),
                        CompilerRegionEvidence::NonInstrumentable => result
                            .compiler_noninstrumentable_deferred_bodies
                            .push(format!("{}::{}", self.path, body.label)),
                    }
                }
            }
        }

        for opaque in &self.analysis.opaque_macros {
            if !opaque.required_on_host
                || !changed_span(instrumentable_lines, opaque.start, opaque.end)
                || (opaque.kind == OpaqueMacroKind::Invocation
                    && self.body_is_covered(opaque.start, opaque.end))
            {
                continue;
            }
            self.record_absent_file(absent_files);
            result
                .changed_opaque_macros
                .push(format!("{}::{}", self.path, opaque.label));
        }
        Ok(())
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
    if classify_path(path).is_some() {
        result.critical.measurable += 1;
        if count > 0 {
            result.critical.covered += 1;
        }
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
    if lane.enforces_overall_changed_lines() && !meets_threshold(&result.overall) {
        violations.push(format!(
            "{} changed executable Rust lines are {:.2}% covered ({} / {}), below {}%",
            lane.name(),
            coverage_percent(&result.overall).unwrap_or(0.0),
            result.overall.covered,
            result.overall.measurable,
            CHANGED_LINE_THRESHOLD_PERCENT
        ));
    }
    if result.critical.measurable > 0 && !meets_threshold(&result.critical) {
        violations.push(format!(
            "{} changed critical-path lines are {:.2}% covered ({} / {}), below {}%",
            lane.name(),
            coverage_percent(&result.critical).unwrap_or(0.0),
            result.critical.covered,
            result.critical.measurable,
            CHANGED_LINE_THRESHOLD_PERCENT
        ));
    }
    let absent_files = result
        .absent_instrumentable_files
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let absent_critical_behavior_files = audited_zero_body_findings(lane, result)
        .into_iter()
        .filter(|finding| {
            matches!(finding.audit, ZeroBodyAudit::Critical(_))
                && finding.kind != ZeroBodyKind::OpaqueMacro
        })
        .filter_map(|finding| {
            let path = finding
                .finding
                .split_once("::")
                .map_or(finding.finding, |(path, _)| path);
            absent_files.contains(path).then(|| path.to_string())
        })
        .collect::<BTreeSet<_>>();
    if !absent_critical_behavior_files.is_empty() {
        violations.push(format!(
            "critical executable bodies are absent from the {} LCOV artifact: {}",
            lane.name(),
            absent_critical_behavior_files
                .into_iter()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !result.mixed_test_production_lines.is_empty() {
        violations.push(format!(
            "changed source lines mix cfg(test)-only and production Rust, so line-only LCOV cannot attribute execution safely; split test-only and production syntax onto separate lines: {}",
            result.mixed_test_production_lines.join(", ")
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
