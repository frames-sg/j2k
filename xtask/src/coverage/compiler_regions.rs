// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use proc_macro2::Span;

mod parsing;

pub(super) use parsing::parse_compiler_regions;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct SourcePosition {
    line: usize,
    column: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SourceSpan {
    start: SourcePosition,
    end: SourcePosition,
}

impl SourceSpan {
    pub(super) fn new(
        start_line: usize,
        start_column: usize,
        end_line: usize,
        end_column: usize,
    ) -> Result<Self, String> {
        let start = source_position(start_line, start_column, "start")?;
        let end = source_position(end_line, end_column, "end")?;
        if start >= end {
            return Err(format!(
                "compiler coverage span must have a positive extent, found {start_line}:{start_column}-{end_line}:{end_column}"
            ));
        }
        Ok(Self { start, end })
    }

    pub(super) fn from_proc_macro(span: Span) -> Result<Self, String> {
        let start = span.start();
        let end = span.end();
        Self::new(
            start.line,
            start
                .column
                .checked_add(1)
                .ok_or_else(|| "source start column overflowed".to_string())?,
            end.line,
            end.column
                .checked_add(1)
                .ok_or_else(|| "source end column overflowed".to_string())?,
        )
    }

    fn contains(self, other: Self) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    fn intersects_line(self, line: usize) -> bool {
        self.start.line <= line
            && (line < self.end.line || (line == self.end.line && self.end.column > 1))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompilerRegionEvidence {
    Covered,
    Uncovered,
    NonInstrumentable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CompilerRegion {
    span: SourceSpan,
    count: u64,
}

#[derive(Debug, Default)]
pub(super) struct CompilerRegionReport {
    files: BTreeSet<String>,
    regions: BTreeMap<String, Vec<CompilerRegion>>,
}

impl CompilerRegionReport {
    pub(super) fn evidence_for_line(
        &self,
        path: &str,
        line: usize,
    ) -> Option<CompilerRegionEvidence> {
        if !self.files.contains(path) {
            return None;
        }
        let candidates = self
            .regions
            .get(path)
            .into_iter()
            .flatten()
            .filter(|region| region.span.intersects_line(line))
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Some(CompilerRegionEvidence::NonInstrumentable);
        }
        let most_specific = candidates.iter().copied().filter(|candidate| {
            !candidates
                .iter()
                .any(|nested| candidate.span != nested.span && candidate.span.contains(nested.span))
        });
        if most_specific.clone().all(|region| region.count > 0) {
            Some(CompilerRegionEvidence::Covered)
        } else {
            Some(CompilerRegionEvidence::Uncovered)
        }
    }

    pub(super) fn evidence_for(
        &self,
        path: &str,
        body: SourceSpan,
    ) -> Result<CompilerRegionEvidence, String> {
        if !self.files.contains(path) {
            return Err(format!(
                "compiler coverage JSON has no source record for changed file `{path}`"
            ));
        }
        let nested = self
            .regions
            .get(path)
            .into_iter()
            .flatten()
            .filter(|region| body.contains(region.span))
            .collect::<Vec<_>>();
        if nested.is_empty() {
            return Ok(CompilerRegionEvidence::NonInstrumentable);
        }
        if nested.iter().any(|region| region.count > 0) {
            Ok(CompilerRegionEvidence::Covered)
        } else {
            Ok(CompilerRegionEvidence::Uncovered)
        }
    }

    #[cfg(test)]
    pub(super) fn for_test(path: &str, regions: &[(SourceSpan, u64)]) -> Self {
        Self {
            files: BTreeSet::from([path.to_string()]),
            regions: BTreeMap::from([(
                path.to_string(),
                regions
                    .iter()
                    .map(|(span, count)| CompilerRegion {
                        span: *span,
                        count: *count,
                    })
                    .collect(),
            )]),
        }
    }
}

fn source_position(line: usize, column: usize, label: &str) -> Result<SourcePosition, String> {
    if line == 0 || column == 0 {
        return Err(format!(
            "compiler coverage {label} position must be one-based, found {line}:{column}"
        ));
    }
    Ok(SourcePosition { line, column })
}

#[cfg(test)]
mod tests;
