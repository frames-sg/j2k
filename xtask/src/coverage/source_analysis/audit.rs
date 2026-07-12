// SPDX-License-Identifier: MIT OR Apache-2.0

//! Narrow production-source audit facade over coverage's cfg/Syn analysis.

use std::collections::BTreeSet;
use std::path::Path;

use super::{
    analyze_source, CoverageCfgContext, ReachKind, SourceFileAnalysis, SourceRole,
    TestOnlyLineDisposition,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SourceAuditTestSpan {
    pub(crate) start_line: usize,
    pub(crate) start_column: usize,
    pub(crate) end_line: usize,
    pub(crate) end_column: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SourceAuditSyntax {
    pub(crate) test_only_spans: Vec<SourceAuditTestSpan>,
    pub(crate) mixed_lines: BTreeSet<usize>,
    pub(crate) fully_test_only: bool,
}

pub(crate) fn analyze_test_only_syntax(
    root: &Path,
    path: &str,
    source: &str,
) -> Result<SourceAuditSyntax, String> {
    let cfg = CoverageCfgContext::for_current_target(BTreeSet::new(), None);
    let parsed = analyze_source(root, path, source, ReachKind::Production, true, &cfg)?;
    let analysis = SourceFileAnalysis {
        role: SourceRole::Production,
        test_only_lines: parsed.test_only_lines,
        test_only_spans: parsed.test_only_spans,
        executable_lines: parsed.executable_lines,
        functions: parsed.functions,
        executable_bodies: parsed.executable_bodies,
        opaque_macros: parsed.opaque_macros,
    };
    let mut mixed_lines = BTreeSet::new();
    let mut production_lines = 0usize;
    let mut test_only_lines = 0usize;
    for (index, line) in source.lines().enumerate() {
        let line_number = index + 1;
        match analysis.test_only_disposition(line_number, line) {
            TestOnlyLineDisposition::Production => production_lines += 1,
            TestOnlyLineDisposition::TestOnly => test_only_lines += 1,
            TestOnlyLineDisposition::Mixed => {
                mixed_lines.insert(line_number);
            }
        }
    }
    let test_only_spans = analysis
        .test_only_spans
        .iter()
        .map(|span| SourceAuditTestSpan {
            start_line: span.start_line,
            start_column: span.start_column,
            end_line: span.end_line,
            end_column: span.end_column,
        })
        .collect();
    let fully_test_only = production_lines == 0 && mixed_lines.is_empty() && test_only_lines != 0;
    Ok(SourceAuditSyntax {
        test_only_spans,
        mixed_lines,
        fully_test_only,
    })
}
