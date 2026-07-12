// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::path::Path;

use crate::coverage::{analyze_test_only_syntax, SourceAuditSyntax, SourceAuditTestSpan};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MaskedRustSource {
    pub(crate) text: String,
    pub(crate) masked_nodes: usize,
    pub(crate) mixed_lines: BTreeSet<usize>,
}

pub(crate) fn mask_test_only_syntax(
    repository_root: &Path,
    relative_path: &Path,
    source: &str,
) -> Result<MaskedRustSource, String> {
    let relative = relative_path.to_str().ok_or_else(|| {
        format!(
            "source-audit path is not UTF-8: {}",
            relative_path.display()
        )
    })?;
    let analysis = analyze_test_only_syntax(repository_root, relative, source)
        .map_err(|error| format!("source-aware audit of {relative} failed: {error}"))?;
    mask_analysis(source, analysis)
        .map_err(|error| format!("mask test-only syntax in {relative}: {error}"))
}

fn mask_analysis(source: &str, analysis: SourceAuditSyntax) -> Result<MaskedRustSource, String> {
    let mut bytes = source.as_bytes().to_vec();
    let masked_nodes = analysis.test_only_spans.len();
    if analysis.fully_test_only {
        blank_range(&mut bytes, 0, source.len());
    } else {
        let line_starts = line_starts(source);
        let mut ranges = analysis
            .test_only_spans
            .iter()
            .map(|span| span_range(source, &line_starts, *span))
            .collect::<Result<Vec<_>, _>>()?;
        ranges.sort_unstable();
        for (start, end) in merge_ranges(&ranges) {
            blank_range(&mut bytes, start, end);
        }
    }
    let text = String::from_utf8(bytes)
        .map_err(|error| format!("masked Rust source is not UTF-8: {error}"))?;
    if text.len() != source.len()
        || text
            .bytes()
            .enumerate()
            .any(|(index, byte)| byte == b'\n' && source.as_bytes()[index] != b'\n')
        || source
            .bytes()
            .enumerate()
            .any(|(index, byte)| byte == b'\n' && text.as_bytes()[index] != b'\n')
    {
        return Err("masking changed source byte or line positions".to_string());
    }
    Ok(MaskedRustSource {
        text,
        masked_nodes,
        mixed_lines: analysis.mixed_lines,
    })
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    starts.extend(
        source
            .bytes()
            .enumerate()
            .filter_map(|(index, byte)| (byte == b'\n').then_some(index + 1)),
    );
    starts
}

fn span_range(
    source: &str,
    line_starts: &[usize],
    span: SourceAuditTestSpan,
) -> Result<(usize, usize), String> {
    let start = source_offset(source, line_starts, span.start_line, span.start_column)?;
    let mut end = source_offset(source, line_starts, span.end_line, span.end_column)?;
    let bytes = source.as_bytes();
    while bytes
        .get(end)
        .is_some_and(|byte| matches!(byte, b' ' | b'\t'))
    {
        end += 1;
    }
    if bytes
        .get(end)
        .is_some_and(|byte| matches!(byte, b',' | b';'))
    {
        end += 1;
    }
    if start > end {
        return Err(format!(
            "test-only span has reversed offsets {start}..{end}"
        ));
    }
    Ok((start, end))
}

fn source_offset(
    source: &str,
    line_starts: &[usize],
    line: usize,
    column: usize,
) -> Result<usize, String> {
    let line_start = *line_starts
        .get(line.saturating_sub(1))
        .ok_or_else(|| format!("span line {line} exceeds source line count"))?;
    let line_end = line_starts
        .get(line)
        .map_or(source.len(), |next_start| next_start.saturating_sub(1));
    let offset = line_start
        .checked_add(column)
        .ok_or_else(|| "span column offset overflow".to_string())?;
    if offset > line_end {
        return Err(format!(
            "span column {column} exceeds line {line} byte length {}",
            line_end.saturating_sub(line_start)
        ));
    }
    Ok(offset)
}

fn merge_ranges(ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in ranges {
        if let Some((_, previous_end)) = merged.last_mut() {
            if start <= *previous_end {
                *previous_end = (*previous_end).max(end);
                continue;
            }
        }
        merged.push((start, end));
    }
    merged
}

fn blank_range(bytes: &mut [u8], start: usize, end: usize) {
    for byte in &mut bytes[start..end] {
        if !matches!(*byte, b'\n' | b'\r') {
            *byte = b' ';
        }
    }
}
