// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SourceFileAnalysis, TestOnlyLineDisposition, TestOnlySpan};

impl SourceFileAnalysis {
    pub(in crate::coverage) fn test_only_disposition(
        &self,
        line_number: usize,
        source_line: &str,
    ) -> TestOnlyLineDisposition {
        if !self.test_only_lines.contains(&line_number) {
            return TestOnlyLineDisposition::Production;
        }
        let intervals = self
            .test_only_spans
            .iter()
            .filter_map(|span| test_only_interval(*span, line_number, source_line.len()))
            .map(|interval| include_trailing_separator(source_line, interval))
            .collect::<Vec<_>>();
        if intervals.is_empty() || !has_production_text_outside(source_line, &intervals) {
            TestOnlyLineDisposition::TestOnly
        } else {
            TestOnlyLineDisposition::Mixed
        }
    }
}

fn include_trailing_separator(source_line: &str, (start, end): (usize, usize)) -> (usize, usize) {
    let bytes = source_line.as_bytes();
    let mut extended = end;
    while bytes.get(extended).is_some_and(u8::is_ascii_whitespace) {
        extended += 1;
    }
    if bytes
        .get(extended)
        .is_some_and(|byte| matches!(byte, b',' | b';'))
    {
        extended += 1;
    }
    (start, extended)
}

fn test_only_interval(
    span: TestOnlySpan,
    line_number: usize,
    line_len: usize,
) -> Option<(usize, usize)> {
    if !(span.start_line..=span.end_line).contains(&line_number) {
        return None;
    }
    let start = if line_number == span.start_line {
        span.start_column.min(line_len)
    } else {
        0
    };
    let end = if line_number == span.end_line {
        span.end_column.min(line_len)
    } else {
        line_len
    };
    Some((start.min(end), end))
}

fn has_production_text_outside(source_line: &str, test_only: &[(usize, usize)]) -> bool {
    let bytes = source_line.as_bytes();
    let mut column = 0;
    while column < bytes.len() {
        if test_only
            .iter()
            .any(|(start, end)| (*start..*end).contains(&column))
            || bytes[column].is_ascii_whitespace()
        {
            column += 1;
            continue;
        }
        if bytes.get(column..column + 2) == Some(b"//") {
            return false;
        }
        return true;
    }
    false
}
