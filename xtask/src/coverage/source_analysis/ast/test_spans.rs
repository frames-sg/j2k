// SPDX-License-Identifier: MIT OR Apache-2.0

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::Attribute;

use super::super::TestOnlySpan;
use super::{span_lines, AstCollector};

impl AstCollector<'_> {
    pub(super) fn record_test_span(&mut self, attrs: &[Attribute], span: Span) {
        let (start, end) = span_lines(attrs, span);
        self.test_only_lines.extend(start..=end);
        let start_location = attrs
            .first()
            .map_or_else(|| span.start(), |attribute| attribute.span().start());
        let end_location = span.end();
        self.test_only_spans.push(TestOnlySpan {
            start_line: start_location.line.max(1),
            start_column: start_location.column,
            end_line: end_location.line.max(start_location.line.max(1)),
            end_column: end_location.column,
        });
    }
}
