// SPDX-License-Identifier: MIT OR Apache-2.0

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::{Attribute, Expr};

use super::super::{DeferredBodyEvidence, ExecutableBodySpan, OpaqueMacroKind, OpaqueMacroSpan};
use super::{span_lines, AstCollector};

impl AstCollector<'_> {
    pub(super) fn record_executable_span(&mut self, span: Span, required: bool) {
        if !required {
            return;
        }
        let (start, end) = span_lines(&[], span);
        self.record_executable_lines(start, end);
    }

    fn record_executable_lines(&mut self, start: usize, end: usize) {
        self.executable_lines.extend(start..=end);
    }

    pub(super) fn record_closure(
        &mut self,
        attrs: &[Attribute],
        item_span: Span,
        body: &Expr,
        required: bool,
    ) {
        self.record_deferred_body(
            "closure",
            attrs,
            item_span,
            body.span(),
            matches!(body, Expr::Block(_)),
            required,
        );
    }

    pub(super) fn record_async_block(
        &mut self,
        attrs: &[Attribute],
        item_span: Span,
        body_span: Span,
        required: bool,
    ) {
        self.record_deferred_body("async", attrs, item_span, body_span, true, required);
    }

    fn record_deferred_body(
        &mut self,
        kind: &str,
        attrs: &[Attribute],
        item_span: Span,
        body_span: Span,
        body_is_block: bool,
        required: bool,
    ) {
        let (start, end) = span_lines(attrs, item_span);
        let (body_start, body_end) = span_lines(&[], body_span);
        let evidence_start = body_start.max(start.saturating_add(1));
        let evidence_end = if body_is_block {
            body_end.saturating_sub(1)
        } else {
            body_end
        };
        let evidence = if evidence_start <= evidence_end {
            DeferredBodyEvidence::DistinctLines {
                start: evidence_start,
                end: evidence_end,
            }
        } else {
            DeferredBodyEvidence::SharedCreationLine
        };
        self.executable_bodies.push(ExecutableBodySpan {
            label: format!("{kind}@{start}"),
            start,
            end,
            evidence,
            required_on_host: required,
        });
        self.record_executable_span(body_span, required);
    }

    pub(super) fn record_opaque_macro(
        &mut self,
        label: &str,
        attrs: &[Attribute],
        span: Span,
        required: bool,
    ) {
        let (start, end) = span_lines(attrs, span);
        self.opaque_macros.push(OpaqueMacroSpan {
            label: format!("{label}@{start}"),
            start,
            end,
            kind: if label.starts_with("opaque-macro-invocation") {
                OpaqueMacroKind::Invocation
            } else {
                OpaqueMacroKind::Definition
            },
            required_on_host: required,
        });
        if required {
            self.record_executable_lines(start, end);
        }
    }

    pub(super) fn visit_executable_node(
        &mut self,
        attrs: &[Attribute],
        span: Span,
        visit_node: impl FnOnce(&mut Self),
    ) {
        let state = self.item_state(attrs);
        if state.test_only {
            self.record_test_span(attrs, span);
            return;
        }
        if self.context.executable {
            self.record_executable_span(span, state.required);
        }
        self.with_context(false, state.required, state.active, visit_node);
    }

    pub(super) fn with_executable_context(&mut self, executable: bool, f: impl FnOnce(&mut Self)) {
        let previous = self.context.executable;
        self.context.executable = executable;
        f(self);
        self.context.executable = previous;
    }
}
