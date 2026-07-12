// SPDX-License-Identifier: MIT OR Apache-2.0

use syn::spanned::Spanned;
use syn::visit;
use syn::{Arm, Expr, Local, Pat, StmtMacro, TypeMacro};

use super::super::super::node_attrs;
use super::super::AstCollector;
use super::macro_invocation_label;

pub(super) fn visit_local(collector: &mut AstCollector<'_>, local: &Local) {
    collector.visit_executable_node(&local.attrs, local.span(), |collector| {
        visit::visit_local(collector, local);
    });
}

pub(super) fn visit_pattern(collector: &mut AstCollector<'_>, pattern: &Pat) {
    match node_attrs::pattern(pattern) {
        Ok(attrs) => {
            let state = collector.item_state(attrs);
            if state.test_only {
                collector.record_test_span(attrs, pattern.span());
                return;
            }
            if let Pat::Macro(pattern_macro) = pattern {
                collector.record_opaque_macro(
                    &macro_invocation_label(&pattern_macro.mac),
                    attrs,
                    pattern.span(),
                    state.required,
                );
            }
            collector.with_context(false, state.required, state.active, |collector| {
                visit::visit_pat(collector, pattern);
            });
        }
        Err(error) => collector.record_error(format!("{error} in `{}`", collector.path)),
    }
}

pub(super) fn visit_arm(collector: &mut AstCollector<'_>, arm: &Arm) {
    collector.visit_executable_node(&arm.attrs, arm.span(), |collector| {
        visit::visit_arm(collector, arm);
    });
}

pub(super) fn visit_expression(collector: &mut AstCollector<'_>, expr: &Expr) {
    match node_attrs::expression(expr) {
        Ok(attrs) => {
            let state = collector.item_state(attrs);
            if state.test_only {
                collector.record_test_span(attrs, expr.span());
                return;
            }
            match expr {
                Expr::Closure(closure) => {
                    collector.record_closure(attrs, closure.span(), &closure.body, state.required);
                    collector.with_context(false, state.required, state.active, |collector| {
                        collector.with_executable_context(true, |collector| {
                            visit::visit_expr_closure(collector, closure);
                        });
                    });
                }
                Expr::Async(async_block) => {
                    collector.record_async_block(
                        attrs,
                        async_block.span(),
                        async_block.block.span(),
                        state.required,
                    );
                    collector.with_context(false, state.required, state.active, |collector| {
                        collector.with_executable_context(true, |collector| {
                            visit::visit_expr_async(collector, async_block);
                        });
                    });
                }
                Expr::Macro(expression_macro) => {
                    collector.record_opaque_macro(
                        &macro_invocation_label(&expression_macro.mac),
                        attrs,
                        expr.span(),
                        state.required,
                    );
                    collector.with_context(false, state.required, state.active, |collector| {
                        visit::visit_expr_macro(collector, expression_macro);
                    });
                }
                _ => {
                    if collector.context.executable {
                        collector.record_executable_span(expr.span(), state.required);
                    }
                    collector.with_context(false, state.required, state.active, |collector| {
                        visit::visit_expr(collector, expr);
                    });
                }
            }
        }
        Err(error) => collector.record_error(format!("{error} in `{}`", collector.path)),
    }
}

pub(super) fn visit_statement_macro(collector: &mut AstCollector<'_>, statement: &StmtMacro) {
    let state = collector.item_state(&statement.attrs);
    if state.test_only {
        collector.record_test_span(&statement.attrs, statement.span());
        return;
    }
    collector.record_opaque_macro(
        &macro_invocation_label(&statement.mac),
        &statement.attrs,
        statement.span(),
        state.required,
    );
    collector.with_context(false, state.required, state.active, |collector| {
        visit::visit_stmt_macro(collector, statement);
    });
}

pub(super) fn visit_type_macro(collector: &mut AstCollector<'_>, type_macro: &TypeMacro) {
    collector.record_opaque_macro(
        &macro_invocation_label(&type_macro.mac),
        &[],
        type_macro.span(),
        collector.context.item.required,
    );
    visit::visit_type_macro(collector, type_macro);
}
