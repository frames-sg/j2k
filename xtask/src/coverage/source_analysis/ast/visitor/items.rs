// SPDX-License-Identifier: MIT OR Apache-2.0

use syn::spanned::Spanned;
use syn::visit;
use syn::{ForeignItem, ImplItem, Item, TraitItem};

use super::super::super::graph::ReachKind;
use super::super::super::module_resolver::resolve_external_module;
use super::super::super::node_attrs;
use super::super::{AstCollector, ModuleEdge};
use super::{item_macro_label, macro_invocation_label};

pub(super) fn visit_item(collector: &mut AstCollector<'_>, item: &Item) {
    if matches!(item, Item::Verbatim(_)) {
        collector.record_error(format!(
            "unclassified verbatim Rust item in `{}`",
            collector.path
        ));
        return;
    }
    let attrs = node_attrs::item(item);
    let state = collector.item_state(attrs);
    if state.test_only {
        collector.record_test_span(attrs, item.span());
    }

    match item {
        Item::Mod(module) => {
            if let Some((_, items)) = &module.content {
                collector.visit_inline_module(
                    module,
                    items,
                    state.test_only,
                    state.required,
                    state.active,
                );
            } else {
                match resolve_external_module(
                    collector.root,
                    collector.path,
                    &collector.module_dir,
                    &collector.path_attr_dir,
                    module,
                    state.active,
                ) {
                    Ok(Some(path)) => collector.edges.push(ModuleEdge {
                        path,
                        kind: if collector.kind == ReachKind::TestTarget {
                            ReachKind::TestTarget
                        } else if state.test_only {
                            ReachKind::TestOnly
                        } else {
                            collector.kind
                        },
                        required_on_host: state.required,
                    }),
                    Ok(None) => {}
                    Err(error) => collector.record_error(error),
                }
            }
        }
        Item::Fn(function) if !state.test_only => {
            collector.record_function(
                &function.sig.ident,
                attrs,
                function.span(),
                function.block.span(),
                state.required,
            );
            collector.with_context(false, state.required, state.active, |collector| {
                collector.with_executable_context(true, |collector| {
                    visit::visit_item_fn(collector, function);
                });
            });
        }
        Item::Macro(item_macro) if !state.test_only => {
            collector.record_opaque_macro(
                &item_macro_label(item_macro),
                attrs,
                item.span(),
                state.required,
            );
            collector.with_context(false, state.required, state.active, |collector| {
                visit::visit_item_macro(collector, item_macro);
            });
        }
        _ if state.test_only => {}
        _ => collector.with_context(false, state.required, state.active, |collector| {
            collector.with_executable_context(false, |collector| {
                visit::visit_item(collector, item);
            });
        }),
    }
}

pub(super) fn visit_impl_item(collector: &mut AstCollector<'_>, item: &ImplItem) {
    if matches!(item, ImplItem::Verbatim(_)) {
        collector.record_error(format!(
            "unclassified verbatim impl item in `{}`",
            collector.path
        ));
        return;
    }
    let attrs = node_attrs::impl_item(item);
    let state = collector.item_state(attrs);
    if state.test_only {
        collector.record_test_span(attrs, item.span());
        return;
    }
    match item {
        ImplItem::Fn(function) => {
            collector.record_function(
                &function.sig.ident,
                attrs,
                function.span(),
                function.block.span(),
                state.required,
            );
            collector.with_context(false, state.required, state.active, |collector| {
                collector.with_executable_context(true, |collector| {
                    visit::visit_impl_item_fn(collector, function);
                });
            });
        }
        ImplItem::Macro(item_macro) => {
            collector.record_opaque_macro(
                &macro_invocation_label(&item_macro.mac),
                attrs,
                item.span(),
                state.required,
            );
            collector.with_context(false, state.required, state.active, |collector| {
                visit::visit_impl_item_macro(collector, item_macro);
            });
        }
        _ => collector.with_context(false, state.required, state.active, |collector| {
            collector.with_executable_context(false, |collector| {
                visit::visit_impl_item(collector, item);
            });
        }),
    }
}

pub(super) fn visit_trait_item(collector: &mut AstCollector<'_>, item: &TraitItem) {
    if matches!(item, TraitItem::Verbatim(_)) {
        collector.record_error(format!(
            "unclassified verbatim trait item in `{}`",
            collector.path
        ));
        return;
    }
    let attrs = node_attrs::trait_item(item);
    let state = collector.item_state(attrs);
    if state.test_only {
        collector.record_test_span(attrs, item.span());
        return;
    }
    match item {
        TraitItem::Fn(function) => {
            if let Some(default) = &function.default {
                collector.record_function(
                    &function.sig.ident,
                    attrs,
                    function.span(),
                    default.span(),
                    state.required,
                );
                collector.with_context(false, state.required, state.active, |collector| {
                    collector.with_executable_context(true, |collector| {
                        visit::visit_trait_item_fn(collector, function);
                    });
                });
            } else {
                collector.with_context(false, state.required, state.active, |collector| {
                    visit::visit_trait_item_fn(collector, function);
                });
            }
        }
        TraitItem::Macro(item_macro) => {
            collector.record_opaque_macro(
                &macro_invocation_label(&item_macro.mac),
                attrs,
                item.span(),
                state.required,
            );
            collector.with_context(false, state.required, state.active, |collector| {
                visit::visit_trait_item_macro(collector, item_macro);
            });
        }
        _ => collector.with_context(false, state.required, state.active, |collector| {
            collector.with_executable_context(false, |collector| {
                visit::visit_trait_item(collector, item);
            });
        }),
    }
}

pub(super) fn visit_foreign_item(collector: &mut AstCollector<'_>, item: &ForeignItem) {
    match node_attrs::foreign_item(item) {
        Ok(attrs) => {
            let state = collector.item_state(attrs);
            if state.test_only {
                collector.record_test_span(attrs, item.span());
                return;
            }
            if let ForeignItem::Macro(item_macro) = item {
                collector.record_opaque_macro(
                    &macro_invocation_label(&item_macro.mac),
                    attrs,
                    item.span(),
                    state.required,
                );
            }
            collector.with_context(false, state.required, state.active, |collector| {
                visit::visit_foreign_item(collector, item);
            });
        }
        Err(error) => collector.record_error(format!("{error} in `{}`", collector.path)),
    }
}
