// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use proc_macro2::Span;
use syn::spanned::Spanned;
use syn::visit::Visit;
use syn::{Attribute, Item};

use super::cfg_eval::{attributes_state, CoverageCfgContext};
use super::graph::ReachKind;
use super::module_resolver::{has_module_path_attribute, source_module_dir, source_parent_dir};
use super::{ExecutableBodySpan, FunctionSpan, OpaqueMacroSpan, TestOnlySpan};

mod executable;
mod test_spans;
mod visitor;

#[derive(Debug)]
pub(super) struct ModuleEdge {
    pub(super) path: String,
    pub(super) kind: ReachKind,
    pub(super) required_on_host: bool,
}

#[derive(Debug)]
pub(super) struct ParsedSource {
    pub(super) edges: Vec<ModuleEdge>,
    pub(super) test_only_lines: BTreeSet<usize>,
    pub(super) test_only_spans: Vec<TestOnlySpan>,
    pub(super) executable_lines: BTreeSet<usize>,
    pub(super) functions: Vec<FunctionSpan>,
    pub(super) executable_bodies: Vec<ExecutableBodySpan>,
    pub(super) opaque_macros: Vec<OpaqueMacroSpan>,
}

pub(super) fn validate_source(path: &str, source: &str) -> Result<(), String> {
    syn::parse_file(source)
        .map(|_| ())
        .map_err(|error| format!("failed to parse Rust source `{path}` for coverage: {error}"))
}

pub(super) fn analyze_source(
    root: &Path,
    path: &str,
    source: &str,
    kind: ReachKind,
    required_on_host: bool,
    cfg: &CoverageCfgContext,
) -> Result<ParsedSource, String> {
    let file = syn::parse_file(source)
        .map_err(|error| format!("failed to parse Rust source `{path}` for coverage: {error}"))?;
    let file_state = attributes_state(&file.attrs, cfg)
        .map_err(|error| format!("failed to classify crate attributes in `{path}`: {error}"))?;
    let classified_attributes = file
        .attrs
        .iter()
        .filter(|attribute| is_coverage_attribute(attribute))
        .map(attribute_location)
        .collect();
    let file_test_only =
        matches!(kind, ReachKind::TestOnly | ReachKind::TestTarget) || file_state.implies_test;
    let file_required = required_on_host && file_state.active;
    let mut collector = AstCollector {
        root,
        path,
        module_dir: source_module_dir(path)?,
        path_attr_dir: source_parent_dir(path),
        kind,
        context: TraversalContext {
            item: ItemState {
                test_only: file_test_only,
                required: file_required,
                active: file_required,
            },
            executable: false,
        },
        cfg,
        edges: Vec::new(),
        test_only_lines: BTreeSet::new(),
        test_only_spans: Vec::new(),
        executable_lines: BTreeSet::new(),
        functions: Vec::new(),
        executable_bodies: Vec::new(),
        opaque_macros: Vec::new(),
        classified_attributes,
        error: None,
    };
    collector.visit_file(&file);
    if file_test_only {
        collector.test_only_lines.extend(1..=source.lines().count());
    }
    if let Some(error) = collector.error {
        return Err(error);
    }
    collector
        .functions
        .sort_by_key(|span| (span.start, span.end));
    collector
        .executable_bodies
        .sort_by_key(|span| (span.start, span.end));
    collector
        .opaque_macros
        .sort_by_key(|span| (span.start, span.end));
    Ok(ParsedSource {
        edges: collector.edges,
        test_only_lines: collector.test_only_lines,
        test_only_spans: collector.test_only_spans,
        executable_lines: collector.executable_lines,
        functions: collector.functions,
        executable_bodies: collector.executable_bodies,
        opaque_macros: collector.opaque_macros,
    })
}

#[derive(Clone, Copy)]
struct ItemState {
    test_only: bool,
    required: bool,
    active: bool,
}

#[derive(Clone, Copy)]
struct TraversalContext {
    item: ItemState,
    executable: bool,
}

struct AstCollector<'a> {
    root: &'a Path,
    path: &'a str,
    module_dir: PathBuf,
    path_attr_dir: PathBuf,
    kind: ReachKind,
    context: TraversalContext,
    cfg: &'a CoverageCfgContext,
    edges: Vec<ModuleEdge>,
    test_only_lines: BTreeSet<usize>,
    test_only_spans: Vec<TestOnlySpan>,
    executable_lines: BTreeSet<usize>,
    functions: Vec<FunctionSpan>,
    executable_bodies: Vec<ExecutableBodySpan>,
    opaque_macros: Vec<OpaqueMacroSpan>,
    classified_attributes: BTreeSet<(usize, usize, usize, usize)>,
    error: Option<String>,
}

impl AstCollector<'_> {
    fn item_state(&mut self, attrs: &[Attribute]) -> ItemState {
        self.classified_attributes.extend(
            attrs
                .iter()
                .filter(|attribute| is_coverage_attribute(attribute))
                .map(attribute_location),
        );
        match attributes_state(attrs, self.cfg) {
            Ok(state) => ItemState {
                test_only: self.context.item.test_only || state.implies_test,
                required: self.context.item.required && state.active,
                active: self.context.item.active && state.active,
            },
            Err(error) => {
                self.record_error(format!(
                    "failed to classify cfg attributes in `{}`: {error}",
                    self.path
                ));
                ItemState {
                    test_only: false,
                    required: false,
                    active: false,
                }
            }
        }
    }

    fn record_error(&mut self, error: String) {
        self.error.get_or_insert(error);
    }

    fn record_function(
        &mut self,
        name: &syn::Ident,
        attrs: &[Attribute],
        item_span: Span,
        body_span: Span,
        required: bool,
    ) {
        let (start, end) = span_lines(attrs, item_span);
        let (body_start, body_end) = span_lines(&[], body_span);
        self.functions.push(FunctionSpan {
            name: name.to_string(),
            start,
            end,
            body_start,
            body_end,
            required_on_host: required,
        });
    }

    fn visit_attributed_node(
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
        self.with_context(false, state.required, state.active, visit_node);
    }

    fn with_context(
        &mut self,
        inherited_test: bool,
        required: bool,
        active: bool,
        f: impl FnOnce(&mut Self),
    ) {
        let previous = self.context.item;
        self.context.item = ItemState {
            test_only: inherited_test,
            required,
            active,
        };
        f(self);
        self.context.item = previous;
    }

    fn visit_inline_module(
        &mut self,
        module: &syn::ItemMod,
        items: &[Item],
        test_only: bool,
        required: bool,
        active: bool,
    ) {
        match has_module_path_attribute(&module.attrs) {
            Ok(true) => {
                self.record_error(format!(
                    "inline module `{}` in `{}` must not have a path attribute",
                    module.ident, self.path
                ));
                return;
            }
            Ok(false) => {}
            Err(error) => {
                self.record_error(format!("invalid module path in `{}`: {error}", self.path));
                return;
            }
        }
        let previous_module_dir = self.module_dir.clone();
        let previous_path_attr_dir = self.path_attr_dir.clone();
        self.module_dir.push(module.ident.to_string());
        self.path_attr_dir = self.module_dir.clone();
        self.with_context(test_only, required, active, |collector| {
            collector.with_executable_context(false, |collector| {
                for nested in items {
                    collector.visit_item(nested);
                }
            });
        });
        self.module_dir = previous_module_dir;
        self.path_attr_dir = previous_path_attr_dir;
    }
}

fn span_lines(attrs: &[Attribute], span: Span) -> (usize, usize) {
    let start = attrs
        .first()
        .map_or_else(
            || span.start().line,
            |attribute| attribute.span().start().line,
        )
        .max(1);
    let end = span.end().line.max(start);
    (start, end)
}

fn is_coverage_attribute(attribute: &Attribute) -> bool {
    attribute.path().is_ident("cfg")
        || attribute.path().is_ident("cfg_attr")
        || attribute.path().is_ident("test")
        || attribute.path().is_ident("bench")
}

fn attribute_location(attribute: &Attribute) -> (usize, usize, usize, usize) {
    let start = attribute.span().start();
    let end = attribute.span().end();
    (start.line, start.column, end.line, end.column)
}
