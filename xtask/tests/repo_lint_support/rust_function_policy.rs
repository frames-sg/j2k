// SPDX-License-Identifier: MIT OR Apache-2.0

//! Small syntax-aware helpers for repository source-relationship ratchets.

use syn::visit::{self, Visit};
use syn::{Block, Expr, ExprCall, ExprMacro, ExprMethodCall, ExprTry, ImplItem, Item, Lit, Type};

pub(crate) struct FunctionCalls {
    ordered: Vec<String>,
    propagated: Vec<String>,
}

impl FunctionCalls {
    pub(crate) fn parse(source_name: &str, source: &str, function_name: &str) -> Self {
        Self::parse_many(source_name, &[source], function_name)
    }

    pub(crate) fn parse_many(source_name: &str, sources: &[&str], function_name: &str) -> Self {
        assert!(
            !sources.is_empty(),
            "{source_name} source family must not be empty"
        );
        let mut block_count = 0usize;
        let mut collector = CallCollector::default();
        for source in sources {
            let file = syn::parse_file(source)
                .unwrap_or_else(|error| panic!("parse {source_name} as Rust: {error}"));
            for block in file
                .items
                .iter()
                .flat_map(|item| callable_blocks(item, function_name))
            {
                block_count += 1;
                collector.visit_block(block);
            }
        }
        assert_eq!(
            block_count, 1,
            "{source_name} must define exactly one function named {function_name}"
        );
        Self {
            ordered: collector.ordered,
            propagated: collector.propagated,
        }
    }

    pub(crate) fn assert_ordered(&self, label: &str, required: &[&str]) {
        assert!(
            !required.is_empty(),
            "{label} ordered call set must not be empty"
        );
        let mut search_start = 0usize;
        for expected in required {
            let relative = self.ordered[search_start..]
                .iter()
                .position(|actual| actual == expected)
                .unwrap_or_else(|| {
                    panic!(
                        "{label} must call {expected} after the prior required calls; observed {:?}",
                        self.ordered
                    )
                });
            search_start += relative + 1;
        }
    }

    pub(crate) fn assert_contains(&self, label: &str, required: &[&str]) {
        assert!(
            !required.is_empty(),
            "{label} required call set must not be empty"
        );
        for expected in required {
            assert!(
                self.ordered.iter().any(|actual| actual == expected),
                "{label} must call {expected}; observed {:?}",
                self.ordered
            );
        }
    }

    pub(crate) fn assert_propagated(&self, label: &str, required: &[&str]) {
        assert!(
            !required.is_empty(),
            "{label} propagated call set must not be empty"
        );
        for expected in required {
            assert!(
                self.propagated.iter().any(|actual| actual == expected),
                "{label} must propagate {expected} with `?`; observed {:?}",
                self.propagated
            );
        }
    }

    pub(crate) fn assert_absent(&self, label: &str, forbidden: &[&str]) {
        assert!(
            !forbidden.is_empty(),
            "{label} forbidden call set must not be empty"
        );
        for unexpected in forbidden {
            assert!(
                self.ordered.iter().all(|actual| actual != unexpected),
                "{label} must not call {unexpected}; observed {:?}",
                self.ordered
            );
        }
    }

    pub(crate) fn assert_count(&self, label: &str, call: &str, expected: usize) {
        let actual = self
            .ordered
            .iter()
            .filter(|observed| observed.as_str() == call)
            .count();
        assert_eq!(
            actual, expected,
            "{label} must call {call} exactly {expected} times; observed {:?}",
            self.ordered
        );
    }
}

pub(crate) fn assert_struct_field_type(
    source_name: &str,
    source: &str,
    struct_name: &str,
    field_name: &str,
    expected_type: &str,
) {
    let file = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("parse {source_name} as Rust: {error}"));
    let item = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Struct(item) if item.ident == struct_name => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{source_name} must define struct {struct_name}"));
    let field = item
        .fields
        .iter()
        .find(|field| {
            field
                .ident
                .as_ref()
                .is_some_and(|ident| ident == field_name)
        })
        .unwrap_or_else(|| panic!("{source_name}::{struct_name} must define field {field_name}"));
    assert_eq!(
        type_name(&field.ty),
        expected_type,
        "{source_name}::{struct_name}.{field_name} type"
    );
}

pub(crate) fn assert_usize_const(
    source_name: &str,
    source: &str,
    const_name: &str,
    expected: usize,
) {
    let file = syn::parse_file(source)
        .unwrap_or_else(|error| panic!("parse {source_name} as Rust: {error}"));
    let item = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Const(item) if item.ident == const_name => Some(item),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{source_name} must define const {const_name}"));
    assert_eq!(
        type_name(&item.ty),
        "usize",
        "{source_name}::{const_name} type"
    );
    let Expr::Lit(expression) = item.expr.as_ref() else {
        panic!("{source_name}::{const_name} must use an integer literal");
    };
    let Lit::Int(value) = &expression.lit else {
        panic!("{source_name}::{const_name} must use an integer literal");
    };
    let actual = value
        .base10_parse::<usize>()
        .unwrap_or_else(|error| panic!("parse {source_name}::{const_name}: {error}"));
    assert_eq!(actual, expected, "{source_name}::{const_name} value");
}

fn callable_blocks<'a>(item: &'a Item, function_name: &str) -> Vec<&'a Block> {
    match item {
        Item::Fn(function) if function.sig.ident == function_name => vec![&function.block],
        Item::Impl(item_impl) => item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method) if method.sig.ident == function_name => Some(&method.block),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn type_name(ty: &Type) -> String {
    match ty {
        Type::Path(path) => rust_path_name(&path.path),
        _ => panic!("policy field type must be a path type"),
    }
}

fn rust_path_name(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

#[derive(Default)]
struct CallCollector {
    ordered: Vec<String>,
    propagated: Vec<String>,
    try_depth: usize,
}

impl CallCollector {
    fn record(&mut self, name: String) {
        if self.try_depth > 0 {
            self.propagated.push(name.clone());
        }
        self.ordered.push(name);
    }
}

impl<'ast> Visit<'ast> for CallCollector {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let syn::Expr::Path(path) = node.func.as_ref() {
            self.record(rust_path_name(&path.path));
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        self.record(node.method.to_string());
        visit::visit_expr_method_call(self, node);
    }

    fn visit_expr_macro(&mut self, node: &'ast ExprMacro) {
        self.record(rust_path_name(&node.mac.path));
        visit::visit_expr_macro(self, node);
    }

    fn visit_expr_try(&mut self, node: &'ast ExprTry) {
        self.try_depth += 1;
        visit::visit_expr(self, &node.expr);
        self.try_depth -= 1;
    }
}

#[test]
fn rust_function_policy_stays_focused() {
    assert!(
        include_str!("rust_function_policy.rs").lines().count() < 275,
        "syntax-aware policy helper must stay below its focused-module ratchet"
    );
}
