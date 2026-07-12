// SPDX-License-Identifier: MIT OR Apache-2.0

//! Syntax-aware call-argument relationships for cross-module budget threading.

use syn::visit::{self, Visit};
use syn::{Block, Expr, ExprCall, ImplItem, Item};

pub(super) struct FunctionCallArguments {
    calls: Vec<ObservedCall>,
}

struct ObservedCall {
    name: String,
    identifiers: Vec<String>,
    mutable_identifiers: Vec<String>,
}

impl FunctionCallArguments {
    pub(super) fn parse_many(source_name: &str, sources: &[&str], function_name: &str) -> Self {
        let mut block_count = 0usize;
        let mut collector = CallCollector::default();
        for source in sources {
            let file = syn::parse_file(source)
                .unwrap_or_else(|error| panic!("parse {source_name} as Rust: {error}"));
            for item in &file.items {
                for block in callable_blocks(item, function_name) {
                    block_count += 1;
                    collector.visit_block(block);
                }
            }
        }
        assert_eq!(
            block_count, 1,
            "{source_name} must define exactly one function named {function_name}"
        );
        Self {
            calls: collector.calls,
        }
    }

    pub(super) fn assert_ident_argument(
        &self,
        label: &str,
        call_name: &str,
        identifier: &str,
        expected: usize,
    ) {
        self.assert_argument(label, call_name, identifier, expected, false);
    }

    pub(super) fn assert_mut_ident_argument(
        &self,
        label: &str,
        call_name: &str,
        identifier: &str,
        expected: usize,
    ) {
        self.assert_argument(label, call_name, identifier, expected, true);
    }

    fn assert_argument(
        &self,
        label: &str,
        call_name: &str,
        identifier: &str,
        expected: usize,
        mutable: bool,
    ) {
        let matching = self
            .calls
            .iter()
            .filter(|call| call.name == call_name)
            .collect::<Vec<_>>();
        assert_eq!(
            matching.len(),
            expected,
            "{label} must call {call_name} exactly {expected} times"
        );
        let observed = matching
            .iter()
            .filter(|call| {
                let identifiers = if mutable {
                    &call.mutable_identifiers
                } else {
                    &call.identifiers
                };
                identifiers.iter().any(|actual| actual == identifier)
            })
            .count();
        assert_eq!(
            observed,
            expected,
            "every {label} call to {call_name} must pass {}{identifier}",
            if mutable { "&mut " } else { "" }
        );
    }
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

#[derive(Default)]
struct CallCollector {
    calls: Vec<ObservedCall>,
}

impl<'ast> Visit<'ast> for CallCollector {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Expr::Path(path) = node.func.as_ref() {
            let name = path
                .path
                .segments
                .last()
                .expect("call path segment")
                .ident
                .to_string();
            let identifiers = node.args.iter().filter_map(path_identifier).collect();
            let mutable_identifiers = node
                .args
                .iter()
                .filter_map(mutable_reference_identifier)
                .collect();
            self.calls.push(ObservedCall {
                name,
                identifiers,
                mutable_identifiers,
            });
        }
        visit::visit_expr_call(self, node);
    }
}

fn path_identifier(expression: &Expr) -> Option<String> {
    let Expr::Path(path) = expression else {
        return None;
    };
    path.path.get_ident().map(ToString::to_string)
}

fn mutable_reference_identifier(expression: &Expr) -> Option<String> {
    let Expr::Reference(reference) = expression else {
        return None;
    };
    reference.mutability?;
    path_identifier(&reference.expr)
}
