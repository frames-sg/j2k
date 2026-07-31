// SPDX-License-Identifier: MIT OR Apache-2.0

//! Syntax-aware call-order checks for the few ordering policies that survive triage.

use syn::{
    visit::{self, Visit},
    Block, ExprCall, ExprMacro, ExprMethodCall, ExprTry, ImplItem, Item,
};

pub(crate) struct FunctionCalls {
    ordered: Vec<String>,
    propagated: Vec<String>,
}

impl FunctionCalls {
    pub(crate) fn parse(source_name: &str, source: &str, function_name: &str) -> Self {
        Self::parse_many(source_name, &[source], function_name)
    }

    fn parse_many(source_name: &str, sources: &[&str], function_name: &str) -> Self {
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
