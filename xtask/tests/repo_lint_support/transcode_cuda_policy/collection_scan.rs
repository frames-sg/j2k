// SPDX-License-Identifier: MIT OR Apache-2.0

use syn::visit::{self, Visit};
use syn::{ExprMethodCall, ImplItemFn, ItemFn};

use super::CudaTranscodeSources;

pub(super) fn assert_no_infallible_collects(sources: &CudaTranscodeSources) {
    let mut owners = Vec::new();
    for source in &sources.files {
        let syntax = syn::parse_file(&source.production)
            .unwrap_or_else(|error| panic!("parse {}: {error}", source.relative));
        let mut collector = CollectOwnerVisitor::default();
        collector.visit_file(&syntax);
        owners.extend(collector.owners);
    }
    assert!(
        owners.is_empty(),
        "CUDA transcode production `.collect()` owners {owners:?} must use explicit fallible reservation and ordered pushes"
    );
}

#[derive(Default)]
struct CollectOwnerVisitor {
    current_function: Option<String>,
    owners: Vec<String>,
}

impl CollectOwnerVisitor {
    fn visit_function(&mut self, name: String, block: &syn::Block) {
        let prior = self.current_function.replace(name);
        self.visit_block(block);
        self.current_function = prior;
    }
}

impl<'ast> Visit<'ast> for CollectOwnerVisitor {
    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        self.visit_function(node.sig.ident.to_string(), &node.block);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        self.visit_function(node.sig.ident.to_string(), &node.block);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        if node.method == "collect" {
            self.owners.push(
                self.current_function
                    .clone()
                    .unwrap_or_else(|| "<module scope>".to_string()),
            );
        }
        visit::visit_expr_method_call(self, node);
    }
}
