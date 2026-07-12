// SPDX-License-Identifier: MIT OR Apache-2.0

use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    Arm, Attribute, BareFnArg, BareVariadic, Expr, Field, FieldPat, FieldValue, FnArg, ForeignItem,
    GenericParam, ImplItem, Item, ItemMacro, Local, Macro, Pat, StmtMacro, TraitItem, TypeMacro,
    Variadic, Variant,
};

use super::super::node_attrs;
use super::{attribute_location, is_coverage_attribute, AstCollector};

mod items;
mod runtime;

impl<'ast> Visit<'ast> for AstCollector<'_> {
    fn visit_attribute(&mut self, attribute: &'ast Attribute) {
        if is_coverage_attribute(attribute)
            && !self
                .classified_attributes
                .contains(&attribute_location(attribute))
        {
            self.record_error(format!(
                "unclassified cfg/test attribute in `{}` at line {}",
                self.path,
                attribute.span().start().line
            ));
        }
        // Attribute payloads are compile-time metadata. Descending into them
        // would misclassify literals in `#[doc = "..."]` and similar metadata
        // as runtime expressions whenever the enclosing item is executable.
    }

    fn visit_item(&mut self, item: &'ast Item) {
        items::visit_item(self, item);
    }

    fn visit_impl_item(&mut self, item: &'ast ImplItem) {
        items::visit_impl_item(self, item);
    }

    fn visit_trait_item(&mut self, item: &'ast TraitItem) {
        items::visit_trait_item(self, item);
    }

    fn visit_local(&mut self, local: &'ast Local) {
        runtime::visit_local(self, local);
    }

    fn visit_fn_arg(&mut self, argument: &'ast FnArg) {
        self.visit_attributed_node(
            node_attrs::function_argument(argument),
            argument.span(),
            |collector| visit::visit_fn_arg(collector, argument),
        );
    }

    fn visit_variadic(&mut self, variadic: &'ast Variadic) {
        self.visit_attributed_node(&variadic.attrs, variadic.span(), |collector| {
            visit::visit_variadic(collector, variadic);
        });
    }

    fn visit_bare_fn_arg(&mut self, argument: &'ast BareFnArg) {
        self.visit_attributed_node(&argument.attrs, argument.span(), |collector| {
            visit::visit_bare_fn_arg(collector, argument);
        });
    }

    fn visit_bare_variadic(&mut self, variadic: &'ast BareVariadic) {
        self.visit_attributed_node(&variadic.attrs, variadic.span(), |collector| {
            visit::visit_bare_variadic(collector, variadic);
        });
    }

    fn visit_generic_param(&mut self, parameter: &'ast GenericParam) {
        self.visit_attributed_node(
            node_attrs::generic_parameter(parameter),
            parameter.span(),
            |collector| visit::visit_generic_param(collector, parameter),
        );
    }

    fn visit_pat(&mut self, pattern: &'ast Pat) {
        runtime::visit_pattern(self, pattern);
    }

    fn visit_field(&mut self, field: &'ast Field) {
        self.visit_attributed_node(&field.attrs, field.span(), |collector| {
            visit::visit_field(collector, field);
        });
    }

    fn visit_variant(&mut self, variant: &'ast Variant) {
        self.visit_attributed_node(&variant.attrs, variant.span(), |collector| {
            visit::visit_variant(collector, variant);
        });
    }

    fn visit_arm(&mut self, arm: &'ast Arm) {
        runtime::visit_arm(self, arm);
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        runtime::visit_expression(self, expr);
    }

    fn visit_field_value(&mut self, field: &'ast FieldValue) {
        self.visit_attributed_node(&field.attrs, field.span(), |collector| {
            visit::visit_field_value(collector, field);
        });
    }

    fn visit_field_pat(&mut self, field: &'ast FieldPat) {
        self.visit_attributed_node(&field.attrs, field.span(), |collector| {
            visit::visit_field_pat(collector, field);
        });
    }

    fn visit_stmt_macro(&mut self, statement: &'ast StmtMacro) {
        runtime::visit_statement_macro(self, statement);
    }

    fn visit_type_macro(&mut self, type_macro: &'ast TypeMacro) {
        runtime::visit_type_macro(self, type_macro);
    }

    fn visit_foreign_item(&mut self, item: &'ast ForeignItem) {
        items::visit_foreign_item(self, item);
    }
}

fn item_macro_label(item: &ItemMacro) -> String {
    if item.mac.path.is_ident("macro_rules") {
        return item.ident.as_ref().map_or_else(
            || "opaque-macro-definition".to_string(),
            |ident| format!("opaque-macro-definition:{ident}"),
        );
    }
    macro_invocation_label(&item.mac)
}

fn macro_invocation_label(value: &Macro) -> String {
    let path = value
        .path
        .segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::");
    if path.is_empty() {
        "opaque-macro-invocation".to_string()
    } else {
        format!("opaque-macro-invocation:{path}")
    }
}
