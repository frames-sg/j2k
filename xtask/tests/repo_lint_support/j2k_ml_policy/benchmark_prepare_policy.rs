// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use syn::{
    spanned::Spanned,
    visit::{self, Visit},
    Expr, ExprCall, ExprMethodCall, Local, Pat,
};

use super::read;

#[derive(Default)]
struct PrepareClosureEvidence {
    prepare_calls: usize,
    prepared_bindings: BTreeSet<String>,
    validated_bindings: BTreeSet<String>,
}

impl<'ast> Visit<'ast> for PrepareClosureEvidence {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == "prepare" {
            self.prepare_calls += 1;
        }
        visit::visit_expr_method_call(self, call);
    }

    fn visit_local(&mut self, local: &'ast Local) {
        if let (Pat::Ident(binding), Some(initializer)) = (&local.pat, &local.init) {
            let mut prepare = ContainsPrepare::default();
            prepare.visit_expr(&initializer.expr);
            if prepare.found {
                self.prepared_bindings.insert(binding.ident.to_string());
            }
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if matches!(
            call.func.as_ref(),
            Expr::Path(path)
                if path.path.segments.last().is_some_and(|segment| {
                    segment.ident == "require_prepared_success"
                })
        ) {
            if let Some(binding) = call.args.first().and_then(referenced_binding) {
                self.validated_bindings.insert(binding);
            }
        }
        visit::visit_expr_call(self, call);
    }
}

#[derive(Default)]
struct ContainsPrepare {
    found: bool,
}

impl<'ast> Visit<'ast> for ContainsPrepare {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        self.found |= call.method == "prepare";
        visit::visit_expr_method_call(self, call);
    }
}

fn referenced_binding(expression: &Expr) -> Option<String> {
    let expression = match expression {
        Expr::Reference(reference) => reference.expr.as_ref(),
        expression => expression,
    };
    let Expr::Path(path) = expression else {
        return None;
    };
    path.path.get_ident().map(ToString::to_string)
}

#[derive(Default)]
struct TimedPrepareVisitor {
    prepare_iterations: usize,
    unvalidated_lines: Vec<usize>,
}

impl<'ast> Visit<'ast> for TimedPrepareVisitor {
    fn visit_expr_method_call(&mut self, call: &'ast ExprMethodCall) {
        if call.method == "iter" {
            for argument in &call.args {
                let Expr::Closure(closure) = argument else {
                    continue;
                };
                let mut evidence = PrepareClosureEvidence::default();
                evidence.visit_expr(&closure.body);
                if evidence.prepare_calls > 0 {
                    self.prepare_iterations += 1;
                    if evidence.prepared_bindings.len() != evidence.prepare_calls
                        || !evidence
                            .prepared_bindings
                            .is_subset(&evidence.validated_bindings)
                    {
                        self.unvalidated_lines.push(closure.span().start().line);
                    }
                }
            }
        }
        visit::visit_expr_method_call(self, call);
    }
}

#[test]
fn gpu_prepare_benchmark_iterations_reject_indexed_errors() {
    for relative in [
        "crates/j2k-ml/benches/batch_decode_cuda.rs",
        "crates/j2k-ml/benches/batch_decode_metal.rs",
    ] {
        let syntax = syn::parse_file(&read(relative))
            .unwrap_or_else(|error| panic!("parse {relative}: {error}"));
        let mut visitor = TimedPrepareVisitor::default();
        visitor.visit_file(&syntax);
        assert_eq!(
            visitor.prepare_iterations, 1,
            "{relative} must contain exactly one timed preparation iteration"
        );
        assert!(
            visitor.unvalidated_lines.is_empty(),
            "{relative} timed preparation must reject indexed errors at lines {:?}",
            visitor.unvalidated_lines
        );
    }
}

#[test]
fn timed_prepare_validation_must_use_the_prepared_value() {
    let syntax = syn::parse_file(
        r"
fn benchmark(bencher: &mut Bencher, decoder: &mut Decoder, inputs: Inputs, other: Prepared) {
    bencher.iter(|| {
        let prepared = decoder.prepare(inputs).unwrap();
        require_prepared_success(&other);
        black_box(prepared)
    });
}
",
    )
    .expect("parse spoofed preparation benchmark");
    let mut visitor = TimedPrepareVisitor::default();
    visitor.visit_file(&syntax);
    assert_eq!(visitor.unvalidated_lines.len(), 1);
}
