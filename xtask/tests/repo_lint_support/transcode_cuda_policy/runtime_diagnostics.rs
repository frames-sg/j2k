// SPDX-License-Identifier: MIT OR Apache-2.0

use syn::visit::{self, Visit};
use syn::{Expr, ExprCall, ExprMethodCall, Lit};

use super::super::{assert_pattern_checks, PatternCheck};
use super::CudaTranscodeSources;

fn is_runtime_operation(method: &str) -> bool {
    method == "system_default"
        || method.starts_with("j2k_transcode_")
        || method.starts_with("encode_htj2k_")
        || method.starts_with("upload_htj2k_")
}

fn path_name(call: &ExprCall) -> Option<String> {
    let Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    Some(
        path.path
            .segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
    )
}

fn runtime_call_name(call: &ExprCall) -> Option<String> {
    let name = path_name(call)?;
    let operation = name.rsplit("::").next()?.to_string();
    is_runtime_operation(&operation).then_some(operation)
}

#[derive(Default)]
struct MethodNames(Vec<String>);

impl<'ast> Visit<'ast> for MethodNames {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(operation) = runtime_call_name(node) {
            self.0.push(operation);
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = node.method.to_string();
        if is_runtime_operation(&method) {
            self.0.push(method);
        }
        visit::visit_expr_method_call(self, node);
    }
}

#[derive(Default)]
struct MapperFacts {
    preserves_runtime_detail: bool,
    maps_to_static_kernel_error: bool,
}

impl<'ast> Visit<'ast> for MapperFacts {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        match path_name(node).as_deref() {
            Some("CudaTranscodeError::runtime") => {
                let operation_is_named = node.args.first().is_some_and(|argument| {
                    matches!(argument, Expr::Lit(literal) if matches!(&literal.lit, Lit::Str(value) if !value.value().is_empty()))
                });
                self.preserves_runtime_detail |= operation_is_named && node.args.len() == 2;
            }
            Some("CudaTranscodeError::Kernel") => self.maps_to_static_kernel_error = true,
            _ => {}
        }
        visit::visit_expr_call(self, node);
    }
}

#[derive(Default)]
struct RuntimeCoverage {
    all_operations: Vec<String>,
    detail_preserving_operations: Vec<String>,
    static_kernel_mappers: usize,
}

impl<'ast> Visit<'ast> for RuntimeCoverage {
    fn visit_expr_call(&mut self, node: &'ast ExprCall) {
        if let Some(operation) = runtime_call_name(node) {
            self.all_operations.push(operation);
        }
        visit::visit_expr_call(self, node);
    }

    fn visit_expr_method_call(&mut self, node: &'ast ExprMethodCall) {
        let method = node.method.to_string();
        if is_runtime_operation(&method) {
            self.all_operations.push(method.clone());
        }
        if method == "map_err" {
            let mut receiver = MethodNames::default();
            receiver.visit_expr(&node.receiver);
            let mut mapper = MapperFacts::default();
            for argument in &node.args {
                mapper.visit_expr(argument);
            }
            if !receiver.0.is_empty() {
                if mapper.preserves_runtime_detail {
                    self.detail_preserving_operations.extend(receiver.0);
                }
                self.static_kernel_mappers += usize::from(mapper.maps_to_static_kernel_error);
            }
        }
        visit::visit_expr_method_call(self, node);
    }
}

fn assert_every_runtime_operation_preserves_detail(sources: &CudaTranscodeSources) {
    let mut coverage = RuntimeCoverage::default();
    for source in &sources.files {
        let syntax = syn::parse_file(&source.production)
            .unwrap_or_else(|error| panic!("parse {}: {error}", source.relative));
        coverage.visit_file(&syntax);
    }
    coverage.all_operations.sort();
    coverage.detail_preserving_operations.sort();
    assert!(
        !coverage.all_operations.is_empty(),
        "CUDA transcode runtime operation inventory must not be empty"
    );
    assert_eq!(
        coverage.detail_preserving_operations, coverage.all_operations,
        "every CUDA transcode runtime call must map its backend error through `CudaTranscodeError::runtime(operation, error)`"
    );
    assert_eq!(
        coverage.static_kernel_mappers, 0,
        "CUDA runtime failures must not be collapsed into static Kernel messages"
    );
}

fn assert_runtime_error_contract(sources: &CudaTranscodeSources) {
    let production = sources.combined();
    let full = sources.full_combined();
    assert_pattern_checks(&[
        PatternCheck::new("CUDA runtime failure type", &production)
            .required(&[
                "Runtime(CudaRuntimeFailure)",
                "source: j2k_cuda_runtime::CudaError",
                "pub struct CudaRuntimeFailure",
                "operation: &'static str",
                "unavailable: bool",
                "source: Box<dyn std::error::Error + Send + Sync + 'static>",
                "source: Box::new(source)",
                "pub const fn operation(&self) -> &'static str",
                "pub const fn is_unavailable(&self) -> bool",
                "Self::Runtime(failure) if failure.is_unavailable()",
                "Self::Runtime(failure) => Some(failure)",
                "Some(self.source.as_ref())",
                "Self::backend(",
                "CudaTranscodeError::Runtime(failure)",
            ])
            .forbidden(&[
                "detail: String",
                "source.to_string()",
                "failure.to_string()",
                "Self::Backend(",
            ]),
        PatternCheck::new("CUDA context diagnostic classification", &production)
            .normalized_required(&[
                "CudaContext::system_default().map_err(|error| { CudaTranscodeError::runtime(\"CUDA context initialization\", error) })?",
            ]),
        PatternCheck::new("CUDA runtime diagnostic regression", &full).required(&[
            "runtime_failure_retains_operation_detail_and_error_source",
            "unavailable_runtime_failure_retains_diagnostic_and_allows_auto_recovery",
            "driver rejected launch",
            "downcast_ref::<CudaTranscodeError>()",
            "downcast_ref::<TestRuntimeError>()",
        ]),
    ]);
}

pub(super) fn assert_policy(sources: &CudaTranscodeSources) {
    assert!(
        include_str!("runtime_diagnostics.rs").lines().count() < 200,
        "CUDA runtime diagnostic policy must remain a focused module"
    );
    assert_every_runtime_operation_preserves_detail(sources);
    assert_runtime_error_contract(sources);
}
