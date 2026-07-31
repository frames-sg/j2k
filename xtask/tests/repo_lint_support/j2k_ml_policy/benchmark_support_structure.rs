// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    process::Command,
};

use syn::{
    spanned::Spanned,
    visit::{self, Visit},
    Expr, ExprCall, File,
};

use super::read;
use crate::repo_lint_support::repo_root;

fn parse(relative: &str) -> File {
    syn::parse_file(&read(relative)).unwrap_or_else(|error| panic!("parse {relative}: {error}"))
}

fn benchmark_targets() -> BTreeMap<String, (BTreeSet<String>, bool)> {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--no-deps", "--format-version=1"])
        .current_dir(repo_root())
        .output()
        .expect("run cargo metadata for benchmark targets");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("parse cargo metadata JSON");
    let package = metadata["packages"]
        .as_array()
        .expect("metadata packages")
        .iter()
        .find(|package| package["name"] == "j2k-ml")
        .expect("j2k-ml package in cargo metadata");

    package["targets"]
        .as_array()
        .expect("j2k-ml targets")
        .iter()
        .filter(|target| {
            target["kind"]
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "bench"))
        })
        .map(|target| {
            let name = target["name"].as_str().expect("benchmark target name");
            let required_features = target["required-features"]
                .as_array()
                .expect("benchmark required features")
                .iter()
                .map(|feature| feature.as_str().expect("feature string").to_owned())
                .collect();
            let test = target["test"].as_bool().expect("benchmark test flag");
            (name.to_owned(), (required_features, test))
        })
        .collect()
}

#[test]
fn j2k_ml_benchmark_targets_match_supported_backends() {
    let targets = benchmark_targets();
    assert_eq!(
        targets,
        BTreeMap::from([
            (
                "batch_decode".to_owned(),
                (BTreeSet::from(["cpu".to_owned()]), false),
            ),
            (
                "batch_decode_cuda".to_owned(),
                (BTreeSet::from(["cpu".to_owned(), "cuda".to_owned()]), false,),
            ),
            (
                "batch_decode_metal".to_owned(),
                (
                    BTreeSet::from(["cpu".to_owned(), "metal".to_owned()]),
                    false,
                ),
            ),
        ])
    );
}

struct MaterializationFinder {
    first_line: Option<usize>,
}

impl<'ast> Visit<'ast> for MaterializationFinder {
    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        if called_path(call).as_deref() == Some("materialize_workload") {
            let line = call.span().start().line;
            self.first_line = Some(self.first_line.map_or(line, |current| current.min(line)));
        }
        visit::visit_expr_call(self, call);
    }
}

struct SessionScopeVisitor<'a> {
    expected: &'a BTreeSet<&'a str>,
    active_materializations: Vec<Option<usize>>,
    observed: BTreeSet<String>,
    violations: Vec<String>,
}

impl<'a> SessionScopeVisitor<'a> {
    fn new(expected: &'a BTreeSet<&'a str>) -> Self {
        Self {
            expected,
            active_materializations: Vec::new(),
            observed: BTreeSet::new(),
            violations: Vec::new(),
        }
    }
}

impl<'ast> Visit<'ast> for SessionScopeVisitor<'_> {
    fn visit_expr_for_loop(&mut self, loop_expression: &'ast syn::ExprForLoop) {
        let mut finder = MaterializationFinder { first_line: None };
        finder.visit_block(&loop_expression.body);
        self.active_materializations.push(finder.first_line);
        visit::visit_expr_for_loop(self, loop_expression);
        self.active_materializations.pop();
    }

    fn visit_expr_call(&mut self, call: &'ast ExprCall) {
        let Some(constructor) = called_path(call) else {
            visit::visit_expr_call(self, call);
            return;
        };
        if !self.expected.contains(constructor.as_str()) {
            visit::visit_expr_call(self, call);
            return;
        }

        self.observed.insert(constructor.clone());
        let call_line = call.span().start().line;
        let materialization_line = self
            .active_materializations
            .iter()
            .rev()
            .flatten()
            .next()
            .copied();
        if materialization_line.is_none_or(|line| line >= call_line) {
            self.violations.push(format!(
                "{constructor} at line {call_line} is not after workload materialization in its loop"
            ));
        }
        visit::visit_expr_call(self, call);
    }
}

fn called_path(call: &ExprCall) -> Option<String> {
    let Expr::Path(path) = call.func.as_ref() else {
        return None;
    };
    let mut segments = path.path.segments.iter().rev();
    let function = segments.next()?.ident.to_string();
    let owner = segments.next().map(|segment| segment.ident.to_string());
    Some(owner.map_or(function.clone(), |owner| format!("{owner}::{function}")))
}

fn assert_sessions_follow_workload_materialization(relative: &str, expected: BTreeSet<&str>) {
    let file = parse(relative);
    let mut visitor = SessionScopeVisitor::new(&expected);
    visitor.visit_file(&file);
    assert!(
        visitor.violations.is_empty(),
        "{relative} session-scope violations: {:?}",
        visitor.violations
    );
    assert_eq!(
        visitor.observed,
        expected.into_iter().map(str::to_owned).collect(),
        "{relative} must exercise every persistent session family"
    );
}

#[test]
fn accelerator_benchmark_sessions_are_scoped_to_one_materialized_workload() {
    assert_sessions_follow_workload_materialization(
        "crates/j2k-ml/benches/batch_decode_cuda.rs",
        BTreeSet::from([
            "CpuBurnDecoder::new",
            "CudaBatchDecoder::with_options",
            "CudaUploadBurnDecoder::new",
        ]),
    );
    assert_sessions_follow_workload_materialization(
        "crates/j2k-ml/benches/batch_decode_metal.rs",
        BTreeSet::from([
            "CpuBurnDecoder::new",
            "MetalBatchDecoder::system_default_with_options",
            "MetalUploadBurnDecoder::system_default",
        ]),
    );
}
