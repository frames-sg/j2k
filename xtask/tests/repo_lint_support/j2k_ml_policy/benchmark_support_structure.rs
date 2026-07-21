// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    process::Command,
};

use syn::{
    punctuated::Punctuated,
    spanned::Spanned,
    visit::{self, Visit},
    Attribute, Expr, ExprCall, File, Item, Meta, Token,
};

use super::read;
use crate::repo_lint_support::repo_root;

fn parse(relative: &str) -> File {
    syn::parse_file(&read(relative)).unwrap_or_else(|error| panic!("parse {relative}: {error}"))
}

fn external_modules(file: &File) -> BTreeSet<String> {
    file.items
        .iter()
        .filter_map(|item| match item {
            Item::Mod(item) if item.content.is_none() => Some(item.ident.to_string()),
            _ => None,
        })
        .collect()
}

fn declared_types(file: &File) -> BTreeSet<String> {
    file.items
        .iter()
        .filter_map(|item| match item {
            Item::Enum(item) => Some(item.ident.to_string()),
            Item::Struct(item) => Some(item.ident.to_string()),
            Item::Trait(item) => Some(item.ident.to_string()),
            _ => None,
        })
        .collect()
}

fn declared_functions(file: &File) -> BTreeSet<String> {
    file.items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(item) => Some(item.sig.ident.to_string()),
            _ => None,
        })
        .collect()
}

fn declared_consts(file: &File) -> BTreeSet<String> {
    file.items
        .iter()
        .filter_map(|item| match item {
            Item::Const(item) => Some(item.ident.to_string()),
            _ => None,
        })
        .collect()
}

#[derive(Default)]
struct DeadCodeSuppressionVisitor {
    locations: Vec<usize>,
}

impl<'ast> Visit<'ast> for DeadCodeSuppressionVisitor {
    fn visit_attribute(&mut self, attribute: &'ast Attribute) {
        let Meta::List(list) = &attribute.meta else {
            return;
        };
        if !(list.path.is_ident("allow") || list.path.is_ident("expect")) {
            return;
        }
        let arguments = attribute
            .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
            .unwrap_or_else(|error| panic!("parse lint attribute: {error}"));
        if arguments
            .iter()
            .any(|argument| matches!(argument, Meta::Path(path) if path.is_ident("dead_code")))
        {
            self.locations.push(attribute.span().start().line);
        }
    }
}

fn assert_no_dead_code_suppressions(relative: &str, file: &File) {
    let mut visitor = DeadCodeSuppressionVisitor::default();
    visitor.visit_file(file);
    assert!(
        visitor.locations.is_empty(),
        "{relative} must not suppress dead_code at lines {:?}",
        visitor.locations
    );
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
fn j2k_ml_benchmark_support_has_focused_ownership() {
    let support_paths = [
        "crates/j2k-ml/benches/support/mod.rs",
        "crates/j2k-ml/benches/support/decode_case.rs",
        "crates/j2k-ml/benches/support/fixture.rs",
        "crates/j2k-ml/benches/support/input_selection.rs",
        "crates/j2k-ml/benches/support/process_policy.rs",
        "crates/j2k-ml/benches/support/workload.rs",
        "crates/j2k-ml/benches/support/workload_catalog.rs",
    ];
    let support = support_paths
        .iter()
        .map(|path| (*path, parse(path)))
        .collect::<BTreeMap<_, _>>();
    for (path, file) in &support {
        assert_no_dead_code_suppressions(path, file);
    }

    assert_eq!(
        external_modules(&support[support_paths[0]]),
        BTreeSet::from([
            "decode_case".to_owned(),
            "fixture".to_owned(),
            "input_selection".to_owned(),
            "process_policy".to_owned(),
            "workload".to_owned(),
            "workload_catalog".to_owned(),
        ])
    );
    assert!(
        declared_functions(&support[support_paths[1]]).is_superset(&BTreeSet::from([
            "requests".to_owned(),
            "decoded_pixels_per_batch".to_owned(),
            "require_prepared_success".to_owned(),
        ]))
    );
    assert!(declared_functions(&support[support_paths[2]]).contains("encode_ht_fixture"));
    assert_eq!(
        declared_types(&support[support_paths[3]]),
        BTreeSet::from(["InputMode".to_owned()])
    );
    assert_eq!(
        declared_types(&support[support_paths[4]]),
        BTreeSet::from(["ProcessMode".to_owned()])
    );
    assert!(
        declared_types(&support[support_paths[5]]).is_superset(&BTreeSet::from([
            "Workload".to_owned(),
            "WorkloadSpec".to_owned(),
        ]))
    );
    assert!(declared_functions(&support[support_paths[5]]).contains("materialize_workload"));
    assert!(declared_functions(&support[support_paths[6]]).contains("workload_specs"));
    assert_eq!(
        declared_consts(&support[support_paths[6]]),
        BTreeSet::from(["BATCH_SIZES".to_owned(), "LOW_BATCH_SIZES".to_owned()])
    );

    let metal = parse("crates/j2k-ml/benches/batch_decode_metal.rs");
    let instrumentation = parse("crates/j2k-ml/benches/batch_decode_metal/instrumentation.rs");
    assert!(external_modules(&metal).contains("instrumentation"));
    assert!(
        declared_functions(&instrumentation).contains("ensure_criterion_instrumentation_disabled")
    );
    assert!(!declared_types(&instrumentation).contains("ProcessMode"));
    assert_no_dead_code_suppressions("Metal benchmark", &metal);
    assert_no_dead_code_suppressions("Metal instrumentation", &instrumentation);

    let benchmark_input_test = parse("crates/j2k-ml/tests/benchmark_inputs.rs");
    assert_eq!(
        external_modules(&benchmark_input_test),
        BTreeSet::from([
            "decode_case".to_owned(),
            "fixture".to_owned(),
            "input_selection".to_owned(),
            "workload".to_owned(),
        ]),
        "benchmark input tests must load only reusable construction modules"
    );

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
            "CudaBurnDecoder::new",
        ]),
    );
    assert_sessions_follow_workload_materialization(
        "crates/j2k-ml/benches/batch_decode_metal.rs",
        BTreeSet::from([
            "CpuBurnDecoder::new",
            "MetalBatchDecoder::system_default_with_options",
            "MetalBurnDecoder::system_default",
        ]),
    );
}
