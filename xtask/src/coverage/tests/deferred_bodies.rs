// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::support::TestRepository;
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;

fn changed(
    path: &str,
    lines: impl IntoIterator<Item = usize>,
) -> BTreeMap<String, BTreeSet<usize>> {
    BTreeMap::from([(path.to_string(), lines.into_iter().collect())])
}

fn report(path: &str, lines: impl IntoIterator<Item = (usize, u64)>) -> LcovReport {
    LcovReport {
        lines: BTreeMap::from([(path.to_string(), lines.into_iter().collect())]),
    }
}

#[test]
fn unpolled_multiline_async_requires_coverage_inside_its_body() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "\
pub fn build_future() {
    let future = async {
        changed();
    };
    let _future = future;
}
";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [2]),
        &report(path, [(2, 1), (5, 1)]),
        &index,
    )
    .unwrap();

    assert_eq!(
        result.changed_executable_bodies_without_covered_body,
        [format!("{path}::async@2")]
    );
    assert!(coverage_violations(CoverageLane::Host, &result)
        .iter()
        .any(|violation| violation.contains("async@2")));
}

#[test]
fn one_line_closure_requires_distinct_body_evidence() {
    assert_one_line_deferred_body_is_ambiguous(
        "pub fn build() { let _callback = || changed(); }\n",
        "closure@1",
    );
}

#[test]
fn one_line_async_requires_distinct_body_evidence() {
    assert_one_line_deferred_body_is_ambiguous(
        "pub fn build() { let _future = async { changed(); }; }\n",
        "async@1",
    );
}

#[test]
fn executed_multiline_deferred_bodies_accept_distinct_positive_records() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "\
pub fn build() {
    let callback = || {
        callback_body();
    };
    callback();
    let _future = async {
        async_body();
    };
}
";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [2, 6]),
        &report(path, [(2, 1), (3, 1), (5, 1), (6, 1), (7, 1)]),
        &index,
    )
    .unwrap();

    assert!(result
        .changed_executable_bodies_without_covered_body
        .is_empty());
    assert!(result
        .changed_deferred_bodies_without_distinct_line_evidence
        .is_empty());
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}

fn assert_one_line_deferred_body_is_ambiguous(source: &str, label: &str) {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [1]),
        &report(path, [(1, 1)]),
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.covered, 1);
    assert_eq!(
        result.changed_deferred_bodies_without_distinct_line_evidence,
        [format!("{path}::{label}")]
    );
    let violations = coverage_violations(CoverageLane::Host, &result);
    assert!(violations
        .iter()
        .any(|violation| violation.contains("line coverage cannot prove")));
}
