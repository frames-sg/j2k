// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::support::TestRepository;
use crate::coverage::compiler_regions::{CompilerRegionReport, SourceSpan};
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
        ..LcovReport::default()
    }
}

fn report_with_regions(
    path: &str,
    lines: impl IntoIterator<Item = (usize, u64)>,
    regions: &[(SourceSpan, u64)],
) -> LcovReport {
    LcovReport {
        lines: BTreeMap::from([(path.to_string(), lines.into_iter().collect())]),
        compiler_regions: CompilerRegionReport::for_test(path, regions),
    }
}

fn shared_body_span(index: &SourceIndex, path: &str) -> SourceSpan {
    let body = index.file(path).unwrap().executable_bodies[0].evidence;
    let crate::coverage::source_analysis::DeferredBodyEvidence::CompilerRegion(span) = body else {
        panic!("expected a same-line compiler-region body")
    };
    span
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
fn executed_one_line_closure_accepts_its_own_compiler_region() {
    assert_one_line_deferred_body(
        "pub fn build() { let callback = || changed(); callback(); }\n",
        "closure@1",
        Some(1),
    );
}

#[test]
fn unpolled_one_line_async_rejects_its_zero_count_compiler_region() {
    assert_one_line_deferred_body(
        "pub fn build() { let _future = async { changed(); }; }\n",
        "async@1",
        Some(0),
    );
}

#[test]
fn body_without_a_compiler_region_is_recorded_as_noninstrumentable() {
    assert_one_line_deferred_body(
        "pub fn build() { let _callback = || UnreachableError; }\n",
        "closure@1",
        None,
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
        .changed_deferred_bodies_without_covered_compiler_region
        .is_empty());
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}

fn assert_one_line_deferred_body(source: &str, label: &str, region_count: Option<u64>) {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let span = shared_body_span(&index, path);
    let regions = region_count.map_or_else(Vec::new, |count| vec![(span, count)]);
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [1]),
        &report_with_regions(path, [(1, 1)], &regions),
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.covered, 1);
    let violations = coverage_violations(CoverageLane::Host, &result);
    match region_count {
        Some(0) => {
            assert_eq!(
                result.changed_deferred_bodies_without_covered_compiler_region,
                [format!("{path}::{label}")]
            );
            assert!(violations
                .iter()
                .any(|violation| violation.contains("no covered region")));
        }
        Some(_) => {
            assert!(result
                .changed_deferred_bodies_without_covered_compiler_region
                .is_empty());
            assert!(result.compiler_noninstrumentable_deferred_bodies.is_empty());
            assert!(violations.is_empty());
        }
        None => {
            assert_eq!(
                result.compiler_noninstrumentable_deferred_bodies,
                [format!("{path}::{label}")]
            );
            assert!(violations.is_empty());
        }
    }
}
