// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::changed;
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;
use crate::coverage::tests::support::TestRepository;

#[test]
fn residual_unmeasured_lines_remain_explicit() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "pub struct Value {\n    pub field: u32,\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [1]),
        &LcovReport::default(),
        &index,
    )
    .unwrap();

    assert_eq!(result.unmeasured, [(path.to_string(), 1)]);
    assert_eq!(result.overall.measurable, 0);
}

#[test]
fn compiler_mapped_documentation_is_not_an_executable_changed_line() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "/// Updated public documentation.\npub fn value() -> u32 {\n    7\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(1, 0), (3, 1)]))]),
        ..LcovReport::default()
    };

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [1]),
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 0);
    assert!(result.uncovered.is_empty());
    assert!(result.changed_functions_without_covered_body.is_empty());
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}
