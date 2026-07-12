// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::support::TestRepository;
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;

fn changed(path: &str, line: usize) -> BTreeMap<String, BTreeSet<usize>> {
    BTreeMap::from([(path.to_string(), BTreeSet::from([line]))])
}

#[test]
fn changed_uncalled_closure_requires_coverage_in_its_own_body() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "\
pub fn build_callback() {
    let callback = || {
        let _changed = 1;
    };
    let _callback = callback;
}
";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(5, 1)]))]),
    };

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, 3),
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.uncovered, [(path.to_string(), 3)]);
    assert_eq!(result.unmeasured, [(path.to_string(), 3)]);
    assert!(result.changed_functions_without_covered_body.is_empty());
    assert_eq!(
        result.changed_executable_bodies_without_covered_body,
        [format!("{path}::closure@2")]
    );
    assert!(coverage_violations(CoverageLane::Host, &result)
        .iter()
        .any(|violation| violation.contains("closure@2")));
}

#[test]
fn changed_opaque_macro_definition_and_invocation_fail_closed() {
    let cases = [
        (
            "macro_rules! generated {\n    () => { pub fn value() -> u32 { 7 } };\n}\n",
            2,
            "opaque-macro-definition:generated@1",
        ),
        (
            "pub fn invoke() {\n    generated!(\n        changed\n    );\n}\n",
            3,
            "opaque-macro-invocation:generated@2",
        ),
    ];

    for (source, changed_line, expected) in cases {
        let repository = TestRepository::new();
        let path = "crates/example/src/lib.rs";
        repository.write(path, source);
        let index = SourceIndex::single(path, source).unwrap();
        let result = evaluate_changed_coverage(
            CoverageLane::Host,
            repository.root(),
            &changed(path, changed_line),
            &LcovReport::default(),
            &index,
        )
        .unwrap();

        assert_eq!(
            result.changed_opaque_macros,
            [format!("{path}::{expected}")]
        );
        assert!(coverage_violations(CoverageLane::Host, &result)
            .iter()
            .any(|violation| violation.contains(expected)));
    }
}

#[test]
fn cfg_test_macro_remains_test_only() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "#[cfg(test)]\nmacro_rules! test_helper {\n    () => { panic!() };\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, 3),
        &LcovReport::default(),
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 0);
    assert!(result.changed_opaque_macros.is_empty());
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}
