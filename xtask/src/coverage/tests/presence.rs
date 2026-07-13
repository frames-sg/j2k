// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::support::TestRepository;
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;

#[test]
fn partial_file_lcov_does_not_mask_second_changed_function_without_covered_body() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "\
pub fn present() {
    let _present = 1;
}
pub fn absent() {
    let _absent = 2;
}
";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([1, 4]))]);
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(2, 1)]))]),
        ..LcovReport::default()
    };

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed,
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(
        result.changed_functions_without_covered_body,
        [format!("{path}::absent@4")]
    );
    assert!(result.absent_instrumentable_files.is_empty());
}

#[test]
fn shared_accelerator_source_absent_from_metal_lcov_is_a_violation() {
    let repository = TestRepository::new();
    let path = "crates/j2k-core/src/accelerator.rs";
    let source = "pub fn choose_accelerator() {\n    let _route = 1;\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([1]))]);

    let result = evaluate_changed_coverage(
        CoverageLane::Metal,
        repository.root(),
        &changed,
        &LcovReport::default(),
        &index,
    )
    .unwrap();
    let violations = coverage_violations(CoverageLane::Metal, &result);

    assert_eq!(result.absent_instrumentable_files, vec![path.to_string()]);
    assert!(violations.iter().any(|violation| violation.contains(path)));
}

#[test]
fn zero_count_body_record_does_not_prove_changed_signature_coverage() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "pub fn never_called() {\n    let _value = 1;\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([1]))]);
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(2, 0)]))]),
        ..LcovReport::default()
    };

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed,
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(
        result.changed_functions_without_covered_body,
        [format!("{path}::never_called@1")]
    );
    assert!(result.absent_instrumentable_files.is_empty());
}

#[test]
fn changed_executable_body_line_without_da_is_uncovered() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "pub fn partly_mapped() {\n    let _changed = 1;\n    let _mapped = 2;\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([2]))]);
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(3, 1)]))]),
        ..LcovReport::default()
    };

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed,
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.overall.covered, 0);
    assert_eq!(result.uncovered, [(path.to_string(), 2)]);
    assert_eq!(result.unmeasured, [(path.to_string(), 2)]);
    assert!(result.changed_functions_without_covered_body.is_empty());
    assert!(!coverage_violations(CoverageLane::Host, &result).is_empty());
}
