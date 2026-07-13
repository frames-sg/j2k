// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::changed;
use crate::coverage::compiler_regions::{CompilerRegionReport, SourceSpan};
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;
use crate::coverage::tests::support::TestRepository;

fn multiline_expression_case(region_count: u64) -> (String, LcovReport, SourceIndex) {
    let path = "crates/example/src/lib.rs";
    let source = "pub fn values() -> [u8; 2] {\n    let values = [\n        1,\n        2,\n    ];\n    values\n}\n";
    let index = SourceIndex::single(path, source).unwrap();
    assert!(
        index.file(path).unwrap().executable_lines.contains(&3),
        "multiline local initializer must be classified as executable"
    );
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(2, 1)]))]),
        compiler_regions: CompilerRegionReport::for_test(
            path,
            &[(SourceSpan::new(2, 5, 5, 7).unwrap(), region_count)],
        ),
    };
    (source.to_string(), report, index)
}

#[test]
fn covered_compiler_region_owns_multiline_expression_lines_without_da_records() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let (source, report, index) = multiline_expression_case(1);
    repository.write(path, &source);

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [3]),
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.overall.covered, 1);
    assert!(result.uncovered.is_empty());
    assert!(result.unmeasured.is_empty());
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}

#[test]
fn zero_compiler_region_keeps_an_executable_line_without_da_uncovered() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let (source, report, index) = multiline_expression_case(0);
    repository.write(path, &source);

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [3]),
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.overall.covered, 0);
    assert_eq!(result.uncovered, [(path.to_string(), 3)]);
    assert!(result.unmeasured.is_empty());
    assert!(!coverage_violations(CoverageLane::Host, &result).is_empty());
}
