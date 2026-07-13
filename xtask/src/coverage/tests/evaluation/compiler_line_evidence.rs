// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::changed;
use crate::coverage::compiler_regions::{CompilerRegionReport, SourceSpan};
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{ChangedCoverageResult, CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;
use crate::coverage::tests::support::TestRepository;

const PATH: &str = "crates/example/src/lib.rs";

fn evaluate_case(region_count: Option<u64>) -> ChangedCoverageResult {
    let source = "pub fn values() -> [u8; 2] {\n    let values = [\n        1,\n        2,\n    ];\n    values\n}\n";
    let index = SourceIndex::single(PATH, source).unwrap();
    assert!(
        index.file(PATH).unwrap().executable_lines.contains(&3),
        "multiline local initializer must be classified as executable"
    );
    let report = LcovReport {
        lines: BTreeMap::from([(PATH.to_string(), BTreeMap::from([(2, 1)]))]),
        compiler_regions: CompilerRegionReport::for_test(
            PATH,
            &region_count.map_or_else(Vec::new, |count| {
                vec![(SourceSpan::new(2, 5, 5, 7).unwrap(), count)]
            }),
        ),
    };
    let repository = TestRepository::new();
    repository.write(PATH, source);
    evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(PATH, [3]),
        &report,
        &index,
    )
    .unwrap()
}

#[test]
fn covered_compiler_region_owns_multiline_expression_lines_without_da_records() {
    let result = evaluate_case(Some(1));

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.overall.covered, 1);
    assert_eq!(result.critical.measurable, 1);
    assert_eq!(result.critical.covered, 1);
    assert!(result.uncovered.is_empty());
    assert!(result.unmeasured.is_empty());
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}

#[test]
fn zero_compiler_region_keeps_an_executable_line_without_da_uncovered() {
    let result = evaluate_case(Some(0));

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.overall.covered, 0);
    assert_eq!(result.uncovered, [(PATH.to_string(), 3)]);
    assert!(result.unmeasured.is_empty());
    assert!(!coverage_violations(CoverageLane::Host, &result).is_empty());
}

#[test]
fn compiler_noninstrumentable_line_is_recorded_without_entering_the_denominator() {
    let result = evaluate_case(None);

    assert_eq!(result.overall.measurable, 0);
    assert!(result.uncovered.is_empty());
    assert!(result.unmeasured.is_empty());
    assert_eq!(
        result.compiler_noninstrumentable_lines,
        [format!("{PATH}:3")]
    );
    assert!(coverage_violations(CoverageLane::Host, &result).is_empty());
}
