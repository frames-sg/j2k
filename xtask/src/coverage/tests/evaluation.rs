// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::support::TestRepository;
use crate::coverage::accelerator_ownership::shared_accelerator_sources;
use crate::coverage::critical_path_policy::{
    audited_zero_body_findings, CriticalPathClass, ZeroBodyAudit,
};
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;

mod compiler_line_evidence;
mod non_executable;

fn changed(
    path: &str,
    lines: impl IntoIterator<Item = usize>,
) -> BTreeMap<String, BTreeSet<usize>> {
    BTreeMap::from([(path.to_string(), lines.into_iter().collect())])
}

#[test]
fn changed_signature_requires_a_positive_da_record_in_the_function_body() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "\
/// Changed contract.
pub fn calculate() -> u32 {
    7
}
";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let changed = changed(path, [1]);
    let signature_only = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(1, 1)]))]),
        ..LcovReport::default()
    };

    let without_covered_body = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed,
        &signature_only,
        &index,
    )
    .unwrap();
    assert_eq!(
        without_covered_body.changed_functions_without_covered_body,
        [format!("{path}::calculate@1")]
    );
    assert!(without_covered_body.absent_instrumentable_files.is_empty());

    let body_covered = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(1, 1), (3, 1)]))]),
        ..LcovReport::default()
    };
    let present = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed,
        &body_covered,
        &index,
    )
    .unwrap();
    assert!(present.changed_functions_without_covered_body.is_empty());
}

#[test]
fn changed_function_without_covered_body_is_a_critical_audit_finding() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "pub unsafe fn changed() {\n    let _value = 1;\n}\n";
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

    assert_eq!(result.absent_instrumentable_files, vec![path.to_string()]);
    assert!(coverage_violations(CoverageLane::Host, &result)
        .iter()
        .any(|violation| violation.contains("critical executable bodies are absent")));
    assert_eq!(
        audited_zero_body_findings(CoverageLane::Host, &result)[0].audit,
        ZeroBodyAudit::Critical(CriticalPathClass::PublicApi)
    );
}

#[test]
fn cfg_test_lines_are_disposed_without_hiding_later_production() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "\
#[cfg(test)]
mod tests {
    fn helper() {}
}
pub fn production() {
    let _value = 1;
}
";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(6, 1)]))]),
        ..LcovReport::default()
    };
    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed(path, [2, 6]),
        &report,
        &index,
    )
    .unwrap();

    assert_eq!(result.overall.measurable, 1);
    assert_eq!(result.overall.covered, 1);
    assert_eq!(
        result.source_dispositions["syntax-test-only"].changed_lines,
        1
    );
    assert_eq!(result.source_dispositions["production"].changed_lines, 1);
}

#[test]
fn same_line_cfg_test_and_production_code_fails_closed_even_with_positive_da() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "pub fn mixed() { #[cfg(test)] test_only(); production_behavior(); }\n";
    repository.write(path, source);
    let index = SourceIndex::single(path, source).unwrap();
    let report = LcovReport {
        lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(1, 1)]))]),
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

    assert_eq!(result.mixed_test_production_lines, [format!("{path}:1")]);
    assert_eq!(result.overall.measurable, 0);
    assert!(coverage_violations(CoverageLane::Host, &result)
        .iter()
        .any(|violation| violation.contains("split test-only and production syntax")));
}

#[test]
fn registered_shared_accelerator_sources_reach_both_gpu_denominators() {
    for source_owner in shared_accelerator_sources() {
        for lane in [CoverageLane::Metal, CoverageLane::Cuda] {
            let repository = TestRepository::new();
            let path = source_owner.path;
            let source = "pub fn route() {}\n";
            repository.write(path, source);
            let index = SourceIndex::single(path, source).unwrap();
            let report = LcovReport {
                lines: BTreeMap::from([(path.to_string(), BTreeMap::from([(1, 1)]))]),
                ..LcovReport::default()
            };
            let result = evaluate_changed_coverage(
                lane,
                repository.root(),
                &changed(path, [1]),
                &report,
                &index,
            )
            .unwrap();

            assert_eq!(result.accelerator.measurable, 1, "{lane:?}: {path}");
            assert_eq!(result.accelerator.covered, 1, "{lane:?}: {path}");
        }
    }
}

#[test]
fn generated_and_vendored_sources_have_reviewed_dispositions() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let generated = "crates/j2k-codec-math/generated/dwt97_constants.rs";
    let generated_changed = changed(generated, [1]);
    let generated_index = SourceIndex::repository_subset(&root, &generated_changed, &[]).unwrap();
    let generated_result = evaluate_changed_coverage(
        CoverageLane::Host,
        &root,
        &generated_changed,
        &LcovReport::default(),
        &generated_index,
    )
    .unwrap();
    assert_eq!(
        generated_result.source_dispositions["generated-codec-math-fragment"].changed_lines,
        1
    );
    assert_eq!(generated_result.overall.measurable, 0);

    let vendored = "third_party/block-0.1.6-patched/src/lib.rs";
    let vendored_changed = changed(vendored, [1]);
    let vendored_index = SourceIndex::repository_subset(&root, &vendored_changed, &[]).unwrap();
    let vendored_result = evaluate_changed_coverage(
        CoverageLane::Metal,
        &root,
        &vendored_changed,
        &LcovReport::default(),
        &vendored_index,
    )
    .unwrap();
    assert_eq!(
        vendored_result.source_dispositions["vendored-block-ffi-binding"].changed_lines,
        1
    );
    assert_eq!(vendored_result.overall.measurable, 0);
}
