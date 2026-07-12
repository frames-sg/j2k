// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::support::TestRepository;
use crate::coverage::evaluation::{coverage_violations, evaluate_changed_coverage};
use crate::coverage::model::{CoverageLane, LcovReport};
use crate::coverage::source_analysis::SourceIndex;

#[test]
fn cfg_active_changed_source_cannot_evade_coverage_gate() {
    let repository = TestRepository::new();
    let path = "crates/example/src/lib.rs";
    let source = "#[cfg(build_script_decides)]\npub fn active() {\n    let _value = 1;\n}\n";
    repository.write(path, source);
    let index = SourceIndex::single_with_custom_cfg(path, source, [("build_script_decides", true)])
        .unwrap();
    let changed = BTreeMap::from([(path.to_string(), BTreeSet::from([2]))]);

    let result = evaluate_changed_coverage(
        CoverageLane::Host,
        repository.root(),
        &changed,
        &LcovReport::default(),
        &index,
    )
    .unwrap();

    assert_eq!(result.absent_instrumentable_files, [path.to_string()]);
    assert!(!coverage_violations(CoverageLane::Host, &result).is_empty());
}
