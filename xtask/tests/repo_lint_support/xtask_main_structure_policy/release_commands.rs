// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::{assert_pattern_checks, repo_root, PatternCheck};

pub(super) fn assert_regressions_stay_focused() {
    let root = repo_root();
    let tests = fs::read_to_string(root.join("xtask/src/release_commands/tests.rs"))
        .expect("read release command tests");
    let integrity = fs::read_to_string(root.join("xtask/src/release_commands/tests/integrity.rs"))
        .expect("read release integrity command tests");
    let validation =
        fs::read_to_string(root.join("xtask/src/release_commands/tests/validation.rs"))
            .expect("read release validation command tests");

    for (relative, source, max_lines) in [
        ("xtask/src/release_commands/tests.rs", &tests, 250),
        (
            "xtask/src/release_commands/tests/integrity.rs",
            &integrity,
            200,
        ),
        (
            "xtask/src/release_commands/tests/validation.rs",
            &validation,
            180,
        ),
    ] {
        assert!(
            source.lines().count() < max_lines,
            "{relative} must stay below its focused {max_lines}-line ownership ratchet"
        );
    }

    assert_pattern_checks(&[
        PatternCheck::new("release command test modules", &tests)
            .required(&["mod integrity;", "mod validation;"]),
        PatternCheck::new("release integrity command regressions", &integrity).required(&[
            "fn release_integrity_accepts_complete_hermetic_metadata_in_pre_candidate_mode()",
            "fn release_integrity_aggregates_invalid_package_metadata_without_publishing()",
            "fn release_integrity_rejects_non_json_cargo_metadata()",
            "run_test_from_workspace(",
        ]),
        PatternCheck::new("release validation command regressions", &validation).required(&[
            "fn publish_workflow_validation_reports_parse_and_release_contract_failures()",
            "fn publish_script_validation_fails_closed_for_missing_and_drifted_contracts()",
            "fn release_docs_validation_reports_missing_packages_and_operational_guards()",
            "fn unpublished_dependency_validation_skips_external_edges_and_accepts_path_only_dev_edges()",
        ]),
    ]);
}
