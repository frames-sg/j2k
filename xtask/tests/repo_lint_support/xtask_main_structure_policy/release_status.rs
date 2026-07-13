// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_module_stays_focused, assert_pattern_checks, read, PatternCheck};

pub(super) fn assert_ownership_and_focus() {
    let policy =
        read("xtask/tests/repo_lint_support/xtask_main_structure_policy/release_status.rs");
    let production = read("xtask/src/release_status.rs");
    let tests = read("xtask/src/release_status/tests.rs");
    let boundary_tests = read("xtask/src/release_status/tests/boundary_errors.rs");

    assert_module_stays_focused(
        "xtask/tests/repo_lint_support/xtask_main_structure_policy/release_status.rs",
        &policy,
        75,
    );
    assert_module_stays_focused("xtask/src/release_status.rs", &production, 350);
    assert_module_stays_focused("xtask/src/release_status/tests.rs", &tests, 225);
    assert_module_stays_focused(
        "xtask/src/release_status/tests/boundary_errors.rs",
        &boundary_tests,
        175,
    );
    assert_pattern_checks(&[
        PatternCheck::new("release-status test module ownership", &production)
            .required(&["#[cfg(test)]", "mod tests;"]),
        PatternCheck::new("release-status command regressions", &tests).required(&[
            "mod boundary_errors;",
            "options_reject_missing_values_duplicates_help_and_unknown_arguments",
            "release_status_derives_remote_and_executes_exact_verifier_contract",
        ]),
        PatternCheck::new("release-status boundary regressions", &boundary_tests).required(&[
            "repository_environment_empty_and_present_paths_execute_exact_contracts",
            "repository_discovery_rejects_process_and_unicode_failures",
        ]),
    ]);
}
