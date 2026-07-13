// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    assert_line_budget, assert_module_boundary, assert_pattern_checks, read, PatternCheck,
};

pub(super) fn assert_contract() {
    let production = read("crates/j2k-compare/src/fixture_compare/manifest.rs");
    let tests = read("crates/j2k-compare/src/fixture_compare/manifest/tests.rs");
    let policy = read("xtask/tests/repo_lint_support/fixture_compare_structure_policy/manifest.rs");

    assert_module_boundary("fixture_compare/manifest.rs", &production, 325);
    assert_module_boundary("fixture_compare/manifest/tests.rs", &tests, 425);
    assert_line_budget(
        "xtask/tests/repo_lint_support/fixture_compare_structure_policy/manifest.rs",
        &policy,
        75,
    );
    assert_pattern_checks(&[
        PatternCheck::new("fixture_compare manifest ownership", &production).required(&[
            "fn fixture_manifest_from_path(",
            "pub(super) fn fixture_manifest_from_env",
            "pub(super) fn external_fixture_metadata",
            "#[cfg(test)]\nmod tests;",
        ]),
        PatternCheck::new("fixture_compare manifest regressions", &tests).required(&[
            "manifest_parser_reports_read_header_and_row_structure_errors",
            "manifest_parser_rejects_duplicate_paths_and_invalid_type_pins",
            "external_metadata_rejects_hash_codec_and_container_mismatches",
        ]),
    ]);
}
