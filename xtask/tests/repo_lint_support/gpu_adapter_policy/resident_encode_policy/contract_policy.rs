// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

mod ownership;

#[test]
fn resident_encode_contract_has_host_regression_coverage() {
    let root = repo_root();
    let tests = fs::read_to_string(root.join("crates/j2k/src/encode/resident/tests.rs"))
        .expect("read resident encode contract tests");
    assert_pattern_checks(&[
        PatternCheck::new("resident encode behavior tests", &tests).required(&[
            "resident_encode_matches_host_hook_codestream_and_reports_dispatch",
            "resident_encode_decline_and_accelerator_error_are_explicit",
            "resident_encode_rejects_success_without_required_dispatch_accounting",
            "resident_encode_preserves_strict_option_contract_without_host_fallback",
            "resident_encode_rejects_cpu_backend_kind_before_dispatch",
            "resident_input_rejects_invalid_geometry_without_calling_accelerator",
            "resident_encode_handles_huge_logical_geometry_without_host_image_allocation",
        ]),
    ]);
}
