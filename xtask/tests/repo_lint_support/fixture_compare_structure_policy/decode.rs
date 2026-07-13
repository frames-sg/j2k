// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    assert_line_budget, assert_module_boundary, assert_pattern_checks, read, PatternCheck,
};

pub(super) fn assert_contract() {
    let production = read("crates/j2k-compare/src/fixture_compare/decode.rs");
    let tests = read("crates/j2k-compare/src/fixture_compare/decode/tests.rs");
    let policy = read("xtask/tests/repo_lint_support/fixture_compare_structure_policy/decode.rs");

    assert_module_boundary("fixture_compare/decode.rs", &production, 650);
    assert_module_boundary("fixture_compare/decode/tests.rs", &tests, 425);
    assert_line_budget(
        "xtask/tests/repo_lint_support/fixture_compare_structure_policy/decode.rs",
        &policy,
        75,
    );
    assert_pattern_checks(&[
        PatternCheck::new("fixture_compare decode ownership", &production).required(&[
            "pub(super) fn decode_j2k_batch",
            "pub(super) fn decode_external_once",
            "pub(super) fn decode_method_label",
            "pub(super) fn crop_interleaved",
            "#[cfg(test)]\nmod tests;",
        ]),
        PatternCheck::new("fixture_compare decode regressions", &tests).required(&[
            "j2k_decode_failures_preserve_operation_and_batch_context",
            "openjpeg_router_matches_j2k_for_gray_and_rgb_operations",
            "external_batch_dispatch_preserves_order_for_homogeneous_and_mixed_inputs",
            "portable_emulation_crops_full_scaled_openjpeg_output",
            "external_router_rejects_decoded_length_mismatches",
        ]),
    ]);
}
