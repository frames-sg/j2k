// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_module_boundary, assert_pattern_checks, read, PatternCheck};

pub(super) fn assert_contract() {
    let production = read("crates/j2k-compare/src/fixture_compare/fixtures.rs");
    let tests = read("crates/j2k-compare/src/fixture_compare/fixtures/tests.rs");

    assert_module_boundary("fixture_compare/fixtures.rs", &production, 525);
    assert_module_boundary("fixture_compare/fixtures/tests.rs", &tests, 350);
    assert_pattern_checks(&[
        PatternCheck::new("fixture_compare fixture ownership", &production).required(&[
            "pub(super) fn all_fixture_cases",
            "pub(super) fn fixture_cases",
            "pub(super) fn load_external_fixture_cases",
            "pub(super) fn encode_lossless",
            "#[cfg(test)]\nmod tests;",
        ]),
        PatternCheck::new("fixture_compare fixture regressions", &tests).required(&[
            "fn fixture_classification_covers_supported_extensions_containers_and_codecs()",
            "fn case_materialization_validates_codestream_shape_and_roi()",
            "fn external_loader_reports_directory_and_fixture_failures()",
            "fn external_loader_sorts_cases_and_applies_region_scaled_corpus_policy()",
            "fn mixed_external_batches_group_compatible_distinct_inputs_only()",
            "fn generated_encoder_rejects_invalid_samples_and_unknown_codec()",
        ]),
    ]);
}
