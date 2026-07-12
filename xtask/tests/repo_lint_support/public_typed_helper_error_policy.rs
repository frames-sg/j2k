// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed-error ratchets for reachable HT validation and diagnostic helpers.

use std::fs;

use super::{assert_pattern_checks, read_source_files, repo_root, PatternCheck};

fn read(relative: &str) -> String {
    fs::read_to_string(repo_root().join(relative))
        .unwrap_or_else(|error| panic!("read {relative}: {error}"))
}

#[test]
fn public_scalar_code_block_adapters_preserve_native_encode_errors() {
    let adapter = read("crates/j2k-native/src/scalar/encode.rs");
    let tests = read("crates/j2k-native/src/tests.rs");

    assert_pattern_checks(&[
        PatternCheck::new("typed scalar code-block adapters", &adapter)
            .required(&[
                "pub fn encode_j2k_code_block_scalar_with_style(",
                "pub fn pack_j2k_code_block_scalar_from_tier1_tokens(",
                "pub fn encode_ht_code_block_scalar(",
                "pub fn encode_ht_code_block_scalar_with_passes(",
                ") -> EncodeResult<EncodedJ2kCodeBlock>",
                ") -> EncodeResult<EncodedHtJ2kCodeBlock>",
                "try_pack_classic_selective_bypass_tier1_tokens(",
                "try_encode_code_block_with_passes(",
            ])
            .forbidden(&[
                "core::result::Result<EncodedJ2kCodeBlock, &'static str>",
                "core::result::Result<EncodedHtJ2kCodeBlock, &'static str>",
                "legacy_coefficient_view_error",
            ]),
        PatternCheck::new("typed scalar adapter regressions", &tests).required(&[
            "scalar_encode_adapters_preserve_typed_input_and_cap_categories",
            "EncodeError::InvalidInput",
            "EncodeError::AllocationTooLarge",
            "classic Tier-1 token worker allocation",
        ]),
    ]);
}

#[test]
fn public_ht_segment_math_returns_a_typed_closed_taxonomy() {
    let source = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/packet_math.rs",
            "crates/j2k-native/src/packet_math/error.rs",
            "crates/j2k-native/src/j2c/packet_encode/header.rs",
        ],
    );

    assert_pattern_checks(&[PatternCheck::new("typed HT segment math", &source)
        .required(&[
            ") -> Result<(u32, u32), HtSegmentLengthError>",
            "#[non_exhaustive]\npub enum HtSegmentLengthError",
            "ContributionLengthExceedsU32",
            "MultiPassLengthOverflow",
            "impl core::error::Error for HtSegmentLengthError",
            "ht_segment_length_validation_returns_each_semantic_failure",
            "packet_math::HtSegmentLengthError::ContributionLengthExceedsU32",
            "packet_math::HtSegmentLengthError::MultiPassLengthOverflow",
        ])
        .forbidden(&[
            ") -> Result<(u32, u32), &'static str>",
            "what == \"multi-pass HTJ2K packet contribution length overflow\"",
        ])]);
}

#[test]
fn cleanup_distribution_preserves_native_encode_categories() {
    let source = read_source_files(
        repo_root(),
        &[
            "crates/j2k-native/src/scalar/encode.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/distribution.rs",
            "crates/j2k-native/src/j2c/ht_block_encode/distribution/tests.rs",
        ],
    );

    assert_pattern_checks(
        &[PatternCheck::new("typed HT cleanup distribution", &source)
            .required(&[
                "pub fn collect_ht_cleanup_encode_distribution(",
                ") -> EncodeResult<HtCleanupEncodeDistribution>",
                "validate_tier1_code_block_geometry(width, height)?;",
                "CoefficientBlockView::try_contiguous(coefficients, width, height)?;",
                "distribution_rejects_bitplane_range_and_magnitude_with_typed_input_errors",
            ])
            .forbidden(&[
                ") -> core::result::Result<HtCleanupEncodeDistribution, &'static str>",
                "map_err(legacy_coefficient_view_error)",
            ])],
    );
}

#[test]
fn shared_97_option_validation_returns_typed_backend_neutral_failures() {
    let oracle = read("crates/j2k-transcode/src/htj2k97_codeblock_oracle.rs");
    let error = read("crates/j2k-transcode/src/htj2k97_codeblock_error.rs");
    let source = format!("{oracle}\n{error}");

    assert_pattern_checks(&[
        PatternCheck::new("typed HTJ2K 9/7 option validation", &source)
            .required(&[
                ") -> Result<(usize, usize), Htj2k97CodeBlockOptionsError>",
                "#[non_exhaustive]\npub enum Htj2k97CodeBlockOptionsError",
                "NumericOptionsOutOfRange",
                "QuantizationOptionsOutOfRange",
                "DimensionExponentUnsupported {",
                "DimensionsExceedLimits {",
                "impl std::error::Error for Htj2k97CodeBlockOptionsError",
                "shared_validator_returns_each_typed_failure_variant",
            ])
            .forbidden(&[
                ") -> Result<(usize, usize), &'static str>",
                "Err(\"9/7 code-block",
            ]),
    ]);
}
