// SPDX-License-Identifier: MIT OR Apache-2.0

//! Architectural ratchets for facade-owned encode round-trip validation.

use std::fs;

use super::{assert_file_pattern_checks, repo_root, FilePatternCheck};

#[test]
fn encode_validation_counts_retained_output_and_preserves_native_errors() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k/src/encode/validation.rs")
                .named("facade encode validation")
                .required(&[
                    "fn psnr_from_validated_bitmap(",
                    "raw_bitmap_metadata_matches(",
                    "output_validation_error(",
                ]),
            FilePatternCheck::new("crates/j2k/src/encode/validation/decode.rs")
                .named("facade retained validation decode")
                .required(&[
                    "codestream.capacity()",
                    "Image::new_with_retained_baseline(",
                    ".decode_native_with_retained_capacity(retained_capacity)",
                    ".decode_native_components_with_retained_capacity(retained_capacity)",
                    "fn validation_decode_error(source: DecodeError, context: &'static str)",
                    "source: crate::NativeBackendError::decode(source)",
                    "fn output_validation_error(",
                    "BackendErrorKind::Validation",
                    "fn raw_bitmap_metadata_matches(",
                    "decoded.signed == signed",
                    "decoded.component_signed.len() == usize::from(components)",
                    "DecodeError::AllocationTooLarge",
                ])
                .forbidden(&[
                    "Image::new(codestream",
                    ".decode_native()",
                    "validation_backend",
                    "encoded codestream validation failed: {err}",
                    "from_native_decode_error_with_context",
                ]),
            FilePatternCheck::new("crates/j2k/src/error.rs")
                .named("facade validation source-chain regression")
                .required(&[
                    "generated_output_validation_errors_do_not_masquerade_as_caller_failures",
                    "source: NativeBackendError::decode(source)",
                    "assert_native_source_chain(&error, &source);",
                ]),
            FilePatternCheck::new("crates/j2k/src/error/native_source.rs")
                .named("facade-owned native source chain")
                .required(&[
                    "impl core::error::Error for NativeBackendError",
                    "NativeBackendErrorSource::Decode(source) => source",
                ])
                .forbidden(&["message: String", "error.to_string()"]),
        ],
    );
}

#[test]
fn component_validation_streams_canonical_samples_without_reference_grid_owner() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k/src/encode/validation/component.rs")
                .named("facade component validation")
                .required(&[
                    "decode_native_components_for_validation(",
                    "fn validate_component_plane(",
                    "fn native_sample_matches(",
                    "canonical_native_sample_bytes(",
                    "actual.data()",
                    "output_validation_error(format!(",
                ])
                .forbidden(&[
                    "Vec::with_capacity",
                    ".to_vec()",
                    ".collect::<Vec",
                    "canonical_native_typed_component_bytes_for_reference_grid",
                ]),
        ],
    );
}

#[test]
fn rate_target_searches_do_not_retain_best_and_candidate_codestreams_together() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k/src/encode/lossy.rs")
                .named("rate search ownership")
                .required(&[
                    "let initial_codestream = encode_at_scale(high)?;",
                    "drop(initial_codestream);",
                    "let mut best_scale = high;",
                    "while current_len > maximum_bytes",
                    "if best_len > maximum_bytes",
                    "let codestream = encode_at_scale(best_scale)?;",
                    "if final_len > maximum_bytes",
                    "if final_psnr + tolerance < target_psnr_db",
                ])
                .forbidden(&["best.codestream.capacity()", "let mut best = LossyAttempt"]),
            FilePatternCheck::new("crates/j2k/tests/encode_lossy_rate_targets.rs")
                .named("rate search behavior coverage")
                .required(&[
                    "psnr_target_returns_a_validated_candidate_at_the_selected_scale",
                    "byte_target_reencoding_is_deterministic",
                    "DecodeSettings::strict()",
                ]),
            FilePatternCheck::new("crates/j2k/src/encode/lossy/tests/rate_validation.rs")
                .named("rate search boundary coverage")
                .required(&[
                    "byte_target_accepts_first_under_limit_candidate_after_large_overshoot",
                    "byte_target_revalidates_the_final_stateful_encode",
                    "external_psnr_target_rejects_final_wrong_signedness_metadata",
                    "assert_eq!(attempt.codestream.len(), 300)",
                ]),
            FilePatternCheck::new("crates/j2k/src/encode.rs")
                .named("final lossy dispatch ownership")
                .required(&[
                    "let mut final_dispatch = J2kEncodeDispatchReport::default();",
                    "final_dispatch = accelerator.dispatch_report().saturating_delta(before);",
                    "dispatch_report: final_dispatch",
                ]),
            FilePatternCheck::new("crates/j2k/tests/encode_lossy.rs")
                .named("final lossy dispatch behavior")
                .required(&[
                    "lossy_require_device_uses_only_the_selected_final_attempt_dispatch",
                    "the returned final attempt did not dispatch required device stages",
                ]),
        ],
    );

    let source = fs::read_to_string(repo_root().join("crates/j2k/src/encode/lossy.rs"))
        .expect("read lossy rate-search source");
    for pattern in [
        "let initial_codestream = encode_at_scale(high)?;",
        "drop(initial_codestream);",
        "let codestream = encode_at_scale(best_scale)?;",
    ] {
        assert_eq!(
            source.matches(pattern).count(),
            2,
            "byte-target and PSNR searches must both contain `{pattern}`"
        );
    }

    assert!(source.contains("if len <= maximum_bytes && diff < best_diff"));
}

#[test]
fn retained_decode_adapters_use_one_typed_capacity_budget() {
    assert_file_pattern_checks(
        repo_root(),
        &[
            FilePatternCheck::new("crates/j2k-native/src/image/output_api.rs")
                .named("retained packed decode adapter")
                .required(&[
                    "pub fn decode_native_with_retained_capacity(",
                    "combine_retained_bytes(",
                    "decode_native_with_context_and_retained_baseline(",
                ]),
            FilePatternCheck::new("crates/j2k-native/src/image.rs")
                .named("retained component decode adapter")
                .required(&[
                    "pub fn decode_native_components_with_retained_capacity(",
                    "allocation::combine_retained_bytes(",
                    "decode_native_components_with_context_and_retained_baseline(",
                ]),
            FilePatternCheck::new("crates/j2k-native/src/image/allocation.rs")
                .named("retained decode owner budget")
                .required(&[
                    "DecodeError::AllocationTooLarge",
                    "requested: usize::MAX",
                    "requested: updated",
                    "cap: DEFAULT_MAX_DECODE_BYTES",
                ])
                .forbidden(&["return Err(ValidationError::ImageTooLarge.into())"]),
            FilePatternCheck::new("crates/j2k/src/error.rs")
                .named("facade native decode resource mapping")
                .required(&[
                    "native_decode_resource_errors_preserve_context_and_source",
                    "NativeDecodeError::AllocationTooLarge",
                    "NativeDecodeError::HostAllocationFailed",
                    "J2kError::NativeDecode",
                    "J2kError::NativeValidation",
                    "generated_output_validation_errors_do_not_masquerade_as_caller_failures",
                ]),
            FilePatternCheck::new("crates/j2k/src/encode/validation/tests/errors.rs")
                .named("facade validation semantic tests")
                .required(&[
                    "interleaved_validation_requires_uniform_signedness_metadata",
                    "generated_codestream_failures_keep_validation_context",
                    "oversized_validation_capacity_is_a_typed_resource_error",
                    "psnr_validation_rejects_matching_bytes_with_wrong_metadata",
                ]),
        ],
    );
}

#[test]
fn encode_validation_modules_remain_reviewable() {
    for (relative, max_lines) in [
        ("crates/j2k/src/encode/validation.rs", 300_usize),
        ("crates/j2k/src/encode/validation/component.rs", 380_usize),
        ("crates/j2k/src/encode/validation/decode.rs", 140_usize),
        ("crates/j2k/src/encode/validation/tests.rs", 60_usize),
        (
            "crates/j2k/src/encode/validation/tests/errors.rs",
            120_usize,
        ),
        ("crates/j2k/src/encode/lossy.rs", 420_usize),
        ("crates/j2k/src/encode/lossy/tests.rs", 60_usize),
        (
            "crates/j2k/src/encode/lossy/tests/rate_validation.rs",
            130_usize,
        ),
    ] {
        let source = fs::read_to_string(repo_root().join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"));
        let lines = source.lines().count();
        assert!(
            lines <= max_lines,
            "{relative} grew to {lines} lines; split the validation concern before {max_lines}"
        );
    }
}
