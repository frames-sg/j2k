// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn jpeg_dct_reemission_input_contract_is_typed_and_complete() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let transcode = read("crates/j2k-jpeg/src/transcode.rs");
    let validation = read("crates/j2k-jpeg/src/transcode/validation.rs");
    let coefficients = read("crates/j2k-jpeg/src/transcode/validation/coefficients.rs");
    let encoder = read("crates/j2k-jpeg/src/encoder.rs");
    let info = read("crates/j2k-jpeg/src/info.rs");
    let contract_tests = read("crates/j2k-jpeg/tests/dct_reemit_contract.rs");
    let parity_tests = read("crates/j2k-jpeg/tests/dct_reemit_parity.rs");

    assert_source_sizes(
        &transcode,
        &validation,
        &coefficients,
        &contract_tests,
        &parity_tests,
    );
    assert_product_contract(&transcode, &validation, &coefficients, &encoder);
    for (source, owner) in [
        (encoder.as_str(), "EncodedJpeg"),
        (transcode.as_str(), "JpegDctImage"),
        (transcode.as_str(), "JpegDctComponent"),
        (info.as_str(), "RestartIndex"),
    ] {
        assert_move_only_owner(source, owner);
    }
    assert_regression_contract(&contract_tests, &parity_tests);
    assert_validation_order(&transcode);
}

fn assert_source_sizes(
    transcode: &str,
    validation: &str,
    coefficients: &str,
    contract_tests: &str,
    parity_tests: &str,
) {
    for (name, source, maximum) in [
        ("transcode.rs", transcode, 700),
        ("transcode/validation.rs", validation, 350),
        ("transcode/validation/coefficients.rs", coefficients, 125),
        ("tests/dct_reemit_contract.rs", contract_tests, 250),
        ("tests/dct_reemit_parity.rs", parity_tests, 100),
    ] {
        assert!(
            source.lines().count() < maximum,
            "j2k-jpeg/{name} must stay below its DCT re-emission contract ratchet"
        );
    }
}

fn assert_product_contract(transcode: &str, validation: &str, coefficients: &str, encoder: &str) {
    assert_pattern_checks(&[
        PatternCheck::new("DCT re-emission public error", encoder)
            .required(&[
                "InvalidDctImage {",
                "reason: crate::transcode::JpegDctImageError",
                "InternalInvariant {",
            ])
            .forbidden(&["Internal(String)", "JpegEncodeError::Internal("]),
        PatternCheck::new("DCT re-emission facade", transcode)
            .required(&[
                "mod validation;",
                "pub use self::validation::JpegDctImageError;",
                "validate_baseline_dct_image(image)",
                "JpegEncodeError::InvalidDctImage { reason }",
            ])
            .forbidden(&["JpegEncodeError::Internal(", "format!("]),
        PatternCheck::new("typed DCT input validation", validation)
            .required(&[
                "#[non_exhaustive]",
                "pub enum JpegDctImageError",
                "UnsupportedCodingMode",
                "EmptyDimensions",
                "DimensionsTooLarge",
                "UnsupportedComponentCount",
                "ComponentOrderMismatch",
                "SamplingFactorOutOfRange",
                "UnsupportedGrayscaleSampling",
                "TooManyBlocksPerMcu",
                "BlockGridArithmeticOverflow",
                "BlockGridMismatch",
                "QuantizedBlockCountMismatch",
                "DcMagnitudeCategoryOutOfRange",
                "AcMagnitudeCategoryOutOfRange",
                "QuantizationValueOutOfRange",
                "ChromaQuantizationTableMismatch",
                "MAX_SAMPLING_FACTOR: u8 = 4",
                "MAX_BLOCKS_PER_MCU: u16 = 10",
                ".checked_mul(",
                "usize::try_from(blocks)",
                "u8::try_from(value)",
                "validate_baseline_coefficients(image, sampling)?",
            ])
            .forbidden(&[
                "JpegEncodeError::Internal(",
                "format!(",
                "image.color_space",
                "image.scan_count",
                "image.restart_index",
                "component.width",
                "component.height",
                "dequantized_blocks",
            ]),
        PatternCheck::new("baseline DCT coefficient categories", coefficients)
            .required(&[
                "MAX_BASELINE_DC_CATEGORY: u8 = 11",
                "MAX_BASELINE_AC_CATEGORY: u8 = 10",
                "validate_baseline_coefficients",
                "use crate::encoder::magnitude",
                "magnitude(difference).0",
                "magnitude(i32::from(value)).0",
                "DcMagnitudeCategoryOutOfRange",
                "AcMagnitudeCategoryOutOfRange",
            ])
            .forbidden(&[
                "JpegEncodeError::Internal(",
                "format!(",
                "fn magnitude_category",
            ]),
    ]);
}

fn assert_move_only_owner(source: &str, owner: &str) {
    let declaration = format!("pub struct {owner}");
    let (prefix, _) = source
        .split_once(&declaration)
        .unwrap_or_else(|| panic!("missing public JPEG owner {owner}"));
    let derive = prefix
        .rsplit("#[derive(")
        .next()
        .unwrap_or_else(|| panic!("missing derive for {owner}"));
    assert!(
        derive.starts_with("Debug, PartialEq, Eq)]"),
        "{owner} must preserve Debug/PartialEq/Eq without infallible Clone"
    );
}

fn assert_regression_contract(contract_tests: &str, parity_tests: &str) {
    assert_pattern_checks(&[
        PatternCheck::new("DCT re-emission parity contract", parity_tests).required(&[
            "valid_dct_reemission_bytes_remain_exact",
            "golden/dct_reemit_grayscale.hex",
            "golden/dct_reemit_420.hex",
            "assert_exact_hex(&encoded, golden)",
            "0xeb28_3c65_094a_b76a",
            "0x44da_dc89_e927_2c00",
            "valid_dct_reemission_preserves_quantized_component_parity",
        ]),
        PatternCheck::new("DCT re-emission invalid-input contract", contract_tests).required(&[
            "dct_reemission_rejects_each_invalid_sampling_factor_and_mcu_sum",
            "dct_reemission_rejects_grid_and_quantized_block_count_mismatches",
            "dct_reemission_rejects_nonbaseline_coefficient_categories",
            "dct_reemission_accepts_baseline_coefficient_category_boundaries",
            "dct_reemission_rejects_nonbaseline_and_unshared_quantization_tables",
            "dct_reemission_ignores_metadata_that_does_not_affect_baseline_output",
        ]),
    ]);
}

fn assert_validation_order(transcode: &str) {
    let function = transcode
        .split("pub fn encode_baseline_dct_image")
        .nth(1)
        .expect("DCT re-emission entrypoint")
        .split("fn validate_dct_reemission_live_bytes")
        .next()
        .expect("DCT re-emission function body");
    let validation_position = function
        .find("validate_baseline_dct_image(image)")
        .expect("typed input validation");
    let capacity_position = function
        .find("jpeg_baseline_entropy_capacity_bytes")
        .expect("entropy capacity planning");
    let entropy_position = function
        .find("encode_dct_entropy")
        .expect("entropy encoding");
    let frame_position = function
        .find("assemble_jpeg_baseline_frame_with_quant_tables")
        .expect("frame assembly");
    assert!(validation_position < capacity_position);
    assert!(capacity_position < entropy_position);
    assert!(entropy_position < frame_position);
}
