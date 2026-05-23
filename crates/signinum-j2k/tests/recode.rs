// SPDX-License-Identifier: Apache-2.0

use signinum_core::{CodecError, CompressedPayloadKind, CompressedTransferSyntax};
use signinum_j2k::{
    encode_j2k_lossless, J2kBlockCodingMode, J2kEncodeValidation, J2kError,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kToHtj2kMode, J2kToHtj2kOptions,
    ReversibleTransform,
};
use signinum_j2k_native::{DecodeSettings, EncodeOptions, Image};
use signinum_test_support::{patterned_gray8, patterned_rgb8};

fn decode_native(codestream: &[u8]) -> signinum_j2k_native::RawBitmap {
    Image::new(codestream, &DecodeSettings::default())
        .expect("codestream should parse")
        .decode_native()
        .expect("codestream should decode")
}

fn lossless_options(block_coding_mode: J2kBlockCodingMode) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions {
        block_coding_mode,
        validation: J2kEncodeValidation::External,
        ..J2kLosslessEncodeOptions::default()
    }
}

fn native_encode_options(reversible: bool, use_mct: bool) -> EncodeOptions {
    EncodeOptions {
        reversible,
        use_mct,
        use_ht_block_coding: false,
        num_decomposition_levels: 1,
        validate_high_throughput_codestream: false,
        ..EncodeOptions::default()
    }
}

fn wrap_codestream_jp2(
    codestream: &[u8],
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    colorspace_enum: u32,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&[0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0D, 0x0A, 0x87, 0x0A]);
    bytes.extend_from_slice(&[
        0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'p', b'2', b' ', 0, 0, 0, 0, b'j', b'p', b'2',
        b' ',
    ]);

    let bpc = bit_depth.saturating_sub(1);
    bytes.extend_from_slice(&[
        0, 0, 0, 45, b'j', b'p', b'2', b'h', 0, 0, 0, 22, b'i', b'h', b'd', b'r',
    ]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&components.to_be_bytes());
    bytes.extend_from_slice(&[bpc, 7, 0, 0]);
    bytes.extend_from_slice(&[0, 0, 0, 15, b'c', b'o', b'l', b'r', 1, 0, 0]);
    bytes.extend_from_slice(&colorspace_enum.to_be_bytes());

    let len = (8 + codestream.len()) as u32;
    bytes.extend_from_slice(&len.to_be_bytes());
    bytes.extend_from_slice(b"jp2c");
    bytes.extend_from_slice(codestream);
    bytes
}

#[test]
fn classic_lossless_53_rgb_recode_to_htj2k_decodes_pixel_exact() {
    let width = 64;
    let height = 64;
    let pixels = patterned_rgb8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let classic = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            reversible_transform: ReversibleTransform::Rct53,
            ..lossless_options(J2kBlockCodingMode::Classic)
        },
    )
    .expect("classic lossless encode")
    .codestream;

    let recoded =
        signinum_j2k::recode_j2k_to_htj2k_lossless(&classic, J2kToHtj2kOptions::default())
            .expect("coefficient-domain recode");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::CoefficientPreserving);
    assert_eq!(
        recoded.report.output_transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless
    );
    assert!(recoded.bytes.starts_with(&[0xff, 0x4f]));

    let decoded = decode_native(&recoded.bytes);
    assert_eq!((decoded.width, decoded.height), (width, height));
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn classic_lossless_53_gray16_recode_to_htj2k_decodes_pixel_exact() {
    let width = 64;
    let height = 64;
    let mut pixels = Vec::new();
    for sample in patterned_gray8(width, height) {
        let value = u16::from(sample) * 257;
        pixels.extend_from_slice(&value.to_le_bytes());
    }
    let samples = J2kLosslessSamples::new(&pixels, width, height, 1, 16, false)
        .expect("valid gray16 samples");
    let classic = encode_j2k_lossless(samples, &lossless_options(J2kBlockCodingMode::Classic))
        .expect("classic lossless encode")
        .codestream;

    let recoded =
        signinum_j2k::recode_j2k_to_htj2k_lossless(&classic, J2kToHtj2kOptions::default())
            .expect("coefficient-domain recode");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::CoefficientPreserving);
    let decoded = decode_native(&recoded.bytes);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn jp2_wrapped_classic_lossless_53_recode_emits_raw_htj2k_codestream() {
    let width = 64;
    let height = 64;
    let pixels = patterned_rgb8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let classic = encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions {
            reversible_transform: ReversibleTransform::Rct53,
            ..lossless_options(J2kBlockCodingMode::Classic)
        },
    )
    .expect("classic lossless encode")
    .codestream;
    let jp2 = wrap_codestream_jp2(&classic, width, height, 3, 8, 16);

    let recoded = signinum_j2k::recode_j2k_to_htj2k_lossless(&jp2, J2kToHtj2kOptions::default())
        .expect("JP2 coefficient-domain recode");

    assert_eq!(
        recoded.report.input_payload_kind,
        CompressedPayloadKind::Jp2File
    );
    assert_eq!(
        recoded.report.output_payload_kind,
        CompressedPayloadKind::Jpeg2000Codestream
    );
    assert!(recoded.bytes.starts_with(&[0xff, 0x4f]));
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn already_raw_htj2k_lossless_returns_passthrough() {
    let width = 32;
    let height = 32;
    let pixels = patterned_gray8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("valid gray samples");
    let htj2k = encode_j2k_lossless(
        samples,
        &lossless_options(J2kBlockCodingMode::HighThroughput),
    )
    .expect("HTJ2K encode")
    .codestream;

    let recoded = signinum_j2k::recode_j2k_to_htj2k_lossless(&htj2k, J2kToHtj2kOptions::default())
        .expect("passthrough recode");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::Passthrough);
    assert_eq!(recoded.bytes, htj2k);
}

#[test]
fn malformed_input_returns_explicit_error() {
    let err =
        signinum_j2k::recode_j2k_to_htj2k_lossless(b"not jpeg 2000", J2kToHtj2kOptions::default())
            .expect_err("malformed input should fail");

    assert!(matches!(err, J2kError::Unsupported(_)) || err.is_truncated());
}

#[test]
fn lossy_97_source_is_rejected_for_lossless_53_coefficient_recode() {
    let width = 32;
    let height = 32;
    let pixels = patterned_gray8(width, height);
    let lossy = signinum_j2k_native::encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &native_encode_options(false, false),
    )
    .expect("lossy 9/7 encode");

    let err = signinum_j2k::recode_j2k_to_htj2k_lossless(&lossy, J2kToHtj2kOptions::default())
        .expect_err("lossy source should fail");

    assert!(matches!(err, J2kError::Unsupported(_)));
}

#[test]
fn signed_source_is_rejected_before_recode() {
    let pixels = [0_u8, 1, 255, 127];
    let signed = signinum_j2k_native::encode(
        &pixels,
        2,
        2,
        1,
        8,
        true,
        &native_encode_options(true, false),
    )
    .expect("signed classic encode");

    let err = signinum_j2k::recode_j2k_to_htj2k_lossless(&signed, J2kToHtj2kOptions::default())
        .expect_err("signed source should fail");

    assert!(matches!(err, J2kError::Unsupported(_)));
}

#[test]
fn unsupported_component_count_is_rejected() {
    let pixels = vec![127_u8; 16 * 16 * 4];
    let four_component = signinum_j2k_native::encode(
        &pixels,
        16,
        16,
        4,
        8,
        false,
        &native_encode_options(true, false),
    )
    .expect("four-component classic encode");

    let err =
        signinum_j2k::recode_j2k_to_htj2k_lossless(&four_component, J2kToHtj2kOptions::default())
            .expect_err("four-component source should fail");

    assert!(matches!(err, J2kError::Unsupported(_)));
}
