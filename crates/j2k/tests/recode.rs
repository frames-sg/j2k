// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossless, extract_j2k_codestream_payload, wrap_j2k_codestream, J2kBlockCodingMode,
    J2kChannelAssociation, J2kChannelDefinition, J2kChannelType, J2kColorSpec, J2kComponentMapping,
    J2kComponentMappingType, J2kEncodeValidation, J2kError, J2kFileBoxMetadata, J2kFileColorSpec,
    J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kPaletteColumn,
    J2kPaletteMetadata, J2kToHtj2kMode, J2kToHtj2kOptions, ReversibleTransform,
};
use j2k_core::{CodecError, Colorspace, CompressedPayloadKind, CompressedTransferSyntax};
use j2k_native::{
    encode_precomputed_htj2k_97, encode_precomputed_j2k_53, encode_typed_component_planes_53,
    DecodeSettings, EncodeOptions, EncodeTypedComponentPlane, Image, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output,
    PrecomputedHtj2k53Component, PrecomputedHtj2k53Image, PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image,
};
use j2k_test_support::{patterned_gray8, patterned_rgb8, wrap_jp2_codestream};

fn decode_native(codestream: &[u8]) -> j2k_native::RawBitmap {
    Image::new(codestream, &DecodeSettings::default())
        .expect("codestream should parse")
        .decode_native()
        .expect("codestream should decode")
}

fn lossless_options(block_coding_mode: J2kBlockCodingMode) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_block_coding_mode(block_coding_mode)
        .with_validation(J2kEncodeValidation::External)
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

fn rgb_channel_definitions() -> [J2kChannelDefinition; 3] {
    [
        J2kChannelDefinition {
            channel_index: 0,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 1 },
        },
        J2kChannelDefinition {
            channel_index: 1,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 2 },
        },
        J2kChannelDefinition {
            channel_index: 2,
            channel_type: J2kChannelType::Color,
            association: J2kChannelAssociation::Color { index: 3 },
        },
    ]
}

fn rgb_palette() -> J2kPaletteMetadata {
    J2kPaletteMetadata {
        columns: vec![
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
            J2kPaletteColumn {
                bit_depth: 8,
                signed: false,
            },
        ],
        entries: vec![vec![2, 20, 200], vec![200, 40, 3]],
    }
}

fn rgb_palette_mappings() -> [J2kComponentMapping; 3] {
    [
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column: 0 },
        },
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column: 1 },
        },
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Palette { column: 2 },
        },
    ]
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
        &lossless_options(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::Rct53),
    )
    .expect("classic lossless encode")
    .codestream;

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&classic, J2kToHtj2kOptions::default())
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

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&classic, J2kToHtj2kOptions::default())
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
        &lossless_options(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::Rct53),
    )
    .expect("classic lossless encode")
    .codestream;
    let jp2 = wrap_jp2_codestream(&classic, width, height, 3, 8, 16);

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&jp2, J2kToHtj2kOptions::default())
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

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&htj2k, J2kToHtj2kOptions::default())
        .expect("passthrough recode");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::Passthrough);
    assert_eq!(recoded.bytes, htj2k);
}

#[test]
fn raw_htj2k_lossless_can_be_wrapped_as_jph_without_reencode() {
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

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &htj2k,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("raw HTJ2K wraps as JPH");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::CodestreamPreserving);
    assert_eq!(
        recoded.report.input_payload_kind,
        CompressedPayloadKind::Jpeg2000Codestream
    );
    assert_eq!(
        recoded.report.output_payload_kind,
        CompressedPayloadKind::JphFile
    );
    let payload =
        extract_j2k_codestream_payload(&recoded.bytes).expect("recoded codestream payload");
    assert_eq!(payload.codestream(), htj2k.as_slice());
    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect JPH");
    assert_eq!(support.payload_kind, CompressedPayloadKind::JphFile);
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn recode_can_emit_jph_file_wrapper() {
    let width = 32;
    let height = 32;
    let pixels = patterned_gray8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("valid gray samples");
    let classic = encode_j2k_lossless(samples, &lossless_options(J2kBlockCodingMode::Classic))
        .expect("classic lossless encode")
        .codestream;

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &classic,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("JPH recode");

    assert_eq!(
        recoded.report.output_payload_kind,
        CompressedPayloadKind::JphFile
    );
    assert!(recoded
        .bytes
        .starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']));
    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect JPH");
    assert_eq!(support.payload_kind, CompressedPayloadKind::JphFile);
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn recode_jph_preserves_input_icc_color_spec() {
    let width = 64;
    let height = 64;
    let pixels = patterned_rgb8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let classic = encode_j2k_lossless(
        samples,
        &lossless_options(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::Rct53),
    )
    .expect("classic lossless encode")
    .codestream;
    let jp2 = wrap_j2k_codestream(
        &classic,
        J2kFileWrapOptions::jp2().with_color(J2kFileColorSpec::IccProfile(b"test-icc")),
    )
    .expect("wrap JP2 with ICC");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &jp2,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("JPH recode preserves ICC");

    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded JPH");
    let metadata = support.file_metadata.as_ref().expect("JPH metadata");
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [J2kColorSpec::IccProfile { profile }] if profile == b"test-icc"
    ));
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn recode_jph_preserves_multiple_colr_boxes_for_coefficient_path() {
    let width = 64;
    let height = 64;
    let pixels = patterned_rgb8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let classic = encode_j2k_lossless(
        samples,
        &lossless_options(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::Rct53),
    )
    .expect("classic lossless encode")
    .codestream;
    let colors = [
        J2kFileColorSpec::Enumerated(Colorspace::SRgb),
        J2kFileColorSpec::IccProfile(b"test-icc"),
    ];
    let jp2 = wrap_j2k_codestream(
        &classic,
        J2kFileWrapOptions::jp2().with_color_specs(&colors),
    )
    .expect("wrap JP2 with multiple COLR boxes");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &jp2,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("JPH recode preserves multiple COLR boxes");

    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded JPH");
    let metadata = support.file_metadata.expect("JPH metadata");
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [
            J2kColorSpec::Enumerated { value: 16 },
            J2kColorSpec::IccProfile { profile },
        ] if profile == b"test-icc"
    ));
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn recode_jph_preserves_channel_definition_metadata_for_coefficient_path() {
    let width = 64;
    let height = 64;
    let pixels = patterned_rgb8(width, height);
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let classic = encode_j2k_lossless(
        samples,
        &lossless_options(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::Rct53),
    )
    .expect("classic lossless encode")
    .codestream;
    let channels = rgb_channel_definitions();
    let jp2 = wrap_j2k_codestream(
        &classic,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &[],
                channel_definitions: &channels,
            }),
    )
    .expect("wrap JP2 with CDEF");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &jp2,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("JPH recode preserves CDEF");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::CoefficientPreserving);
    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded JPH");
    let metadata = support.file_metadata.expect("JPH metadata");
    assert_eq!(metadata.channel_definitions, channels);
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn recode_jph_drops_palette_metadata_on_pixel_fallback() {
    let width = 16;
    let height = 16;
    let indices = (0..width * height)
        .map(|idx| (idx & 1) as u8)
        .collect::<Vec<_>>();
    let samples =
        J2kLosslessSamples::new(&indices, width, height, 1, 8, false).expect("palette indices");
    let classic = encode_j2k_lossless(samples, &lossless_options(J2kBlockCodingMode::Classic))
        .expect("classic lossless encode")
        .codestream;
    let palette = rgb_palette();
    let mappings = rgb_palette_mappings();
    let channels = rgb_channel_definitions();
    let jp2 = wrap_j2k_codestream(
        &classic,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: Some(&palette),
                component_mappings: &mappings,
                channel_definitions: &channels,
            }),
    )
    .expect("wrap paletted JP2");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &jp2,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("JPH recode uses pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(recoded.report.components, 3);
    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded JPH");
    let metadata = support.file_metadata.expect("JPH metadata");
    assert!(matches!(
        metadata.color_specs.as_slice(),
        [J2kColorSpec::Enumerated { value: 16 }]
    ));
    assert!(metadata.palette.is_none());
    assert!(metadata.component_mappings.is_empty());
    assert!(metadata.channel_definitions.is_empty());
    let decoded = Image::new(&recoded.bytes, &DecodeSettings::default())
        .expect("recoded direct RGB JPH")
        .decode_native_components()
        .expect("resolved RGB component decode");
    assert_eq!(decoded.planes().len(), 3);
    let expected_red = indices
        .iter()
        .map(|index| if *index == 0 { 2 } else { 200 })
        .collect::<Vec<_>>();
    let expected_green = indices
        .iter()
        .map(|index| if *index == 0 { 20 } else { 40 })
        .collect::<Vec<_>>();
    let expected_blue = indices
        .iter()
        .map(|index| if *index == 0 { 200 } else { 3 })
        .collect::<Vec<_>>();
    assert_eq!(decoded.planes()[0].data(), expected_red);
    assert_eq!(decoded.planes()[1].data(), expected_green);
    assert_eq!(decoded.planes()[2].data(), expected_blue);
}

#[test]
fn recode_jph_drops_component_mapping_metadata_on_sampled_pixel_fallback() {
    let width = 4;
    let height = 4;
    let red = [
        10_u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
    ];
    let green = [20_u8, 50, 90, 130];
    let blue = [30_u8, 60, 100, 140];
    let planes = [
        EncodeTypedComponentPlane {
            data: &red,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 8,
            signed: false,
        },
        EncodeTypedComponentPlane {
            data: &green,
            x_rsiz: 2,
            y_rsiz: 2,
            bit_depth: 8,
            signed: false,
        },
        EncodeTypedComponentPlane {
            data: &blue,
            x_rsiz: 2,
            y_rsiz: 2,
            bit_depth: 8,
            signed: false,
        },
    ];
    let classic = encode_typed_component_planes_53(
        &planes,
        width,
        height,
        &native_encode_options(true, false),
    )
    .expect("sampled direct-mapped codestream");
    let mappings = [
        J2kComponentMapping {
            component_index: 0,
            mapping_type: J2kComponentMappingType::Direct,
        },
        J2kComponentMapping {
            component_index: 1,
            mapping_type: J2kComponentMappingType::Direct,
        },
        J2kComponentMapping {
            component_index: 2,
            mapping_type: J2kComponentMappingType::Direct,
        },
    ];
    let jp2 = wrap_j2k_codestream(
        &classic,
        J2kFileWrapOptions::jp2()
            .with_color(J2kFileColorSpec::Enumerated(Colorspace::SRgb))
            .with_metadata(J2kFileBoxMetadata {
                palette: None,
                component_mappings: &mappings,
                channel_definitions: &[],
            }),
    )
    .expect("wrap sampled direct-mapped JP2");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(
        &jp2,
        J2kToHtj2kOptions::new(
            CompressedPayloadKind::JphFile,
            j2k::J2kProgressionOrder::Lrcp,
            J2kEncodeValidation::CpuRoundTrip,
        ),
    )
    .expect("sampled direct-mapped fallback recodes as resolved pixels");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded JPH");
    let metadata = support.file_metadata.as_ref().expect("JPH metadata");
    assert!(metadata.palette.is_none());
    assert!(metadata.component_mappings.is_empty());
    assert!(!support.has_component_subsampling());
    assert_eq!(decode_native(&recoded.bytes).data, decode_native(&jp2).data);
}

#[test]
fn malformed_input_returns_explicit_error() {
    let err = j2k::recode_j2k_to_htj2k_lossless(b"not jpeg 2000", J2kToHtj2kOptions::default())
        .expect_err("malformed input should fail");

    assert!(matches!(err, J2kError::Unsupported(_)) || err.is_truncated());
}

#[test]
fn lossy_97_source_uses_pixel_preserving_recode() {
    let width = 32;
    let height = 32;
    let pixels = patterned_gray8(width, height);
    let lossy = j2k_native::encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &native_encode_options(false, false),
    )
    .expect("lossy 9/7 encode");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&lossy, J2kToHtj2kOptions::default())
        .expect("lossy source should use pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(
        decode_native(&recoded.bytes).data,
        decode_native(&lossy).data
    );
}

#[test]
fn signed_source_uses_pixel_preserving_recode() {
    let pixels = [0_u8, 1, 255, 127];
    let signed = j2k_native::encode(
        &pixels,
        2,
        2,
        1,
        8,
        true,
        &native_encode_options(true, false),
    )
    .expect("signed classic encode");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&signed, J2kToHtj2kOptions::default())
        .expect("signed source should use pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn four_component_source_uses_pixel_preserving_recode() {
    let pixels = vec![127_u8; 16 * 16 * 4];
    let four_component = j2k_native::encode(
        &pixels,
        16,
        16,
        4,
        8,
        false,
        &native_encode_options(true, false),
    )
    .expect("four-component classic encode");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&four_component, J2kToHtj2kOptions::default())
        .expect("four-component source should use pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(decode_native(&recoded.bytes).data, pixels);
}

#[test]
fn mixed_typed_source_uses_pixel_preserving_recode() {
    let unsigned = [3_u8, 17, 99, 201];
    let signed_values = [-12_i16, -1, 0, 511];
    let signed = signed_values
        .iter()
        .flat_map(|sample| {
            let raw = u16::try_from(i32::from(*sample) & 0x0fff).expect("masked 12-bit sample");
            raw.to_le_bytes()
        })
        .collect::<Vec<_>>();
    let planes = [
        EncodeTypedComponentPlane {
            data: &unsigned,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 8,
            signed: false,
        },
        EncodeTypedComponentPlane {
            data: &signed,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 12,
            signed: true,
        },
    ];
    let source =
        encode_typed_component_planes_53(&planes, 2, 2, &native_encode_options(true, false))
            .expect("mixed typed classic source");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&source, J2kToHtj2kOptions::default())
        .expect("mixed typed source should use pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(recoded.report.components, 2);
    assert_eq!(recoded.report.bit_depth, 12);
    let decoded = Image::new(&recoded.bytes, &DecodeSettings::default())
        .expect("recoded mixed typed should parse")
        .decode_native_components()
        .expect("recoded mixed typed should decode as components");
    assert_eq!(decoded.planes()[0].bit_depth(), 8);
    assert!(!decoded.planes()[0].signed());
    assert_eq!(decoded.planes()[0].data(), unsigned);
    assert_eq!(decoded.planes()[1].bit_depth(), 12);
    assert!(decoded.planes()[1].signed());
    assert_eq!(
        decoded.planes()[1].data(),
        signed_values
            .iter()
            .flat_map(|sample| sample.to_le_bytes())
            .collect::<Vec<_>>()
            .as_slice()
    );
}

#[test]
fn high_bit_source_uses_pixel_preserving_recode() {
    let samples = [0_u32, 1, (1_u32 << 28) + 17, (1_u32 << 29) - 1];
    let pixels = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let source_options = EncodeOptions {
        reversible: true,
        use_mct: false,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let source =
        j2k_native::encode(&pixels, 2, 2, 1, 29, false, &source_options).expect("gray29 source");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&source, J2kToHtj2kOptions::default())
        .expect("high-bit source should use pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(recoded.report.components, 1);
    assert_eq!(recoded.report.bit_depth, 29);
    let decoded = decode_native(&recoded.bytes);
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn lossy_sampled_source_uses_pixel_fallback_and_preserves_sampling() {
    let source = PrecomputedHtj2k97Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![
            PrecomputedHtj2k97Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt97(16, 16),
            },
            PrecomputedHtj2k97Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt97(8, 8),
            },
            PrecomputedHtj2k97Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt97(8, 8),
            },
        ],
    };
    let lossy = encode_precomputed_htj2k_97(&source, &native_encode_options(false, false))
        .expect("sampled lossy HTJ2K fixture");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&lossy, J2kToHtj2kOptions::default())
        .expect("sampled lossy source should use pixel fallback");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::PixelPreserving);
    assert_eq!(
        decode_native(&recoded.bytes).data,
        decode_native(&lossy).data
    );

    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded HTJ2K");
    assert_eq!(
        support.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless
    );
    assert!(support.has_component_subsampling());
    let sampling = support
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
}

#[test]
fn recode_subsampled_classic_53_uses_coefficient_path_and_preserves_sampling() {
    let source = PrecomputedHtj2k53Image {
        width: 16,
        height: 16,
        bit_depth: 8,
        signed: false,
        components: vec![
            PrecomputedHtj2k53Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: zero_dwt53(16, 16),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
            PrecomputedHtj2k53Component {
                x_rsiz: 2,
                y_rsiz: 2,
                dwt: zero_dwt53(8, 8),
            },
        ],
    };
    let classic = encode_precomputed_j2k_53(&source, &native_encode_options(true, false))
        .expect("sampled classic 5/3 fixture");

    let recoded = j2k::recode_j2k_to_htj2k_lossless(&classic, J2kToHtj2kOptions::default())
        .expect("sampled coefficient-domain recode");

    assert_eq!(recoded.report.mode, J2kToHtj2kMode::CoefficientPreserving);
    assert_eq!(
        decode_native(&recoded.bytes).data,
        decode_native(&classic).data
    );

    let support = j2k::J2kDecoder::inspect_support(&recoded.bytes).expect("inspect recoded HTJ2K");
    assert_eq!(
        support.transfer_syntax,
        CompressedTransferSyntax::HtJpeg2000Lossless
    );
    assert!(support.has_component_subsampling());
    let sampling = support
        .components
        .iter()
        .map(|component| (component.x_rsiz, component.y_rsiz))
        .collect::<Vec<_>>();
    assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
}

fn zero_dwt53(width: u32, height: u32) -> J2kForwardDwt53Output {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    J2kForwardDwt53Output {
        ll: vec![0.0; (low_width * low_height) as usize],
        ll_width: low_width,
        ll_height: low_height,
        levels: vec![J2kForwardDwt53Level {
            hl: vec![0.0; (high_width * low_height) as usize],
            lh: vec![0.0; (low_width * high_height) as usize],
            hh: vec![0.0; (high_width * high_height) as usize],
            width,
            height,
            low_width,
            low_height,
            high_width,
            high_height,
        }],
    }
}

fn zero_dwt97(width: u32, height: u32) -> J2kForwardDwt97Output {
    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    J2kForwardDwt97Output {
        ll: vec![0.0; (low_width * low_height) as usize],
        ll_width: low_width,
        ll_height: low_height,
        levels: vec![J2kForwardDwt97Level {
            hl: vec![0.0; (high_width * low_height) as usize],
            lh: vec![0.0; (low_width * high_height) as usize],
            hh: vec![0.0; (high_width * high_height) as usize],
            width,
            height,
            low_width,
            low_height,
            high_width,
            high_height,
        }],
    }
}
