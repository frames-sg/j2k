// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use j2k::{
    encode_j2k_lossless, encode_j2k_lossless_components, encode_j2k_lossless_typed_components,
    encode_j2k_lossless_with_accelerator, encode_j2k_lossless_with_roi_regions,
    j2k_lossless_decomposition_levels, j2k_lossless_decomposition_levels_for_options,
    j2k_lossless_decomposition_levels_for_progression, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessComponentPlane, J2kLosslessComponentSamples,
    J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLosslessTypedComponentPlane,
    J2kLosslessTypedComponentSamples, J2kLossyEncodeOptions, J2kMarkerSegment, J2kProgressionOrder,
    J2kRoiRegion, ReversibleTransform,
};
use j2k::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kCodeBlockStyle, J2kDeinterleaveToF32Job,
    J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kEncodeStageError, J2kEncodeStageResult,
    J2kHtCodeBlockEncodeJob, J2kPacketizationEncodeJob, J2kQuantizeSubbandJob, J2kSubBandType,
    J2kTier1CodeBlockEncodeJob,
};
use j2k_core::{BackendKind, CodecError};
use j2k_native::{inspect_j2k_codestream_header, DecodeSettings, DecoderContext, Image};

fn masked_u8(value: usize) -> u8 {
    u8::try_from(value & 0xff).expect("masked fixture byte fits u8")
}

fn clamped_u8(value: i32) -> u8 {
    u8::try_from(value.clamp(0, 255)).expect("clamped fixture byte fits u8")
}

fn decode_native(codestream: &[u8]) -> j2k_native::RawBitmap {
    Image::new(codestream, &DecodeSettings::default())
        .expect("encoded codestream should parse")
        .decode_native()
        .expect("encoded codestream should decode")
}

fn strict_decode_native(codestream: &[u8]) -> j2k_native::RawBitmap {
    Image::new(
        codestream,
        &DecodeSettings {
            strict: true,
            ..DecodeSettings::default()
        },
    )
    .expect("strict encoded codestream should parse")
    .decode_native()
    .expect("strict encoded codestream should decode")
}

fn cpu_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default().with_backend(EncodeBackendPreference::CpuOnly)
}

fn auto_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default().with_backend(EncodeBackendPreference::Auto)
}

fn require_device_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default().with_backend(EncodeBackendPreference::RequireDevice)
}

#[test]
fn backend_preference_helpers_select_clear_routes() {
    assert_eq!(
        J2kLosslessEncodeOptions::default()
            .with_accelerated_backend()
            .backend,
        EncodeBackendPreference::Auto
    );
    assert_eq!(
        J2kLosslessEncodeOptions::default()
            .with_cpu_only_backend()
            .backend,
        EncodeBackendPreference::CpuOnly
    );
    assert_eq!(
        J2kLosslessEncodeOptions::default()
            .with_strict_device_backend()
            .backend,
        EncodeBackendPreference::RequireDevice
    );
    assert_eq!(
        J2kLossyEncodeOptions::default()
            .with_accelerated_backend()
            .backend,
        EncodeBackendPreference::Auto
    );
}

#[test]
fn lossless_component_plane_encode_preserves_sampling_for_classic_and_htj2k() {
    let luma = vec![128_u8; 16 * 16];
    let chroma_blue = vec![96_u8; 8 * 8];
    let chroma_red = vec![160_u8; 8 * 8];
    let planes = [
        J2kLosslessComponentPlane {
            data: &luma,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        J2kLosslessComponentPlane {
            data: &chroma_blue,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        J2kLosslessComponentPlane {
            data: &chroma_red,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let samples =
        J2kLosslessComponentSamples::new(&planes, 16, 16, 8, false).expect("component samples");

    for block_coding_mode in [
        J2kBlockCodingMode::Classic,
        J2kBlockCodingMode::HighThroughput,
    ] {
        let encoded = encode_j2k_lossless_components(
            samples,
            &cpu_options()
                .with_block_coding_mode(block_coding_mode)
                .with_reversible_transform(ReversibleTransform::None53)
                .with_validation(J2kEncodeValidation::CpuRoundTrip),
        )
        .expect("component-plane encode");

        assert_eq!(encoded.components, 3);
        let image = Image::new(&encoded.codestream, &DecodeSettings::default())
            .expect("parse component-plane codestream");
        let mut context = DecoderContext::default();
        let components = image
            .decode_components_with_context(&mut context)
            .expect("decode component-plane codestream");
        let sampling = components
            .planes()
            .iter()
            .map(j2k_native::ComponentPlane::sampling)
            .collect::<Vec<_>>();

        assert_eq!(sampling, [(1, 1), (2, 2), (2, 2)]);
    }
}

#[test]
fn lossless_component_plane_encode_round_trips_full_resolution_gray29_planes() {
    let first_values = [0_u32, 1, (1_u32 << 28) + 7, (1_u32 << 29) - 1];
    let second_values = [42_u32, 65_537, (1_u32 << 27) + 9, (1_u32 << 28) - 1];
    let first = first_values
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let second = second_values
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessComponentPlane {
            data: &first,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        J2kLosslessComponentPlane {
            data: &second,
            x_rsiz: 1,
            y_rsiz: 1,
        },
    ];
    let samples = J2kLosslessComponentSamples::new(&planes, 2, 2, 29, false)
        .expect("high-bit component samples");

    let encoded = encode_j2k_lossless_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect("high-bit component-plane encode");

    assert_eq!(encoded.components, 2);
    assert_eq!(encoded.bit_depth, 29);
    let image = Image::new(&encoded.codestream, &DecodeSettings::default()).expect("parse gray29");
    let decoded = image
        .decode_native_components()
        .expect("decode native components");
    assert_eq!(decoded.planes().len(), 2);
    assert_eq!(decoded.planes()[0].data(), first);
    assert_eq!(decoded.planes()[1].data(), second);
}

#[test]
fn lossless_component_plane_encode_round_trips_sampled_high_bit_planes() {
    let luma = (0..16_u32)
        .flat_map(|idx| unsigned_31_bytes(((idx * 1_000_003) + 17) & ((1_u32 << 29) - 1)))
        .collect::<Vec<_>>();
    let chroma = (0..4_u32)
        .flat_map(|idx| unsigned_31_bytes(((idx * 3_000_011) + 255) & ((1_u32 << 29) - 1)))
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessComponentPlane {
            data: &luma,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        J2kLosslessComponentPlane {
            data: &chroma,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let samples = J2kLosslessComponentSamples::new(&planes, 4, 4, 29, false)
        .expect("sampled high-bit component samples");

    let encoded = encode_j2k_lossless_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("sampled high-bit component-plane encode");

    assert_eq!(encoded.components, 2);
    assert_eq!(encoded.bit_depth, 29);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse sampled high-bit component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode sampled high-bit components");

    assert_eq!(decoded.planes().len(), 2);
    assert_eq!(decoded.planes()[0].sampling(), (1, 1));
    assert_eq!(decoded.planes()[0].data(), luma);
    assert_eq!(decoded.planes()[1].sampling(), (2, 2));
    let chroma_expanded = expand_component_plane_bytes(&chroma, 4, 4, 2, 2, 4);
    assert_eq!(decoded.planes()[1].data(), chroma_expanded);
}

#[test]
fn lossless_component_plane_encode_round_trips_sampled_high_bit_multi_tile_planes() {
    let luma = (0..16_u32)
        .flat_map(|idx| unsigned_31_bytes((idx * 17_000_003) & ((1_u32 << 29) - 1)))
        .collect::<Vec<_>>();
    let chroma = [0_u32, 1, (1_u32 << 28) + 17, (1_u32 << 29) - 1]
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessComponentPlane {
            data: &luma,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        J2kLosslessComponentPlane {
            data: &chroma,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let samples =
        J2kLosslessComponentSamples::new(&planes, 4, 4, 29, false).expect("component samples");

    let encoded = encode_j2k_lossless_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_tile_size(Some((2, 2)))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("sampled high-bit multi-tile component-plane encode");

    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xff, 0x90])
        .count();
    assert_eq!(sot_count, 4);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse sampled high-bit multi-tile component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode sampled high-bit multi-tile components");

    assert_eq!(decoded.planes().len(), 2);
    assert_eq!(decoded.planes()[0].sampling(), (1, 1));
    assert_eq!(decoded.planes()[0].data(), luma);
    assert_eq!(decoded.planes()[1].sampling(), (2, 2));
    let chroma_expanded = expand_component_plane_bytes(&chroma, 4, 4, 2, 2, 4);
    assert_eq!(decoded.planes()[1].data(), chroma_expanded);
}

#[test]
fn lossless_component_plane_encode_round_trips_unaligned_sampled_high_bit_multi_tile_planes() {
    let luma = (0..25_u32)
        .flat_map(|idx| unsigned_31_bytes((idx * 11_000_017) & ((1_u32 << 29) - 1)))
        .collect::<Vec<_>>();
    let chroma = (0..9_u32)
        .flat_map(|idx| unsigned_31_bytes(((idx * 19_000_031) + 7) & ((1_u32 << 29) - 1)))
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessComponentPlane {
            data: &luma,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        J2kLosslessComponentPlane {
            data: &chroma,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    let samples =
        J2kLosslessComponentSamples::new(&planes, 5, 5, 29, false).expect("component samples");

    let encoded = encode_j2k_lossless_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_tile_size(Some((3, 3)))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("unaligned sampled high-bit multi-tile component-plane encode");

    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xff, 0x90])
        .count();
    assert_eq!(sot_count, 4);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse unaligned sampled high-bit multi-tile component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode unaligned sampled high-bit multi-tile components");

    assert_eq!(decoded.planes().len(), 2);
    assert_eq!(decoded.planes()[0].sampling(), (1, 1));
    assert_eq!(decoded.planes()[0].data(), luma);
    assert_eq!(decoded.planes()[1].sampling(), (2, 2));
    let chroma_expanded = expand_component_plane_bytes(&chroma, 5, 5, 2, 2, 4);
    assert_eq!(decoded.planes()[1].data(), chroma_expanded);
}

#[test]
fn lossless_component_plane_encode_round_trips_full_resolution_gray35_planes() {
    let mask = (1_u64 << 35) - 1;
    let gray = (0..64_u64 * 64)
        .flat_map(|idx| {
            let sample = idx
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add((idx / 64).wrapping_mul(1_048_583))
                & mask;
            unsigned_35_bytes(sample)
        })
        .collect::<Vec<_>>();
    let planes = [J2kLosslessComponentPlane {
        data: &gray,
        x_rsiz: 1,
        y_rsiz: 1,
    }];
    let samples =
        J2kLosslessComponentSamples::new(&planes, 64, 64, 35, false).expect("gray35 samples");

    let encoded = encode_j2k_lossless_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("gray35 component-plane encode");

    assert_eq!(encoded.components, 1);
    assert_eq!(encoded.bit_depth, 35);
    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse gray35 component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode gray35 components");

    assert_eq!(decoded.planes().len(), 1);
    assert_eq!(decoded.planes()[0].sampling(), (1, 1));
    assert_eq!(decoded.planes()[0].bit_depth(), 35);
    assert_eq!(decoded.planes()[0].data(), gray);
}

fn expand_component_plane_bytes(
    component: &[u8],
    width: u32,
    height: u32,
    x_rsiz: u8,
    y_rsiz: u8,
    bytes_per_sample: usize,
) -> Vec<u8> {
    let component_width = width.div_ceil(u32::from(x_rsiz)) as usize;
    let mut expanded = Vec::with_capacity(width as usize * height as usize * bytes_per_sample);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let component_idx =
                (y / usize::from(y_rsiz)) * component_width + (x / usize::from(x_rsiz));
            let start = component_idx * bytes_per_sample;
            expanded.extend_from_slice(&component[start..start + bytes_per_sample]);
        }
    }
    expanded
}

#[test]
fn lossless_typed_component_plane_encode_preserves_mixed_metadata() {
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
        J2kLosslessTypedComponentPlane {
            data: &unsigned,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 8,
            signed: false,
        },
        J2kLosslessTypedComponentPlane {
            data: &signed,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 12,
            signed: true,
        },
    ];
    let samples =
        J2kLosslessTypedComponentSamples::new(&planes, 2, 2).expect("typed component samples");

    let encoded = encode_j2k_lossless_typed_components(
        samples,
        &cpu_options()
            .with_reversible_transform(ReversibleTransform::None53)
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("typed component encode");

    assert_eq!(encoded.components, 2);
    assert_eq!(encoded.bit_depth, 12);
    assert!(!encoded.signed);

    let support = j2k::J2kDecoder::inspect_support(&encoded.codestream).expect("support info");
    assert!(support.has_mixed_bit_depths());
    assert!(support.has_signed_components());
    assert_eq!(support.components[0].bit_depth, 8);
    assert!(!support.components[0].signed);
    assert_eq!(support.components[1].bit_depth, 12);
    assert!(support.components[1].signed);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse typed component codestream");
    let mut context = DecoderContext::default();
    let decoded = image
        .decode_components_with_context(&mut context)
        .expect("decode typed component codestream");
    #[expect(
        clippy::cast_possible_truncation,
        reason = "decoded fixture samples are rounded within the asserted i16 domain"
    )]
    let decoded_signed = decoded.planes()[1]
        .samples()
        .iter()
        .map(|sample| sample.round() as i16)
        .collect::<Vec<_>>();

    assert_eq!(decoded_signed, signed_values);
}

#[test]
fn lossless_typed_component_plane_encode_round_trips_mixed_high_bit_metadata() {
    let unsigned = [0_u32, 1, (1_u32 << 28) + 17, (1_u32 << 29) - 1]
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let signed_values = [-2048_i16, -1, 0, 2047];
    let signed = signed_values
        .iter()
        .flat_map(|sample| {
            let raw = u16::try_from(i32::from(*sample) & 0x0fff).expect("masked 12-bit sample");
            raw.to_le_bytes()
        })
        .collect::<Vec<_>>();
    let canonical_signed = signed_values
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessTypedComponentPlane {
            data: &unsigned,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 29,
            signed: false,
        },
        J2kLosslessTypedComponentPlane {
            data: &signed,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 12,
            signed: true,
        },
    ];
    let samples = J2kLosslessTypedComponentSamples::new(&planes, 2, 2)
        .expect("mixed high-bit typed component samples");

    let encoded = encode_j2k_lossless_typed_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("mixed high-bit typed component encode");

    assert_eq!(encoded.components, 2);
    assert_eq!(encoded.bit_depth, 29);
    assert!(!encoded.signed);

    let support = j2k::J2kDecoder::inspect_support(&encoded.codestream).expect("support info");
    assert!(support.has_mixed_bit_depths());
    assert!(support.has_signed_components());
    assert_eq!(support.components[0].bit_depth, 29);
    assert!(!support.components[0].signed);
    assert_eq!(support.components[1].bit_depth, 12);
    assert!(support.components[1].signed);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse mixed high-bit typed component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode mixed high-bit typed component codestream");

    assert_eq!(decoded.planes()[0].data(), unsigned);
    assert_eq!(decoded.planes()[1].data(), canonical_signed);
}

#[test]
fn lossless_typed_component_plane_encode_round_trips_mixed_high_bit_multi_tile_metadata() {
    let unsigned = (0..16_u32)
        .flat_map(|idx| {
            let sample = (idx * 17_000_003 + (idx / 4) * 9_000_001) & ((1_u32 << 29) - 1);
            unsigned_31_bytes(sample)
        })
        .collect::<Vec<_>>();
    let signed_values = [
        -2048_i16, -1024, -257, -1, 0, 1, 255, 1024, 2047, 1536, -1536, 17, -17, 33, -33, 511,
    ];
    let signed = signed_values
        .iter()
        .flat_map(|sample| {
            let raw = u16::try_from(i32::from(*sample) & 0x0fff).expect("masked 12-bit sample");
            raw.to_le_bytes()
        })
        .collect::<Vec<_>>();
    let canonical_signed = signed_values
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessTypedComponentPlane {
            data: &unsigned,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 29,
            signed: false,
        },
        J2kLosslessTypedComponentPlane {
            data: &signed,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 12,
            signed: true,
        },
    ];
    let samples = J2kLosslessTypedComponentSamples::new(&planes, 4, 4)
        .expect("mixed high-bit typed component samples");

    let encoded = encode_j2k_lossless_typed_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_tile_size(Some((2, 2)))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("mixed high-bit typed multi-tile component encode");

    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xff, 0x90])
        .count();
    assert_eq!(sot_count, 4);

    let support = j2k::J2kDecoder::inspect_support(&encoded.codestream).expect("support info");
    assert!(support.has_mixed_bit_depths());
    assert!(support.has_signed_components());
    assert_eq!(support.components[0].bit_depth, 29);
    assert!(!support.components[0].signed);
    assert_eq!(support.components[1].bit_depth, 12);
    assert!(support.components[1].signed);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse mixed high-bit typed multi-tile component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode mixed high-bit typed multi-tile component codestream");

    assert_eq!(decoded.planes()[0].data(), unsigned);
    assert_eq!(decoded.planes()[1].data(), canonical_signed);
}

#[test]
fn lossless_typed_component_plane_encode_round_trips_mixed_35_bit_metadata() {
    let mask = (1_u64 << 35) - 1;
    let unsigned = (0..64_u64 * 64)
        .flat_map(|idx| {
            let sample = idx
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add((idx / 64).wrapping_mul(1_048_583))
                & mask;
            unsigned_35_bytes(sample)
        })
        .collect::<Vec<_>>();
    let signed_values = (0..64_i32 * 64)
        .map(|idx| ((idx * 37) % 4096) as i16 - 2048)
        .collect::<Vec<_>>();
    let signed = signed_values
        .iter()
        .flat_map(|sample| {
            let raw = u16::try_from(i32::from(*sample) & 0x0fff).expect("masked 12-bit sample");
            raw.to_le_bytes()
        })
        .collect::<Vec<_>>();
    let canonical_signed = signed_values
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let planes = [
        J2kLosslessTypedComponentPlane {
            data: &unsigned,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 35,
            signed: false,
        },
        J2kLosslessTypedComponentPlane {
            data: &signed,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 12,
            signed: true,
        },
    ];
    let samples = J2kLosslessTypedComponentSamples::new(&planes, 64, 64)
        .expect("mixed 35-bit typed component samples");

    let encoded = encode_j2k_lossless_typed_components(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(1))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("mixed 35-bit typed component encode");

    assert_eq!(encoded.components, 2);
    assert_eq!(encoded.bit_depth, 35);
    assert!(!encoded.signed);

    let support = j2k::J2kDecoder::inspect_support(&encoded.codestream).expect("support info");
    assert!(support.has_mixed_bit_depths());
    assert!(support.has_signed_components());
    assert_eq!(support.components[0].bit_depth, 35);
    assert!(!support.components[0].signed);
    assert_eq!(support.components[1].bit_depth, 12);
    assert!(support.components[1].signed);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("parse mixed 35-bit typed component codestream");
    let decoded = image
        .decode_native_components()
        .expect("decode mixed 35-bit typed component codestream");

    assert_eq!(decoded.planes()[0].bit_depth(), 35);
    assert_eq!(decoded.planes()[0].data(), unsigned);
    assert_eq!(decoded.planes()[1].bit_depth(), 12);
    assert_eq!(decoded.planes()[1].data(), canonical_signed);
}

#[test]
fn cpu_lossless_rectangular_roi_roundtrips_and_writes_rgn() {
    let pixels: Vec<_> = (0_usize..64 * 64)
        .map(|idx| u8::try_from(idx % 251).expect("fixture sample fits u8"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();
    let roi = [J2kRoiRegion {
        component: 0,
        x: 8,
        y: 12,
        width: 24,
        height: 20,
        shift: 12,
    }];

    let encoded = encode_j2k_lossless_with_roi_regions(
        samples,
        &cpu_options()
            .with_reversible_transform(ReversibleTransform::None53)
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        &roi,
    )
    .expect("lossless ROI encode");
    let rgn = marker_offset(&encoded.codestream, 0x5E).expect("RGN marker");

    assert_eq!(
        &encoded.codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x0C]
    );

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_multi_tile_rectangular_roi_roundtrips_and_writes_rgn() {
    let pixels: Vec<_> = (0_usize..96 * 80)
        .map(|idx| u8::try_from(idx % 251).expect("fixture sample fits u8"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 96, 80, 1, 8, false).unwrap();
    let roi = [J2kRoiRegion {
        component: 0,
        x: 24,
        y: 20,
        width: 48,
        height: 44,
        shift: 12,
    }];

    let encoded = encode_j2k_lossless_with_roi_regions(
        samples,
        &cpu_options()
            .with_reversible_transform(ReversibleTransform::None53)
            .with_tile_size(Some((32, 32)))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        &roi,
    )
    .expect("multi-tile lossless ROI encode");
    let rgn = marker_offset(&encoded.codestream, 0x5E).expect("RGN marker");
    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xff, 0x90])
        .count();

    assert_eq!(
        &encoded.codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x0C]
    );
    assert!(sot_count > 1, "fixture should exercise multi-tile output");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_htj2k_rectangular_roi_roundtrips_at_31_coded_bitplanes() {
    let pixels: Vec<_> = (0_usize..16 * 16)
        .map(|idx| u8::try_from(idx % 251).expect("fixture sample fits u8"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 16, 16, 1, 8, false).unwrap();
    let roi = [J2kRoiRegion {
        component: 0,
        x: 4,
        y: 5,
        width: 7,
        height: 6,
        shift: 23,
    }];

    let encoded = encode_j2k_lossless_with_roi_regions(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        &roi,
    )
    .expect("HTJ2K ROI encode at 31 coded bitplanes");
    let rgn = marker_offset(&encoded.codestream, 0x5E).expect("RGN marker");

    assert_eq!(
        &encoded.codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x17]
    );

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_high_bit_rectangular_roi_roundtrips_at_50_coded_bitplanes() {
    let pixels = (0_u32..16 * 16)
        .flat_map(|idx| unsigned_31_bytes((idx * 37) & ((1_u32 << 25) - 1)))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 16, 16, 1, 25, false).unwrap();
    let roi = [J2kRoiRegion {
        component: 0,
        x: 2,
        y: 2,
        width: 8,
        height: 8,
        shift: 25,
    }];

    let encoded = encode_j2k_lossless_with_roi_regions(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        &roi,
    )
    .expect("classic high-bit ROI encode at 50 coded bitplanes");
    let rgn = marker_offset(&encoded.codestream, 0x5E).expect("RGN marker");

    assert_eq!(
        &encoded.codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x19]
    );
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_high_bit_rectangular_roi_rejects_56_coded_bitplanes_explicitly() {
    let pixels = (0_u32..16 * 16)
        .flat_map(|idx| unsigned_31_bytes((idx * 37) & ((1_u32 << 25) - 1)))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 16, 16, 1, 25, false).unwrap();
    let roi = [J2kRoiRegion {
        component: 0,
        x: 2,
        y: 2,
        width: 8,
        height: 8,
        shift: 31,
    }];

    let err = encode_j2k_lossless_with_roi_regions(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::None53)
            .with_max_decomposition_levels(Some(0)),
        &roi,
    )
    .expect_err("classic ROI maxshift must report the coded-bitplane limit");

    let j2k::J2kError::NativeEncode { context, source } = err else {
        panic!("expected typed native unsupported error, got {err:?}");
    };
    assert_eq!(context, "native JPEG 2000 lossless ROI encode failed");
    assert!(matches!(
        std::error::Error::source(&source)
            .and_then(|source| source.downcast_ref::<j2k_native::EncodeError>()),
        Some(j2k_native::EncodeError::Unsupported {
            what: "ROI maxshift exceeds supported coded bitplane count",
        })
    ));
}

fn deinterleave_to_f32_for_test(job: J2kDeinterleaveToF32Job<'_>) -> Vec<Vec<f32>> {
    let num_components = usize::from(job.num_components);
    let bytes_per_sample = if job.bit_depth <= 8 { 1 } else { 2 };
    assert_eq!(
        job.pixels.len(),
        job.num_pixels * num_components * bytes_per_sample
    );

    let unsigned_offset = if job.signed {
        0
    } else {
        1_i32 << u32::from(job.bit_depth.saturating_sub(1))
    };
    let mut components = vec![vec![0.0; job.num_pixels]; num_components];
    for pixel_idx in 0..job.num_pixels {
        for (component_idx, component) in components.iter_mut().enumerate() {
            let sample_idx = pixel_idx * num_components + component_idx;
            let sample = if job.bit_depth <= 8 {
                let byte = job.pixels[sample_idx];
                if job.signed {
                    i16::from(i8::from_le_bytes([byte]))
                } else {
                    i16::try_from(i32::from(byte) - unsigned_offset)
                        .expect("level-shifted 8-bit sample fits in i16")
                }
            } else {
                let byte_idx = sample_idx * 2;
                let bytes = [job.pixels[byte_idx], job.pixels[byte_idx + 1]];
                if job.signed {
                    i16::from_le_bytes(bytes)
                } else {
                    i16::try_from(i32::from(u16::from_le_bytes(bytes)) - unsigned_offset)
                        .expect("level-shifted 16-bit sample fits in i16")
                }
            };
            component[pixel_idx] = f32::from(sample);
        }
    }
    components
}

fn native_subband(subband: J2kSubBandType) -> j2k_native::J2kSubBandType {
    match subband {
        J2kSubBandType::LowLow => j2k_native::J2kSubBandType::LowLow,
        J2kSubBandType::HighLow => j2k_native::J2kSubBandType::HighLow,
        J2kSubBandType::LowHigh => j2k_native::J2kSubBandType::LowHigh,
        J2kSubBandType::HighHigh => j2k_native::J2kSubBandType::HighHigh,
    }
}

fn native_code_block_style(style: J2kCodeBlockStyle) -> j2k_native::J2kCodeBlockStyle {
    style
}

fn public_encoded_j2k(block: j2k_native::EncodedJ2kCodeBlock) -> EncodedJ2kCodeBlock {
    block
}

fn public_encoded_ht(block: j2k_native::EncodedHtJ2kCodeBlock) -> EncodedHtJ2kCodeBlock {
    block
}

#[test]
fn default_lossless_options_use_auto_cpu_safe_profile() {
    let options = J2kLosslessEncodeOptions::default();

    assert_eq!(options.backend, EncodeBackendPreference::Auto);
    assert_eq!(options.block_coding_mode, J2kBlockCodingMode::Classic);
    assert_eq!(options.progression, J2kProgressionOrder::Lrcp);
    assert_eq!(options.max_decomposition_levels, None);
    assert_eq!(options.reversible_transform, ReversibleTransform::Rct53);
    assert_eq!(options.validation, J2kEncodeValidation::CpuRoundTrip);
}

#[test]
fn lossless_encode_can_skip_facade_cpu_validation_for_external_validation() {
    let pixels: Vec<u8> = (0_usize..8 * 8 * 3).map(|i| masked_u8(i * 17)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options().with_validation(J2kEncodeValidation::External),
    )
    .expect("lossless encode without facade CPU validation");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn cpu_htj2k_lossless_round_trips_gray8() {
    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(9)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options().with_block_coding_mode(J2kBlockCodingMode::HighThroughput),
    )
    .expect("HTJ2K lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert!(encoded
        .codestream
        .windows(2)
        .any(|window| window == [0xFF, 0x50]));
    let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
    assert_eq!(encoded.codestream[cod_offset + 12], 0x40);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn cpu_htj2k_rpcl_writes_cod_rpcl_and_tlm() {
    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(11)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_progression(J2kProgressionOrder::Rpcl),
    )
    .expect("HTJ2K RPCL lossless encode");

    let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
    assert_eq!(encoded.codestream[cod_offset + 5], 0x02);
    assert!(marker_offset(&encoded.codestream, 0x55).is_some());
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn cpu_lossless_all_progression_orders_write_cod_marker_and_round_trip() {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u8 {
        for x in 0..64u8 {
            pixels.push(x.wrapping_mul(3).wrapping_add(y));
            pixels.push(y.wrapping_mul(5).wrapping_add(x / 2));
            pixels.push(x.wrapping_mul(7).wrapping_sub(y.wrapping_mul(2)));
        }
    }

    for (progression, marker_byte) in [
        (J2kProgressionOrder::Lrcp, 0x00),
        (J2kProgressionOrder::Rlcp, 0x01),
        (J2kProgressionOrder::Rpcl, 0x02),
        (J2kProgressionOrder::Pcrl, 0x03),
        (J2kProgressionOrder::Cprl, 0x04),
    ] {
        let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).unwrap();
        let encoded = encode_j2k_lossless(
            samples,
            &cpu_options()
                .with_progression(progression)
                .with_reversible_transform(ReversibleTransform::None53),
        )
        .unwrap_or_else(|err| panic!("{progression:?} encode failed: {err}"));

        let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
        assert_eq!(encoded.codestream[cod_offset + 5], marker_byte);
        assert_eq!(
            decode_native(&encoded.codestream).data,
            pixels,
            "{progression:?} round trip"
        );
    }
}

#[test]
fn cpu_lossless_multi_tile_codestream_decodes() {
    let pixels: Vec<u8> = (0..96 * 80)
        .map(|index| masked_u8(index * 17 + index / 9))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 96, 80, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options().with_tile_size(Some((48, 40))))
        .expect("lossless multi-tile encode");

    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xff, 0x90])
        .count();
    assert_eq!(sot_count, 4);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 96);
    assert_eq!(decoded.height, 80);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_emits_packet_markers_that_strict_decode_uses() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 31) ^ (index / 7)))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options().with_marker_segments(&[
            J2kMarkerSegment::Tlm,
            J2kMarkerSegment::Plt,
            J2kMarkerSegment::Plm,
            J2kMarkerSegment::Sop,
            J2kMarkerSegment::Eph,
        ]),
    )
    .expect("lossless packet-marker encode");

    for marker in [0x55, 0x57, 0x58, 0x91, 0x92] {
        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xff, marker]),
            "marker FF{marker:02X} must be emitted"
        );
    }
    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_emits_ppm_and_ppt_that_strict_decode_uses() {
    for (marker, marker_byte) in [(J2kMarkerSegment::Ppm, 0x60), (J2kMarkerSegment::Ppt, 0x61)] {
        let pixels: Vec<u8> = (0..64 * 64)
            .map(|index| masked_u8((index * 17) ^ (index / 5)))
            .collect();
        let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let encoded = encode_j2k_lossless(
            samples,
            &cpu_options()
                .with_max_decomposition_levels(Some(0))
                .with_marker_segments(&[marker]),
        )
        .expect("lossless separated packet-header encode");

        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xff, marker_byte]),
            "marker FF{marker_byte:02X} must be emitted"
        );
        let decoded = strict_decode_native(&encoded.codestream);
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(decoded.data, pixels);
    }
}

#[test]
fn cpu_lossless_multi_tile_emits_ppm_and_ppt_that_strict_decode_uses() {
    for (marker, marker_byte) in [(J2kMarkerSegment::Ppm, 0x60), (J2kMarkerSegment::Ppt, 0x61)] {
        let pixels: Vec<u8> = (0..64 * 64)
            .map(|index| masked_u8((index * 29) ^ (index / 11)))
            .collect();
        let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let encoded = encode_j2k_lossless(
            samples,
            &cpu_options()
                .with_max_decomposition_levels(Some(0))
                .with_tile_size(Some((32, 32)))
                .with_marker_segments(&[marker]),
        )
        .expect("lossless multi-tile separated packet-header encode");

        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xff, marker_byte]),
            "marker FF{marker_byte:02X} must be emitted"
        );
        let sot_count = encoded
            .codestream
            .windows(2)
            .filter(|marker| *marker == [0xff, 0x90])
            .count();
        assert_eq!(sot_count, 4);
        let decoded = strict_decode_native(&encoded.codestream);
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(decoded.data, pixels);
    }
}

#[test]
fn cpu_lossless_emits_multiple_tile_parts_that_strict_decode_uses() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 19) ^ (index / 3)))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_max_decomposition_levels(Some(2))
            .with_tile_part_packet_limit(Some(1)),
    )
    .expect("lossless multi-tile-part encode");

    let tile_parts = sot_tile_part_fields(&encoded.codestream);
    assert!(tile_parts.len() > 1, "expected multiple tile-parts");
    for (index, (tile_index, tile_part_index, num_tile_parts)) in tile_parts.iter().enumerate() {
        assert_eq!(*tile_index, 0);
        assert_eq!(
            *tile_part_index,
            u8::try_from(index).expect("tile-part index fits u8")
        );
        assert_eq!(
            *num_tile_parts,
            u8::try_from(tile_parts.len()).expect("tile-part count fits u8")
        );
    }

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_emits_tlm_for_multiple_tile_parts() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 37) ^ (index / 13)))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_max_decomposition_levels(Some(2))
            .with_tile_part_packet_limit(Some(1))
            .with_marker_segments(&[J2kMarkerSegment::Tlm]),
    )
    .expect("lossless multi-tile-part TLM encode");

    let tlm = tlm_tile_part_lengths(&encoded.codestream);
    let sot = sot_tile_part_lengths(&encoded.codestream);
    assert!(sot.len() > 1, "expected multiple tile-parts");
    assert_eq!(tlm, sot);

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_emits_ppm_and_ppt_across_multiple_tile_parts_that_strict_decode_uses() {
    for (marker, marker_byte) in [(J2kMarkerSegment::Ppm, 0x60), (J2kMarkerSegment::Ppt, 0x61)] {
        let pixels: Vec<u8> = (0..64 * 64)
            .map(|index| masked_u8((index * 41) ^ (index / 19)))
            .collect();
        let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let encoded = encode_j2k_lossless(
            samples,
            &cpu_options()
                .with_max_decomposition_levels(Some(2))
                .with_tile_part_packet_limit(Some(1))
                .with_marker_segments(&[marker]),
        )
        .expect("lossless multi-tile-part separated packet-header encode");

        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xff, marker_byte]),
            "marker FF{marker_byte:02X} must be emitted"
        );
        assert!(
            sot_tile_part_fields(&encoded.codestream).len() > 1,
            "expected multiple tile-parts"
        );
        let decoded = strict_decode_native(&encoded.codestream);
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.num_components, 1);
        assert_eq!(decoded.data, pixels);
    }
}

#[test]
fn default_lossless_policy_enables_one_reversible_dwt_level_for_wsi_tiles() {
    let gray = vec![0; 64 * 64];
    let gray_samples = J2kLosslessSamples::new(&gray, 64, 64, 1, 8, false).unwrap();
    assert_eq!(j2k_lossless_decomposition_levels(gray_samples), 1);

    let rgb = vec![0; 512 * 512 * 3];
    let rgb_samples = J2kLosslessSamples::new(&rgb, 512, 512, 3, 8, false).unwrap();
    assert_eq!(j2k_lossless_decomposition_levels(rgb_samples), 1);
}

#[test]
fn default_lossless_policy_keeps_edge_tiles_undecomposed() {
    let gray = vec![0; 63 * 512];
    let samples = J2kLosslessSamples::new(&gray, 63, 512, 1, 8, false).unwrap();

    assert_eq!(j2k_lossless_decomposition_levels(samples), 0);
}

#[test]
fn rpcl_lossless_policy_reduces_base_resolution_to_64_or_less() {
    for (tile_size, expected_levels) in [(512usize, 3u8), (1024, 4), (2048, 5)] {
        let pixels = vec![0; tile_size * tile_size];
        let samples = J2kLosslessSamples::new(
            &pixels,
            u32::try_from(tile_size).expect("fixture tile size fits u32"),
            u32::try_from(tile_size).expect("fixture tile size fits u32"),
            1,
            8,
            false,
        )
        .unwrap();

        assert_eq!(
            j2k_lossless_decomposition_levels_for_progression(samples, J2kProgressionOrder::Rpcl),
            expected_levels
        );
    }
}

#[test]
fn max_decomposition_level_option_caps_rpcl_without_forcing_small_tiles() {
    let large_pixels = vec![0; 256 * 256];
    let large =
        J2kLosslessSamples::new(&large_pixels, 256, 256, 1, 8, false).expect("valid large tile");
    assert_eq!(
        j2k_lossless_decomposition_levels_for_options(
            large,
            J2kLosslessEncodeOptions::default()
                .with_progression(J2kProgressionOrder::Rpcl)
                .with_max_decomposition_levels(Some(1))
        ),
        1
    );

    let small_pixels = vec![0; 8 * 8];
    let small =
        J2kLosslessSamples::new(&small_pixels, 8, 8, 1, 8, false).expect("valid small tile");
    assert_eq!(
        j2k_lossless_decomposition_levels_for_options(
            small,
            J2kLosslessEncodeOptions::default()
                .with_progression(J2kProgressionOrder::Rpcl)
                .with_max_decomposition_levels(Some(1))
        ),
        0
    );
}

#[test]
fn cpu_lossless_round_trips_gray8() {
    let pixels: Vec<u8> = (0_u8..35).map(|value| value * 7).collect();
    let samples = J2kLosslessSamples::new(&pixels, 7, 5, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.width, 7);
    assert_eq!(encoded.height, 5);
    assert_eq!(encoded.components, 1);
    assert_eq!(encoded.bit_depth, 8);
    assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 7);
    assert_eq!(decoded.height, 5);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_component_count_above_u8() {
    let pixels = (0_u8..=255).collect::<Vec<_>>();
    let samples =
        J2kLosslessSamples::new(&pixels, 1, 1, 256, 8, false).expect("256-component sample");

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("lossless encode");

    assert_eq!(encoded.components, 256);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 1);
    assert_eq!(decoded.height, 1);
    assert_eq!(decoded.num_components, 256);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_two_component_no_mct_with_strict_decode() {
    let mut pixels = Vec::with_capacity(11 * 7 * 2);
    for y in 0..7u8 {
        for x in 0..11u8 {
            pixels.push(x.wrapping_mul(17).wrapping_add(y.wrapping_mul(3)));
            pixels.push(255u8.wrapping_sub(x.wrapping_mul(5).wrapping_add(y.wrapping_mul(11))));
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, 11, 7, 2, 8, false).expect("2-component samples");

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options().with_reversible_transform(ReversibleTransform::None53),
    )
    .expect("2-component lossless encode");

    let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
    assert_eq!(
        encoded.codestream[cod_offset + 8],
        0,
        "2-component output must not use MCT"
    );

    let image = Image::new(
        &encoded.codestream,
        &DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        },
    )
    .expect("strict parse of 2-component codestream");
    let decoded = image
        .decode_native()
        .expect("strict decode of 2-component codestream");

    assert_eq!(decoded.width, 11);
    assert_eq!(decoded.height, 7);
    assert_eq!(decoded.num_components, 2);
    assert_eq!(decoded.bit_depth, 8);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_classic_lossless_cod_marker_length_reaches_next_marker() {
    let pixels = vec![127u8; 64 * 64 * 3];
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let cod_offset = marker_offset(&encoded.codestream, 0x52).expect("COD marker");
    let qcd_offset = marker_offset(&encoded.codestream, 0x5C).expect("QCD marker");
    let lcod = u16::from_be_bytes([
        encoded.codestream[cod_offset + 2],
        encoded.codestream[cod_offset + 3],
    ]) as usize;

    assert_eq!(cod_offset + 2 + lcod, qcd_offset);
}

#[test]
fn auto_lossless_round_trips_rgb16_odd_dimensions() {
    let mut pixels = Vec::new();
    for y in 0..3u16 {
        for x in 0..5u16 {
            for c in 0..3u16 {
                pixels.extend_from_slice(&(x * 101 + y * 307 + c * 997).to_le_bytes());
            }
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 5, 3, 3, 16, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default())
        .expect("auto lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.components, 3);
    assert_eq!(encoded.bit_depth, 16);

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 5);
    assert_eq!(decoded.height, 3);
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 16);
    assert_eq!(decoded.data, pixels);
}

fn unsigned_24_bytes(sample: u32) -> [u8; 3] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
    ]
}

fn signed_24_bytes(sample: i32) -> [u8; 3] {
    unsigned_24_bytes(u32::from_le_bytes(sample.to_le_bytes()) & 0x00ff_ffff)
}

fn unsigned_38_bytes(sample: u64) -> [u8; 5] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0xff) as u8,
        ((sample >> 32) & 0x3f) as u8,
    ]
}

fn unsigned_31_bytes(sample: u32) -> [u8; 4] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0x7f) as u8,
    ]
}

fn unsigned_32_bytes(sample: u32) -> [u8; 4] {
    sample.to_le_bytes()
}

fn unsigned_35_bytes(sample: u64) -> [u8; 5] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0xff) as u8,
        ((sample >> 32) & 0x07) as u8,
    ]
}

fn unsigned_37_bytes(sample: u64) -> [u8; 5] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0xff) as u8,
        ((sample >> 32) & 0x1f) as u8,
    ]
}

fn signed_29_bytes(sample: i32) -> [u8; 4] {
    sample.to_le_bytes()
}

#[test]
fn cpu_lossless_round_trips_gray24() {
    let values = [0_u32, 1, 255, 256, 65_535, 65_536, 0x12_34_56, 0xff_ff_ff];
    let pixels = values
        .iter()
        .flat_map(|sample| unsigned_24_bytes(*sample))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 4, 2, 1, 24, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu gray24 encode");

    assert_eq!(encoded.bit_depth, 24);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 4);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.bit_depth, 24);
    assert_eq!(decoded.bytes_per_sample, 3);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_gray29_round_trips_native_bytes() {
    let values = [
        0_u32,
        1,
        255,
        65_535,
        16_777_216,
        (1_u32 << 28) + 17,
        (1_u32 << 28) - 1,
        1_u32 << 28,
        (1_u32 << 29) - 1,
    ];
    let pixels = values
        .iter()
        .flat_map(|sample| unsigned_31_bytes(*sample))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 1, 29, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(2)),
    )
    .expect("cpu gray29 classic encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.width, 3);
    assert_eq!(encoded.height, 3);
    assert_eq!(encoded.components, 1);
    assert_eq!(encoded.bit_depth, 29);
    assert!(!encoded.signed);

    let header = inspect_j2k_codestream_header(&encoded.codestream).expect("inspect gray29");
    assert_eq!(header.dimensions, (3, 3));
    assert_eq!(header.components, 1);
    assert_eq!(header.bit_depth, 29);
    assert_eq!(header.component_info[0].bit_depth, 29);
    assert!(!header.component_info[0].signed);

    let image = Image::new(&encoded.codestream, &DecodeSettings::default()).expect("parse gray29");
    assert_eq!(image.width(), 3);
    assert_eq!(image.height(), 3);
    assert_eq!(image.original_bit_depth(), 29);
    let decoded = image.decode_native().expect("decode gray29");
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_gray32_dwt_round_trips_native_bytes() {
    let pixels = (0..64_u32 * 64)
        .flat_map(|idx| {
            let sample = idx
                .wrapping_mul(2_654_435_761)
                .wrapping_add((idx / 64).wrapping_mul(97_531));
            unsigned_32_bytes(sample)
        })
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 32, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect("cpu gray32 classic encode");

    assert_eq!(encoded.bit_depth, 32);
    let header = inspect_j2k_codestream_header(&encoded.codestream).expect("inspect gray32");
    assert_eq!(header.bit_depth, 32);
    assert_eq!(header.component_info[0].bit_depth, 32);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 32);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_gray35_dwt_round_trips_native_bytes() {
    let mask = (1_u64 << 35) - 1;
    let pixels = (0..64_u64 * 64)
        .flat_map(|idx| {
            let sample = idx
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add((idx / 64).wrapping_mul(1_048_583))
                & mask;
            unsigned_35_bytes(sample)
        })
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 35, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect("cpu gray35 classic encode");

    assert_eq!(encoded.bit_depth, 35);
    let header = inspect_j2k_codestream_header(&encoded.codestream).expect("inspect gray35");
    assert_eq!(header.bit_depth, 35);
    assert_eq!(header.component_info[0].bit_depth, 35);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 35);
    assert_eq!(decoded.bytes_per_sample, 5);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_gray37_without_dwt_round_trips_native_bytes() {
    let values = [
        0_u64,
        1,
        255,
        65_535,
        16_777_216,
        1_u64 << 32,
        (1_u64 << 36) + 17,
        (1_u64 << 37) - 1,
        (1_u64 << 36) - 3,
    ];
    let pixels = values
        .iter()
        .flat_map(|sample| unsigned_37_bytes(*sample))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 1, 37, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(0)),
    )
    .expect("cpu gray37 classic encode");

    assert_eq!(encoded.bit_depth, 37);
    let header = inspect_j2k_codestream_header(&encoded.codestream).expect("inspect gray37");
    assert_eq!(header.bit_depth, 37);
    assert_eq!(header.component_info[0].bit_depth, 37);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 37);
    assert_eq!(decoded.bytes_per_sample, 5);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_gray29_multi_tile_round_trips_native_bytes() {
    let mut pixels = Vec::new();
    for y in 0..6_u32 {
        for x in 0..6_u32 {
            let sample = (x * 17_000_003 + y * 9_000_001 + (x ^ y) * 123_457) & ((1_u32 << 29) - 1);
            pixels.extend_from_slice(&unsigned_31_bytes(sample));
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 6, 6, 1, 29, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(1))
            .with_tile_size(Some((3, 3))),
    )
    .expect("cpu gray29 classic multi-tile encode");

    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xff, 0x90])
        .count();
    assert_eq!(sot_count, 4);

    let image =
        Image::new(&encoded.codestream, &DecodeSettings::default()).expect("parse gray29 tiles");
    let decoded = image.decode_native().expect("decode gray29 tiles");
    assert_eq!(decoded.width, 6);
    assert_eq!(decoded.height, 6);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_rejects_gray38_explicitly() {
    let pixels = [
        0_u64,
        1,
        255,
        65_535,
        16_777_216,
        (1_u64 << 32) + 17,
        (1_u64 << 37) - 1,
        1_u64 << 37,
        (1_u64 << 38) - 1,
    ]
    .iter()
    .flat_map(|sample| unsigned_38_bytes(*sample))
    .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 1, 38, false).unwrap();

    let err = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(2)),
    )
    .expect_err("classic gray38 encode should be explicitly unsupported");

    assert!(
        err.to_string()
            .contains("no-quantization guard/exponent signaling limit"),
        "unexpected error: {err}"
    );
}

#[test]
fn cpu_lossless_classic_signed_gray29_round_trips_native_bytes() {
    let values = [
        -(1_i32 << 28),
        -65_537,
        -1,
        0,
        1,
        65_537,
        (1_i32 << 28) - 1,
        123_456,
        -123_456,
    ];
    let pixels = values
        .iter()
        .flat_map(|sample| signed_29_bytes(*sample))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 1, 29, true).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(2)),
    )
    .expect("cpu signed gray29 classic encode");

    assert_eq!(encoded.bit_depth, 29);
    assert!(encoded.signed);
    let image =
        Image::new(&encoded.codestream, &DecodeSettings::default()).expect("parse signed gray29");
    let decoded = image.decode_native().expect("decode signed gray29");
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert!(decoded.signed);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_classic_rgb29_rct_round_trips_native_bytes() {
    let mut pixels = Vec::new();
    for y in 0..3_u32 {
        for x in 0..3_u32 {
            for c in 0..3_u32 {
                let sample =
                    (x * 17_000_003 + y * 9_000_001 + c * 33_333_331) & ((1_u32 << 29) - 1);
                pixels.extend_from_slice(&unsigned_31_bytes(sample));
            }
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 3, 29, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_reversible_transform(ReversibleTransform::Rct53)
            .with_max_decomposition_levels(Some(2)),
    )
    .expect("cpu rgb29 classic encode");

    assert_eq!(encoded.components, 3);
    assert_eq!(encoded.bit_depth, 29);
    let image = Image::new(&encoded.codestream, &DecodeSettings::default()).expect("parse rgb29");
    let decoded = image.decode_native().expect("decode rgb29");
    assert_eq!(decoded.num_components, 3);
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_htj2k_gray31_without_dwt_round_trips_native_bytes() {
    let pixels = [
        0_u32,
        1,
        255,
        65_535,
        16_777_216,
        (1_u32 << 30) + 17,
        (1_u32 << 30) - 1,
        1_u32 << 30,
        (1_u32 << 31) - 1,
    ]
    .iter()
    .flat_map(|sample| unsigned_31_bytes(*sample))
    .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 1, 31, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0)),
    )
    .expect("cpu HTJ2K gray31 encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.bit_depth, 31);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 31);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_htj2k_high_bit_dwt_rejects_explicitly() {
    let pixels = (0_u32..(64 * 64))
        .map(|index| match index % 8 {
            0 => 0,
            1 => 1,
            2 => 65_535,
            3 => 16_777_216,
            4 => (1_u32 << 28) + 17,
            5 => (1_u32 << 28) - 1,
            6 => 1_u32 << 28,
            _ => (1_u32 << 29) - 1,
        })
        .flat_map(unsigned_31_bytes)
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 29, false).unwrap();

    let err = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect_err("HTJ2K high-bit DWT encode should be explicitly unsupported");

    assert!(
        err.to_string()
            .contains("high-bit lossless encode with DWT"),
        "unexpected error: {err}"
    );
}

#[test]
fn cpu_lossless_htj2k_gray29_without_dwt_round_trips_native_bytes() {
    let pixels = [
        0_u32,
        1,
        255,
        65_535,
        16_777_216,
        (1_u32 << 28) + 17,
        (1_u32 << 28) - 1,
        1_u32 << 28,
        (1_u32 << 29) - 1,
    ]
    .iter()
    .flat_map(|sample| unsigned_31_bytes(*sample))
    .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 1, 29, false).unwrap();

    let encoded = encode_j2k_lossless(
        samples,
        &cpu_options()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0)),
    )
    .expect("cpu HTJ2K gray29 encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.bit_depth, 29);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_signed_gray24() {
    let values = [-8_388_608_i32, -65_536, -257, -1, 0, 257, 65_536, 8_388_607];
    let pixels = values
        .iter()
        .flat_map(|sample| signed_24_bytes(*sample))
        .collect::<Vec<_>>();
    let samples = J2kLosslessSamples::new(&pixels, 4, 2, 1, 24, true).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu signed gray24 encode");

    assert_eq!(encoded.bit_depth, 24);
    assert!(encoded.signed);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 4);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.num_components, 1);
    assert_eq!(decoded.bit_depth, 24);
    assert!(decoded.signed);
    assert_eq!(decoded.bytes_per_sample, 3);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_high_variance_512() {
    let mut pixels = Vec::with_capacity(512 * 512 * 3);
    let mut state = 0x5eed_1234_u32;
    for _ in 0..512 * 512 * 3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push((state >> 24) as u8);
    }
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_constant_gray_512() {
    let pixels = vec![243u8; 512 * 512 * 3];
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_low_variance_slide_like_512() {
    let mut pixels = Vec::with_capacity(512 * 512 * 3);
    for y in 0..512u32 {
        for x in 0..512u32 {
            let base = 240u8.wrapping_add(((x / 19 + y / 23 + x * y / 4096) & 7) as u8);
            pixels.push(base);
            pixels.push(base.saturating_sub(2));
            pixels.push(base.saturating_add(2));
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_rgb8_variable_chroma_512() {
    let mut pixels = Vec::with_capacity(512 * 512 * 3);
    for y in 0..512i32 {
        for x in 0..512i32 {
            let base = 238 + ((x / 17 + y / 29 + x * y / 8192) & 15);
            let red_delta = ((x * 3 + y * 5) & 31) - 15;
            let blue_delta = ((x * 7 - y * 3) & 31) - 15;
            pixels.push(clamped_u8(base + red_delta));
            pixels.push(clamped_u8(base));
            pixels.push(clamped_u8(base + blue_delta));
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
#[ignore = "requires J2K_APERIO_TILE_FIXTURE"]
fn cpu_lossless_round_trips_aperio_jp2k_problem_tile_512() {
    let Some(path) = std::env::var_os("J2K_APERIO_TILE_FIXTURE").map(PathBuf::from) else {
        return;
    };
    let pixels = std::fs::read(&path).expect("problem tile fixture");
    assert_eq!(pixels.len(), 512 * 512 * 3);
    let samples = J2kLosslessSamples::new(&pixels, 512, 512, 3, 8, false).unwrap();

    let codestream = j2k_native::encode(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("cpu lossless encode");
    let decoded = decode_native(&codestream);
    let mismatch = decoded
        .data
        .iter()
        .zip(pixels.iter())
        .position(|(actual, expected)| actual != expected);
    assert!(
        mismatch.is_none(),
        "first mismatch at byte {:?}: expected {:?}, actual {:?}",
        mismatch,
        mismatch.map(|idx| pixels[idx]),
        mismatch.map(|idx| decoded.data[idx])
    );
}

#[test]
fn cpu_lossless_round_trips_rgb8_seed_130_64() {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    let mut state = 0x0082_u32 ^ 0x9e37_79b9;
    for _ in 0..64 * 64 * 3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push((state >> 24) as u8);
    }
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn cpu_lossless_round_trips_gray8_seed_104_64() {
    let mut pixels = Vec::with_capacity(64 * 64);
    let mut state = 0x0068_u32 ^ 0x517c_c1b7;
    for _ in 0..64 * 64 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        pixels.push((state >> 24) as u8);
    }
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossless(samples, &cpu_options()).expect("cpu lossless encode");

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn auto_falls_back_to_validated_cpu_until_device_encode_is_complete() {
    let pixels: Vec<u8> = (0_u8..27).map(|value| value * 3).collect();
    let samples = J2kLosslessSamples::new(&pixels, 3, 3, 3, 8, false).unwrap();

    let encoded =
        encode_j2k_lossless(samples, &auto_options()).expect("prefer-device lossless encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.data, pixels);
}

#[test]
fn require_device_errors_clearly_when_encode_backend_is_unavailable() {
    let pixels = vec![0u8; 4 * 4];
    let samples = J2kLosslessSamples::new(&pixels, 4, 4, 1, 8, false).unwrap();

    let err = encode_j2k_lossless(samples, &require_device_options()).unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("device"));
    assert!(err.to_string().contains("encode"));
}

#[test]
fn accelerator_facade_auto_falls_back_when_no_stage_dispatches() {
    #[derive(Default)]
    struct NoDispatchAccelerator;

    impl J2kEncodeStageAccelerator for NoDispatchAccelerator {}

    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(5)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = NoDispatchAccelerator;

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &auto_options(),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("prefer-device encode should fall back to CPU without dispatch");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn accelerator_facade_preserves_native_stage_error() {
    struct FailingDeinterleaveAccelerator;

    impl J2kEncodeStageAccelerator for FailingDeinterleaveAccelerator {
        fn encode_deinterleave(
            &mut self,
            _job: J2kDeinterleaveToF32Job<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
            Err(J2kEncodeStageError::internal_invariant(
                "facade accelerator fixture",
            ))
        }
    }

    let pixels = vec![0_u8; 8 * 8];
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let error = encode_j2k_lossless_with_accelerator(
        samples,
        &auto_options(),
        BackendKind::Metal,
        &mut FailingDeinterleaveAccelerator,
    )
    .expect_err("accepted accelerator failure must surface");

    let j2k::J2kError::NativeEncode { context, source } = error else {
        panic!("expected native encode failure");
    };
    assert_eq!(
        context,
        "accelerated native JPEG 2000 lossless encode failed"
    );
    let Some(j2k_native::EncodeError::Accelerator { operation, source }) =
        std::error::Error::source(&source)
            .and_then(|source| source.downcast_ref::<j2k_native::EncodeError>())
    else {
        panic!("expected accelerator failure");
    };
    assert_eq!(*operation, "pixel deinterleave");
    assert_eq!(
        *source,
        J2kEncodeStageError::internal_invariant("facade accelerator fixture")
    );
}

#[test]
fn accelerator_facade_require_device_errors_when_no_stage_dispatches() {
    #[derive(Default)]
    struct NoDispatchAccelerator;

    impl J2kEncodeStageAccelerator for NoDispatchAccelerator {}

    let pixels = vec![0u8; 8 * 8];
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = NoDispatchAccelerator;

    let err = encode_j2k_lossless_with_accelerator(
        samples,
        &require_device_options(),
        BackendKind::Metal,
        &mut accelerator,
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("did not dispatch"));
}

#[test]
fn accelerator_facade_reports_partial_auto_dispatch_and_strictly_rejects_it() {
    #[derive(Default)]
    struct PacketizationDispatchAccelerator {
        deinterleave: usize,
        quantize_subband: usize,
        packetization: usize,
    }

    impl J2kEncodeStageAccelerator for PacketizationDispatchAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                deinterleave: self.deinterleave,
                quantize_subband: self.quantize_subband,
                packetization: self.packetization,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_deinterleave(
            &mut self,
            job: J2kDeinterleaveToF32Job<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
            self.deinterleave = self.deinterleave.saturating_add(1);
            Ok(Some(deinterleave_to_f32_for_test(job)))
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "mock accelerator fixture coefficients are rounded within the i32 domain"
        )]
        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<i32>>> {
            self.quantize_subband = self.quantize_subband.saturating_add(1);
            Ok(Some(
                job.coefficients
                    .iter()
                    .map(|sample| sample.round() as i32)
                    .collect(),
            ))
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
            self.packetization = self.packetization.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(7)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut auto_accelerator = PacketizationDispatchAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &auto_options(),
        BackendKind::Metal,
        &mut auto_accelerator,
    )
    .expect("Auto should preserve partial dispatch evidence while falling back");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.dispatch_report, auto_accelerator.dispatch_report());
    assert!(encoded.dispatch_report.deinterleave > 0);
    assert!(encoded.dispatch_report.quantize_subband > 0);
    assert!(encoded.dispatch_report.packetization > 0);
    assert_eq!(encoded.dispatch_report.tier1_code_block, 0);

    let mut strict_accelerator = PacketizationDispatchAccelerator::default();

    let err = encode_j2k_lossless_with_accelerator(
        samples,
        &require_device_options(),
        BackendKind::Metal,
        &mut strict_accelerator,
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("tier1_code_block"));
}

#[test]
fn accelerator_facade_reports_requested_backend_after_all_required_stages_dispatch() {
    #[derive(Default)]
    struct FullClassicAccelerator {
        deinterleave: usize,
        quantize_subband: usize,
        tier1_code_block: usize,
        packetization: usize,
    }

    impl J2kEncodeStageAccelerator for FullClassicAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                deinterleave: self.deinterleave,
                quantize_subband: self.quantize_subband,
                tier1_code_block: self.tier1_code_block,
                packetization: self.packetization,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_deinterleave(
            &mut self,
            job: J2kDeinterleaveToF32Job<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
            self.deinterleave = self.deinterleave.saturating_add(1);
            Ok(Some(deinterleave_to_f32_for_test(job)))
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "mock accelerator fixture coefficients are rounded within the i32 domain"
        )]
        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<i32>>> {
            self.quantize_subband = self.quantize_subband.saturating_add(1);
            Ok(Some(
                job.coefficients
                    .iter()
                    .map(|sample| sample.round() as i32)
                    .collect(),
            ))
        }

        fn encode_tier1_code_block(
            &mut self,
            job: J2kTier1CodeBlockEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<EncodedJ2kCodeBlock>> {
            self.tier1_code_block = self.tier1_code_block.saturating_add(1);
            j2k_native::encode_j2k_code_block_scalar_with_style(
                job.coefficients,
                job.width,
                job.height,
                native_subband(job.sub_band_type),
                job.total_bitplanes,
                native_code_block_style(job.style),
            )
            .map(public_encoded_j2k)
            .map(Some)
            .map_err(|source| {
                J2kEncodeStageError::backend(
                    "native scalar",
                    "classic Tier-1 code-block encode",
                    source,
                )
            })
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
            self.packetization = self.packetization.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(7)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = FullClassicAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &auto_options(),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("all required device stages should produce encoded codestream");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(encoded.dispatch_report, accelerator.dispatch_report());
    assert!(encoded.dispatch_report.deinterleave > 0);
    assert!(encoded.dispatch_report.quantize_subband > 0);
    assert!(encoded.dispatch_report.tier1_code_block > 0);
    assert!(encoded.dispatch_report.packetization > 0);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn accelerator_facade_ht_require_device_checks_ht_code_block_stage() {
    #[derive(Default)]
    struct FullHtAccelerator {
        deinterleave: usize,
        quantize_subband: usize,
        ht_code_block: usize,
        packetization: usize,
    }

    impl J2kEncodeStageAccelerator for FullHtAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                deinterleave: self.deinterleave,
                quantize_subband: self.quantize_subband,
                ht_code_block: self.ht_code_block,
                packetization: self.packetization,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_deinterleave(
            &mut self,
            job: J2kDeinterleaveToF32Job<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
            self.deinterleave = self.deinterleave.saturating_add(1);
            Ok(Some(deinterleave_to_f32_for_test(job)))
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "mock accelerator fixture coefficients are rounded within the i32 domain"
        )]
        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<i32>>> {
            self.quantize_subband = self.quantize_subband.saturating_add(1);
            Ok(Some(
                job.coefficients
                    .iter()
                    .map(|sample| sample.round() as i32)
                    .collect(),
            ))
        }

        fn encode_ht_code_block(
            &mut self,
            job: J2kHtCodeBlockEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
            self.ht_code_block = self.ht_code_block.saturating_add(1);
            j2k_native::encode_ht_code_block_scalar(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
            .map(public_encoded_ht)
            .map(Some)
            .map_err(|source| {
                J2kEncodeStageError::backend("native scalar", "HT Tier-1 code-block encode", source)
            })
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
            self.packetization = self.packetization.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(13)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = FullHtAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &require_device_options().with_block_coding_mode(J2kBlockCodingMode::HighThroughput),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("HT required stages should dispatch");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

#[test]
fn accelerator_facade_ht_lossless_quality_layers_request_refinement_passes() {
    #[derive(Default)]
    struct RefinementHtAccelerator {
        max_target_coding_passes: u8,
        ht_code_block: usize,
    }

    impl J2kEncodeStageAccelerator for RefinementHtAccelerator {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            J2kEncodeDispatchReport {
                ht_code_block: self.ht_code_block,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_ht_code_block(
            &mut self,
            job: J2kHtCodeBlockEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<EncodedHtJ2kCodeBlock>> {
            self.ht_code_block = self.ht_code_block.saturating_add(1);
            self.max_target_coding_passes =
                self.max_target_coding_passes.max(job.target_coding_passes);
            j2k_native::encode_ht_code_block_scalar_with_passes(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
                job.target_coding_passes,
            )
            .map(public_encoded_ht)
            .map(Some)
            .map_err(|source| {
                J2kEncodeStageError::backend("native scalar", "HT Tier-1 refinement encode", source)
            })
        }
    }

    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(13)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let mut accelerator = RefinementHtAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &auto_options()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_quality_layers(3),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("HT lossless layered encode should dispatch");

    assert_eq!(accelerator.max_target_coding_passes, 3);
    assert!(accelerator.ht_code_block > 0);
    assert_eq!(decode_native(&encoded.codestream).data, pixels);
}

fn marker_offset(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
}

fn sot_tile_part_fields(codestream: &[u8]) -> Vec<(u16, u8, u8)> {
    codestream
        .windows(2)
        .enumerate()
        .filter_map(|(offset, marker)| {
            if marker != [0xFF, 0x90] || offset + 12 > codestream.len() {
                return None;
            }
            let tile_index = u16::from_be_bytes([codestream[offset + 4], codestream[offset + 5]]);
            let tile_part_index = codestream[offset + 10];
            let num_tile_parts = codestream[offset + 11];
            Some((tile_index, tile_part_index, num_tile_parts))
        })
        .collect()
}

fn sot_tile_part_lengths(codestream: &[u8]) -> Vec<(u16, u32)> {
    let mut offset = 0usize;
    let mut fields = Vec::new();
    while offset + 12 <= codestream.len() {
        if codestream[offset] == 0xff && codestream[offset + 1] == 0x90 {
            let tile_index = u16::from_be_bytes([codestream[offset + 4], codestream[offset + 5]]);
            let tile_part_length = u32::from_be_bytes([
                codestream[offset + 6],
                codestream[offset + 7],
                codestream[offset + 8],
                codestream[offset + 9],
            ]);
            fields.push((tile_index, tile_part_length));
            offset += 12;
        } else {
            offset += 1;
        }
    }
    fields
}

fn tlm_tile_part_lengths(codestream: &[u8]) -> Vec<(u16, u32)> {
    let mut offset = 0usize;
    let mut fields = Vec::new();
    while offset + 12 <= codestream.len() {
        if codestream[offset] == 0xff && codestream[offset + 1] == 0x55 {
            let marker_len =
                u16::from_be_bytes([codestream[offset + 2], codestream[offset + 3]]) as usize;
            assert_eq!(marker_len, 10);
            assert_eq!(codestream[offset + 5], 0x22);
            let tile_index = u16::from_be_bytes([codestream[offset + 6], codestream[offset + 7]]);
            let tile_part_length = u32::from_be_bytes([
                codestream[offset + 8],
                codestream[offset + 9],
                codestream[offset + 10],
                codestream[offset + 11],
            ]);
            fields.push((tile_index, tile_part_length));
            offset += 2 + marker_len;
        } else {
            offset += 1;
        }
    }
    fields
}

#[test]
fn sample_descriptor_rejects_short_pixel_buffers() {
    let pixels = vec![0u8; 5];

    let err = J2kLosslessSamples::new(&pixels, 2, 2, 3, 8, false).unwrap_err();

    assert!(err.to_string().contains("pixel data too short"));
}
