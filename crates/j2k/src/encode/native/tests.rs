// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{string::ToString, vec};

use j2k_native::{
    DecodeSettings, EncodeError, EncodeProgressionOrder, Image, ResidentHtj2kEncodeError,
};

use super::{
    encode_cpu, encode_cpu_components, encode_cpu_typed_components, encode_cpu_with_roi_regions,
    interleave_component_planes, map_native_resident_encode_error,
    native_lossless_component_options, native_lossless_options,
    native_lossless_typed_component_options, native_lossy_options, native_progression_order,
    native_roi_regions_for_samples, validate_lossless_high_bit_options,
    validate_lossy_high_bit_options, EncodeBackendPreference, J2kBlockCodingMode,
    J2kLosslessComponentSamples, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kLosslessTypedComponentSamples, J2kLossyEncodeOptions, J2kLossySamples, J2kMarkerSegment,
    J2kProgressionOrder, J2kRoiRegion, ReversibleTransform,
};
use crate::{J2kError, J2kLosslessComponentPlane, J2kLosslessTypedComponentPlane};

fn lossless_samples(data: &[u8], bit_depth: u8, width: u32, height: u32) -> J2kLosslessSamples<'_> {
    J2kLosslessSamples::new(data, width, height, 1, bit_depth, false).unwrap()
}

fn lossy_samples(data: &[u8], bit_depth: u8) -> J2kLossySamples<'_> {
    J2kLossySamples::new(data, 1, 1, 1, bit_depth, false).unwrap()
}

fn invalid_samples_what(error: J2kError) -> String {
    match error {
        J2kError::InvalidSamples { what } => what,
        other => panic!("expected invalid samples, got {other:?}"),
    }
}

#[test]
fn high_bit_guards_preserve_supported_cpu_shapes_and_specific_rejections() {
    let sample_24 = [0_u8; 3];
    let sample_25 = [0_u8; 4];
    let sample_32 = [0_u8; 4];
    let dwt_25 = vec![0_u8; 64 * 64 * 4];
    let dwt_38 = vec![0_u8; 64 * 64 * 5];
    let permissive = J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        ..J2kLosslessEncodeOptions::default()
    };
    assert!(validate_lossless_high_bit_options(
        lossless_samples(&sample_24, 24, 1, 1),
        &permissive
    )
    .is_ok());

    let classic_cpu = J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        block_coding_mode: J2kBlockCodingMode::Classic,
        max_decomposition_levels: Some(0),
        ..J2kLosslessEncodeOptions::default()
    };
    assert!(validate_lossless_high_bit_options(
        lossless_samples(&sample_25, 25, 1, 1),
        &classic_cpu
    )
    .is_ok());

    let require_device = J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::RequireDevice,
        ..classic_cpu
    };
    assert!(validate_lossless_high_bit_options(
        lossless_samples(&sample_25, 25, 1, 1),
        &require_device
    )
    .unwrap_err()
    .to_string()
    .contains("CPU reversible path only"));

    let ht_no_dwt = J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        ..classic_cpu
    };
    assert!(
        validate_lossless_high_bit_options(lossless_samples(&sample_32, 32, 1, 1), &ht_no_dwt)
            .unwrap_err()
            .to_string()
            .contains("HT block bitplane limit")
    );

    let ht_with_dwt = J2kLosslessEncodeOptions {
        max_decomposition_levels: Some(1),
        ..ht_no_dwt
    };
    assert!(validate_lossless_high_bit_options(
        lossless_samples(&dwt_25, 25, 64, 64),
        &ht_with_dwt
    )
    .unwrap_err()
    .to_string()
    .contains("high-bit lossless encode with DWT"));

    let classic_with_dwt = J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::Classic,
        ..ht_with_dwt
    };
    assert!(validate_lossless_high_bit_options(
        lossless_samples(&dwt_38, 38, 64, 64),
        &classic_with_dwt
    )
    .unwrap_err()
    .to_string()
    .contains("no-quantization guard"));
}

#[test]
fn lossy_high_bit_guards_keep_cpu_classic_as_the_only_high_bit_route() {
    let sample_24 = [0_u8; 3];
    let sample_25 = [0_u8; 4];
    let sample_38 = [0_u8; 5];
    let ht_device = J2kLossyEncodeOptions {
        backend: EncodeBackendPreference::RequireDevice,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        ..J2kLossyEncodeOptions::default()
    };
    assert!(validate_lossy_high_bit_options(lossy_samples(&sample_24, 24), &ht_device).is_ok());
    assert!(
        validate_lossy_high_bit_options(lossy_samples(&sample_25, 25), &ht_device)
            .unwrap_err()
            .to_string()
            .contains("HTJ2K high-bit lossy")
    );

    let device_classic = J2kLossyEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::Classic,
        ..ht_device
    };
    assert!(
        validate_lossy_high_bit_options(lossy_samples(&sample_25, 25), &device_classic)
            .unwrap_err()
            .to_string()
            .contains("CPU irreversible path only")
    );

    let cpu_classic = J2kLossyEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        ..device_classic
    };
    assert!(validate_lossy_high_bit_options(lossy_samples(&sample_38, 38), &cpu_classic).is_ok());
}

#[test]
fn roi_conversion_is_transactional_and_preserves_valid_descriptors() {
    let valid = J2kRoiRegion {
        component: 1,
        x: 2,
        y: 3,
        width: 4,
        height: 5,
        shift: 6,
    };
    let converted = native_roi_regions_for_samples(8, 9, 2, &[valid]).unwrap();
    assert_eq!(converted.len(), 1);
    assert_eq!(converted[0].component, 1);
    assert_eq!((converted[0].x, converted[0].y), (2, 3));
    assert_eq!((converted[0].width, converted[0].height), (4, 5));
    assert_eq!(converted[0].shift, 6);

    let cases = [
        (
            J2kRoiRegion {
                component: 2,
                ..valid
            },
            "ROI region component index out of range",
        ),
        (
            J2kRoiRegion { width: 0, ..valid },
            "ROI region dimensions must be non-zero",
        ),
        (
            J2kRoiRegion { shift: 0, ..valid },
            "ROI region maxshift must be non-zero",
        ),
        (
            J2kRoiRegion {
                x: u32::MAX,
                ..valid
            },
            "ROI region bounds overflow",
        ),
        (
            J2kRoiRegion {
                x: 7,
                width: 2,
                ..valid
            },
            "ROI region must be inside image bounds",
        ),
    ];
    for (invalid, expected) in cases {
        assert_eq!(
            invalid_samples_what(
                native_roi_regions_for_samples(8, 9, 2, &[valid, invalid]).unwrap_err()
            ),
            expected
        );
    }
}

#[test]
fn component_interleave_preserves_little_endian_sample_and_plane_order() {
    let first = [1_u8, 2, 3, 4];
    let second = [11_u8, 12, 13, 14];
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
    let samples = J2kLosslessComponentSamples::new(&planes, 2, 1, 16, false).unwrap();

    assert_eq!(
        interleave_component_planes(samples).unwrap(),
        [1, 2, 11, 12, 3, 4, 13, 14]
    );
}

#[test]
fn cpu_encode_adapters_emit_decodable_lossless_codestreams() {
    let pixels: Vec<u8> = (0_u8..64).map(|value| value.wrapping_mul(3)).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).unwrap();
    let options = J2kLosslessEncodeOptions {
        backend: EncodeBackendPreference::CpuOnly,
        max_decomposition_levels: Some(0),
        ..J2kLosslessEncodeOptions::default()
    };
    let roi = J2kRoiRegion {
        component: 0,
        x: 2,
        y: 1,
        width: 4,
        height: 5,
        shift: 12,
    };

    let interleaved = encode_cpu(samples, options).unwrap();
    let roi_encoded = encode_cpu_with_roi_regions(samples, options, &[roi]).unwrap();

    let plane = [J2kLosslessComponentPlane {
        data: &pixels,
        x_rsiz: 1,
        y_rsiz: 1,
    }];
    let component_samples = J2kLosslessComponentSamples::new(&plane, 8, 8, 8, false).unwrap();
    let components = encode_cpu_components(component_samples, options).unwrap();

    let typed_plane = [J2kLosslessTypedComponentPlane {
        data: &pixels,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 8,
        signed: false,
    }];
    let typed_samples = J2kLosslessTypedComponentSamples::new(&typed_plane, 8, 8).unwrap();
    let typed = encode_cpu_typed_components(typed_samples, options).unwrap();

    for codestream in [interleaved, roi_encoded, components, typed] {
        let image = Image::new(&codestream, &DecodeSettings::default()).unwrap();
        assert_eq!((image.width(), image.height()), (8, 8));
        assert_eq!(image.decode().unwrap(), pixels);
    }
}

#[test]
fn native_options_preserve_facade_contract_without_enabling_internal_validation() {
    let samples = J2kLosslessSamples::new(&[0; 12], 2, 2, 3, 8, false).unwrap();
    let options = J2kLosslessEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        progression: J2kProgressionOrder::Rpcl,
        max_decomposition_levels: Some(0),
        tile_size: Some((1, 2)),
        tile_part_packet_limit: Some(7),
        quality_layers: 3,
        write_plt: true,
        write_plm: true,
        write_ppm: true,
        write_ppt: true,
        write_sop: true,
        write_eph: true,
        reversible_transform: ReversibleTransform::Rct53,
        ..J2kLosslessEncodeOptions::default()
    };
    let native = native_lossless_options(samples, options);
    assert!(native.reversible);
    assert!(native.use_ht_block_coding);
    assert_eq!(native.progression_order, EncodeProgressionOrder::Rpcl);
    assert!(
        native.write_tlm,
        "RPCL requires TLM even without an explicit marker request"
    );
    assert!(native.write_plt && native.write_plm && native.write_ppm && native.write_ppt);
    assert!(native.write_sop && native.write_eph && native.use_mct);
    assert_eq!(native.tile_size, Some((1, 2)));
    assert_eq!(native.tile_part_packet_limit, Some(7));
    assert_eq!(native.num_layers, 3);
    assert!(!native.validate_high_throughput_codestream);
}

#[test]
fn planar_options_disable_mct_and_lossy_options_preserve_markers_and_layers() {
    let component_data = [0_u8; 4];
    let planes = [J2kLosslessComponentPlane {
        data: &component_data,
        x_rsiz: 1,
        y_rsiz: 1,
    }; 3];
    let components = J2kLosslessComponentSamples::new(&planes, 2, 2, 8, false).unwrap();
    assert!(
        !native_lossless_component_options(components, J2kLosslessEncodeOptions::default()).use_mct
    );

    let typed_planes = [J2kLosslessTypedComponentPlane {
        data: &component_data,
        x_rsiz: 1,
        y_rsiz: 1,
        bit_depth: 8,
        signed: false,
    }; 3];
    let typed = J2kLosslessTypedComponentSamples::new(&typed_planes, 2, 2).unwrap();
    assert!(
        !native_lossless_typed_component_options(typed, J2kLosslessEncodeOptions::default())
            .use_mct
    );

    let lossy = J2kLossySamples::new(&[0_u8; 12], 2, 2, 3, 8, false).unwrap();
    let lossy_options = J2kLossyEncodeOptions {
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        progression: J2kProgressionOrder::Cprl,
        max_decomposition_levels: Some(0),
        marker_segments: vec![
            J2kMarkerSegment::Tlm,
            J2kMarkerSegment::Plt,
            J2kMarkerSegment::Plm,
            J2kMarkerSegment::Ppm,
            J2kMarkerSegment::Ppt,
            J2kMarkerSegment::Sop,
            J2kMarkerSegment::Eph,
        ],
        precinct_exponents: vec![(4, 5), (6, 7)],
        tile_size: Some((2, 1)),
        tile_part_packet_limit: Some(9),
        ..J2kLossyEncodeOptions::default()
    };
    let native = native_lossy_options(lossy, &lossy_options, 2.5).unwrap();
    assert!(!native.reversible);
    assert!(native.use_ht_block_coding && native.use_mct);
    assert_eq!(native.progression_order, EncodeProgressionOrder::Cprl);
    assert!(native.write_tlm && native.write_plt && native.write_plm);
    assert!(native.write_ppm && native.write_ppt && native.write_sop && native.write_eph);
    assert_eq!(native.precinct_exponents, [(4, 5), (6, 7)]);
    assert_eq!(
        native.irreversible_quantization_scale.to_bits(),
        2.5_f32.to_bits()
    );
}

#[test]
fn progression_and_resident_error_adapters_preserve_semantics() {
    for (facade, native) in [
        (J2kProgressionOrder::Lrcp, EncodeProgressionOrder::Lrcp),
        (J2kProgressionOrder::Rlcp, EncodeProgressionOrder::Rlcp),
        (J2kProgressionOrder::Rpcl, EncodeProgressionOrder::Rpcl),
        (J2kProgressionOrder::Pcrl, EncodeProgressionOrder::Pcrl),
        (J2kProgressionOrder::Cprl, EncodeProgressionOrder::Cprl),
    ] {
        assert_eq!(native_progression_order(facade), native);
    }

    assert!(matches!(
        map_native_resident_encode_error(ResidentHtj2kEncodeError::Unsupported("shape")),
        J2kError::Unsupported(error) if error.what == "shape"
    ));
    assert!(matches!(
        map_native_resident_encode_error(ResidentHtj2kEncodeError::Declined),
        J2kError::Unsupported(error) if error.what.contains("declined")
    ));
    for resident in [
        ResidentHtj2kEncodeError::Resource(EncodeError::AllocationTooLarge {
            what: "resident output",
            requested: 2,
            cap: 1,
        }),
        ResidentHtj2kEncodeError::Backend(EncodeError::InternalInvariant { what: "broken" }),
    ] {
        assert!(matches!(
            map_native_resident_encode_error(resident),
            J2kError::NativeEncode { .. }
        ));
    }
    assert!(matches!(
        map_native_resident_encode_error(ResidentHtj2kEncodeError::InvalidInput("bad geometry")),
        J2kError::NativeEncode { .. }
    ));
}
