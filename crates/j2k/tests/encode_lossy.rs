// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    encode_j2k_lossy, encode_j2k_lossy_with_accelerator, encode_j2k_lossy_with_roi_regions,
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLossyEncodeOptions,
    J2kLossySamples, J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer, J2kRateTarget,
    J2kRoiRegion,
};
use j2k::{
    EncodedHtJ2kCodeBlock, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kEncodeStageError, J2kEncodeStageResult, J2kHtCodeBlockEncodeJob,
    J2kPacketizationEncodeJob, J2kQuantizeSubbandJob,
};
use j2k_core::{BackendKind, CodecError};
use j2k_native::{DecodeSettings, Image};

fn masked_u8(value: usize) -> u8 {
    u8::try_from(value & 0xff).expect("masked fixture byte fits u8")
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
    .expect("encoded codestream should parse strictly")
    .decode_native()
    .expect("encoded codestream should decode strictly")
}

fn public_encoded_ht(block: j2k_native::EncodedHtJ2kCodeBlock) -> EncodedHtJ2kCodeBlock {
    block
}

fn plt_packet_length_count(codestream: &[u8]) -> usize {
    plt_packet_lengths(codestream).len()
}

fn marker_offset(codestream: &[u8], marker: u8) -> Option<usize> {
    codestream
        .windows(2)
        .position(|window| window == [0xFF, marker])
}

fn unsigned_29_bytes(sample: u32) -> [u8; 4] {
    [
        (sample & 0xff) as u8,
        ((sample >> 8) & 0xff) as u8,
        ((sample >> 16) & 0xff) as u8,
        ((sample >> 24) & 0x1f) as u8,
    ]
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

fn plt_packet_lengths(codestream: &[u8]) -> Vec<u32> {
    let plt = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x58])
        .expect("PLT marker");
    let marker_len = u16::from_be_bytes([codestream[plt + 2], codestream[plt + 3]]) as usize;
    let payload = &codestream[plt + 5..plt + 2 + marker_len];
    let mut lengths = Vec::new();
    let mut value = 0_u32;
    let mut in_progress = false;
    for &byte in payload {
        value = (value << 7) + u32::from(byte & 0x7F);
        in_progress = true;
        if byte & 0x80 == 0 {
            lengths.push(value);
            value = 0;
            in_progress = false;
        }
    }
    assert!(!in_progress, "PLT packet length is incomplete");
    lengths
}

#[test]
fn cpu_lossy_rectangular_roi_writes_rgn_and_decodes() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 13) ^ (index / 5)))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();
    let roi = [J2kRoiRegion {
        component: 0,
        x: 8,
        y: 12,
        width: 24,
        height: 20,
        shift: 12,
    }];

    let encoded = encode_j2k_lossy_with_roi_regions(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_max_decomposition_levels(Some(1)),
        &roi,
    )
    .expect("lossy ROI encode");
    let rgn = marker_offset(&encoded.codestream, 0x5E).expect("RGN marker");

    assert_eq!(
        &encoded.codestream[rgn + 2..rgn + 7],
        &[0x00, 0x05, 0x00, 0x00, 0x0C]
    );

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn lossy_sample_descriptor_accepts_part1_high_bit_depths() {
    let pixels = vec![0u8; 2 * 2 * 5];

    let samples =
        J2kLossySamples::new(&pixels, 2, 2, 1, 38, false).expect("38-bit lossy sample descriptor");

    assert_eq!(samples.bit_depth, 38);
}

#[test]
fn cpu_classic_lossy_gray29_decodes_native_bytes() {
    let pixels = (0_u32..32 * 32)
        .map(|idx| ((idx * 524_287) ^ (idx / 3)) & ((1_u32 << 29) - 1))
        .flat_map(unsigned_29_bytes)
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 29, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect("classic gray29 lossy encode");

    assert_eq!(encoded.bit_depth, 29);
    assert!(encoded.report.psnr_db.is_some());
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 29);
    assert_eq!(decoded.bytes_per_sample, 4);
    assert_eq!(decoded.data.len(), pixels.len());
}

#[test]
fn cpu_classic_lossy_gray38_decodes_native_bytes() {
    let pixels = (0_u64..16 * 16)
        .map(|idx| ((idx * 34_359_738_337) ^ (idx / 5)) & ((1_u64 << 38) - 1))
        .flat_map(unsigned_38_bytes)
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 16, 16, 1, 38, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(J2kBlockCodingMode::Classic)
            .with_max_decomposition_levels(Some(1)),
    )
    .expect("classic gray38 lossy encode");

    assert_eq!(encoded.bit_depth, 38);
    assert!(encoded.report.psnr_db.is_some());
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.bit_depth, 38);
    assert_eq!(decoded.bytes_per_sample, 5);
    assert_eq!(decoded.data.len(), pixels.len());
}

#[test]
fn cpu_lossy_htj2k_high_bit_rejects_explicitly() {
    let pixels = (0_u32..16 * 16)
        .map(|idx| (idx * 1_048_573) & ((1_u32 << 29) - 1))
        .flat_map(unsigned_29_bytes)
        .collect::<Vec<_>>();
    let samples = J2kLossySamples::new(&pixels, 16, 16, 1, 29, false).unwrap();

    let err = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput),
    )
    .expect_err("HTJ2K high-bit lossy encode should be explicitly unsupported");

    assert!(err.is_unsupported());
    assert!(
        err.to_string().contains("HTJ2K high-bit lossy encode"),
        "unexpected error: {err}"
    );
}

#[test]
fn lossy_quality_layer_targets_must_be_cumulative() {
    let pixels = vec![0u8; 16 * 16];
    let samples = J2kLossySamples::new(&pixels, 16, 16, 1, 8, false).unwrap();

    let err = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_quality_layers(vec![
                J2kQualityLayer::new(J2kRateTarget::Bytes(2048)),
                J2kQualityLayer::new(J2kRateTarget::Bytes(1024)),
            ]),
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("cumulative"));
}

#[test]
fn cpu_classic_lossy_bits_per_pixel_target_encodes_parseable_codestream() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 13) ^ (index / 5)))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_rate_target(Some(J2kRateTarget::BitsPerPixel(2.0)))
            .with_progression(J2kProgressionOrder::Rlcp),
    )
    .expect("classic lossy encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(encoded.width, 64);
    assert_eq!(encoded.height, 64);
    assert_eq!(encoded.components, 1);
    assert_eq!(encoded.bit_depth, 8);
    assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));
    assert!(encoded.report.actual_bits_per_pixel.is_finite());
    assert!(encoded.report.psnr_db.expect("PSNR report").is_finite());

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_roundtrips_more_than_four_components() {
    let pixels: Vec<u8> = (0..16 * 16 * 5)
        .map(|index| masked_u8(index * 7 + index / 11))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 16, 16, 5, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default().with_backend(EncodeBackendPreference::CpuOnly),
    )
    .expect("classic lossy five-component encode");
    let decoded = strict_decode_native(&encoded.codestream);

    assert_eq!(encoded.components, 5);
    assert_eq!(decoded.width, 16);
    assert_eq!(decoded.height, 16);
    assert_eq!(decoded.num_components, 5);
}

#[test]
fn cpu_classic_lossy_preserves_signed_component_metadata() {
    let pixels: Vec<u8> = (0..16)
        .map(|index| (i8::try_from(index).unwrap() - 8).to_le_bytes()[0])
        .collect();
    let samples = J2kLossySamples::new(&pixels, 4, 4, 1, 8, true).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_max_decomposition_levels(Some(0))
            .with_rate_target(Some(J2kRateTarget::BitsPerPixel(8.0))),
    )
    .expect("classic signed lossy encode");
    let decoded = strict_decode_native(&encoded.codestream);

    assert_eq!(encoded.components, 1);
    assert_eq!(encoded.bit_depth, 8);
    assert!(encoded.signed);
    assert!(decoded.signed);
    assert_eq!(decoded.component_signed, vec![true]);
    assert_eq!(decoded.data.len(), pixels.len());
}

#[test]
fn cpu_classic_lossy_multiple_quality_layers_encode_scalable_codestream() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8(index * 17 + index / 3))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_quality_layers(vec![
                J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(0.75)),
                J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(1.5)),
            ]),
    )
    .expect("classic lossy multilayer encode");

    assert_eq!(encoded.report.quality_layers, 2);
    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);

    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(
        u16::from_be_bytes([encoded.codestream[cod + 6], encoded.codestream[cod + 7]]),
        2
    );
    assert_eq!(encoded.codestream[cod + 12] & 0x04, 0x04);
}

#[test]
fn cpu_classic_lossy_round_trips_two_component_no_mct_with_strict_decode() {
    let mut pixels = Vec::with_capacity(17 * 13 * 2);
    for y in 0..13u8 {
        for x in 0..17u8 {
            pixels.push(x.wrapping_mul(11).wrapping_add(y.wrapping_mul(7)));
            pixels.push(255u8.wrapping_sub(x.wrapping_mul(3).wrapping_add(y.wrapping_mul(19))));
        }
    }
    let samples = J2kLossySamples::new(&pixels, 17, 13, 2, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_max_decomposition_levels(Some(0)),
    )
    .expect("2-component lossy encode");

    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(
        encoded.codestream[cod + 8],
        0,
        "2-component lossy output must not use MCT"
    );

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 17);
    assert_eq!(decoded.height, 13);
    assert_eq!(decoded.num_components, 2);
    assert_eq!(decoded.bit_depth, 8);
}

#[test]
fn cpu_classic_lossy_quality_layer_byte_targets_bound_early_layer_packets() {
    let pixels: Vec<u8> = (0..128 * 128)
        .map(|index| masked_u8((index * 53) ^ (index / 11)))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 128, 128, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_max_decomposition_levels(Some(0))
            .with_marker_segments(vec![J2kMarkerSegment::Plt])
            .with_quality_layers(vec![
                J2kQualityLayer::new(J2kRateTarget::Bytes(256)),
                J2kQualityLayer::new(J2kRateTarget::Bytes(20_000)),
            ]),
    )
    .expect("classic lossy PCRD encode");

    let packet_lengths = plt_packet_lengths(&encoded.codestream);
    assert_eq!(packet_lengths.len(), 2);
    assert!(
        packet_lengths[0] > 1,
        "first layer packet should carry legal pass-truncated data"
    );
    assert!(
        packet_lengths[0] <= 768,
        "first layer packet length {} exceeded target+tolerance",
        packet_lengths[0]
    );
    assert!(packet_lengths[1] > 0);

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_multiple_quality_layers_decode_all_progressions() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 29) ^ (index / 7)))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    for progression in [
        J2kProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl,
    ] {
        let encoded = encode_j2k_lossy(
            samples,
            &J2kLossyEncodeOptions::default()
                .with_backend(EncodeBackendPreference::CpuOnly)
                .with_progression(progression)
                .with_quality_layers(vec![
                    J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(0.75)),
                    J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(1.5)),
                ]),
        )
        .unwrap_or_else(|err| panic!("{progression:?} multilayer encode failed: {err}"));

        let decoded = decode_native(&encoded.codestream);
        assert_eq!(decoded.width, 64, "{progression:?}");
        assert_eq!(decoded.height, 64, "{progression:?}");
        assert_eq!(decoded.num_components, 1, "{progression:?}");
    }
}

#[test]
fn cpu_classic_lossy_emits_plt_and_plm_that_strict_decode_uses() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8(index * 11 + index / 9))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_marker_segments(vec![J2kMarkerSegment::Plt, J2kMarkerSegment::Plm]),
    )
    .expect("classic lossy PLT/PLM encode");

    assert!(
        encoded
            .codestream
            .windows(2)
            .any(|marker| marker == [0xFF, 0x58]),
        "PLT marker must be emitted"
    );
    assert!(
        encoded
            .codestream
            .windows(2)
            .any(|marker| marker == [0xFF, 0x57]),
        "PLM marker must be emitted"
    );

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_emits_sop_and_eph_that_strict_decode_uses() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 19) ^ (index / 11)))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_marker_segments(vec![J2kMarkerSegment::Sop, J2kMarkerSegment::Eph]),
    )
    .expect("classic lossy SOP/EPH encode");

    assert!(
        encoded
            .codestream
            .windows(2)
            .any(|marker| marker == [0xFF, 0x91]),
        "SOP marker must be emitted"
    );
    assert!(
        encoded
            .codestream
            .windows(2)
            .any(|marker| marker == [0xFF, 0x92]),
        "EPH marker must be emitted"
    );
    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(encoded.codestream[cod + 4] & 0x06, 0x06);

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_emits_ppm_and_ppt_that_strict_decode_uses() {
    for (marker, marker_byte) in [(J2kMarkerSegment::Ppm, 0x60), (J2kMarkerSegment::Ppt, 0x61)] {
        let pixels: Vec<u8> = (0..64 * 64)
            .map(|index| masked_u8((index * 23) ^ (index / 13)))
            .collect();
        let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let encoded = encode_j2k_lossy(
            samples,
            &J2kLossyEncodeOptions::default()
                .with_backend(EncodeBackendPreference::CpuOnly)
                .with_max_decomposition_levels(Some(0))
                .with_marker_segments(vec![marker]),
        )
        .expect("classic lossy separated packet-header encode");

        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xFF, marker_byte]),
            "marker FF{marker_byte:02X} must be emitted"
        );
        let decoded = strict_decode_native(&encoded.codestream);
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.num_components, 1);
    }
}

#[test]
fn cpu_classic_lossy_multi_tile_emits_ppm_and_ppt_that_strict_decode_uses() {
    for (marker, marker_byte) in [(J2kMarkerSegment::Ppm, 0x60), (J2kMarkerSegment::Ppt, 0x61)] {
        let pixels: Vec<u8> = (0..64 * 64)
            .map(|index| masked_u8((index * 37) ^ (index / 17)))
            .collect();
        let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let encoded = encode_j2k_lossy(
            samples,
            &J2kLossyEncodeOptions::default()
                .with_backend(EncodeBackendPreference::CpuOnly)
                .with_max_decomposition_levels(Some(0))
                .with_tile_size(Some((32, 32)))
                .with_marker_segments(vec![marker]),
        )
        .expect("classic lossy multi-tile separated packet-header encode");

        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xFF, marker_byte]),
            "marker FF{marker_byte:02X} must be emitted"
        );
        let sot_count = encoded
            .codestream
            .windows(2)
            .filter(|marker| *marker == [0xFF, 0x90])
            .count();
        assert_eq!(sot_count, 4);
        let decoded = strict_decode_native(&encoded.codestream);
        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert_eq!(decoded.num_components, 1);
    }
}

#[test]
fn cpu_classic_lossy_multi_tile_codestream_decodes() {
    let pixels: Vec<u8> = (0..96 * 80)
        .map(|index| masked_u8(index * 23 + index / 13))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 96, 80, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(1.5)))
        .with_marker_segments(vec![J2kMarkerSegment::Tlm]);
    options.tile_size = Some((48, 40));

    let encoded = encode_j2k_lossy(samples, &options).expect("classic lossy multi-tile encode");

    let sot_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xFF, 0x90])
        .count();
    assert_eq!(sot_count, 4);
    let tlm_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xFF, 0x55])
        .count();
    assert_eq!(tlm_count, 4);

    let decoded = decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 96);
    assert_eq!(decoded.height, 80);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_multi_tile_emits_plt_and_plm() {
    let pixels: Vec<u8> = (0..96 * 80)
        .map(|index| masked_u8(index * 41 + index / 23))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 96, 80, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(1.5)))
        .with_marker_segments(vec![J2kMarkerSegment::Plt, J2kMarkerSegment::Plm]);
    options.tile_size = Some((48, 40));

    let encoded =
        encode_j2k_lossy(samples, &options).expect("classic lossy multi-tile PLT/PLM encode");

    let plt_count = encoded
        .codestream
        .windows(2)
        .filter(|marker| *marker == [0xFF, 0x58])
        .count();
    assert_eq!(plt_count, 4);
    assert!(
        encoded
            .codestream
            .windows(2)
            .any(|marker| marker == [0xFF, 0x57]),
        "PLM marker must be emitted"
    );

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 96);
    assert_eq!(decoded.height, 80);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_emits_multiple_tile_parts_that_strict_decode_uses() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8(index * 29 + index / 11))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();
    let options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(1.75)))
        .with_max_decomposition_levels(Some(2))
        .with_tile_part_packet_limit(Some(1));

    let encoded =
        encode_j2k_lossy(samples, &options).expect("classic lossy multi-tile-part encode");

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
}

#[test]
fn cpu_classic_lossy_emits_tlm_for_multiple_tile_parts() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8(index * 43 + index / 17))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();
    let options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_rate_target(Some(J2kRateTarget::BitsPerPixel(1.75)))
        .with_max_decomposition_levels(Some(2))
        .with_tile_part_packet_limit(Some(1))
        .with_marker_segments(vec![J2kMarkerSegment::Tlm]);

    let encoded = encode_j2k_lossy(samples, &options).expect("classic lossy TLM encode");

    let tlm = tlm_tile_part_lengths(&encoded.codestream);
    let sot = sot_tile_part_lengths(&encoded.codestream);
    assert!(sot.len() > 1, "expected multiple tile-parts");
    assert_eq!(tlm, sot);

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_emits_ppm_and_ppt_across_multiple_tile_parts_that_strict_decode_uses() {
    for (marker, marker_byte) in [(J2kMarkerSegment::Ppm, 0x60), (J2kMarkerSegment::Ppt, 0x61)] {
        let pixels: Vec<u8> = (0..64 * 64)
            .map(|index| masked_u8((index * 47) ^ (index / 23)))
            .collect();
        let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

        let encoded = encode_j2k_lossy(
            samples,
            &J2kLossyEncodeOptions::default()
                .with_backend(EncodeBackendPreference::CpuOnly)
                .with_max_decomposition_levels(Some(2))
                .with_tile_part_packet_limit(Some(1))
                .with_marker_segments(vec![marker]),
        )
        .expect("classic lossy multi-tile-part separated packet-header encode");

        assert!(
            encoded
                .codestream
                .windows(2)
                .any(|window| window == [0xFF, marker_byte]),
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
    }
}

#[test]
fn cpu_classic_lossy_writes_explicit_single_precinct_exponents() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8(index * 31 + index / 17))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_max_decomposition_levels(Some(1));
    options.precinct_exponents = vec![(15, 15), (15, 15)];

    let encoded = encode_j2k_lossy(samples, &options).expect("classic lossy precinct encode");

    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(encoded.codestream[cod + 4] & 0x01, 0x01);
    assert_eq!(
        u16::from_be_bytes([encoded.codestream[cod + 2], encoded.codestream[cod + 3]]),
        14
    );
    assert_eq!(&encoded.codestream[cod + 14..cod + 16], &[0xFF, 0xFF]);

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 64);
    assert_eq!(decoded.height, 64);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_classic_lossy_splits_packets_by_precinct() {
    let pixels: Vec<u8> = (0..128 * 128)
        .map(|index| masked_u8(index * 37 + index / 19))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 128, 128, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_max_decomposition_levels(Some(0))
        .with_marker_segments(vec![J2kMarkerSegment::Plt]);
    options.precinct_exponents = vec![(6, 6)];

    let encoded = encode_j2k_lossy(samples, &options).expect("classic lossy precinct encode");

    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(encoded.codestream[cod + 4] & 0x01, 0x01);
    assert_eq!(
        u16::from_be_bytes([encoded.codestream[cod + 2], encoded.codestream[cod + 3]]),
        13
    );
    assert_eq!(encoded.codestream[cod + 14], 0x66);
    assert_eq!(plt_packet_length_count(&encoded.codestream), 4);

    let decoded = strict_decode_native(&encoded.codestream);
    assert_eq!(decoded.width, 128);
    assert_eq!(decoded.height, 128);
    assert_eq!(decoded.num_components, 1);
}

#[test]
fn cpu_htj2k_lossy_reports_rate_granularity() {
    let pixels: Vec<u8> = (0..32 * 32).map(masked_u8).collect();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("HTJ2K lossy encode");

    assert_eq!(
        encoded.report.ht_rate_granularity_bytes,
        Some(encoded.codestream.len() as u64)
    );
    assert_eq!(decode_native(&encoded.codestream).num_components, 1);
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
            assert_eq!(codestream[offset + 5], 0x60);
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
fn cpu_htj2k_lossy_three_quality_layers_use_three_pass_segment_granularity() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| masked_u8((index * 43) ^ (index / 29)))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).unwrap();

    let encoded = encode_j2k_lossy(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::CpuOnly)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_quality_layers(vec![
                J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(0.75)),
                J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(1.5)),
                J2kQualityLayer::new(J2kRateTarget::BitsPerPixel(3.0)),
            ])
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("HTJ2K multilayer lossy encode");

    assert_eq!(encoded.report.quality_layers, 3);
    assert!(encoded.report.ht_rate_granularity_bytes.is_some());
    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(
        u16::from_be_bytes([encoded.codestream[cod + 6], encoded.codestream[cod + 7]]),
        3
    );
    assert_eq!(decode_native(&encoded.codestream).num_components, 1);
}

#[test]
fn accelerator_facade_htj2k_lossy_multilayer_require_device_checks_supported_stages() {
    #[derive(Default)]
    struct FullHtj2kLossyAccelerator {
        deinterleave: usize,
        quantize_subband: usize,
        ht_code_block: usize,
        packetization: usize,
    }

    impl J2kEncodeStageAccelerator for FullHtj2kLossyAccelerator {
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
            assert_eq!(job.bit_depth, 8);
            assert!(!job.signed);
            let mut component = Vec::with_capacity(job.num_pixels);
            for &sample in job.pixels {
                component.push(f32::from(sample) - 128.0);
            }
            Ok(Some(vec![component]))
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
            assert_eq!(job.target_coding_passes, 3);
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

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
            self.packetization = self.packetization.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..32 * 32)
        .map(|index| masked_u8(index * 47 + index / 31))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_quality_layers(vec![
            J2kQualityLayer::new(J2kRateTarget::Bytes(100_000)),
            J2kQualityLayer::new(J2kRateTarget::Bytes(100_000)),
            J2kQualityLayer::new(J2kRateTarget::Bytes(100_000)),
        ])
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    let mut accelerator = FullHtj2kLossyAccelerator::default();

    let encoded =
        encode_j2k_lossy_with_accelerator(samples, &options, BackendKind::Cuda, &mut accelerator)
            .expect("HTJ2K lossy multilayer required stages should dispatch");

    assert_eq!(encoded.backend, BackendKind::Cuda);
    assert!(accelerator.deinterleave > 0);
    assert!(accelerator.quantize_subband > 0);
    assert!(accelerator.ht_code_block > 0);
    assert_eq!(decode_native(&encoded.codestream).num_components, 1);
}

#[test]
fn lossy_require_device_uses_only_the_selected_final_attempt_dispatch() {
    #[derive(Default)]
    struct FirstAttemptOnlyDispatch {
        packetization_attempts: usize,
    }

    impl J2kEncodeStageAccelerator for FirstAttemptOnlyDispatch {
        fn dispatch_report(&self) -> J2kEncodeDispatchReport {
            if self.packetization_attempts == 0 {
                return J2kEncodeDispatchReport::default();
            }
            J2kEncodeDispatchReport {
                deinterleave: 1,
                quantize_subband: 1,
                ht_code_block: 1,
                packetization: 1,
                ..J2kEncodeDispatchReport::default()
            }
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> J2kEncodeStageResult<Option<Vec<u8>>> {
            self.packetization_attempts = self.packetization_attempts.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..32 * 32)
        .map(|index| masked_u8(index * 31 + index / 17))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_rate_target(Some(J2kRateTarget::Bytes(100_000)))
        .with_validation(J2kEncodeValidation::External);
    options.psnr_iteration_budget = 1;
    let mut accelerator = FirstAttemptOnlyDispatch::default();

    let error =
        encode_j2k_lossy_with_accelerator(samples, &options, BackendKind::Cuda, &mut accelerator)
            .expect_err("the returned final attempt did not dispatch required device stages");

    assert!(error.is_unsupported());
    assert!(
        accelerator.packetization_attempts >= 2,
        "fixture must include search attempts plus the selected final re-encode"
    );
}
