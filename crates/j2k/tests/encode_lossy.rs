// SPDX-License-Identifier: Apache-2.0

use j2k::adapter::encode_stage::{
    EncodedHtJ2kCodeBlock, J2kDeinterleaveToF32Job, J2kEncodeDispatchReport,
    J2kEncodeStageAccelerator, J2kHtCodeBlockEncodeJob, J2kPacketizationEncodeJob,
    J2kQuantizeSubbandJob,
};
use j2k::{
    encode_j2k_lossy, encode_j2k_lossy_with_accelerator, EncodeBackendPreference,
    J2kBlockCodingMode, J2kEncodeValidation, J2kLossyEncodeOptions, J2kLossySamples,
    J2kMarkerSegment, J2kProgressionOrder, J2kQualityLayer, J2kRateTarget,
};
use j2k_core::{BackendKind, CodecError};
use j2k_native::{DecodeSettings, Image};

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
    EncodedHtJ2kCodeBlock {
        data: block.data,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        num_coding_passes: block.num_coding_passes,
        num_zero_bitplanes: block.num_zero_bitplanes,
    }
}

fn plt_packet_length_count(codestream: &[u8]) -> usize {
    plt_packet_lengths(codestream).len()
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
fn lossy_sample_descriptor_rejects_more_than_sixteen_bits_explicitly() {
    let pixels = vec![0u8; 2 * 2 * 2];

    let err = J2kLossySamples::new(&pixels, 2, 2, 1, 17, false).unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("1-16 bits per sample"));
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
        .map(|index| (((index * 13) ^ (index / 5)) & 0xFF) as u8)
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
fn cpu_classic_lossy_multiple_quality_layers_encode_scalable_codestream() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| (((index * 17) + (index / 3)) & 0xFF) as u8)
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
        .map(|index| (((index * 53) ^ (index / 11)) & 0xFF) as u8)
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
        .map(|index| (((index * 29) ^ (index / 7)) & 0xFF) as u8)
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
        .map(|index| (((index * 11) + (index / 9)) & 0xFF) as u8)
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
        .map(|index| (((index * 19) ^ (index / 11)) & 0xFF) as u8)
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
fn cpu_classic_lossy_multi_tile_codestream_decodes() {
    let pixels: Vec<u8> = (0..96 * 80)
        .map(|index| (((index * 23) + (index / 13)) & 0xFF) as u8)
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
        .map(|index| (((index * 41) + (index / 23)) & 0xFF) as u8)
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
fn cpu_classic_lossy_writes_explicit_single_precinct_exponents() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| (((index * 31) + (index / 17)) & 0xFF) as u8)
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
        .map(|index| (((index * 37) + (index / 19)) & 0xFF) as u8)
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
    let pixels: Vec<u8> = (0..32 * 32).map(|index| (index & 0xFF) as u8).collect();
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

#[test]
fn cpu_htj2k_lossy_multiple_quality_layers_use_segment_granularity() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|index| (((index * 43) ^ (index / 29)) & 0xFF) as u8)
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
            ])
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
    )
    .expect("HTJ2K multilayer lossy encode");

    assert_eq!(encoded.report.quality_layers, 2);
    assert!(encoded.report.ht_rate_granularity_bytes.is_some());
    let cod = encoded
        .codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x52])
        .expect("COD marker");
    assert_eq!(
        u16::from_be_bytes([encoded.codestream[cod + 6], encoded.codestream[cod + 7]]),
        2
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
        ) -> core::result::Result<Option<Vec<Vec<f32>>>, &'static str> {
            self.deinterleave = self.deinterleave.saturating_add(1);
            assert_eq!(job.bit_depth, 8);
            assert!(!job.signed);
            let mut component = Vec::with_capacity(job.num_pixels);
            for &sample in job.pixels {
                component.push(f32::from(sample) - 128.0);
            }
            Ok(Some(vec![component]))
        }

        fn encode_quantize_subband(
            &mut self,
            job: J2kQuantizeSubbandJob<'_>,
        ) -> core::result::Result<Option<Vec<i32>>, &'static str> {
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
        ) -> core::result::Result<Option<EncodedHtJ2kCodeBlock>, &'static str> {
            self.ht_code_block = self.ht_code_block.saturating_add(1);
            j2k_native::encode_ht_code_block_scalar(
                job.coefficients,
                job.width,
                job.height,
                job.total_bitplanes,
            )
            .map(public_encoded_ht)
            .map(Some)
        }

        fn encode_packetization(
            &mut self,
            _job: J2kPacketizationEncodeJob<'_>,
        ) -> core::result::Result<Option<Vec<u8>>, &'static str> {
            self.packetization = self.packetization.saturating_add(1);
            Ok(None)
        }
    }

    let pixels: Vec<u8> = (0..32 * 32)
        .map(|index| (((index * 47) + (index / 31)) & 0xFF) as u8)
        .collect();
    let samples = J2kLossySamples::new(&pixels, 32, 32, 1, 8, false).unwrap();
    let mut options = J2kLossyEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(0))
        .with_quality_layers(vec![
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
