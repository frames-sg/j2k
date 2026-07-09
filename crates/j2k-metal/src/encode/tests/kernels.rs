// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_dispatch_round_trips_gray8_lossless_tile() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..64 * 64).map(|i| ((i * 5) & 0xFF) as u8).collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_max_decomposition_levels(Some(1));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with metal forward DWT 5/3");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(accelerator.forward_dwt53_attempts(), 1);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_lossless_facade_dispatches_rct_and_dwt_for_wsi_sized_rgb_tile() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(128 * 128 * 3);
    for y in 0..128u32 {
        for x in 0..128u32 {
            pixels.push(((x * 3 + y * 5) & 0xFF) as u8);
            pixels.push(((x * 7 + y * 11) & 0xFF) as u8);
            pixels.push(((x * 13 + y * 17) & 0xFF) as u8);
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, 128, 128, 3, 8, false).expect("valid RGB samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(accelerator.forward_rct_dispatches(), 1);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 3);
    assert!(accelerator.tier1_code_block_attempts() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert!(accelerator.tier1_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_uses_one_batched_dispatch_for_multiple_code_blocks() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..256 * 256)
        .map(|idx| ((idx * 17 + 3) & 0xFF) as u8)
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_max_decomposition_levels(Some(0));
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("encode with batched Metal classic Tier-1");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert!(accelerator.tier1_code_block_attempts() > 1);
    assert_eq!(accelerator.tier1_code_block_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_resident_uses_mq_byte_split_gpu_token_pack_by_default() {
    if !should_run_metal_runtime() {
        return;
    }

    let _profile_guard = compute::force_metal_profile_stages_for_test(true);
    compute::reset_classic_gpu_token_pack_dispatches_for_test();
    compute::reset_classic_split_mq_byte_gpu_token_pack_dispatches_for_test();
    let first: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x + y * 5) & 0xFF) as u8
        })
        .collect();
    let second: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x * 3 + y * 7) & 0xFF) as u8
        })
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
    ];

    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::Classic,
            validation: J2kEncodeValidation::External,
        },
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("resident batch encode with default MQ-byte GPU token-pack Classic Tier-1");
    assert_eq!(encoded.outcomes.len(), 2);
    for (outcome, expected) in encoded.outcomes.iter().zip([&first, &second]) {
        let codestream = outcome
            .encoded
            .to_encoded_j2k()
            .expect("codestream readback");
        let decoded = Image::new(&codestream.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(codestream.backend, BackendKind::Metal);
        assert_eq!(&decoded.data, expected);
    }
    assert!(
        compute::classic_gpu_token_pack_dispatches_for_test() > 0,
        "default Classic GPU token-pack route was not dispatched"
    );
    assert!(
        compute::classic_split_mq_byte_gpu_token_pack_dispatches_for_test() > 0,
        "default Classic GPU token-pack route did not use MQ-byte split token emit"
    );
    assert_eq!(
        encoded
            .stats
            .stage_stats
            .tier1_token_pack_output_bytes_total,
        encoded.stats.stage_stats.tier1_output_used_bytes_total,
        "default Classic GPU token-pack route should attribute Tier-1 output bytes to token pack"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_resident_gpu_token_pack_route_round_trips() {
    if !should_run_metal_runtime() {
        return;
    }

    let _guard = compute::force_classic_gpu_token_pack_route_for_test(true);
    let _profile_guard = compute::force_metal_profile_stages_for_test(true);
    compute::reset_classic_gpu_token_pack_dispatches_for_test();
    let first: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x + y * 3) & 0xFF) as u8
        })
        .collect();
    let second: Vec<u8> = (0..256 * 256)
        .map(|idx| {
            let x = idx % 256;
            let y = idx / 256;
            ((x * 2 + y) & 0xFF) as u8
        })
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let first_buffer = crate::benchmark_private_buffer_with_bytes(&session, &first)
        .expect("private benchmark input buffer");
    let second_buffer = crate::benchmark_private_buffer_with_bytes(&session, &second)
        .expect("private benchmark input buffer");
    let tiles = [
        super::super::MetalLosslessEncodeTile {
            buffer: &first_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width: 256,
            height: 256,
            pitch_bytes: 256,
            output_width: 256,
            output_height: 256,
            format: PixelFormat::Gray8,
        },
    ];

    let encoded = super::super::encode_lossless_from_padded_metal_buffers_to_metal_batch(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::Classic,
            validation: J2kEncodeValidation::External,
        },
        &session,
        super::super::MetalLosslessEncodeConfig {
            gpu_encode_inflight_tiles: Some(2),
            gpu_encode_memory_budget_bytes: Some(1024 * 1024 * 1024),
        },
    )
    .expect("resident batch encode with gated GPU token-pack Classic Tier-1");
    assert_eq!(encoded.outcomes.len(), 2);
    for (outcome, expected) in encoded.outcomes.iter().zip([&first, &second]) {
        let codestream = outcome
            .encoded
            .to_encoded_j2k()
            .expect("codestream readback");
        let decoded = Image::new(&codestream.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");

        assert_eq!(codestream.backend, BackendKind::Metal);
        assert_eq!(&decoded.data, expected);
    }
    assert!(
        compute::classic_gpu_token_pack_dispatches_for_test() > 0,
        "gated Classic GPU token-pack route was not dispatched"
    );
    assert!(
        encoded.stats.stage_stats.tier1_token_emit_token_bytes_total > 0,
        "gated Classic GPU token-pack route did not expose token-emitter byte counters"
    );
    assert!(
        encoded
            .stats
            .stage_stats
            .tier1_token_emit_segment_count_total
            > 0,
        "gated Classic GPU token-pack route did not expose token segment counters"
    );
    assert_eq!(
        encoded
            .stats
            .stage_stats
            .tier1_token_pack_output_bytes_total,
        encoded.stats.stage_stats.tier1_output_used_bytes_total,
        "gated Classic GPU token-pack route should attribute Tier-1 output bytes to token pack"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_uses_one_batched_dispatch_for_multiple_code_blocks() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..256 * 256)
        .map(|idx| ((idx * 23 + 9) & 0xFF) as u8)
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 256, 256, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert!(accelerator.ht_code_block_attempts() > 1);
    assert_eq!(accelerator.ht_code_block_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_lossless_facade_dispatches_ht_code_blocks_and_packetization() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..64).map(|value| ((value * 13) & 0xFF) as u8).collect();
    let samples = J2kLosslessSamples::new(&pixels, 8, 8, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &lossless_options! {
            backend: EncodeBackendPreference::RequireDevice,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
        },
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossless encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(encoded.dispatch_report.deinterleave, 1);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert!(encoded.dispatch_report.quantize_subband > 0);
    assert!(accelerator.quantize_subband_attempts() > 0);
    assert!(accelerator.quantize_subband_dispatches() > 0);
    assert!(accelerator.ht_code_block_attempts() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_lossy_facade_require_device_dispatches_supported_stages() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..16 * 16)
        .map(|idx| ((idx * 17 + idx / 3) & 0xFF) as u8)
        .collect();
    let samples = J2kLossySamples::new(&pixels, 16, 16, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossy encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(encoded.dispatch_report.deinterleave, 1);
    assert_eq!(accelerator.deinterleave_attempts(), 1);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert!(encoded.dispatch_report.quantize_subband > 0);
    assert!(accelerator.quantize_subband_attempts() > 0);
    assert!(accelerator.quantize_subband_dispatches() > 0);
    assert!(accelerator.ht_code_block_attempts() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_attempts(), 1);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_lossy_rgb_facade_reports_forward_ict_dispatch() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 16;
    let height = 16;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|idx| ((idx * 19 + idx / 5 + 7) & 0xFF) as u8)
        .collect();
    let samples =
        J2kLossySamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated RGB HTJ2K lossy encode");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert_eq!(encoded.dispatch_report.deinterleave, 1);
    assert_eq!(encoded.dispatch_report.forward_ict, 1);
    assert_eq!(encoded.dispatch_report.forward_rct, 0);
    assert_eq!(encoded.dispatch_report.forward_dwt97, 0);
    assert!(encoded.dispatch_report.quantize_subband > 0);
    assert_eq!(accelerator.forward_ict_attempts(), 1);
    assert_eq!(accelerator.forward_ict_dispatches(), 1);
    assert!(accelerator.quantize_subband_attempts() > 0);
    assert!(accelerator.quantize_subband_dispatches() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_dispatches(), 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_lossy_facade_reports_forward_dwt97_dispatch() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 64;
    let height = 64;
    let pixels: Vec<u8> = (0..width * height)
        .map(|idx| ((idx * 23 + idx / 11 + 31) & 0xFF) as u8)
        .collect();
    let samples =
        J2kLossySamples::new(&pixels, width, height, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let encoded = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(2))
            .with_validation(J2kEncodeValidation::CpuRoundTrip),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Metal-accelerated HTJ2K lossy encode with DWT 9/7");

    assert_eq!(encoded.backend, BackendKind::Metal);
    assert!(accelerator.forward_dwt97_attempts() > 0);
    assert!(accelerator.forward_dwt97_dispatches() > 0);
    assert!(encoded.dispatch_report.forward_dwt97 > 0);
    assert!(encoded.dispatch_report.quantize_subband > 0);
    assert!(accelerator.quantize_subband_dispatches() > 0);
    assert!(accelerator.ht_code_block_dispatches() > 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_kernel_matches_scalar_oracle() {
    if !should_run_metal_runtime() {
        return;
    }

    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 37 + 11) & 0x1ff) - 255;
            if idx % 5 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        sub_band_type: j2k_native::J2kSubBandType::HighHigh,
        total_bitplanes: 9,
        style,
    };

    let gpu = compute::encode_classic_tier1_code_block(job).expect("Metal classic encode");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        8,
        8,
        j2k_native::J2kSubBandType::HighHigh,
        9,
        style,
    )
    .expect("scalar classic encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_kernel_matches_scalar_for_terminated_passes() {
    if !should_run_metal_runtime() {
        return;
    }

    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 43 + 5) & 0x3ff) - 511;
            if idx % 6 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: true,
        termination_on_each_pass: true,
        vertically_causal_context: false,
        segmentation_symbols: true,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        sub_band_type: j2k_native::J2kSubBandType::LowHigh,
        total_bitplanes: 10,
        style,
    };

    let gpu =
        compute::encode_classic_tier1_code_block(job).expect("Metal classic terminated encode");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        8,
        8,
        j2k_native::J2kSubBandType::LowHigh,
        10,
        style,
    )
    .expect("scalar classic terminated encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_kernel_matches_scalar_for_selective_bypass() {
    if !should_run_metal_runtime() {
        return;
    }

    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 61 + 29) & 0x7ff) - 1023;
            if idx % 4 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        sub_band_type: j2k_native::J2kSubBandType::HighLow,
        total_bitplanes: 11,
        style,
    };

    let gpu = compute::encode_classic_tier1_code_block(job).expect("Metal classic bypass encode");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        8,
        8,
        j2k_native::J2kSubBandType::HighLow,
        11,
        style,
    )
    .expect("scalar classic bypass encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_batched_bypass_u16_32_matches_scalar() {
    if !should_run_metal_runtime() {
        return;
    }

    let coeffs: Vec<i32> = (0..32 * 32)
        .map(|idx| {
            let value = ((idx * 97 + idx / 3 + 19) & 0x7ff) - 1023;
            if idx % 11 == 0 || idx % 17 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let job = j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 32,
        height: 32,
        sub_band_type: j2k_native::J2kSubBandType::HighHigh,
        total_bitplanes: 11,
        style,
    };

    let gpu = compute::encode_classic_tier1_code_blocks(&[job])
        .expect("batched Metal classic bypass_u16_32 encode")
        .pop()
        .expect("one encoded codeblock");
    let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
        &coeffs,
        32,
        32,
        j2k_native::J2kSubBandType::HighHigh,
        11,
        style,
    )
    .expect("scalar classic bypass encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.segments.len(), cpu.segments.len());
    for (gpu_segment, cpu_segment) in gpu.segments.iter().zip(cpu.segments.iter()) {
        assert_eq!(gpu_segment.data_offset, cpu_segment.data_offset);
        assert_eq!(gpu_segment.data_length, cpu_segment.data_length);
        assert_eq!(gpu_segment.start_coding_pass, cpu_segment.start_coding_pass);
        assert_eq!(gpu_segment.end_coding_pass, cpu_segment.end_coding_pass);
        assert_eq!(gpu_segment.use_arithmetic, cpu_segment.use_arithmetic);
    }
    assert_eq!(gpu.number_of_coding_passes, cpu.number_of_coding_passes);
    assert_eq!(gpu.missing_bit_planes, cpu.missing_bit_planes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_classic_tier1_token_routes_match_scalar_bytes() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_coeffs: Vec<i32> = (0..32 * 32)
        .map(|idx| {
            let value = ((idx * 37 + idx / 5 + 31) & 0xff) - 127;
            if idx % 5 == 0 || idx % 11 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let second_coeffs: Vec<i32> = (0..17 * 29)
        .map(|idx| {
            let value = ((idx * 73 + idx / 7 + 11) & 0xff) - 127;
            if idx % 7 == 0 || idx % 23 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let style = J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: true,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    };
    let jobs = [
        j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &first_coeffs,
            width: 32,
            height: 32,
            sub_band_type: j2k_native::J2kSubBandType::HighHigh,
            total_bitplanes: 8,
            style,
        },
        j2k_native::J2kTier1CodeBlockEncodeJob {
            coefficients: &second_coeffs,
            width: 17,
            height: 29,
            sub_band_type: j2k_native::J2kSubBandType::LowLow,
            total_bitplanes: 8,
            style,
        },
    ];

    let gpu_packed = compute::encode_classic_tier1_code_blocks_via_gpu_token_pack_for_test(&jobs)
        .expect("Metal classic GPU token-pack encode");
    let cpu_packed =
        compute::encode_classic_tier1_code_blocks_via_ordered_tokens_cpu_pack_for_test(&jobs)
            .expect("Metal classic ordered-token CPU-pack encode");
    let split_packed =
        compute::encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_cpu_pack_for_test(&jobs)
            .expect("Metal classic split MQ/raw token CPU-pack encode");
    let split_gpu_packed =
        compute::encode_classic_tier1_code_blocks_via_split_mq_raw_tokens_gpu_pack_for_test(&jobs)
            .expect("Metal classic split MQ/raw token GPU-pack encode");
    let mq_byte_split_gpu_packed =
        compute::encode_classic_tier1_code_blocks_via_split_mq_byte_raw_tokens_gpu_pack_for_test(
            &jobs,
        )
        .expect("Metal classic split MQ-byte/raw-bit token GPU-pack encode");

    assert_eq!(gpu_packed.len(), jobs.len());
    assert_eq!(cpu_packed.len(), jobs.len());
    assert_eq!(split_packed.len(), jobs.len());
    assert_eq!(split_gpu_packed.len(), jobs.len());
    assert_eq!(mq_byte_split_gpu_packed.len(), jobs.len());
    for (
        (
            (((gpu_block, cpu_packed_block), split_packed_block), split_gpu_packed_block),
            mq_byte_split_gpu_packed_block,
        ),
        job,
        coeffs,
    ) in gpu_packed
        .iter()
        .zip(cpu_packed.iter())
        .zip(split_packed.iter())
        .zip(split_gpu_packed.iter())
        .zip(mq_byte_split_gpu_packed.iter())
        .zip(jobs.iter())
        .zip([&first_coeffs, &second_coeffs])
        .map(|((blocks, job), coeffs)| (blocks, job, coeffs))
    {
        let cpu = j2k_native::encode_j2k_code_block_scalar_with_style(
            coeffs,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            style,
        )
        .expect("scalar classic bypass encode");

        assert_eq!(gpu_block.data, cpu.data);
        assert_eq!(gpu_block.segments, cpu.segments);
        assert_eq!(
            gpu_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(gpu_block.missing_bit_planes, cpu.missing_bit_planes);
        assert_eq!(cpu_packed_block.data, cpu.data);
        assert_eq!(cpu_packed_block.segments, cpu.segments);
        assert_eq!(
            cpu_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(cpu_packed_block.missing_bit_planes, cpu.missing_bit_planes);
        assert_eq!(split_packed_block.data, cpu.data);
        assert_eq!(split_packed_block.segments, cpu.segments);
        assert_eq!(
            split_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(
            split_packed_block.missing_bit_planes,
            cpu.missing_bit_planes
        );
        assert_eq!(split_gpu_packed_block.data, cpu.data);
        assert_eq!(split_gpu_packed_block.segments, cpu.segments);
        assert_eq!(
            split_gpu_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(
            split_gpu_packed_block.missing_bit_planes,
            cpu.missing_bit_planes
        );
        assert_eq!(mq_byte_split_gpu_packed_block.data, cpu.data);
        assert_eq!(mq_byte_split_gpu_packed_block.segments, cpu.segments);
        assert_eq!(
            mq_byte_split_gpu_packed_block.number_of_coding_passes,
            cpu.number_of_coding_passes
        );
        assert_eq!(
            mq_byte_split_gpu_packed_block.missing_bit_planes,
            cpu.missing_bit_planes
        );
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_htj2k_cleanup_kernel_matches_scalar_oracle() {
    if !should_run_metal_runtime() {
        return;
    }

    let coeffs: Vec<i32> = (0..64)
        .map(|idx| {
            let value = ((idx * 19 + 7) & 0xff) - 127;
            if idx % 7 == 0 {
                0
            } else {
                value
            }
        })
        .collect();
    let job = j2k_native::J2kHtCodeBlockEncodeJob {
        coefficients: &coeffs,
        width: 8,
        height: 8,
        total_bitplanes: 8,
        target_coding_passes: 1,
    };

    let gpu = compute::encode_ht_cleanup_code_block(job).expect("Metal HT encode");
    let cpu = j2k_native::encode_ht_code_block_scalar(&coeffs, 8, 8, 8).expect("scalar HT encode");

    assert_eq!(gpu.data, cpu.data);
    assert_eq!(gpu.num_coding_passes, cpu.num_coding_passes);
    assert_eq!(gpu.num_zero_bitplanes, cpu.num_zero_bitplanes);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tier2_packetization_kernel_matches_scalar_oracle() {
    if !should_run_metal_runtime() {
        return;
    }

    let block0 = [0x12, 0x34, 0x56, 0x78];
    let block1 = [0x9a, 0xbc];
    let code_blocks = vec![
        j2k_native::J2kPacketizationCodeBlock {
            data: &block0,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 2,
            previously_included: false,
            l_block: 3,
            block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
        },
        j2k_native::J2kPacketizationCodeBlock {
            data: &block1,
            ht_cleanup_length: u32::try_from(block1.len()).expect("test payload fits u32"),
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: 1,
            previously_included: false,
            l_block: 3,
            block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::HighThroughput,
        },
    ];
    let subband = j2k_native::J2kPacketizationSubband {
        code_blocks,
        num_cbs_x: 2,
        num_cbs_y: 1,
    };
    let resolution = j2k_native::J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let resolutions = [resolution];
    let packet_descriptors = [j2k_native::J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    }];
    let job = j2k_native::J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 2,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
    let cpu = j2k_native::encode_j2k_packetization_scalar(job).expect("scalar packet encode");

    assert_eq!(gpu, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tier2_packetization_reuses_descriptor_state_across_layers() {
    if !should_run_metal_runtime() {
        return;
    }

    let block0 = vec![0x11];
    let block1 = vec![0x22];
    let first = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block0,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let second = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block1,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let resolutions = [first, second];
    let packet_descriptors = [
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let job = j2k_native::J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 2,
        num_components: 1,
        code_block_count: 2,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Rpcl,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
    let cpu = j2k_native::encode_j2k_packetization_scalar(job).expect("scalar packet encode");

    assert_eq!(gpu, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_tier2_packetization_honors_explicit_descriptor_order() {
    if !should_run_metal_runtime() {
        return;
    }

    let block0 = vec![0xA0];
    let block1 = vec![0xB0];
    let first = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block0,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let second = j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &block1,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: j2k_native::J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let resolutions = [first, second];
    let packet_descriptors = [
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 1,
            layer: 0,
            resolution: 1,
            component: 0,
            precinct: 0,
        },
        j2k_native::J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let job = j2k_native::J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 1,
        num_components: 1,
        code_block_count: 2,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Rpcl,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let gpu = compute::encode_tier2_packetization(job).expect("Metal packet encode");
    let cpu = j2k_native::encode_j2k_packetization_scalar(job).expect("scalar packet encode");

    assert_eq!(gpu, cpu);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_handles_single_sample_edge_dimensions() {
    if !should_run_metal_runtime() {
        return;
    }

    for (width, height) in [(1, 8), (8, 1)] {
        let samples: Vec<f32> = (0..width * height)
            .map(|i| {
                f32::from(
                    u8::try_from((i * 11 + width * 3 + height * 5) & 0xFF)
                        .expect("masked sample fits in u8"),
                ) - 128.0
            })
            .collect();
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let output = accelerator
            .encode_forward_dwt53(J2kForwardDwt53Job {
                samples: &samples,
                width,
                height,
                num_levels: 1,
            })
            .expect("metal DWT 5/3 stage")
            .expect("metal DWT 5/3 dispatch");

        assert_eq!(output.ll_width, width.div_ceil(2));
        assert_eq!(output.ll_height, height.div_ceil(2));
        assert_eq!(output.levels.len(), 1);
        assert_eq!(accelerator.forward_dwt53_attempts(), 1);
        assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    }
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt53_matches_reference_for_fractional_stage_samples() {
    fn assert_slice_near(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.0001,
                "{label}[{index}] mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    if !should_run_metal_runtime() {
        return;
    }

    let width = 8;
    let height = 8;
    let samples = (0..width * height)
        .map(|idx| f32::from(u16::try_from(idx).expect("test index fits u16")) * 0.5 - 15.25)
        .collect::<Vec<_>>();
    let expected = forward_dwt53_reference(&samples, width, height, 1);
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let actual = accelerator
        .encode_forward_dwt53(J2kForwardDwt53Job {
            samples: &samples,
            width,
            height,
            num_levels: 1,
        })
        .expect("metal DWT 5/3 stage")
        .expect("metal DWT 5/3 dispatch");

    assert_eq!(actual.ll_width, expected.ll_width);
    assert_eq!(actual.ll_height, expected.ll_height);
    assert_slice_near(&actual.ll, &expected.ll, "LL");
    assert_eq!(actual.levels.len(), expected.levels.len());
    for (index, (actual, expected)) in actual.levels.iter().zip(&expected.levels).enumerate() {
        assert_eq!(actual.width, expected.width, "level {index} width");
        assert_eq!(actual.height, expected.height, "level {index} height");
        assert_eq!(
            actual.low_width, expected.low_width,
            "level {index} low width"
        );
        assert_eq!(
            actual.low_height, expected.low_height,
            "level {index} low height"
        );
        assert_eq!(
            actual.high_width, expected.high_width,
            "level {index} high width"
        );
        assert_eq!(
            actual.high_height, expected.high_height,
            "level {index} high height"
        );
        assert_slice_near(&actual.hl, &expected.hl, "HL");
        assert_slice_near(&actual.lh, &expected.lh, "LH");
        assert_slice_near(&actual.hh, &expected.hh, "HH");
    }
}

#[cfg(target_os = "macos")]
fn native_lossy_dwt97_options(num_decomposition_levels: u8) -> EncodeOptions {
    EncodeOptions {
        num_decomposition_levels,
        reversible: false,
        guard_bits: 2,
        use_ht_block_coding: true,
        ..Default::default()
    }
}

#[cfg(target_os = "macos")]
fn assert_metal_dwt97_matches_native_encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_decomposition_levels: u8,
) {
    let options = native_lossy_dwt97_options(num_decomposition_levels);
    let expected = j2k_native::encode(pixels, width, height, 1, 8, false, &options)
        .expect("native lossy DWT 9/7 encode");
    let mut accelerator = MetalEncodeStageAccelerator::for_forward_dwt97_encode();
    let actual = {
        j2k_native::encode_with_accelerator(
            pixels,
            width,
            height,
            1,
            8,
            false,
            &options,
            &mut accelerator,
        )
        .expect("Metal DWT 9/7 lossy encode")
    };

    assert_eq!(actual, expected);
    assert_eq!(accelerator.forward_dwt97_attempts(), 1);
    assert_eq!(accelerator.forward_dwt97_dispatches(), 1);
    let report = accelerator.dispatch_report();
    assert_eq!(report.forward_dwt97, 1);
    assert_eq!(report.forward_dwt53, 0);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt97_single_level_matches_native_encode_output() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 17;
    let height = 15;
    let pixels = (0..width * height)
        .map(|idx| ((idx * 29 + idx / 5 + 17) & 0xFF) as u8)
        .collect::<Vec<_>>();

    assert_metal_dwt97_matches_native_encode(&pixels, width, height, 1);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_forward_dwt97_multi_level_matches_native_encode_output() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 32;
    let height = 16;
    let pixels = (0..width * height)
        .map(|idx| ((idx * 43 + idx / 7 + 91) & 0xFF) as u8)
        .collect::<Vec<_>>();

    assert_metal_dwt97_matches_native_encode(&pixels, width, height, 3);
}
