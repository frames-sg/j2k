// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn hybrid_encode_upload_owns_bytes_after_caller_storage_is_reused() {
    if !should_run_metal_runtime() {
        return;
    }

    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let expected = vec![3u8, 5, 8, 13, 21];
    let mut caller_bytes = expected.clone();
    let buffer = super::super::copy_padded_metal_buffer_from_bytes(&session, &caller_bytes)
        .expect("copy hybrid encode input");

    caller_bytes.fill(0xff);
    drop(caller_bytes);
    // SAFETY: The buffer is shared, Metal-owned storage initialized by the
    // synchronous upload helper and has not been submitted for GPU mutation.
    let uploaded =
        unsafe { j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 0, expected.len()) }
            .expect("read Metal-owned hybrid input");
    assert_eq!(uploaded, expected);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_host_output_encode_options_preserve_auto_for_hybrid_path() {
    let routed = super::super::host_output_encode_options(lossless_options! {
        backend: EncodeBackendPreference::Auto,
        validation: J2kEncodeValidation::CpuRoundTrip,
    });

    assert_eq!(routed.backend, EncodeBackendPreference::Auto);
    assert_eq!(routed.validation, J2kEncodeValidation::External);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_classic_host_output_stays_cpu_without_metal_dispatches() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|i| u8::try_from((i * 17) & 0xff).expect("masked pixel fits u8"))
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("hybrid host-output encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
    assert_eq!(accelerator.tier1_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
    assert!(accelerator.prefer_parallel_cpu_code_block_fallback());
}

#[cfg(target_os = "macos")]
#[test]
fn auto_classic_large_host_output_dispatches_benchmark_gated_prep_stages_only() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 1024u32;
    let height = 1024u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|idx| ((idx * 19 + idx / 13) & 0xff) as u8)
        .collect();
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("valid gray samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        max_decomposition_levels: Some(1),
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("benchmark-gated Auto host-output encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.deinterleave_dispatches(), 1);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 1);
    assert!(accelerator.quantize_subband_dispatches() > 0);
    assert_eq!(accelerator.tier1_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn auto_lossy_host_output_stays_cpu_without_metal_dispatches() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|idx| ((idx * 29 + idx / 7) & 0xff) as u8)
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::Auto)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_validation(J2kEncodeValidation::External),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Auto lossy host-output encode should fall back to CPU");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.ht_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
}

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::cast_sign_loss,
    reason = "bounded synthetic pixel expression is nonnegative"
)]
fn auto_lossy_packet_marker_shape_stays_cpu_without_packetization_dispatch() {
    let pixels: Vec<u8> = (0..64 * 64)
        .map(|idx| ((idx * 31 + idx / 5) & 0xff) as u8)
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::Auto)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_marker_segments(vec![J2kMarkerSegment::Sop])
            .with_validation(J2kEncodeValidation::External),
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Auto lossy marker shape should fall back to CPU");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.packetization_attempts(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
}

#[cfg(target_os = "macos")]
#[test]
fn strict_metal_lossy_packet_marker_shape_requires_packetization_dispatch() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..64 * 64)
        .map(|idx| u8::try_from((idx * 37 + idx / 9) & 0xff).expect("masked pixel fits u8"))
        .collect();
    let samples = J2kLossySamples::new(&pixels, 64, 64, 1, 8, false).expect("valid gray samples");
    let mut accelerator = MetalEncodeStageAccelerator::default();

    let err = encode_j2k_lossy_with_accelerator(
        samples,
        &J2kLossyEncodeOptions::default()
            .with_backend(EncodeBackendPreference::RequireDevice)
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_max_decomposition_levels(Some(0))
            .with_marker_segments(vec![J2kMarkerSegment::Sop])
            .with_validation(J2kEncodeValidation::External),
        BackendKind::Metal,
        &mut accelerator,
    )
    .unwrap_err();

    assert!(err.is_unsupported());
    assert!(err.to_string().contains("packetization"));
    assert!(accelerator.ht_code_block_dispatches() > 0);
    assert_eq!(accelerator.packetization_attempts(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_small_host_output_stays_cpu_below_resident_gate() {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    let samples = J2kLosslessSamples::new(&pixels, 64, 64, 3, 8, false).expect("valid RGB samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Auto HTJ2K host-output encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
    assert_eq!(accelerator.ht_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_large_host_output_stays_cpu_for_single_frame() {
    let width = 1024u32;
    let height = 1024u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Auto HTJ2K host-output encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
    assert_eq!(accelerator.ht_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_kodak_sized_rgb_host_output_stays_cpu_for_single_frame() {
    let width = 768u32;
    let height = 512u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 3, 8, false).expect("valid RGB samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Auto HTJ2K host-output encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
    assert_eq!(accelerator.ht_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_gray_host_output_stays_cpu_for_single_frame() {
    let width = 512u32;
    let height = 512u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 11 + (x ^ y)) & 0xff) as u8);
        }
    }
    let samples =
        J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("valid gray samples");
    let options = lossless_options! {
        backend: EncodeBackendPreference::Auto,
        block_coding_mode: J2kBlockCodingMode::HighThroughput,
        validation: J2kEncodeValidation::External,
    };
    let mut accelerator = MetalEncodeStageAccelerator::for_auto_host_output();

    let encoded = encode_j2k_lossless_with_accelerator(
        samples,
        &options,
        BackendKind::Metal,
        &mut accelerator,
    )
    .expect("Auto HTJ2K host-output encode");
    let decoded = Image::new(&encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert_eq!(accelerator.forward_rct_dispatches(), 0);
    assert_eq!(accelerator.forward_dwt53_dispatches(), 0);
    assert_eq!(accelerator.ht_code_block_dispatches(), 0);
    assert_eq!(accelerator.packetization_dispatches(), 0);
    assert_eq!(encoded.dispatch_report, J2kEncodeDispatchReport::default());
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_padded_rgb8_stays_cpu_below_resident_gate() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            pixels.push(((x * 19 + y * 3) & 0xff) as u8);
            pixels.push(((x * 5 + y * 23) & 0xff) as u8);
            pixels.push(((x * 11 + y * 13) & 0xff) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 64,
            height: 64,
            pitch_bytes: 64 * 3,
            output_width: 64,
            output_height: 64,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto HTJ2K host-output encode");
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");

    assert_eq!(decoded.data, pixels);
    assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
    assert!(!encoded.resident.coefficient_prep_used);
    assert!(!encoded.resident.packetization_used);
    assert!(!encoded.resident.codestream_assembly_used);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_padded_private_rgb8_auto_host_encode_routes_away_from_resident_prep() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| u8::try_from((i * 43) & 0xFF).expect("masked pixel fits u8"))
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width: 8,
            height: 8,
            pitch_bytes: 8 * 3,
            output_width: 8,
            output_height: 8,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto host-output encode should avoid resident prep and still succeed");

    assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
    assert!(!encoded.resident.coefficient_prep_used);
    assert!(!encoded.resident.packetization_used);
    assert!(!encoded.resident.codestream_assembly_used);
    assert_eq!(
        encoded.encoded.dispatch_report,
        J2kEncodeDispatchReport::default()
    );
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_decoded_bytes_match(&decoded.data, &pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_resident_host_output_policy_requires_batched_work() {
    assert!(
        !super::super::should_try_auto_resident_lossless_host_format(
            PixelFormat::Rgb8,
            ReversibleTransform::Rct53,
            1,
            64,
            64,
        )
    );
    assert!(
        !super::super::should_try_auto_resident_lossless_host_format(
            PixelFormat::Rgb8,
            ReversibleTransform::Rct53,
            1,
            512,
            512,
        )
    );
    assert!(
        !super::super::should_try_auto_resident_lossless_host_format(
            PixelFormat::Rgb8,
            ReversibleTransform::Rct53,
            1,
            1024,
            1024,
        )
    );
    assert!(super::super::should_try_auto_resident_lossless_host_format(
        PixelFormat::Rgb8,
        ReversibleTransform::Rct53,
        2,
        1024,
        1024,
    ));
    assert!(super::super::should_try_auto_resident_lossless_host_format(
        PixelFormat::Gray8,
        ReversibleTransform::None53,
        2,
        512,
        512,
    ));
    assert!(
        !super::super::should_try_auto_resident_lossless_host_format(
            PixelFormat::Gray8,
            ReversibleTransform::None53,
            1,
            512,
            512,
        )
    );
    assert!(
        !super::super::should_try_auto_resident_lossless_host_format(
            PixelFormat::Rgb8,
            ReversibleTransform::Rct53,
            1,
            768,
            512,
        )
    );
    assert!(
        !super::super::should_try_auto_resident_lossless_host_format(
            PixelFormat::Rgb8,
            ReversibleTransform::Rct53,
            2,
            768,
            512,
        )
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn auto_htj2k_padded_private_rgb8_single_host_output_stays_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 512u32;
    let height = 512u32;
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 3 + y * 5) & 0xff) as u8);
            pixels.push(((x * 7 + y * 11) & 0xff) as u8);
            pixels.push(((x * 13 + y * 17) & 0xff) as u8);
        }
    }
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize * 3,
            output_width: width,
            output_height: height,
            format: PixelFormat::Rgb8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto HTJ2K single host-output encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
    assert!(encoded.input_copy_used);
    assert!(!encoded.resident.coefficient_prep_used);
    assert!(!encoded.resident.packetization_used);
    assert!(!encoded.resident.codestream_assembly_used);
    assert_eq!(
        encoded.encoded.dispatch_report,
        J2kEncodeDispatchReport::default()
    );
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "requires Metal runtime; exercised by the fail-closed Metal release lane"]
fn auto_htj2k_padded_private_gray8_single_host_output_stays_cpu() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 512u32;
    let height = 512u32;
    let pixels: Vec<u8> = (0..width * height)
        .map(|index| ((index * 17 + index / 13) & 0xff) as u8)
        .collect();
    let session = crate::MetalBackendSession::system_default().expect("Metal session");
    let buffer = crate::benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("private benchmark input buffer");

    let encoded = super::super::encode_lossless_from_padded_metal_buffer_with_report(
        super::super::MetalLosslessEncodeTile {
            buffer: &buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize,
            output_width: width,
            output_height: height,
            format: PixelFormat::Gray8,
        },
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto HTJ2K single Gray8 host-output encode");

    assert_eq!(encoded.encoded.backend, BackendKind::Cpu);
    assert!(!encoded.resident.coefficient_prep_used);
    assert!(!encoded.resident.packetization_used);
    assert!(!encoded.resident.codestream_assembly_used);
    assert_eq!(
        encoded.encoded.dispatch_report,
        J2kEncodeDispatchReport::default()
    );
    let decoded = Image::new(&encoded.encoded.codestream, &DecodeSettings::default())
        .expect("codestream parses")
        .decode_native()
        .expect("codestream decodes");
    assert_eq!(decoded.data, pixels);
}

#[cfg(target_os = "macos")]
#[test]
fn auto_htj2k_padded_private_gray8_batch_host_output_uses_full_resident_path() {
    if !should_run_metal_runtime() {
        return;
    }

    let width = 512u32;
    let height = 512u32;
    let first: Vec<u8> = (0..width * height)
        .map(|index| ((index * 17 + index / 13) & 0xff) as u8)
        .collect();
    let second: Vec<u8> = (0..width * height)
        .map(|index| ((index * 23 + index / 7 + 5) & 0xff) as u8)
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
            width,
            height,
            pitch_bytes: width as usize,
            output_width: width,
            output_height: height,
            format: PixelFormat::Gray8,
        },
        super::super::MetalLosslessEncodeTile {
            buffer: &second_buffer,
            byte_offset: 0,
            width,
            height,
            pitch_bytes: width as usize,
            output_width: width,
            output_height: height,
            format: PixelFormat::Gray8,
        },
    ];

    let encoded = super::super::encode_lossless_from_padded_metal_buffers_with_report(
        &tiles,
        &lossless_options! {
            backend: EncodeBackendPreference::Auto,
            block_coding_mode: J2kBlockCodingMode::HighThroughput,
            validation: J2kEncodeValidation::External,
        },
        &session,
    )
    .expect("Auto HTJ2K batched Gray8 host-output encode");

    assert_eq!(encoded.len(), 2);
    for (frame, expected) in encoded.iter().zip([first, second]) {
        assert_eq!(frame.encoded.backend, BackendKind::Metal);
        assert!(!frame.input_copy_used);
        assert!(frame.resident.coefficient_prep_used);
        assert!(frame.resident.packetization_used);
        assert!(frame.resident.codestream_assembly_used);
        let decoded = Image::new(&frame.encoded.codestream, &DecodeSettings::default())
            .expect("codestream parses")
            .decode_native()
            .expect("codestream decodes");
        assert_eq!(decoded.data, expected);
    }
}
