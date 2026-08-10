// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use objc2_metal::MTLResource as _;

#[test]
fn full_classic_grayscale_decode_to_metal_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn full_classic_signed_gray4_decode_to_metal_matches_host_exactly() {
    if !should_run_metal_runtime() {
        return;
    }

    let (bytes, expected) = fixture_classic_signed_gray4();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::from(bytes))])
        .expect("decode signed classic gray4 batch");
    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );
    let surface = &batch.groups()[0].surfaces()[0];
    let actual = surface
        .as_bytes()
        .expect("signed classic gray4 surface bytes")
        .chunks_exact(2)
        .map(|sample| i16::from_ne_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn full_classic_signed_gray4_roi_decode_to_metal_matches_host_exactly() {
    if !should_run_metal_runtime() {
        return;
    }

    let (bytes, expected) = fixture_classic_signed_gray4_roi();
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::from(bytes))])
        .expect("decode signed classic gray4 ROI batch");
    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );
    let surface = &batch.groups()[0].surfaces()[0];
    let actual = surface
        .as_bytes()
        .expect("signed classic gray4 ROI surface bytes")
        .chunks_exact(2)
        .map(|sample| i16::from_ne_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn full_htj2k_decode_to_metal_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn full_irreversible_htj2k_decode_to_metal_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(32, 32, 1);
    let bytes = encode_htj2k(
        &pixels,
        32,
        32,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible HTJ2K gray8");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; 32 * 32];
    host_decoder
        .decode_into(&mut host, 32, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.as_bytes().expect("surface byte access"), host);
}

#[test]
fn completed_htj2k_batch_exposes_observed_metal_stages() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::from(fixture_ht_gray8()))])
        .expect("decode HTJ2K batch");
    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );

    let report = batch.groups()[0].dispatch_report();
    assert!(report.tier1 > 0);
    assert!(report.ht_tier1 > 0);
    assert_eq!(report.ht_refinement, 0);
    assert_eq!(report.classic_tier1, 0);
    assert!(report.dequantization > 0);
    assert!(report.idwt > 0);
    assert_eq!(report.mct, 0);
    assert!(report.color_output > 0);
    assert!(report.host_to_device > 0);
}

#[test]
fn completed_refinement_batch_exposes_observed_metal_refinement() {
    if !should_run_metal_runtime() {
        return;
    }

    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::from(
            j2k_test_support::openhtj2k_refinement_fixture(),
        ))])
        .expect("decode HTJ2K refinement batch");
    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );

    let report = batch.groups()[0].dispatch_report();
    assert!(report.ht_tier1 > 0);
    assert!(report.ht_refinement > 0);
    assert_eq!(report.classic_tier1, 0);
}

#[test]
fn htj2k_direct_decode_clears_reused_classic_scratch_buffers() {
    if !should_run_metal_runtime() {
        return;
    }

    let classic_bytes = fixture_gray8();
    let mut classic_decoder = J2kDecoder::new(&classic_bytes).expect("classic decoder");
    let classic_surface = classic_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("classic device decode");
    assert_eq!(classic_surface.backend_kind(), BackendKind::Metal);

    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn full_irreversible_j2k_decode_to_metal_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8_irreversible();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn full_irreversible_rgb_j2k_decode_to_metal_matches_host_decode_exactly() {
    if !should_run_metal_runtime() {
        return;
    }

    let pixels = j2k_test_support::gradient_u8(16, 16, 3);
    let bytes = encode(
        &pixels,
        16,
        16,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible RGB8");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; 16 * 16 * 3];
    host_decoder
        .decode_into(&mut host, 16 * 3, PixelFormat::Rgb8)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
        .expect("device decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.as_bytes().expect("surface byte access"), host);
}

#[test]
fn openjpeg_irreversible_rgb_roi_decode_to_metal_matches_cpu_exactly() {
    if !should_run_metal_runtime() {
        return;
    }

    let codestream = j2k_test_support::OPENJPEG_IRREVERSIBLE_RGB8_8X8;
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let mut decoder = J2kDecoder::new(codestream).expect("decoder");
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut cpu_decoder = J2kDecoder::new(codestream).expect("CPU decoder");
    let cpu = cpu_decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::region(PixelFormat::Rgb8, roi, BackendRequest::Cpu),
            &session,
        )
        .expect("CPU surface decode");
    let mut full_decoder = J2kDecoder::new(codestream).expect("full CPU decoder");
    let full = full_decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Cpu),
            &session,
        )
        .expect("full CPU surface decode");
    let full = full.as_bytes().expect("full CPU surface byte access");
    let mut cropped = Vec::with_capacity(4 * 4 * 3);
    for y in roi.y..roi.y + roi.h {
        let start = (y as usize * 8 + roi.x as usize) * 3;
        cropped.extend_from_slice(&full[start..start + roi.w as usize * 3]);
    }
    assert_eq!(cpu.as_bytes().expect("CPU surface byte access"), cropped);
    let surface = decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::region(PixelFormat::Rgb8, roi, BackendRequest::Metal),
            &session,
        )
        .expect("device decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        cpu.as_bytes().expect("CPU surface byte access")
    );
}

#[test]
fn auto_full_grayscale_prefers_cpu_for_small_classic_fixture() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("auto decode");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
}

#[test]
fn auto_full_htj2k_prefers_cpu_for_small_fixture() {
    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Auto)
        .expect("auto decode");
    assert_eq!(surface.backend_kind(), BackendKind::Cpu);
}

#[test]
fn auto_repeated_grayscale_keeps_short_512_batch_on_cpu() {
    let bytes = fixture_gray8_sized(512, 512);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_auto_to_device(PixelFormat::Gray8, 8)
        .expect("auto repeated decode");
    assert_eq!(surfaces.len(), 8);
    assert!(surfaces
        .iter()
        .all(|surface| surface.backend_kind() == BackendKind::Cpu));
}

#[test]
fn auto_repeated_grayscale_keeps_unqualified_512_batch_on_cpu() {
    let bytes = fixture_gray8_sized(512, 512);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_auto_to_device(PixelFormat::Gray8, 16)
        .expect("auto repeated decode");
    assert_eq!(surfaces.len(), 16);
    assert!(surfaces
        .iter()
        .all(|surface| surface.backend_kind() == BackendKind::Cpu));
}

#[test]
fn tile_full_grayscale_device_path_uses_metal_direct() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let mut ctx = J2kContext::default();
    let mut pool = J2kScratchPool::new();
    let surface = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("tile surface");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (4, 4));
}

#[test]
fn metal_surface_exposes_buffer_for_on_device_consumers() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let mut metal_decoder = J2kDecoder::new(&bytes).expect("metal decoder");
    let metal_surface = metal_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("metal surface");
    let (buffer, byte_offset) =
        completed_surface_metal_buffer(&metal_surface).expect("metal buffer");
    assert_eq!(byte_offset, 0);
    let buffer_len = buffer.length();
    assert!(buffer_len >= metal_surface.byte_len());

    let mut cpu_decoder = J2kDecoder::new(&bytes).expect("cpu decoder");
    let cpu_surface = cpu_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cpu)
        .expect("cpu surface");
    assert!(completed_surface_metal_buffer(&cpu_surface).is_none());
}

#[test]
fn metal_encoded_raw_parts_validate_ranges_and_support_consuming_handoff() {
    if !should_run_metal_runtime() {
        return;
    }

    let Ok(device) = j2k_metal_support::system_default_device() else {
        j2k_test_support::metal_device_unavailable_is_skip(module_path!());
        return;
    };
    let invalid_buffer =
        j2k_metal_support::checked_shared_buffer(&device, 64).expect("test buffer allocation");
    // SAFETY: This fresh allocation has no prior or concurrent writers and is
    // retained only for this constructor call.
    let invalid = unsafe {
        j2k_metal::MetalEncodedJ2k::from_raw_parts(invalid_buffer, 16..32, 64, (4, 4), 1, 8, false)
    };
    assert!(matches!(
        invalid,
        Err(Error::MetalKernel { message }) if message.contains("exceeds allocation length")
    ));

    let buffer =
        j2k_metal_support::checked_shared_buffer(&device, 64).expect("test buffer allocation");
    let expected_ptr = Retained::as_ptr(&buffer);
    // SAFETY: This fresh allocation has no writers and stays immutable until
    // the encoded object is consumed below.
    let encoded = unsafe {
        j2k_metal::MetalEncodedJ2k::from_raw_parts(buffer, 8..24, 32, (4, 4), 1, 8, false)
    }
    .expect("valid raw Metal codestream parts");
    assert_eq!(encoded.byte_offset(), 8);
    assert_eq!(encoded.byte_len(), 16);
    assert_eq!(encoded.capacity(), 32);
    assert_eq!(encoded.dimensions(), (4, 4));
    assert_eq!(encoded.components(), 1);
    assert_eq!(encoded.bit_depth(), 8);
    assert!(!encoded.is_signed());
    // SAFETY: This encoded descriptor is the allocation's only owner and no
    // sibling descriptor or cloned handle exists.
    let handed_off = unsafe { encoded.into_codestream_buffer() };
    assert_eq!(Retained::as_ptr(&handed_off), expected_ptr);
}

#[cfg(target_os = "macos")]
#[test]
fn decode_to_device_with_session_uses_session_device() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut decoder = J2kDecoder::new(&bytes).expect("metal decoder");

    let surface = decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Metal),
            &session,
        )
        .expect("session decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    let (buffer, _) = completed_surface_metal_buffer(&surface).expect("metal buffer");
    let buffer_device = buffer.device();
    assert!(ptr::eq(buffer_device.as_ref(), session.device()));
}

#[cfg(target_os = "macos")]
#[test]
fn decode_scaled_to_device_with_session_supports_rgb8_resident_surface() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_rgb8_sized(8, 8);
    let scale = Downscale::Half;
    let scaled = Rect {
        x: 0,
        y: 0,
        w: 8,
        h: 8,
    }
    .scaled_covering(scale);
    let session = MetalBackendSession::system_default().expect("Metal backend session");

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut pool = J2kScratchPool::new();
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_scaled_into(&mut pool, &mut host, stride, PixelFormat::Rgb8, scale)
        .expect("host scaled RGB8 decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("metal decoder");
    let surface = decoder
        .decode_request_to_device_with_session(
            MetalDecodeRequest::scaled(PixelFormat::Rgb8, scale, BackendRequest::Metal),
            &session,
        )
        .expect("session scaled RGB8 decode");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
    let (buffer, _) = completed_surface_metal_buffer(&surface).expect("metal buffer");
    let buffer_device = buffer.device();
    assert!(ptr::eq(buffer_device.as_ref(), session.device()));
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_cpu_staged_metal_api_uses_session_device_and_marks_residency() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_rgb8();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");

    let surface = decoder
        .decode_request_to_cpu_staged_metal_surface_with_session(
            MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Metal),
            &session,
        )
        .expect("CPU-staged Metal surface");

    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::CpuStagedMetalUpload);
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
    let (buffer, byte_offset) = completed_surface_metal_buffer(&surface).expect("Metal buffer");
    assert_eq!(byte_offset, 0);
    let buffer_device = buffer.device();
    assert!(ptr::eq(buffer_device.as_ref(), session.device()));
}

#[cfg(target_os = "macos")]
#[test]
fn decode_to_device_with_session_unsupported_rgba16_is_rejected() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_rgb12();
    let session = MetalBackendSession::system_default().expect("Metal backend session");
    let mut decoder = J2kDecoder::new(&bytes).expect("metal decoder");

    let result = decoder.decode_request_to_device_with_session(
        MetalDecodeRequest::full(PixelFormat::Rgba16, BackendRequest::Metal),
        &session,
    );

    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert_eq!(reason, UNSUPPORTED_RGBA16_REASON);
        }
        Err(other) => panic!("unexpected explicit Metal session error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal session request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}
