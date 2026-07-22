// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

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
fn auto_repeated_grayscale_uses_metal_for_512_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8_sized(512, 512);
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_auto_to_device(PixelFormat::Gray8, 16)
        .expect("auto repeated decode");
    assert_eq!(surfaces.len(), 16);
    assert!(surfaces
        .iter()
        .all(|surface| surface.backend_kind() == BackendKind::Metal));
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
    let buffer_len = usize::try_from(buffer.length()).expect("metal buffer length fits usize");
    assert!(buffer_len >= metal_surface.byte_len());

    let mut cpu_decoder = J2kDecoder::new(&bytes).expect("cpu decoder");
    let cpu_surface = cpu_decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cpu)
        .expect("cpu surface");
    assert!(completed_surface_metal_buffer(&cpu_surface).is_none());
}

#[test]
fn metal_encoded_raw_parts_validate_ranges_and_support_consuming_handoff() {
    use metal::foreign_types::ForeignType;

    if !should_run_metal_runtime() {
        return;
    }

    let Some(device) = metal::Device::system_default() else {
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
    let expected_ptr = buffer.as_ptr();
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
    assert_eq!(handed_off.as_ptr(), expected_ptr);
}

#[cfg(target_os = "macos")]
#[test]
fn decode_to_device_with_session_uses_session_device() {
    use metal::foreign_types::ForeignTypeRef;

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
    assert_eq!(buffer.device().as_ptr(), session.device().as_ptr());
}

#[cfg(target_os = "macos")]
#[test]
fn decode_scaled_to_device_with_session_supports_rgb8_resident_surface() {
    use metal::foreign_types::ForeignTypeRef;

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
    assert_eq!(buffer.device().as_ptr(), session.device().as_ptr());
}

#[cfg(target_os = "macos")]
#[test]
fn explicit_cpu_staged_metal_api_uses_session_device_and_marks_residency() {
    use metal::foreign_types::ForeignTypeRef;

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
    assert_eq!(buffer.device().as_ptr(), session.device().as_ptr());
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
