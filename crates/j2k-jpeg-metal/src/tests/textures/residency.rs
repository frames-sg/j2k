// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
fn jpeg_device_decode_uses_private_internal_planes() {
    let Some(session) = metal_session() else {
        return;
    };
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    compute::reset_jpeg_private_buffer_allocations_for_test();
    let surface = decoder
        .decode_to_device_with_session(PixelFormat::Rgb8, &session)
        .expect("resident JPEG Metal decode");
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert!(
        compute::jpeg_private_buffer_allocations_for_test() > 0,
        "resident JPEG Metal decode should use Private internal planes"
    );
    let _ = surface.as_bytes().expect("surface byte access");
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_private_rgb8_tile_uses_private_output_buffer() {
    let Some(session) = metal_session() else {
        return;
    };
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let tile = decoder
        .decode_private_rgb8_tile_with_session(&session)
        .expect("resident private JPEG Metal decode");

    assert_eq!(tile.dimensions(), (16, 16));
    assert_eq!(tile.pixel_format(), PixelFormat::Rgb8);
    assert_eq!(tile.pitch_bytes(), 16 * PixelFormat::Rgb8.bytes_per_pixel());
    assert_eq!(tile.byte_offset(), 0);
    // SAFETY: The private decode waited for completion and no command accesses
    // the raw buffer while this test inspects its storage metadata.
    let raw_buffer = unsafe { tile.buffer() };
    assert_eq!(raw_buffer.storage_mode(), metal::MTLStorageMode::Private);
    assert!(tile.status_buffer_trusted().length() > 0);

    let handed_off = tile.clone().into_buffer();
    assert_eq!(handed_off.storage_mode(), metal::MTLStorageMode::Private);
    assert_eq!(tile.dimensions(), (16, 16));
}

#[cfg(target_os = "macos")]
#[test]
fn jpeg_gray_region_decode_uses_private_internal_planes() {
    if !should_run_metal_runtime() {
        return;
    }

    let roi = Rect {
        x: 4,
        y: 4,
        w: 8,
        h: 8,
    };
    let mut expected_decoder = Decoder::new(BASELINE_420).expect("expected decoder");
    let mut expected = vec![0; roi.w as usize * roi.h as usize];
    expected_decoder
        .decode_region_into(
            &mut CpuScratchPool::new(),
            &mut expected,
            roi.w as usize,
            PixelFormat::Gray8,
            roi,
        )
        .expect("expected CPU region decode");

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    compute::reset_jpeg_private_buffer_allocations_for_test();
    let surface = decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
        .expect("resident JPEG Metal region decode");
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert!(
        compute::jpeg_private_buffer_allocations_for_test() >= 3,
        "resident Gray8 region decode should keep decoded Y/Cb/Cr planes Private"
    );
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        expected.as_slice()
    );
}

#[cfg(target_os = "macos")]
#[test]
fn uploaded_metal_surface_is_marked_cpu_staged() {
    if !should_run_metal_runtime() {
        return;
    }

    let surface = upload_surface(
        vec![1, 2, 3],
        (1, 1),
        PixelFormat::Rgb8,
        BackendRequest::Metal,
    )
    .expect("CPU staged Metal upload");

    assert_eq!(surface.residency(), SurfaceResidency::CpuStagedMetalUpload);
}

#[test]
fn auto_route_prefers_cpu_host_for_region_scaled_even_with_restart_packets() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
    let packet = build_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

    assert_eq!(
        choose_route(
            &decoder,
            BackendRequest::Auto,
            PixelFormat::Rgb8,
            batch::BatchOp::RegionScaled {
                roi: Rect {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 16,
                },
                scale: Downscale::Quarter,
            },
            test_fast_packets(None, None, Some(&packet)),
        ),
        routing::RouteDecision::CpuHost
    );
}

#[cfg(not(target_os = "macos"))]
#[test]
fn session_decode_rejects_unsupported_shape_before_host_unavailability() {
    let mut decoder = Decoder::new(GRAYSCALE).expect("decoder");
    let session = MetalBackendSession::default();

    assert!(matches!(
        decoder.decode_to_device_with_session(PixelFormat::Gray8, &session),
        Err(Error::UnsupportedMetalRequest { .. })
    ));
}
