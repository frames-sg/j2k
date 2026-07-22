// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn repeated_classic_grayscale_direct_decode_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_direct_to_device(PixelFormat::Gray8, 3)
        .expect("repeated direct decode");
    assert_eq!(surfaces.len(), 3);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    for surface in surfaces {
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
}

#[test]
fn repeated_ht_grayscale_direct_decode_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surfaces = decoder
        .decode_repeated_grayscale_direct_to_device(PixelFormat::Gray8, 3)
        .expect("repeated direct decode");
    assert_eq!(surfaces.len(), 3);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    for surface in surfaces {
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
}

#[test]
fn metal_gray16_matches_host_decode_for_12bit_source() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 8];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray16)
        .expect("host decode");

    let surface = decoder
        .decode_to_device(PixelFormat::Gray16, BackendRequest::Metal)
        .expect("device decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn explicit_metal_rgb_full_tile_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let rgb8 = fixture_rgb8();
    {
        let mut decoder = J2kDecoder::new(&rgb8).expect("rgb8 decoder");
        let mut host_decoder = J2kDecoder::new(&rgb8).expect("rgb8 host decoder");
        let mut host = [0u8; 12];
        host_decoder
            .decode_into(&mut host, 6, PixelFormat::Rgb8)
            .expect("host rgb8 decode");
        let surface = decoder
            .decode_to_device(PixelFormat::Rgb8, BackendRequest::Metal)
            .expect("explicit Metal rgb8 decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 2));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }

    {
        let mut decoder = J2kDecoder::new(&rgb8).expect("rgba8 decoder");
        let mut host_decoder = J2kDecoder::new(&rgb8).expect("rgba8 host decoder");
        let mut host = [0u8; 16];
        host_decoder
            .decode_into(&mut host, 8, PixelFormat::Rgba8)
            .expect("host rgba8 decode");
        let surface = decoder
            .decode_to_device(PixelFormat::Rgba8, BackendRequest::Metal)
            .expect("explicit Metal rgba8 decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 2));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }

    let rgb12 = fixture_rgb12();
    {
        let mut decoder = J2kDecoder::new(&rgb12).expect("rgb12 decoder");
        let mut host_decoder = J2kDecoder::new(&rgb12).expect("rgb12 host decoder");
        let mut host = [0u8; 12];
        host_decoder
            .decode_into(&mut host, 12, PixelFormat::Rgb16)
            .expect("host rgb16 decode");
        let surface = decoder
            .decode_to_device(PixelFormat::Rgb16, BackendRequest::Metal)
            .expect("explicit Metal rgb16 decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (2, 1));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
}

#[test]
fn explicit_metal_unsupported_rgba16_full_decode_is_rejected() {
    let bytes = fixture_rgb12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let result = decoder.decode_to_device(PixelFormat::Rgba16, BackendRequest::Metal);

    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert_eq!(reason, UNSUPPORTED_RGBA16_REASON);
        }
        Err(other) => panic!("unexpected explicit Metal error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn explicit_metal_unsupported_rgba16_error_is_codec_unsupported() {
    let bytes = fixture_rgb12();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let err = match decoder.decode_to_device(PixelFormat::Rgba16, BackendRequest::Metal) {
        Err(err) => err,
        Ok(surface) => panic!(
            "explicit Metal must not silently fall back; got {:?}",
            surface.backend_kind()
        ),
    };

    assert!(err.is_unsupported());
}

#[test]
fn auto_decode_report_explains_cpu_fallback_and_residency() {
    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let reported = decoder
        .decode_request_to_device_with_report(MetalDecodeRequest::full(
            PixelFormat::Gray8,
            BackendRequest::Auto,
        ))
        .expect("reported Auto decode");

    assert_eq!(reported.surface.backend_kind(), BackendKind::Cpu);
    assert_eq!(reported.surface.residency(), SurfaceResidency::Host);
    assert!(completed_surface_metal_buffer(&reported.surface).is_none());
    assert_eq!(reported.report.operation, DecodeOperation::Full);
    assert_eq!(reported.report.requested_backend, BackendRequest::Auto);
    assert_eq!(reported.report.selected_backend, BackendKind::Cpu);
    assert_eq!(reported.report.pixel_format, PixelFormat::Gray8);
    assert_eq!(reported.report.surface_residency, SurfaceResidency::Host);
    assert_eq!(
        reported.report.fallback_reason,
        Some(AUTO_DECODE_CPU_FALLBACK_REASON)
    );
}

#[test]
fn explicit_metal_decode_report_records_resident_surface() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");

    let reported = decoder
        .decode_request_to_device_with_report(MetalDecodeRequest::full(
            PixelFormat::Gray8,
            BackendRequest::Metal,
        ))
        .expect("reported explicit Metal decode");

    assert_eq!(reported.surface.backend_kind(), BackendKind::Metal);
    assert_eq!(
        reported.surface.residency(),
        SurfaceResidency::MetalResidentDecode
    );
    assert_eq!(reported.report.operation, DecodeOperation::Full);
    assert_eq!(reported.report.requested_backend, BackendRequest::Metal);
    assert_eq!(reported.report.selected_backend, BackendKind::Metal);
    assert_eq!(
        reported.report.surface_residency,
        SurfaceResidency::MetalResidentDecode
    );
    assert_eq!(reported.report.fallback_reason, None);
}

#[test]
fn explicit_metal_unsupported_rgba16_report_variants_are_rejected() {
    let bytes = fixture_rgb12();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 1,
    };
    let scale = Downscale::Half;

    let mut decoder = J2kDecoder::new(&bytes).expect("full decoder");
    assert_unsupported_rgba16_report(decoder.decode_request_to_device_with_report(
        MetalDecodeRequest::full(PixelFormat::Rgba16, BackendRequest::Metal),
    ));

    let mut decoder = J2kDecoder::new(&bytes).expect("region decoder");
    assert_unsupported_rgba16_report(decoder.decode_request_to_device_with_report(
        MetalDecodeRequest::region(PixelFormat::Rgba16, roi, BackendRequest::Metal),
    ));

    let mut decoder = J2kDecoder::new(&bytes).expect("scaled decoder");
    assert_unsupported_rgba16_report(decoder.decode_request_to_device_with_report(
        MetalDecodeRequest::scaled(PixelFormat::Rgba16, scale, BackendRequest::Metal),
    ));

    let mut decoder = J2kDecoder::new(&bytes).expect("region scaled decoder");
    assert_unsupported_rgba16_report(decoder.decode_request_to_device_with_report(
        MetalDecodeRequest::region_scaled(PixelFormat::Rgba16, roi, scale, BackendRequest::Metal),
    ));
}
