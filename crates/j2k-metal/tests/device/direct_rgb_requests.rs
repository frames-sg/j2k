// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn explicit_metal_region_scaled_rgb_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_direct_rgb8_variant(3);
    let roi = Rect {
        x: 1,
        y: 2,
        w: 5,
        h: 4,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host region scaled RGB decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Metal)
        .expect("explicit Metal region scaled RGB decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );

    let mut host_decoder = J2kDecoder::new(&bytes).expect("rgba8 host decoder");
    let stride = scaled.w as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Rgba8,
            roi,
            scale,
        )
        .expect("host region scaled RGBA decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("rgba8 decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgba8, roi, scale, BackendRequest::Metal)
        .expect("explicit Metal region scaled RGBA decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );

    let bytes = fixture_rgb12();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 1,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut host_decoder = J2kDecoder::new(&bytes).expect("rgb16 host decoder");
    let stride = scaled.w as usize * PixelFormat::Rgb16.bytes_per_pixel();
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Rgb16,
            roi,
            scale,
        )
        .expect("host region scaled RGB16 decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("rgb16 decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb16, roi, scale, BackendRequest::Metal)
        .expect("explicit Metal region scaled RGB16 decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn explicit_metal_region_scaled_rgb_large_cropped_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_rgb8_sized(1024, 1024);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 512,
        h: 512,
    };

    for scale in [Downscale::Half, Downscale::None] {
        let scaled = roi.scaled_covering(scale);
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
            let stride = scaled.w as usize * fmt.bytes_per_pixel();
            let mut host = vec![0u8; stride * scaled.h as usize];
            host_decoder
                .decode_region_scaled_into(
                    &mut J2kScratchPool::new(),
                    &mut host,
                    stride,
                    fmt,
                    roi,
                    scale,
                )
                .expect("host region scaled RGB decode");

            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            let surface = decoder
                .decode_region_scaled_to_device(fmt, roi, scale, BackendRequest::Metal)
                .expect("explicit Metal region scaled RGB decode");
            assert_eq!(surface.backend_kind(), BackendKind::Metal);
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
            let surface_bytes = surface.as_bytes().expect("surface byte access");
            if surface_bytes.as_ref() != host.as_slice() {
                let mismatch = surface_bytes
                    .iter()
                    .zip(&host)
                    .position(|(actual, expected)| actual != expected)
                    .expect("mismatched buffers should have a differing byte");
                panic!(
                    "fmt={fmt:?} scale={scale:?} first mismatch at byte {mismatch}: metal={} host={}",
                    surface_bytes[mismatch],
                    host[mismatch]
                );
            }
        }
    }
}

#[test]
fn auto_region_and_scaled_fallback_to_cpu_surface_and_match_host_decode() {
    let bytes = fixture_rgb8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto)
        .expect("region surface");
    assert_eq!(region_surface.backend_kind(), BackendKind::Cpu);
    assert!(completed_surface_metal_buffer(&region_surface).is_none());

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut region_host = [0u8; 3];
    host_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut region_host,
            3,
            PixelFormat::Rgb8,
            roi,
        )
        .expect("host region");
    assert_eq!(
        region_surface.as_bytes().expect("surface byte access"),
        region_host.as_slice()
    );

    let scaled_surface = decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Auto)
        .expect("scaled surface");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Cpu);
    assert!(completed_surface_metal_buffer(&scaled_surface).is_none());

    let mut scaled_host = [0u8; 3];
    host_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut scaled_host,
            3,
            PixelFormat::Rgb8,
            Downscale::Half,
        )
        .expect("host scaled");
    assert_eq!(
        scaled_surface.as_bytes().expect("surface byte access"),
        scaled_host.as_slice()
    );
}

#[test]
fn invalid_region_reports_error_instead_of_panicking() {
    let bytes = fixture_rgb8();
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let roi = Rect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    match decoder.decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Auto) {
        Err(Error::Decode(j2k::J2kError::InvalidRegion { .. })) => {}
        Err(other) => panic!("unexpected error for invalid ROI: {other:?}"),
        Ok(_) => panic!("invalid ROI should fail"),
    }
}

#[test]
fn explicit_metal_tile_unsupported_rgba16_is_rejected() {
    let bytes = fixture_rgb12();
    let mut ctx = J2kContext::default();
    let mut pool = J2kScratchPool::new();

    let result = Codec::decode_tile_to_device(
        &mut ctx,
        &mut pool,
        &bytes,
        PixelFormat::Rgba16,
        BackendRequest::Metal,
    );

    match result {
        Err(Error::UnsupportedMetalRequest { reason }) => {
            assert_eq!(reason, UNSUPPORTED_RGBA16_REASON);
        }
        Err(other) => panic!("unexpected explicit Metal tile error: {other:?}"),
        Ok(surface) => panic!(
            "explicit Metal tile request must not fall back; got {:?}",
            surface.backend_kind()
        ),
    }
}

#[test]
fn hybrid_ht_cpuupload_uses_worker_local_decode_workspace() {
    let source = include_str!("../../src/compute/direct_cpu.rs");

    assert!(
        source.contains("decode_prepared_ht_jobs_on_cpu_with_workspace"),
        "HT CPUUpload decode must expose a workspace-aware helper"
    );
    assert!(
        source.contains("HtCodeBlockDecodeWorkspace::default()"),
        "parallel HT CPUUpload decode must initialize worker-local HT decode workspaces"
    );
    assert!(
        source.contains("decode_ht_code_block_scalar_with_workspace"),
        "HT CPUUpload decode must call the scratch-reusing scalar helper"
    );
}
