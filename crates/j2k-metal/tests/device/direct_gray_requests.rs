// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn explicit_metal_region_and_scaled_grayscale_match_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };

    let mut host_region_decoder = J2kDecoder::new(&bytes).expect("host region decoder");
    let mut host_region = [0u8; 4];
    host_region_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut host_region,
            2,
            PixelFormat::Gray8,
            roi,
        )
        .expect("host region decode");

    let mut region_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = region_decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
        .expect("explicit Metal region decode");
    assert_eq!(region_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(region_surface.dimensions(), (2, 2));
    assert_eq!(
        region_surface.as_bytes().expect("surface byte access"),
        host_region.as_slice()
    );

    let mut host_scaled_decoder = J2kDecoder::new(&bytes).expect("host scaled decoder");
    let mut host_scaled = [0u8; 4];
    host_scaled_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host_scaled,
            2,
            PixelFormat::Gray8,
            Downscale::Half,
        )
        .expect("host scaled decode");

    let mut scaled_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let scaled_surface = scaled_decoder
        .decode_scaled_to_device(PixelFormat::Gray8, Downscale::Half, BackendRequest::Metal)
        .expect("explicit Metal scaled decode");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(scaled_surface.dimensions(), (2, 2));
    assert_eq!(
        scaled_surface.as_bytes().expect("surface byte access"),
        host_scaled.as_slice()
    );
}

#[test]
fn explicit_metal_scaled_rgb8_matches_host_decode() {
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

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host scaled decoder");
    let stride = scaled.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Rgb8,
            scale,
        )
        .expect("host scaled RGB8 decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, scale, BackendRequest::Metal)
        .expect("explicit Metal scaled RGB8 decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn explicit_metal_region_and_scaled_htj2k_grayscale_match_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    };

    let mut host_region_decoder = J2kDecoder::new(&bytes).expect("host region decoder");
    let mut host_region = [0u8; 4];
    host_region_decoder
        .decode_region_into(
            &mut J2kScratchPool::new(),
            &mut host_region,
            2,
            PixelFormat::Gray8,
            roi,
        )
        .expect("host region decode");

    let mut region_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let region_surface = region_decoder
        .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
        .expect("explicit Metal region decode");
    assert_eq!(region_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(region_surface.dimensions(), (2, 2));
    assert_eq!(
        region_surface.as_bytes().expect("surface byte access"),
        host_region.as_slice()
    );

    let mut host_scaled_decoder = J2kDecoder::new(&bytes).expect("host scaled decoder");
    let mut host_scaled = [0u8; 4];
    host_scaled_decoder
        .decode_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host_scaled,
            2,
            PixelFormat::Gray8,
            Downscale::Half,
        )
        .expect("host scaled decode");

    let mut scaled_decoder = J2kDecoder::new(&bytes).expect("decoder");
    let scaled_surface = scaled_decoder
        .decode_scaled_to_device(PixelFormat::Gray8, Downscale::Half, BackendRequest::Metal)
        .expect("explicit Metal scaled decode");
    assert_eq!(scaled_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(scaled_surface.dimensions(), (2, 2));
    assert_eq!(
        scaled_surface.as_bytes().expect("surface byte access"),
        host_scaled.as_slice()
    );
}

#[test]
fn explicit_metal_region_scaled_grayscale_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host region scaled decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Metal)
        .expect("explicit Metal region scaled decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn explicit_metal_region_scaled_grayscale_large_cropped_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8_sized(1024, 1024);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 512,
        h: 512,
    };

    for scale in [Downscale::Half, Downscale::None] {
        let scaled = roi.scaled_covering(scale);
        let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
        let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
        host_decoder
            .decode_region_scaled_into(
                &mut J2kScratchPool::new(),
                &mut host,
                scaled.w as usize,
                PixelFormat::Gray8,
                roi,
                scale,
            )
            .expect("host region scaled decode");

        let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
        let surface = decoder
            .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Metal)
            .expect("explicit Metal region scaled decode");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        let surface_bytes = surface.as_bytes().expect("surface byte access");
        if surface_bytes.as_ref() != host.as_slice() {
            let mismatch = surface_bytes
                .iter()
                .zip(&host)
                .position(|(actual, expected)| actual != expected)
                .expect("mismatched buffers should have a differing byte");
            panic!(
                "scale={scale:?} first mismatch at byte {mismatch}: metal={} host={}",
                surface_bytes[mismatch], host[mismatch]
            );
        }
    }
}

#[test]
fn explicit_metal_region_scaled_htj2k_grayscale_matches_host_decode() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_gray8();
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host region scaled decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Metal)
        .expect("explicit Metal region scaled decode");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}

#[test]
fn explicit_metal_region_scaled_htj2k_falls_back_when_direct_width_is_unsupported() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_gray8_unsupported_direct_width();
    let roi = Rect {
        x: 48,
        y: 2,
        w: 96,
        h: 4,
    };
    let scale = Downscale::None;
    let scaled = roi.scaled_covering(scale);

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            scaled.w as usize,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host region scaled decode");

    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Gray8, roi, scale, BackendRequest::Metal)
        .expect("explicit Metal should fall back after unsupported direct HT geometry");
    assert_eq!(surface.backend_kind(), BackendKind::Metal);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        surface.as_bytes().expect("surface byte access"),
        host.as_slice()
    );
}
