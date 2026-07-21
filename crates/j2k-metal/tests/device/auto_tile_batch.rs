// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn submitted_auto_region_scaled_grayscale_keeps_short_batch_on_cpu() {
    let bytes = fixture_gray8_sized(512, 512);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();
    let submissions = (0..16)
        .map(|_| {
            submit_tile_region_scaled_to_device!(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("submit auto region-scaled tile")
        })
        .collect::<Vec<_>>();

    for submission in submissions {
        let surface = submission.wait().expect("auto region-scaled surface");
        assert_eq!(surface.backend_kind(), BackendKind::Cpu);
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "short auto ROI+scaled grayscale tile batches should use one CPU batch fallback"
    );
}

#[test]
fn submitted_auto_region_scaled_rgb_tiles_flush_as_one_cpu_batch() {
    let bytes = fixture_rgb8();
    let roi = Rect {
        x: 0,
        y: 0,
        w: 1,
        h: 1,
    };
    let scale = Downscale::None;
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();
    let submissions = (0..3)
        .map(|_| {
            submit_tile_region_scaled_to_device!(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Rgb8,
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("submit auto RGB region-scaled tile")
        })
        .collect::<Vec<_>>();

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 3];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            3,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host region-scaled decode");

    for submission in submissions {
        let surface = submission.wait().expect("auto RGB region-scaled surface");
        assert_eq!(surface.backend_kind(), BackendKind::Cpu);
        assert_eq!(surface.residency(), SurfaceResidency::Host);
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "auto RGB ROI+scaled tile batches should flush through one CPU batch fallback"
    );
}

#[test]
fn submitted_auto_region_scaled_grayscale_batch64_uses_one_metal_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8_sized(512, 512);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 256,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();
    let submissions = (0..64)
        .map(|_| {
            submit_tile_region_scaled_to_device!(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("submit auto region-scaled tile")
        })
        .collect::<Vec<_>>();

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let stride = scaled.w as usize;
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host region-scaled decode");

    for submission in submissions {
        let surface = submission.wait().expect("auto region-scaled surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "large auto ROI+scaled grayscale tile batches should use one Metal batch"
    );
}

#[test]
fn submitted_auto_region_scaled_ht_grayscale_1024_batch16_uses_one_metal_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_ht_gray8_sized(1024, 1024);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 512,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();
    let submissions = (0..16)
        .map(|_| {
            submit_tile_region_scaled_to_device!(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("submit auto region-scaled HT tile")
        })
        .collect::<Vec<_>>();

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let stride = scaled.w as usize;
    let mut host = vec![0u8; stride * scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Gray8,
            roi,
            scale,
        )
        .expect("host region-scaled decode");

    for submission in submissions {
        let surface = submission.wait().expect("auto region-scaled surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "1024-class auto HT ROI+scaled grayscale tile batches should use one Metal batch"
    );
}

#[test]
fn submitted_auto_region_scaled_rgb_1024_batch16_uses_hybrid_metal() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_rgb8_sized(1024, 1024);
    let roi = Rect {
        x: 128,
        y: 128,
        w: 512,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();
    let submissions = (0..16)
        .map(|_| {
            submit_tile_region_scaled_to_device!(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Rgb8,
                roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("submit auto region-scaled RGB tile")
        })
        .collect::<Vec<_>>();

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
        .expect("host region-scaled RGB decode");

    for submission in submissions {
        let surface = submission.wait().expect("auto region-scaled RGB surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "1024-class auto ROI+scaled RGB tile batches should use one resident hybrid Metal batch"
    );
}

#[test]
fn submitted_auto_region_scaled_ht_grayscale_batch16_is_not_order_dependent() {
    if !should_run_metal_runtime() {
        return;
    }

    let small_bytes = fixture_ht_gray8_sized(64, 64);
    let large_bytes = fixture_ht_gray8_sized(1024, 1024);
    let small_roi = Rect {
        x: 8,
        y: 8,
        w: 32,
        h: 32,
    };
    let large_roi = Rect {
        x: 128,
        y: 128,
        w: 512,
        h: 256,
    };
    let scale = Downscale::Quarter;
    let large_scaled = large_roi.scaled_covering(scale);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let mut submissions = Vec::with_capacity(17);
    submissions.push(
        submit_tile_region_scaled_to_device!(
            &mut ctx,
            &mut session,
            &mut pool,
            &small_bytes,
            PixelFormat::Gray8,
            small_roi,
            scale,
            BackendRequest::Auto,
        )
        .expect("submit small leading auto region-scaled tile"),
    );
    for _ in 0..16 {
        submissions.push(
            submit_tile_region_scaled_to_device!(
                &mut ctx,
                &mut session,
                &mut pool,
                &large_bytes,
                PixelFormat::Gray8,
                large_roi,
                scale,
                BackendRequest::Auto,
            )
            .expect("submit large auto region-scaled tile"),
        );
    }

    let mut host_decoder = J2kDecoder::new(&large_bytes).expect("host decoder");
    let stride = large_scaled.w as usize;
    let mut host = vec![0u8; stride * large_scaled.h as usize];
    host_decoder
        .decode_region_scaled_into(
            &mut J2kScratchPool::new(),
            &mut host,
            stride,
            PixelFormat::Gray8,
            large_roi,
            scale,
        )
        .expect("host region-scaled decode");

    let mut surfaces = Vec::with_capacity(submissions.len());
    for submission in submissions {
        surfaces.push(submission.wait().expect("auto region-scaled surface"));
    }
    assert_eq!(
        surfaces[1].backend_kind(),
        BackendKind::Metal,
        "large 1024-class tiles should not be routed to CPU just because a small tile was submitted first"
    );
    assert_eq!(surfaces[1].dimensions(), (large_scaled.w, large_scaled.h));
    assert_eq!(
        surfaces[1].as_bytes().expect("surface byte access"),
        host.as_slice()
    );
    assert_eq!(
        session.submissions().expect("session submissions"),
        2,
        "auto ROI+scaled should use one Metal batch for the sixteen qualifying 1024-class tiles and leave the leading small tile on CPU"
    );
}
