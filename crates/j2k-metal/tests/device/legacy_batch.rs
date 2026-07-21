// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn submitted_full_grayscale_tiles_flush_as_one_device_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8();
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = (0..3)
        .map(|_| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                BackendRequest::Metal,
            )
            .expect("submit tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions().expect("session submissions"),
        0,
        "submitted tile surfaces should stay queued until a wait flushes the session"
    );

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 16];
    host_decoder
        .decode_into(&mut host, 4, PixelFormat::Gray8)
        .expect("host decode");

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "compatible queued grayscale tiles should flush through one repeated Metal batch"
    );
}

#[test]
fn submitted_auto_512_grayscale_tiles_flush_as_one_metal_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_gray8_sized(512, 512);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = (0..16)
        .map(|_| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Gray8,
                BackendRequest::Auto,
            )
            .expect("submit auto tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions().expect("session submissions"),
        0,
        "auto submitted tile surfaces should stay queued until a wait flushes the session"
    );

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.dimensions(), (512, 512));
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "compatible auto grayscale tiles should flush through one repeated Metal batch"
    );
}

#[test]
fn submitted_distinct_full_grayscale_tiles_flush_as_one_device_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let classic_bytes = fixture_gray8();
    let reversed_bytes = fixture_gray8_reversed();
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let classic_submission = Codec::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        &classic_bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("submit classic tile");
    let reversed_submission = Codec::submit_tile_to_device(
        &mut ctx,
        &mut session,
        &mut pool,
        &reversed_bytes,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("submit reversed tile");

    assert_eq!(
        session.submissions().expect("session submissions"),
        0,
        "distinct submitted tile surfaces should stay queued until wait"
    );

    let mut classic_host_decoder = J2kDecoder::new(&classic_bytes).expect("classic host decoder");
    let mut classic_host = [0u8; 16];
    classic_host_decoder
        .decode_into(&mut classic_host, 4, PixelFormat::Gray8)
        .expect("classic host decode");

    let mut reversed_host_decoder =
        J2kDecoder::new(&reversed_bytes).expect("reversed host decoder");
    let mut reversed_host = [0u8; 16];
    reversed_host_decoder
        .decode_into(&mut reversed_host, 4, PixelFormat::Gray8)
        .expect("reversed host decode");

    let classic_surface = classic_submission.wait().expect("classic surface");
    let reversed_surface = reversed_submission.wait().expect("reversed surface");
    assert_eq!(classic_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(reversed_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(
        classic_surface.as_bytes().expect("surface byte access"),
        classic_host.as_slice()
    );
    assert_eq!(
        reversed_surface.as_bytes().expect("surface byte access"),
        reversed_host.as_slice()
    );
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "distinct queued grayscale tiles should flush through one Metal command buffer"
    );
}

#[test]
fn submitted_non_stackable_grayscale_tiles_decode_every_input() {
    if !should_run_metal_runtime() {
        return;
    }

    let small = fixture_gray8();
    let large = fixture_gray8_sized(8, 8);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = [&small, &large]
        .into_iter()
        .map(|bytes| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Gray8,
                BackendRequest::Metal,
            )
            .expect("submit non-stackable grayscale tile")
        })
        .collect::<Vec<_>>();
    let expected = [&small, &large]
        .into_iter()
        .map(|bytes| {
            let mut decoder = J2kDecoder::new(bytes).expect("host decoder");
            let (width, height) = decoder.inner().info().dimensions;
            let mut pixels = vec![0_u8; width as usize * height as usize];
            decoder
                .decode_into(&mut pixels, width as usize, PixelFormat::Gray8)
                .expect("host grayscale decode");
            (width, height, pixels)
        })
        .collect::<Vec<_>>();

    for (submission, (width, height, pixels)) in submissions.into_iter().zip(expected) {
        let surface = submission.wait().expect("non-stackable Metal surface");
        assert_eq!(surface.dimensions(), (width, height));
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            pixels.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "non-stackable grayscale inputs should still share one producer command buffer"
    );
}

#[test]
fn submitted_full_rgb_tiles_flush_as_one_device_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let bytes = fixture_direct_rgb8();
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = (0..3)
        .map(|_| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                &bytes,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit rgb tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions().expect("session submissions"),
        0,
        "submitted RGB tile surfaces should stay queued until a wait flushes the session"
    );

    let mut host_decoder = J2kDecoder::new(&bytes).expect("host decoder");
    let mut host = [0u8; 12];
    host_decoder
        .decode_into(&mut host, 6, PixelFormat::Rgb8)
        .expect("host decode");

    for submission in submissions {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
    }
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "compatible queued RGB tiles should flush through one Metal batch"
    );
}

#[test]
fn submitted_distinct_full_rgb_tiles_stay_resident_when_batch_route_falls_back() {
    if !should_run_metal_runtime() {
        return;
    }

    let rgb_tiles = [
        fixture_direct_rgb8_variant(0),
        fixture_direct_rgb8_variant(5),
        fixture_direct_rgb8_variant(11),
    ];
    assert_ne!(rgb_tiles[0], rgb_tiles[1], "RGB batch fixtures must differ");
    assert_ne!(rgb_tiles[1], rgb_tiles[2], "RGB batch fixtures must differ");
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let submissions = rgb_tiles
        .iter()
        .map(|bytes| {
            Codec::submit_tile_to_device(
                &mut ctx,
                &mut session,
                &mut pool,
                bytes,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
            )
            .expect("submit distinct rgb tile")
        })
        .collect::<Vec<_>>();

    assert_eq!(
        session.submissions().expect("session submissions"),
        0,
        "distinct RGB tile surfaces should stay queued until a wait flushes the session"
    );

    let expected = rgb_tiles
        .iter()
        .map(|bytes| {
            let mut host_decoder = J2kDecoder::new(bytes).expect("host decoder");
            let stride = 8 * 3;
            let mut host = vec![0u8; stride * 8];
            host_decoder
                .decode_into(&mut host, stride, PixelFormat::Rgb8)
                .expect("host decode");
            host
        })
        .collect::<Vec<_>>();

    let mut surfaces = Vec::with_capacity(submissions.len());
    for (submission, host) in submissions.into_iter().zip(expected) {
        let surface = submission.wait().expect("surface");
        assert_eq!(surface.backend_kind(), BackendKind::Metal);
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(
            surface.as_bytes().expect("surface byte access"),
            host.as_slice()
        );
        surfaces.push(surface);
    }
    assert!(
        session.submissions().expect("session submissions") >= 1,
        "queued RGB tiles should submit at least one resident Metal decode"
    );
    for surface in surfaces {
        assert!(completed_surface_metal_buffer(&surface).is_some());
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
    }
}

#[test]
fn metal_tile_batch_decodes_submitted_tiles_in_order() {
    if !should_run_metal_runtime() {
        return;
    }

    let classic_bytes = fixture_gray8();
    let reversed_bytes = fixture_gray8_reversed();
    let mut batch = MetalTileBatch::new();

    assert!(batch.is_empty());
    assert_eq!(
        batch
            .push_tile_request(
                &classic_bytes,
                MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Metal)
            )
            .expect("push classic tile"),
        0
    );
    assert_eq!(
        batch
            .push_shared_tile_request(
                Arc::<[u8]>::from(reversed_bytes.as_slice()),
                MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Metal)
            )
            .expect("push reversed tile"),
        1
    );
    assert_eq!(batch.len(), 2);
    assert_eq!(batch.submissions().expect("batch submissions"), 0);

    let surfaces = batch.decode_all().expect("batch decode");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Metal);
    assert_eq!(surfaces[1].backend_kind(), BackendKind::Metal);

    let mut classic_host_decoder = J2kDecoder::new(&classic_bytes).expect("classic host decoder");
    let mut classic_host = [0u8; 16];
    classic_host_decoder
        .decode_into(&mut classic_host, 4, PixelFormat::Gray8)
        .expect("classic host decode");

    let mut reversed_host_decoder =
        J2kDecoder::new(&reversed_bytes).expect("reversed host decoder");
    let mut reversed_host = [0u8; 16];
    reversed_host_decoder
        .decode_into(&mut reversed_host, 4, PixelFormat::Gray8)
        .expect("reversed host decode");

    assert_eq!(
        surfaces[0].as_bytes().expect("surface byte access"),
        classic_host.as_slice()
    );
    assert_eq!(
        surfaces[1].as_bytes().expect("surface byte access"),
        reversed_host.as_slice()
    );
}
