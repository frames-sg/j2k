// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn tile_batch_decode_many_device_preserves_full_tile_order() {
    if !should_run_metal_runtime() {
        return;
    }

    let classic_bytes = fixture_gray8();
    let reversed_bytes = fixture_gray8_reversed();
    let mut ctx = J2kContext::default();
    let mut pool = J2kScratchPool::new();
    let inputs = [classic_bytes.as_slice(), reversed_bytes.as_slice()];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Metal,
    )
    .expect("decode full-tile batch");

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

    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Metal);
    assert_eq!(surfaces[1].backend_kind(), BackendKind::Metal);
    assert_eq!(
        surfaces[0].as_bytes().expect("surface byte access"),
        classic_host.as_slice()
    );
    assert_eq!(
        surfaces[1].as_bytes().expect("surface byte access"),
        reversed_host.as_slice()
    );
}

#[test]
fn metal_tile_batch_supports_region_and_scaled_requests() {
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
    let mut batch = MetalTileBatch::with_capacity(2);

    assert_eq!(
        batch
            .push_tile_request(
                &bytes,
                MetalDecodeRequest::region(PixelFormat::Gray8, roi, BackendRequest::Metal)
            )
            .expect("push region tile"),
        0
    );
    assert_eq!(
        batch
            .push_tile_request(
                &bytes,
                MetalDecodeRequest::scaled(
                    PixelFormat::Gray8,
                    Downscale::Half,
                    BackendRequest::Metal
                )
            )
            .expect("push scaled tile"),
        1
    );

    let surfaces = batch.decode_all().expect("batch decode");
    assert_eq!(surfaces.len(), 2);
    assert_eq!(surfaces[0].dimensions(), (2, 2));
    assert_eq!(surfaces[1].dimensions(), (2, 2));
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Metal);
    assert_eq!(surfaces[1].backend_kind(), BackendKind::Metal);
}

#[test]
fn metal_tile_batch_supports_region_scaled_requests() {
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
    let mut batch = MetalTileBatch::with_capacity(1);

    assert_eq!(
        batch
            .push_tile_request(
                &bytes,
                MetalDecodeRequest::region_scaled(
                    PixelFormat::Gray8,
                    roi,
                    scale,
                    BackendRequest::Metal
                )
            )
            .expect("push region scaled tile"),
        0
    );

    let surfaces = batch.decode_all().expect("batch decode");
    assert_eq!(surfaces.len(), 1);
    assert_eq!(surfaces[0].dimensions(), (scaled.w, scaled.h));
    assert_eq!(surfaces[0].backend_kind(), BackendKind::Metal);
}

#[test]
fn submitted_distinct_region_scaled_htj2k_grayscale_tiles_flush_as_one_device_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let ht_bytes = fixture_ht_gray8();
    let reversed_bytes = fixture_ht_gray8_reversed();
    assert_ne!(ht_bytes, reversed_bytes, "HTJ2K batch fixtures must differ");
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let ht_submission = submit_tile_region_scaled_to_device!(
        &mut ctx,
        &mut session,
        &mut pool,
        &ht_bytes,
        PixelFormat::Gray8,
        roi,
        scale,
        BackendRequest::Metal,
    )
    .expect("submit ht region-scaled tile");
    let reversed_submission = submit_tile_region_scaled_to_device!(
        &mut ctx,
        &mut session,
        &mut pool,
        &reversed_bytes,
        PixelFormat::Gray8,
        roi,
        scale,
        BackendRequest::Metal,
    )
    .expect("submit reversed ht region-scaled tile");

    assert_eq!(
        session.submissions().expect("session submissions"),
        0,
        "region-scaled submitted tile surfaces should stay queued until wait"
    );

    let expected = [&ht_bytes, &reversed_bytes]
        .into_iter()
        .map(|bytes| {
            let mut decoder = J2kDecoder::new(bytes).expect("host decoder");
            let stride = scaled.w as usize;
            let mut host = vec![0u8; stride * scaled.h as usize];
            decoder
                .decode_region_scaled_into(
                    &mut J2kScratchPool::new(),
                    &mut host,
                    stride,
                    PixelFormat::Gray8,
                    roi,
                    scale,
                )
                .expect("host region-scaled decode");
            host
        })
        .collect::<Vec<_>>();

    let ht_surface = ht_submission.wait().expect("ht region-scaled surface");
    let reversed_surface = reversed_submission
        .wait()
        .expect("reversed ht region-scaled surface");
    assert_eq!(ht_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(reversed_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(ht_surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(reversed_surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        ht_surface.as_bytes().expect("surface byte access"),
        expected[0].as_slice()
    );
    assert_eq!(
        reversed_surface.as_bytes().expect("surface byte access"),
        expected[1].as_slice()
    );
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "distinct queued HTJ2K region-scaled grayscale tiles should flush through one Metal command buffer"
    );
}

#[test]
fn submitted_distinct_region_scaled_htj2k_gray16_tiles_flush_as_one_device_batch() {
    if !should_run_metal_runtime() {
        return;
    }

    let first_bytes = fixture_ht_gray12_offset(0);
    let second_bytes = fixture_ht_gray12_offset(37);
    assert_ne!(
        first_bytes, second_bytes,
        "HTJ2K Gray16 batch fixtures must differ"
    );
    let roi = Rect {
        x: 1,
        y: 0,
        w: 2,
        h: 3,
    };
    let scale = Downscale::Half;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = J2kContext::default();
    let mut session = MetalSession::default();
    let mut pool = J2kScratchPool::new();

    let first_submission = submit_tile_region_scaled_to_device!(
        &mut ctx,
        &mut session,
        &mut pool,
        &first_bytes,
        PixelFormat::Gray16,
        roi,
        scale,
        BackendRequest::Metal,
    )
    .expect("submit first ht gray16 region-scaled tile");
    let second_submission = submit_tile_region_scaled_to_device!(
        &mut ctx,
        &mut session,
        &mut pool,
        &second_bytes,
        PixelFormat::Gray16,
        roi,
        scale,
        BackendRequest::Metal,
    )
    .expect("submit second ht gray16 region-scaled tile");

    let expected = [&first_bytes, &second_bytes]
        .into_iter()
        .map(|bytes| {
            let mut decoder = J2kDecoder::new(bytes).expect("host decoder");
            let stride = scaled.w as usize * PixelFormat::Gray16.bytes_per_pixel();
            let mut host = vec![0u8; stride * scaled.h as usize];
            decoder
                .decode_region_scaled_into(
                    &mut J2kScratchPool::new(),
                    &mut host,
                    stride,
                    PixelFormat::Gray16,
                    roi,
                    scale,
                )
                .expect("host region-scaled gray16 decode");
            host
        })
        .collect::<Vec<_>>();

    let first_surface = first_submission.wait().expect("first gray16 surface");
    let second_surface = second_submission.wait().expect("second gray16 surface");
    assert_eq!(first_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(second_surface.backend_kind(), BackendKind::Metal);
    assert_eq!(first_surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(second_surface.dimensions(), (scaled.w, scaled.h));
    assert_eq!(
        first_surface.as_bytes().expect("surface byte access"),
        expected[0].as_slice()
    );
    assert_eq!(
        second_surface.as_bytes().expect("surface byte access"),
        expected[1].as_slice()
    );
    assert_eq!(
        session.submissions().expect("session submissions"),
        1,
        "distinct queued HTJ2K region-scaled Gray16 tiles should flush through one Metal command buffer"
    );
}
