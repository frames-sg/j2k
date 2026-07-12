// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    BackendRequest, CodecError, DecoderContext, DeviceSurface, Downscale, ImageDecode,
    ImageDecodeDevice, PixelFormat, Rect, TileBatchDecodeDevice,
};
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_cuda::{Codec, Decoder};

use super::support::BASELINE_420;

#[test]
fn auto_region_scaled_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let mut host_decoder = Decoder::new(BASELINE_420).expect("host decoder");
    let mut host = vec![0u8; scaled.w as usize * scaled.h as usize * 3];
    host_decoder
        .decode_region_scaled_into(
            &mut j2k_jpeg::ScratchPool::new(),
            &mut host,
            scaled.w as usize * 3,
            PixelFormat::Rgb8,
            roi,
            scale,
        )
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(host.as_slice()));
}

#[test]
fn tile_batch_region_scaled_auto_surface_matches_host_decode() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let scaled = roi.scaled_covering(scale);
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let surface = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Auto,
    )
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(surface.dimensions(), (scaled.w, scaled.h));

    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::region_scaled(
            PixelFormat::Rgb8,
            j2k_jpeg::Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        ))
        .expect("host decode");
    assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
}

#[test]
fn tile_batch_region_scaled_cuda_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let error = Codec::decode_tile_region_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        scale,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA tile-batch region+scaled decode should be unsupported");
    assert!(error.is_unsupported());
}

#[test]
fn tile_batch_region_cuda_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();

    let error = Codec::decode_tile_region_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        roi,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA tile-batch region decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("region output"));
}

#[test]
fn tile_batch_scaled_cuda_surface_fails_without_owned_cuda_path() {
    let mut ctx = DecoderContext::<j2k_jpeg::DecoderContext>::new();
    let mut pool = j2k_jpeg::ScratchPool::new();

    let error = Codec::decode_tile_scaled_to_device(
        &mut ctx,
        &mut pool,
        BASELINE_420,
        PixelFormat::Rgb8,
        Downscale::Half,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA tile-batch scaled decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("scaled output"));
}
