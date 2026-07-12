// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_core::PixelFormat;
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_cuda::{Codec, CudaSession};
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

use super::support::{BASELINE_420, BASELINE_422, BASELINE_444, OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA};

#[test]
fn explicit_cuda_session_batch_records_owned_packet_cache_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let inputs = [BASELINE_420, BASELINE_420];
    let mut session = CudaSession::default();
    let surfaces = Codec::decode_tiles_to_device_with_session(
        &inputs,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cuda,
        &mut session,
    )
    .expect("cuda session batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    assert_eq!(session.owned_cuda_packet_cache_len(), 1);
    for surface in surfaces {
        let stats = surface.cuda_surface().expect("cuda surface").stats();
        assert!(stats.used_owned_cuda_decode());
    }
}

#[test]
fn explicit_cuda_decodes_into_caller_owned_buffer_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let mut session = CudaSession::default();
    let pitch = 16 * PixelFormat::Rgb8.bytes_per_pixel();
    let byte_len = pitch * 16;
    let buffer = session
        .take_owned_cuda_output_buffer(byte_len)
        .expect("device output buffer");

    let stats = Codec::decode_tile_rgb8_into_cuda_buffer_with_session(
        BASELINE_420,
        &buffer,
        pitch,
        &mut session,
    )
    .expect("direct owned CUDA decode");

    assert!(stats.used_owned_cuda_decode());
    assert_eq!(session.owned_cuda_packet_cache_len(), 1);

    let mut downloaded = vec![0u8; byte_len];
    buffer
        .copy_to_host(&mut downloaded)
        .expect("download buffer");
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    let max_delta = downloaded
        .iter()
        .zip(expected)
        .map(|(actual, expected)| actual.abs_diff(expected))
        .max()
        .unwrap_or(0);
    assert!(
        max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
        "direct J2K-owned CUDA decode differed from the CPU reference by max channel delta {max_delta}"
    );
}

#[test]
fn explicit_cuda_decodes_422_and_444_into_caller_owned_buffers_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    for (input, dimensions) in [
        (BASELINE_422, (16_u32, 8_u32)),
        (BASELINE_444, (8_u32, 8_u32)),
    ] {
        let mut session = CudaSession::default();
        let pitch = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let byte_len = pitch * dimensions.1 as usize;
        let buffer = session
            .take_owned_cuda_output_buffer(byte_len)
            .expect("device output buffer");

        let stats = Codec::decode_tile_rgb8_into_cuda_buffer_with_session(
            input,
            &buffer,
            pitch,
            &mut session,
        )
        .expect("direct owned CUDA decode");

        assert!(stats.used_owned_cuda_decode());
        assert_eq!(session.owned_cuda_packet_cache_len(), 1);

        let mut downloaded = vec![0u8; byte_len];
        buffer
            .copy_to_host(&mut downloaded)
            .expect("download buffer");
        let (expected, _) = j2k_jpeg::Decoder::new(input)
            .expect("host decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("host decode");
        let max_delta = downloaded
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
            "direct J2K-owned CUDA decode differed from the CPU reference by max channel delta {max_delta}"
        );
    }
}

#[test]
fn explicit_cuda_decodes_batch_into_caller_owned_buffers_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let cases = [
        (BASELINE_420, (16_u32, 16_u32)),
        (BASELINE_422, (16_u32, 8_u32)),
    ];
    let mut session = CudaSession::default();
    let buffers = cases
        .iter()
        .map(|(_, dimensions)| {
            let pitch = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            session
                .take_owned_cuda_output_buffer(pitch * dimensions.1 as usize)
                .expect("device output buffer")
        })
        .collect::<Vec<_>>();
    let tiles = cases
        .iter()
        .zip(buffers.iter())
        .map(
            |((input, dimensions), buffer)| j2k_jpeg_cuda::CudaJpegDecodeOutputTile {
                input,
                output: buffer,
                pitch_bytes: dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel(),
            },
        )
        .collect::<Vec<_>>();

    let stats = Codec::decode_tiles_rgb8_into_cuda_buffers_with_session(&tiles, &mut session)
        .expect("direct owned CUDA batch decode");

    assert_eq!(stats.len(), cases.len());
    assert_eq!(session.owned_cuda_packet_cache_len(), cases.len());
    for ((input, dimensions), (buffer, stats)) in cases.iter().zip(buffers.iter().zip(stats)) {
        assert!(stats.used_owned_cuda_decode());
        let pitch = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let mut downloaded = vec![0u8; pitch * dimensions.1 as usize];
        buffer
            .copy_to_host(&mut downloaded)
            .expect("download buffer");
        let (expected, _) = j2k_jpeg::Decoder::new(input)
            .expect("host decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("host decode");
        let max_delta = downloaded
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
            "direct J2K-owned CUDA batch decode differed from the CPU reference by max channel delta {max_delta}"
        );
    }
}
