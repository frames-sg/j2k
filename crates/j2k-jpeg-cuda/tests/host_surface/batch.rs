// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, CodecError, DeviceSurface, PixelFormat, TileBatchDecodeManyDevice};
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_cuda::{Codec, CudaSession, Error};
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

use super::support::{assert_cuda_surface, assert_surface_bytes_match_or_are_close, BASELINE_420};

#[test]
fn decode_tiles_to_device_auto_preserves_order_and_matches_host_bytes() {
    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (16, 16));
        match surface.backend_kind() {
            j2k_core::BackendKind::Cpu => {
                assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
            }
            j2k_core::BackendKind::Cuda => {
                let mut downloaded = vec![0u8; surface.byte_len()];
                surface
                    .download_into(&mut downloaded, surface.pitch_bytes())
                    .expect("download cuda surface");
                assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
            }
            j2k_core::BackendKind::Metal => panic!("JPEG CUDA batch returned Metal surface"),
        }
    }
}

#[test]
fn decode_tiles_to_device_with_session_auto_preserves_order_and_matches_host_bytes() {
    let inputs = [BASELINE_420, BASELINE_420];
    let mut session = CudaSession::default();

    let surfaces = Codec::decode_tiles_to_device_with_session(
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
        &mut session,
    )
    .expect("session-backed batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    for surface in surfaces {
        assert_eq!(surface.dimensions(), (16, 16));
        match surface.backend_kind() {
            j2k_core::BackendKind::Cpu => {
                assert_eq!(surface.as_host_bytes(), Some(expected.as_slice()));
            }
            j2k_core::BackendKind::Cuda => {
                let mut downloaded = vec![0u8; surface.byte_len()];
                surface
                    .download_into(&mut downloaded, surface.pitch_bytes())
                    .expect("download cuda surface");
                assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
            }
            j2k_core::BackendKind::Metal => panic!("JPEG CUDA batch returned Metal surface"),
        }
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_returns_cuda_surfaces_or_clear_unavailable_error() {
    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    match Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    ) {
        Ok(surfaces) => {
            assert_eq!(surfaces.len(), inputs.len());
            for surface in surfaces {
                assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
                assert_eq!(surface.as_host_bytes(), None);
                assert_cuda_surface(&surface);
            }
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}

#[test]
fn decode_tiles_to_device_explicit_cuda_gray8_fails_without_cpu_upload() {
    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let error = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Gray8,
        BackendRequest::Cuda,
    )
    .expect_err("strict CUDA Gray8 batch decode should be unsupported");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn decode_tiles_to_device_explicit_cuda_uses_owned_decode_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let mut ctx = j2k_jpeg::DecoderContext::default();
    let mut pool = j2k_jpeg::ScratchPool::new();
    let inputs = [BASELINE_420, BASELINE_420];

    let surfaces = Codec::decode_tiles_to_device(
        &mut ctx,
        &mut pool,
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("cuda batch surfaces");

    assert_eq!(surfaces.len(), inputs.len());
    for surface in surfaces {
        let stats = surface.cuda_surface().expect("cuda surface").stats();
        assert!(
            stats.used_owned_cuda_decode(),
            "explicit full-tile RGB8 CUDA batch decode must use the J2K-owned CUDA path when required"
        );
        assert!(
            stats.decode_kernel_dispatches() > 0,
            "owned CUDA batch decode path must report decode dispatches"
        );
        assert_eq!(
            stats.copy_kernel_dispatches(),
            0,
            "owned CUDA batch decode path should not be reported as CPU decode plus copy"
        );
    }
}
