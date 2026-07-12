// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    BackendRequest, CodecError, DeviceSurface, Downscale, ImageDecodeDevice, PixelFormat, Rect,
};
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_cuda::{Decoder, Error};
use j2k_test_support::cuda_runtime_gate;

use super::support::{assert_cuda_surface, assert_surface_bytes_match_or_are_close, BASELINE_420};

#[test]
fn auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Auto)
        .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_or_clear_unavailable_error() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    match decoder.decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda) {
        Ok(surface) => {
            assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
            assert_eq!(surface.as_host_bytes(), None);
            #[cfg(feature = "cuda-runtime")]
            assert_ne!(
                surface.cuda_surface().expect("cuda surface").device_ptr(),
                0
            );
        }
        Err(error) => assert!(error.is_unsupported()),
    }
}

#[test]
fn explicit_cuda_request_validates_decode_before_upload() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Rgba16, BackendRequest::Cuda)
        .expect_err("unsupported decode");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn explicit_cuda_gray8_request_fails_without_cpu_upload() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Cuda)
        .expect_err("strict CUDA Gray8 decode should be unsupported");
    assert!(error.is_unsupported());
    assert!(!matches!(error, Error::CudaUnavailable));
}

#[test]
fn explicit_cuda_request_returns_cuda_surface_when_cuda_runtime_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    assert_eq!(surface.dimensions(), (16, 16));

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");

    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
}

#[test]
fn explicit_cuda_region_scaled_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let scale = Downscale::Quarter;

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let error = decoder
        .decode_region_scaled_to_device(PixelFormat::Rgb8, roi, scale, BackendRequest::Cuda)
        .expect_err("strict CUDA region+scaled decode should be unsupported");
    assert!(error.is_unsupported());
}

#[test]
fn explicit_cuda_region_surface_fails_without_owned_cuda_path() {
    let roi = Rect {
        x: 4,
        y: 4,
        w: 10,
        h: 10,
    };
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_region_to_device(PixelFormat::Rgb8, roi, BackendRequest::Cuda)
        .expect_err("strict CUDA region decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("region output"));
}

#[test]
fn explicit_cuda_scaled_surface_fails_without_owned_cuda_path() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

    let error = decoder
        .decode_scaled_to_device(PixelFormat::Rgb8, Downscale::Half, BackendRequest::Cuda)
        .expect_err("strict CUDA scaled decode should be unsupported");

    assert!(error.is_unsupported());
    assert!(error.to_string().contains("scaled output"));
}

#[test]
fn explicit_cuda_download_respects_padded_stride_when_cuda_runtime_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_cuda_surface(&surface);
    let row_bytes = surface.pitch_bytes();
    let stride = row_bytes + 5;
    let mut downloaded = vec![0xCD; stride * surface.dimensions().1 as usize];
    surface
        .download_into(&mut downloaded, stride)
        .expect("download cuda surface");

    let (expected, _) = j2k_jpeg::Decoder::new(BASELINE_420)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    for (row, expected_row) in expected.chunks(row_bytes).enumerate() {
        let start = row * stride;
        assert_surface_bytes_match_or_are_close(
            &surface,
            &downloaded[start..start + row_bytes],
            expected_row,
        );
        assert_eq!(&downloaded[start + row_bytes..start + stride], &[0xCD; 5]);
    }
}
