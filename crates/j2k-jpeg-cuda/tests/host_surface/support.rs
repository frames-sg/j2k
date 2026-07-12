// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, DeviceSurface, ImageDecodeDevice, PixelFormat};
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_cuda::Decoder;
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

pub(super) const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");
pub(super) const BASELINE_422: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_422_16x8.jpg");
pub(super) const BASELINE_444: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_444_8x8.jpg");
pub(super) const OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA: u8 = 2;

#[cfg(feature = "cuda-runtime")]
pub(super) fn generated_rgb_jpeg(
    subsampling: j2k_jpeg::JpegSubsampling,
    width: u32,
    height: u32,
) -> Vec<u8> {
    generated_rgb_jpeg_with_restart(subsampling, width, height, None)
}

pub(super) fn generated_rgb_jpeg_with_restart(
    subsampling: j2k_jpeg::JpegSubsampling,
    width: u32,
    height: u32,
    restart_interval: Option<u16>,
) -> Vec<u8> {
    let rgb = j2k_test_support::gpu_bench_rgb8(width, height);
    j2k_jpeg::encode_jpeg_baseline(
        j2k_jpeg::JpegSamples::Rgb8 {
            data: &rgb,
            width,
            height,
        },
        j2k_jpeg::JpegEncodeOptions {
            quality: 90,
            subsampling,
            restart_interval,
            backend: j2k_jpeg::JpegBackend::Cpu,
        },
    )
    .expect("generated JPEG")
    .data
}

pub(super) fn assert_cuda_surface(surface: &j2k_jpeg_cuda::Surface) {
    let cuda = surface.cuda_surface().expect("cuda surface");
    assert_ne!(cuda.device_ptr(), 0);
    assert!(cuda.stats().kernel_dispatches() > 0);
}

pub(super) fn assert_surface_bytes_match_or_are_close(
    surface: &j2k_jpeg_cuda::Surface,
    actual: &[u8],
    expected: &[u8],
) {
    assert_eq!(actual.len(), expected.len());
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    if stats.used_owned_cuda_decode() {
        let max_delta = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| actual.abs_diff(*expected))
            .max()
            .unwrap_or(0);
        assert!(
            max_delta <= OWNED_CUDA_RGB8_MAX_CHANNEL_DELTA,
            "J2K-owned CUDA decode differed from the CPU reference by max channel delta {max_delta}"
        );
        return;
    }
    assert_eq!(actual, expected);
}

pub(super) fn assert_full_frame_owned_cuda_decode_when_required(
    input: &[u8],
    dimensions: (u32, u32),
) {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let mut decoder = Decoder::new(input).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_eq!(surface.dimensions(), dimensions);
    assert_eq!(surface.as_host_bytes(), None);
    assert_cuda_surface(&surface);
    let stats = surface.cuda_surface().expect("cuda surface").stats();
    assert!(stats.used_owned_cuda_decode());
    assert!(!stats.used_hardware_decode());
    assert_eq!(stats.copy_kernel_dispatches(), 0);

    let mut downloaded = vec![0u8; surface.byte_len()];
    surface
        .download_into(&mut downloaded, surface.pitch_bytes())
        .expect("download cuda surface");
    let (expected, _) = j2k_jpeg::Decoder::new(input)
        .expect("host decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
        .expect("host decode");
    assert_surface_bytes_match_or_are_close(&surface, &downloaded, &expected);
}
