// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, ImageDecodeDevice, PixelFormat};
use j2k_jpeg_cuda::Decoder;
use j2k_test_support::cuda_jpeg_hardware_decode_gate;

#[cfg(feature = "cuda-runtime")]
use super::support::generated_rgb_jpeg;
use super::support::{
    assert_full_frame_owned_cuda_decode_when_required, generated_rgb_jpeg_with_restart,
    BASELINE_420, BASELINE_422, BASELINE_444,
};

#[test]
fn explicit_cuda_full_frame_uses_owned_decode_when_required() {
    if !cuda_jpeg_hardware_decode_gate(module_path!()) {
        return;
    }

    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let surface = decoder
        .decode_to_device(PixelFormat::Rgb8, BackendRequest::Cuda)
        .expect("cuda surface");
    let cuda = surface.cuda_surface().expect("cuda surface");
    let stats = cuda.stats();
    assert!(
        stats.used_owned_cuda_decode(),
        "explicit full-frame RGB8 CUDA decode must use the J2K-owned CUDA JPEG path when required"
    );
    assert!(
        !stats.used_hardware_decode(),
        "strict J2K-owned CUDA JPEG decode must not report external hardware decode"
    );
    assert!(
        stats.decode_kernel_dispatches() > 0,
        "owned CUDA decode path must report decode kernel dispatches"
    );
    assert_eq!(
        stats.copy_kernel_dispatches(),
        0,
        "owned CUDA decode path should not be reported as the CPU decode plus copy fallback"
    );
}

#[test]
fn explicit_cuda_full_frame_422_uses_owned_decode_when_required() {
    assert_full_frame_owned_cuda_decode_when_required(BASELINE_422, (16, 8));
}

#[test]
fn explicit_cuda_full_frame_444_uses_owned_decode_when_required() {
    assert_full_frame_owned_cuda_decode_when_required(BASELINE_444, (8, 8));
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_multi_checkpoint_420_uses_owned_decode_when_required() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16);
    assert_full_frame_owned_cuda_decode_when_required(&input, (32, 16));
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_multi_row_and_column_420_matches_cpu_when_required() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr420, 32, 32);
    assert_full_frame_owned_cuda_decode_when_required(&input, (32, 32));
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_multi_mcu_422_matches_cpu_when_required() {
    let input = generated_rgb_jpeg(j2k_jpeg::JpegSubsampling::Ybr422, 32, 8);
    assert_full_frame_owned_cuda_decode_when_required(&input, (32, 8));
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_odd_subsampled_dimensions_match_cpu_when_required() {
    for (subsampling, dimensions) in [
        (j2k_jpeg::JpegSubsampling::Ybr420, (17, 17)),
        (j2k_jpeg::JpegSubsampling::Ybr422, (17, 9)),
    ] {
        let input = generated_rgb_jpeg(subsampling, dimensions.0, dimensions.1);
        assert_full_frame_owned_cuda_decode_when_required(&input, dimensions);
    }
}

#[test]
fn explicit_cuda_restart_checkpoint_420_uses_owned_decode_when_required() {
    let input = generated_rgb_jpeg_with_restart(j2k_jpeg::JpegSubsampling::Ybr420, 32, 16, Some(1));
    assert_full_frame_owned_cuda_decode_when_required(&input, (32, 16));
}
