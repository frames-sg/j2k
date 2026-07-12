// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, DeviceSubmission, DeviceSurface, ImageDecodeSubmit, PixelFormat};
use j2k_jpeg_cuda::{CudaSession, Decoder};
#[cfg(feature = "cuda-runtime")]
use j2k_test_support::cuda_runtime_gate;

#[cfg(feature = "cuda-runtime")]
use super::support::assert_cuda_surface;
use super::support::BASELINE_420;

#[test]
fn cuda_session_owned_decode_cache_starts_empty() {
    let session = CudaSession::default();

    assert_eq!(session.owned_cuda_packet_cache_len(), 0);
}

#[test]
fn submit_to_device_auto_falls_back_to_cpu_surface() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert!(surface.as_host_bytes().is_some());
    assert!(session.submissions() >= 1);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn submit_to_device_auto_does_not_initialize_cuda_runtime() {
    let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
    let mut session = CudaSession::default();
    let surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut decoder,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
    )
    .expect("submission")
    .wait()
    .expect("surface");
    assert_eq!(surface.backend_kind(), j2k_core::BackendKind::Cpu);
    assert_eq!(session.submissions(), 1);
    assert!(!session.is_runtime_initialized());
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn explicit_cuda_submissions_reuse_session_runtime_when_required() {
    if !cuda_runtime_gate(module_path!()) {
        return;
    }

    let mut session = CudaSession::default();
    assert!(!session.is_runtime_initialized());

    let mut first = Decoder::new(BASELINE_420).expect("decoder");
    let first_surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut first,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("first submission")
    .wait()
    .expect("first surface");
    assert_eq!(first_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_cuda_surface(&first_surface);
    assert!(session.is_runtime_initialized());

    let mut second = Decoder::new(BASELINE_420).expect("decoder");
    let second_surface = <Decoder<'_> as ImageDecodeSubmit<'_>>::submit_to_device(
        &mut second,
        &mut session,
        PixelFormat::Rgb8,
        BackendRequest::Cuda,
    )
    .expect("second submission")
    .wait()
    .expect("second surface");
    assert_eq!(second_surface.backend_kind(), j2k_core::BackendKind::Cuda);
    assert_cuda_surface(&second_surface);
    assert_eq!(session.submissions(), 2);
    assert!(session.is_runtime_initialized());
}
