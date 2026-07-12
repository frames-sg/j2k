// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda-runtime")]

use j2k_core::{BackendRequest, DeviceSurface, PixelFormat};
use j2k_jpeg_cuda::{Codec, CudaSession};

use super::support::BASELINE_420;

#[test]
fn session_batch_and_host_surfaces_remain_charged_until_the_batch_drops() {
    let inputs = [BASELINE_420, BASELINE_420];
    let mut session = CudaSession::default();
    let surfaces = Codec::decode_tiles_to_device_with_session(
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cpu,
        &mut session,
    )
    .expect("session-backed host surfaces");

    let minimum_surface_bytes = surfaces.iter().map(DeviceSurface::byte_len).sum::<usize>();
    assert!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes
            >= minimum_surface_bytes
    );

    drop(surfaces);
    assert_eq!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes,
        0
    );
}

#[test]
fn failed_session_batch_releases_completed_surface_and_batch_leases() {
    let inputs: [&[u8]; 2] = [BASELINE_420, b"not a JPEG stream"];
    let mut session = CudaSession::default();

    Codec::decode_tiles_to_device_with_session(
        &inputs,
        PixelFormat::Rgb8,
        BackendRequest::Cpu,
        &mut session,
    )
    .expect_err("malformed second input must reject the batch");

    assert_eq!(
        session
            .owned_cuda_host_memory_diagnostics()
            .unwrap()
            .active_owner_bytes,
        0
    );
}
