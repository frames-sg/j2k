// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_batch_private_host_bytes, checked_single_private_host_bytes,
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEncodeStatus,
};
use crate::CudaError;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

#[test]
fn cuda_runtime_single_host_peak_has_an_exact_cap_boundary() {
    assert_eq!(
        checked_single_private_host_bytes(0, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
            .expect("exact cap is valid"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_single_private_host_bytes(0, DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1),
        Err(CudaError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "JPEG baseline single encode output",
        }) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
    ));
}

#[test]
fn cuda_runtime_batch_host_peak_has_an_exact_cap_boundary() {
    let tile_count = 2;
    let fixed = tile_count
        * (core::mem::size_of::<CudaJpegBaselineEncodeParams>()
            + core::mem::size_of::<CudaJpegBaselineEncodeStatus>()
            + core::mem::size_of::<Vec<u8>>());
    let entropy_capacity = DEFAULT_MAX_HOST_ALLOCATION_BYTES - fixed;
    assert_eq!(
        checked_batch_private_host_bytes(
            0,
            tile_count,
            tile_count,
            tile_count,
            tile_count,
            entropy_capacity,
        )
        .expect("exact cap is valid"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_batch_private_host_bytes(
            0,
            tile_count,
            tile_count,
            tile_count,
            tile_count,
            entropy_capacity + 1,
        ),
        Err(CudaError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "JPEG baseline batch encode output",
        }) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
    ));
}

#[test]
fn cuda_runtime_batch_counts_retained_parameter_capacity_and_validation_scratch() {
    let param_size = core::mem::size_of::<CudaJpegBaselineEncodeParams>();
    let over_cap_capacity = DEFAULT_MAX_HOST_ALLOCATION_BYTES / param_size + 1;
    assert!(matches!(
        checked_batch_private_host_bytes(0, over_cap_capacity, 1, 1, 1, 1),
        Err(CudaError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "JPEG baseline batch encode validation",
        }) if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn cuda_runtime_encode_counts_adapter_external_live_bytes() {
    let external = 17;
    assert_eq!(
        checked_single_private_host_bytes(external, DEFAULT_MAX_HOST_ALLOCATION_BYTES - external,)
            .expect("external plus output at the cap is valid"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_single_private_host_bytes(
            external,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES - external + 1,
        ),
        Err(CudaError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what: "JPEG baseline single encode output",
        }) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
    ));
}
