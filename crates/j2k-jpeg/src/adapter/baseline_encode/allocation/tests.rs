// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::encoded_output::JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

#[test]
fn aggregate_live_allocation_accepts_the_cap_and_rejects_one_more_byte() {
    assert_eq!(
        checked_encode_host_live_bytes([DEFAULT_MAX_HOST_ALLOCATION_BYTES - 1, 1])
            .expect("exact cap is valid"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_encode_host_live_bytes([DEFAULT_MAX_HOST_ALLOCATION_BYTES, 1]),
        Err(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES
        }) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
    ));
}

#[test]
fn gpu_single_accepts_exact_live_cap_and_rejects_one_more_entropy_byte() {
    let entropy = (DEFAULT_MAX_HOST_ALLOCATION_BYTES - JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY) / 2;
    assert_eq!(
        checked_gpu_tile_live_bytes(entropy).expect("exact live cap"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_gpu_tile_live_bytes(entropy + 1),
        Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
            if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 2
                && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn cpu_grayscale_formula_counts_borrow_metadata_entropy_and_frame() {
    let plane_metadata = size_of::<Cow<'static, [u8]>>();
    let entropy = (DEFAULT_MAX_HOST_ALLOCATION_BYTES
        - JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY
        - plane_metadata)
        / 2;
    assert_eq!(
        checked_cpu_encode_live_bytes(0, 1, entropy, entropy).expect("exact live cap"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_cpu_encode_live_bytes(0, 1, entropy + 1, entropy + 1),
        Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
            if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 2
                && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn gpu_batch_carries_retained_frames_across_source_groups() {
    let mut budget = GpuBatchAllocationBudget::new(4).expect("fixed metadata");
    let fixed_metadata = 4 * (size_of::<EncodedJpeg>() + size_of::<JpegBaselineGpuEncodeTile>());
    let entropy_outer = 2 * size_of::<Vec<u8>>();
    let plan_metadata = 2 * size_of::<JpegBaselineGpuEncodeParams>();
    let mut first = GpuEncodeGroupAllocation::default();
    first.add_tile(64).unwrap();
    first.add_tile(96).unwrap();
    budget.add_group(&first).unwrap();
    let first_frame_bytes =
        64 + JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY + 96 + JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;
    let first_peak =
        fixed_metadata + 64 + 96 + entropy_outer + first_frame_bytes.max(plan_metadata);
    assert_eq!(budget.peak_bytes(), first_peak);

    let mut second = GpuEncodeGroupAllocation::default();
    second.add_tile(80).unwrap();
    second.add_tile(112).unwrap();
    budget.add_group(&second).unwrap();
    let second_frame_bytes =
        80 + JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY + 112 + JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;
    let second_peak = fixed_metadata
        + first_frame_bytes
        + 80
        + 112
        + entropy_outer
        + second_frame_bytes.max(plan_metadata);
    assert_eq!(budget.peak_bytes(), second_peak);
    assert!(second_peak > first_peak);
}

#[test]
fn gpu_runtime_params_and_returned_outer_metadata_share_the_group_cap() {
    let retained = DEFAULT_MAX_HOST_ALLOCATION_BYTES - 3;
    assert_eq!(
        checked_gpu_group_runtime_live_bytes(retained, 1, 1, 1, 1)
            .expect("exact runtime group boundary"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_gpu_group_runtime_live_bytes(retained, 1, 1, 2, 1),
        Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
            if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
                && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn second_gpu_source_group_keeps_prior_actual_frames_live() {
    let prior_frame_bytes = DEFAULT_MAX_HOST_ALLOCATION_BYTES - 3;
    assert_eq!(
        checked_gpu_group_runtime_live_bytes(prior_frame_bytes, 1, 1, 1, 1)
            .expect("exact second-group boundary"),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    assert!(matches!(
        checked_gpu_group_runtime_live_bytes(prior_frame_bytes + 1, 1, 1, 1, 1),
        Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
            if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
                && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}
