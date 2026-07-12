// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_tag_tree_retained_bytes, try_tag_tree_vec_with_capacity,
    CudaHtj2kPacketizationPlanError, CUDA_PACKETIZATION_TAG_TREE_ALLOCATION,
};
use crate::allocation::HostPhaseBudget;

#[test]
fn tag_tree_oversized_request_is_rejected_before_allocation() {
    let mut budget = HostPhaseBudget::new("test tag tree");
    assert!(matches!(
        try_tag_tree_vec_with_capacity::<u32>(&mut budget, usize::MAX),
        Err(CudaHtj2kPacketizationPlanError::MemoryCapExceeded {
            what: "test tag tree",
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })
    ));
}

#[test]
fn tag_tree_actual_capacity_has_exact_and_one_over_boundaries() {
    assert_eq!(
        checked_tag_tree_retained_bytes([1, 1], 1, [1, 1, 1], 28),
        Ok(28)
    );
    assert_eq!(
        checked_tag_tree_retained_bytes([1, 1], 1, [1, 1, 1], 27),
        Err(CudaHtj2kPacketizationPlanError::MemoryCapExceeded {
            what: CUDA_PACKETIZATION_TAG_TREE_ALLOCATION,
            requested: 28,
            cap: 27,
        })
    );
}
