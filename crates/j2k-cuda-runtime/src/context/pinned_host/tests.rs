// SPDX-License-Identifier: MIT OR Apache-2.0

use super::validate_non_null_pinned_host_allocation;
use crate::CudaError;

#[test]
fn nonzero_null_pinned_allocation_is_rejected_before_safe_slice_construction() {
    assert!(matches!(
        validate_non_null_pinned_host_allocation(std::ptr::null_mut(), 1),
        Err(CudaError::InternalInvariant { .. })
    ));
    validate_non_null_pinned_host_allocation(std::ptr::dangling_mut(), 1)
        .expect("non-null allocation");
}
