// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    host_element_bytes, try_vec_defaulted, try_vec_filled, try_vec_from_slice,
    try_vec_with_capacity, HostPhaseBudget,
};
use crate::CudaError;

#[test]
fn logically_oversized_capacities_are_rejected_before_allocation() {
    for result in [
        try_vec_with_capacity::<u32>(usize::MAX),
        try_vec_filled(usize::MAX, 0u32),
        try_vec_defaulted::<u32>(usize::MAX),
    ] {
        assert!(matches!(
            result,
            Err(CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA host vector capacity",
            })
        ));
    }
}

#[test]
fn initialized_and_copied_allocations_preserve_contents() {
    assert_eq!(try_vec_filled(3, 7u16).unwrap(), [7, 7, 7]);
    assert_eq!(try_vec_from_slice(&[1u8, 2, 3]).unwrap(), [1, 2, 3]);
}

#[test]
fn element_byte_accounting_saturates() {
    assert_eq!(host_element_bytes::<u32>(3), 12);
    assert_eq!(host_element_bytes::<u32>(usize::MAX), usize::MAX);
}

#[test]
fn actual_capacity_phase_budget_counts_every_live_owner() {
    let first = j2k_core::try_host_vec_with_capacity::<u8>(8).unwrap();
    let second = j2k_core::try_host_vec_with_capacity::<u8>(8).unwrap();
    let actual = first.capacity().saturating_add(second.capacity());

    let mut budget = HostPhaseBudget::with_cap("test phase", actual);
    budget.account_vec(&first).unwrap();
    budget.account_vec(&second).unwrap();
    assert_eq!(budget.live_bytes(), actual);

    let mut one_under = HostPhaseBudget::with_cap("test phase", actual.saturating_sub(1));
    one_under.account_vec(&first).unwrap();
    assert!(matches!(
        one_under.account_vec(&second),
        Err(CudaError::HostAllocationTooLarge {
            requested,
            cap,
            what: "test phase",
        }) if requested == actual && cap == actual.saturating_sub(1)
    ));
}

#[test]
fn allocator_overcapacity_is_rejected_with_phase_context() {
    let values = j2k_core::try_host_vec_with_capacity::<u8>(17).unwrap();
    let mut budget = HostPhaseBudget::with_cap("overcapacity test", 16);

    assert!(matches!(
        budget.account_vec(&values),
        Err(CudaError::HostAllocationTooLarge {
            requested,
            cap: 16,
            what: "overcapacity test",
        }) if requested == values.capacity()
    ));
}
