// SPDX-License-Identifier: MIT OR Apache-2.0

use super::SharedCudaHostBudget;
use crate::CudaError;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

#[test]
fn owner_first_rejects_pinned_growth_without_mutation() {
    let authority = SharedCudaHostBudget::new();
    let owner = authority
        .register_external_owner(DEFAULT_MAX_HOST_ALLOCATION_BYTES - 7)
        .expect("register owner");
    let before = authority.current_bytes().unwrap();
    assert!(matches!(
        authority.reserve_pinned(8),
        Err(CudaError::HostAllocationTooLarge { .. })
    ));
    assert_eq!(authority.current_bytes().unwrap(), before);
    assert_eq!(owner.bytes(), DEFAULT_MAX_HOST_ALLOCATION_BYTES - 7);
}

#[test]
fn pinned_first_rejects_external_growth_transactionally() {
    let authority = SharedCudaHostBudget::new();
    authority.reserve_pinned(7).expect("seed pinned");
    assert!(matches!(
        authority.register_external_owner(DEFAULT_MAX_HOST_ALLOCATION_BYTES - 6),
        Err(CudaError::HostAllocationTooLarge { .. })
    ));
    assert_eq!(authority.current_bytes().unwrap(), 7);
}

#[test]
fn external_owner_growth_shrink_and_drop_are_exact() {
    let authority = SharedCudaHostBudget::new();
    let mut owner = authority.register_external_owner(11).unwrap();
    assert_eq!(authority.current_bytes().unwrap(), 11);
    owner.reconcile(17).unwrap();
    assert_eq!(authority.current_bytes().unwrap(), 17);
    owner.reconcile(5).unwrap();
    assert_eq!(authority.current_bytes().unwrap(), 5);
    drop(owner);
    assert_eq!(authority.current_bytes().unwrap(), 0);
}

#[test]
fn replacement_reserves_headroom_and_rolls_back_on_drop() {
    let authority = SharedCudaHostBudget::new();
    let mut owner = authority.register_external_owner(11).unwrap();
    let reservation = owner.reserve_replacement(7).unwrap();
    assert_eq!(reservation.external_live_bytes(), 4);
    assert!(matches!(
        authority.reserve_pinned(1),
        Err(CudaError::HostAllocationTooLarge { .. })
    ));
    drop(reservation);
    authority.reserve_pinned(1).unwrap();
    assert_eq!(owner.bytes(), 11);
}

#[test]
fn replacement_allows_same_context_calls_without_recursive_locking() {
    let authority = SharedCudaHostBudget::new();
    let mut owner = authority.register_external_owner(0).unwrap();
    let reservation = owner.reserve_replacement(0).unwrap();
    assert!(matches!(
        authority.reserve_pinned(1),
        Err(CudaError::HostAllocationTooLarge { .. })
    ));
    reservation.commit(0).unwrap();
}

#[test]
fn external_owner_can_outlive_context_authority_cleanup() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let context = super::super::CudaContext::system_default().expect("CUDA context");
    let owner = context
        .register_external_host_owner(17)
        .expect("external owner");
    drop(context);
    assert_eq!(owner.context_live_bytes().unwrap(), 17);
    drop(owner);
}
