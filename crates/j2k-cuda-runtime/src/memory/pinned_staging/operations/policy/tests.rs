// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Barrier, Mutex,
};

use super::{
    lock_pinned_upload_operation, validate_pinned_upload_operation_context,
    validate_pinned_upload_staging_len, PINNED_UPLOAD_STAGING_ALLOCATION,
};
use crate::CudaError;

#[test]
fn pinned_staging_allocation_accepts_exact_cap_and_rejects_one_over() {
    validate_pinned_upload_staging_len(64, 64).expect("exact cap must fit");
    assert!(matches!(
        validate_pinned_upload_staging_len(65, 64),
        Err(CudaError::HostAllocationTooLarge {
            requested: 65,
            cap: 64,
            what: PINNED_UPLOAD_STAGING_ALLOCATION,
        })
    ));
}

#[test]
fn explicit_empty_staging_checkout_is_rejected_without_driver_work() {
    assert!(matches!(
        validate_pinned_upload_staging_len(0, 64),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn foreign_pinned_upload_operation_is_rejected() {
    validate_pinned_upload_operation_context(true).expect("matching context");
    assert!(matches!(
        validate_pinned_upload_operation_context(false),
        Err(CudaError::InvalidArgument { .. })
    ));
}

#[test]
fn clone_shared_operation_gate_serializes_checkout_through_recycle_window() {
    let gate = Arc::new(Mutex::new(()));
    let barrier = Arc::new(Barrier::new(3));
    let active = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        for _ in 0..2 {
            let gate = Arc::clone(&gate);
            let barrier = Arc::clone(&barrier);
            let active = Arc::clone(&active);
            let peak = Arc::clone(&peak);
            scope.spawn(move || {
                barrier.wait();
                let _operation = lock_pinned_upload_operation(&gate).expect("lock operation gate");
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                for _ in 0..1_000 {
                    std::thread::yield_now();
                }
                active.fetch_sub(1, Ordering::SeqCst);
            });
        }
        barrier.wait();
    });
    assert_eq!(peak.load(Ordering::SeqCst), 1);
}

#[test]
fn poisoned_operation_gate_surfaces_typed_error() {
    let gate = Mutex::new(());
    let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = lock_pinned_upload_operation(&gate).expect("lock operation gate");
        panic!("poison operation gate");
    }));
    assert!(panic_result.is_err());
    assert!(matches!(
        lock_pinned_upload_operation(&gate),
        Err(CudaError::StatePoisoned { .. })
    ));
}

#[test]
fn reentrant_operation_gate_is_rejected_instead_of_deadlocking() {
    let gate = Mutex::new(());
    let held = lock_pinned_upload_operation(&gate).expect("lock operation gate");
    assert!(matches!(
        lock_pinned_upload_operation(&gate),
        Err(CudaError::InvalidArgument { .. })
    ));
    let other_gate = Mutex::new(());
    drop(lock_pinned_upload_operation(&other_gate).expect("independent gate"));
    drop(held);
    drop(lock_pinned_upload_operation(&gate).expect("gate reusable after release"));
}

#[test]
fn operation_gate_hold_is_tracked_per_thread() {
    let gate = Arc::new(Mutex::new(()));
    let held = lock_pinned_upload_operation(&gate).expect("lock operation gate");
    let peer_gate = Arc::clone(&gate);
    let peer = std::thread::spawn(move || {
        // A peer thread blocks on the gate rather than seeing this thread's hold.
        lock_pinned_upload_operation(&peer_gate).map(drop)
    });
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(!peer.is_finished(), "peer must wait for the held gate");
    drop(held);
    peer.join()
        .expect("peer thread")
        .expect("peer acquires the released gate");
}
