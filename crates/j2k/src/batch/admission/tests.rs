// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn reconciling_actual_capacity_releases_admission_headroom() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    let mut claim = budget
        .claim(J2K_BATCH_EXECUTION_CAP_BYTES)
        .expect("full claim");
    claim
        .reconcile(J2K_BATCH_EXECUTION_CAP_BYTES / 2)
        .expect("shrink claim");
    let second = budget
        .claim(J2K_BATCH_EXECUTION_CAP_BYTES / 2)
        .expect("released half is immediately admissible");
    drop((claim, second));
    assert_eq!(budget.lock_state().live, 0);
}

#[test]
fn shared_baseline_is_subtracted_from_each_initial_worker_claim() {
    let shared = super::super::allocation::GENERIC_WORKER_CLAIM_BYTES / 8;
    let worker = super::super::allocation::GENERIC_WORKER_CLAIM_BYTES - shared;
    let budget = BatchAllocationBudget::with_baseline(shared).expect("shared-plan budget");
    let claims = (0..super::super::allocation::MAX_GENERIC_BATCH_WORKERS)
        .map(|_| {
            budget
                .claim(worker)
                .expect("worker claim excluding shared plan")
        })
        .collect::<Vec<_>>();
    drop(claims);
    assert_eq!(budget.lock_state().live, shared);
}

#[test]
fn waiting_claim_resumes_after_an_owner_reconciles() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    let mut owners = (0..super::super::allocation::MAX_GENERIC_BATCH_WORKERS)
        .map(|_| {
            budget
                .claim(super::super::allocation::GENERIC_WORKER_CLAIM_BYTES)
                .expect("generic owner")
        })
        .collect::<Vec<_>>();
    let waiting_budget = Arc::clone(&budget);
    let waiter = std::thread::spawn(move || {
        waiting_budget
            .claim(1)
            .expect("waiting direct claim resumes")
    });
    budget.wait_until_waiting();
    owners[0]
        .reconcile(super::super::allocation::GENERIC_WORKER_CLAIM_BYTES - 1)
        .expect("release one byte of actual-capacity headroom");
    let resumed = waiter.join().expect("waiting worker joined");
    drop((owners, resumed));
    assert_eq!(budget.lock_state().live, 0);
}

#[test]
fn unwinding_owner_releases_its_admission_claim() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    let unwind_budget = Arc::clone(&budget);
    assert!(std::panic::catch_unwind(move || {
        let _claim = unwind_budget
            .claim(J2K_BATCH_EXECUTION_CAP_BYTES)
            .expect("full claim");
        panic!("unwind admitted owner");
    })
    .is_err());
    let replacement = budget
        .claim(J2K_BATCH_EXECUTION_CAP_BYTES)
        .expect("unwound claim was released");
    drop(replacement);
    assert_eq!(budget.lock_state().live, 0);
}

#[test]
fn failed_reconciliation_releases_the_original_claim_on_drop() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    {
        let mut claim = budget
            .claim(J2K_BATCH_EXECUTION_CAP_BYTES)
            .expect("full claim");
        assert!(matches!(
            claim.reconcile(J2K_BATCH_EXECUTION_CAP_BYTES + 1),
            Err(BatchInfrastructureError::AllocationTooLarge { .. })
        ));
    }
    let replacement = budget
        .claim(J2K_BATCH_EXECUTION_CAP_BYTES)
        .expect("failed reconciliation did not leak its claim");
    drop(replacement);
    assert_eq!(budget.lock_state().live, 0);
}

#[test]
fn poisoned_admission_state_is_a_typed_infrastructure_error() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    let poison = Arc::clone(&budget);
    assert!(std::thread::spawn(move || {
        let _guard = poison.state.lock().expect("initial lock");
        panic!("poison admission state");
    })
    .join()
    .is_err());
    assert!(matches!(
        budget.claim(1),
        Err(BatchInfrastructureError::SchedulerPoisoned)
    ));
}
