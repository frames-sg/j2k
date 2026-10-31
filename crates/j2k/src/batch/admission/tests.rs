// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn reconciling_actual_capacity_releases_admission_headroom() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    let mut claim = budget
        .claim(batch_execution_cap_bytes())
        .expect("full claim");
    claim
        .reconcile(batch_execution_cap_bytes() / 2)
        .expect("shrink claim");
    let second = budget
        .claim(batch_execution_cap_bytes() / 2)
        .expect("released half is immediately admissible");
    drop((claim, second));
    assert_eq!(budget.lock_state().live, 0);
}

#[test]
fn shared_baseline_is_subtracted_from_each_initial_worker_claim() {
    let shared = super::super::allocation::GENERIC_WORKER_CLAIM_BYTES / 8;
    let worker = super::super::allocation::GENERIC_WORKER_CLAIM_BYTES - shared;
    let budget = BatchAllocationBudget::with_baseline(shared).expect("shared-plan budget");
    let claims = (0..super::super::allocation::generic_batch_worker_limit())
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
fn claim_that_cannot_fit_beside_the_baseline_fails_without_waiting() {
    let baseline = super::super::allocation::GENERIC_WORKER_CLAIM_BYTES / 8;
    let budget = BatchAllocationBudget::with_baseline(baseline).expect("shared-plan budget");
    let admissible = batch_execution_cap_bytes() - baseline;
    let claiming = Arc::clone(&budget);
    let (sender, receiver) = std::sync::mpsc::channel();
    let claimant = std::thread::spawn(move || {
        // Fits the cap, but never beside the permanent baseline.
        let result = claiming.claim(admissible + 1).map(drop);
        sender.send(result).expect("report the claim result");
    });
    let result = receiver
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap_or_else(|_| {
            panic!(
                "claim blocked; registered waiters: {}",
                budget.lock_state().waiters
            )
        });
    claimant.join().expect("claimant joined");
    assert!(matches!(
        result,
        Err(BatchInfrastructureError::AllocationTooLarge { requested, cap, .. })
            if requested == admissible + 1 && cap == admissible
    ));
    assert_eq!(budget.lock_state().waiters, 0, "the claim never waited");
    let largest = budget.claim(admissible).expect("largest admissible claim");
    drop(largest);
    assert_eq!(budget.lock_state().live, baseline);
}

#[test]
fn waiting_claim_resumes_after_an_owner_reconciles() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    let mut owners = (0..super::super::allocation::generic_batch_worker_limit())
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
            .claim(batch_execution_cap_bytes())
            .expect("full claim");
        panic!("unwind admitted owner");
    })
    .is_err());
    let replacement = budget
        .claim(batch_execution_cap_bytes())
        .expect("unwound claim was released");
    drop(replacement);
    assert_eq!(budget.lock_state().live, 0);
}

#[test]
fn failed_reconciliation_releases_the_original_claim_on_drop() {
    let budget = BatchAllocationBudget::with_baseline(0).expect("budget");
    {
        let mut claim = budget
            .claim(batch_execution_cap_bytes())
            .expect("full claim");
        assert!(matches!(
            claim.reconcile(batch_execution_cap_bytes() + 1),
            Err(BatchInfrastructureError::AllocationTooLarge { .. })
        ));
    }
    let replacement = budget
        .claim(batch_execution_cap_bytes())
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
