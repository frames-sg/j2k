// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn empty_batch_plan_is_a_typed_infrastructure_error() {
    assert_eq!(
        select_batch_plan(0, 1),
        Err(BatchInfrastructureError::EmptyBatchPlan)
    );
}

#[test]
fn exact_aggregate_cap_is_accepted_and_one_over_is_rejected() {
    let (metadata, _) = batch_metadata_bytes(1, 1).expect("small metadata");
    let exact_cap = GENERIC_WORKER_CLAIM_BYTES + metadata;
    let plan = select_batch_plan_with_limits(1, 1, GENERIC_WORKER_CLAIM_BYTES, metadata, exact_cap)
        .expect("exact aggregate boundary");
    assert_eq!(plan.live_bytes, exact_cap);

    let error =
        select_batch_plan_with_limits(1, 1, GENERIC_WORKER_CLAIM_BYTES, metadata, exact_cap - 1)
            .expect_err("one byte above aggregate cap");
    assert!(matches!(
        error,
        BatchInfrastructureError::AllocationTooLarge {
            requested,
            cap,
            ..
        } if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn requested_worker_count_reduces_to_fit_aggregate_claims() {
    let (two_worker_metadata, _) = batch_metadata_bytes(4, 2).expect("small metadata");
    let two_worker_cap = GENERIC_WORKER_CLAIM_BYTES
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(two_worker_metadata))
        .expect("test cap");
    let plan = select_batch_plan_with_limits(
        4,
        4,
        GENERIC_WORKER_CLAIM_BYTES,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        two_worker_cap,
    )
    .expect("two workers fit");
    assert_eq!(plan.worker_count, 2);
    assert_eq!(plan.chunk_size, 2);
}

#[test]
fn one_worker_rejects_when_claim_and_metadata_cannot_fit() {
    let (metadata, _) = batch_metadata_bytes(1, 1).expect("small metadata");
    let cap = GENERIC_WORKER_CLAIM_BYTES + metadata - 1;
    assert!(matches!(
        select_batch_plan_with_limits(1, 1, GENERIC_WORKER_CLAIM_BYTES, metadata, cap,),
        Err(BatchInfrastructureError::AllocationTooLarge { .. })
    ));
}

#[test]
fn metadata_one_over_is_public_infrastructure_error() {
    let (metadata, _) = batch_metadata_bytes(1, 1).expect("small metadata");
    let metadata_cap = metadata - 1;
    let infrastructure =
        select_batch_plan_with_limits(1, 1, GENERIC_WORKER_CLAIM_BYTES, metadata_cap, usize::MAX)
            .expect_err("metadata is one byte over policy");
    let public_error: super::super::TileBatchError = infrastructure.into();
    assert!(matches!(
        public_error,
        super::super::TileBatchError::Infrastructure(
            BatchInfrastructureError::AllocationTooLarge {
                what: "J2K batch metadata",
                requested,
                cap,
            }
        ) if requested == metadata && cap == metadata_cap
    ));
}

#[test]
fn aggregate_overflow_is_a_typed_cap_failure() {
    assert!(matches!(
        select_batch_plan_with_limits(1, 1, usize::MAX, usize::MAX, usize::MAX),
        Err(BatchInfrastructureError::AllocationTooLarge {
            requested: usize::MAX,
            ..
        })
    ));
}

#[test]
fn planning_claims_do_not_allocate_worker_workspace() {
    let jobs = generic_batch_worker_limit() * 2;
    let plan = select_batch_plan(jobs, usize::MAX).expect("bounded arithmetic plan");
    assert_eq!(plan.worker_count, generic_batch_worker_limit());
    assert!(plan.live_bytes <= batch_host_cap_bytes());
}

#[test]
fn generic_worker_claims_are_bounded_by_the_given_limit() {
    let plan = select_batch_plan_with_worker_limit(
        64,
        64,
        GENERIC_WORKER_CLAIM_BYTES,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        usize::MAX,
        4,
    )
    .expect("four claims fit");
    assert_eq!(plan.worker_count, 4);
    assert_eq!(plan.chunk_size, 16);
}

#[test]
fn dynamically_admitted_direct_batch_is_not_clamped_to_generic_workers() {
    let plan = select_direct_batch_plan(16, 12).expect("bounded direct metadata plan");

    // The scheduler's chunking contract turns 12 desired workers over 16 jobs
    // into eight two-job workers. Aggregate direct allocations are admitted at
    // runtime, so the fixed four-generic-worker ceiling must not truncate this
    // metadata-only plan.
    assert_eq!(plan.worker_count, 8);
    assert_eq!(plan.chunk_size, 2);
    assert!(plan.metadata_bytes <= J2K_BATCH_METADATA_ALLOWANCE_BYTES);
}

#[test]
fn dynamically_admitted_workers_remain_structurally_bounded() {
    let jobs = admitted_batch_worker_limit() * 8;
    let plan = select_direct_batch_plan(jobs, jobs).expect("bounded direct metadata plan");

    assert_eq!(plan.worker_count, admitted_batch_worker_limit());
    assert_eq!(plan.chunk_size, 8);
}
