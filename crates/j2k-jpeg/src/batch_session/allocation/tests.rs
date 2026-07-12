// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn decode_jobs(claims: &[usize]) -> Vec<PlannedJob> {
    claims
        .iter()
        .map(|&worker_live_bytes| PlannedJob::Decode {
            worker_live_bytes,
            retained_result_bytes: 0,
        })
        .collect()
}

fn zero_metadata() -> BatchMetadataLayout {
    BatchMetadataLayout {
        fixed_bytes: 0,
        worker_slot_capacity: 0,
        worker_slot_bytes: 0,
        worker_result_bytes: 0,
        ordered_result_bytes: 0,
        handle_bytes: 0,
    }
}

#[test]
fn empty_job_plan_is_a_typed_infrastructure_error() {
    let error = select_batch_plan_with_limits(&[], 1, zero_metadata(), |_| 0, limits(0, 0, 0))
        .expect_err("empty batches must not reach division-based planning");

    assert_eq!(error, BatchInfrastructureError::EmptyBatchPlan);
}

const fn limits(metadata: usize, codec: usize, aggregate: usize) -> BatchPlanLimits {
    BatchPlanLimits {
        metadata,
        codec,
        aggregate,
    }
}

#[test]
fn exact_cap_is_accepted_and_one_over_is_rejected() {
    let jobs = decode_jobs(&[80]);
    let plan = select_batch_plan_with_limits(&jobs, 1, zero_metadata(), |_| 0, limits(0, 80, 80))
        .expect("exact live cap");
    assert_eq!(plan.worker_count, 1);
    assert_eq!(plan.live_bytes, 80);
    assert_eq!(plan.codec_bytes, 80);
    assert_eq!(plan.metadata_bytes, 0);

    let error = select_batch_plan_with_limits(&jobs, 1, zero_metadata(), |_| 0, limits(0, 79, 79))
        .expect_err("one byte over cap");
    assert!(matches!(
        error,
        BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG batch codec claims",
            requested: 80,
            cap: 79,
        }
    ));
}

#[test]
fn planner_reduces_concurrency_to_fit_aggregate_worker_claims() {
    let jobs = decode_jobs(&[60, 60, 60, 60]);
    let plan = select_batch_plan_with_limits(&jobs, 4, zero_metadata(), |_| 0, limits(0, 120, 120))
        .expect("two sequential worker chunks fit");
    assert_eq!(plan.worker_count, 2);
    assert_eq!(plan.chunk_size, 2);
    assert_eq!(plan.live_bytes, 120);
}

#[test]
fn stale_retained_worker_bytes_participate_in_the_next_plan() {
    let jobs = decode_jobs(&[20, 20]);
    let error = select_batch_plan_with_limits(
        &jobs,
        2,
        zero_metadata(),
        |worker| if worker == 0 { 80 } else { 0 },
        limits(0, 50, 50),
    )
    .expect_err("stale capacity prevents either worker layout");
    assert!(matches!(
        error,
        BatchInfrastructureError::AllocationTooLarge { cap: 50, .. }
    ));

    let plan = select_batch_plan_with_limits(&jobs, 2, zero_metadata(), |_| 0, limits(0, 50, 50))
        .expect("releasing stale capacity restores a valid plan");
    assert_eq!(plan.worker_count, 2);
    assert_eq!(plan.live_bytes, 40);
}

#[test]
fn overflow_is_a_cap_error_not_an_allocator_error() {
    let jobs = decode_jobs(&[usize::MAX]);
    let error = select_batch_plan_with_limits(
        &jobs,
        1,
        BatchMetadataLayout {
            fixed_bytes: 1,
            ..zero_metadata()
        },
        |_| 0,
        limits(usize::MAX, usize::MAX, usize::MAX),
    )
    .expect_err("aggregate overflow");
    assert!(matches!(
        error,
        BatchInfrastructureError::AllocationTooLarge {
            requested: usize::MAX,
            ..
        }
    ));
}

#[test]
fn overflowing_high_concurrency_candidate_can_reduce_to_a_fitting_worker() {
    let jobs = decode_jobs(&[0, 0]);
    let plan = select_batch_plan_with_limits(
        &jobs,
        2,
        BatchMetadataLayout {
            handle_bytes: usize::MAX,
            ..zero_metadata()
        },
        |_| 0,
        limits(usize::MAX, usize::MAX, usize::MAX),
    )
    .expect("single-worker handle metadata fits exactly");
    assert_eq!(plan.worker_count, 1);
    assert_eq!(plan.live_bytes, usize::MAX);
}

#[test]
fn allocator_failure_category_is_not_flattened_into_a_cap_error() {
    let error = host_allocation_error(
        "JPEG batch test vector",
        HostAllocationError::for_elements::<u64>(512),
    );
    assert_eq!(
        error,
        BatchInfrastructureError::HostAllocationFailed {
            what: "JPEG batch test vector",
            bytes: 4096,
        }
    );
}

#[test]
fn fallible_vector_rejects_over_cap_before_allocator_entry() {
    let count = JPEG_BATCH_METADATA_ALLOWANCE_BYTES / size_of::<u64>() + 1;
    let error = try_vec_with_capacity::<u64>(count, "JPEG batch test vector")
        .expect_err("one element over the vector cap");
    assert!(matches!(
        error,
        BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG batch test vector",
            requested,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        } if requested == count * size_of::<u64>()
    ));
}

#[test]
fn planning_phase_accepts_maximum_codec_plus_exact_metadata_and_rejects_one_over() {
    assert_eq!(
        ensure_planning_phase(JPEG_BATCH_METADATA_ALLOWANCE_BYTES).expect("exact planning domains"),
        JPEG_BATCH_HOST_CAP_BYTES
    );
    assert!(matches!(
        ensure_planning_phase(JPEG_BATCH_METADATA_ALLOWANCE_BYTES + 1),
        Err(BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG batch metadata",
            requested,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        }) if requested == JPEG_BATCH_METADATA_ALLOWANCE_BYTES + 1
    ));
}

#[test]
fn metadata_vectors_share_one_allowance_instead_of_individual_caps() {
    let exact_retained = JPEG_BATCH_METADATA_ALLOWANCE_BYTES - size_of::<u64>();
    assert_eq!(
        ensure_metadata_bytes(
            exact_retained,
            size_of::<u64>(),
            "JPEG shared metadata test",
        )
        .expect("one small vector reaches the exact shared boundary"),
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES
    );

    assert!(matches!(
        ensure_metadata_bytes(
            exact_retained + 1,
            size_of::<u64>(),
            "JPEG shared metadata test",
        ),
        Err(BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG shared metadata test",
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            ..
        })
    ));
}

#[test]
fn retained_metadata_and_actual_summary_share_one_exact_boundary() {
    let retained = JPEG_BATCH_METADATA_ALLOWANCE_BYTES - size_of::<usize>();
    assert_eq!(
        ensure_metadata_bytes(
            retained,
            size_of::<usize>(),
            "JPEG retained summary metadata",
        )
        .expect("exact retained plus summary boundary"),
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES
    );
    assert!(matches!(
        ensure_metadata_bytes(
            retained + 1,
            size_of::<usize>(),
            "JPEG retained summary metadata",
        ),
        Err(BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG retained summary metadata",
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            ..
        })
    ));
}
