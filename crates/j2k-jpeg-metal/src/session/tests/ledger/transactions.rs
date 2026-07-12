// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn transactional_queue_growth_rejects_live_old_plus_new_without_mutating_session() {
    let mut session = SessionState::default();
    let plan = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("cached fixture plan");
    let repeated = ResolvedJpegPlan {
        input: plan.input.clone(),
        fast_packet: plan.fast_packet.clone(),
        shape: plan.shape,
    };
    session
        .queue_request(request_from_plan(plan))
        .expect("initial request");
    session.queued.shrink_to_fit();
    session.completed.shrink_to_fit();
    assert_eq!(session.queued.len(), session.queued.capacity());
    assert_eq!(session.completed.len(), session.completed.capacity());

    let projected_queued = projected_push_capacity(
        session.queued.len(),
        session.queued.capacity(),
        "test queued capacity",
    )
    .expect("projected queued capacity");
    let projected_completed = projected_push_capacity(
        session.completed.len(),
        session.completed.capacity(),
        "test completed capacity",
    )
    .expect("projected completed capacity");
    let final_without_submissions = session
        .collective_queue_state_bytes(
            session.queued_plan_ledger.retained_bytes(),
            0,
            projected_queued,
            projected_completed,
        )
        .expect("final projected bytes");
    let submission_size = std::mem::size_of::<crate::batch::MetalSubmission>();
    let retained_submission_capacity =
        (DEFAULT_MAX_HOST_ALLOCATION_BYTES - final_without_submissions) / submission_size;
    let retained_submission_bytes =
        submission_capacity_bytes(retained_submission_capacity).expect("submission bytes");
    let final_bytes = final_without_submissions + retained_submission_bytes;
    let old_metadata_bytes = session
        .session_metadata_live_bytes()
        .expect("old metadata bytes");
    assert!(final_bytes <= DEFAULT_MAX_HOST_ALLOCATION_BYTES);
    assert!(final_bytes + old_metadata_bytes > DEFAULT_MAX_HOST_ALLOCATION_BYTES);

    let before = (
        session.queued.as_ptr(),
        session.queued.len(),
        session.queued.capacity(),
        session.completed.as_ptr(),
        session.completed.len(),
        session.completed.capacity(),
        session.queued_plan_ledger.retained_bytes(),
        session.retained_execution_metadata_bytes,
        session.peak_collective_host_bytes,
    );
    let error = session
        .queue_request_with_retained(request_from_plan(repeated), retained_submission_capacity)
        .expect_err("old plus replacement allocations exceed the cap");
    assert!(matches!(
        error,
        Error::BatchInfrastructure(j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal transactional queue growth",
            ..
        })
    ));
    assert_eq!(
        before,
        (
            session.queued.as_ptr(),
            session.queued.len(),
            session.queued.capacity(),
            session.completed.as_ptr(),
            session.completed.len(),
            session.completed.capacity(),
            session.queued_plan_ledger.retained_bytes(),
            session.retained_execution_metadata_bytes,
            session.peak_collective_host_bytes,
        )
    );
}

#[test]
fn transactional_queue_growth_rolls_back_injected_allocator_overcapacity() {
    let mut session = SessionState::default();
    let plan = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("cached fixture plan");
    let repeated = ResolvedJpegPlan {
        input: plan.input.clone(),
        fast_packet: plan.fast_packet.clone(),
        shape: plan.shape,
    };
    session
        .queue_request(request_from_plan(plan))
        .expect("initial request");
    session.queued.shrink_to_fit();
    session.completed.shrink_to_fit();
    session.queue_growth_capacity_override = Some((usize::MAX, 0));

    let before = (
        session.queued.as_ptr(),
        session.queued.len(),
        session.queued.capacity(),
        session.completed.as_ptr(),
        session.completed.len(),
        session.completed.capacity(),
        session.queued_plan_ledger.retained_bytes(),
        session.retained_execution_metadata_bytes,
        session.peak_collective_host_bytes,
    );
    let error = session
        .queue_request(request_from_plan(repeated))
        .expect_err("injected allocator overcapacity must fail admission");
    assert!(matches!(
        error,
        Error::BatchInfrastructure(j2k_core::BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG Metal transactional queue growth",
            ..
        })
    ));
    assert_eq!(
        before,
        (
            session.queued.as_ptr(),
            session.queued.len(),
            session.queued.capacity(),
            session.completed.as_ptr(),
            session.completed.len(),
            session.completed.capacity(),
            session.queued_plan_ledger.retained_bytes(),
            session.retained_execution_metadata_bytes,
            session.peak_collective_host_bytes,
        )
    );
}
