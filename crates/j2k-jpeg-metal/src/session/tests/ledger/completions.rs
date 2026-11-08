// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_core::DeviceSubmission;

fn pending_completions(count: usize) -> j2k_metal_support::CompletionSlots<Result<Surface, Error>> {
    let mut slots = j2k_metal_support::CompletionSlots::default();
    for _ in 0..count {
        slots
            .try_reserve_slot("test pending completions")
            .expect("reserve completion");
        slots.allocate().expect("pending completion");
    }
    slots
}

#[test]
fn completed_host_outputs_compose_at_exact_cap_and_reject_one_over_transactionally() {
    let first = host_surface_with_capacity(32);
    let second = host_surface_with_capacity(48);
    let first_bytes = first.retained_host_capacity_bytes();
    let second_bytes = second.retained_host_capacity_bytes();
    let exact_external = DEFAULT_MAX_HOST_ALLOCATION_BYTES - first_bytes - second_bytes;
    let mut exact = SessionState {
        completed: pending_completions(2),
        ..SessionState::default()
    };
    exact
        .store_completed_result(0, Ok(first), exact_external, 0)
        .expect("first exact-cap host completion");
    exact
        .store_completed_result(1, Ok(second), exact_external, 0)
        .expect("two host completions at exact cap");
    assert_eq!(exact.completed_host_bytes(), first_bytes + second_bytes);

    let first = host_surface_with_capacity(32);
    let second = host_surface_with_capacity(48);
    let first_bytes = first.retained_host_capacity_bytes();
    let second_bytes = second.retained_host_capacity_bytes();
    let one_over_external = DEFAULT_MAX_HOST_ALLOCATION_BYTES - first_bytes - second_bytes + 1;
    let mut over = SessionState {
        completed: pending_completions(2),
        ..SessionState::default()
    };
    over.store_completed_result(0, Ok(first), one_over_external, 0)
        .expect("first host completion remains below cap");
    let error = over
        .store_completed_result(1, Ok(second), one_over_external, 0)
        .expect_err("second host completion exceeds cap by one byte");
    assert!(matches!(
        error,
        Error::BatchInfrastructure(
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal completed host surface retention",
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
        ) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
    ));
    assert_eq!(over.completed_host_bytes(), first_bytes);
    assert!(over.completed[0].is_some());
    assert!(over.completed[1].is_none());
}

#[test]
fn queue_and_execution_stamp_include_an_earlier_retained_host_completion() {
    let mut session = SessionState {
        completed: pending_completions(1),
        ..SessionState::default()
    };
    let surface = host_surface_with_capacity(4096);
    let host_bytes = surface.retained_host_capacity_bytes();
    session
        .store_completed_result(0, Ok(surface), 0, 0)
        .expect("retained host completion");
    let plan = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("plan resolution with retained completion");
    session
        .queue_request(request_from_plan(plan))
        .expect("queue with retained completion");
    let expected_external = session
        .session_metadata_live_bytes()
        .expect("session metadata including completion");
    assert!(expected_external >= host_bytes);

    let queued = session
        .take_queued_requests()
        .expect("execution baseline with retained completion");
    assert_eq!(queued[0].execution_external_live_bytes(), expected_external);
}

#[test]
fn fully_waited_cpu_session_reuses_bounded_completion_state() {
    let shared = SharedSession::default();
    let mut warmed_peak = None;
    for _ in 0..32 {
        let slot = {
            let mut state = shared.lock().expect("session lock");
            let plan = state
                .resolve_jpeg_plan(BASELINE_420, BackendRequest::Cpu)
                .expect("CPU plan");
            state
                .queue_request(crate::batch::QueuedRequest::new_shared(
                    plan.input,
                    PixelFormat::Rgb8,
                    BackendRequest::Cpu,
                    crate::batch::BatchOp::Full,
                    None,
                    plan.shape,
                ))
                .expect("CPU request")
        };
        let surface = crate::batch::MetalSubmission {
            session: shared.clone(),
            slot: Some(slot),
        }
        .wait()
        .expect("CPU completion");
        assert_eq!(surface.retained_host_capacity_bytes(), 16 * 16 * 3);

        let state = shared.lock().expect("session lock after wait");
        assert!(state.queued.is_empty());
        assert!(state.completed.is_empty());
        assert_eq!(state.completed_host_bytes(), 0);
        let peak = state.peak_collective_host_bytes();
        if let Some(warmed_peak) = warmed_peak {
            assert_eq!(peak, warmed_peak);
        } else {
            warmed_peak = Some(peak);
        }
    }
}

fn cpu_submission(
    shared: &SharedSession,
    input: std::sync::Arc<[u8]>,
) -> crate::batch::MetalSubmission {
    let mut state = shared.lock().expect("session lock");
    let plan = state
        .resolve_arc_jpeg_plan_with_external_live(input, BackendRequest::Cpu, 0)
        .expect("CPU input owner");
    let slot = state
        .queue_request(crate::batch::QueuedRequest::new_shared(
            plan.input,
            PixelFormat::Rgb8,
            BackendRequest::Cpu,
            crate::batch::BatchOp::Full,
            None,
            plan.shape,
        ))
        .expect("CPU request");
    crate::batch::MetalSubmission {
        session: shared.clone(),
        slot: Some(slot),
    }
}

#[test]
fn abandoned_submission_cancels_queue_and_reuses_slot_without_retaining_input() {
    let shared = SharedSession::default();
    let input = std::sync::Arc::<[u8]>::from(BASELINE_420);
    let weak = std::sync::Arc::downgrade(&input);
    let submission = cpu_submission(&shared, input);
    let slot = submission.slot;
    drop(submission);
    let state = shared.lock().expect("session after cancellation");
    assert!(state.queued.is_empty(), "dropped submissions never execute");
    assert!(state.completed.is_empty());
    assert_eq!(state.queued_plan_ledger.retained_bytes(), 0);
    assert_eq!(state.submissions, 0);
    assert!(
        weak.upgrade().is_none(),
        "cancelled input owner is released"
    );
    drop(state);
    let next = cpu_submission(&shared, std::sync::Arc::from(BASELINE_420));
    assert_eq!(next.slot, slot);
    // Keep a second request sharing the same owners while cancelling one.
    let (other_slot, owner_bytes, capacity) = {
        let mut state = shared.lock().expect("shared pending owners");
        let request = state.queued[0].clone();
        let other_slot = state
            .queue_request(request)
            .expect("shared pending request");
        (
            other_slot,
            state.queued_plan_ledger.retained_bytes(),
            state.completed.capacity(),
        )
    };
    let other = crate::batch::MetalSubmission {
        session: shared.clone(),
        slot: Some(other_slot),
    };
    drop(next);
    {
        let state = shared.lock().expect("one shared owner remains");
        assert_eq!(state.queued.len(), 1);
        assert_eq!(state.queued_plan_ledger.retained_bytes(), owner_bytes);
    }
    let reused = cpu_submission(&shared, std::sync::Arc::from(BASELINE_420));
    assert_eq!(
        reused.slot, slot,
        "reuse a hole while another request remains active"
    );
    assert_eq!(
        shared.lock().expect("reused slot").completed.capacity(),
        capacity
    );
    drop(other);
    reused.wait().expect("reused slot completes");
}

#[test]
fn abandoned_completion_releases_host_owner_and_host_budget() {
    let shared = SharedSession::default();
    let first = cpu_submission(&shared, std::sync::Arc::from(BASELINE_420));
    let abandoned = cpu_submission(&shared, std::sync::Arc::from(BASELINE_420));
    first.wait().expect("flush both requests");
    let weak = {
        let state = shared.lock().expect("completed session");
        let surface = state.completed[abandoned.slot.expect("active slot")]
            .as_ref()
            .expect("completion")
            .as_ref()
            .expect("surface");
        let Storage::Host(bytes) = &surface.storage else {
            panic!("CPU output")
        };
        assert_eq!(state.completed_host_bytes(), bytes.capacity());
        std::sync::Arc::downgrade(bytes)
    };
    drop(abandoned);
    let state = shared.lock().expect("session after abandoned completion");
    assert_eq!(state.completed_host_bytes(), 0);
    assert!(state.completed.is_empty());
    assert!(
        weak.upgrade().is_none(),
        "unreachable completed surface is released"
    );
}

#[test]
fn early_collection_error_releases_unconsumed_completions() {
    let shared = SharedSession::default();
    let bad = cpu_submission(&shared, std::sync::Arc::from([0_u8]));
    let good = cpu_submission(&shared, std::sync::Arc::from(BASELINE_420));
    let result = j2k_metal_support::FallibleSubmissionQueue::from_retained(vec![bad, good])
        .try_finish(
            "test completion collection",
            "test completion outputs",
            DeviceSubmission::wait,
        );
    assert!(result.is_err());
    let state = shared.lock().expect("session after failed collection");
    assert!(state.queued.is_empty());
    assert!(
        state.completed.is_empty(),
        "remaining submission handles release completed outputs"
    );
    assert_eq!(state.completed_host_bytes(), 0);
    assert_eq!(state.queued_plan_ledger.retained_bytes(), 0);
}

#[test]
fn abandoned_metal_completion_releases_its_device_buffer() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let device = j2k_metal_support::system_default_device().expect("required Metal device");
    let shared = SharedSession::default();
    let (submission, weak) = objc2::rc::autoreleasepool(|_| {
        let buffer =
            j2k_metal_support::checked_shared_buffer(&device, 64).expect("Metal completion buffer");
        let weak = objc2::rc::Weak::from_retained(&buffer);
        let surface = Surface::from_metal_buffer(buffer, (8, 8), PixelFormat::Gray8)
            .expect("completed Metal surface");
        let mut state = shared.lock().expect("completion state");
        state
            .completed
            .try_reserve_slot("test Metal completion")
            .expect("reserve completion");
        let slot = state.completed.allocate().expect("active completion");
        state
            .store_completed_result(slot, Ok(surface), 0, 0)
            .expect("store Metal completion");
        (
            crate::batch::MetalSubmission {
                session: shared.clone(),
                slot: Some(slot),
            },
            weak,
        )
    });
    assert!(weak.load().is_some());
    drop(submission);
    assert!(shared
        .lock()
        .expect("released Metal state")
        .completed
        .is_empty());
    assert!(
        weak.load().is_none(),
        "the abandoned completion was the device buffer's last owner"
    );
}
