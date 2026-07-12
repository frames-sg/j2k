// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_core::DeviceSubmission;

#[test]
fn completed_host_outputs_compose_at_exact_cap_and_reject_one_over_transactionally() {
    let first = host_surface_with_capacity(32);
    let second = host_surface_with_capacity(48);
    let first_bytes = first.retained_host_capacity_bytes();
    let second_bytes = second.retained_host_capacity_bytes();
    let exact_external = DEFAULT_MAX_HOST_ALLOCATION_BYTES - first_bytes - second_bytes;
    let mut exact = SessionState {
        completed: vec![None, None],
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
        completed: vec![None, None],
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
        completed: vec![None],
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
            slot,
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
