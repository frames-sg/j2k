// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

use crate::{Error, MetalSession};

use super::execute::complete_repeated_device_failure;
use super::heuristics::{
    can_decode_requests_as_repeated_full_color_batch,
    can_decode_requests_as_repeated_full_grayscale_batch, group_metal_requests,
    profile_route_label, same_input_bytes, BatchRoute,
};
use super::request::{BatchOp, QueuedRequest};
use super::session::{queue_tile_request_shared, release_surface_slot, SessionState};

fn auto_rgb_region_scaled_request(input: Arc<[u8]>) -> QueuedRequest {
    QueuedRequest::new(
        input,
        PixelFormat::Rgb8,
        BackendRequest::Auto,
        BatchOp::RegionScaled {
            roi: Rect {
                x: 128,
                y: 128,
                w: 512,
                h: 256,
            },
            scale: Downscale::Quarter,
        },
        0,
    )
}

fn auto_full_request(input: Arc<[u8]>, fmt: PixelFormat) -> QueuedRequest {
    QueuedRequest::new(input, fmt, BackendRequest::Auto, BatchOp::Full, 0)
}

#[test]
fn auto_repeated_full_candidates_are_limited_to_measured_formats() {
    let shared = Arc::<[u8]>::from([1_u8]);
    let requests = |fmt| {
        vec![
            auto_full_request(shared.clone(), fmt),
            auto_full_request(shared.clone(), fmt),
        ]
    };

    assert!(can_decode_requests_as_repeated_full_color_batch(&requests(
        PixelFormat::Rgb8
    )));
    assert!(!can_decode_requests_as_repeated_full_color_batch(
        &requests(PixelFormat::Rgba8)
    ));
    assert!(can_decode_requests_as_repeated_full_grayscale_batch(
        &requests(PixelFormat::Gray8)
    ));
    assert!(!can_decode_requests_as_repeated_full_grayscale_batch(
        &requests(PixelFormat::Gray16)
    ));
}

#[test]
fn auto_region_scaled_distinct_inputs_stay_on_cpu_without_promotion_evidence() {
    let requests = (0_u8..16)
        .map(|idx| auto_rgb_region_scaled_request(Arc::from([idx])))
        .collect::<Vec<_>>();

    let grouped = group_metal_requests(requests).expect("group requests");
    assert_eq!(grouped.len(), 1);
    assert_eq!(grouped[0].route, BatchRoute::AutoRegionScaledDirectCpu);
}

#[test]
fn repeated_input_check_uses_pointer_identity_before_fingerprint() {
    let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
    let first = auto_rgb_region_scaled_request(shared.clone());
    let next = auto_rgb_region_scaled_request(shared);

    assert!(same_input_bytes(&first, &next));
    assert!(!first.input_fingerprint_cache_filled_for_test());
    assert!(!next.input_fingerprint_cache_filled_for_test());
}

#[test]
fn dropping_an_unwaited_submission_releases_and_reuses_its_session_slot() {
    let mut session = MetalSession::default();
    let submission = queue_tile_request_shared(
        &mut session,
        Arc::<[u8]>::from([0xff_u8, 0x4f]),
        PixelFormat::Gray8,
        BackendRequest::Cpu,
        BatchOp::Full,
    )
    .expect("queue submission");
    let first_slot = submission.slot.expect("active submission slot");
    assert_eq!(session.shared.lock().expect("session").queued.len(), 1);

    drop(submission);

    let state = session.shared.lock().expect("session");
    assert!(state.queued.is_empty());
    assert_eq!(state.free_slots, [first_slot]);
    drop(state);

    let next = queue_tile_request_shared(
        &mut session,
        Arc::<[u8]>::from([0xff_u8, 0x4f]),
        PixelFormat::Gray8,
        BackendRequest::Cpu,
        BatchOp::Full,
    )
    .expect("reuse submission slot");
    assert_eq!(next.slot, Some(first_slot));
}

#[test]
fn slot_release_reports_missing_reserved_capacity_without_panicking() {
    let mut state = SessionState::default();

    assert!(matches!(
        release_surface_slot(&mut state, 0),
        Err(Error::MetalStateInvariant {
            state: "J2K Metal batch free-slot ledger",
            ..
        })
    ));
    assert!(state.free_slots.is_empty());
}

#[test]
fn selected_repeated_device_failure_is_reported_without_cpu_retry() {
    let shared = Arc::<[u8]>::from([1_u8]);
    let requests = (0..2)
        .map(|slot| {
            let mut request = auto_full_request(shared.clone(), PixelFormat::Rgb8);
            request.output_slot = slot;
            request
        })
        .collect::<Vec<_>>();
    let mut session = SessionState {
        submissions: 0,
        queued: Vec::new(),
        completed: (0..requests.len()).map(|_| None).collect(),
        free_slots: Vec::new(),
    };

    complete_repeated_device_failure(
        &mut session,
        &requests,
        &Error::MetalKernel {
            message: "synthetic dispatch failure".to_string(),
        },
    );

    assert_eq!(session.submissions, 1);
    assert!(session.completed.iter().all(|result| {
        matches!(
            result,
            Some(Err(Error::MetalRuntime { message }))
                if message.contains("synthetic dispatch failure")
        )
    }));
}

#[test]
fn auto_region_scaled_grouping_keeps_repeated_rgb_on_cpu_without_evidence() {
    let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
    let requests = (0..2)
        .map(|_| auto_rgb_region_scaled_request(shared.clone()))
        .collect::<Vec<_>>();

    let grouped = group_metal_requests(requests).expect("group requests");

    assert_eq!(grouped.len(), 1);
    assert_eq!(grouped[0].route, BatchRoute::AutoRegionScaledDirectCpu);
    assert_eq!(grouped[0].requests.len(), 2);
    assert!(
        grouped[0]
            .requests
            .iter()
            .all(|request| !request.input_fingerprint_cache_filled_for_test()),
        "shared repeated inputs should be classified by Arc identity without fingerprinting"
    );
}

#[test]
fn auto_region_scaled_distinct_rgb_grouping_preserves_cpu_decision() {
    let requests = (0_u8..16)
        .map(|idx| auto_rgb_region_scaled_request(Arc::from([idx])))
        .collect::<Vec<_>>();

    let grouped = group_metal_requests(requests).expect("group requests");

    assert_eq!(grouped.len(), 1);
    assert_eq!(grouped[0].route, BatchRoute::AutoRegionScaledDirectCpu);
    assert_eq!(grouped[0].requests.len(), 16);
}

#[test]
fn profile_route_labels_are_stable_for_decode_batch_slices() {
    assert_eq!(profile_route_label(BatchRoute::Generic), "generic");
    assert_eq!(
        profile_route_label(BatchRoute::AutoRegionScaledDirectCpu),
        "auto_region_scaled_direct_cpu"
    );
}
