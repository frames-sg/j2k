// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

use crate::{Error, MetalSession};

#[cfg(target_os = "macos")]
use super::execute::process_batch;
#[cfg(target_os = "macos")]
use super::heuristics::GroupedRequests;
use super::heuristics::{
    auto_region_scaled_direct_metal_min_dim, can_decode_requests_as_repeated_region_scaled_batch,
    group_metal_requests, profile_route_label, same_input_bytes, BatchRoute,
    AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT, AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM,
    AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT,
    AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
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

fn auto_rgb_region_scaled_request_with_max_dim(
    input: Arc<[u8]>,
    max_image_dim: u32,
) -> QueuedRequest {
    let request = auto_rgb_region_scaled_request(input);
    request.max_image_dim.set(Some(max_image_dim)).ok();
    request
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "bounded test fixture index fits in u8"
)]
fn auto_region_scaled_rgb_threshold_requires_repeated_inputs() {
    let requests = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
        .map(|idx| auto_rgb_region_scaled_request(Arc::from([idx as u8])))
        .collect::<Vec<_>>();

    assert!(!can_decode_requests_as_repeated_region_scaled_batch(
        &requests
    ));
    assert_eq!(
        auto_region_scaled_direct_metal_min_dim(&requests),
        None,
        "distinct RGB ROI+scaled Auto batches must stay CPU until hybrid wins for distinct inputs"
    );

    let shared = Arc::<[u8]>::from([1_u8]);
    let repeated = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
        .map(|_| auto_rgb_region_scaled_request(shared.clone()))
        .collect::<Vec<_>>();
    assert!(can_decode_requests_as_repeated_region_scaled_batch(
        &repeated
    ));
}

#[test]
fn auto_region_scaled_repeated_rgb_uses_measured_batch_two_metal_threshold() {
    let shared = Arc::<[u8]>::from([1_u8]);
    let repeated = (0..2)
        .map(|_| auto_rgb_region_scaled_request_with_max_dim(shared.clone(), 512))
        .collect::<Vec<_>>();

    assert_eq!(
        auto_region_scaled_direct_metal_min_dim(&repeated),
        Some(512),
        "measured repeated RGB ROI+scaled batches should route to Metal from batch 2 at 512px"
    );

    let single = vec![auto_rgb_region_scaled_request_with_max_dim(shared, 512)];
    assert_eq!(auto_region_scaled_direct_metal_min_dim(&single), None);
}

#[test]
fn queued_request_caches_image_dimension_probe() {
    let request = auto_rgb_region_scaled_request(Arc::from([0_u8]));

    assert!(!request.max_image_dim_cache_filled_for_test());
    assert_eq!(request.max_image_dim(), None);
    assert!(request.max_image_dim_cache_filled_for_test());
    assert_eq!(request.max_image_dim(), None);
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
fn auto_region_scaled_grouping_preserves_repeated_rgb_metal_decision() {
    let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
    let requests = (0..AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT)
        .map(|_| {
            auto_rgb_region_scaled_request_with_max_dim(
                shared.clone(),
                AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
            )
        })
        .collect::<Vec<_>>();

    let grouped = group_metal_requests(requests).expect("group requests");

    assert_eq!(grouped.len(), 1);
    assert_eq!(
        grouped[0].route,
        BatchRoute::AutoRepeatedRegionScaledDirectMetal
    );
    assert_eq!(
        grouped[0].requests.len(),
        AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT
    );
    assert!(
        grouped[0]
            .requests
            .iter()
            .all(|request| !request.input_fingerprint_cache_filled_for_test()),
        "shared repeated inputs should be classified by Arc identity without fingerprinting"
    );
}

#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "bounded test fixture index fits in u8"
)]
fn auto_region_scaled_distinct_rgb_grouping_preserves_cpu_decision() {
    let requests = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
        .map(|idx| {
            auto_rgb_region_scaled_request_with_max_dim(
                Arc::from([idx as u8]),
                AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM,
            )
        })
        .collect::<Vec<_>>();

    let grouped = group_metal_requests(requests).expect("group requests");

    assert_eq!(grouped.len(), 1);
    assert_eq!(grouped[0].route, BatchRoute::AutoRegionScaledDirectCpu);
    assert_eq!(
        grouped[0].requests.len(),
        AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT
    );
}

#[test]
fn profile_route_labels_are_stable_for_decode_batch_slices() {
    assert_eq!(profile_route_label(BatchRoute::Generic), "generic");
    assert_eq!(
        profile_route_label(BatchRoute::AutoRegionScaledDirectCpu),
        "auto_region_scaled_direct_cpu"
    );
    assert_eq!(
        profile_route_label(BatchRoute::AutoRegionScaledDirectMetal),
        "auto_region_scaled_direct_metal"
    );
    assert_eq!(
        profile_route_label(BatchRoute::AutoRepeatedRegionScaledDirectMetal),
        "auto_repeated_region_scaled_direct_metal"
    );
}

#[cfg(target_os = "macos")]
#[test]
fn auto_region_scaled_prechecked_error_does_not_retry_generic_direct_path() {
    let _guard = crate::hybrid::region_scaled_color_plan_test_lock_for_test();
    crate::hybrid::reset_region_scaled_color_plan_builds_for_test();
    let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
    let requests = (0..AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT)
        .map(|slot| {
            let mut request = auto_rgb_region_scaled_request_with_max_dim(
                shared.clone(),
                AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
            );
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

    process_batch(
        &mut session,
        GroupedRequests {
            route: BatchRoute::AutoRepeatedRegionScaledDirectMetal,
            requests,
        },
        None,
    );

    assert_eq!(
        crate::hybrid::region_scaled_color_plan_builds_for_test(),
        1,
        "failed prechecked Auto Metal routing should fall back to CPU without retrying generic direct Metal"
    );
    assert!(
        session
            .completed
            .iter()
            .all(|result| matches!(result, Some(Err(_)))),
        "invalid inputs should be surfaced on every fallback request"
    );
}
