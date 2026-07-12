// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn repeated_fully_shared_plan_is_charged_once_in_the_queue() {
    let mut session = SessionState::default();
    let plan = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("shared cached plan");
    let owner_bytes = plan_owner_bytes(&plan);
    let limit = session
        .jpeg_plan_cache_diagnostics()
        .retained_bytes
        .checked_add(owner_bytes)
        .expect("shared combined limit");
    session.queued_plan_ledger.set_host_byte_limit(limit);
    let repeated = ResolvedJpegPlan {
        input: plan.input.clone(),
        fast_packet: plan.fast_packet.clone(),
        shape: plan.shape,
    };

    session
        .queue_request(request_from_plan(plan))
        .expect("first shared request");
    session
        .queue_request(request_from_plan(repeated))
        .expect("repeated shared request");

    assert_eq!(session.queued.len(), 2);
    assert_eq!(session.queued_plan_ledger.retained_bytes(), owner_bytes);
}

#[test]
fn same_input_adds_a_later_distinct_packet_owner() {
    let input = SharedJpegInput::try_copy_from_slice(BASELINE_420).expect("shared fixture");
    let plan = JpegCachedPlan::build(input.clone()).expect("fixture cached plan");
    let packet = plan.fast_packet().cloned().expect("fixture packet");
    let shape = batch::BatchShape::from_summary(plan.batch_summary(), plan.color_space());
    let input_bytes = input.retained_cache_bytes().expect("input bytes");
    let packet_bytes = packet.retained_cache_bytes().expect("packet bytes");
    let mut session = SessionState::default();
    session
        .queued_plan_ledger
        .set_host_byte_limit(input_bytes + packet_bytes);

    session
        .queue_request(crate::batch::QueuedRequest::new_shared(
            input.clone(),
            PixelFormat::Rgb8,
            BackendRequest::Cpu,
            crate::batch::BatchOp::Full,
            None,
            batch::BatchShape::unknown(),
        ))
        .expect("input-only request");
    assert_eq!(session.queued_plan_ledger.retained_bytes(), input_bytes);

    session
        .queue_request(crate::batch::QueuedRequest::new_shared(
            input,
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            crate::batch::BatchOp::Full,
            Some(packet),
            shape,
        ))
        .expect("same input with a new packet owner");
    assert_eq!(
        session.queued_plan_ledger.retained_bytes(),
        input_bytes + packet_bytes
    );
}

#[test]
fn disabled_cache_charges_each_new_packet_arc_for_the_same_input() {
    let input = SharedJpegInput::try_copy_from_slice(BASELINE_420).expect("shared fixture");
    let mut session = SessionState {
        jpeg_plans: JpegPlanCache::with_limits(0, usize::MAX),
        ..SessionState::default()
    };
    let first = session
        .resolve_shared_jpeg_plan(input.clone(), BackendRequest::Metal)
        .expect("first disabled-cache plan");
    let second = session
        .resolve_shared_jpeg_plan(input, BackendRequest::Metal)
        .expect("second disabled-cache plan");
    assert!(SharedJpegInput::ptr_eq(&first.input, &second.input));
    assert!(!SharedJpegFastPacket::ptr_eq(
        first.fast_packet.as_ref().expect("first packet"),
        second.fast_packet.as_ref().expect("second packet"),
    ));
    let expected = first
        .input
        .retained_cache_bytes()
        .expect("shared input bytes")
        .checked_add(
            first
                .fast_packet
                .as_ref()
                .expect("first packet")
                .retained_cache_bytes()
                .expect("first packet bytes"),
        )
        .and_then(|bytes| {
            bytes.checked_add(
                second
                    .fast_packet
                    .as_ref()
                    .expect("second packet")
                    .retained_cache_bytes()
                    .expect("second packet bytes"),
            )
        })
        .expect("disabled-cache queued bytes");
    session.queued_plan_ledger.set_host_byte_limit(expected);

    session
        .queue_request(request_from_plan(first))
        .expect("first disabled-cache request");
    session
        .queue_request(request_from_plan(second))
        .expect("second disabled-cache request");

    assert_eq!(session.queued_plan_ledger.retained_bytes(), expected);
    assert_eq!(session.jpeg_plan_cache_diagnostics().entries, 0);
    assert_eq!(session.jpeg_plan_cache_diagnostics().disabled_rejections, 2);
}

#[test]
fn more_than_eight_distinct_queued_plans_remain_accounted_after_cache_eviction() {
    let mut session = SessionState::default();
    for suffix in 0_u8..12 {
        let mut input = BASELINE_420.to_vec();
        input.extend_from_slice(&[0, suffix]);
        let plan = session
            .resolve_jpeg_plan(&input, BackendRequest::Metal)
            .expect("distinct cached plan");
        session
            .queue_request(request_from_plan(plan))
            .expect("distinct queued plan");
    }

    assert_eq!(session.queued.len(), 12);
    assert_eq!(session.jpeg_plan_cache_diagnostics().entries, 8);
    assert_eq!(session.jpeg_plan_cache_diagnostics().evictions, 4);
    let expected = session.queued.iter().fold(0_usize, |bytes, request| {
        bytes
            .checked_add(request.retained_input_bytes().expect("queued input bytes"))
            .and_then(|bytes| {
                bytes.checked_add(
                    request
                        .retained_packet_bytes()
                        .expect("queued packet bytes"),
                )
            })
            .expect("queued owner sum")
    });
    assert_eq!(session.queued_plan_ledger.retained_bytes(), expected);
    assert!(session
        .jpeg_plan_cache_diagnostics()
        .retained_bytes
        .checked_add(expected)
        .is_some_and(|bytes| bytes <= DEFAULT_MAX_HOST_ALLOCATION_BYTES));

    let queued = session
        .take_queued_requests()
        .expect("stamp executing owner baseline");
    assert_eq!(queued.len(), 12);
    assert!(session.queued.is_empty());
    assert_eq!(session.queued_plan_ledger.retained_bytes(), 0);
}
