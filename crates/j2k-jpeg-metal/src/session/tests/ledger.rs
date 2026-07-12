// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use j2k_core::{BackendKind, PixelFormat, SurfaceResidency, DEFAULT_MAX_HOST_ALLOCATION_BYTES};
use j2k_jpeg::adapter::JpegPlanCacheError;

use crate::{Storage, Surface};

fn request_from_plan(plan: ResolvedJpegPlan) -> crate::batch::QueuedRequest {
    crate::batch::QueuedRequest::new_shared(
        plan.input,
        PixelFormat::Rgb8,
        BackendRequest::Metal,
        crate::batch::BatchOp::Full,
        plan.fast_packet,
        plan.shape,
    )
}

fn plan_owner_bytes(plan: &ResolvedJpegPlan) -> usize {
    let input_bytes = plan
        .input
        .retained_cache_bytes()
        .expect("test input retained bytes");
    let packet_bytes = plan.fast_packet.as_ref().map_or(0, |packet| {
        packet
            .retained_cache_bytes()
            .expect("test packet retained bytes")
    });
    input_bytes
        .checked_add(packet_bytes)
        .expect("test plan retained bytes")
}

fn host_surface_with_capacity(capacity: usize) -> Surface {
    let bytes = vec![0; capacity];
    let width = u32::try_from(capacity).expect("test surface width fits u32");
    Surface {
        backend: BackendKind::Cpu,
        residency: SurfaceResidency::Host,
        dimensions: (width, 1),
        fmt: PixelFormat::Gray8,
        pitch_bytes: capacity,
        storage: Storage::Host(std::sync::Arc::new(bytes)),
    }
}

#[test]
fn queued_and_cached_plan_limit_accepts_exact_and_rejects_one_over_before_mutation() {
    let mut exact = SessionState::default();
    assert_eq!(
        exact.queued_plan_ledger.host_byte_limit(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES
    );
    let exact_plan = exact
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("exact cached plan");
    let exact_owner_bytes = plan_owner_bytes(&exact_plan);
    let exact_limit = exact
        .jpeg_plan_cache_diagnostics()
        .retained_bytes
        .checked_add(exact_owner_bytes)
        .expect("exact combined limit");
    exact.queued_plan_ledger.set_host_byte_limit(exact_limit);

    exact
        .queue_request(request_from_plan(exact_plan))
        .expect("exact queued/cache owner limit");
    assert_eq!(exact.queued_plan_ledger.retained_bytes(), exact_owner_bytes);
    assert_eq!(exact.queued.len(), 1);

    let mut over = SessionState::default();
    let over_plan = over
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("one-over cached plan");
    let over_owner_bytes = plan_owner_bytes(&over_plan);
    let requested = over
        .jpeg_plan_cache_diagnostics()
        .retained_bytes
        .checked_add(over_owner_bytes)
        .expect("one-over combined request");
    over.queued_plan_ledger.set_host_byte_limit(requested - 1);

    let error = over
        .queue_request(request_from_plan(over_plan))
        .expect_err("one byte over combined owner limit");
    assert!(matches!(
        error,
        Error::JpegPlanCache(JpegPlanCacheError::Limit {
            what: "JPEG Metal queued and cached plan owner graphs",
            requested: actual,
            cap,
        }) if actual == requested && cap == requested - 1
    ));
    assert!(over.queued.is_empty());
    assert!(over.completed.is_empty());
    assert_eq!(over.queued_plan_ledger.retained_bytes(), 0);
}

#[test]
fn plan_build_baseline_accepts_exact_hit_and_rejects_one_over_before_admission() {
    let mut session = SessionState::default();
    let cached = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("cached fixture plan");
    session
        .queue_request(request_from_plan(cached))
        .expect("pre-existing queued plan");

    let cache_bytes = session.jpeg_plan_cache_diagnostics().retained_bytes;
    let queue_bytes = session.queued_plan_ledger.retained_bytes();
    let metadata_bytes = session
        .session_metadata_live_bytes()
        .expect("pre-existing session metadata");
    let available = DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(cache_bytes)
        .and_then(|bytes| bytes.checked_sub(queue_bytes))
        .and_then(|bytes| bytes.checked_sub(metadata_bytes))
        .expect("pre-existing graph is below the global cap");

    let exact = session
        .resolve_jpeg_plan_with_external_live(BASELINE_420, BackendRequest::Metal, available)
        .expect("an owner-sharing cache hit at the exact cap");
    assert!(SharedJpegInput::ptr_eq(
        &exact.input,
        &session.queued[0].input
    ));

    let before = session.jpeg_plan_cache_diagnostics();
    let error = session
        .resolve_jpeg_plan_with_external_live(BASELINE_444, BackendRequest::Metal, available + 1)
        .expect_err("one byte over must fail before constructing a miss plan");
    assert!(
        matches!(
            &error,
            Error::JpegPlanCache(JpegPlanCacheError::Limit {
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                ..
            }) if *requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
        ),
        "unexpected one-over error: {error:?}"
    );
    let after = session.jpeg_plan_cache_diagnostics();
    assert_eq!(after.entries, before.entries);
    assert_eq!(after.retained_bytes, before.retained_bytes);
    assert_eq!(after.evictions, before.evictions);
    assert_eq!(session.queued.len(), 1);
    assert_eq!(session.queued_plan_ledger.retained_bytes(), queue_bytes);
}

#[test]
fn collective_queue_preflight_composes_nonzero_owners_and_metadata_exactly() {
    let mut session = SessionState::default();
    let plan = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("cached fixture plan");
    let owner_bytes = plan_owner_bytes(&plan);
    let cache_bytes = session.jpeg_plan_cache_diagnostics().retained_bytes;
    let queued_capacity = 1;
    let completed_capacity = 1;
    let fixed_bytes = owner_bytes
        .checked_add(cache_bytes)
        .and_then(|bytes| {
            bytes.checked_add(queued_capacity * std::mem::size_of::<crate::batch::QueuedRequest>())
        })
        .and_then(|bytes| {
            bytes.checked_add(
                completed_capacity
                    * std::mem::size_of::<Option<Result<crate::Surface, crate::Error>>>(),
            )
        })
        .expect("collective fixed bytes");
    let exact_submission_bytes = DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(fixed_bytes)
        .expect("fixture owners leave metadata headroom");

    session
        .preflight_collective_queue_state(
            owner_bytes,
            exact_submission_bytes,
            queued_capacity,
            completed_capacity,
        )
        .expect("owners and metadata at exact global cap");
    let error = session
        .preflight_collective_queue_state(
            owner_bytes,
            exact_submission_bytes + 1,
            queued_capacity,
            completed_capacity,
        )
        .expect_err("owners and metadata one byte over global cap");
    assert!(matches!(
        error,
        Error::BatchInfrastructure(
            j2k_core::BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG Metal collective queued request state",
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
        ) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
    ));
}

#[test]
fn execution_stamp_includes_completed_and_submission_capacities() {
    let mut session = SessionState::default();
    let plan = session
        .resolve_jpeg_plan(BASELINE_420, BackendRequest::Metal)
        .expect("cached fixture plan");
    let submission_capacity = 3;
    session
        .queue_request_with_retained(request_from_plan(plan), submission_capacity)
        .expect("queue with retained submissions");

    let queued_bytes =
        session.queued.capacity() * std::mem::size_of::<crate::batch::QueuedRequest>();
    let completed_bytes = session.completed.capacity()
        * std::mem::size_of::<Option<Result<crate::Surface, crate::Error>>>();
    let submission_bytes =
        submission_capacity * std::mem::size_of::<crate::batch::MetalSubmission>();
    assert!(completed_bytes > 0);

    let queued = session
        .take_queued_requests()
        .expect("stamp executing owner baseline");
    assert_eq!(queued.len(), 1);
    assert_eq!(
        queued[0].execution_external_live_bytes(),
        queued_bytes + completed_bytes + submission_bytes
    );
}

#[path = "ledger/completions.rs"]
mod completions;
#[path = "ledger/owners.rs"]
mod owners;
#[path = "ledger/transactions.rs"]
mod transactions;
