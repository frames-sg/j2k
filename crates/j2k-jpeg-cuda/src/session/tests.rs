// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Barrier};

use j2k_jpeg::adapter::{SharedJpegFastPacket, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES};

use super::{select_operation_accounting_result, CudaSession};
use crate::Error;

const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");

#[test]
fn clones_created_before_runtime_initialization_share_lazy_runtime_owner() {
    let session = CudaSession::default();
    let clone = session.clone();

    assert!(Arc::ptr_eq(&session.runtime_state, &clone.runtime_state));
    assert!(!session.is_runtime_initialized());
    assert!(!clone.is_runtime_initialized());
}

#[test]
fn clone_before_init_reuses_one_context_and_output_pool_when_cuda_is_available() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let mut session = CudaSession::default();
    let mut clone = session.clone();
    let first_context = session.cuda_context().expect("first session context");
    let second_context = clone.cuda_context().expect("clone context");
    assert!(first_context.is_same_context(&second_context));
    assert!(session.is_runtime_initialized());
    assert!(clone.is_runtime_initialized());

    let buffer = session
        .take_owned_cuda_output_buffer(4096)
        .expect("take shared output buffer");
    session
        .recycle_owned_cuda_output_buffer(buffer)
        .expect("recycle through first clone");
    assert_eq!(clone.retained_owned_cuda_output_buffers().unwrap(), 1);

    let buffer = clone
        .take_owned_cuda_output_buffer(4096)
        .expect("reuse through second clone");
    assert_eq!(session.retained_owned_cuda_output_buffers().unwrap(), 0);
    clone
        .recycle_owned_cuda_output_buffer(buffer)
        .expect("return through second clone");
    assert_eq!(session.retained_owned_cuda_output_buffers().unwrap(), 1);
}

#[test]
fn retained_pinned_upload_capacity_blocks_later_host_owner_over_admission() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let mut session = CudaSession::default();
    let report = crate::owned_decode::diagnose_owned_cuda_420_entropy(
        BASELINE_420,
        j2k_cuda_runtime::CudaJpegChunkedEntropyConfig::default(),
        &mut session,
    )
    .expect("seed retained CUDA pinned upload staging");
    drop(report);

    let diagnostics = session.owned_cuda_host_memory_diagnostics().unwrap();
    assert!(diagnostics.pinned_upload_retained_bytes > 0);
    assert_eq!(diagnostics.active_owner_bytes, 0);
    let available_without_pinned = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(diagnostics.cache_retained_bytes)
        .expect("cache remains inside host cap");
    let error = session
        .reserve_existing_host_owner(available_without_pinned)
        .expect_err("retained pinned staging must remain part of admission");
    assert!(matches!(
        error,
        Error::HostAllocationTooLarge { requested, cap, .. }
            if requested
                == j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
                    + diagnostics.pinned_upload_retained_bytes
                && cap == j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
    ));
}

#[test]
fn external_pinned_growth_obeys_live_session_context_owners() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let session = CudaSession::default();
    let context = session.cuda_context().expect("session context");
    let external_clone = context.clone();
    let owner = context
        .register_external_host_owner(j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES - 7)
        .expect("seed context owner");
    let before = context
        .pinned_upload_staging_pool_diagnostics()
        .expect("pinned diagnostics before rejection");
    let error = external_clone
        .upload_pinned(&[0_u8; 8])
        .expect_err("external upload must share context authority");
    assert!(matches!(
        error,
        j2k_cuda_runtime::CudaError::HostAllocationTooLarge { .. }
    ));
    assert_eq!(
        owner.bytes(),
        j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES - 7
    );
    assert_eq!(
        context
            .pinned_upload_staging_pool_diagnostics()
            .expect("pinned diagnostics after rejection"),
        before
    );
}

#[test]
fn independent_sessions_on_one_context_cannot_race_past_the_host_cap() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let first = CudaSession::default();
    let context = first.cuda_context().expect("shared context");
    let second = CudaSession::default();
    second
        .bind_cuda_context(&context)
        .expect("bind second independent session");
    let start = Arc::new(Barrier::new(3));
    let finish = Arc::new(Barrier::new(3));
    let requested = j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES / 2 + 1;
    let workers = [first, second].map(|session| {
        let start = Arc::clone(&start);
        let finish = Arc::clone(&finish);
        std::thread::spawn(move || {
            start.wait();
            let result = session.allocate_owned_host_owner(|_| Ok(((), requested)));
            finish.wait();
            result.map(|((), lease)| lease)
        })
    });
    start.wait();
    finish.wait();
    let results = workers.map(|worker| worker.join().expect("allocation worker"));
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(results.iter().filter(|result| result.is_err()).count(), 1);
    assert!(results
        .iter()
        .filter_map(|result| result.as_ref().err())
        .any(|error| matches!(
            error,
            Error::CudaRuntime {
                source: j2k_cuda_runtime::CudaError::HostAllocationTooLarge { .. }
            }
        )));
}

#[test]
fn allocation_transaction_same_context_pinned_reentry_fails_typed_without_deadlock() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let session = CudaSession::default();
    let context = session.cuda_context().expect("session context");
    let before = context
        .pinned_upload_staging_pool_diagnostics()
        .expect("pinned diagnostics before transaction");
    let error = session
        .allocate_owned_host_owner::<()>(|_| {
            let source = context
                .upload_pinned(&[1])
                .expect_err("reserved headroom must block same-context pinned growth");
            Err(Error::CudaRuntime { source })
        })
        .expect_err("allocation operation keeps the pinned admission failure");
    assert!(matches!(
        error,
        Error::CudaRuntime {
            source: j2k_cuda_runtime::CudaError::HostAllocationTooLarge { .. }
        }
    ));
    assert_eq!(
        context
            .pinned_upload_staging_pool_diagnostics()
            .expect("pinned diagnostics after transaction"),
        before
    );
}

#[test]
fn pinned_accounting_guard_rejects_a_foreign_context_operation() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let session = CudaSession::default();
    let context = session.cuda_context().expect("session context");
    let foreign = j2k_cuda_runtime::CudaContext::system_default().expect("foreign context");
    let operation = foreign
        .begin_pinned_upload_operation()
        .expect("foreign pinned operation");
    let Err(error) = session.reserve_pinned_upload_retention(&context, &operation) else {
        panic!("foreign operation must not be charged to the session");
    };
    assert!(matches!(error, Error::CudaRuntime { .. }));
}

#[test]
fn operation_and_accounting_failures_preserve_both_typed_sources() {
    let result = select_operation_accounting_result::<()>(
        Err(Error::UnsupportedCudaRequest { reason: "primary" }),
        Err(Error::InFlightHostLedgerPoisoned),
    )
    .expect_err("both failures must remain visible");
    let Error::OperationAndHostAccountingFailed {
        primary,
        accounting,
    } = result
    else {
        panic!("expected compound operation/accounting failure");
    };
    assert!(matches!(
        *primary,
        Error::UnsupportedCudaRequest { reason: "primary" }
    ));
    assert!(matches!(*accounting, Error::InFlightHostLedgerPoisoned));
}

#[test]
fn cloned_sessions_share_one_cache_and_hit_the_same_one_family_owners() {
    let session = CudaSession::default();
    let first = session
        .resolve_owned_packet(BASELINE_420)
        .expect("first plan")
        .expect("first packet");
    let clone = session.clone();
    let second = clone
        .resolve_owned_packet(BASELINE_420)
        .expect("clone hit")
        .expect("second packet");

    assert!(SharedJpegFastPacket::ptr_eq(&first.packet, &second.packet));
    let packet = &second.packet;
    let family_count = usize::from(packet.fast420().is_some())
        + usize::from(packet.fast422().is_some())
        + usize::from(packet.fast444().is_some());
    assert_eq!(family_count, 1);
    let diagnostics = clone.owned_cuda_packet_cache_diagnostics().unwrap();
    assert_eq!(diagnostics.entries, 1);
    assert_eq!(diagnostics.misses, 1);
    assert_eq!(diagnostics.hits, 1);
    assert_eq!(
        session.owned_packet_cache.active_host_bytes().unwrap(),
        first.packet.retained_cache_bytes().unwrap() * 2
    );
}

#[test]
fn clone_shared_host_memory_diagnostics_preserve_high_water_after_release() {
    let session = CudaSession::default();
    let clone = session.clone();
    assert_eq!(
        session.owned_cuda_host_memory_diagnostics().unwrap(),
        super::CudaJpegHostMemoryDiagnostics {
            cache_retained_bytes: 0,
            active_owner_bytes: 0,
            pinned_upload_retained_bytes: 0,
            current_combined_bytes: 0,
            peak_active_owner_bytes: 0,
            peak_combined_bytes: 0,
        }
    );

    let leased = session
        .resolve_owned_packet(BASELINE_420)
        .expect("diagnostic plan")
        .expect("diagnostic packet");
    let packet_bytes = leased.packet.retained_cache_bytes().unwrap();
    let live = clone.owned_cuda_host_memory_diagnostics().unwrap();
    assert_eq!(live.active_owner_bytes, packet_bytes);
    assert_eq!(
        live.current_combined_bytes,
        live.cache_retained_bytes + packet_bytes
    );
    assert_eq!(live.peak_active_owner_bytes, packet_bytes);
    assert!(live.peak_combined_bytes >= live.current_combined_bytes);

    drop(leased);
    let released = session.owned_cuda_host_memory_diagnostics().unwrap();
    assert_eq!(released.active_owner_bytes, 0);
    assert_eq!(
        released.current_combined_bytes,
        released.cache_retained_bytes
    );
    assert_eq!(released.peak_active_owner_bytes, packet_bytes);
    assert!(released.peak_combined_bytes >= live.peak_combined_bytes);
}

#[test]
fn malformed_input_is_never_cached() {
    let session = CudaSession::default();
    assert!(matches!(
        session.resolve_owned_packet(&[0xff, 0xd8]),
        Err(Error::Decode(_))
    ));
    assert_eq!(session.owned_cuda_packet_cache_len(), 0);
    assert_eq!(
        session
            .owned_cuda_packet_cache_diagnostics()
            .unwrap()
            .entries,
        0
    );
}

#[test]
fn disabled_and_oversized_admission_return_current_plans_without_retention() {
    let disabled =
        CudaSession::with_owned_packet_cache_limits(0, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES);
    let disabled_packet = disabled
        .resolve_owned_packet(BASELINE_420)
        .expect("disabled resolution")
        .expect("disabled current packet");
    assert!(disabled.owned_packet_cache.active_host_bytes().unwrap() > 0);
    drop(disabled_packet);
    assert_eq!(disabled.owned_packet_cache.active_host_bytes().unwrap(), 0);
    let diagnostics = disabled.owned_cuda_packet_cache_diagnostics().unwrap();
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.disabled_rejections, 1);

    let oversized = CudaSession::with_owned_packet_cache_limits(1, 1);
    let oversized_packet = oversized
        .resolve_owned_packet(BASELINE_420)
        .expect("oversized resolution")
        .expect("oversized current packet");
    assert!(oversized.owned_packet_cache.active_host_bytes().unwrap() > 0);
    drop(oversized_packet);
    assert_eq!(oversized.owned_packet_cache.active_host_bytes().unwrap(), 0);
    let diagnostics = oversized.owned_cuda_packet_cache_diagnostics().unwrap();
    assert_eq!(diagnostics.entries, 0);
    assert_eq!(diagnostics.oversized_rejections, 1);
}

#[test]
fn reused_source_pointer_with_changed_bytes_cannot_cross_hit() {
    let session = CudaSession::default();
    let mut source = BASELINE_420.to_vec();
    let original = source.clone();
    let pointer = source.as_ptr();
    session
        .resolve_owned_packet(&source)
        .expect("initial plan")
        .expect("initial packet");
    source.fill(0);

    assert_eq!(source.as_ptr(), pointer);
    assert!(matches!(
        session.resolve_owned_packet(&source),
        Err(Error::Decode(_))
    ));
    assert!(session
        .resolve_owned_packet(&original)
        .expect("original hit")
        .is_some());
    let diagnostics = session.owned_cuda_packet_cache_diagnostics().unwrap();
    assert_eq!(diagnostics.entries, 1);
    assert_eq!(diagnostics.misses, 2);
    assert_eq!(diagnostics.hits, 1);
}

#[test]
fn long_sequence_stays_bounded_and_evicts_by_lru() {
    let session =
        CudaSession::with_owned_packet_cache_limits(3, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES);
    let inputs: Vec<Vec<u8>> = (0_u8..10).map(encoded_fixture).collect();
    for input in &inputs {
        session
            .resolve_owned_packet(input)
            .expect("resolve sequence plan")
            .expect("resolve sequence packet");
        assert!(session.owned_cuda_packet_cache_len() <= 3);
    }

    let diagnostics = session.owned_cuda_packet_cache_diagnostics().unwrap();
    assert_eq!(diagnostics.entries, 3);
    assert_eq!(diagnostics.peak_entries, 3);
    assert_eq!(diagnostics.evictions, 7);
    let misses = diagnostics.misses;
    session
        .resolve_owned_packet(&inputs[0])
        .expect("oldest input rebuilds after eviction")
        .expect("oldest packet rebuilds");
    assert_eq!(
        session
            .owned_cuda_packet_cache_diagnostics()
            .unwrap()
            .misses,
        misses + 1
    );
}

#[test]
fn more_than_eight_leased_packets_survive_cache_eviction_until_release() {
    let session =
        CudaSession::with_owned_packet_cache_limits(3, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES);
    let inputs: Vec<Vec<u8>> = (0_u8..10).map(encoded_fixture).collect();
    let mut leased = Vec::new();
    for input in &inputs {
        leased.push(
            session
                .resolve_owned_packet(input)
                .expect("resolve leased sequence")
                .expect("ready leased packet"),
        );
    }
    assert_eq!(leased.len(), 10);
    assert_eq!(session.owned_cuda_packet_cache_len(), 3);
    assert_eq!(
        session
            .owned_cuda_packet_cache_diagnostics()
            .unwrap()
            .evictions,
        7
    );
    assert!(leased
        .iter()
        .all(|packet| packet.packet.fast420().is_some()));
    assert!(session.owned_packet_cache.active_host_bytes().unwrap() > 0);
    drop(leased);
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);
}

#[test]
fn packet_lease_has_exact_cap_and_release_reacquire_boundaries() {
    let session = CudaSession::default();
    let probe = session
        .resolve_owned_packet(BASELINE_420)
        .expect("build probe plan")
        .expect("probe packet");
    let packet_bytes = probe.packet.retained_cache_bytes().unwrap();
    drop(probe);
    let cache_bytes = session
        .owned_cuda_packet_cache_diagnostics()
        .unwrap()
        .retained_bytes;
    let exact_cap = cache_bytes + packet_bytes;
    let leased = session
        .owned_packet_cache
        .reserve_with_cap_for_test(packet_bytes, exact_cap)
        .expect("exact lease cap");
    assert_eq!(
        session.owned_packet_cache.active_host_bytes().unwrap(),
        packet_bytes
    );
    drop(leased);
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);

    let leased = session
        .owned_packet_cache
        .reserve_with_cap_for_test(packet_bytes, exact_cap)
        .expect("released bytes can be reacquired");
    drop(leased);

    assert!(matches!(
        session
            .owned_packet_cache
            .reserve_with_cap_for_test(packet_bytes, exact_cap - 1),
        Err(Error::HostAllocationTooLarge { requested, cap, .. })
            if requested == exact_cap && cap == exact_cap - 1
    ));
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);
}

#[test]
fn clone_threads_count_duplicate_packet_handles_until_each_release() {
    let session = CudaSession::default();
    let probe = session
        .resolve_owned_packet(BASELINE_420)
        .expect("probe resolution")
        .expect("probe packet");
    let packet_bytes = probe.packet.retained_cache_bytes().unwrap();
    drop(probe);

    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let worker_session = session.clone();
        let worker_barrier = Arc::clone(&barrier);
        workers.push(std::thread::spawn(move || {
            let packet = worker_session
                .resolve_owned_packet(BASELINE_420)
                .expect("thread resolution")
                .expect("thread packet");
            worker_barrier.wait();
            worker_barrier.wait();
            drop(packet);
        }));
    }
    barrier.wait();
    assert_eq!(
        session.owned_packet_cache.active_host_bytes().unwrap(),
        packet_bytes * 2
    );
    barrier.wait();
    for worker in workers {
        worker.join().expect("lease worker");
    }
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);
}

#[test]
fn existing_decoder_retained_bytes_are_part_of_the_in_flight_lease() {
    let decoder = j2k_jpeg::Decoder::new(BASELINE_420).expect("existing decoder");
    let decoder_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(&decoder).unwrap();
    let session = CudaSession::default();
    let leased = session
        .resolve_owned_packet_from_decoder(&decoder)
        .expect("resolve existing decoder")
        .expect("existing-decoder packet");
    let packet_bytes = leased.packet.retained_cache_bytes().unwrap();
    assert_eq!(
        session.owned_packet_cache.active_host_bytes().unwrap(),
        decoder_bytes + packet_bytes
    );
    drop(leased);
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);
}

#[test]
fn operation_gate_serializes_clone_threads_without_holding_the_cache_mutex() {
    let session = CudaSession::default();
    let gate = session.jpeg_host_operation_gate();
    let held = gate.lock().unwrap();
    let clone = session.clone();
    let (sender, receiver) = std::sync::mpsc::channel();
    let worker = std::thread::spawn(move || {
        let worker_gate = clone.jpeg_host_operation_gate();
        let _operation = worker_gate.lock().unwrap();
        sender.send(()).unwrap();
    });
    assert!(receiver.try_recv().is_err());
    drop(held);
    receiver.recv().expect("worker enters after release");
    worker.join().unwrap();
}

#[test]
fn unwind_releases_exact_host_owner_bytes() {
    let session = CudaSession::default();
    let result = std::panic::catch_unwind(|| {
        let _leased = session
            .resolve_owned_packet(BASELINE_420)
            .expect("resolve before unwind")
            .expect("packet before unwind");
        panic!("intentional in-flight release unwind");
    });
    assert!(result.is_err());
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);
}

#[test]
fn debug_uses_a_nonblocking_cache_snapshot() {
    let session = CudaSession::default();
    let _guard = session.owned_packet_cache.state.lock().unwrap();
    let rendered = format!("{session:?}");
    assert!(rendered.contains("owned_cuda_packet_cache_diagnostics: Ok(None)"));
}

#[test]
fn poison_is_typed_while_infallible_len_keeps_last_coherent_shadow() {
    let session = CudaSession::default();
    let leased = session
        .resolve_owned_packet(BASELINE_420)
        .expect("seed cache")
        .expect("seed packet");
    assert_eq!(session.owned_cuda_packet_cache_len(), 1);
    let cache = Arc::clone(&session.owned_packet_cache);
    let poisoned = std::thread::spawn(move || {
        let _guard = cache.state.lock().expect("lock cache before poison");
        panic!("intentional cache poison test");
    });
    assert!(poisoned.join().is_err());

    drop(leased);
    assert_eq!(session.owned_packet_cache.active_host_bytes().unwrap(), 0);

    assert!(matches!(
        session.owned_cuda_packet_cache_diagnostics(),
        Err(Error::OwnedPacketCachePoisoned)
    ));
    assert_eq!(session.owned_cuda_packet_cache_len(), 1);
    assert!(format!("{session:?}").contains("OwnedPacketCachePoisoned"));
}

fn encoded_fixture(seed: u8) -> Vec<u8> {
    let mut rgb = vec![0_u8; 8 * 8 * 3];
    for (index, sample) in rgb.iter_mut().enumerate() {
        *sample = seed
            .wrapping_mul(17)
            .wrapping_add(u8::try_from(index).expect("8x8 RGB index fits u8"));
    }
    j2k_jpeg::encode_jpeg_baseline(
        j2k_jpeg::JpegSamples::Rgb8 {
            data: &rgb,
            width: 8,
            height: 8,
        },
        j2k_jpeg::JpegEncodeOptions {
            quality: 90,
            subsampling: j2k_jpeg::JpegSubsampling::Ybr420,
            restart_interval: None,
            backend: j2k_jpeg::JpegBackend::Cpu,
        },
    )
    .expect("encode fixture")
    .data
}
