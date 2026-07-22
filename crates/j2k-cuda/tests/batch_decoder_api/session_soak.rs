// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
use j2k_cuda::{CudaBatchDecoder, CudaSession};
use j2k_cuda_runtime::CudaContext;

use super::support::submit_external_for_test;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "one soak scenario audits pool, transfer, event, synchronization, and output invariants together"
)]
fn resident_and_external_decode_pools_stabilize_for_one_thousand_batches_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-s12-53-single-raw")
        .expect("independent signed RGB12 single-tile fixture");
    let encoded = Arc::<[u8]>::from(fixture.encoded);
    let inputs = (0..8)
        .map(|_| EncodedImage::full(Arc::clone(&encoded)))
        .collect::<Vec<_>>();
    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context.clone());
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    let prepared = decoder
        .prepare(inputs)
        .expect("prepare reusable RGB12 batch");
    let [group] = prepared.groups() else {
        panic!("expected one homogeneous signed RGB12 group")
    };
    let external_bytes = fixture
        .oracle
        .len()
        .checked_mul(8)
        .expect("external batch byte length");
    let mut external = context
        .allocate(external_bytes)
        .expect("persistent external destination");

    for _ in 0..16 {
        drop(
            decoder
                .decode_prepared(&prepared)
                .expect("warm persistent CUDA batch pools"),
        );
        // SAFETY: the allocation remains live and inaccessible until wait.
        unsafe { submit_external_for_test(&mut decoder, group, &context, &mut external) }
            .wait()
            .expect("warm persistent external CUDA batch pools");
    }
    let warm = decoder
        .diagnostics()
        .expect("warm CUDA session diagnostics");
    let warm_runtime = warm.runtime.expect("CUDA runtime must be initialized");
    let warm_batch = warm
        .pools
        .batch_decode
        .expect("dense batch decode pool must be initialized");
    assert_eq!(warm_batch.deferred_bytes, 0);
    assert_eq!(warm_batch.reuse_holds, 0);
    assert!(warm.pools.retained_bytes() > 0);

    for _ in 0..1_000 {
        drop(
            decoder
                .decode_prepared(&prepared)
                .expect("reuse prepared CUDA batch"),
        );
    }
    for _ in 0..1_000 {
        // SAFETY: the allocation remains live and inaccessible until wait.
        unsafe { submit_external_for_test(&mut decoder, group, &context, &mut external) }
            .wait()
            .expect("reuse external CUDA destination");
    }

    let after = decoder
        .diagnostics()
        .expect("post-soak CUDA session diagnostics");
    let after_runtime = after.runtime.expect("CUDA runtime remains initialized");
    let after_batch = after
        .pools
        .batch_decode
        .expect("batch pool remains initialized");
    assert_eq!(after_batch.deferred_bytes, 0);
    assert_eq!(after_batch.reuse_holds, 0);
    assert_eq!(after.pools.retained_bytes(), warm.pools.retained_bytes());
    assert_eq!(
        after.pools.peak_retained_bytes_upper_bound(),
        warm.pools.peak_retained_bytes_upper_bound()
    );
    assert_eq!(
        after_runtime.status_device_to_host_operations
            - warm_runtime.status_device_to_host_operations,
        2_000,
        "each resident/external group must use one status readback"
    );
    assert_eq!(
        after_runtime.device_to_host_operations - warm_runtime.device_to_host_operations,
        2_000,
        "the soak must not download decoded pixels"
    );
    assert_eq!(
        after_runtime.device_to_host_bytes - warm_runtime.device_to_host_bytes,
        after_runtime.status_device_to_host_bytes - warm_runtime.status_device_to_host_bytes,
        "all pre-verification D2H bytes must be the one small status transfer"
    );
    assert_eq!(
        after_runtime.event_driver_allocations, warm_runtime.event_driver_allocations,
        "completion events must stabilize after warmup"
    );
    assert!(after_runtime.event_reuses > warm_runtime.event_reuses);
    assert_eq!(
        after_runtime.event_host_synchronizations, warm_runtime.event_host_synchronizations,
        "the group status readback must replace a second final-store event wait"
    );
    assert_eq!(
        after_runtime.context_host_synchronizations, warm_runtime.context_host_synchronizations,
        "normal batch completion must not synchronize the whole context"
    );

    let mut actual = vec![0_u8; external_bytes];
    external
        .copy_to_host(&mut actual)
        .expect("download final external soak output");
    for image in actual.chunks_exact(fixture.oracle.len()) {
        assert_eq!(image, fixture.oracle);
    }
}
