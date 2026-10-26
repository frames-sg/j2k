// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, EncodedImage,
    PreparedBatch,
};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

use super::test_fixtures::{four_level_rgb8_fixture, invalidate_first_cleanup_scup};
use crate::decoder::color_batch::COLOR_BATCH_CHUNK_IMAGES;
use crate::{CudaBatchDecodeResult, CudaBatchDecoder, CudaSession, Error, Surface};

fn alternating_inputs(first: &Arc<[u8]>, second: &Arc<[u8]>, count: usize) -> Vec<EncodedImage> {
    (0..count)
        .map(|index| {
            let encoded = if index.is_multiple_of(2) {
                first
            } else {
                second
            };
            EncodedImage::full(Arc::clone(encoded))
        })
        .collect()
}

fn cpu_rgb8_samples(prepared: &PreparedBatch, options: BatchDecodeOptions) -> Vec<u8> {
    let oracle = CpuBatchDecoder::new(options)
        .decode_prepared(prepared)
        .expect("CPU RGB8 oracle");
    let [group] = oracle.groups() else {
        panic!("CPU oracle must produce one RGB8 group")
    };
    let CpuBatchSamples::U8(samples) = group.samples() else {
        panic!("CPU oracle must produce RGB8 samples")
    };
    samples.clone()
}

fn dense_rgb8_bytes(result: &CudaBatchDecodeResult, image_count: usize) -> Vec<u8> {
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    assert!(
        result.group_errors().is_empty(),
        "{:?}",
        result.group_errors()
    );
    let [group] = result.groups() else {
        panic!("expected one resident RGB8 group")
    };
    assert_eq!(group.source_indices(), (0..image_count).collect::<Vec<_>>());
    let dense = group.dense_output();
    assert_eq!(dense.ranges().len(), image_count);
    let image_bytes = dense.ranges()[0].len;
    for (index, range) in dense.ranges().iter().enumerate() {
        assert_eq!(range.offset, index * image_bytes, "dense range {index}");
        assert_eq!(range.len, image_bytes, "dense range {index}");
    }
    let mut bytes = vec![0_u8; image_bytes * image_count];
    dense
        .buffer()
        .copy_range_to_host(dense.ranges()[0].offset, &mut bytes)
        .expect("download dense RGB8 group");
    bytes
}

fn device_allocations(decoder: &CudaBatchDecoder) -> u64 {
    decoder
        .diagnostics()
        .expect("CUDA decoder diagnostics")
        .runtime
        .expect("CUDA runtime diagnostics")
        .device_allocation_operations
}

/// Warm the session once more, then return the allocations made by one decode.
fn warm_decode_allocations(
    decoder: &mut CudaBatchDecoder,
    prepared: &PreparedBatch,
) -> (u64, CudaBatchDecodeResult) {
    drop(decoder.decode_prepared(prepared).expect("warm-up decode"));
    let before = device_allocations(decoder);
    let result = decoder.decode_prepared(prepared).expect("measured decode");
    (device_allocations(decoder) - before, result)
}

#[test]
fn owned_large_rgb_group_matches_cpu_in_order_and_reuses_bounded_scratch() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let first: Arc<[u8]> = Arc::from(four_level_rgb8_fixture(17).0);
    let second: Arc<[u8]> = Arc::from(four_level_rgb8_fixture(29).0);
    let count = 2 * COLOR_BATCH_CHUNK_IMAGES + 1;
    let inputs = alternating_inputs(&first, &second, count);

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = {
            let mut options = BatchDecodeOptions::default();
            options.layout = layout;
            options
        };
        let large = prepare_batch(inputs.clone(), options).expect("prepare large RGB8 batch");
        let one_chunk = prepare_batch(inputs[..COLOR_BATCH_CHUNK_IMAGES].to_vec(), options)
            .expect("prepare one-chunk RGB8 batch");
        let expected = cpu_rgb8_samples(&large, options);

        // Measure each size on its own session so the cache holds only its working set.
        let one_chunk_allocations = {
            let mut decoder = CudaBatchDecoder::with_options(options);
            drop(
                decoder
                    .decode_prepared(&one_chunk)
                    .expect("cold one-chunk RGB8 decode"),
            );
            warm_decode_allocations(&mut decoder, &one_chunk).0
        };
        let mut decoder = CudaBatchDecoder::with_options(options);
        let cold = decoder
            .decode_prepared(&large)
            .expect("cold large RGB8 decode");
        assert_eq!(dense_rgb8_bytes(&cold, count), expected, "{layout:?} cold");
        drop(cold);
        let (large_allocations, warm) = warm_decode_allocations(&mut decoder, &large);
        assert_eq!(dense_rgb8_bytes(&warm, count), expected, "{layout:?} warm");
        if layout == BatchLayout::Nhwc {
            assert_eq!(
                Surface::download_batch_tight(warm.groups()[0].surfaces())
                    .expect("download NHWC surfaces"),
                expected
            );
        }
        let chunks = u64::try_from(count.div_ceil(COLOR_BATCH_CHUNK_IMAGES)).expect("chunks");
        assert!(
            large_allocations <= one_chunk_allocations * chunks,
            "{layout:?}: a warm {count}-image group made {large_allocations} device \
             allocations, more than {chunks} warm {COLOR_BATCH_CHUNK_IMAGES}-image groups \
             ({one_chunk_allocations} each)"
        );
    }
}

#[test]
fn external_large_rgb_group_matches_cpu_in_one_caller_allocation() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let first: Arc<[u8]> = Arc::from(four_level_rgb8_fixture(17).0);
    let second: Arc<[u8]> = Arc::from(four_level_rgb8_fixture(29).0);
    let count = 2 * COLOR_BATCH_CHUNK_IMAGES + 1;
    let options = {
        let mut options = BatchDecodeOptions::default();
        options.layout = BatchLayout::Nchw;
        options
    };
    let prepared = prepare_batch(alternating_inputs(&first, &second, count), options)
        .expect("prepare large external RGB8 batch");
    let expected = cpu_rgb8_samples(&prepared, options);
    let [group] = prepared.groups() else {
        panic!("expected one prepared RGB8 group")
    };
    let context = CudaContext::system_default().expect("CUDA context");
    let mut decoder = CudaBatchDecoder::with_session_and_options(
        CudaSession::with_context(context.clone()),
        options,
    );
    let mut allocation = context
        .allocate(expected.len())
        .expect("external RGB8 destination");
    let ptr = allocation.device_ptr();
    let len = allocation.byte_len();
    // SAFETY: the allocation stays live and exclusively owned until the
    // synchronous decode below returns.
    let external = unsafe {
        let mut destination =
            CudaExternalDeviceBufferViewMut::from_raw_parts(&context, ptr, len, 1, &mut allocation)
                .expect("external RGB8 view");
        decoder.decode_batch_into(group, &mut destination)
    }
    .expect("decode large external RGB8 group");

    assert_eq!(external.ranges().len(), count);
    let image_bytes = expected.len() / count;
    for (index, range) in external.ranges().iter().enumerate() {
        assert_eq!(range.offset, index * image_bytes, "external range {index}");
        assert_eq!(range.len, image_bytes, "external range {index}");
    }
    let mut actual = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut actual)
        .expect("download external RGB8 group");
    assert_eq!(actual, expected);
}

#[test]
fn owned_large_rgb_group_maps_late_chunk_kernel_failure_to_global_source_and_job() {
    if !j2k_test_support::cuda_runtime_and_strict_oxide_gate(module_path!()) {
        return;
    }
    let (valid, _) = j2k_test_support::htj2k_rgb8_fixture_with_pixels(129, 131);
    let mut corrupted = valid.clone();
    let (corrupted_job, jobs_per_image) = invalidate_first_cleanup_scup(&mut corrupted);
    let valid: Arc<[u8]> = Arc::from(valid);
    let corrupted: Arc<[u8]> = Arc::from(corrupted);
    let options = BatchDecodeOptions::default();
    let single = prepare_batch(vec![EncodedImage::full(Arc::clone(&valid))], options)
        .expect("prepare single RGB8 image");
    let expected_single = cpu_rgb8_samples(&single, options);
    let corrupted_source = COLOR_BATCH_CHUNK_IMAGES + 1;
    // The final chunk fails at wait; a middle chunk fails while submitting.
    for count in [
        COLOR_BATCH_CHUNK_IMAGES + 2,
        2 * COLOR_BATCH_CHUNK_IMAGES + 1,
    ] {
        let mut inputs = vec![EncodedImage::full(Arc::clone(&valid)); count];
        inputs[corrupted_source] = EncodedImage::full(Arc::clone(&corrupted));
        let mut decoder = CudaBatchDecoder::with_options(options);
        drop(
            decoder
                .decode_batch(vec![EncodedImage::full(Arc::clone(&valid)); count])
                .expect("warm valid RGB8 batch"),
        );

        let result = decoder
            .decode_batch(inputs)
            .expect("recoverable RGB8 group failure");
        assert!(result.groups().is_empty());
        let [group_error] = result.group_errors() else {
            panic!(
                "expected one RGB8 group failure, got {:?}",
                result.group_errors()
            )
        };
        assert_eq!(group_error.source_indices(), (0..count).collect::<Vec<_>>());
        assert!(
            matches!(
                group_error.source(),
                Error::CudaTier1JobFailed {
                    source_index,
                    original_job_index,
                    ..
                } if *source_index == corrupted_source
                    && *original_job_index == jobs_per_image * corrupted_source + corrupted_job
            ),
            "{count} images: {:?}",
            group_error.source()
        );
        let batch_pool = decoder
            .decode_pool_diagnostics()
            .expect("pool diagnostics after RGB8 failure")
            .batch_decode
            .expect("batch pool");
        assert_eq!(batch_pool.reuse_holds, 0);
        assert_eq!(batch_pool.deferred_buffers, 0);

        let reused = decoder
            .decode_prepared(&single)
            .expect("session reuse after RGB8 failure");
        assert_eq!(dense_rgb8_bytes(&reused, 1), expected_single);
    }
}
