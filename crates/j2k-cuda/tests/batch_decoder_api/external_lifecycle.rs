// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, CpuBatchDecoder, CpuBatchSamples, DecodeRequest, Downscale,
    EncodedImage, PreparedBatchGroup, Rect,
};
use j2k_cuda::{CudaBatchDecoder, CudaExternalBatchTryFinish, CudaSession};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};

use super::support::submit_external_for_test;

#[test]
fn external_batch_handles_roi_reduction_and_drop_safe_session_reuse_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels = (0_u8..=255).collect::<Vec<_>>();
    let encoded = Arc::<[u8]>::from(
        j2k_native::encode_htj2k(
            &pixels,
            16,
            16,
            1,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode HTJ2K ROI/reduction fixture"),
    );
    let inputs = vec![
        EncodedImage::new(
            Arc::clone(&encoded),
            DecodeRequest::Region {
                roi: Rect {
                    x: 4,
                    y: 4,
                    w: 8,
                    h: 8,
                },
            },
        ),
        EncodedImage::new(
            encoded,
            DecodeRequest::Reduced {
                scale: Downscale::Half,
            },
        ),
    ];
    let options = BatchDecodeOptions::default();
    let prepared = prepare_batch(inputs, options).expect("prepare shared batch");
    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].source_indices(), &[0]);
    assert_eq!(prepared.groups()[1].source_indices(), &[1]);

    let mut cpu = CpuBatchDecoder::new(options);
    let cpu_output = cpu
        .decode_prepared(&prepared)
        .expect("CPU oracle batch decode");
    assert_eq!(cpu_output.groups().len(), prepared.groups().len());

    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
    for (prepared_group, oracle_group) in prepared.groups().iter().zip(cpu_output.groups().iter()) {
        assert_eq!(
            prepared_group.source_indices(),
            oracle_group.source_indices()
        );
        let expected = match oracle_group.samples() {
            CpuBatchSamples::U8(samples) => samples,
            samples => panic!("expected U8 CPU oracle, got {samples:?}"),
        };
        assert_drop_retirement_and_reuse(&context, &mut decoder, prepared_group, expected);
    }
}

fn assert_drop_retirement_and_reuse(
    context: &CudaContext,
    decoder: &mut CudaBatchDecoder,
    prepared_group: &PreparedBatchGroup,
    expected: &[u8],
) {
    let mut allocation = context
        .allocate(expected.len())
        .expect("external destination allocation");

    // The first submission is intentionally dropped unfinished. Drop must
    // retire its resources before the persistent session is reused.
    {
        let ptr = allocation.device_ptr();
        let len = allocation.byte_len();
        // SAFETY: `allocation` is exclusively borrowed through submission.
        let mut destination = unsafe {
            CudaExternalDeviceBufferViewMut::from_raw_parts(
                context,
                ptr,
                len,
                std::mem::align_of::<u8>(),
                &mut allocation,
            )
        }
        .expect("external destination view");
        // SAFETY: dropping the completion owner retires work before reuse.
        let submitted = unsafe { decoder.submit_batch_into(prepared_group, &mut destination) }
            .expect("submit external batch for drop retirement");
        drop(submitted);
    }

    let submitted = {
        let ptr = allocation.device_ptr();
        let len = allocation.byte_len();
        // SAFETY: the allocation remains live and inaccessible until the
        // returned completion owner is retired below.
        let mut destination = unsafe {
            CudaExternalDeviceBufferViewMut::from_raw_parts(
                context,
                ptr,
                len,
                std::mem::align_of::<u8>(),
                &mut allocation,
            )
        }
        .expect("reused external destination view");
        // SAFETY: there is no overlapping access before completion.
        unsafe { decoder.submit_batch_into(prepared_group, &mut destination) }
            .expect("submit external batch after drop retirement")
    };
    let group = match submitted.try_finish().expect("poll external batch") {
        CudaExternalBatchTryFinish::Pending(submitted) => {
            submitted.wait().expect("wait external batch")
        }
        CudaExternalBatchTryFinish::Complete(group) => group,
    };
    assert_eq!(group.source_indices(), prepared_group.source_indices());
    assert_eq!(group.ranges().len(), 1);

    let mut actual = vec![0_u8; expected.len()];
    allocation
        .copy_to_host(&mut actual)
        .expect("download completed external output");
    assert_eq!(actual, expected);
}

#[test]
fn mixed_decomposition_shapes_keep_async_pool_resources_live_when_runtime_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let pixels_a = (0_u8..=255).collect::<Vec<_>>();
    let pixels_b = pixels_a
        .iter()
        .map(|value| value.wrapping_mul(29).wrapping_add(7))
        .collect::<Vec<_>>();
    let encode = |pixels: &[u8], levels| {
        Arc::<[u8]>::from(
            j2k_native::encode_htj2k(
                pixels,
                16,
                16,
                1,
                8,
                false,
                &j2k_native::EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: levels,
                    ..j2k_native::EncodeOptions::default()
                },
            )
            .expect("encode mixed-decomposition HTJ2K fixture"),
        )
    };
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(encode(&pixels_a, 1)),
            EncodedImage::full(encode(&pixels_b, 2)),
        ],
        BatchDecodeOptions::default(),
    )
    .expect("prepare mixed-decomposition batch");
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].source_indices(), [0]);
    assert_eq!(prepared.groups()[1].source_indices(), [1]);

    let mut cpu = CpuBatchDecoder::new(BatchDecodeOptions::default());
    let oracle = cpu
        .decode_prepared(&prepared)
        .expect("CPU mixed-decomposition oracle");
    let expected = oracle
        .groups()
        .iter()
        .map(|group| match group.samples() {
            CpuBatchSamples::U8(samples) => samples,
            samples => panic!("expected U8 CPU oracle, got {samples:?}"),
        })
        .collect::<Vec<_>>();

    let context = CudaContext::system_default().expect("CUDA context");
    let session = CudaSession::with_context(context.clone());
    let mut decoder = CudaBatchDecoder::with_session(session);
    let mut first_output = context
        .allocate(expected[0].len())
        .expect("first mixed-shape output");
    let mut second_output = context
        .allocate(expected[1].len())
        .expect("second mixed-shape output");
    // SAFETY: both allocations remain live and untouched until both pending
    // owners are waited below. Distinct destinations do not overlap.
    let first = unsafe {
        submit_external_for_test(
            &mut decoder,
            &prepared.groups()[0],
            &context,
            &mut first_output,
        )
    };
    // Submit again before waiting the first batch. This catches recycling of
    // IDWT scratch still referenced by the first default-stream graph.
    // SAFETY: same lifetime guarantee as the first distinct allocation.
    let second = unsafe {
        submit_external_for_test(
            &mut decoder,
            &prepared.groups()[1],
            &context,
            &mut second_output,
        )
    };
    second.wait().expect("wait second mixed-shape batch");
    first.wait().expect("wait first mixed-shape batch");

    let mut first_actual = vec![0_u8; expected[0].len()];
    let mut second_actual = vec![0_u8; expected[1].len()];
    first_output
        .copy_to_host(&mut first_actual)
        .expect("download first mixed-shape batch");
    second_output
        .copy_to_host(&mut second_actual)
        .expect("download second mixed-shape batch");
    assert_eq!(first_actual, *expected[0]);
    assert_eq!(second_actual, *expected[1]);
}
