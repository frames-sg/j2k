// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec, vec::Vec};
use core::num::NonZeroUsize;

use super::super::{
    prepare_batch, BatchDecodeOptions, BatchErrorStage, BatchItemError, BatchLayout,
    CpuBatchDecoder, CpuBatchSamples, CpuExecution, DecodeRequest, EncodedImage, IndexedBatchError,
    PreparedBatch,
};
use super::image::{PLANNED_DECODE_BOUND, STREAM_ABOVE_OVERRIDE, STREAM_ABOVE_PLANNED_BYTES};
use j2k_core::Downscale;

fn classic_fixture(width: u32, height: u32, components: u16, precision: u8) -> Arc<[u8]> {
    let count = width * height * u32::from(components);
    let samples = if precision <= 8 {
        (0..count)
            .map(|index| u8::try_from((index * 7 + index / 97) & 0xff).expect("masked sample"))
            .collect::<Vec<_>>()
    } else {
        (0..count)
            .flat_map(|index| {
                u16::try_from((index * 977 + 31) % (1 << precision))
                    .expect("masked sample")
                    .to_le_bytes()
            })
            .collect()
    };
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..j2k_native::EncodeOptions::default()
    };
    Arc::from(
        j2k_native::encode(
            &samples, width, height, components, precision, false, &options,
        )
        .expect("encode classic fixture"),
    )
}

fn classic_rgb8(width: u32, height: u32) -> Arc<[u8]> {
    classic_fixture(width, height, 3, 8)
}

/// One worker keeps preparation and decode on this thread, where the
/// planned-bound override applies.
fn single_worker_options() -> BatchDecodeOptions {
    BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    }
}

struct PlannedBound;

impl PlannedBound {
    fn hold(bytes: usize) -> Self {
        PLANNED_DECODE_BOUND.with(|bound| bound.set(Some(bytes)));
        Self
    }
}

impl Drop for PlannedBound {
    fn drop(&mut self) {
        PLANNED_DECODE_BOUND.with(|bound| bound.set(None));
    }
}

struct StreamAbove;

impl StreamAbove {
    fn hold(bytes: usize) -> Self {
        STREAM_ABOVE_OVERRIDE.with(|threshold| threshold.set(Some(bytes)));
        Self
    }
}

impl Drop for StreamAbove {
    fn drop(&mut self) {
        STREAM_ABOVE_OVERRIDE.with(|threshold| threshold.set(None));
    }
}

fn planned_bytes(prepared: &PreparedBatch) -> Vec<(usize, CpuExecution)> {
    let mut planned = prepared
        .groups()
        .iter()
        .flat_map(|group| group.source_indices().iter().zip(group.images()))
        .map(|(&index, image)| (index, image.cpu_execution()))
        .collect::<Vec<_>>();
    planned.sort_by_key(|(index, _)| *index);
    planned
}

fn u8_samples(groups: &[super::super::CpuBatchGroup], source_index: usize) -> Vec<u8> {
    let group = groups
        .iter()
        .find(|group| group.source_indices() == [source_index])
        .expect("decoded group");
    let CpuBatchSamples::U8(samples) = group.samples() else {
        panic!("RGB8 decodes to u8 samples");
    };
    samples.clone()
}

fn is_decode_bound_error(error: &IndexedBatchError, index: usize, stage: BatchErrorStage) -> bool {
    error.index == index
        && matches!(
            &error.source,
            BatchItemError::Codec { stage: actual, source }
                if *actual == stage && **source == CpuExecution::decode_bound_error()
        )
}

fn single_worker_options_with(layout: BatchLayout) -> BatchDecodeOptions {
    let mut options = single_worker_options();
    options.layout = layout;
    options
}

fn all_u8_samples(result: &super::super::CpuBatchDecodeResult) -> Vec<(Vec<usize>, Vec<u8>)> {
    assert!(result.errors().is_empty(), "{:?}", result.errors());
    let mut groups = result
        .groups()
        .iter()
        .map(|group| {
            let CpuBatchSamples::U8(samples) = group.samples() else {
                panic!("8-bit decodes to u8 samples");
            };
            (group.source_indices().to_vec(), samples.clone())
        })
        .collect::<Vec<_>>();
    groups.sort();
    groups
}

#[test]
fn cpu_batch_streams_rgb8_and_gray8_plans_over_the_decode_bound() {
    let small = classic_rgb8(16, 16);
    let large = classic_rgb8(96, 64);
    let gray = classic_fixture(80, 72, 1, 8);
    let inputs = || {
        vec![
            EncodedImage::full(Arc::clone(&small)),
            EncodedImage::full(Arc::clone(&large)),
            EncodedImage::full(Arc::clone(&gray)),
            EncodedImage::new(
                Arc::clone(&large),
                DecodeRequest::Reduced {
                    scale: Downscale::Half,
                },
            ),
        ]
    };
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let mut decoder = CpuBatchDecoder::new(single_worker_options_with(layout));
        let unbounded = decoder.prepare(inputs()).expect("prepare without a bound");
        let planned = planned_bytes(&unbounded);
        let CpuExecution::OffsetPlan { bytes: small_bytes } = planned[0].1 else {
            panic!("the small image keeps its offset plan: {planned:?}");
        };
        let reference = all_u8_samples(
            &decoder
                .decode_prepared(&unbounded)
                .expect("decode without a bound"),
        );

        let _bound = PlannedBound::hold(small_bytes);
        let bounded = decoder.prepare(inputs()).expect("prepare under the bound");
        assert!(bounded.errors().is_empty(), "{:?}", bounded.errors());
        let planned = planned_bytes(&bounded);
        assert!(matches!(planned[0].1, CpuExecution::OffsetPlan { .. }));
        for (index, execution) in &planned[1..] {
            assert_eq!(*execution, CpuExecution::StreamedRows, "input {index}");
        }
        let streamed = all_u8_samples(&decoder.decode_prepared(&bounded).expect("decode"));
        assert_eq!(streamed, reference, "{layout:?}");
    }
}

#[test]
fn cpu_prepare_rejects_plans_over_the_bound_without_a_streamed_route() {
    let small = classic_rgb8(16, 16);
    let gray12 = classic_fixture(96, 64, 1, 12);
    let inputs = || {
        vec![
            EncodedImage::full(Arc::clone(&small)),
            EncodedImage::full(Arc::clone(&gray12)),
        ]
    };
    let mut decoder = CpuBatchDecoder::new(single_worker_options());
    let unbounded = decoder.prepare(inputs()).expect("prepare without a bound");
    let CpuExecution::OffsetPlan { bytes: small_bytes } = planned_bytes(&unbounded)[0].1 else {
        panic!("the small image keeps its offset plan");
    };
    let reference = decoder
        .decode_prepared(&unbounded)
        .expect("decode without a bound");
    assert_eq!(reference.errors(), &[] as &[IndexedBatchError]);

    let _bound = PlannedBound::hold(small_bytes);
    let rejected = decoder.prepare(inputs()).expect("prepare under the bound");
    assert_eq!(rejected.errors().len(), 1);
    assert!(is_decode_bound_error(
        &rejected.errors()[0],
        1,
        BatchErrorStage::Prepare
    ));
    let output = decoder
        .decode_prepared(&rejected)
        .expect("decode the admitted image");
    assert_eq!(
        u8_samples(output.groups(), 0),
        u8_samples(reference.groups(), 0)
    );

    // Shared preparation keeps the image for device decoders; the CPU
    // decoder still rejects it before decoding.
    let shared = prepare_batch(inputs(), single_worker_options()).expect("shared prepare");
    assert_eq!(shared.errors(), &[] as &[IndexedBatchError]);
    assert_eq!(
        planned_bytes(&shared)[1],
        (1, CpuExecution::ExceedsDecodeBound)
    );
    let output = decoder
        .decode_prepared(&shared)
        .expect("decode the shared batch");
    assert_eq!(output.errors().len(), 1);
    assert!(is_decode_bound_error(
        &output.errors()[0],
        1,
        BatchErrorStage::Decode
    ));
}

#[test]
fn cpu_batch_streams_large_rgb8_and_gray8_plans_below_the_decode_bound() {
    let small = classic_rgb8(16, 16);
    let large = classic_rgb8(96, 64);
    let gray = classic_fixture(80, 72, 1, 8);
    let gray12 = classic_fixture(96, 64, 1, 12);
    let inputs = || {
        vec![
            EncodedImage::full(Arc::clone(&small)),
            EncodedImage::full(Arc::clone(&large)),
            EncodedImage::full(Arc::clone(&gray)),
            EncodedImage::new(
                Arc::clone(&large),
                DecodeRequest::Region {
                    roi: j2k_core::Rect {
                        x: 13,
                        y: 7,
                        w: 61,
                        h: 50,
                    },
                },
            ),
            // Two Gray8 images in flight stream; a lone one decodes its plan.
            EncodedImage::full(Arc::clone(&gray)),
            EncodedImage::full(Arc::clone(&gray12)),
        ]
    };
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let mut decoder = CpuBatchDecoder::new(single_worker_options_with(layout));
        let planned_route = decoder.prepare(inputs()).expect("prepare small plans");
        let planned = planned_bytes(&planned_route);
        assert!(
            planned
                .iter()
                .all(|(_, execution)| matches!(execution, CpuExecution::OffsetPlan { .. })),
            "fixtures this small keep their plans: {planned:?}"
        );
        let CpuExecution::OffsetPlan { bytes: small_bytes } = planned[0].1 else {
            unreachable!("checked above");
        };
        let reference = decoder
            .decode_prepared(&planned_route)
            .expect("decode the plans");

        let _threshold = StreamAbove::hold(small_bytes);
        let streamed_route = decoder
            .prepare(inputs())
            .expect("prepare under the threshold");
        assert_eq!(streamed_route.errors(), &[] as &[IndexedBatchError]);
        let streamed = planned_bytes(&streamed_route);
        assert!(matches!(streamed[0].1, CpuExecution::OffsetPlan { .. }));
        for (index, execution) in &streamed[1..5] {
            assert!(
                matches!(execution, CpuExecution::StreamedRowsOrPlan { .. }),
                "input {index}: {execution:?}"
            );
        }
        // 12-bit output has no streamed route and stays within the bound.
        assert_eq!(streamed[5], planned[5], "{layout:?}");
        let output = decoder
            .decode_prepared(&streamed_route)
            .expect("decode streamed items");
        assert_eq!(output.errors(), &[] as &[IndexedBatchError]);
        assert_eq!(output.groups(), reference.groups(), "{layout:?}");
    }
}

#[test]
fn cpu_batch_streams_rgb8_images_whose_plans_pass_the_default_threshold() {
    // About 14 bytes of plan per 8-bit sample: 1.4 MiB of RGB8 output
    // plans several MiB, over the default threshold.
    let rgb = classic_rgb8(704, 704);
    let rgb12 = classic_fixture(704, 704, 3, 12);
    let inputs = || {
        vec![
            EncodedImage::full(Arc::clone(&rgb)),
            EncodedImage::full(Arc::clone(&rgb12)),
        ]
    };
    let mut decoder = CpuBatchDecoder::new(single_worker_options());
    let prepared = decoder.prepare(inputs()).expect("prepare");
    let planned = planned_bytes(&prepared);
    assert!(
        matches!(planned[0].1, CpuExecution::StreamedRowsOrPlan { .. }),
        "{planned:?}"
    );
    let CpuExecution::OffsetPlan { bytes } = planned[1].1 else {
        panic!("12-bit output keeps its plan: {planned:?}");
    };
    assert!(bytes > STREAM_ABOVE_PLANNED_BYTES, "{bytes}");
    let output = decoder.decode_prepared(&prepared).expect("decode");
    assert_eq!(output.errors(), &[] as &[IndexedBatchError]);

    // The streamed samples equal the plan route's.
    let _threshold = StreamAbove::hold(usize::MAX);
    let planned_route = decoder.prepare(inputs()).expect("prepare plans");
    assert!(matches!(
        planned_bytes(&planned_route)[0].1,
        CpuExecution::OffsetPlan { .. }
    ));
    let reference = decoder
        .decode_prepared(&planned_route)
        .expect("decode plans");
    assert_eq!(output.groups(), reference.groups());
}

#[test]
fn a_lone_gray8_image_decodes_its_plan_and_gray8_images_in_flight_stream() {
    let gray = classic_fixture(80, 72, 1, 8);
    let rgb = classic_rgb8(80, 72);
    let decode = |copies: usize, image: &Arc<[u8]>| {
        let inputs = vec![EncodedImage::full(Arc::clone(image)); copies];
        let mut decoder = CpuBatchDecoder::new(single_worker_options());
        let reference = decoder
            .decode_prepared(&decoder.prepare(inputs.clone()).expect("prepare plans"))
            .expect("decode plans");
        let plan_calls = decoder.workspace_stats().prepared_plan_decode_calls();
        let _threshold = StreamAbove::hold(0);
        let prepared = decoder.prepare(inputs).expect("prepare");
        assert!(planned_bytes(&prepared)
            .iter()
            .all(|(_, execution)| matches!(execution, CpuExecution::StreamedRowsOrPlan { .. })));
        let output = decoder.decode_prepared(&prepared).expect("decode");
        assert_eq!(output.groups(), reference.groups(), "{copies} copies");
        decoder.workspace_stats().prepared_plan_decode_calls() - plan_calls
    };
    assert_eq!(decode(1, &gray), 1, "a lone Gray8 image decodes its plan");
    assert_eq!(decode(2, &gray), 0, "Gray8 images in flight stream");
    assert_eq!(decode(1, &rgb), 0, "a lone RGB8 image streams");
}

#[test]
fn explicit_workers_decode_streamed_images_on_a_pool_of_that_many_threads() {
    let rgb = classic_rgb8(96, 64);
    let gray = classic_fixture(80, 72, 1, 8);
    for image in [&rgb, &gray] {
        let inputs = |copies: usize| vec![EncodedImage::full(Arc::clone(image)); copies];
        // Prepared on this thread, where the threshold override applies.
        let mut preparer = CpuBatchDecoder::new(single_worker_options());
        // (workers, images, pool threads): a multi-image group with an
        // explicit count of two or more workers decodes on a pool of that
        // many threads; one worker, the codec's own count, and a lone image
        // keep their threads.
        for (workers, copies, pool) in [
            (Some(1), 4, None),
            (Some(2), 3, Some(2)),
            (Some(3), 4, Some(3)),
            (Some(3), 1, None),
            (None, 4, None),
        ] {
            let reference = preparer
                .decode_prepared(&preparer.prepare(inputs(copies)).expect("prepare plans"))
                .expect("decode plans");
            let _threshold = StreamAbove::hold(0);
            let prepared = preparer.prepare(inputs(copies)).expect("prepare");
            assert!(planned_bytes(&prepared)
                .iter()
                .all(|(_, execution)| matches!(
                    execution,
                    CpuExecution::StreamedRowsOrPlan { .. }
                )));
            let mut decoder = CpuBatchDecoder::new(BatchDecodeOptions {
                workers: workers.and_then(NonZeroUsize::new),
                ..BatchDecodeOptions::default()
            });
            let decoded = decoder.decode_prepared(&prepared).expect("decode");
            let label = format!("{workers:?} workers, {copies} images");
            assert_eq!(decoded.errors(), &[] as &[IndexedBatchError], "{label}");
            assert_eq!(decoded.groups(), reference.groups(), "{label}");
            assert_eq!(decoder.stream_pool_threads(), pool, "{label}");
        }
    }
}
