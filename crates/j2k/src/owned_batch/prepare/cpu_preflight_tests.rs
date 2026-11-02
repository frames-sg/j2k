// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec, vec::Vec};
use core::num::NonZeroUsize;

use super::super::{
    prepare_batch, BatchDecodeOptions, BatchErrorStage, BatchItemError, BatchLayout,
    CpuBatchDecoder, CpuBatchSamples, CpuDecodeParallelism, CpuExecution, DecodeRequest,
    EncodedImage, IndexedBatchError, PreparedBatch,
};
use super::image::{PLANNED_DECODE_BOUND, STREAM_ABOVE_OVERRIDE, STREAM_ABOVE_PLANNED_BYTES};
use j2k_core::Downscale;

fn classic_fixture(width: u32, height: u32, components: u16, precision: u8) -> Arc<[u8]> {
    fixture(width, height, components, precision, false)
}

fn fixture(width: u32, height: u32, components: u16, precision: u8, ht: bool) -> Arc<[u8]> {
    seeded_fixture(width, height, components, precision, ht, 0)
}

/// A fixture whose samples differ for every `seed`: the same size, other
/// coded bytes.
fn seeded_fixture(
    width: u32,
    height: u32,
    components: u16,
    precision: u8,
    ht: bool,
    seed: u32,
) -> Arc<[u8]> {
    let count = width * height * u32::from(components);
    let samples = if precision <= 8 {
        (0..count)
            .map(|index| {
                u8::try_from((index * 7 + index / 97 + seed * index / 13) & 0xff)
                    .expect("masked sample")
            })
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
        use_ht_block_coding: ht,
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
    assert!(planned.iter().all(|(_, execution)| {
        execution.claim_bytes() <= crate::batch::allocation::GENERIC_WORKER_CLAIM_BYTES
    }));
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

/// `codestream` as JP2 in `color`, with `channel_definitions` in a cdef box.
fn wrap_jp2(
    codestream: &[u8],
    color: j2k_core::Colorspace,
    channel_definitions: &[crate::J2kChannelDefinition],
) -> Arc<[u8]> {
    Arc::from(
        crate::wrap_j2k_codestream(
            codestream,
            crate::J2kFileWrapOptions::jp2()
                .with_color(crate::J2kFileColorSpec::Enumerated(color))
                .with_metadata(crate::J2kFileBoxMetadata {
                    palette: None,
                    component_mappings: &[],
                    channel_definitions,
                }),
        )
        .expect("wrap JP2"),
    )
}

fn identity_rgb_channel_definitions() -> [crate::J2kChannelDefinition; 3] {
    [1, 2, 3].map(|color| crate::J2kChannelDefinition {
        channel_index: color - 1,
        channel_type: crate::J2kChannelType::Color,
        association: crate::J2kChannelAssociation::Color { index: color },
    })
}

/// `codestream` with the image area and tile grid moved to `origin` on
/// both axes. A multiple of every code-block and DWT alignment keeps the
/// same coded samples.
fn with_siz_origin(codestream: &[u8], origin: u32) -> Arc<[u8]> {
    let mut bytes = codestream.to_vec();
    assert_eq!(&bytes[..4], &[0xff, 0x4f, 0xff, 0x51], "SOC then SIZ");
    let mut add = |offset: usize| {
        let field = <[u8; 4]>::try_from(&bytes[offset..offset + 4]).expect("SIZ field");
        let value = u32::from_be_bytes(field) + origin;
        bytes[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
    };
    // Xsiz, Ysiz, XOsiz, YOsiz, then XTOsiz, YTOsiz.
    for offset in [8, 12, 16, 20, 32, 36] {
        add(offset);
    }
    Arc::from(bytes)
}

#[test]
fn gray8_and_rgb8_images_whose_region_decode_cannot_stream_keep_the_plan_route() {
    let rgb = classic_rgb8(96, 64);
    let gray = classic_fixture(80, 72, 1, 8);
    let streams = [
        Arc::clone(&rgb),
        Arc::clone(&gray),
        wrap_jp2(&gray, j2k_core::Colorspace::SGray, &[]),
    ];
    // A channel definition, or offsets, keep the region decode on whole
    // component planes, which it hands over at the end.
    let planes = [
        wrap_jp2(
            &rgb,
            j2k_core::Colorspace::SRgb,
            &identity_rgb_channel_definitions(),
        ),
        with_siz_origin(&rgb, 4096),
        with_siz_origin(&gray, 4096),
    ];
    let inputs = || {
        streams
            .iter()
            .chain(&planes)
            .map(|image| EncodedImage::full(Arc::clone(image)))
            .collect::<Vec<_>>()
    };
    let mut decoder = CpuBatchDecoder::new(single_worker_options());
    let reference = all_u8_samples(
        &decoder
            .decode_prepared(&decoder.prepare(inputs()).expect("prepare plans"))
            .expect("decode plans"),
    );
    // Moved, wrapped and plain images decode the same samples.
    let samples = |index: usize| {
        reference
            .iter()
            .find(|(indices, _)| indices.contains(&index))
            .map(|(indices, samples)| {
                let per_image = samples.len() / indices.len();
                let slot = indices.iter().position(|&i| i == index).expect("slot");
                samples[slot * per_image..(slot + 1) * per_image].to_vec()
            })
            .expect("decoded image")
    };
    for (index, plain) in [(2, 1), (3, 0), (4, 0), (5, 1)] {
        assert_eq!(samples(index), samples(plain), "input {index}");
    }

    {
        let _threshold = StreamAbove::hold(0);
        let prepared = decoder
            .prepare(inputs())
            .expect("prepare over the threshold");
        let planned = planned_bytes(&prepared);
        for (index, execution) in &planned {
            assert_eq!(
                matches!(execution, CpuExecution::StreamedRowsOrPlan { .. }),
                *index < streams.len(),
                "input {index}: {execution:?}"
            );
            assert_eq!(
                matches!(execution, CpuExecution::OffsetPlan { .. }),
                *index >= streams.len(),
                "input {index}: {execution:?}"
            );
        }
        let streamed = all_u8_samples(&decoder.decode_prepared(&prepared).expect("decode"));
        assert_eq!(streamed, reference);
    }

    // Over the bound, only the images that stream their rows are admitted.
    let _bound = PlannedBound::hold(0);
    let bounded = decoder.prepare(inputs()).expect("prepare under the bound");
    assert!(planned_bytes(&bounded)
        .iter()
        .all(|(_, execution)| *execution == CpuExecution::StreamedRows));
    let rejected = bounded.errors();
    assert_eq!(rejected.len(), planes.len(), "{rejected:?}");
    for (error, index) in rejected.iter().zip(streams.len()..) {
        assert!(
            is_decode_bound_error(error, index, BatchErrorStage::Prepare),
            "{error:?}"
        );
    }
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

#[derive(Debug, Clone, Copy)]
enum Pattern {
    /// Copies of one image.
    Copies(usize),
    /// Four different images.
    Distinct,
    /// One image twice and two others: `[a, b, a, c]`.
    Mixed,
    /// One image twice around another: `[a, b, a]`.
    AroundOne,
}

#[test]
fn groups_whose_images_stream_never_stage_and_others_stage_at_one_worker() {
    let limit = crate::batch::allocation::generic_batch_worker_limit();
    for is_gray in [false, true] {
        let (width, height, components) = if is_gray { (80, 72, 1) } else { (96, 64, 3) };
        let images = (0..4)
            .map(|seed| seeded_fixture(width, height, components, 8, false, seed))
            .collect::<Vec<_>>();
        let inputs = |pattern: Pattern| -> Vec<EncodedImage> {
            let picks: Vec<usize> = match pattern {
                Pattern::Copies(copies) => vec![0; copies],
                Pattern::Distinct => vec![0, 1, 2, 3],
                Pattern::Mixed => vec![0, 1, 0, 2],
                Pattern::AroundOne => vec![0, 1, 0],
            };
            picks
                .into_iter()
                .map(|pick| EncodedImage::full(Arc::clone(&images[pick])))
                .collect()
        };
        // Prepared on this thread, where the threshold override applies.
        let batch_preparer = CpuBatchDecoder::new(single_worker_options());
        // Copies, distinct and mixed images stream at every worker count, on
        // a pool of that many threads when there are several; the codec's
        // own count and a lone image keep their threads.
        for (workers, pattern, pool) in [
            (Some(1), Pattern::Copies(4), None),
            (Some(1), Pattern::Distinct, None),
            (Some(1), Pattern::Mixed, None),
            (Some(1), Pattern::AroundOne, None),
            (Some(2), Pattern::Copies(3), Some(2)),
            (Some(2), Pattern::Mixed, Some(2)),
            (Some(2), Pattern::AroundOne, Some(2)),
            (Some(3), Pattern::Distinct, Some(3)),
            (Some(4), Pattern::Mixed, Some(4)),
            (Some(6), Pattern::Copies(3), Some(6)),
            (Some(8), Pattern::Mixed, Some(8)),
            (Some(3), Pattern::Copies(1), None),
            (None, Pattern::Copies(4), None),
        ] {
            let label = format!("gray {is_gray}, {workers:?} workers, {pattern:?}");
            // Plans that do not stream stage at one worker, as before.
            let mut plan_decoder = CpuBatchDecoder::new(single_worker_options());
            let planned = batch_preparer
                .prepare(inputs(pattern))
                .expect("prepare plans");
            assert!(planned_bytes(&planned)
                .iter()
                .all(|(_, execution)| matches!(execution, CpuExecution::OffsetPlan { .. })));
            let reference = plan_decoder
                .decode_prepared(&planned)
                .expect("decode plans");
            let payloads = planned.groups()[0]
                .images()
                .iter()
                .map(|image| image.classic_plan().expect("classic plan").payload_count() as u64)
                .sum::<u64>();
            let staged = planned.groups()[0].images().len() > 1;
            assert_eq!(
                plan_decoder.workspace_stats().flattened_payload_jobs(),
                if staged { payloads } else { 0 },
                "{label}"
            );
            let _threshold = StreamAbove::hold(0);
            let prepared = batch_preparer.prepare(inputs(pattern)).expect("prepare");
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
            let output = decoder.decode_prepared(&prepared).expect("decode");
            assert_eq!(output.errors(), &[] as &[IndexedBatchError], "{label}");
            assert_eq!(output.groups(), reference.groups(), "{label}");
            let stats = decoder.workspace_stats();
            assert_eq!(stats.flattened_payload_jobs(), 0, "{label}");
            assert_eq!(stats.cross_image_entropy_windows(), 0, "{label}");
            assert_eq!(
                decoder.stream_pool_threads(),
                pool.map(|threads: usize| threads.min(limit)),
                "{label}"
            );
        }
    }
}

#[test]
fn htj2k_groups_stream_with_one_or_two_workers() {
    let ht = fixture(96, 64, 3, 8, true);
    let inputs = || vec![EncodedImage::full(Arc::clone(&ht)); 3];
    let mut batch_preparer = CpuBatchDecoder::new(single_worker_options());
    let reference = batch_preparer
        .decode_prepared(&batch_preparer.prepare(inputs()).expect("prepare plans"))
        .expect("decode plans");
    let _threshold = StreamAbove::hold(0);
    let prepared = batch_preparer.prepare(inputs()).expect("prepare");
    assert!(planned_bytes(&prepared)
        .iter()
        .all(|(_, execution)| matches!(execution, CpuExecution::StreamedRowsOrPlan { .. })));
    // HTJ2K gains nothing from the staged route's back-to-back code-blocks.
    for (workers, pool) in [(1, None), (2, Some(2))] {
        let mut decoder = CpuBatchDecoder::new(BatchDecodeOptions {
            workers: NonZeroUsize::new(workers),
            ..BatchDecodeOptions::default()
        });
        let output = decoder.decode_prepared(&prepared).expect("decode");
        assert_eq!(output.errors(), &[] as &[IndexedBatchError], "{workers}");
        assert_eq!(output.groups(), reference.groups(), "{workers}");
        assert_eq!(
            decoder.workspace_stats().cross_image_entropy_windows(),
            0,
            "{workers}"
        );
        assert_eq!(decoder.stream_pool_threads(), pool, "{workers}");
    }
}

#[test]
fn explicit_worker_counts_set_the_policy_of_images_decoded_one_at_a_time() {
    use super::super::cpu_execute::image_policy;
    let workers = NonZeroUsize::new;
    let serial = Some(CpuDecodeParallelism::Serial);
    // Several images: streamed ones in the pool of two or more threads, and
    // plans serial; one worker decodes serially.
    assert_eq!(image_policy(3, true, workers(1)), (false, serial));
    assert_eq!(image_policy(3, true, workers(2)), (true, serial));
    assert_eq!(image_policy(3, false, workers(2)), (false, serial));
    // A lone image and automatic workers keep the scheduler's choice.
    assert_eq!(image_policy(1, true, workers(2)), (false, None));
    assert_eq!(image_policy(3, true, None), (false, None));
}

#[test]
fn batch_images_share_the_stream_pool_only_when_they_fill_it() {
    use super::super::cpu_execute::shares_stream_pool;
    let limit = crate::batch::allocation::generic_batch_worker_limit();
    // The session's workers, and so its stream pool, stop at the limit.
    let cases = [
        // Eight images on 2, 3, 4 or 8 workers, and three on 2, keep an image
        // in flight for every thread of the pool.
        (8, 2, true),
        (8, 3, true),
        (8, 4, true),
        (8, 8, true),
        (3, 2, true),
        // Eight on 6 run four at a time, five on 4 three, two on 18 two:
        // fewer images in flight than threads.
        (8, 6, false),
        (5, 4, false),
        (2, 18, false),
    ];
    for (images, workers, shared) in cases.into_iter().filter(|case| case.1 <= limit) {
        let options = crate::TileBatchOptions::new(NonZeroUsize::new(workers));
        assert_eq!(
            shares_stream_pool(images, options, workers).expect("plan"),
            shared,
            "{images} on {workers}"
        );
    }
}

#[test]
fn images_far_above_their_group_mean_keep_the_lone_rings() {
    use super::super::cpu_execute::shared_ring_bytes;
    // Seven crops of about the same coded size share the rings; one noise
    // image of the same shape with 3.7 times their bytes keeps its own.
    let crops = [
        3_585_095, 3_622_220, 3_720_453, 3_745_469, 3_787_355, 3_807_463, 3_648_167,
    ];
    let mut group = crops.to_vec();
    group.push(13_840_854);
    let limit = shared_ring_bytes(group.into_iter());
    assert!(crops.iter().all(|&bytes| bytes <= limit), "{limit}");
    assert!(13_840_854 > limit, "{limit}");
    // A uniform group shares them all, half again above the mean at most.
    assert_eq!(shared_ring_bytes([100, 100, 100].into_iter()), 150);
    assert_eq!(shared_ring_bytes([90, 100, 110, 160].into_iter()), 172);
    assert_eq!(shared_ring_bytes(core::iter::empty()), 0);
}

#[test]
fn plans_beyond_the_flattening_cap_decode_one_image_at_a_time() {
    use super::super::cpu_fast::{
        classic_flattened_metadata_bytes, FLATTENED_METADATA_CAP_OVERRIDE,
    };
    struct FlattenedCap;
    impl FlattenedCap {
        fn hold(bytes: usize) -> Self {
            FLATTENED_METADATA_CAP_OVERRIDE.with(|cap| cap.set(Some(bytes)));
            Self
        }
    }
    impl Drop for FlattenedCap {
        fn drop(&mut self) {
            FLATTENED_METADATA_CAP_OVERRIDE.with(|cap| cap.set(None));
        }
    }
    let images = (0..3)
        .map(|seed| seeded_fixture(96, 64, 3, 8, false, seed))
        .collect::<Vec<_>>();
    let inputs = || {
        images
            .iter()
            .map(|bytes| EncodedImage::full(Arc::clone(bytes)))
            .collect::<Vec<_>>()
    };
    // Small plans that do not stream: one worker stages them.
    let decoder = CpuBatchDecoder::new(single_worker_options());
    let prepared = decoder.prepare(inputs()).expect("prepare plans");
    assert!(planned_bytes(&prepared)
        .iter()
        .all(|(_, execution)| matches!(execution, CpuExecution::OffsetPlan { .. })));
    let payloads = prepared.groups()[0]
        .images()
        .iter()
        .map(|image| image.classic_plan().expect("classic plan").payload_count())
        .sum::<usize>();
    let needed = classic_flattened_metadata_bytes(3, payloads);
    let mut reference = None;
    // At the cap the plans stage; one byte below it each decodes on its
    // own, where flattening them would have failed the batch.
    for (cap, staged) in [(needed, true), (needed - 1, false), (0, false)] {
        let _cap = FlattenedCap::hold(cap);
        let mut decoder = CpuBatchDecoder::new(single_worker_options());
        let (groups, errors) = decoder
            .decode_prepared(&prepared)
            .expect("decode")
            .into_parts();
        assert_eq!(errors, [], "cap {cap}");
        match &reference {
            Some(reference) => assert_eq!(&groups, reference, "cap {cap}"),
            None => reference = Some(groups),
        }
        assert_eq!(
            decoder.workspace_stats().flattened_payload_jobs(),
            if staged { payloads as u64 } else { 0 },
            "cap {cap}"
        );
        assert_eq!(decoder.parallel_plan_decodes(), 0, "cap {cap}");
    }
}

#[test]
fn planned_images_among_streamed_ones_keep_the_group_worker_policy() {
    let limit = crate::batch::allocation::generic_batch_worker_limit();
    let rgb = classic_rgb8(96, 64);
    let gray = classic_fixture(80, 72, 1, 8);
    // Copies `a` stream; `b` keeps its offset plan: for RGB8 the same JP2
    // with identity channel definitions, for Gray8 offsets.
    let srgb = j2k_core::Colorspace::SRgb;
    let cases = [
        (
            wrap_jp2(&rgb, srgb, &[]),
            wrap_jp2(&rgb, srgb, &identity_rgb_channel_definitions()),
            false,
        ),
        (Arc::clone(&gray), with_siz_origin(&gray, 4096), true),
    ];
    for (copy, planned, is_gray) in cases {
        let inputs = || {
            [&copy, &planned, &copy]
                .map(|bytes| EncodedImage::full(Arc::clone(bytes)))
                .to_vec()
        };
        let mut batch_preparer = CpuBatchDecoder::new(single_worker_options());
        let reference = batch_preparer
            .decode_prepared(&batch_preparer.prepare(inputs()).expect("prepare plans"))
            .expect("decode plans");
        let _threshold = StreamAbove::hold(0);
        let prepared = batch_preparer.prepare(inputs()).expect("prepare");
        assert_eq!(prepared.groups().len(), 1, "gray {is_gray}");
        let streams = planned_bytes(&prepared)
            .iter()
            .map(|(_, execution)| execution.streams())
            .collect::<Vec<_>>();
        assert_eq!(streams, [true, false, true], "gray {is_gray}");
        // `[a, b, a]`: `b` decodes its plan without nested parallelism, as
        // the group's other plans would, and in the pool of two threads.
        for workers in [1, 2] {
            let label = format!("gray {is_gray}, {workers} workers");
            let mut decoder = CpuBatchDecoder::new(BatchDecodeOptions {
                workers: NonZeroUsize::new(workers),
                ..BatchDecodeOptions::default()
            });
            let output = decoder.decode_prepared(&prepared).expect("decode");
            assert_eq!(output.errors(), &[] as &[IndexedBatchError], "{label}");
            assert_eq!(output.groups(), reference.groups(), "{label}");
            assert_eq!(
                decoder.workspace_stats().flattened_payload_jobs(),
                0,
                "{label}"
            );
            assert_eq!(decoder.parallel_plan_decodes(), 0, "{label}");
            assert_eq!(
                decoder.stream_pool_threads(),
                (workers > 1).then_some(workers.min(limit)),
                "{label}"
            );
        }
    }
}
