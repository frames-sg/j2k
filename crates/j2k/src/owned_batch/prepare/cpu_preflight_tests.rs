// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec, vec::Vec};
use core::num::NonZeroUsize;

use super::super::cpu_execute::STAGED_PLANS_CAP_OVERRIDE;
use super::super::{
    prepare_batch, BatchDecodeOptions, BatchErrorStage, BatchItemError, BatchLayout,
    CpuBatchDecoder, CpuBatchSamples, CpuExecution, DecodeRequest, EncodedImage, IndexedBatchError,
    PreparedBatch,
};
use super::image::{PLANNED_DECODE_BOUND, STREAM_ABOVE_OVERRIDE, STREAM_ABOVE_PLANNED_BYTES};
use j2k_core::Downscale;

fn classic_fixture(width: u32, height: u32, components: u16, precision: u8) -> Arc<[u8]> {
    fixture(width, height, components, precision, false)
}

fn fixture(width: u32, height: u32, components: u16, precision: u8, ht: bool) -> Arc<[u8]> {
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

struct StagedPlansCap;

impl StagedPlansCap {
    fn hold(bytes: Option<usize>) -> Self {
        STAGED_PLANS_CAP_OVERRIDE.with(|cap| cap.set(bytes));
        Self
    }
}

impl Drop for StagedPlansCap {
    fn drop(&mut self) {
        STAGED_PLANS_CAP_OVERRIDE.with(|cap| cap.set(None));
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

/// `codestream` as JP2, with `channel_definitions` in a cdef box.
fn wrap_jp2(codestream: &[u8], channel_definitions: &[crate::J2kChannelDefinition]) -> Arc<[u8]> {
    let color = if channel_definitions.is_empty() {
        j2k_core::Colorspace::SGray
    } else {
        j2k_core::Colorspace::SRgb
    };
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
    let streams = [Arc::clone(&rgb), Arc::clone(&gray), wrap_jp2(&gray, &[])];
    // A channel definition, or offsets, keep the region decode on whole
    // component planes, which it hands over at the end.
    let planes = [
        wrap_jp2(&rgb, &identity_rgb_channel_definitions()),
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
        let decoded = all_u8_samples(&decoder.decode_prepared(&prepared).expect("decode"));
        assert_eq!(decoded, reference);
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
        // One worker keeps several images' plans within the cap (see
        // `explicit_workers_choose_the_staged_plans_or_the_stream_pool`).
        let _cap = StagedPlansCap::hold(Some(0));
        let output = decoder.decode_prepared(&prepared).expect("decode");
        assert_eq!(output.groups(), reference.groups(), "{copies} copies");
        decoder.workspace_stats().prepared_plan_decode_calls() - plan_calls
    };
    assert_eq!(decode(1, &gray), 1, "a lone Gray8 image decodes its plan");
    assert_eq!(decode(2, &gray), 0, "Gray8 images in flight stream");
    assert_eq!(decode(1, &rgb), 0, "a lone RGB8 image streams");
}

#[test]
fn explicit_workers_choose_the_staged_plans_or_the_stream_pool() {
    let rgb = classic_rgb8(96, 64);
    let gray = classic_fixture(80, 72, 1, 8);
    for (image, is_gray) in [(&rgb, false), (&gray, true)] {
        let inputs = |copies: usize| vec![EncodedImage::full(Arc::clone(image)); copies];
        // Prepared on this thread, where the threshold override applies.
        let mut preparer = CpuBatchDecoder::new(single_worker_options());
        // (workers, images, cap on the group's plans, staged windows, pool
        // threads): one worker (up to four for Gray8) keeps a multi-image
        // group's plans on the staged route while they fit the cap and
        // streams them otherwise; more workers stream the images on a pool of
        // that many threads; the codec's own count and a lone image keep
        // their threads.
        let streams = |workers: usize| (!is_gray).then_some(workers);
        let five_workers = 5.min(crate::batch::allocation::generic_batch_worker_limit());
        for (workers, copies, cap, staged, pool) in [
            (Some(1), 4, None, true, None),
            (Some(2), 3, None, is_gray, streams(2)),
            (Some(3), 4, None, is_gray, streams(3)),
            (Some(1), 4, Some(1), false, None),
            (Some(2), 3, Some(1), false, Some(2)),
            (Some(5), 6, None, false, Some(five_workers)),
            (Some(3), 1, None, false, None),
            (None, 4, None, false, None),
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
            let _cap = StagedPlansCap::hold(cap);
            let decoded = decoder.decode_prepared(&prepared).expect("decode");
            let label =
                format!("gray {is_gray}, {workers:?} workers, {copies} images, cap {cap:?}");
            assert_eq!(decoded.errors(), &[] as &[IndexedBatchError], "{label}");
            assert_eq!(decoded.groups(), reference.groups(), "{label}");
            assert_eq!(
                decoder.workspace_stats().cross_image_entropy_windows() > 0,
                staged,
                "{label}"
            );
            assert_eq!(decoder.stream_pool_threads(), pool, "{label}");
        }
    }
}

#[test]
fn htj2k_groups_stream_with_one_or_two_workers() {
    let ht = fixture(96, 64, 3, 8, true);
    let inputs = || vec![EncodedImage::full(Arc::clone(&ht)); 3];
    let mut preparer = CpuBatchDecoder::new(single_worker_options());
    let reference = preparer
        .decode_prepared(&preparer.prepare(inputs()).expect("prepare plans"))
        .expect("decode plans");
    let _threshold = StreamAbove::hold(0);
    let prepared = preparer.prepare(inputs()).expect("prepare");
    assert!(planned_bytes(&prepared)
        .iter()
        .all(|(_, execution)| matches!(execution, CpuExecution::StreamedRowsOrPlan { .. })));
    // HTJ2K gains nothing from the staged route's back-to-back code-blocks.
    for (workers, pool) in [(1, None), (2, Some(2))] {
        let mut decoder = CpuBatchDecoder::new(BatchDecodeOptions {
            workers: NonZeroUsize::new(workers),
            ..BatchDecodeOptions::default()
        });
        let decoded = decoder.decode_prepared(&prepared).expect("decode");
        assert_eq!(decoded.errors(), &[] as &[IndexedBatchError], "{workers}");
        assert_eq!(decoded.groups(), reference.groups(), "{workers}");
        assert_eq!(
            decoder.workspace_stats().cross_image_entropy_windows(),
            0,
            "{workers}"
        );
        assert_eq!(decoder.stream_pool_threads(), pool, "{workers}");
    }
}
