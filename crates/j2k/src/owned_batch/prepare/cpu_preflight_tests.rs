// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec, vec::Vec};
use core::num::NonZeroUsize;

use super::super::{
    prepare_batch, BatchDecodeOptions, BatchErrorStage, BatchItemError, BatchLayout,
    CpuBatchDecoder, CpuBatchSamples, CpuExecution, DecodeRequest, EncodedImage, IndexedBatchError,
    PreparedBatch,
};
use super::image::PLANNED_DECODE_BOUND;
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
