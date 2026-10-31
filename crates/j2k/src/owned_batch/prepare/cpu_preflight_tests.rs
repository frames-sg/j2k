// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{sync::Arc, vec, vec::Vec};
use core::num::NonZeroUsize;

use super::super::{
    prepare_batch, BatchDecodeOptions, BatchErrorStage, BatchItemError, CpuBatchDecoder,
    CpuBatchSamples, CpuExecution, EncodedImage, IndexedBatchError, PreparedBatch,
};
use super::image::PLANNED_DECODE_BOUND;

fn classic_rgb8(width: u32, height: u32) -> Arc<[u8]> {
    let samples = (0..width * height * 3)
        .map(|index| u8::try_from((index * 7 + index / 97) & 0xff).expect("masked sample"))
        .collect::<Vec<_>>();
    let options = j2k_native::EncodeOptions {
        reversible: true,
        num_decomposition_levels: 3,
        ..j2k_native::EncodeOptions::default()
    };
    Arc::from(
        j2k_native::encode(&samples, width, height, 3, 8, false, &options)
            .expect("encode classic RGB8 fixture"),
    )
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

#[test]
fn cpu_prepare_rejects_plans_over_the_decode_bound_before_decoding() {
    let small = classic_rgb8(16, 16);
    let large = classic_rgb8(96, 64);
    let inputs = || {
        vec![
            EncodedImage::full(Arc::clone(&small)),
            EncodedImage::full(Arc::clone(&large)),
        ]
    };
    let mut decoder = CpuBatchDecoder::new(single_worker_options());
    let unbounded = decoder.prepare(inputs()).expect("prepare without a bound");
    let planned = planned_bytes(&unbounded);
    let [(0, CpuExecution::OffsetPlan { bytes: small_bytes }), (1, CpuExecution::OffsetPlan { bytes: large_bytes })] =
        planned[..]
    else {
        panic!("both images keep offset plans: {planned:?}");
    };
    assert!(small_bytes < large_bytes, "{small_bytes} < {large_bytes}");
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
    assert_eq!(planned_bytes(&rejected).len(), 1);
    let decoded = decoder
        .decode_prepared(&rejected)
        .expect("decode the admitted image");
    assert_eq!(
        u8_samples(decoded.groups(), 0),
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
    let decoded = decoder
        .decode_prepared(&shared)
        .expect("decode the shared batch");
    assert_eq!(decoded.errors().len(), 1);
    assert!(is_decode_bound_error(
        &decoded.errors()[0],
        1,
        BatchErrorStage::Decode
    ));
    assert_eq!(
        u8_samples(decoded.groups(), 0),
        u8_samples(reference.groups(), 0)
    );
}
