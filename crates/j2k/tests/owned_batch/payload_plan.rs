// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{prepare_batch, BatchDecodeOptions, EncodedImage};

pub(super) fn native_prepared_plan(
    plan: &j2k::PreparedHtj2kPlan,
) -> &j2k_native::J2kReferencedHtj2kPlan {
    plan.adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedHtj2kPlan>()
        .expect("j2k-native prepared-plan adapter")
}

pub(super) fn native_prepared_classic_plan(
    plan: &j2k::PreparedClassicPlan,
) -> &j2k_native::J2kReferencedClassicPlan {
    plan.adapter_view()
        .downcast_ref::<j2k_native::J2kReferencedClassicPlan>()
        .expect("j2k-native prepared classic-plan adapter")
}

pub(super) fn assert_prepared_ht_payload_ranges_reconstruct_owned_bytes(bytes: Vec<u8>) {
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::<[u8]>::from(bytes))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare HTJ2K offset plan");
    let prepared_image = &prepared.groups()[0].images()[0];
    let referenced = prepared_image.htj2k_plan().expect("referenced HTJ2K plan");
    let geometry = native_prepared_plan(referenced)
        .grayscale_geometry()
        .expect("grayscale referenced geometry");
    let native_image = j2k_native::Image::new(
        prepared_image.bytes(),
        &j2k_native::DecodeSettings::strict(),
    )
    .expect("parse owned HTJ2K plan source");
    let mut context = j2k_native::DecoderContext::default();
    let owned = native_image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build owned HTJ2K direct plan");
    let mut payload_cursor = 0usize;

    for (owned_step, referenced_step) in owned.steps.iter().zip(&geometry.steps) {
        let (
            j2k_native::J2kDirectGrayscaleStep::HtSubBand(owned_sub_band),
            j2k_native::J2kDirectGrayscaleStep::HtSubBand(referenced_sub_band),
        ) = (owned_step, referenced_step)
        else {
            continue;
        };
        for (owned_job, referenced_job) in owned_sub_band.jobs.iter().zip(&referenced_sub_band.jobs)
        {
            assert!(referenced_job.data.is_empty());
            let payload = referenced
                .payload(payload_cursor)
                .expect("first payload record for referenced HT job");
            payload_cursor += 1;
            assert_eq!(payload.cleanup.length, owned_job.cleanup_length as usize);
            let cleanup_end = payload.cleanup.end().expect("cleanup range end");
            let mut reconstructed = prepared_image
                .bytes()
                .get(payload.cleanup.offset..cleanup_end)
                .expect("cleanup range inside retained encoded owner")
                .to_vec();
            let mut refinement_bytes = 0usize;
            if let Some(refinement) = payload.refinement {
                let refinement_end = refinement.end().expect("refinement range end");
                let bytes = prepared_image
                    .bytes()
                    .get(refinement.offset..refinement_end)
                    .expect("refinement range inside retained encoded owner");
                refinement_bytes += bytes.len();
                reconstructed.extend_from_slice(bytes);
            }
            while refinement_bytes < owned_job.refinement_length as usize {
                let continuation = referenced
                    .payload(payload_cursor)
                    .expect("continuation record for referenced HT job");
                payload_cursor += 1;
                assert_eq!(continuation.cleanup.length, 0);
                let refinement = continuation
                    .refinement
                    .expect("continuation record carries refinement bytes");
                let refinement_end = refinement.end().expect("continuation range end");
                let bytes = prepared_image
                    .bytes()
                    .get(refinement.offset..refinement_end)
                    .expect("continuation range inside retained encoded owner");
                assert!(!bytes.is_empty());
                refinement_bytes += bytes.len();
                reconstructed.extend_from_slice(bytes);
            }
            assert_eq!(refinement_bytes, owned_job.refinement_length as usize);
            assert_eq!(reconstructed, owned_job.data);
        }
    }
    assert_eq!(payload_cursor, referenced.payload_count());
}

pub(super) fn assert_prepared_classic_payload_ranges_reconstruct_owned_bytes(bytes: Vec<u8>) {
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::<[u8]>::from(bytes))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare classic offset plan");
    let prepared_image = &prepared.groups()[0].images()[0];
    let referenced = prepared_image
        .classic_plan()
        .expect("referenced classic plan");
    let geometry = native_prepared_classic_plan(referenced)
        .grayscale_geometry()
        .expect("grayscale referenced classic geometry");
    let native_image = j2k_native::Image::new(
        prepared_image.bytes(),
        &j2k_native::DecodeSettings::strict(),
    )
    .expect("parse owned classic plan source");
    let mut context = j2k_native::DecoderContext::default();
    let owned = native_image
        .build_direct_grayscale_plan_with_context(&mut context)
        .expect("build owned classic direct plan");
    let mut payload_cursor = 0usize;

    for (owned_step, referenced_step) in owned.steps.iter().zip(&geometry.steps) {
        let (
            j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(owned_sub_band),
            j2k_native::J2kDirectGrayscaleStep::ClassicSubBand(referenced_sub_band),
        ) = (owned_step, referenced_step)
        else {
            continue;
        };
        for (owned_job, referenced_job) in owned_sub_band.jobs.iter().zip(&referenced_sub_band.jobs)
        {
            assert_eq!(referenced_job.data.capacity(), 0);
            assert_eq!(referenced_job.segments, owned_job.segments);
            let payload = referenced
                .payload(payload_cursor)
                .expect("payload descriptor for referenced classic job");
            payload_cursor += 1;
            let end_range = payload.end_range().expect("classic fragment span end");
            let mut reconstructed = Vec::with_capacity(payload.combined_length);
            for range_index in payload.first_range..end_range {
                let range = referenced
                    .range(range_index)
                    .expect("classic fragment range");
                let range_end = range.end().expect("classic fragment range end");
                reconstructed.extend_from_slice(
                    prepared_image
                        .bytes()
                        .get(range.offset..range_end)
                        .expect("classic fragment inside retained encoded owner"),
                );
            }
            assert_eq!(reconstructed.len(), payload.combined_length);
            assert_eq!(reconstructed, owned_job.data);
        }
    }
    assert_eq!(payload_cursor, referenced.payload_count());
    assert_eq!(
        referenced
            .payloads()
            .map(|payload| payload.range_count)
            .sum::<usize>(),
        referenced.range_count()
    );
}
