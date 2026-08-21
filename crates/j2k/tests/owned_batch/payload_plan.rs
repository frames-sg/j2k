// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{prepare_batch, BatchDecodeOptions, EncodedImage};

use super::fixtures::{
    classic_gray8_fixture, htj2k_gray8_fixture, htj2k_native_fixture, rgb8_fixture,
};

#[test]
fn classic_and_ht_plans_share_one_image_geometry_contract() {
    let classic_bytes = Arc::<[u8]>::from(classic_gray8_fixture(8, 6));
    let ht_bytes = Arc::<[u8]>::from(htj2k_gray8_fixture(8, 6));

    let classic_batch = prepare_batch(
        vec![EncodedImage::full(classic_bytes)],
        BatchDecodeOptions::default(),
    )
    .expect("prepare classic geometry");
    let ht_batch = prepare_batch(
        vec![EncodedImage::full(ht_bytes)],
        BatchDecodeOptions::default(),
    )
    .expect("prepare HT geometry");
    let classic = classic_batch.groups()[0].images()[0]
        .classic_plan()
        .expect("classic plan");
    let ht = ht_batch.groups()[0].images()[0]
        .htj2k_plan()
        .expect("HT plan");
    let classic_image = classic.geometry().image_geometry();
    let ht_image = ht.geometry().image_geometry();

    for geometry in [classic_image, ht_image] {
        assert!(!geometry.is_empty());
        assert!(geometry.is_grayscale());
        assert!(!geometry.is_color());
        assert!(!geometry.is_rgba());
        assert_eq!(geometry.full_dimensions(), (8, 6));
        let output = geometry.output_rect();
        assert_eq!((output.x1 - output.x0, output.y1 - output.y0), (8, 6));
        assert_eq!(geometry.tiles().len(), 1);
        assert!(geometry.grayscale_geometry().is_some());
        assert!(geometry.color_geometry().is_none());
        assert!(geometry.rgba_geometry().is_none());
        assert_eq!(
            geometry.uniform_wavelet_transform(),
            Some(j2k_native::J2kWaveletTransform::Reversible53)
        );
    }

    assert_eq!(classic.is_grayscale(), classic_image.is_grayscale());
    assert_eq!(ht.is_grayscale(), ht_image.is_grayscale());
}

#[test]
fn classic_and_ht_color_plans_use_the_shared_component_geometry() {
    let sources = [
        Arc::<[u8]>::from(rgb8_fixture()),
        Arc::<[u8]>::from(htj2k_native_fixture(3, 8, false, 4, 4)),
    ];
    let prepared = prepare_batch(
        sources.into_iter().map(EncodedImage::full).collect(),
        BatchDecodeOptions::default(),
    )
    .expect("prepare classic and HT color geometry");

    for image in prepared
        .groups()
        .iter()
        .flat_map(j2k::PreparedBatchGroup::images)
    {
        let geometry = image
            .classic_plan()
            .map(j2k::PreparedClassicPlan::image_geometry)
            .or_else(|| {
                image
                    .htj2k_plan()
                    .map(j2k::PreparedHtj2kPlan::image_geometry)
            })
            .expect("prepared image geometry");
        assert!(!geometry.is_grayscale());
        assert!(geometry.is_color());
        assert!(!geometry.is_rgba());
        let color = geometry.color_geometry().expect("single-tile RGB geometry");
        assert_eq!(color.component_plans.len(), 3);
        assert_eq!(color.dimensions, (4, 4));
    }
}

pub(super) fn native_prepared_plan(plan: &j2k::PreparedHtj2kPlan) -> &j2k::Htj2kPreparedGeometry {
    plan.geometry()
}

pub(super) fn native_prepared_classic_plan(
    plan: &j2k::PreparedClassicPlan,
) -> &j2k::ClassicPreparedGeometry {
    plan.geometry()
}

pub(super) fn assert_prepared_ht_payload_ranges_reconstruct_owned_bytes(bytes: Vec<u8>) {
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::<[u8]>::from(bytes))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare HTJ2K offset plan");
    let prepared_image = &prepared.groups()[0].images()[0];
    let referenced = prepared_image.htj2k_plan().expect("referenced HTJ2K plan");
    let shared = referenced.clone();
    assert!(core::ptr::eq(referenced.geometry(), shared.geometry()));
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
    let shared = referenced.clone();
    assert!(core::ptr::eq(referenced.geometry(), shared.geometry()));
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
