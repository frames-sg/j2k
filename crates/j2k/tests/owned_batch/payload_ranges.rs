// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, wrap_j2k_codestream, BatchDecodeOptions, EncodedImage, J2kFileWrapOptions,
    PreparationDepth,
};
use j2k_native::{encode, EncodeOptions};

use super::fixtures::{classic_gray8_fixture, htj2k_gray8_fixture};
use super::payload_plan::{
    assert_prepared_classic_payload_ranges_reconstruct_owned_bytes,
    assert_prepared_ht_payload_ranges_reconstruct_owned_bytes, native_prepared_plan,
};

#[test]
fn prepared_htj2k_uses_codestream_ranges_without_retaining_payload_copies() {
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::<[u8]>::from(htj2k_gray8_fixture(
            8, 8,
        )))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare HTJ2K offset plan");
    let image = &prepared.groups()[0].images()[0];

    assert_eq!(image.preparation_depth(), PreparationDepth::Htj2kOffsetPlan);
    let codestream = image.codestream_range();
    assert_eq!(codestream.offset, 0);
    assert_eq!(codestream.length, image.bytes().len());
    let plan = image.htj2k_plan().expect("HTJ2K reference plan");
    assert!(!plan.is_empty());
    assert!(plan.payloads().all(|payload| {
        payload
            .cleanup
            .end()
            .is_some_and(|end| end <= codestream.length)
            && payload
                .refinement
                .is_none_or(|range| range.end().is_some_and(|end| end <= codestream.length))
    }));
    let geometry = native_prepared_plan(plan)
        .grayscale_geometry()
        .expect("grayscale geometry");
    let retained_payload_bytes = geometry
        .steps
        .iter()
        .filter_map(|step| match step {
            j2k_native::J2kDirectGrayscaleStep::HtSubBand(sub_band) => Some(
                sub_band
                    .jobs
                    .iter()
                    .map(|job| job.data.capacity())
                    .sum::<usize>(),
            ),
            _ => None,
        })
        .sum::<usize>();
    assert_eq!(retained_payload_bytes, 0);
}

#[test]
fn prepared_raw_htj2k_payload_ranges_reconstruct_original_job_bytes() {
    assert_prepared_ht_payload_ranges_reconstruct_owned_bytes(htj2k_gray8_fixture(8, 8));
}

#[test]
fn prepared_jph_payload_ranges_reconstruct_original_job_bytes() {
    let codestream = htj2k_gray8_fixture(8, 8);
    let wrapped = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jph()).expect("wrap JPH");
    assert_prepared_ht_payload_ranges_reconstruct_owned_bytes(wrapped);
}

#[test]
fn prepared_raw_classic_payload_ranges_reconstruct_original_job_bytes() {
    assert_prepared_classic_payload_ranges_reconstruct_owned_bytes(classic_gray8_fixture(8, 8));
}

#[test]
fn prepared_jp2_classic_payload_ranges_reconstruct_original_job_bytes() {
    let codestream = classic_gray8_fixture(8, 8);
    let wrapped = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jp2()).expect("wrap JP2");
    assert_prepared_classic_payload_ranges_reconstruct_owned_bytes(wrapped);
}

#[test]
fn prepared_raw_jp2_and_jph_codestream_ranges_reconstruct_original_codestreams() {
    let classic = encode(
        &(0_u8..64).collect::<Vec<_>>(),
        8,
        8,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode classic J2K fixture");
    let ht = htj2k_gray8_fixture(8, 8);
    let inputs = [
        classic.clone(),
        wrap_j2k_codestream(&classic, J2kFileWrapOptions::jp2()).expect("wrap JP2"),
        wrap_j2k_codestream(&ht, J2kFileWrapOptions::jph()).expect("wrap JPH"),
    ];
    let expected = [&classic[..], &classic[..], &ht[..]];

    for (bytes, expected_codestream) in inputs.into_iter().zip(expected) {
        let prepared = prepare_batch(
            vec![EncodedImage::full(Arc::<[u8]>::from(bytes))],
            BatchDecodeOptions::default(),
        )
        .expect("prepare encoded owner");
        let image = &prepared.groups()[0].images()[0];
        let range = image.codestream_range();
        let end = range.end().expect("codestream range end");
        assert_eq!(
            image
                .bytes()
                .get(range.offset..end)
                .expect("codestream range inside retained encoded owner"),
            expected_codestream
        );
    }
}

#[test]
fn prepared_jph_payload_ranges_resolve_inside_the_original_owned_bytes() {
    let codestream = htj2k_gray8_fixture(8, 8);
    let wrapped = wrap_j2k_codestream(&codestream, J2kFileWrapOptions::jph()).expect("wrap JPH");
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::<[u8]>::from(wrapped))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare JPH offset plan");
    let image = &prepared.groups()[0].images()[0];
    let codestream_range = image.codestream_range();
    let plan = image.htj2k_plan().expect("JPH HTJ2K reference plan");

    assert!(codestream_range.offset > 0);
    assert!(plan.payloads().all(|payload| {
        payload
            .cleanup
            .end()
            .is_some_and(|absolute_end| absolute_end <= image.bytes().len())
    }));
}
