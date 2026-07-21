// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    prepare_batch, BatchCodecRoute, BatchDecodeOptions, BatchItemError, BatchLayout,
    CpuBatchDecoder, CpuBatchSamples, DecodeSettings, EncodedImage, NativeSampleType,
    PreparationDepth,
};

use super::fixtures::{classic_gray8_fixture_with_tile_size, htj2k_gray8_fixture};

#[test]
fn prepared_batch_is_reusable_and_groups_heterogeneous_images_without_padding() {
    let gray_4x3 = Arc::<[u8]>::from(htj2k_gray8_fixture(4, 3));
    let gray_2x2 = Arc::<[u8]>::from(htj2k_gray8_fixture(2, 2));
    let inputs = vec![
        EncodedImage::full(Arc::clone(&gray_4x3)),
        EncodedImage::full(gray_2x2),
        EncodedImage::full(gray_4x3),
    ];
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        workers: NonZeroUsize::new(2),
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(inputs, options).expect("prepare batch");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].source_indices(), &[0, 2]);
    assert_eq!(prepared.groups()[1].source_indices(), &[1]);
    assert_eq!(prepared.groups()[0].options().layout, options.layout);
    assert_eq!(
        prepared.groups()[0]
            .options()
            .settings
            .lenient_tolerance_enabled(),
        options.settings.lenient_tolerance_enabled()
    );
    assert_eq!(
        prepared.groups()[0].info().sample_type,
        NativeSampleType::U8
    );
    assert_eq!(prepared.groups()[0].info().route, BatchCodecRoute::Htj2k);

    let mut session = CpuBatchDecoder::new(options);
    let first = session
        .decode_prepared(&prepared)
        .expect("first prepared decode");
    let second = session
        .decode_prepared(&prepared)
        .expect("second prepared decode");

    assert!(
        first.errors().is_empty(),
        "decode errors: {:?}",
        first.errors()
    );
    assert_eq!(first.groups(), second.groups());
    let CpuBatchSamples::U8(samples) = first.groups()[0].samples() else {
        panic!("expected u8 samples")
    };
    let expected = (0_u8..12).collect::<Vec<_>>();
    assert_eq!(&samples[..expected.len()], expected);
    assert_eq!(&samples[expected.len()..], expected);
}

#[test]
fn caller_supplied_prepared_images_regroup_in_new_submission_order_without_reparse() {
    let preparation_options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let larger_bytes = Arc::<[u8]>::from(htj2k_gray8_fixture(4, 3));
    let larger_preparation = prepare_batch(
        vec![
            EncodedImage::full(Arc::from([0_u8, 1, 2, 3])),
            EncodedImage::full(Arc::clone(&larger_bytes)),
        ],
        preparation_options,
    )
    .expect("prepare larger image after an indexed failure");
    let larger = larger_preparation.groups()[0].images()[0].clone();
    assert_eq!(larger.source_index(), 1);

    let smaller_preparation = prepare_batch(
        vec![EncodedImage::full(Arc::from(htj2k_gray8_fixture(2, 2)))],
        preparation_options,
    )
    .expect("prepare smaller image");
    let smaller = smaller_preparation.groups()[0].images()[0].clone();

    let decode_options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        workers: NonZeroUsize::new(2),
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBatchDecoder::new(decode_options);
    let regrouped = decoder
        .prepare_prepared_images(vec![larger.clone(), smaller, larger])
        .expect("regroup caller-supplied prepared images");

    assert!(regrouped.errors().is_empty());
    assert_eq!(regrouped.groups().len(), 2);
    assert_eq!(regrouped.groups()[0].source_indices(), [0, 2]);
    assert_eq!(regrouped.groups()[1].source_indices(), [1]);
    assert_eq!(regrouped.groups()[0].info().layout, BatchLayout::Nhwc);
    assert_eq!(
        regrouped.groups()[0]
            .images()
            .iter()
            .map(j2k::PreparedImage::source_index)
            .collect::<Vec<_>>(),
        [1, 1]
    );
    assert!(regrouped.groups()[0]
        .images()
        .iter()
        .all(|image| Arc::ptr_eq(image.bytes(), &larger_bytes)));

    let batch_output = decoder
        .decode_prepared(&regrouped)
        .expect("decode regrouped prepared images");
    assert!(batch_output.errors().is_empty());
    assert_eq!(batch_output.groups()[0].source_indices(), [0, 2]);
    assert_eq!(batch_output.groups()[1].source_indices(), [1]);
    let CpuBatchSamples::U8(samples) = batch_output.groups()[0].samples() else {
        panic!("expected native u8 samples")
    };
    let expected = (0_u8..12).collect::<Vec<_>>();
    assert_eq!(samples, &[expected.clone(), expected].concat());
}

#[test]
fn regrouping_prepared_images_reports_settings_mismatches_at_submission_indices() {
    let lenient_options = BatchDecodeOptions {
        settings: DecodeSettings::lenient(),
        ..BatchDecodeOptions::default()
    };
    let lenient = prepare_batch(
        vec![EncodedImage::full(Arc::from(htj2k_gray8_fixture(3, 2)))],
        lenient_options,
    )
    .expect("prepare lenient image")
    .groups()[0]
        .images()[0]
        .clone();
    let strict = prepare_batch(
        vec![EncodedImage::full(Arc::from(htj2k_gray8_fixture(2, 2)))],
        BatchDecodeOptions::default(),
    )
    .expect("prepare strict image")
    .groups()[0]
        .images()[0]
        .clone();
    let mut decoder = CpuBatchDecoder::new(BatchDecodeOptions::default());

    let regrouped = decoder
        .prepare_prepared_images(vec![lenient, strict])
        .expect("regroup remains an infrastructure success");

    assert_eq!(regrouped.errors().len(), 1);
    assert_eq!(regrouped.errors()[0].index, 0);
    assert!(matches!(
        regrouped.errors()[0].source,
        BatchItemError::PreparedDecodeSettingsMismatch {
            prepared,
            requested,
        } if prepared == DecodeSettings::lenient() && requested == DecodeSettings::strict()
    ));
    assert_eq!(regrouped.groups().len(), 1);
    assert_eq!(regrouped.groups()[0].source_indices(), [1]);

    let batch_output = decoder
        .decode_prepared(&regrouped)
        .expect("decode compatible regrouped image");
    assert_eq!(batch_output.errors().len(), 1);
    assert_eq!(batch_output.errors()[0].index, 0);
    assert_eq!(batch_output.groups()[0].source_indices(), [1]);
}

#[test]
fn cpu_session_reuses_multitile_classic_prepared_workspace_across_decodes() {
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(
            classic_gray8_fixture_with_tile_size(16, 16, Some((8, 8))),
        ))],
        options,
    )
    .expect("prepare reusable CPU batch");
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::ClassicOffsetPlan
    );
    let mut session = CpuBatchDecoder::new(options);

    let first_output = session
        .decode_prepared(&prepared)
        .expect("first prepared decode");
    let first = session.workspace_stats();
    let second_output = session
        .decode_prepared(&prepared)
        .expect("second prepared decode");
    let second = session.workspace_stats();

    assert!(first_output.errors().is_empty());
    assert_eq!(first_output.groups(), second_output.groups());
    assert_eq!(first.prepared_plan_decode_calls(), 1);
    assert_eq!(second.prepared_plan_decode_calls(), 2);
    assert_eq!(first.decode_calls(), 0);
    assert_eq!(second.decode_calls(), 0);
    assert_eq!(second.scratch_capacity_retries(), 0);
    assert!(first.retained_prepared_plan_classic_workspace_bytes() > 0);
    assert_eq!(
        second.retained_prepared_plan_classic_workspace_bytes(),
        first.retained_prepared_plan_classic_workspace_bytes()
    );
}

#[test]
fn cpu_session_reuses_preparation_workers_across_one_shot_batches() {
    let options = BatchDecodeOptions {
        workers: NonZeroUsize::new(1),
        ..BatchDecodeOptions::default()
    };
    let encoded = Arc::<[u8]>::from(htj2k_gray8_fixture(16, 16));
    let session = CpuBatchDecoder::new(options);

    let first = session
        .prepare(vec![EncodedImage::full(Arc::clone(&encoded))])
        .expect("first retained preparation");
    assert!(first.errors().is_empty());
    let first_stats = session.workspace_stats();

    let second = session
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("second retained preparation");
    assert!(second.errors().is_empty());
    let second_stats = session.workspace_stats();

    assert_eq!(first_stats.preparation_calls(), 1);
    assert_eq!(first_stats.preparation_worker_reuses(), 0);
    assert_eq!(second_stats.preparation_calls(), 2);
    assert_eq!(second_stats.preparation_worker_reuses(), 1);
}
