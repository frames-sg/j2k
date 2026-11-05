// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    prepare_batch, BatchCodecRoute, BatchDecodeOptions, BatchItemError, BatchLayout,
    CpuBatchDecodeResult, CpuBatchDecoder, CpuBatchSamples, DecodeSettings, EncodedImage,
    J2kDecodeWarning, NativeSampleType, PreparationDepth,
};

use super::fixtures::{
    classic_gray8_fixture_with_tile_size, htj2k_gray8_fixture, lenient_trailing_metadata_jp2,
};

#[test]
fn prepared_batch_is_reusable_and_groups_heterogeneous_images_without_padding() {
    let gray_4x3 = Arc::<[u8]>::from(htj2k_gray8_fixture(4, 3));
    let gray_2x2 = Arc::<[u8]>::from(htj2k_gray8_fixture(2, 2));
    let inputs = vec![
        EncodedImage::full(Arc::clone(&gray_4x3)),
        EncodedImage::full(gray_2x2),
        EncodedImage::full(gray_4x3),
    ];
    let options = {
        let mut options = BatchDecodeOptions::default();
        options.layout = BatchLayout::Nhwc;
        options.workers = NonZeroUsize::new(2);
        options
    };
    let prepared = prepare_batch(inputs, options).expect("prepare batch");

    assert!(prepared.errors().is_empty(), "{:?}", prepared.errors());
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
    let preparation_options = {
        let mut options = BatchDecodeOptions::default();
        options.workers = NonZeroUsize::new(1);
        options
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

    let decode_options = {
        let mut options = BatchDecodeOptions::default();
        options.layout = BatchLayout::Nhwc;
        options.workers = NonZeroUsize::new(2);
        options
    };
    let mut decoder = CpuBatchDecoder::new(decode_options);
    let regrouped = decoder
        .prepare_prepared_images(vec![larger.clone(), smaller, larger])
        .expect("regroup caller-supplied prepared images");

    assert!(regrouped.errors().is_empty(), "{:?}", regrouped.errors());
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
    assert!(
        batch_output.errors().is_empty(),
        "{:?}",
        batch_output.errors()
    );
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
    let lenient_options = {
        let mut options = BatchDecodeOptions::default();
        options.settings = DecodeSettings::lenient();
        options
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
fn valid_lenient_batch_does_not_report_a_recovery_warning() {
    let options = {
        let mut options = BatchDecodeOptions::default();
        options.settings = DecodeSettings::lenient();
        options
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(htj2k_gray8_fixture(3, 2)))],
        options,
    )
    .expect("prepare valid lenient image");
    let mut decoder = CpuBatchDecoder::new(options);

    let output = decoder
        .decode_prepared(&prepared)
        .expect("decode valid lenient image");

    assert!(output.errors().is_empty(), "{:?}", output.errors());
    assert_eq!(output.groups().len(), 1);
    assert_eq!(output.groups()[0].warnings(), &[Vec::new()]);
}

#[test]
fn cpu_session_reuses_multitile_classic_prepared_workspace_across_decodes() {
    let options = {
        let mut options = BatchDecodeOptions::default();
        options.workers = NonZeroUsize::new(1);
        options
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

    assert!(
        first_output.errors().is_empty(),
        "{:?}",
        first_output.errors()
    );
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
    let options = {
        let mut options = BatchDecodeOptions::default();
        options.workers = NonZeroUsize::new(1);
        options
    };
    let encoded = Arc::<[u8]>::from(htj2k_gray8_fixture(16, 16));
    let session = CpuBatchDecoder::new(options);

    let first = session
        .prepare(vec![EncodedImage::full(Arc::clone(&encoded))])
        .expect("first retained preparation");
    assert!(first.errors().is_empty(), "{:?}", first.errors());
    let first_stats = session.workspace_stats();

    let second = session
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("second retained preparation");
    assert!(second.errors().is_empty(), "{:?}", second.errors());
    let second_stats = session.workspace_stats();

    assert_eq!(first_stats.preparation_calls(), 1);
    assert_eq!(first_stats.preparation_worker_reuses(), 0);
    assert_eq!(second_stats.preparation_calls(), 2);
    assert_eq!(second_stats.preparation_worker_reuses(), 1);
}

/// A JP2 only a lenient parse recovers fails preparation under strict
/// settings, and under lenient settings prepares and decodes through shared
/// preparation, CPU preparation and the one-shot decode, reporting the
/// recovery among its warnings as a single-image decode does.
#[test]
fn lenient_batches_recover_trailing_metadata_and_report_it() {
    let (width, height) = (5, 3);
    let bytes = Arc::<[u8]>::from(lenient_trailing_metadata_jp2(width, height));
    let inputs = || vec![EncodedImage::full(Arc::clone(&bytes))];
    let pixels = (0_u8..15).collect::<Vec<_>>();

    let strict = prepare_batch(inputs(), BatchDecodeOptions::default())
        .expect("strict preparation is an infrastructure success");
    assert_eq!(strict.errors().len(), 1, "strict settings reject the box");
    assert!(strict.groups().is_empty());

    let lenient = {
        let mut options = BatchDecodeOptions::default();
        options.settings = DecodeSettings::lenient();
        options
    };
    let mut decoder = CpuBatchDecoder::new(lenient);
    let shared = prepare_batch(inputs(), lenient).expect("shared preparation");
    let cpu = decoder.prepare(inputs()).expect("CPU preparation");
    for (route, prepared) in [("shared", &shared), ("CPU", &cpu)] {
        assert!(
            prepared.errors().is_empty(),
            "{route}: {:?}",
            prepared.errors()
        );
        let output = decoder.decode_prepared(prepared).expect("decode prepared");
        assert_recovered_gray(&output, &pixels, route);
    }
    let one_shot = decoder.decode(inputs()).expect("one-shot decode");
    assert_recovered_gray(&one_shot, &pixels, "one-shot");
}

fn assert_recovered_gray(decoded: &CpuBatchDecodeResult, pixels: &[u8], route: &str) {
    assert!(
        decoded.errors().is_empty(),
        "{route}: {:?}",
        decoded.errors()
    );
    assert_eq!(decoded.groups().len(), 1, "{route}");
    let group = &decoded.groups()[0];
    let CpuBatchSamples::U8(samples) = group.samples() else {
        panic!("{route}: expected u8 samples");
    };
    assert_eq!(samples.as_slice(), pixels, "{route}");
    assert_eq!(
        group.warnings(),
        &[vec![J2kDecodeWarning::LenientMetadataRecovery]],
        "{route}"
    );
}

/// A four-component sRGB JP2 without channel definitions has no alpha a
/// strict batch can name, while a lenient decode takes the fourth component
/// as straight alpha, as a lenient single-image decode does.
#[test]
fn lenient_batches_infer_undeclared_alpha_as_single_image_decodes_do() {
    let (width, height) = (3_u32, 2_u32);
    let pixels = (0_u8..24).map(|value| value * 9).collect::<Vec<_>>();
    let codestream = j2k_native::encode(
        &pixels,
        width,
        height,
        4,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode four components");
    let bytes = Arc::<[u8]>::from(j2k_test_support::wrap_jp2_codestream(
        &codestream,
        width,
        height,
        4,
        8,
        16,
    ));
    let inputs = || vec![EncodedImage::full(Arc::clone(&bytes))];

    let strict = prepare_batch(inputs(), BatchDecodeOptions::default()).expect("strict prepare");
    assert!(
        matches!(
            strict.errors(),
            [error] if matches!(error.source, BatchItemError::NonRepresentableBatchOutput { .. })
        ),
        "{:?}",
        strict.errors()
    );

    let lenient = {
        let mut options = BatchDecodeOptions::default();
        options.settings = DecodeSettings::lenient();
        options.layout = BatchLayout::Nhwc;
        options
    };
    let mut single = j2k::J2kDecoder::new_with_settings(&bytes, DecodeSettings::lenient())
        .expect("lenient single-image decoder");
    let mut expected = vec![0_u8; pixels.len()];
    single
        .decode_into(&mut expected, width as usize * 4, j2k::PixelFormat::Rgba8)
        .expect("lenient single-image RGBA decode");
    assert_eq!(expected, pixels);

    let decoded = CpuBatchDecoder::new(lenient)
        .decode(inputs())
        .expect("lenient batch decode");
    assert!(decoded.errors().is_empty(), "{:?}", decoded.errors());
    let group = &decoded.groups()[0];
    assert_eq!(group.info().color, j2k::BatchColor::Rgba);
    assert_eq!(group.info().alpha, j2k::BatchAlpha::Straight);
    let CpuBatchSamples::U8(samples) = group.samples() else {
        panic!("expected u8 samples");
    };
    assert_eq!(samples.as_slice(), expected);
    assert_eq!(
        group.warnings(),
        &[vec![J2kDecodeWarning::LenientMetadataRecovery]]
    );
}
