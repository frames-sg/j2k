// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn persistent_metal_batch_decoder_reuses_one_session_for_distinct_and_repeated_ht_batches() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = Arc::<[u8]>::from(fixture_ht_gray8());
    let second = Arc::<[u8]>::from(fixture_ht_gray8_reversed());
    let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
    let registry_id = decoder.backend_session().device().registry_id();

    let before = decoder.submissions().expect("initial submissions");
    let distinct = decoder
        .decode_batch(vec![
            EncodedImage::full(first.clone()),
            EncodedImage::full(second),
        ])
        .expect("distinct HT batch");
    let after_distinct = decoder.submissions().expect("distinct submissions");

    assert!(distinct.errors().is_empty());
    assert!(distinct.group_errors().is_empty());
    assert_eq!(distinct.groups().len(), 1);
    assert_eq!(distinct.groups()[0].surfaces().len(), 2);
    assert!(distinct.groups()[0].surfaces().iter().all(|surface| {
        surface.backend_kind() == BackendKind::Metal
            && surface.residency() == SurfaceResidency::MetalResidentDecode
    }));
    assert_eq!(
        after_distinct - before,
        1,
        "distinct HT inputs must coalesce"
    );

    let repeated = decoder
        .decode_batch(vec![
            EncodedImage::full(first.clone()),
            EncodedImage::full(first),
        ])
        .expect("repeated HT batch");
    let after_repeated = decoder.submissions().expect("repeated submissions");

    assert!(repeated.errors().is_empty());
    assert!(repeated.group_errors().is_empty());
    assert_eq!(repeated.groups().len(), 1);
    assert_eq!(repeated.groups()[0].surfaces().len(), 2);
    assert_eq!(
        after_repeated - after_distinct,
        1,
        "repeated HT inputs must use one batch submission"
    );
    assert_eq!(
        decoder.backend_session().device().registry_id(),
        registry_id
    );
}

#[test]
fn persistent_metal_batch_decoder_accepts_and_reuses_shared_prepared_groups() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = Arc::<[u8]>::from(fixture_ht_gray8());
    let second = Arc::<[u8]>::from(fixture_ht_gray8_reversed());
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::new(first.clone(), DecodeRequest::Full),
            EncodedImage::new(second, DecodeRequest::Full),
        ])
        .expect("shared batch preparation");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 1);
    assert_eq!(prepared.groups()[0].source_indices(), &[0, 1]);
    assert!(Arc::ptr_eq(
        prepared.groups()[0].images()[0].bytes(),
        &first
    ));

    let before = decoder.submissions().expect("initial submissions");
    for expected_after in [before + 1, before + 2] {
        let result = decoder
            .decode_prepared(&prepared)
            .expect("prepared Metal batch decode");
        assert!(result.errors().is_empty());
        assert_eq!(result.groups().len(), 1);
        let group = &result.groups()[0];
        assert_eq!(group.source_indices(), &[0, 1]);
        assert_eq!(group.decoded_rects().len(), 2);
        assert_eq!(group.surfaces().len(), 2);
        let resident = group
            .resident_batch()
            .expect("NCHW Gray group must retain its dense allocation");
        assert_eq!(resident.image_count(), 2);
        let dense = completed_resident_batch_bytes(group);
        assert_eq!(resident.image_stride_bytes(), dense.len() / 2);
        for (index, surface) in group.surfaces().iter().enumerate() {
            let start = index * resident.image_stride_bytes();
            let end = start + resident.image_stride_bytes();
            assert_eq!(
                surface.as_bytes().expect("resident Gray view").as_ref(),
                &dense[start..end]
            );
        }
        assert_eq!(group.info().route, BatchCodecRoute::Htj2k);
        assert_eq!(group.info().sample_type, NativeSampleType::U8);
        assert!(group
            .surfaces()
            .iter()
            .all(|surface| surface.residency() == SurfaceResidency::MetalResidentDecode));
        assert_eq!(
            decoder.submissions().expect("completed submissions"),
            expected_after
        );
    }
}

#[test]
fn submitted_shared_prepared_batch_preserves_indexed_errors_and_resident_output() {
    if !should_run_metal_runtime() {
        return;
    }

    let malformed = Arc::<[u8]>::from(&b"not a codestream"[..]);
    let valid = Arc::<[u8]>::from(fixture_ht_gray8());
    let options = BatchDecodeOptions::default();
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(vec![EncodedImage::full(valid.clone())])
        .expect("CPU submitted batch oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("gray8 fixture must use U8 batch storage")
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let submitted = decoder
        .submit_batch(vec![
            EncodedImage::full(malformed),
            EncodedImage::full(valid),
        ])
        .expect("submit shared encoded Metal batch");

    assert_eq!(submitted.len(), 1);
    let result = submitted
        .wait()
        .expect("complete shared encoded Metal batch");
    assert_eq!(result.errors().len(), 1);
    assert_eq!(result.errors()[0].index, 0);
    assert!(result.group_errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].source_indices(), &[1]);
    assert_eq!(result.groups()[0].surfaces().len(), 1);
    assert_eq!(
        result.groups()[0].surfaces()[0]
            .as_bytes()
            .expect("submitted resident bytes"),
        expected.as_slice()
    );
}

#[test]
fn submitted_shared_batch_continues_after_nonfatal_group_submit_failure() {
    if !should_run_metal_runtime() {
        return;
    }

    let valid_gray = Arc::<[u8]>::from(fixture_ht_gray8());
    let unsupported_roi_rgb = unsupported_classic_roi_rgb();
    let options = BatchDecodeOptions::default();
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(valid_gray),
            EncodedImage::full(unsupported_roi_rgb),
        ])
        .expect("prepare mixed shared Metal batch");
    let submitted = decoder
        .submit_prepared(&prepared)
        .expect("submit supported shared group");

    assert_eq!(submitted.len(), 1);
    let result = submitted.wait().expect("complete supported shared group");
    assert!(result.errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].source_indices(), &[0]);
    assert_eq!(result.group_errors().len(), 1);
    assert_eq!(result.group_errors()[0].source_indices(), &[1]);
}

#[test]
fn dropped_shared_prepared_batch_retires_work_and_reuses_decoder() {
    if !should_run_metal_runtime() {
        return;
    }

    let first = Arc::<[u8]>::from(fixture_ht_gray8());
    let second = Arc::<[u8]>::from(fixture_ht_gray8_reversed());
    let options = BatchDecodeOptions::default();
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(first), EncodedImage::full(second)])
        .expect("prepare reusable shared Metal batch");

    let dropped = decoder
        .submit_prepared(&prepared)
        .expect("submit dropped shared Metal batch");
    assert_eq!(dropped.len(), 1);
    drop(dropped);

    let completed = decoder
        .submit_prepared(&prepared)
        .expect("resubmit shared Metal batch")
        .wait()
        .expect("complete resubmitted shared Metal batch");
    assert_eq!(completed.groups().len(), 1);
    assert_eq!(completed.groups()[0].surfaces().len(), 2);
}

#[test]
fn shared_metal_prepared_batch_decodes_classic_resident_color_without_legacy_staging() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let inputs = vec![EncodedImage::full(Arc::from(fixture_rgb8()))];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU classic RGB oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("classic RGB8 must use U8 batch storage")
    };
    let prepared = decoder.prepare(inputs).expect("prepare RGB batch");
    assert!(prepared.errors().is_empty());
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::ClassicOffsetPlan
    );

    let result = decoder
        .decode_prepared(&prepared)
        .expect("classic resident RGB decode");
    assert!(result.group_errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    assert_eq!(result.groups()[0].surfaces().len(), 1);
    assert_eq!(
        result.groups()[0].surfaces()[0]
            .as_bytes()
            .expect("classic resident RGB bytes")
            .as_ref(),
        expected.as_slice()
    );
}

#[test]
fn prepared_ht_rgb_nchw_resident_group_is_exact_without_interleaved_surface_views() {
    if !should_run_metal_runtime() {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![
        EncodedImage::full(Arc::from(fixture_ht_rgb_u8_sized(8, 8, 0))),
        EncodedImage::full(Arc::from(fixture_ht_rgb_u8_sized(8, 8, 17))),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU NCHW RGB oracle");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("six-bit RGB must use U8 batch storage")
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder.prepare(inputs).expect("prepare NCHW HT RGB batch");
    assert!(prepared.errors().is_empty());

    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode NCHW resident RGB group");
    assert!(result.group_errors().is_empty());
    assert_eq!(result.groups().len(), 1);
    let group = &result.groups()[0];
    assert_eq!(group.info().layout, BatchLayout::Nchw);
    assert_eq!(group.source_indices(), &[0, 1]);
    assert!(
        group.surfaces().is_empty(),
        "planar RGB bytes must not be mislabeled as interleaved Surface values"
    );
    let resident = group
        .resident_batch()
        .expect("NCHW RGB group must retain its dense allocation");
    assert_eq!(resident.image_count(), 2);
    assert_eq!(resident.image_stride_bytes(), expected.len() / 2);
    assert_eq!(completed_resident_batch_bytes(group), expected.as_slice());
}
