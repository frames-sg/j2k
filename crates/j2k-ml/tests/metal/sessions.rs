// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn repeated_default_decoders_reuse_one_initialized_burn_device() {
    if !metal_runtime_gate("j2k-ml repeated paired Metal device") {
        return;
    }

    let first = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("first paired J2K/Burn Metal session");
    let second = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("second paired J2K/Burn Metal session");

    assert_eq!(
        first.device(),
        second.device(),
        "one cached wgpu setup must map to one initialized CubeCL device identity"
    );
}

#[test]
fn persistent_metal_burn_decoder_writes_independent_ht_directly() {
    if !metal_runtime_gate("j2k-ml direct Metal Burn batch") {
        return;
    }
    let mut decoder = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("paired J2K/Burn Metal session");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(Arc::from(openhtj2k_refinement_fixture())),
            EncodedImage::full(Arc::from(openhtj2k_refinement_fixture())),
        ])
        .expect("prepare independent HTJ2K fixture");

    for _ in 0..2 {
        let output = decoder
            .decode_prepared(&prepared)
            .expect("direct Metal Burn decode");
        assert!(output.errors.is_empty());
        let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
            panic!("expected native U8 Metal tensor")
        };
        assert_eq!(tensor.dims()[0], 2);
        let actual = tensor.into_data().into_vec::<u8>().expect("U8 data");
        let expected = openhtj2k_refinement_pixels();
        assert_eq!(
            actual,
            expected.iter().chain(expected).copied().collect::<Vec<_>>()
        );
    }
    assert!(
        decoder
            .codec()
            .submissions()
            .expect("Metal batch submissions")
            >= 2
    );
}

#[test]
fn metal_burn_regroups_prepared_images_with_submission_indices_and_settings_errors() {
    if !metal_runtime_gate("j2k-ml Metal prepared-image regrouping") {
        return;
    }

    let encoded = Arc::<[u8]>::from(openhtj2k_refinement_fixture());
    let strict = prepare_batch(
        vec![EncodedImage::full(encoded.clone())],
        BatchDecodeOptions::default(),
    )
    .expect("strict preparation")
    .groups()[0]
        .images()[0]
        .clone();
    let lenient_options = BatchDecodeOptions {
        settings: DecodeSettings::lenient(),
        ..BatchDecodeOptions::default()
    };
    let lenient = prepare_batch(vec![EncodedImage::full(encoded)], lenient_options)
        .expect("lenient preparation")
        .groups()[0]
        .images()[0]
        .clone();
    let mut decoder = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("paired J2K/Burn Metal session");

    let regrouped = decoder
        .prepare_prepared_images(vec![lenient, strict.clone(), strict.clone()])
        .expect("regroup prepared images");
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
    assert_eq!(regrouped.groups()[0].source_indices(), [1, 2]);

    let output = decoder
        .decode_prepared_images(vec![strict.clone(), strict])
        .expect("decode regrouped prepared images");
    assert!(output.errors.is_empty());
    assert!(output.group_errors.is_empty());
    assert_eq!(output.groups.len(), 1);
    assert_eq!(output.groups[0].source_indices, [0, 1]);
    assert_eq!(output.groups[0].decoded_rects.len(), 2);
    assert_eq!(output.groups[0].warnings, [Vec::new(), Vec::new()]);
    assert_eq!(output.groups[0].tensor.tensor().dims()[0], 2);
}

#[test]
fn dropped_pending_metal_burn_batch_retires_storage_and_decoder_reuses() {
    if !metal_runtime_gate("j2k-ml dropped direct Metal Burn batch") {
        return;
    }
    let mut decoder = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("paired J2K/Burn Metal session");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(Arc::from(
            openhtj2k_refinement_fixture(),
        ))])
        .expect("prepare HTJ2K fixture");

    let pending = decoder
        .submit_prepared(&prepared)
        .expect("submit disposable Burn batch");
    drop(pending);

    let output = decoder
        .decode_prepared(&prepared)
        .expect("reuse decoder after dropped pending batch");
    let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected native U8 Metal tensor")
    };
    assert_eq!(
        tensor.into_data().into_vec::<u8>().expect("U8 data"),
        openhtj2k_refinement_pixels()
    );
}

#[test]
fn metal_burn_batch_continues_after_one_group_submit_failure() {
    if !metal_runtime_gate("j2k-ml Metal group submit continuation") {
        return;
    }
    let valid_gray = Arc::<[u8]>::from(openhtj2k_refinement_fixture());
    let mut decoder = MetalBurnDecoder::system_default(BatchDecodeOptions::default())
        .expect("paired J2K/Burn Metal session");
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(unsupported_classic_roi_rgb()),
            EncodedImage::full(valid_gray),
        ])
        .expect("prepare two homogeneous Metal groups");
    assert_eq!(prepared.groups().len(), 2);

    let submitted = decoder
        .submit_prepared(&prepared)
        .expect("unsupported group must remain a result-level failure");
    assert_eq!(submitted.len(), 1);
    let output = submitted.wait().expect("finish supported Metal group");

    assert!(output.errors.is_empty());
    assert_eq!(output.groups.len(), 1);
    assert_eq!(output.groups[0].source_indices, [1]);
    assert_eq!(output.group_errors.len(), 1);
    assert_eq!(output.group_errors[0].source_indices(), &[0]);
    assert!(matches!(
        output.group_errors[0].source(),
        BurnDecodeError::Metal(j2k_metal::Error::UnsupportedMetalRequest { .. })
    ));
}
