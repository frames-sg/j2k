// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cpu")]

use std::sync::Arc;

use burn_core::tensor::DType;
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex, FlexDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::NdArrayDevice::Cpu as FlexDevice;
use j2k::{
    encode_j2k_lossless, BatchDecodeOptions, BatchLayout, EncodedImage, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_ml::{BurnBatchTensor, CpuBurnDecoder};
use j2k_test_support::{
    htj2k_gray8_fixture, openhtj2k_refinement_fixture, openhtj2k_refinement_odd_fixture,
    openhtj2k_refinement_odd_pixels, openhtj2k_refinement_pixels,
};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type Flex = burn_ndarray::NdArray<f32, i64, i8>;

#[test]
fn persistent_cpu_burn_decoder_reuses_prepared_integer_batch() {
    let encoded = Arc::<[u8]>::from(htj2k_gray8_fixture(4, 3));
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, options);
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(Arc::clone(&encoded)),
            EncodedImage::full(encoded),
        ])
        .expect("prepare reusable Burn batch");

    let first = decoder
        .decode_prepared(&prepared)
        .expect("first prepared Burn decode");
    let second = decoder
        .decode_prepared(&prepared)
        .expect("second prepared Burn decode");

    assert!(first.errors.is_empty());
    assert!(second.errors.is_empty());
    assert!(first.group_errors.is_empty());
    assert!(second.group_errors.is_empty());
    assert_eq!(first.groups.len(), 1);
    assert_eq!(second.groups.len(), 1);
    assert_eq!(first.groups[0].source_indices, [0, 1]);
    assert_eq!(second.groups[0].source_indices, [0, 1]);

    let BurnBatchTensor::U8(first_tensor) = first.groups.into_iter().next().unwrap().tensor else {
        panic!("expected U8 Burn tensor group")
    };
    let BurnBatchTensor::U8(second_tensor) = second.groups.into_iter().next().unwrap().tensor
    else {
        panic!("expected U8 Burn tensor group")
    };
    assert_eq!(first_tensor.dims(), [2, 1, 3, 4]);
    assert_eq!(first_tensor.dtype(), DType::U8);
    assert_eq!(first_tensor.into_data(), second_tensor.into_data());
}

#[test]
fn cpu_burn_decoder_accepts_caller_supplied_prepared_images() {
    let encoded = Arc::<[u8]>::from(htj2k_gray8_fixture(4, 3));
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nchw,
        ..BatchDecodeOptions::default()
    };
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, options);
    let original = decoder
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("prepare source image");
    let image = original.groups()[0].images()[0].clone();

    let regrouped = decoder
        .prepare_prepared_images(vec![image.clone(), image.clone()])
        .expect("regroup prepared Burn inputs");
    assert_eq!(regrouped.groups()[0].source_indices(), [0, 1]);
    let first = decoder
        .decode_prepared(&regrouped)
        .expect("decode explicitly regrouped Burn inputs");
    let second = decoder
        .decode_prepared_images(vec![image.clone(), image])
        .expect("regroup and decode prepared Burn inputs");

    assert!(first.errors.is_empty());
    assert!(second.errors.is_empty());
    assert_eq!(first.groups[0].source_indices, [0, 1]);
    assert_eq!(second.groups[0].source_indices, [0, 1]);
    let BurnBatchTensor::U8(first_tensor) = first.groups.into_iter().next().unwrap().tensor else {
        panic!("expected U8 Burn tensor group")
    };
    let BurnBatchTensor::U8(second_tensor) = second.groups.into_iter().next().unwrap().tensor
    else {
        panic!("expected U8 Burn tensor group")
    };
    assert_eq!(first_tensor.dims(), [2, 1, 3, 4]);
    assert_eq!(first_tensor.into_data(), second_tensor.into_data());
}

#[test]
fn burn_adapter_preserves_heterogeneous_groups_and_indexed_preflight_errors() {
    let first = Arc::<[u8]>::from(htj2k_gray8_fixture(4, 3));
    let second = Arc::<[u8]>::from(htj2k_gray8_fixture(2, 2));
    let inputs = vec![
        EncodedImage::full(first),
        EncodedImage::full(Arc::<[u8]>::from([0_u8, 1, 2, 3])),
        EncodedImage::full(second),
    ];
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let output = decoder.decode(inputs).expect("decode valid groups");

    assert_eq!(output.errors.len(), 1);
    assert_eq!(output.errors[0].index, 1);
    assert_eq!(output.groups.len(), 2);
    assert_eq!(output.groups[0].source_indices, [0]);
    assert_eq!(output.groups[1].source_indices, [2]);
}

#[test]
fn signed_i16_codec_group_stays_i16_in_burn() {
    let samples = [-300_i16, -1, 0, 300];
    let encoded = signed_gray16_fixture(&samples, 2, 2);
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let output = decoder
        .decode(vec![EncodedImage::full(Arc::<[u8]>::from(encoded))])
        .expect("decode signed Burn batch");

    assert!(output.errors.is_empty());
    let BurnBatchTensor::I16(tensor) = output.groups.into_iter().next().unwrap().tensor else {
        panic!("expected I16 Burn tensor group")
    };
    assert_eq!(tensor.dtype(), DType::I16);
    assert_eq!(
        tensor.into_data().into_vec::<i16>().expect("i16 data"),
        samples
    );
}

#[test]
fn independent_openhtj2k_cleanup_and_refinement_samples_are_exact() {
    for (encoded, expected) in [
        (
            openhtj2k_refinement_fixture(),
            openhtj2k_refinement_pixels(),
        ),
        (
            openhtj2k_refinement_odd_fixture(),
            openhtj2k_refinement_odd_pixels(),
        ),
    ] {
        let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
        let output = decoder
            .decode(vec![EncodedImage::full(Arc::<[u8]>::from(encoded))])
            .expect("decode independent OpenHTJ2K fixture");
        assert!(output.errors.is_empty());
        let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
            panic!("expected U8 OpenHTJ2K tensor")
        };
        assert_eq!(
            tensor.into_data().into_vec::<u8>().expect("u8 data"),
            expected,
        );
    }
}

fn signed_gray16_fixture(samples: &[i16], width: u32, height: u32) -> Vec<u8> {
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let samples =
        J2kLosslessSamples::new(&bytes, width, height, 1, 16, true).expect("signed gray16 samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode signed gray16")
    .codestream
}
