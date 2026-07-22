// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cpu")]

use std::sync::Arc;

use burn_core::tensor::{backend::Backend, DType};
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex, FlexDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::NdArrayDevice::Cpu as FlexDevice;
use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, BatchDecodeOptions, BatchLayout, DecodeRequest,
    Downscale, EncodedImage, J2kBlockCodingMode, J2kEncodeValidation, J2kFileWrapOptions,
    J2kLosslessEncodeOptions, J2kLosslessSamples, Rect,
};
use j2k_ml::{BurnBatchTensor, CpuBurnDecoder};
use j2k_test_support::{
    classic_j2k_gray8_fixture, htj2k_gray8_fixture, htj2k_gray8_large_fixture,
    htj2k_rgb8_fixture_with_pixels,
};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type Flex = burn_ndarray::NdArray<f32, i64, i8>;

#[test]
fn raw_j2k_jp2_raw_htj2k_and_jph_stay_exact_u8_batches() {
    let classic = classic_j2k_gray8_fixture(4, 3);
    let ht = htj2k_gray8_fixture(4, 3);
    let jp2 = wrap_j2k_codestream(&classic, J2kFileWrapOptions::jp2()).expect("wrap JP2");
    let jph = wrap_j2k_codestream(&ht, J2kFileWrapOptions::jph()).expect("wrap JPH");
    let inputs = [classic, jp2, ht, jph]
        .into_iter()
        .map(|bytes| EncodedImage::full(Arc::from(bytes)))
        .collect();
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let output = decoder.decode(inputs).expect("decode wrapper matrix");

    assert!(output.errors.is_empty());
    assert_eq!(
        output.groups.len(),
        4,
        "transfer syntax remains grouping metadata"
    );
    for group in output.groups {
        let BurnBatchTensor::U8(tensor) = group.tensor else {
            panic!("expected U8 group")
        };
        assert_eq!(tensor.dims(), [1, 1, 3, 4]);
        assert_eq!(
            tensor.into_data().into_vec::<u8>().expect("u8 data"),
            (0..12).collect::<Vec<_>>()
        );
    }
}

#[test]
fn classic_and_ht_wrappers_preserve_u16_values_and_dtype() {
    let samples = [0_u16, 1, 0x1234, u16::MAX, 4096, 32_768];
    let classic = encode_gray16(&samples, 3, 2, J2kBlockCodingMode::Classic);
    let ht = encode_gray16(&samples, 3, 2, J2kBlockCodingMode::HighThroughput);
    let inputs = [
        classic.clone(),
        wrap_j2k_codestream(&classic, J2kFileWrapOptions::jp2()).expect("wrap JP2"),
        ht.clone(),
        wrap_j2k_codestream(&ht, J2kFileWrapOptions::jph()).expect("wrap JPH"),
    ]
    .into_iter()
    .map(|bytes| EncodedImage::full(Arc::from(bytes)))
    .collect();
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let output = decoder.decode(inputs).expect("decode U16 wrapper matrix");

    assert!(output.errors.is_empty());
    for group in output.groups {
        let BurnBatchTensor::U16(tensor) = group.tensor else {
            panic!("expected U16 group")
        };
        assert_eq!(tensor.dtype(), DType::U16);
        assert_eq!(
            tensor.into_data().into_vec::<u16>().expect("u16 data"),
            samples
        );
    }
}

#[test]
fn codec_layout_controls_nchw_and_nhwc_without_adapter_repacking() {
    let (encoded, expected_nhwc) = htj2k_rgb8_fixture_with_pixels(3, 2);
    for (layout, expected_shape) in [
        (BatchLayout::Nchw, [2, 3, 2, 3]),
        (BatchLayout::Nhwc, [2, 2, 3, 3]),
    ] {
        let mut decoder = CpuBurnDecoder::<Flex>::new(
            FlexDevice,
            BatchDecodeOptions {
                layout,
                ..BatchDecodeOptions::default()
            },
        );
        let output = decoder
            .decode(vec![
                EncodedImage::full(Arc::from(encoded.clone())),
                EncodedImage::full(Arc::from(encoded.clone())),
            ])
            .expect("decode RGB batch");
        let BurnBatchTensor::U8(tensor) = output.groups.into_iter().next().unwrap().tensor else {
            panic!("expected RGB U8 group")
        };
        assert_eq!(tensor.dims(), expected_shape);
        let actual = tensor.into_data().into_vec::<u8>().expect("RGB bytes");
        let expected_image = match layout {
            BatchLayout::Nhwc => expected_nhwc.clone(),
            BatchLayout::Nchw => (0..3)
                .flat_map(|channel| expected_nhwc.iter().skip(channel).step_by(3).copied())
                .collect(),
            _ => unreachable!("known layouts"),
        };
        assert_eq!(
            actual,
            expected_image
                .iter()
                .chain(&expected_image)
                .copied()
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn roi_reduction_and_every_supported_reduction_report_dense_shapes() {
    let encoded = Arc::<[u8]>::from(htj2k_gray8_large_fixture(64, 64));
    let roi = Rect {
        x: 8,
        y: 12,
        w: 32,
        h: 24,
    };
    let requests = [
        (DecodeRequest::Full, [1, 1, 64, 64]),
        (DecodeRequest::Region { roi }, [1, 1, 24, 32]),
        (
            DecodeRequest::Reduced {
                scale: Downscale::Half,
            },
            [1, 1, 32, 32],
        ),
        (
            DecodeRequest::Reduced {
                scale: Downscale::Quarter,
            },
            [1, 1, 16, 16],
        ),
        (
            DecodeRequest::Reduced {
                scale: Downscale::Eighth,
            },
            [1, 1, 8, 8],
        ),
        (
            DecodeRequest::RegionReduced {
                roi,
                scale: Downscale::Half,
            },
            [1, 1, 12, 16],
        ),
    ];
    for (request, expected_shape) in requests {
        let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
        let output = decoder
            .decode(vec![EncodedImage::new(Arc::clone(&encoded), request)])
            .expect("decode requested geometry");
        let tensor = output
            .groups
            .into_iter()
            .next()
            .unwrap()
            .tensor
            .into_tensor();
        assert_eq!(tensor.dims(), expected_shape, "{request:?}");
    }
}

#[test]
fn empty_batch_is_a_success_and_corrupt_inputs_are_indexed() {
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let empty = decoder.decode(Vec::new()).expect("empty batch");
    assert!(empty.groups.is_empty());
    assert!(empty.errors.is_empty());

    let valid = Arc::<[u8]>::from(classic_j2k_gray8_fixture(2, 2));
    let output = decoder
        .decode(vec![
            EncodedImage::full(valid),
            EncodedImage::full(Arc::from(*b"corrupt")),
        ])
        .expect("valid groups survive indexed preflight errors");
    assert_eq!(output.groups.len(), 1);
    assert_eq!(output.errors.len(), 1);
    assert_eq!(output.errors[0].index, 1);
}

#[test]
fn portable_test_backend_supports_all_fast_batch_integer_dtypes() {
    assert!(Flex::supports_dtype(&FlexDevice, DType::U8));
    assert!(Flex::supports_dtype(&FlexDevice, DType::U16));
    assert!(Flex::supports_dtype(&FlexDevice, DType::I16));
}

fn encode_gray16(samples: &[u16], width: u32, height: u32, mode: J2kBlockCodingMode) -> Vec<u8> {
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let samples =
        J2kLosslessSamples::new(&bytes, width, height, 1, 16, false).expect("valid Gray16");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(mode)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode Gray16")
    .codestream
}
