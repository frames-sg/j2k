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

    assert!(output.errors().is_empty());
    assert_eq!(
        output.groups().len(),
        4,
        "transfer syntax remains grouping metadata"
    );
    for group in output.into_groups() {
        let BurnBatchTensor::U8(tensor) = group.into_tensor() else {
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

    assert!(output.errors().is_empty());
    for group in output.into_groups() {
        let BurnBatchTensor::U16(tensor) = group.into_tensor() else {
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
        let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, {
            let mut options = BatchDecodeOptions::default();
            options.layout = layout;
            options
        });
        let output = decoder
            .decode(vec![
                EncodedImage::full(Arc::from(encoded.clone())),
                EncodedImage::full(Arc::from(encoded.clone())),
            ])
            .expect("decode RGB batch");
        let BurnBatchTensor::U8(tensor) = output
            .into_groups()
            .into_iter()
            .next()
            .unwrap()
            .into_tensor()
        else {
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
            .into_groups()
            .into_iter()
            .next()
            .unwrap()
            .into_tensor()
            .into_tensor();
        assert_eq!(tensor.dims(), expected_shape, "{request:?}");
    }
}

#[test]
fn empty_batch_is_a_success_and_corrupt_inputs_are_indexed() {
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let empty = decoder.decode(Vec::new()).expect("empty batch");
    assert!(empty.groups().is_empty());
    assert!(empty.errors().is_empty());

    let valid = Arc::<[u8]>::from(classic_j2k_gray8_fixture(2, 2));
    let output = decoder
        .decode(vec![
            EncodedImage::full(valid),
            EncodedImage::full(Arc::from(*b"corrupt")),
        ])
        .expect("valid groups survive indexed preflight errors");
    assert_eq!(output.groups().len(), 1);
    assert_eq!(output.errors().len(), 1);
    assert_eq!(output.errors()[0].index, 1);
}

#[test]
fn portable_test_backend_supports_all_fast_batch_integer_dtypes() {
    assert!(Flex::supports_dtype(&FlexDevice, DType::U8));
    assert!(Flex::supports_dtype(&FlexDevice, DType::U16));
    assert!(Flex::supports_dtype(&FlexDevice, DType::I16));
}

#[test]
fn jpeg_materialization_preserves_u8_layout_hints_and_partial_results() {
    use j2k_jpeg::{ColorTransform, JpegBatchDecodeOptions, JpegBatchDecoder, JpegBatchInput};
    use j2k_test_support::{progressive_8x8_jpeg, JPEG_BASELINE_420_16X16};

    let small = progressive_8x8_jpeg();
    let raw_components = JpegBatchInput::new(JPEG_BASELINE_420_16X16).with_options(
        j2k_jpeg::DecodeOptions::default().with_color_transform(ColorTransform::ForceRgb),
    );
    let inputs = [
        JpegBatchInput::new(JPEG_BASELINE_420_16X16),
        JpegBatchInput::new(&small),
        JpegBatchInput::new(b"invalid"),
        raw_components,
    ];
    for layout in [BatchLayout::Nchw, BatchLayout::Nhwc] {
        let options = {
            let mut options = BatchDecodeOptions::default();
            options.layout = layout;
            options
        };
        let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, options);
        let empty = decoder.decode_batch(&[]).expect("empty JPEG batch");
        assert!(empty.groups().is_empty() && empty.errors().is_empty());
        let reference = JpegBatchDecoder::new({
            let mut options = JpegBatchDecodeOptions::default();
            options.layout = layout;
            options
        })
        .decode(&inputs)
        .expect("codec reference");
        let output = decoder
            .decode_batch(&inputs.map(j2k_ml::ImageInput::Jpeg))
            .expect("JPEG tensors");
        assert_eq!(
            output.errors().iter().map(|e| e.index).collect::<Vec<_>>(),
            [2]
        );
        assert_eq!(output.groups().len(), 2);
        assert_eq!(output.groups()[0].source_indices(), [0, 3]);
        assert_eq!(output.groups()[1].source_indices(), [1]);
        // A subsequent decode must not overwrite an earlier Burn tensor.
        decoder
            .decode_batch(
                &inputs[..1]
                    .iter()
                    .copied()
                    .map(j2k_ml::ImageInput::Jpeg)
                    .collect::<Vec<_>>(),
            )
            .expect("reuse JPEG session");
        for (group, expected) in output.into_groups().into_iter().zip(reference.groups()) {
            assert_eq!(group.dimensions(), expected.dimensions());
            assert_eq!(group.layout(), layout);
            assert_eq!(group.source_indices(), expected.source_indices());
            assert_eq!(
                group.warnings(),
                expected
                    .warnings()
                    .iter()
                    .map(|row| row
                        .iter()
                        .cloned()
                        .map(j2k_ml::ImageDecodeWarning::Jpeg)
                        .collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            );
            let (width, height) = group.dimensions();
            let count = group.source_indices().len();
            assert_eq!(
                group.tensor().tensor().dims(),
                if layout == BatchLayout::Nchw {
                    [count, 3, height as usize, width as usize]
                } else {
                    [count, height as usize, width as usize, 3]
                }
            );
            assert_eq!(group.tensor().tensor().dtype(), DType::U8);
            let actual = group
                .into_tensor()
                .into_tensor()
                .into_data()
                .into_vec::<u8>()
                .expect("U8 JPEG pixels");
            assert_eq!(actual, expected.samples());
            if count == 2 {
                let (rgb, components) = actual.split_at(actual.len() / 2);
                assert_ne!(
                    rgb, components,
                    "the per-input color hint changes decoded values"
                );
            }
        }
    }
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

#[test]
fn mixed_codec_groups_report_alpha_precision_signedness_and_decoded_rects() {
    use j2k::BatchAlpha;
    use j2k_ml::ImageInput;
    let jpeg = j2k_test_support::progressive_8x8_jpeg();
    let gray16 = EncodedImage::full(Arc::from(encode_gray16(
        &[0, 1, 0x1234, u16::MAX, 4096, 32_768],
        3,
        2,
        J2kBlockCodingMode::HighThroughput,
    )));
    let inputs = [
        ImageInput::Jpeg(j2k_jpeg::JpegBatchInput::new(&jpeg)),
        ImageInput::Jpeg2000(&gray16),
    ];
    let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, BatchDecodeOptions::default());
    let result = decoder.decode_batch(&inputs).expect("decode mixed batch");

    assert!(result.errors().is_empty());
    assert_eq!(result.groups().len(), 2);
    for group in result.groups() {
        assert_eq!(group.alpha(), BatchAlpha::None);
        assert!(!group.signed());
        assert_eq!(group.decoded_rects(), [Rect::full(group.dimensions())]);
        match group.source_indices() {
            [0] => {
                assert_eq!(group.dimensions(), (8, 8));
                assert_eq!(group.precision(), 8);
            }
            [1] => {
                assert_eq!(group.dimensions(), (3, 2));
                assert_eq!(group.precision(), 16);
            }
            other => panic!("unexpected source rows {other:?}"),
        }
    }
}

#[test]
fn mixed_codec_batch_preserves_indices_channels_and_session_integrity() {
    use j2k_ml::{ImageDecodeError, ImageInput};
    let jpeg = j2k_test_support::progressive_8x8_jpeg();
    let second_scan = jpeg
        .windows(2)
        .enumerate()
        .filter(|(_, p)| *p == [0xff, 0xda])
        .nth(1)
        .unwrap()
        .0;
    let partial = &jpeg[..second_scan];
    let encoded = EncodedImage::full(Arc::from(htj2k_gray8_fixture(8, 8)));
    for strict in [true, false] {
        let mut options = BatchDecodeOptions::default();
        options.settings = if strict {
            j2k::DecodeSettings::strict()
        } else {
            j2k::DecodeSettings::lenient()
        };
        let mut decoder = CpuBurnDecoder::<Flex>::new(FlexDevice, options);
        let input_policy = j2k_jpeg::DecodeOptions::default().with_strict(!strict);
        let inputs = [
            ImageInput::Jpeg(j2k_jpeg::JpegBatchInput::new(&jpeg)),
            ImageInput::Jpeg(j2k_jpeg::JpegBatchInput::new(partial).with_options(input_policy)),
            ImageInput::Jpeg2000(&encoded),
            ImageInput::Jpeg(j2k_jpeg::JpegBatchInput::new(b"bad")),
        ];
        let result = decoder.decode_batch(&inputs).unwrap();
        let errors = result
            .errors()
            .iter()
            .map(|error| error.index)
            .collect::<Vec<_>>();
        assert_eq!(errors, if strict { vec![1, 3] } else { vec![3] });
        assert!(result
            .errors()
            .iter()
            .all(|error| matches!(error.source, ImageDecodeError::Jpeg(_))));
        assert!(result.group_errors().is_empty());
        let mut seen = Vec::new();
        for group in result.groups() {
            for (row, &index) in group.source_indices().iter().enumerate() {
                seen.push(index);
                assert_eq!(group.recovered()[row], index == 1);
                assert_eq!(group.color().channels(), if index == 2 { 1 } else { 3 });
                assert_eq!(group.j2k_info().is_some(), index == 2);
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, if strict { vec![0, 2] } else { vec![0, 1, 2] });
        let source = std::error::Error::source(&result.errors().last().unwrap().source).unwrap();
        assert!(source.downcast_ref::<j2k_jpeg::JpegError>().is_some());
    }
}
