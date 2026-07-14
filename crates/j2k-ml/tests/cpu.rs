// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cpu")]

use burn_core::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, DType},
};
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex, FlexDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::{NdArray as Flex, NdArrayDevice::Cpu as FlexDevice};
use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, DeviceDecodeRequest, Downscale, J2kBlockCodingMode,
    J2kEncodeValidation, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples, Rect,
    ReversibleTransform,
};
use j2k_ml::{
    cpu, ChannelSelection, FloatNormalization, PanicOnDecodeError, TensorDecodeError,
    TensorDecodeOptions, TensorInput, TensorLayout,
};
use j2k_test_support::{
    classic_j2k_gray8_fixture, htj2k_gray8_fixture, htj2k_gray8_large_fixture,
    htj2k_rgb8_fixture_with_pixels,
};

#[test]
fn decodes_gray_u8_to_default_chw_tensor() {
    let encoded = classic_j2k_gray8_fixture(4, 3);
    let decoded = cpu::decode_u8::<Flex>(
        TensorInput::full(&encoded),
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("decode tensor");

    assert_eq!(decoded.tensor.dims(), [1, 3, 4]);
    let data = decoded.tensor.into_data();
    assert_eq!(data.dtype, DType::U8);
    assert_eq!(
        data.into_vec::<u8>().expect("u8 data"),
        (0..12).collect::<Vec<_>>()
    );
}

#[test]
fn decodes_raw_j2k_jp2_raw_htj2k_and_jph_with_exact_values() {
    let classic = classic_j2k_gray8_fixture(4, 3);
    let ht = htj2k_gray8_fixture(4, 3);
    let jp2 = wrap_j2k_codestream(&classic, J2kFileWrapOptions::jp2()).expect("wrap JP2");
    let jph = wrap_j2k_codestream(&ht, J2kFileWrapOptions::jph()).expect("wrap JPH");

    for (name, encoded) in [
        ("raw J2K", classic.as_slice()),
        ("JP2", jp2.as_slice()),
        ("raw HTJ2K", ht.as_slice()),
        ("JPH", jph.as_slice()),
    ] {
        let data = cpu::decode_u8::<Flex>(
            TensorInput::full(encoded),
            &TensorDecodeOptions::default(),
            &FlexDevice,
        )
        .unwrap_or_else(|error| panic!("decode {name}: {error}"))
        .tensor
        .into_data();
        assert_eq!(data.dtype, DType::U8, "{name}");
        assert_eq!(
            data.into_vec::<u8>().expect("u8 data"),
            (0..12).collect::<Vec<_>>(),
            "{name}"
        );
    }
}

#[test]
fn decodes_rgb_float_with_layout_and_unit_normalization() {
    let (encoded, expected) = htj2k_rgb8_fixture_with_pixels(3, 2);
    let options = TensorDecodeOptions {
        layout: TensorLayout::ChannelsLast,
        channels: ChannelSelection::Rgb,
        normalization: FloatNormalization::Unit,
    };
    let decoded = cpu::decode_float::<Flex>(TensorInput::full(&encoded), &options, &FlexDevice)
        .expect("decode tensor");

    assert_eq!(decoded.tensor.dims(), [2, 3, 3]);
    let actual = decoded
        .tensor
        .into_data()
        .into_vec::<f32>()
        .expect("f32 data");
    let expected = expected
        .into_iter()
        .map(|sample| f32::from(sample) / 255.0)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn rejects_invalid_normalization_before_decoding_corrupt_input() {
    let options = TensorDecodeOptions {
        normalization: FloatNormalization::MeanStd {
            mean: vec![0.0],
            std: vec![0.0],
        },
        ..TensorDecodeOptions::default()
    };
    let error = cpu::decode_float::<Flex>(TensorInput::full(b"corrupt"), &options, &FlexDevice)
        .expect_err("zero std must fail first");
    assert!(matches!(
        error,
        TensorDecodeError::InvalidNormalization { .. }
    ));
}

#[test]
fn batch_preserves_order_and_rejects_shape_mismatch_with_index() {
    let first = classic_j2k_gray8_fixture(2, 2);
    let second = classic_j2k_gray8_fixture(3, 2);
    let error = cpu::decode_u8_batch::<Flex>(
        &[
            TensorInput {
                encoded: &first,
                request: DeviceDecodeRequest::Full,
            },
            TensorInput::full(&second),
        ],
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect_err("shape mismatch");
    assert!(matches!(
        error,
        TensorDecodeError::BatchShapeMismatch { index: 1, .. }
    ));
}

#[test]
fn flex_supports_the_integer_dtypes_used_by_the_contract() {
    assert!(Flex::supports_dtype(&FlexDevice, DType::U8));
    assert!(Flex::supports_dtype(&FlexDevice, DType::U16));
}

#[test]
fn decodes_u16_values_and_preserves_u16_dtype() {
    let samples = [0u16, 1, 0x1234, u16::MAX, 4096, 32_768];
    let encoded = encode_gray16(&samples, 3, 2);
    let decoded = cpu::decode_u16::<Flex>(
        TensorInput::full(&encoded),
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("decode u16 tensor");

    assert_eq!(decoded.tensor.dims(), [1, 2, 3]);
    let data = decoded.tensor.into_data();
    assert_eq!(data.dtype, DType::U16);
    assert_eq!(data.into_vec::<u16>().expect("u16 data"), samples);
}

#[test]
fn decodes_u16_classic_htj2k_jp2_and_jph_with_exact_values() {
    let samples = [0u16, 1, 0x1234, u16::MAX, 4096, 32_768];
    let classic = encode_gray16_mode(&samples, 3, 2, J2kBlockCodingMode::Classic);
    let ht = encode_gray16_mode(&samples, 3, 2, J2kBlockCodingMode::HighThroughput);
    let jp2 = wrap_j2k_codestream(&classic, J2kFileWrapOptions::jp2()).expect("wrap JP2");
    let jph = wrap_j2k_codestream(&ht, J2kFileWrapOptions::jph()).expect("wrap JPH");

    for (name, encoded) in [
        ("raw J2K", classic.as_slice()),
        ("JP2", jp2.as_slice()),
        ("raw HTJ2K", ht.as_slice()),
        ("JPH", jph.as_slice()),
    ] {
        let data = cpu::decode_u16::<Flex>(
            TensorInput::full(encoded),
            &TensorDecodeOptions::default(),
            &FlexDevice,
        )
        .unwrap_or_else(|error| panic!("decode {name}: {error}"))
        .tensor
        .into_data();
        assert_eq!(data.dtype, DType::U16, "{name}");
        assert_eq!(data.into_vec::<u16>().expect("u16 data"), samples, "{name}");
    }
}

#[test]
fn channel_selection_and_nhwc_batch_shapes_follow_the_public_contract() {
    let gray = classic_j2k_gray8_fixture(2, 1);
    let (rgb, _) = htj2k_rgb8_fixture_with_pixels(2, 1);
    let rgba = encode_rgba8(&[1, 2, 3, 4, 5, 6, 7, 8], 2, 1);
    for (encoded, selection, channels) in [
        (gray.as_slice(), ChannelSelection::Auto, 1),
        (gray.as_slice(), ChannelSelection::Gray, 1),
        (rgb.as_slice(), ChannelSelection::Auto, 3),
        (rgb.as_slice(), ChannelSelection::Rgb, 3),
        (rgba.as_slice(), ChannelSelection::Auto, 3),
        (rgba.as_slice(), ChannelSelection::Rgba, 4),
    ] {
        let options = TensorDecodeOptions {
            layout: TensorLayout::ChannelsLast,
            channels: selection,
            ..TensorDecodeOptions::default()
        };
        let single = cpu::decode_u8::<Flex>(TensorInput::full(encoded), &options, &FlexDevice)
            .expect("channel-selected single decode");
        assert_eq!(single.tensor.dims(), [1, 2, channels]);

        let batch = cpu::decode_u8_batch::<Flex>(
            &[TensorInput::full(encoded), TensorInput::full(encoded)],
            &options,
            &FlexDevice,
        )
        .expect("channel-selected batch decode");
        assert_eq!(batch.tensor.dims(), [2, 1, 2, channels]);
    }
}

#[test]
fn u16_unit_float_uses_the_full_canonical_denominator() {
    let samples = [0u16, 32_768, u16::MAX];
    let encoded = encode_gray16(&samples, 3, 1);
    let actual = cpu::decode_float::<Flex>(
        TensorInput::full(&encoded),
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("u16 unit float decode")
    .tensor
    .into_data()
    .into_vec::<f32>()
    .expect("f32 data");
    let expected = samples.map(|value| f32::from(value) / 65_535.0);
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= f32::EPSILON);
    }
}

#[test]
fn roi_scaled_decode_reports_shape_and_rectangle() {
    let encoded = classic_j2k_gray8_fixture(8, 8);
    let roi = Rect {
        x: 2,
        y: 2,
        w: 4,
        h: 4,
    };
    let decoded = cpu::decode_u8::<Flex>(
        TensorInput {
            encoded: &encoded,
            request: DeviceDecodeRequest::RegionScaled {
                roi,
                scale: Downscale::Half,
            },
        },
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("decode ROI tensor");

    assert_eq!(decoded.tensor.dims(), [1, 2, 2]);
    assert_eq!(decoded.decoded, roi.scaled_covering(Downscale::Half));
}

#[test]
fn every_supported_power_of_two_scale_has_the_expected_shape() {
    let encoded = htj2k_gray8_large_fixture(64, 64);
    for (scale, side) in [
        (Downscale::None, 64),
        (Downscale::Half, 32),
        (Downscale::Quarter, 16),
        (Downscale::Eighth, 8),
    ] {
        let decoded = cpu::decode_u8::<Flex>(
            TensorInput {
                encoded: &encoded,
                request: DeviceDecodeRequest::Scaled { scale },
            },
            &TensorDecodeOptions::default(),
            &FlexDevice,
        )
        .expect("scaled decode");
        assert_eq!(decoded.tensor.dims(), [1, side, side], "{scale:?}");
    }
}

#[test]
fn raw_float_casts_without_scaling() {
    let encoded = classic_j2k_gray8_fixture(3, 2);
    let options = TensorDecodeOptions {
        normalization: FloatNormalization::Raw,
        ..TensorDecodeOptions::default()
    };
    let actual = cpu::decode_float::<Flex>(TensorInput::full(&encoded), &options, &FlexDevice)
        .expect("raw float decode")
        .tensor
        .into_data()
        .into_vec::<f32>()
        .expect("f32 data");
    assert_eq!(actual, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn mean_std_is_unit_scaled_and_broadcast_per_channel() {
    let (encoded, pixels) = htj2k_rgb8_fixture_with_pixels(2, 1);
    let options = TensorDecodeOptions {
        layout: TensorLayout::ChannelsLast,
        channels: ChannelSelection::Rgb,
        normalization: FloatNormalization::MeanStd {
            mean: vec![0.1, 0.2, 0.3],
            std: vec![0.5, 0.25, 2.0],
        },
    };
    let actual = cpu::decode_float::<Flex>(TensorInput::full(&encoded), &options, &FlexDevice)
        .expect("normalized tensor")
        .tensor
        .into_data()
        .into_vec::<f32>()
        .expect("f32 data");

    for (index, (actual, pixel)) in actual.iter().zip(pixels).enumerate() {
        let channel = index % 3;
        let mean = [0.1, 0.2, 0.3][channel];
        let std = [0.5, 0.25, 2.0][channel];
        let expected = (f32::from(pixel) / 255.0 - mean) / std;
        assert!((actual - expected).abs() <= 1.0e-6);
    }
}

#[test]
fn batch_uses_one_rank_four_tensor_and_preserves_input_order() {
    let first = classic_j2k_gray8_fixture(2, 2);
    let second = classic_j2k_gray8_fixture(2, 2);
    let batch = cpu::decode_u8_batch::<Flex>(
        &[TensorInput::full(&first), TensorInput::full(&second)],
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("decode batch");

    assert_eq!(batch.tensor.dims(), [2, 1, 2, 2]);
    assert_eq!(
        batch.tensor.into_data().into_vec::<u8>().expect("u8 data"),
        vec![0, 1, 2, 3, 0, 1, 2, 3]
    );
    assert_eq!(batch.decoded.len(), 2);
    assert_eq!(batch.warnings.len(), 2);
}

#[test]
fn u16_batch_preserves_values_dtype_and_order() {
    let first_samples = [0u16, 1, 4096, u16::MAX];
    let second_samples = [32_768u16, 17, 255, 1024];
    let first = encode_gray16(&first_samples, 2, 2);
    let second = encode_gray16(&second_samples, 2, 2);
    let batch = cpu::decode_u16_batch::<Flex>(
        &[TensorInput::full(&first), TensorInput::full(&second)],
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect("decode u16 batch");

    assert_eq!(batch.tensor.dims(), [2, 1, 2, 2]);
    let data = batch.tensor.into_data();
    assert_eq!(data.dtype, DType::U16);
    assert_eq!(
        data.into_vec::<u16>().expect("u16 data"),
        first_samples
            .into_iter()
            .chain(second_samples)
            .collect::<Vec<_>>()
    );
}

#[test]
fn float_batch_mean_std_broadcasts_in_both_layouts() {
    let (encoded, pixels) = htj2k_rgb8_fixture_with_pixels(2, 1);
    let mean = [0.1f32, 0.2, 0.3];
    let std = [0.5f32, 0.25, 2.0];

    for layout in [TensorLayout::ChannelsFirst, TensorLayout::ChannelsLast] {
        let options = TensorDecodeOptions {
            layout,
            channels: ChannelSelection::Rgb,
            normalization: FloatNormalization::MeanStd {
                mean: mean.to_vec(),
                std: std.to_vec(),
            },
        };
        let batch = cpu::decode_float_batch::<Flex>(
            &[TensorInput::full(&encoded), TensorInput::full(&encoded)],
            &options,
            &FlexDevice,
        )
        .expect("decode normalized float batch");

        assert_eq!(
            batch.tensor.dims(),
            match layout {
                TensorLayout::ChannelsFirst => [2, 3, 1, 2],
                TensorLayout::ChannelsLast => [2, 1, 2, 3],
            }
        );
        let actual = batch
            .tensor
            .into_data()
            .into_vec::<f32>()
            .expect("f32 data");
        let expected_image = match layout {
            TensorLayout::ChannelsFirst => (0..3)
                .flat_map(|channel| {
                    pixels.iter().skip(channel).step_by(3).map(move |pixel| {
                        (f32::from(*pixel) / 255.0 - mean[channel]) / std[channel]
                    })
                })
                .collect::<Vec<_>>(),
            TensorLayout::ChannelsLast => pixels
                .iter()
                .enumerate()
                .map(|(index, pixel)| {
                    let channel = index % 3;
                    (f32::from(*pixel) / 255.0 - mean[channel]) / std[channel]
                })
                .collect::<Vec<_>>(),
        };
        let expected = expected_image
            .iter()
            .copied()
            .chain(expected_image.iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1.0e-6);
        }
    }
}

#[test]
fn empty_and_corrupt_batches_are_explicit_and_indexed() {
    let empty = cpu::decode_u8_batch::<Flex>(&[], &TensorDecodeOptions::default(), &FlexDevice)
        .expect_err("empty batch");
    assert!(matches!(empty, TensorDecodeError::EmptyBatch));

    let valid = classic_j2k_gray8_fixture(2, 2);
    let corrupt = cpu::decode_u8_batch::<Flex>(
        &[TensorInput::full(&valid), TensorInput::full(b"corrupt")],
        &TensorDecodeOptions::default(),
        &FlexDevice,
    )
    .expect_err("corrupt batch item");
    assert!(matches!(
        corrupt,
        TensorDecodeError::BatchItem { index: 1, .. }
    ));
}

#[test]
fn panic_batcher_includes_actionable_item_context() {
    let valid = classic_j2k_gray8_fixture(2, 2);
    let batcher = PanicOnDecodeError::<Flex>::new(TensorDecodeOptions::default());
    let panic = std::panic::catch_unwind(|| {
        let _ = batcher.batch(
            vec![TensorInput::full(&valid), TensorInput::full(b"corrupt")],
            &FlexDevice,
        );
    })
    .expect_err("adapter must panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("string panic");
    assert!(message.contains("batch item 1"), "{message}");
    assert!(message.contains("decode failed"), "{message}");
}

fn encode_gray16(samples: &[u16], width: u32, height: u32) -> Vec<u8> {
    encode_gray16_mode(samples, width, height, J2kBlockCodingMode::Classic)
}

fn encode_gray16_mode(
    samples: &[u16],
    width: u32,
    height: u32,
    mode: J2kBlockCodingMode,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    let samples =
        J2kLosslessSamples::new(&bytes, width, height, 1, 16, false).expect("valid gray16 samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(mode)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode gray16")
    .codestream
}

fn encode_rgba8(samples: &[u8], width: u32, height: u32) -> Vec<u8> {
    let samples =
        J2kLosslessSamples::new(samples, width, height, 4, 8, false).expect("valid rgba8 samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_reversible_transform(ReversibleTransform::None53)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode rgba8")
    .codestream
}
