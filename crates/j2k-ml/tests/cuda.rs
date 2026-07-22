// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(all(feature = "cuda", not(target_os = "macos")))]

use std::sync::Arc;

use burn_core::tensor::DType;
use burn_cuda::CudaDevice;
use j2k::{
    encode_j2k_lossless, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, Downscale, EncodedImage, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples, Rect,
};
use j2k_ml::{BurnBatchTensor, CudaBurnDecoder};
use j2k_test_support::{
    cuda_runtime_and_strict_oxide_gate, htj2k_gray8_large_fixture, OpenJphBatchFixture,
};

#[test]
fn direct_cuda_batch_writes_exact_u8_pixels_and_reuses_the_session() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA direct batch") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_gray8_large_fixture(8, 8));
    let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(Arc::clone(&encoded)),
            EncodedImage::full(encoded),
        ])
        .expect("prepare CUDA batch");

    for _ in 0..2 {
        let burn_batch = decoder
            .decode_prepared(&prepared)
            .expect("submit prepared CUDA batch");
        assert!(burn_batch.errors.is_empty());
        let BurnBatchTensor::U8(tensor) = burn_batch.groups.into_iter().next().unwrap().tensor
        else {
            panic!("expected U8 tensor")
        };
        assert_eq!(tensor.dims(), [2, 1, 8, 8]);
        let values = tensor.into_data().into_vec::<u8>().expect("CUDA U8 data");
        assert_eq!(&values[..64], &values[64..]);
    }
    assert!(decoder.codec().session().submissions() >= 2);
}

#[test]
fn direct_cuda_preserves_native_u16_and_i16_samples() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA native integer batches") {
        return;
    }
    let unsigned = [0_u16, 1, 2048, 4095];
    let signed = [-2048_i16, -1, 0, 2047];
    let cases = [
        (
            encode_gray(&unsigned, 12, false),
            DType::U16,
            unsigned
                .iter()
                .map(|value| i32::from(*value))
                .collect::<Vec<_>>(),
        ),
        (
            encode_gray(&signed, 12, true),
            DType::I16,
            signed
                .iter()
                .map(|value| i32::from(*value))
                .collect::<Vec<_>>(),
        ),
    ];
    for (encoded, dtype, expected) in cases {
        let mut decoder =
            CudaBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());
        let burn_batch = decoder
            .decode(vec![EncodedImage::full(Arc::from(encoded))])
            .expect("decode native CUDA type");
        let tensor = burn_batch
            .groups
            .into_iter()
            .next()
            .unwrap()
            .tensor
            .into_tensor();
        assert_eq!(tensor.dtype(), dtype);
        let data = tensor.into_data();
        let actual = match dtype {
            DType::U16 => data
                .into_vec::<u16>()
                .expect("U16 data")
                .into_iter()
                .map(i32::from)
                .collect::<Vec<_>>(),
            DType::I16 => data
                .into_vec::<i16>()
                .expect("I16 data")
                .into_iter()
                .map(i32::from)
                .collect::<Vec<_>>(),
            _ => unreachable!("test only covers U16/I16"),
        };
        assert_eq!(actual, expected);
    }
}

#[test]
fn direct_cuda_supports_roi_and_reduction_without_host_staging() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA ROI reduction") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_gray8_large_fixture(64, 64));
    let roi = Rect {
        x: 8,
        y: 12,
        w: 32,
        h: 24,
    };
    let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());
    let burn_batch = decoder
        .decode(vec![EncodedImage::new(
            encoded,
            DecodeRequest::RegionReduced {
                roi,
                scale: Downscale::Half,
            },
        )])
        .expect("decode CUDA ROI reduction");
    let tensor = burn_batch
        .groups
        .into_iter()
        .next()
        .unwrap()
        .tensor
        .into_tensor();
    assert_eq!(tensor.dims(), [1, 1, 12, 16]);
}

#[test]
fn direct_cuda_rgb_preserves_subnative_codes_and_burn_layout() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA exact RGB batches") {
        return;
    }
    let rgb7 = (0_u16..4 * 4 * 3)
        .map(|value| ((value * 29 + 7) & 0x7f) as u8)
        .collect::<Vec<_>>();
    let rgb12 = (0_u32..4 * 4 * 3)
        .map(|value| ((value * 977 + 31) & 0x0fff) as u16)
        .collect::<Vec<_>>();
    let cases = [
        (encode_rgb(&rgb7, 7), DType::U8),
        (encode_rgb(&rgb12, 12), DType::U16),
    ];

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), options);
        let mut cpu = CpuBatchDecoder::new(options);
        for (encoded, expected_dtype) in &cases {
            let prepared = decoder
                .prepare(vec![EncodedImage::full(Arc::from(encoded.clone()))])
                .expect("prepare exact RGB Burn batch");
            let oracle = cpu
                .decode_prepared(&prepared)
                .expect("decode exact RGB CPU oracle");
            let expected = match oracle.groups()[0].samples() {
                CpuBatchSamples::U8(samples) => {
                    samples.iter().copied().map(u16::from).collect::<Vec<_>>()
                }
                CpuBatchSamples::U16(samples) => samples.clone(),
                other => panic!("unexpected exact RGB oracle type: {other:?}"),
            };

            let burn_batch = decoder
                .decode_prepared(&prepared)
                .expect("decode exact RGB directly into Burn storage");
            let group = burn_batch
                .groups
                .into_iter()
                .next()
                .expect("Burn RGB group");
            let tensor = group.tensor.into_tensor();
            assert_eq!(tensor.dtype(), *expected_dtype);
            assert_eq!(
                tensor.dims(),
                match layout {
                    BatchLayout::Nhwc => [1, 4, 4, 3],
                    BatchLayout::Nchw => [1, 3, 4, 4],
                    _ => unreachable!("test only covers public dense layouts"),
                }
            );
            let data = tensor.into_data();
            let actual = match expected_dtype {
                DType::U8 => data
                    .into_vec::<u8>()
                    .expect("exact RGB U8 tensor data")
                    .into_iter()
                    .map(u16::from)
                    .collect::<Vec<_>>(),
                DType::U16 => data.into_vec::<u16>().expect("exact RGB U16 tensor data"),
                _ => unreachable!("test only covers unsigned exact RGB tensors"),
            };
            assert_eq!(actual, expected, "{layout:?} {expected_dtype:?}");
        }
    }
}

#[test]
fn direct_cuda_signed_rgb_matches_cpu_for_geometry_and_burn_layout() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA signed RGB batches") {
        return;
    }
    let fixtures = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .filter(|fixture| {
            matches!(
                fixture.name,
                "openjph-rgb-s8-53-single-raw"
                    | "openjph-rgb-s12-53-single-raw"
                    | "openjph-rgb-s16-53-single-raw"
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(fixtures.len(), 3);
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region {
            roi: Rect {
                x: 2,
                y: 3,
                w: 9,
                h: 7,
            },
        },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi: Rect {
                x: 2,
                y: 4,
                w: 10,
                h: 8,
            },
            scale: Downscale::Half,
        },
    ];

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), options);
        let mut cpu = CpuBatchDecoder::new(options);
        for fixture in &fixtures {
            let encoded = Arc::<[u8]>::from(fixture.encoded);
            for request in requests {
                let prepared = decoder
                    .prepare(vec![EncodedImage::new(Arc::clone(&encoded), request)])
                    .unwrap_or_else(|error| panic!("{} prepare: {error}", fixture.name));
                let oracle = cpu
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{} CPU oracle: {error}", fixture.name));
                let expected = match oracle.groups()[0].samples() {
                    CpuBatchSamples::I16(samples) => samples.clone(),
                    other => panic!(
                        "{}: unexpected signed RGB oracle type: {other:?}",
                        fixture.name
                    ),
                };
                if layout == BatchLayout::Nhwc && request == DecodeRequest::Full {
                    assert_eq!(
                        expected,
                        openjph_i16_oracle(fixture),
                        "{} independent OpenJPH oracle",
                        fixture.name
                    );
                }
                let dimensions = prepared.groups()[0].info().dimensions;

                let burn_batch = decoder
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{} Burn decode: {error}", fixture.name));
                let group = burn_batch
                    .groups
                    .into_iter()
                    .next()
                    .expect("Burn signed RGB group");
                let tensor = group.tensor.into_tensor();
                assert_eq!(tensor.dtype(), DType::I16);
                assert_eq!(
                    tensor.dims(),
                    match layout {
                        BatchLayout::Nhwc => [1, dimensions.1 as usize, dimensions.0 as usize, 3,],
                        BatchLayout::Nchw => [1, 3, dimensions.1 as usize, dimensions.0 as usize,],
                        _ => unreachable!("test only covers public dense layouts"),
                    }
                );
                let actual = tensor
                    .into_data()
                    .into_vec::<i16>()
                    .expect("signed RGB I16 tensor data");
                assert_eq!(actual, expected, "{} {layout:?} {request:?}", fixture.name);
            }
        }
    }
}

fn openjph_i16_oracle(fixture: &OpenJphBatchFixture) -> Vec<i16> {
    if fixture.precision <= 8 {
        fixture
            .oracle
            .iter()
            .map(|sample| i16::from(i8::from_ne_bytes([*sample])))
            .collect()
    } else {
        fixture
            .oracle
            .chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
            .collect()
    }
}

fn encode_gray<T: Copy>(samples: &[T], precision: u8, signed: bool) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(samples);
    // SAFETY: plain integer fixtures are copied immediately into the encoder;
    // their native little-endian representation is the codec's input format.
    let bytes = unsafe { std::slice::from_raw_parts(samples.as_ptr().cast::<u8>(), byte_len) };
    let samples = J2kLosslessSamples::new(bytes, 2, 2, 1, precision, signed)
        .expect("valid native integer samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode native integer fixture")
    .codestream
}

fn encode_rgb<T: Copy>(samples: &[T], precision: u8) -> Vec<u8> {
    let byte_len = std::mem::size_of_val(samples);
    // SAFETY: plain integer fixtures are copied immediately into the encoder;
    // their native little-endian representation is the codec's input format.
    let bytes = unsafe { std::slice::from_raw_parts(samples.as_ptr().cast::<u8>(), byte_len) };
    let samples = J2kLosslessSamples::new(bytes, 4, 4, 3, precision, false)
        .expect("valid RGB native integer samples");
    encode_j2k_lossless(
        samples,
        &J2kLosslessEncodeOptions::default()
            .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
            .with_validation(J2kEncodeValidation::External),
    )
    .expect("encode RGB native integer fixture")
    .codestream
}
