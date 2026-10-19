// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::{DecodeRequest, Decoder};

fn decode_gray16(bytes: &[u8]) -> Vec<u16> {
    let (pixels, _) = Decoder::new(bytes)
        .expect("decoder")
        .decode_request(DecodeRequest::full(PixelFormat::Gray16))
        .expect("decode");
    pixels
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&pair| u16::from_ne_bytes(pair))
        .collect()
}

fn ramp(width: u32, height: u32, precision: u8) -> Vec<u16> {
    let max = (1_u32 << precision) - 1;
    (0..width * height)
        .map(|index| {
            let (x, y) = (index % width, index / width);
            u16::try_from((x * 977 + y * 131 + (x * y) % 61) % (max + 1)).expect("sample")
        })
        .collect()
}

#[test]
fn every_predictor_round_trips_sixteen_bit_samples() {
    let (width, height) = (37, 23);
    let data = ramp(width, height, 16);
    for predictor in 1..=7 {
        let mut options = JpegLosslessEncodeOptions::new(16);
        options.predictor = predictor;
        let encoded = encode_jpeg_lossless(
            JpegLosslessSamples::Gray16 {
                data: &data,
                width,
                height,
            },
            options,
        )
        .expect("encode");
        assert_eq!(encoded.backend, JpegBackend::Cpu);
        assert_eq!(decode_gray16(&encoded.data), data, "predictor {predictor}");
    }
}

#[test]
fn invalid_options_are_rejected_before_any_coding() {
    let data = [0_u16; 4];
    let samples = JpegLosslessSamples::Gray16 {
        data: &data,
        width: 2,
        height: 2,
    };
    let reject = |options: JpegLosslessEncodeOptions| encode_jpeg_lossless(samples, options);
    let mut options = JpegLosslessEncodeOptions::new(17);
    assert!(matches!(
        reject(options),
        Err(JpegEncodeError::UnsupportedPrecision { precision: 17, .. })
    ));
    options.precision = 12;
    options.predictor = 0;
    assert!(matches!(
        reject(options),
        Err(JpegEncodeError::InvalidPredictor { predictor: 0 })
    ));
    options.predictor = 1;
    options.point_transform = 12;
    assert!(matches!(
        reject(options),
        Err(JpegEncodeError::InvalidPointTransform { .. })
    ));
    options.point_transform = 0;
    options.restart_rows = Some(0);
    assert!(matches!(
        reject(options),
        Err(JpegEncodeError::InvalidRestartInterval)
    ));
    options.restart_rows = None;
    options.backend = JpegBackend::Metal;
    assert!(matches!(
        reject(options),
        Err(JpegEncodeError::UnsupportedBackend { .. })
    ));
}

#[test]
fn samples_above_the_precision_are_rejected() {
    let data = [0_u16, 4095, 4096, 1];
    let error = encode_jpeg_lossless(
        JpegLosslessSamples::Gray16 {
            data: &data,
            width: 2,
            height: 2,
        },
        JpegLosslessEncodeOptions::new(12),
    )
    .expect_err("out of range");
    assert!(matches!(
        error,
        JpegEncodeError::SampleOutOfRange {
            index: 2,
            value: 4096,
            precision: 12
        }
    ));
}

#[test]
fn restart_intervals_must_fit_the_dri_field() {
    let data = vec![0_u8; 40_000 * 2];
    let mut options = JpegLosslessEncodeOptions::new(8);
    options.restart_rows = Some(2);
    let error = encode_jpeg_lossless(
        JpegLosslessSamples::Gray8 {
            data: &data,
            width: 40_000,
            height: 2,
        },
        options,
    )
    .expect_err("interval too large");
    assert!(matches!(
        error,
        JpegEncodeError::RestartIntervalTooLarge {
            rows: 2,
            mcus_per_row: 40_000
        }
    ));
}

#[test]
fn wrong_sample_lengths_are_rejected() {
    let data = [0_u8; 5];
    let error = encode_jpeg_lossless(
        JpegLosslessSamples::Rgb8 {
            data: &data,
            width: 2,
            height: 1,
        },
        JpegLosslessEncodeOptions::new(8),
    )
    .expect_err("short input");
    assert!(matches!(
        error,
        JpegEncodeError::SampleLength {
            expected: 6,
            actual: 5
        }
    ));
}
