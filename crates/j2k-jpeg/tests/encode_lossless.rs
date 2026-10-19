// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lossless (SOF3) encode against libjpeg-turbo and DCMTK.

use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_lossless, DecodeRequest, Decoder, JpegLosslessEncodeOptions, JpegLosslessSamples,
};
use j2k_test_support::LosslessMatrixCase;

/// The matrix cases `cjpeg -lossless` wrote (always with optimal tables).
fn cjpeg_cases() -> Vec<LosslessMatrixCase> {
    j2k_test_support::lossless_matrix_cases()
        .into_iter()
        .filter(|case| case.scans.is_none() && matches!(case.layout, "gray" | "rgb444"))
        .collect()
}

fn options_for(case: &LosslessMatrixCase) -> JpegLosslessEncodeOptions {
    let mut options = JpegLosslessEncodeOptions::new(case.precision);
    options.predictor = case.predictor;
    options.point_transform = case.point_transform;
    options.restart_rows = (case.restart_rows != 0)
        .then(|| u16::try_from(case.restart_rows).expect("restart rows fit u16"));
    options
}

fn encode_samples(
    samples: &[u16],
    gray: bool,
    (width, height): (u32, u32),
    options: JpegLosslessEncodeOptions,
) -> Vec<u8> {
    let narrow: Vec<u8>;
    let input = match (options.precision > 8, gray) {
        (true, true) => JpegLosslessSamples::Gray16 {
            data: samples,
            width,
            height,
        },
        (true, false) => JpegLosslessSamples::Rgb16 {
            data: samples,
            width,
            height,
        },
        (false, _) => {
            narrow = samples
                .iter()
                .map(|&sample| u8::try_from(sample).expect("8-bit sample"))
                .collect();
            if gray {
                JpegLosslessSamples::Gray8 {
                    data: &narrow,
                    width,
                    height,
                }
            } else {
                JpegLosslessSamples::Rgb8 {
                    data: &narrow,
                    width,
                    height,
                }
            }
        }
    };
    encode_jpeg_lossless(input, options)
        .expect("lossless encode")
        .data
}

#[test]
fn lossless_encode_matches_cjpeg_byte_for_byte() {
    let cases = cjpeg_cases();
    assert!(cases.len() >= 40, "only {} cjpeg cases", cases.len());
    for case in &cases {
        let samples = case
            .reference_samples()
            .expect("cjpeg cases carry a reference");
        let encoded = encode_samples(
            &samples,
            case.is_gray(),
            (case.width, case.height),
            options_for(case),
        );
        assert!(
            encoded == case.jpeg,
            "{}: output differs from cjpeg",
            case.name
        );
    }
}

#[test]
fn lossless_encode_round_trips_through_cpu_decode() {
    for case in cjpeg_cases() {
        let samples = case.reference_samples().expect("reference");
        let encoded = encode_samples(
            &samples,
            case.is_gray(),
            (case.width, case.height),
            options_for(&case),
        );
        let fmt = match (case.precision > 8, case.is_gray()) {
            (false, true) => PixelFormat::Gray8,
            (false, false) => PixelFormat::Rgb8,
            (true, true) => PixelFormat::Gray16,
            (true, false) => PixelFormat::Rgb16,
        };
        let (decoded, _) = Decoder::new(&encoded)
            .expect("decoder")
            .decode_request(DecodeRequest::full(fmt))
            .expect("decode");
        let decoded: Vec<u16> = if case.precision > 8 {
            decoded
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_ne_bytes(*pair))
                .collect()
        } else {
            decoded.iter().map(|&sample| u16::from(sample)).collect()
        };
        assert_eq!(decoded, samples, "{}", case.name);
    }
}

#[test]
fn dicom_lossless_cases_round_trip_through_lossless_encode() {
    for case in j2k_test_support::dicom_matrix_cases() {
        if !case.options.contains("+e1") && !case.options.contains("+el") {
            continue;
        }
        let samples = case.reference_samples();
        // DCMTK stores 10- to 16-bit images at precision 16.
        let options = JpegLosslessEncodeOptions::new(case.precision);
        let encoded = encode_samples(
            &samples,
            case.channels == 1,
            (case.width, case.height),
            options,
        );
        let fmt = match (case.precision > 8, case.channels == 1) {
            (false, true) => PixelFormat::Gray8,
            (false, false) => PixelFormat::Rgb8,
            (true, true) => PixelFormat::Gray16,
            (true, false) => PixelFormat::Rgb16,
        };
        let (decoded, _) = Decoder::new(&encoded)
            .expect("decoder")
            .decode_request(DecodeRequest::full(fmt))
            .expect("decode");
        let decoded: Vec<u16> = if case.precision > 8 {
            decoded
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_ne_bytes(*pair))
                .collect()
        } else {
            decoded.iter().map(|&sample| u16::from(sample)).collect()
        };
        assert_eq!(decoded, samples, "{}", case.name);
    }
}
