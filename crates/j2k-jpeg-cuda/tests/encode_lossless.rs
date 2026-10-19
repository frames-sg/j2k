// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA lossless (SOF3) encode must equal CPU lossless encode byte for byte.
//! These tests run on CUDA hosts with `J2K_REQUIRE_CUDA_RUNTIME` set and a
//! cuda-oxide build of the lossless encode PTX.

#![cfg(feature = "cuda-runtime")]

use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_lossless, JpegBackend, JpegEncodeError, JpegLosslessEncodeOptions,
    JpegLosslessSamples,
};
use j2k_jpeg_cuda::{
    encode_jpeg_lossless_from_cuda_buffer, CudaSession, Error, JpegLosslessCudaEncodeTile,
};

fn components(format: PixelFormat) -> usize {
    if matches!(format, PixelFormat::Gray8 | PixelFormat::Gray16) {
        1
    } else {
        3
    }
}

/// Samples in a padded device layout: six spare bytes in front and four
/// after each row.
fn device_bytes(values: &[u16], width: u32, format: PixelFormat) -> (Vec<u8>, usize, usize) {
    let wide = matches!(format, PixelFormat::Gray16 | PixelFormat::Rgb16);
    let sample_bytes = if wide { 2 } else { 1 };
    let row_samples = width as usize * components(format);
    let pitch = row_samples * sample_bytes + 4;
    let offset = 6;
    let mut bytes = vec![0x5A_u8; offset + pitch * (values.len() / row_samples)];
    for (index, &value) in values.iter().enumerate() {
        let at = offset + index / row_samples * pitch + index % row_samples * sample_bytes;
        if wide {
            bytes[at..at + 2].copy_from_slice(&value.to_ne_bytes());
        } else {
            bytes[at] = u8::try_from(value).expect("8-bit sample");
        }
    }
    (bytes, offset, pitch)
}

fn cuda_encode(
    format: PixelFormat,
    (width, height): (u32, u32),
    values: &[u16],
    options: JpegLosslessEncodeOptions,
) -> Result<Vec<u8>, Error> {
    let (bytes, offset, pitch) = device_bytes(values, width, format);
    let context = j2k_cuda_runtime::CudaContext::system_default().expect("CUDA context");
    let buffer = context.upload(&bytes).expect("upload samples");
    let mut session = CudaSession::default();
    let encoded = encode_jpeg_lossless_from_cuda_buffer(
        JpegLosslessCudaEncodeTile {
            buffer: &buffer,
            byte_offset: offset,
            width,
            height,
            pitch_bytes: pitch,
            format,
        },
        options,
        &mut session,
    )?;
    assert_eq!(encoded.backend, JpegBackend::Cuda);
    Ok(encoded.data)
}

fn cpu_encode(
    format: PixelFormat,
    (width, height): (u32, u32),
    values: &[u16],
    options: JpegLosslessEncodeOptions,
) -> Vec<u8> {
    let narrow: Vec<u8> = values
        .iter()
        .map(|&value| u8::try_from(value).unwrap_or(u8::MAX))
        .collect();
    let samples = match format {
        PixelFormat::Gray8 => JpegLosslessSamples::Gray8 {
            data: &narrow,
            width,
            height,
        },
        PixelFormat::Rgb8 => JpegLosslessSamples::Rgb8 {
            data: &narrow,
            width,
            height,
        },
        PixelFormat::Gray16 => JpegLosslessSamples::Gray16 {
            data: values,
            width,
            height,
        },
        _ => JpegLosslessSamples::Rgb16 {
            data: values,
            width,
            height,
        },
    };
    let mut cpu = options;
    cpu.backend = JpegBackend::Cpu;
    encode_jpeg_lossless(samples, cpu).expect("CPU encode").data
}

/// Format, precision, predictor, point transform, restart rows, dimensions.
type Case = (PixelFormat, u8, u8, u8, Option<u16>, (u32, u32));

/// Noise, ramps, and steps, so every category and code-word boundary occurs.
fn samples(count: usize, precision: u8, seed: u32) -> Vec<u16> {
    let max = (1_u32 << precision) - 1;
    (0..u32::try_from(count).expect("sample count"))
        .map(|i| {
            let value = (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 9) % 53
                + i * 7 % (max / 4 + 1)
                + (i / 5 % 6) * (max / 8);
            u16::try_from(value.min(max)).expect("sample")
        })
        .collect()
}

#[test]
fn cuda_lossless_encode_matches_cpu_encode_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let cases: &[Case] = &[
        (PixelFormat::Gray16, 16, 1, 0, None, (61, 47)),
        (PixelFormat::Gray16, 16, 7, 0, Some(3), (65, 11)),
        (PixelFormat::Gray16, 12, 6, 2, None, (32, 9)),
        (PixelFormat::Rgb16, 16, 4, 0, Some(2), (37, 23)),
        (PixelFormat::Gray8, 8, 1, 0, None, (97, 41)),
        (PixelFormat::Rgb8, 8, 7, 0, Some(5), (31, 30)),
        (PixelFormat::Gray16, 16, 1, 0, None, (2048, 37)),
    ];
    for (seed, &(format, precision, predictor, point_transform, restart_rows, dims)) in
        cases.iter().enumerate()
    {
        let mut options = JpegLosslessEncodeOptions::new(precision);
        options.predictor = predictor;
        options.point_transform = point_transform;
        options.restart_rows = restart_rows;
        let values = samples(
            dims.0 as usize * dims.1 as usize * components(format),
            precision,
            u32::try_from(seed).expect("seed"),
        );
        let cuda = cuda_encode(format, dims, &values, options).expect("CUDA encode");
        assert!(
            cuda == cpu_encode(format, dims, &values, options),
            "{format:?} P{precision} SV{predictor} Pt{point_transform} {restart_rows:?} {dims:?}: CUDA differs from CPU"
        );
    }
}

#[test]
fn cuda_lossless_encode_matches_cjpeg_matrix_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    for case in j2k_test_support::lossless_matrix_cases() {
        if case.scans.is_some() || !matches!(case.layout, "gray" | "rgb444") {
            continue;
        }
        let values = case.reference_samples().expect("cjpeg reference");
        let format = match (case.precision > 8, case.is_gray()) {
            (false, true) => PixelFormat::Gray8,
            (false, false) => PixelFormat::Rgb8,
            (true, true) => PixelFormat::Gray16,
            (true, false) => PixelFormat::Rgb16,
        };
        let mut options = JpegLosslessEncodeOptions::new(case.precision);
        options.predictor = case.predictor;
        options.point_transform = case.point_transform;
        options.restart_rows = (case.restart_rows != 0)
            .then(|| u16::try_from(case.restart_rows).expect("restart rows"));
        let cuda =
            cuda_encode(format, (case.width, case.height), &values, options).expect("CUDA encode");
        assert!(cuda == case.jpeg, "{}: CUDA differs from cjpeg", case.name);
    }
}

#[test]
fn cuda_lossless_encode_reports_the_first_out_of_range_sample_when_required() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let mut values = vec![100_u16; 64 * 8];
    values[300] = 5000;
    values[200] = 4096;
    let error = cuda_encode(
        PixelFormat::Gray16,
        (64, 8),
        &values,
        JpegLosslessEncodeOptions::new(12),
    )
    .expect_err("out of range");
    assert!(
        matches!(
            error,
            Error::Encode(JpegEncodeError::SampleOutOfRange {
                index: 200,
                value: 4096,
                precision: 12
            })
        ),
        "{error:?}"
    );
}
