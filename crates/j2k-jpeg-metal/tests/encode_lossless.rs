// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal lossless (SOF3) encode must equal CPU lossless encode byte for byte.

#![cfg(target_os = "macos")]

use j2k_core::PixelFormat;
use j2k_jpeg::{
    encode_jpeg_lossless, JpegBackend, JpegEncodeError, JpegLosslessEncodeOptions,
    JpegLosslessSamples,
};
use j2k_jpeg_metal::{
    encode_jpeg_lossless_from_metal_buffer, Error, JpegLosslessMetalEncodeTile, MetalBackendSession,
};

fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

/// Samples in a padded device layout: `offset` spare bytes in front and
/// four spare bytes after each row.
struct DeviceImage {
    bytes: Vec<u8>,
    offset: usize,
    pitch: usize,
}

fn device_image(values: &[u16], width: u32, components: usize, wide: bool) -> DeviceImage {
    let sample_bytes = if wide { 2 } else { 1 };
    let row_samples = width as usize * components;
    let pitch = row_samples * sample_bytes + 4;
    let offset = 6;
    let rows = values.len() / row_samples;
    let mut bytes = vec![0xA5_u8; offset + pitch * rows];
    for (index, &value) in values.iter().enumerate() {
        let at = offset + index / row_samples * pitch + index % row_samples * sample_bytes;
        if wide {
            bytes[at..at + 2].copy_from_slice(&value.to_ne_bytes());
        } else {
            bytes[at] = u8::try_from(value).expect("8-bit sample");
        }
    }
    DeviceImage {
        bytes,
        offset,
        pitch,
    }
}

fn components(format: PixelFormat) -> usize {
    if matches!(format, PixelFormat::Gray8 | PixelFormat::Gray16) {
        1
    } else {
        3
    }
}

fn metal_encode(
    session: &MetalBackendSession,
    format: PixelFormat,
    (width, height): (u32, u32),
    values: &[u16],
    options: JpegLosslessEncodeOptions,
) -> Result<Vec<u8>, Error> {
    let wide = matches!(format, PixelFormat::Gray16 | PixelFormat::Rgb16);
    let image = device_image(values, width, components(format), wide);
    let buffer =
        j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &image.bytes)
            .expect("upload samples");
    // SAFETY: the upload completed synchronously, and nothing writes the
    // buffer while the tile is in use.
    let tile = unsafe {
        JpegLosslessMetalEncodeTile::new(
            &buffer,
            image.offset,
            (width, height),
            image.pitch,
            format,
        )
    };
    let encoded = encode_jpeg_lossless_from_metal_buffer(tile, options, session)?;
    assert_eq!(encoded.backend, JpegBackend::Metal);
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
fn metal_lossless_encode_matches_cpu_encode() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal session");
    let cases: &[Case] = &[
        (PixelFormat::Gray16, 16, 1, 0, None, (61, 47)),
        (PixelFormat::Gray16, 16, 7, 0, Some(3), (65, 11)),
        (PixelFormat::Gray16, 12, 6, 2, None, (32, 9)),
        (PixelFormat::Gray16, 10, 5, 0, Some(1), (33, 13)),
        (PixelFormat::Rgb16, 16, 4, 0, Some(2), (37, 23)),
        (PixelFormat::Rgb16, 9, 1, 3, None, (1, 17)),
        (PixelFormat::Gray8, 8, 1, 0, None, (97, 41)),
        (PixelFormat::Gray8, 5, 3, 1, Some(4), (64, 19)),
        (PixelFormat::Rgb8, 8, 2, 0, None, (45, 33)),
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
        let metal = metal_encode(&session, format, dims, &values, options).expect("Metal encode");
        let cpu = cpu_encode(format, dims, &values, options);
        assert!(
            metal == cpu,
            "{format:?} P{precision} SV{predictor} Pt{point_transform} {restart_rows:?} {dims:?}: Metal differs from CPU"
        );
    }
}

#[test]
fn metal_lossless_encode_matches_cjpeg_matrix() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut compared = 0;
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
        let metal = metal_encode(
            &session,
            format,
            (case.width, case.height),
            &values,
            options,
        )
        .expect("Metal encode");
        assert!(
            metal == case.jpeg,
            "{}: Metal differs from cjpeg",
            case.name
        );
        compared += 1;
    }
    assert!(compared >= 40, "only {compared} cjpeg cases");
}

#[test]
fn metal_lossless_encode_reports_the_first_out_of_range_sample() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut values = vec![100_u16; 64 * 8];
    values[300] = 5000;
    values[200] = 4096;
    let error = metal_encode(
        &session,
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

#[test]
fn metal_lossless_encode_validates_the_device_tile() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal session");
    let bytes = vec![0_u8; 64];
    let buffer = j2k_metal_support::checked_shared_buffer_with_slice(session.device(), &bytes)
        .expect("upload");
    // SAFETY: the upload completed synchronously and the buffer is not written.
    let unaligned =
        unsafe { JpegLosslessMetalEncodeTile::new(&buffer, 1, (4, 4), 8, PixelFormat::Gray16) };
    assert!(encode_jpeg_lossless_from_metal_buffer(
        unaligned,
        JpegLosslessEncodeOptions::new(12),
        &session
    )
    .is_err());
    // SAFETY: as above.
    let outside =
        unsafe { JpegLosslessMetalEncodeTile::new(&buffer, 0, (8, 8), 8, PixelFormat::Gray16) };
    assert!(encode_jpeg_lossless_from_metal_buffer(
        outside,
        JpegLosslessEncodeOptions::new(12),
        &session
    )
    .is_err());
    // SAFETY: as above.
    let rgba =
        unsafe { JpegLosslessMetalEncodeTile::new(&buffer, 0, (2, 2), 8, PixelFormat::Rgba8) };
    assert!(matches!(
        encode_jpeg_lossless_from_metal_buffer(rgba, JpegLosslessEncodeOptions::new(8), &session),
        Err(Error::Encode(
            JpegEncodeError::UnsupportedPixelFormat { .. }
        ))
    ));
}
