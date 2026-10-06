// SPDX-License-Identifier: MIT OR Apache-2.0

//! DCT-scaled decodes against the libjpeg-turbo reference matrix.
//!
//! libjpeg-turbo gives each component its own reduced IDCT size (chroma is
//! decoded larger so it needs less upsampling), replicates instead of
//! smoothing at 1/8 and for 2:1 components at most two samples wide, and
//! applies the reduced IDCT to progressive and 12-bit coefficients. `OpenSlide`
//! scale-decodes NDPI/VMS levels with it, so j2k must match it bit for bit.

use std::panic::{catch_unwind, AssertUnwindSafe};

use j2k_jpeg::{
    decode_tiles_region_scaled_into, decode_tiles_scaled_into, DecodeOutcome, DecodeRequest,
    Decoder, Downscale, JpegCapabilityReport, JpegCapabilityRequest, JpegDecodeOp, JpegError,
    PixelFormat, Rect, TileBatchOptions, TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use j2k_test_support::{
    crop_interleaved_bytes, scaled_matrix_cases, scaled_rect_covering, PixelRect, ScaledMatrixCase,
    SCALED_MATRIX_DENOMINATORS,
};

fn downscale(denominator: u32) -> Downscale {
    match denominator {
        1 => Downscale::None,
        2 => Downscale::Half,
        4 => Downscale::Quarter,
        8 => Downscale::Eighth,
        _ => unreachable!("matrix denominators are 1, 2, 4, 8"),
    }
}

fn pixel_format(case: &ScaledMatrixCase) -> PixelFormat {
    match (case.is_gray(), case.precision > 8) {
        (true, false) => PixelFormat::Gray8,
        (false, false) => PixelFormat::Rgb8,
        (true, true) => PixelFormat::Gray16,
        (false, true) => PixelFormat::Rgb16,
    }
}

/// Reference samples as native values (8-bit bytes or 12-bit values).
fn reference_samples(case: &ScaledMatrixCase, denominator: u32) -> Vec<u16> {
    if case.precision > 8 {
        case.reference_u16(denominator)
    } else {
        case.reference(denominator)
            .iter()
            .map(|&v| u16::from(v))
            .collect()
    }
}

/// Decoder output bytes as native values (16-bit output is little-endian).
fn output_samples(case: &ScaledMatrixCase, bytes: &[u8]) -> Vec<u16> {
    if case.precision > 8 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    } else {
        bytes.iter().map(|&v| u16::from(v)).collect()
    }
}

/// Decodes `request`, turning a panic into a reported failure so one bad case
/// cannot hide the rest of the matrix.
fn decode(decoder: &Decoder<'_>, request: DecodeRequest) -> Result<Vec<u8>, String> {
    let outcome: Result<(Vec<u8>, DecodeOutcome), JpegError> =
        match catch_unwind(AssertUnwindSafe(|| decoder.decode_request(request))) {
            Ok(outcome) => outcome,
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("non-string panic");
                return Err(format!("decode panicked: {message}"));
            }
        };
    outcome
        .map(|(bytes, _)| bytes)
        .map_err(|err| format!("decode failed: {err}"))
}

fn describe_mismatch(expected: &[u16], actual: &[u16]) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "{} samples, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let diffs: Vec<u16> = expected
        .iter()
        .zip(actual)
        .map(|(&e, &a)| e.abs_diff(a))
        .filter(|&d| d > 0)
        .collect();
    let max = diffs.iter().copied().max()?;
    Some(format!(
        "{} of {} samples differ, max {max}",
        diffs.len(),
        expected.len()
    ))
}

fn crop_samples(samples: &[u16], width: u32, channels: usize, rect: PixelRect) -> Vec<u16> {
    let mut out = Vec::new();
    for y in rect.y..rect.y + rect.h {
        let start = (y * width + rect.x) as usize * channels;
        out.extend_from_slice(&samples[start..start + rect.w as usize * channels]);
    }
    out
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} scaled decodes differ from libjpeg-turbo:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn scaled_decodes_match_libjpeg_turbo() {
    let mut failures = Vec::new();
    for case in scaled_matrix_cases() {
        let decoder = match Decoder::new(case.jpeg) {
            Ok(decoder) => decoder,
            Err(err) => {
                failures.push(format!("{}: parse failed: {err}", case.name));
                continue;
            }
        };
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let request = DecodeRequest::scaled(pixel_format(&case), downscale(denominator));
            let label = format!("{} 1/{denominator}", case.name);
            match decode(&decoder, request) {
                Ok(bytes) => {
                    let expected = reference_samples(&case, denominator);
                    if let Some(message) =
                        describe_mismatch(&expected, &output_samples(&case, &bytes))
                    {
                        failures.push(format!("{label}: {message}"));
                    }
                }
                Err(err) => failures.push(format!("{label}: {err}")),
            }
        }
    }
    assert_no_failures(&failures);
}

#[test]
fn scaled_rgba_decodes_match_libjpeg_turbo() {
    let mut failures = Vec::new();
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| !case.is_gray() && case.precision == 8)
    {
        let decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let request = DecodeRequest::scaled(PixelFormat::Rgba8, downscale(denominator));
            let label = format!("{} rgba 1/{denominator}", case.name);
            match decode(&decoder, request) {
                Ok(bytes) => {
                    let rgb: Vec<u16> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|px| px[..3].iter().map(|&v| u16::from(v)))
                        .collect();
                    let alpha_ok = bytes.as_chunks::<4>().0.iter().all(|px| px[3] == u8::MAX);
                    if let Some(message) =
                        describe_mismatch(&reference_samples(&case, denominator), &rgb)
                    {
                        failures.push(format!("{label}: {message}"));
                    } else if !alpha_ok {
                        failures.push(format!("{label}: alpha is not opaque"));
                    }
                }
                Err(err) => failures.push(format!("{label}: {err}")),
            }
        }
    }
    assert_no_failures(&failures);
}

/// Every 12-bit color layout, including those only the component-plane
/// pipeline renders, is CPU-eligible and decodes to opaque `Rgba16`.
#[test]
fn twelve_bit_color_layouts_are_cpu_eligible_and_decode_rgba16() {
    let mut failures = Vec::new();
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| !case.is_gray() && case.precision > 8)
    {
        let report = JpegCapabilityReport::inspect(
            case.jpeg,
            JpegCapabilityRequest {
                op: JpegDecodeOp::Full,
                fmt: PixelFormat::Rgb16,
            },
        )
        .expect("matrix JPEG capability report");
        if !report.cpu.eligible {
            failures.push(format!("{}: CPU rejected Rgb16", case.name));
        }
        let decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let request = DecodeRequest::scaled(PixelFormat::Rgba16, downscale(denominator));
            let label = format!("{} rgba16 1/{denominator}", case.name);
            match decode(&decoder, request) {
                Ok(bytes) => {
                    let samples = output_samples(&case, &bytes);
                    let rgb: Vec<u16> = samples
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .flat_map(|px| px[..3].to_vec())
                        .collect();
                    if let Some(message) =
                        describe_mismatch(&reference_samples(&case, denominator), &rgb)
                    {
                        failures.push(format!("{label}: {message}"));
                    } else if !samples
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|px| px[3] == u16::MAX)
                    {
                        failures.push(format!("{label}: alpha is not opaque"));
                    }
                }
                Err(err) => failures.push(format!("{label}: {err}")),
            }
        }
    }
    assert_no_failures(&failures);
}

/// Scaled regions are crops of the scaled image, wherever the region starts.
#[test]
fn scaled_region_decodes_match_libjpeg_turbo_crops() {
    let mut failures = Vec::new();
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| case.width >= 18 || case.height >= 64)
    {
        let decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
        let rois = [
            Rect {
                x: 5.min(case.width / 3),
                y: 3,
                w: case.width - 9.min(case.width / 2),
                h: case.height - 6,
            },
            Rect {
                x: case.width / 2 + 1,
                y: case.height / 2 - 1,
                w: case.width / 2 - 1,
                h: case.height / 2,
            },
        ];
        for roi in rois {
            for denominator in SCALED_MATRIX_DENOMINATORS {
                let request =
                    DecodeRequest::region_scaled(pixel_format(&case), roi, downscale(denominator));
                let label = format!(
                    "{} roi {},{} {}x{} 1/{denominator}",
                    case.name, roi.x, roi.y, roi.w, roi.h
                );
                let scaled = scaled_rect_covering(
                    PixelRect {
                        x: roi.x,
                        y: roi.y,
                        w: roi.w,
                        h: roi.h,
                    },
                    denominator,
                );
                let (width, _) = case.scaled_dimensions(denominator);
                let expected = crop_samples(
                    &reference_samples(&case, denominator),
                    width,
                    case.channels(),
                    scaled,
                );
                match decode(&decoder, request) {
                    Ok(bytes) => {
                        if let Some(message) =
                            describe_mismatch(&expected, &output_samples(&case, &bytes))
                        {
                            failures.push(format!("{label}: {message}"));
                        }
                    }
                    Err(err) => failures.push(format!("{label}: {err}")),
                }
            }
        }
    }
    assert_no_failures(&failures);
}

#[test]
fn crop_helper_matches_interleaved_byte_crop() {
    let samples: Vec<u8> = (0..60).collect();
    let rect = PixelRect {
        x: 1,
        y: 1,
        w: 2,
        h: 2,
    };
    let widened: Vec<u16> = samples.iter().map(|&v| u16::from(v)).collect();
    let expected: Vec<u16> = crop_interleaved_bytes(&samples, 5, 3, rect)
        .into_iter()
        .map(u16::from)
        .collect();
    assert_eq!(crop_samples(&widened, 5, 3, rect), expected);
}

/// The batch tile API (what `wsi-rs` calls for NDPI/VMS scaled levels) routes
/// through the same scaled geometry as single decodes.
#[test]
fn batch_scaled_tile_decodes_match_libjpeg_turbo() {
    let cases: Vec<ScaledMatrixCase> = scaled_matrix_cases()
        .into_iter()
        .filter(|case| case.precision == 8)
        .collect();
    let mut failures = Vec::new();
    for denominator in SCALED_MATRIX_DENOMINATORS {
        let mut outputs: Vec<Vec<u8>> = cases
            .iter()
            .map(|case| vec![0u8; case.reference(denominator).len() * 3 / case.channels()])
            .collect();
        let mut jobs: Vec<TileScaledDecodeJob<'_, '_>> = cases
            .iter()
            .zip(outputs.iter_mut())
            .map(|(case, out)| TileScaledDecodeJob {
                input: case.jpeg,
                stride: case.scaled_dimensions(denominator).0 as usize * 3,
                out: out.as_mut_slice(),
                scale: downscale(denominator),
            })
            .collect();
        // Grayscale cases decode to RGB here: replicate their references.
        if let Err(err) =
            decode_tiles_scaled_into(&mut jobs, PixelFormat::Rgb8, TileBatchOptions::default())
        {
            failures.push(format!("1/{denominator} batch failed: {err}"));
        }
        drop(jobs);
        for (case, out) in cases.iter().zip(&outputs) {
            let expected = rgb_reference(case, denominator);
            let actual: Vec<u16> = out.iter().map(|&v| u16::from(v)).collect();
            if let Some(message) = describe_mismatch(&expected, &actual) {
                failures.push(format!("{} batch 1/{denominator}: {message}", case.name));
            }
        }
    }
    assert_no_failures(&failures);
}

#[test]
fn batch_scaled_region_tile_decodes_match_libjpeg_turbo_crops() {
    let cases: Vec<ScaledMatrixCase> = scaled_matrix_cases()
        .into_iter()
        .filter(|case| case.precision == 8 && case.width >= 18)
        .collect();
    let mut failures = Vec::new();
    for denominator in SCALED_MATRIX_DENOMINATORS {
        let rois: Vec<Rect> = cases
            .iter()
            .map(|case| Rect {
                x: case.width / 3,
                y: case.height / 4,
                w: case.width / 2,
                h: case.height / 2,
            })
            .collect();
        let scaled: Vec<PixelRect> = rois
            .iter()
            .map(|roi| {
                scaled_rect_covering(
                    PixelRect {
                        x: roi.x,
                        y: roi.y,
                        w: roi.w,
                        h: roi.h,
                    },
                    denominator,
                )
            })
            .collect();
        let mut outputs: Vec<Vec<u8>> = scaled
            .iter()
            .map(|rect| vec![0u8; rect.w as usize * rect.h as usize * 3])
            .collect();
        let mut jobs: Vec<TileRegionScaledDecodeJob<'_, '_>> = cases
            .iter()
            .zip(&rois)
            .zip(&scaled)
            .zip(outputs.iter_mut())
            .map(|(((case, &roi), rect), out)| TileRegionScaledDecodeJob {
                input: case.jpeg,
                out: out.as_mut_slice(),
                stride: rect.w as usize * 3,
                roi: roi.into(),
                scale: downscale(denominator),
            })
            .collect();
        if let Err(err) = decode_tiles_region_scaled_into(
            &mut jobs,
            PixelFormat::Rgb8,
            TileBatchOptions::default(),
        ) {
            failures.push(format!("1/{denominator} batch failed: {err}"));
        }
        drop(jobs);
        for ((case, rect), out) in cases.iter().zip(&scaled).zip(&outputs) {
            let (width, _) = case.scaled_dimensions(denominator);
            let expected = crop_samples(&rgb_reference(case, denominator), width, 3, *rect);
            let actual: Vec<u16> = out.iter().map(|&v| u16::from(v)).collect();
            if let Some(message) = describe_mismatch(&expected, &actual) {
                failures.push(format!(
                    "{} batch region 1/{denominator}: {message}",
                    case.name
                ));
            }
        }
    }
    assert_no_failures(&failures);
}

/// 8-bit reference as interleaved RGB (grayscale expanded to R = G = B).
fn rgb_reference(case: &ScaledMatrixCase, denominator: u32) -> Vec<u16> {
    let samples = reference_samples(case, denominator);
    if case.is_gray() {
        samples.iter().flat_map(|&v| [v, v, v]).collect()
    } else {
        samples
    }
}

/// Region edges on and beside MCU boundaries at every scale, where smoothing
/// filters need chroma from outside the decoded window.
fn edge_sweep_rois(width: u32, height: u32) -> Vec<Rect> {
    let xs = [0, 7, 8, 9, 15, 16, 17];
    let ys = [0, 7, 8, 9, 16, 17];
    let mut rois = Vec::new();
    for &x0 in &xs {
        for &x1 in &xs {
            for &y0 in &ys {
                for &y1 in &ys {
                    let (x1, y1) = (x1 + 8, y1 + 8);
                    if x0 < x1 && y0 < y1 && x1 <= width && y1 <= height {
                        rois.push(Rect {
                            x: x0,
                            y: y0,
                            w: x1 - x0,
                            h: y1 - y0,
                        });
                    }
                }
            }
        }
    }
    rois
}

#[test]
fn scaled_region_edge_sweep_matches_libjpeg_turbo_crops() {
    let mut failures = Vec::new();
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| case.width == 45)
    {
        let decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let reference = reference_samples(&case, denominator);
            let (width, _) = case.scaled_dimensions(denominator);
            for roi in edge_sweep_rois(case.width, case.height) {
                let scaled = scaled_rect_covering(
                    PixelRect {
                        x: roi.x,
                        y: roi.y,
                        w: roi.w,
                        h: roi.h,
                    },
                    denominator,
                );
                let expected = crop_samples(&reference, width, case.channels(), scaled);
                let request =
                    DecodeRequest::region_scaled(pixel_format(&case), roi, downscale(denominator));
                let label = format!(
                    "{} roi {},{} {}x{} 1/{denominator}",
                    case.name, roi.x, roi.y, roi.w, roi.h
                );
                match decode(&decoder, request) {
                    Ok(bytes) => {
                        if let Some(message) =
                            describe_mismatch(&expected, &output_samples(&case, &bytes))
                        {
                            failures.push(format!("{label}: {message}"));
                        }
                    }
                    Err(err) => failures.push(format!("{label}: {err}")),
                }
            }
        }
    }
    assert_no_failures(&failures);
}

/// Grayscale images decoded to RGB or RGBA replicate the gray sample into
/// every channel at every scale, for 8-bit and 12-bit precision.
#[test]
fn scaled_gray_to_color_decodes_match_libjpeg_turbo() {
    let mut failures = Vec::new();
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(ScaledMatrixCase::is_gray)
    {
        let decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
        let formats = if case.precision > 8 {
            [(PixelFormat::Rgb16, 3), (PixelFormat::Rgba16, 4)]
        } else {
            [(PixelFormat::Rgb8, 3), (PixelFormat::Rgba8, 4)]
        };
        let full = Rect {
            x: 0,
            y: 0,
            w: case.width,
            h: case.height,
        };
        let inner = Rect {
            x: case.width / 3,
            y: case.height / 3,
            w: case.width - case.width / 3,
            h: case.height - case.height / 3,
        };
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let gray = reference_samples(&case, denominator);
            let (width, _) = case.scaled_dimensions(denominator);
            for roi in [full, inner] {
                let scaled = scaled_rect_covering(
                    PixelRect {
                        x: roi.x,
                        y: roi.y,
                        w: roi.w,
                        h: roi.h,
                    },
                    denominator,
                );
                let crop = crop_samples(&gray, width, 1, scaled);
                for (fmt, channels) in formats {
                    let alpha = if case.precision > 8 { u16::MAX } else { 255 };
                    let expected: Vec<u16> = crop
                        .iter()
                        .flat_map(|&v| {
                            let mut px = vec![v; 3];
                            if channels == 4 {
                                px.push(alpha);
                            }
                            px
                        })
                        .collect();
                    let request = DecodeRequest::region_scaled(fmt, roi, downscale(denominator));
                    let label = format!(
                        "{} {fmt:?} roi {},{} {}x{} 1/{denominator}",
                        case.name, roi.x, roi.y, roi.w, roi.h
                    );
                    match decode(&decoder, request) {
                        Ok(bytes) => {
                            if let Some(message) =
                                describe_mismatch(&expected, &output_samples(&case, &bytes))
                            {
                                failures.push(format!("{label}: {message}"));
                            }
                        }
                        Err(err) => failures.push(format!("{label}: {err}")),
                    }
                }
            }
        }
    }
    assert_no_failures(&failures);
}
