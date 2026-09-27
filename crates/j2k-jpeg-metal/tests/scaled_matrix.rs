// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal DCT-scaled decodes against the libjpeg-turbo reference matrix.
//!
//! The Metal fast families (8-bit baseline 4:4:4, 4:2:2 and 4:2:0) must
//! reproduce libjpeg-turbo's scaled output exactly: 4:2:0 chroma decoded with
//! the larger IDCT below full size, 4:2:2 chroma replicated at 1/8 and when at
//! most two samples wide. Images at most four pixels wide, whose full-size
//! 2:1 chroma libjpeg-turbo replicates, are not Metal fast shapes and must be
//! rejected by explicit Metal requests rather than decoded differently.
#![cfg(target_os = "macos")]

use j2k_core::{BackendKind, BackendRequest, DeviceSurface, Downscale, PixelFormat, Rect};
use j2k_jpeg_metal::{Decoder, JpegTileBatch, MetalDecodeRequest};
use j2k_test_support::{
    crop_interleaved_bytes, scaled_matrix_cases, scaled_rect_covering, PixelRect, ScaledMatrixCase,
    SCALED_MATRIX_DENOMINATORS,
};

fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}

fn downscale(denominator: u32) -> Downscale {
    match denominator {
        1 => Downscale::None,
        2 => Downscale::Half,
        4 => Downscale::Quarter,
        8 => Downscale::Eighth,
        _ => unreachable!("matrix denominators are 1, 2, 4, 8"),
    }
}

/// 8-bit sequential cases in the Metal fast families.
fn metal_family_cases() -> Vec<ScaledMatrixCase> {
    scaled_matrix_cases()
        .into_iter()
        .filter(|case| {
            case.precision == 8
                && matches!(case.layout, "444" | "422" | "420")
                && matches!(case.coding, "baseline" | "restart")
        })
        .collect()
}

/// Full-size 2:1 chroma at most two samples wide: libjpeg-turbo replicates
/// it, so these images are not Metal fast shapes.
fn is_narrow_subsampled(case: &ScaledMatrixCase) -> bool {
    case.layout != "444" && case.width <= 4
}

fn expected_rgba(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .flat_map(|px| [px[0], px[1], px[2], u8::MAX])
        .collect()
}

fn mismatch(expected: &[u8], actual: &[u8]) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "{} bytes, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let diffs: Vec<u8> = expected
        .iter()
        .zip(actual)
        .map(|(&e, &a)| e.abs_diff(a))
        .filter(|&d| d > 0)
        .collect();
    let max = diffs.iter().copied().max()?;
    Some(format!(
        "{} of {} bytes differ, max {max}",
        diffs.len(),
        expected.len()
    ))
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} Metal scaled decodes differ from libjpeg-turbo:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn region_for(case: &ScaledMatrixCase) -> Rect {
    Rect {
        x: case.width / 3,
        y: case.height / 4,
        w: case.width / 2,
        h: case.height / 2,
    }
}

fn expected_region(case: &ScaledMatrixCase, roi: Rect, denominator: u32) -> (PixelRect, Vec<u8>) {
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
    let crop = crop_interleaved_bytes(case.reference(denominator), width as usize, 3, scaled);
    (scaled, crop)
}

#[test]
fn metal_scaled_decodes_match_libjpeg_turbo() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for case in metal_family_cases()
        .iter()
        .filter(|case| !is_narrow_subsampled(case))
    {
        for denominator in SCALED_MATRIX_DENOMINATORS {
            for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
                let label = format!("{} {fmt:?} 1/{denominator}", case.name);
                let mut decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
                let request =
                    MetalDecodeRequest::scaled(fmt, downscale(denominator), BackendRequest::Metal);
                match decoder.decode_request_to_device(request) {
                    Ok(surface) => {
                        assert_eq!(surface.backend_kind(), BackendKind::Metal, "{label}");
                        let rgb = case.reference(denominator);
                        let expected = if fmt == PixelFormat::Rgba8 {
                            expected_rgba(rgb)
                        } else {
                            rgb.to_vec()
                        };
                        let bytes = surface.as_bytes().expect("surface bytes");
                        if let Some(message) = mismatch(&expected, &bytes) {
                            failures.push(format!("{label}: {message}"));
                        }
                    }
                    Err(err) => failures.push(format!("{label}: Metal decode failed: {err}")),
                }
            }
        }
    }
    assert_no_failures(&failures);
}

#[test]
fn metal_scaled_region_decodes_match_libjpeg_turbo_crops() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for case in metal_family_cases().iter().filter(|case| case.width >= 18) {
        let roi = region_for(case);
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let label = format!("{} region 1/{denominator}", case.name);
            let mut decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
            let request = MetalDecodeRequest::region_scaled(
                PixelFormat::Rgb8,
                roi,
                downscale(denominator),
                BackendRequest::Metal,
            );
            let (scaled, expected) = expected_region(case, roi, denominator);
            match decoder.decode_request_to_device(request) {
                Ok(surface) => {
                    assert_eq!(surface.dimensions(), (scaled.w, scaled.h), "{label}");
                    let bytes = surface.as_bytes().expect("surface bytes");
                    if let Some(message) = mismatch(&expected, &bytes) {
                        failures.push(format!("{label}: {message}"));
                    }
                }
                Err(err) => failures.push(format!("{label}: Metal decode failed: {err}")),
            }
        }
    }
    assert_no_failures(&failures);
}

/// Batches group same-shape tiles into shared dispatches; every tile in a
/// batch must still match its own reference.
#[test]
fn metal_scaled_tile_batches_match_libjpeg_turbo() {
    if !should_run_metal_runtime() {
        return;
    }
    let cases: Vec<ScaledMatrixCase> = metal_family_cases()
        .into_iter()
        .filter(|case| !is_narrow_subsampled(case))
        .collect();
    let mut failures = Vec::new();
    for denominator in SCALED_MATRIX_DENOMINATORS {
        for region in [false, true] {
            let mut batch = JpegTileBatch::with_capacity(cases.len() * 2);
            let mut expected = Vec::new();
            for case in cases.iter().filter(|case| !region || case.width >= 18) {
                // Two copies so compatible tiles share a grouped dispatch.
                for _ in 0..2 {
                    let request = if region {
                        let roi = region_for(case);
                        expected.push((case.name, expected_region(case, roi, denominator).1));
                        MetalDecodeRequest::region_scaled(
                            PixelFormat::Rgb8,
                            roi,
                            downscale(denominator),
                            BackendRequest::Metal,
                        )
                    } else {
                        expected.push((case.name, case.reference(denominator).to_vec()));
                        MetalDecodeRequest::scaled(
                            PixelFormat::Rgb8,
                            downscale(denominator),
                            BackendRequest::Metal,
                        )
                    };
                    batch
                        .push_tile_request(case.jpeg, request)
                        .expect("push batch tile");
                }
            }
            let label = if region { "region batch" } else { "batch" };
            match batch.decode_all() {
                Ok(surfaces) => {
                    for (surface, (name, expected)) in surfaces.iter().zip(&expected) {
                        let bytes = surface.as_bytes().expect("surface bytes");
                        if let Some(message) = mismatch(expected, &bytes) {
                            failures.push(format!("{name} {label} 1/{denominator}: {message}"));
                        }
                    }
                }
                Err(err) => failures.push(format!("{label} 1/{denominator} failed: {err}")),
            }
        }
    }
    assert_no_failures(&failures);
}

#[test]
fn explicit_metal_rejects_narrow_subsampled_images() {
    if !should_run_metal_runtime() {
        return;
    }
    for case in metal_family_cases()
        .iter()
        .filter(|case| is_narrow_subsampled(case))
    {
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let mut decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
            let request = MetalDecodeRequest::scaled(
                PixelFormat::Rgb8,
                downscale(denominator),
                BackendRequest::Metal,
            );
            assert!(
                decoder.decode_request_to_device(request).is_err(),
                "{} 1/{denominator}: narrow 2:1 chroma must not take the Metal fast path",
                case.name
            );
        }
    }
}

/// Region edges on and beside MCU boundaries at every scale, where the
/// smoothing filters need chroma from outside the decoded window.
#[test]
fn metal_scaled_region_edge_sweep_matches_libjpeg_turbo_crops() {
    if !should_run_metal_runtime() {
        return;
    }
    let xs = [0u32, 7, 8, 9, 15, 16, 17];
    let ys = [0u32, 7, 8, 9, 16, 17];
    let mut failures = Vec::new();
    for case in metal_family_cases().iter().filter(|case| case.width == 45) {
        for denominator in SCALED_MATRIX_DENOMINATORS {
            for &x0 in &xs {
                for &x1 in &xs {
                    for &y0 in &ys {
                        for &y1 in &ys {
                            let (x1, y1) = (x1 + 8, y1 + 8);
                            if x0 >= x1 || y0 >= y1 || x1 > case.width || y1 > case.height {
                                continue;
                            }
                            let roi = Rect {
                                x: x0,
                                y: y0,
                                w: x1 - x0,
                                h: y1 - y0,
                            };
                            let label = format!(
                                "{} roi {},{} {}x{} 1/{denominator}",
                                case.name, roi.x, roi.y, roi.w, roi.h
                            );
                            let mut decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
                            let request = MetalDecodeRequest::region_scaled(
                                PixelFormat::Rgb8,
                                roi,
                                downscale(denominator),
                                BackendRequest::Metal,
                            );
                            let (_, expected) = expected_region(case, roi, denominator);
                            match decoder.decode_request_to_device(request) {
                                Ok(surface) => {
                                    let bytes = surface.as_bytes().expect("surface bytes");
                                    if let Some(message) = mismatch(&expected, &bytes) {
                                        failures.push(format!("{label}: {message}"));
                                    }
                                }
                                Err(err) => failures.push(format!("{label}: {err}")),
                            }
                        }
                    }
                }
            }
        }
    }
    assert_no_failures(&failures);
}
