// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hybrid Metal decode against CPU decode and libjpeg-turbo.
//!
//! JPEG classes outside the fast-packet kernels (grayscale, progressive,
//! 12-bit, CMYK/YCCK, APP14 RGB, 4:4:0/4:1:1/4:1:0/1x4 sampling, narrow
//! subsampled images) decode through CPU entropy decode followed by Metal
//! IDCT, upsampling, color conversion, and packing. Explicit Metal requests
//! must accept exactly what CPU decode accepts and return identical bytes in
//! Metal-resident storage.
#![cfg(target_os = "macos")]

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect, SurfaceResidency};
use j2k_jpeg::DecodeRequest;
use j2k_jpeg_metal::{Decoder, Error, MetalBackendSession, MetalDecodeRequest};
use j2k_test_support::{scaled_matrix_cases, SCALED_MATRIX_DENOMINATORS};

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

fn metal_request(request: DecodeRequest) -> MetalDecodeRequest {
    match request.region {
        None => MetalDecodeRequest::scaled(request.fmt, request.scale, BackendRequest::Metal),
        Some(roi) => MetalDecodeRequest::region_scaled(
            request.fmt,
            Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            request.scale,
            BackendRequest::Metal,
        ),
    }
}

fn first_difference(expected: &[u8], actual: &[u8]) -> Option<String> {
    if expected.len() != actual.len() {
        return Some(format!(
            "{} bytes, expected {}",
            actual.len(),
            expected.len()
        ));
    }
    let index = expected.iter().zip(actual).position(|(e, a)| e != a)?;
    let differing = expected.iter().zip(actual).filter(|(e, a)| e != a).count();
    Some(format!(
        "{differing} of {} bytes differ; first at {index}: metal {} cpu {}",
        expected.len(),
        actual[index],
        expected[index]
    ))
}

/// Explicit Metal decode of `request`, checked against CPU decode. Returns
/// whether CPU decode accepted the request; Metal must then succeed with
/// identical Metal-resident bytes, and must fail when CPU decode fails.
fn check_against_cpu(
    name: &str,
    jpeg: &[u8],
    request: DecodeRequest,
    failures: &mut Vec<String>,
) -> bool {
    let cpu = j2k_jpeg::Decoder::new(jpeg)
        .expect("CPU decoder")
        .decode_request(request);
    let mut decoder = Decoder::new(jpeg).expect("Metal decoder");
    let metal = decoder.decode_request_to_device(metal_request(request));
    match (cpu, metal) {
        (Ok((expected, _)), Ok(surface)) => {
            if surface.residency() != SurfaceResidency::MetalResidentDecode {
                failures.push(format!(
                    "{name} {request:?}: residency {:?}",
                    surface.residency()
                ));
            }
            let actual = surface.as_bytes().expect("surface bytes");
            if let Some(message) = first_difference(&expected, &actual) {
                failures.push(format!("{name} {request:?}: {message}"));
            }
            true
        }
        (Ok(_), Err(err)) => {
            failures.push(format!(
                "{name} {request:?}: CPU decodes but Metal failed: {err}"
            ));
            true
        }
        (Err(_), Ok(_)) => {
            failures.push(format!("{name} {request:?}: CPU rejects but Metal decoded"));
            false
        }
        (Err(_), Err(_)) => false,
    }
}

fn formats_for(precision: u8) -> [PixelFormat; 3] {
    if precision == 8 {
        [PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8]
    } else {
        [PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16]
    }
}

fn requests_for(precision: u8, (w, h): (u32, u32)) -> Vec<DecodeRequest> {
    let regions = [
        None,
        Some(j2k_jpeg::Rect {
            x: w / 3,
            y: h / 4,
            w: (w - w / 3).div_ceil(2).max(1),
            h: (h - h / 4).div_ceil(2).max(1),
        }),
        Some(j2k_jpeg::Rect {
            x: w.saturating_sub(1),
            y: h.saturating_sub(1),
            w: 1,
            h: 1,
        }),
    ];
    let mut requests = Vec::new();
    for fmt in formats_for(precision) {
        for denominator in SCALED_MATRIX_DENOMINATORS {
            for region in regions {
                requests.push(DecodeRequest {
                    fmt,
                    region,
                    scale: downscale(denominator),
                });
            }
        }
    }
    requests
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} Metal decodes differ from CPU decode:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn explicit_metal_matches_cpu_across_the_scaled_matrix() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    let (mut compared, mut rejected) = (0_usize, 0_usize);
    for case in scaled_matrix_cases() {
        for request in requests_for(case.precision, (case.width, case.height)) {
            if check_against_cpu(case.name, case.jpeg, request, &mut failures) {
                compared += 1;
                continue;
            }
            // CPU decode rejects only Gray16 output from 12-bit color images.
            if !(request.fmt == PixelFormat::Gray16 && case.precision > 8 && !case.is_gray()) {
                failures.push(format!("{} {request:?}: CPU decode rejected it", case.name));
            }
            rejected += 1;
        }
    }
    assert_no_failures(&failures);
    assert!(
        compared > 10_000 && rejected > 0,
        "compared {compared}, rejected {rejected}"
    );
}

#[test]
fn explicit_metal_matches_libjpeg_turbo_for_every_layout_and_coding() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| case.precision == 8)
    {
        let fmt = if case.is_gray() {
            PixelFormat::Gray8
        } else {
            PixelFormat::Rgb8
        };
        for denominator in SCALED_MATRIX_DENOMINATORS {
            let mut decoder = Decoder::new(case.jpeg).expect("matrix JPEG parses");
            let request =
                MetalDecodeRequest::scaled(fmt, downscale(denominator), BackendRequest::Metal);
            match decoder.decode_request_to_device(request) {
                Ok(surface) => {
                    let actual = surface.as_bytes().expect("surface bytes");
                    if let Some(message) = first_difference(case.reference(denominator), &actual) {
                        failures.push(format!("{} 1/{denominator}: {message}", case.name));
                    }
                }
                Err(err) => failures.push(format!("{} 1/{denominator}: {err}", case.name)),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} Metal decodes differ from libjpeg-turbo:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn fixture_cases() -> Vec<(&'static str, Vec<u8>)> {
    use j2k_test_support as fx;
    vec![
        ("grayscale_8x8", fx::grayscale_8x8_jpeg()),
        ("grayscale_odd", fx::baseline_grayscale_jpeg(37, 29)),
        (
            "grayscale_restart",
            fx::restart_coded_grayscale_jpeg(41, 23),
        ),
        ("progressive_8x8", fx::progressive_8x8_jpeg()),
        ("rgb_app14_8x8", fx::rgb_app14_8x8_jpeg()),
        ("cmyk_8x8", fx::cmyk_8x8_jpeg()),
        ("cmyk_16x8_422", fx::cmyk_16x8_422_jpeg()),
        ("cmyk_16x16_420", fx::cmyk_16x16_420_jpeg()),
        (
            "cmyk_nonleading_max_422",
            fx::cmyk_16x8_nonleading_max_422_jpeg(),
        ),
        ("ycck_8x8", fx::ycck_8x8_jpeg()),
        ("ycck_16x8_422", fx::ycck_16x8_422_jpeg()),
        ("ycck_16x16_420", fx::ycck_16x16_420_jpeg()),
        (
            "ycck_nonleading_max_422",
            fx::ycck_16x8_nonleading_max_422_jpeg(),
        ),
        ("e12_gray_8x8", fx::extended_12bit_grayscale_8x8_jpeg()),
        (
            "e12_gray_restart",
            fx::extended_12bit_grayscale_restart_16x8_jpeg(),
        ),
        ("e12_rgb_8x8", fx::extended_12bit_rgb_8x8_jpeg()),
        ("e12_rgb_420", fx::extended_12bit_rgb_420_32x32_jpeg()),
        (
            "e12_ycbcr_422_restart",
            fx::extended_12bit_ycbcr_422_restart_32x8_jpeg(),
        ),
        (
            "e12_ycbcr_420_restart",
            fx::extended_12bit_ycbcr_420_restart_32x32_jpeg(),
        ),
        (
            "e12_cmyk_nonconstant",
            fx::extended_12bit_cmyk_nonconstant_8x8_jpeg(),
        ),
        (
            "e12_cmyk_restart",
            fx::extended_12bit_cmyk_restart_16x8_jpeg(),
        ),
        ("e12_cmyk_420", fx::extended_12bit_cmyk_16x16_420_jpeg()),
        (
            "e12_ycck_nonconstant",
            fx::extended_12bit_ycck_nonconstant_8x8_jpeg(),
        ),
        ("e12_ycck_422", fx::extended_12bit_ycck_16x8_422_jpeg()),
        ("p12_gray_8x8", fx::progressive_12bit_grayscale_8x8_jpeg()),
        ("p12_rgb_422", fx::progressive_12bit_rgb_422_32x8_jpeg()),
        (
            "p12_ycbcr_420",
            fx::progressive_12bit_ycbcr_420_32x32_jpeg(),
        ),
        ("p12_cmyk_422", fx::progressive_12bit_cmyk_16x8_422_jpeg()),
        ("p12_ycck_420", fx::progressive_12bit_ycck_16x16_420_jpeg()),
    ]
}

#[test]
fn explicit_metal_matches_cpu_for_grayscale_rgb_cmyk_ycck_and_12bit_fixtures() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for (name, jpeg) in fixture_cases() {
        let info = j2k_jpeg::Decoder::new(&jpeg)
            .expect("decoder")
            .info()
            .clone();
        let mut compared = 0;
        for request in requests_for(info.bit_depth, info.dimensions) {
            compared += usize::from(check_against_cpu(name, &jpeg, request, &mut failures));
        }
        assert!(compared > 0, "{name}: CPU decode rejected every request");
    }
    assert_no_failures(&failures);
}

#[test]
fn explicit_metal_rejects_requests_cpu_decode_rejects() {
    if !should_run_metal_runtime() {
        return;
    }
    let cases = [
        (j2k_test_support::progressive_8x8_jpeg(), PixelFormat::Rgb16),
        (
            j2k_test_support::extended_12bit_rgb_8x8_jpeg(),
            PixelFormat::Rgb8,
        ),
        (
            j2k_test_support::extended_12bit_ycbcr_8x8_jpeg(),
            PixelFormat::Gray16,
        ),
    ];
    for (jpeg, fmt) in cases {
        let mut decoder = Decoder::new(&jpeg).expect("decoder");
        let Err(err) =
            decoder.decode_request_to_device(MetalDecodeRequest::full(fmt, BackendRequest::Metal))
        else {
            panic!("{fmt:?}: CPU decode rejects this request, so Metal must too");
        };
        assert!(
            matches!(err, Error::UnsupportedMetalRequest { reason } if reason.starts_with("JPEG Metal")),
            "{fmt:?}: {err:?}"
        );
    }
}

#[test]
fn session_decode_uses_hybrid_metal_decode() {
    if !should_run_metal_runtime() {
        return;
    }
    let session = MetalBackendSession::system_default().expect("Metal session");
    for jpeg in [
        j2k_test_support::progressive_8x8_jpeg(),
        j2k_test_support::baseline_grayscale_jpeg(37, 29),
        j2k_test_support::cmyk_16x16_420_jpeg(),
    ] {
        let (expected, _) = j2k_jpeg::Decoder::new(&jpeg)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgba8))
            .expect("CPU decode");
        let mut decoder = Decoder::new(&jpeg).expect("decoder");
        let surface = decoder
            .decode_to_device_with_session(PixelFormat::Rgba8, &session)
            .expect("session decode");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert_eq!(
            surface.as_bytes().expect("surface bytes").as_ref(),
            expected
        );
    }
}

#[test]
fn truncated_progressive_stream_fails_like_cpu_decode() {
    if !should_run_metal_runtime() {
        return;
    }
    let jpeg = scaled_matrix_cases()
        .into_iter()
        .find(|case| case.coding == "progressive" && case.precision == 8 && case.width == 67)
        .expect("progressive matrix case")
        .jpeg;
    let truncated = &jpeg[..jpeg.len() * 2 / 3];
    let cpu = j2k_jpeg::Decoder::new(truncated)
        .and_then(|decoder| decoder.decode_request(DecodeRequest::full(PixelFormat::Rgb8)));
    let metal = Decoder::new(truncated).and_then(|mut decoder| {
        decoder.decode_request_to_device(MetalDecodeRequest::full(
            PixelFormat::Rgb8,
            BackendRequest::Metal,
        ))
    });
    assert_eq!(cpu.is_ok(), metal.is_ok(), "cpu {:?}", cpu.err());
}

/// Lossless samples up to 8 bits decode to 8-bit formats, wider ones to
/// 16-bit formats; the others are included to check Metal rejects them too.
fn lossless_formats(gray: bool, precision: u8) -> &'static [PixelFormat] {
    match (gray, precision <= 8) {
        (true, true) => &[PixelFormat::Gray8, PixelFormat::Rgb8],
        (true, false) => &[PixelFormat::Gray16, PixelFormat::Rgb16],
        (false, true) => &[PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8],
        (false, false) => &[PixelFormat::Rgb16, PixelFormat::Rgba16, PixelFormat::Gray16],
    }
}

fn lossless_requests(gray: bool, precision: u8, (w, h): (u32, u32)) -> Vec<DecodeRequest> {
    let regions = [
        None,
        Some(j2k_jpeg::Rect {
            x: w / 3,
            y: h / 4,
            w: (w - w / 3).div_ceil(2).max(1),
            h: (h - h / 4).div_ceil(2).max(1),
        }),
    ];
    let mut requests = Vec::new();
    for &fmt in lossless_formats(gray, precision) {
        for denominator in SCALED_MATRIX_DENOMINATORS {
            for region in regions {
                requests.push(DecodeRequest {
                    fmt,
                    region,
                    scale: downscale(denominator),
                });
            }
        }
    }
    requests
}

#[test]
fn explicit_metal_matches_cpu_across_the_lossless_matrix() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    let mut compared = 0_usize;
    for case in j2k_test_support::lossless_matrix_cases() {
        if j2k_jpeg::Decoder::new(case.jpeg).is_err() {
            // CPU decode cannot plan the stream; Metal must not either.
            assert!(Decoder::new(case.jpeg).is_err(), "{}", case.name);
            continue;
        }
        for request in lossless_requests(case.is_gray(), case.precision, (case.width, case.height))
        {
            compared += usize::from(check_against_cpu(
                case.name,
                case.jpeg,
                request,
                &mut failures,
            ));
        }
    }
    assert_no_failures(&failures);
    assert!(compared > 400, "compared {compared}");
}

#[test]
fn explicit_metal_matches_cpu_for_lossless_fixtures() {
    use j2k_test_support as fx;
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for predictor in 1..=7 {
        let cases: [(&str, Vec<u8>); 12] = [
            (
                "gray_3x3",
                fx::lossless_predictor_grayscale_3x3_jpeg(predictor),
            ),
            (
                "gray_restart_3x3",
                fx::lossless_restart_predictor_grayscale_3x3_jpeg(predictor),
            ),
            (
                "gray16_restart_3x3",
                fx::lossless_restart_predictor_grayscale_16bit_3x3_jpeg(predictor),
            ),
            (
                "rgb_restart_3x3",
                fx::lossless_restart_predictor_rgb_3x3_jpeg(predictor),
            ),
            (
                "ycbcr_restart_3x3",
                fx::lossless_restart_predictor_ycbcr_3x3_jpeg(predictor),
            ),
            (
                "ycbcr16_restart_3x3",
                fx::lossless_restart_predictor_ycbcr_16bit_3x3_jpeg(predictor),
            ),
            (
                "rgb_422_restart_4x2",
                fx::lossless_rgb_8bit_422_restart_4x2_jpeg(predictor),
            ),
            (
                "rgb16_420_restart_4x4",
                fx::lossless_rgb_16bit_420_restart_4x4_jpeg(predictor),
            ),
            (
                "ycbcr_422_restart_4x2",
                fx::lossless_ycbcr_8bit_422_restart_4x2_jpeg(predictor),
            ),
            (
                "ycbcr16_422_4x2",
                fx::lossless_ycbcr_16bit_422_4x2_jpeg(predictor),
            ),
            (
                "ycbcr_420_restart_4x4",
                fx::lossless_ycbcr_8bit_420_restart_4x4_jpeg(predictor),
            ),
            (
                "ycbcr16_420_restart_4x4",
                fx::lossless_ycbcr_16bit_420_restart_4x4_jpeg(predictor),
            ),
        ];
        for (name, jpeg) in &cases {
            let info = j2k_jpeg::Decoder::new(jpeg)
                .expect("decoder")
                .info()
                .clone();
            let gray = info.color_space == j2k_jpeg::ColorSpace::Grayscale;
            let mut compared = 0;
            for request in lossless_requests(gray, info.bit_depth, info.dimensions) {
                compared += usize::from(check_against_cpu(name, jpeg, request, &mut failures));
            }
            assert!(
                compared > 0,
                "{name} p{predictor}: CPU decode rejected every request"
            );
        }
    }
    assert_no_failures(&failures);
}

#[test]
fn malformed_lossless_streams_fail_like_cpu_decode() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut errors = 0;
    for case in j2k_test_support::lossless_matrix_cases()
        .into_iter()
        .filter(|case| case.width > 30 && j2k_jpeg::Decoder::new(case.jpeg).is_ok())
    {
        let fmt = lossless_formats(case.is_gray(), case.precision)[0];
        // Truncate mid-scan, and flip a byte inside the entropy data so
        // reconstruction can leave the sample range.
        let mut corrupted = case.jpeg.to_vec();
        let middle = corrupted.len() * 3 / 4;
        corrupted[middle] ^= 0x5A;
        if corrupted[middle] == 0xFF {
            corrupted[middle] = 0x7F;
        }
        for (label, jpeg) in [
            ("truncated", &case.jpeg[..case.jpeg.len() * 2 / 3]),
            ("corrupted", corrupted.as_slice()),
        ] {
            let cpu = j2k_jpeg::Decoder::new(jpeg)
                .and_then(|decoder| decoder.decode_request(DecodeRequest::full(fmt)));
            let metal = Decoder::new(jpeg).and_then(|mut decoder| {
                decoder
                    .decode_request_to_device(MetalDecodeRequest::full(fmt, BackendRequest::Metal))
            });
            match (cpu, metal) {
                (Ok((expected, _)), Ok(surface)) => assert_eq!(
                    surface.as_bytes().expect("surface bytes").as_ref(),
                    expected,
                    "{} {label}",
                    case.name
                ),
                (Err(cpu), Err(Error::Decode(metal))) => {
                    assert_eq!(metal, cpu, "{} {label}", case.name);
                    errors += 1;
                }
                (cpu, metal) => panic!(
                    "{} {label}: cpu ok {}, metal {:?}",
                    case.name,
                    cpu.is_ok(),
                    metal.err()
                ),
            }
        }
    }
    // Every truncated stream fails; corruption fails most of the rest.
    assert!(errors > 30, "only {errors} malformed streams failed");
}

/// A batch of unrelated JPEGs (fast-packet tiles, grayscale, progressive,
/// CMYK, 12-bit, lossless) decodes entirely on Metal under explicit Metal
/// requests: fast tiles through the packet kernels and everything else
/// through hybrid decode, in submission order.
#[test]
fn heterogeneous_explicit_metal_batch_decodes_every_item_on_metal() {
    use j2k_test_support as fx;
    if !should_run_metal_runtime() {
        return;
    }
    let lossless = j2k_test_support::lossless_matrix_cases()
        .into_iter()
        .find(|case| case.layout == "ycbcr420" && case.precision == 16)
        .expect("16-bit lossless 4:2:0 case")
        .jpeg
        .to_vec();
    let inputs: Vec<(Vec<u8>, PixelFormat)> = vec![
        (
            include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg").to_vec(),
            PixelFormat::Rgb8,
        ),
        (fx::baseline_grayscale_jpeg(37, 29), PixelFormat::Gray8),
        (fx::progressive_8x8_jpeg(), PixelFormat::Rgba8),
        (
            include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg").to_vec(),
            PixelFormat::Rgb8,
        ),
        (fx::cmyk_16x16_420_jpeg(), PixelFormat::Rgb8),
        (
            fx::extended_12bit_ycbcr_420_32x32_jpeg(),
            PixelFormat::Rgb16,
        ),
        (lossless, PixelFormat::Rgba16),
    ];
    let mut batch = j2k_jpeg_metal::JpegTileBatch::with_capacity(inputs.len());
    for (jpeg, fmt) in &inputs {
        batch
            .push_tile_request(jpeg, MetalDecodeRequest::full(*fmt, BackendRequest::Metal))
            .expect("push tile");
    }
    let surfaces = batch.decode_all().expect("heterogeneous Metal batch");
    assert_eq!(surfaces.len(), inputs.len());
    for (surface, (jpeg, fmt)) in surfaces.iter().zip(&inputs) {
        let (expected, _) = j2k_jpeg::Decoder::new(jpeg)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(*fmt))
            .expect("CPU decode");
        assert_eq!(
            surface.residency(),
            SurfaceResidency::MetalResidentDecode,
            "{fmt:?}"
        );
        assert_eq!(
            surface.as_bytes().expect("surface bytes").as_ref(),
            expected,
            "{fmt:?}"
        );
    }
}

/// Region edges on and beside MCU boundaries for every layout and coding:
/// windowed entropy decode must reproduce CPU region decode exactly.
#[test]
fn hybrid_region_edge_sweep_matches_cpu_region_decode() {
    if !should_run_metal_runtime() {
        return;
    }
    let xs = [0_u32, 7, 9, 16, 17, 40];
    let ys = [0_u32, 8, 9, 16, 30];
    let mut failures = Vec::new();
    let mut compared = 0;
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| case.width == 67 || case.width == 45)
    {
        let fmt = formats_for(case.precision)[1];
        for (&x0, &x1) in xs.iter().zip(xs.iter().skip(1)) {
            for (&y0, &y1) in ys.iter().zip(ys.iter().skip(2)) {
                let (x1, y1) = (x1.min(case.width), y1.min(case.height));
                if x0 >= x1 || y0 >= y1 {
                    continue;
                }
                for denominator in SCALED_MATRIX_DENOMINATORS {
                    let request = DecodeRequest {
                        fmt,
                        region: Some(j2k_jpeg::Rect {
                            x: x0,
                            y: y0,
                            w: x1 - x0,
                            h: y1 - y0,
                        }),
                        scale: downscale(denominator),
                    };
                    compared += usize::from(check_against_cpu(
                        case.name,
                        case.jpeg,
                        request,
                        &mut failures,
                    ));
                }
            }
        }
    }
    assert_no_failures(&failures);
    assert!(compared > 500, "compared {compared}");
}

/// Regions of a large image whose MCUs the windowed decode skips, entering
/// the scan at restart markers when the stream has them.
#[test]
fn hybrid_windowed_regions_of_large_images_match_cpu() {
    use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
    if !should_run_metal_runtime() {
        return;
    }
    let (width, height) = (1024_u16, 768_u16);
    let rgb = j2k_test_support::gpu_bench_rgb8(u32::from(width), u32::from(height));
    let gray: Vec<u8> = rgb.as_chunks::<3>().0.iter().map(|px| px[1]).collect();
    let mut failures = Vec::new();
    for (name, pixels, color, sampling, restart) in [
        ("gray", &gray, ColorType::Luma, SamplingFactor::F_1_1, 0),
        (
            "gray_restart",
            &gray,
            ColorType::Luma,
            SamplingFactor::F_1_1,
            5,
        ),
        ("ycbcr440", &rgb, ColorType::Rgb, SamplingFactor::F_1_2, 0),
        (
            "ycbcr440_restart",
            &rgb,
            ColorType::Rgb,
            SamplingFactor::F_1_2,
            7,
        ),
    ] {
        let mut jpeg = Vec::new();
        let mut encoder = Encoder::new(&mut jpeg, 88);
        encoder.set_sampling_factor(sampling);
        if restart != 0 {
            encoder.set_restart_interval(restart);
        }
        encoder
            .encode(pixels, width, height, color)
            .expect("encode large fixture");
        let fmt = if color == ColorType::Luma {
            PixelFormat::Gray8
        } else {
            PixelFormat::Rgba8
        };
        for (roi, denominator) in [
            ((517, 301, 129, 77), 1),
            ((0, 700, 1024, 68), 1),
            ((900, 0, 124, 768), 2),
            ((333, 333, 333, 333), 4),
            ((1, 1, 1, 1), 1),
        ] {
            let request = DecodeRequest {
                fmt,
                region: Some(j2k_jpeg::Rect {
                    x: roi.0,
                    y: roi.1,
                    w: roi.2,
                    h: roi.3,
                }),
                scale: downscale(denominator),
            };
            assert!(
                check_against_cpu(name, &jpeg, request, &mut failures),
                "{name}: CPU decode rejected {request:?}"
            );
        }
    }
    assert_no_failures(&failures);
}

/// DCMTK-written DICOM codestreams decode on Metal exactly as DCMTK and CPU
/// decode them.
#[test]
fn explicit_metal_matches_dcmtk_for_dicom_codestreams() {
    if !should_run_metal_runtime() {
        return;
    }
    let mut failures = Vec::new();
    for case in j2k_test_support::dicom_matrix_cases() {
        let wide = case.precision > 8;
        let fmt = match (case.channels, wide) {
            (1, false) => PixelFormat::Gray8,
            (1, true) => PixelFormat::Gray16,
            (_, false) => PixelFormat::Rgb8,
            (_, true) => PixelFormat::Rgb16,
        };
        let request = DecodeRequest::full(fmt);
        check_against_cpu(case.name, case.jpeg, request, &mut failures);
        let surface = Decoder::new(case.jpeg)
            .expect("Metal decoder")
            .decode_request_to_device(metal_request(request))
            .unwrap_or_else(|err| panic!("{}: Metal decode: {err}", case.name));
        let bytes = surface.as_bytes().expect("surface bytes");
        let samples: Vec<u16> = if wide {
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| u16::from_le_bytes(*pair))
                .collect()
        } else {
            bytes.iter().map(|&sample| u16::from(sample)).collect()
        };
        if samples != case.reference_samples() {
            failures.push(format!(
                "{} ({}): Metal differs from DCMTK",
                case.name, case.options
            ));
        }
    }
    assert_no_failures(&failures);
}
