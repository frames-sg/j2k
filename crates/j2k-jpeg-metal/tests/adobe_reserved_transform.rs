// SPDX-License-Identifier: MIT OR Apache-2.0

//! A reserved Adobe APP14 transform byte through explicit Metal decode.
//!
//! A strict decoder refuses it before any Metal work. A decoder built from a
//! lenient view reads it as CPU decode does, YCbCr for three components and
//! YCCK for four (libjpeg-turbo's reading), and returns the same bytes in
//! Metal-resident storage.
#![cfg(target_os = "macos")]

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect, SurfaceResidency};
use j2k_jpeg::{DecodeOptions, DecodeRequest, JpegView};
use j2k_jpeg_metal::{Decoder, MetalDecodeRequest};
use j2k_test_support::{scaled_matrix_cases, with_adobe_transform, SCALED_MATRIX_DENOMINATORS};

fn downscale(denominator: u32) -> Downscale {
    match denominator {
        1 => Downscale::None,
        2 => Downscale::Half,
        4 => Downscale::Quarter,
        8 => Downscale::Eighth,
        _ => unreachable!("matrix denominators are 1, 2, 4, 8"),
    }
}

#[test]
fn explicit_metal_reads_reserved_transforms_as_lenient_cpu_decode() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let lenient = DecodeOptions::default().with_strict(false);
    let mut failures = Vec::new();
    let mut compared = 0_usize;
    for case in scaled_matrix_cases()
        .into_iter()
        .filter(|case| !case.is_gray())
    {
        let jpeg = with_adobe_transform(case.jpeg, 7);
        if Decoder::new(&jpeg).is_ok() {
            failures.push(format!("{}: strict Metal decoder accepted it", case.name));
        }
        let formats = if case.precision > 8 {
            [PixelFormat::Rgb16, PixelFormat::Rgba16]
        } else {
            [PixelFormat::Rgb8, PixelFormat::Rgba8]
        };
        let roi = j2k_jpeg::Rect {
            x: case.width / 3,
            y: case.height / 4,
            w: (case.width / 2).max(1),
            h: (case.height / 2).max(1),
        };
        for fmt in formats {
            for denominator in SCALED_MATRIX_DENOMINATORS {
                let scale = downscale(denominator);
                for region in [None, Some(roi)] {
                    let request = DecodeRequest { fmt, region, scale };
                    let label = format!("{} {request:?}", case.name);
                    let (expected, outcome) = j2k_jpeg::Decoder::from_view(
                        JpegView::parse_with_options(&jpeg, lenient).expect("view"),
                    )
                    .and_then(|decoder| decoder.decode_request(request))
                    .unwrap_or_else(|err| panic!("{label}: lenient CPU decode: {err}"));
                    assert!(outcome.recovered, "{label}");
                    let metal_request = match region {
                        None => MetalDecodeRequest::scaled(fmt, scale, BackendRequest::Metal),
                        Some(roi) => MetalDecodeRequest::region_scaled(
                            fmt,
                            Rect {
                                x: roi.x,
                                y: roi.y,
                                w: roi.w,
                                h: roi.h,
                            },
                            scale,
                            BackendRequest::Metal,
                        ),
                    };
                    let decoded = Decoder::from_view(
                        JpegView::parse_with_options(&jpeg, lenient).expect("view"),
                    )
                    .and_then(|mut decoder| decoder.decode_request_to_device(metal_request));
                    match decoded {
                        Ok(surface) => {
                            if surface.residency() != SurfaceResidency::MetalResidentDecode {
                                failures.push(format!("{label}: {:?}", surface.residency()));
                            }
                            let actual = surface.as_bytes().expect("surface bytes");
                            if actual[..] != expected[..] {
                                failures.push(format!("{label}: Metal bytes differ from CPU"));
                            }
                            compared += 1;
                        }
                        Err(err) => failures.push(format!("{label}: Metal failed: {err}")),
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} reserved-transform Metal decodes differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(compared > 3000, "compared {compared}");
}
