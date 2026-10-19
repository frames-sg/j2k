// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host reference for the lossless device decode contract: reconstruct each
//! sample from the device differences with the per-sample recurrence device
//! kernels evaluate, then pack each output pixel. Matching CPU decode pins
//! the contract.

use super::*;
use crate::color::ycbcr;
use crate::{DecodeRequest, Rect};
use alloc::vec;
use alloc::vec::Vec;
use j2k_core::Downscale;

/// Reconstruct one component plane from its differences.
fn reconstruct(component: JpegDeviceLosslessComponent, diffs: &[u16]) -> Option<Vec<u16>> {
    let (width, height) = (component.width as usize, component.height as usize);
    let mut samples = vec![0u16; width * height];
    for row in 0..height {
        let first_row =
            row == 0 || (component.restart_rows != 0 && row % component.restart_rows as usize == 0);
        for col in 0..width {
            let diff = diffs[row * width + col];
            let at = |r: usize, c: usize| samples[r * width + c];
            let prediction = if first_row {
                if col == 0 {
                    component.initial_predictor
                } else {
                    at(row, col - 1)
                }
            } else if col == 0 {
                at(row - 1, 0)
            } else {
                let (ra, rb, rc) = (
                    i32::from(at(row, col - 1)),
                    i32::from(at(row - 1, col)),
                    i32::from(at(row - 1, col - 1)),
                );
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "predictions are reduced modulo 2^16"
                )]
                let predicted = match component.predictor {
                    1 => ra,
                    2 => rb,
                    3 => rc,
                    4 => ra + rb - rc,
                    5 => ra + ((rb - rc) >> 1),
                    6 => rb + ((ra - rc) >> 1),
                    _ => (ra + rb) >> 1,
                } as u16;
                predicted
            };
            let sample = prediction.wrapping_add(diff);
            if component.sample_limit != u16::MAX && sample > component.sample_limit {
                return None;
            }
            samples[row * width + col] = sample;
        }
    }
    Some(samples)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "reference samples are bounded by the output sample width"
)]
fn ycbcr_at_precision(rgb: [u32; 3], precision: u8) -> [u32; 3] {
    let (r, g, b) = match precision {
        8 => {
            let (r, g, b) = ycbcr::ycbcr_to_rgb(rgb[0] as u8, rgb[1] as u8, rgb[2] as u8);
            (r.into(), g.into(), b.into())
        }
        16 => ycbcr::ycbcr16_to_rgb16(rgb[0] as u16, rgb[1] as u16, rgb[2] as u16),
        _ => ycbcr::ycbcr_to_rgb_with_precision(
            rgb[0] as u16,
            rgb[1] as u16,
            rgb[2] as u16,
            precision,
        ),
    };
    [r, g, b].map(u32::from)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "reference samples are bounded by the output sample width"
)]
fn render_reference(decoder: &Decoder<'_>, request: DecodeRequest) -> Option<Vec<u8>> {
    let layout = device_lossless_layout(decoder, request.fmt).ok()?;
    let mut diffs = vec![0u16; layout.total_samples()];
    decode_device_lossless_differences_into(decoder, request.fmt, &mut diffs).ok()?;
    let bases = layout.sample_bases();
    let mut planes = Vec::new();
    for (index, &component) in layout.components().iter().enumerate() {
        let len = component.width as usize * component.height as usize;
        let samples = reconstruct(component, &diffs[bases[index]..bases[index] + len])?;
        planes.push(
            samples
                .into_iter()
                .map(|s| u32::from(s << component.point_transform))
                .collect::<Vec<u32>>(),
        );
    }
    let (width, height) = decoder.info().dimensions;
    let roi = request.region.unwrap_or(Rect::full((width, height)));
    let denom = request.scale.denominator();
    let (x0, y0) = (roi.x / denom, roi.y / denom);
    let (x1, y1) = (
        (roi.x + roi.w).div_ceil(denom),
        (roi.y + roi.h).div_ceil(denom),
    );
    let wide = layout.precision > 8;
    let mut out = Vec::new();
    let mut push = |value: u32| {
        if wide {
            out.extend_from_slice(&(value as u16).to_le_bytes());
        } else {
            out.push(value as u8);
        }
    };
    let components = layout.components();
    // Each component's sample covering pixel (x, y): libjpeg-turbo's
    // lossless upsampler replicates by the sampling ratio.
    let sample_at = |c: usize, x: usize, y: usize| {
        let component = components[c];
        let (cx, cy) = (
            x / usize::from(component.h_expand),
            y / usize::from(component.v_expand),
        );
        planes[c][cy * component.width as usize + cx]
    };
    for oy in y0..y1 {
        let sy = (oy * denom).min(height - 1) as usize;
        for ox in x0..x1 {
            let sx = (ox * denom).min(width - 1) as usize;
            if components.len() == 1 {
                push(sample_at(0, sx, sy));
                continue;
            }
            let rgb = [
                sample_at(0, sx, sy),
                sample_at(1, sx, sy),
                sample_at(2, sx, sy),
            ];
            let rgb = if layout.color_space == ColorSpace::YCbCr {
                ycbcr_at_precision(rgb, layout.precision)
            } else {
                rgb
            };
            rgb.into_iter().for_each(&mut push);
            if matches!(request.fmt, PixelFormat::Rgba8 | PixelFormat::Rgba16) {
                push(if wide { 0xFFFF } else { 0xFF });
            }
        }
    }
    Some(out)
}

#[test]
fn device_lossless_reference_matches_cpu_on_lossless_matrix() {
    let scales = [
        Downscale::None,
        Downscale::Half,
        Downscale::Quarter,
        Downscale::Eighth,
    ];
    let (mut compared, mut rejected) = (0, 0);
    let mut unparsed = Vec::new();
    for case in j2k_test_support::lossless_matrix_cases() {
        let Ok(decoder) = Decoder::new(case.jpeg) else {
            // The planner rejects the shape outright, so no backend decodes it.
            unparsed.push(case.name);
            continue;
        };
        let formats: &[PixelFormat] = match (case.is_gray(), case.precision > 8) {
            (true, false) => &[PixelFormat::Gray8],
            (true, true) => &[PixelFormat::Gray16],
            (false, false) => &[PixelFormat::Rgb8, PixelFormat::Rgba8],
            (false, true) => &[PixelFormat::Rgb16, PixelFormat::Rgba16],
        };
        let (w, h) = (case.width, case.height);
        let regions = [
            None,
            Some(Rect {
                x: w / 3,
                y: h / 4,
                w: (w - w / 3).div_ceil(2),
                h: (h - h / 4).div_ceil(2),
            }),
        ];
        for &fmt in formats {
            for scale in scales {
                for region in regions {
                    let request = DecodeRequest { fmt, region, scale };
                    let cpu = decoder.decode_request(request);
                    let device = render_reference(&decoder, request);
                    match (cpu, device) {
                        (Ok((expected, _)), Some(actual)) => {
                            assert_eq!(actual, expected, "{} {request:?}", case.name);
                            compared += 1;
                        }
                        (Err(_), None) => rejected += 1,
                        (cpu, device) => panic!(
                            "{} {request:?}: cpu {:?}, device {}",
                            case.name,
                            cpu.map(|_| ()),
                            device.is_some()
                        ),
                    }
                }
            }
        }
    }
    // CPU decode plans every case: any precision and point transform,
    // split scans, and odd subsampled sizes.
    assert!(compared > 300, "compared {compared}");
    assert_eq!(rejected, 0);
    assert!(unparsed.is_empty(), "unparsed {unparsed:?}");
}
