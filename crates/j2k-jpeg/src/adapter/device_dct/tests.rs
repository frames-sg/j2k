// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host reference for the hybrid device decode contract.
//!
//! `render_reference` rebuilds decoded pixels from the device DCT blocks and
//! per-component geometry with per-output-pixel formulas, the same shape GPU
//! kernels evaluate. Matching the CPU decoder here pins the contract device
//! adapters implement.

use super::*;
use crate::color::{cmyk, ycbcr};
use crate::idct::{downscale, scalar};
use crate::info::ColorSpace;
use crate::{DecodeRequest, Rect};
use alloc::vec;
use alloc::vec::Vec;
use j2k_core::PixelFormat;

struct ComponentPlane {
    geometry: JpegDeviceComponentGeometry,
    stride: usize,
    samples: Vec<u16>,
}

impl ComponentPlane {
    fn at(&self, x: u32, y: u32) -> u32 {
        u32::from(self.samples[y as usize * self.stride + x as usize])
    }
}

fn idct_block(precision: u8, size: u32, block: &[i16; 64], out: &mut [u16; 64]) {
    match (precision, size) {
        (8, 8) => {
            let mut pixels = [0u8; 64];
            scalar::idct_islow(block, &mut pixels);
            out.iter_mut()
                .zip(pixels)
                .for_each(|(o, p)| *o = u16::from(p));
        }
        (8, 4) => {
            let mut pixels = [0u8; 16];
            downscale::idct_islow_4x4(block, &mut pixels);
            out.iter_mut()
                .zip(pixels)
                .for_each(|(o, p)| *o = u16::from(p));
        }
        (8, 2) => {
            let mut pixels = [0u8; 4];
            downscale::idct_islow_2x2(block, &mut pixels);
            out.iter_mut()
                .zip(pixels)
                .for_each(|(o, p)| *o = u16::from(p));
        }
        (8, _) => out[0] = u16::from(downscale::idct_islow_1x1(block)),
        (_, 8) => scalar::idct_islow_12bit(block, out),
        (_, 4) => {
            let mut pixels = [0u16; 16];
            downscale::idct_islow_12bit_4x4(block, &mut pixels);
            out[..16].copy_from_slice(&pixels);
        }
        (_, 2) => {
            let mut pixels = [0u16; 4];
            downscale::idct_islow_12bit_2x2(block, &mut pixels);
            out[..4].copy_from_slice(&pixels);
        }
        (_, _) => out[0] = downscale::idct_islow_12bit_1x1(block),
    }
}

fn component_planes(decoder: &Decoder<'_>, scale: Downscale) -> Vec<ComponentPlane> {
    let layout = device_dct_layout(decoder).expect("device layout");
    let mut blocks = vec![[0i16; 64]; layout.total_blocks().unwrap()];
    decode_device_dct_blocks_into(decoder, &mut blocks).expect("device entropy decode");
    let geometry = device_component_geometry(decoder.info(), scale).unwrap();
    let bases = layout.block_bases().unwrap();
    let mut planes = Vec::new();
    for (index, (grid, geometry)) in layout.components().iter().zip(geometry).enumerate() {
        let size = geometry.idct_size;
        let stride = (grid.block_cols * size) as usize;
        let mut samples = vec![0u16; stride * (grid.block_rows * size) as usize];
        let mut pixels = [0u16; 64];
        for by in 0..grid.block_rows {
            for bx in 0..grid.block_cols {
                let block = &blocks[bases[index] + (by * grid.block_cols + bx) as usize];
                idct_block(layout.precision, size, block, &mut pixels);
                for row in 0..size {
                    for col in 0..size {
                        let dst =
                            ((by * size + row) as usize) * stride + (bx * size + col) as usize;
                        samples[dst] = pixels[(row * size + col) as usize];
                    }
                }
            }
        }
        planes.push(ComponentPlane {
            geometry,
            stride,
            samples,
        });
    }
    planes
}

/// libjpeg-turbo `h2v1_fancy_upsample` at one output column.
fn h2v1(plane: &ComponentPlane, row: u32, x: u32) -> u32 {
    let n = plane.geometry.width;
    let s = (x / 2).min(n - 1);
    let near = 3 * plane.at(s, row);
    if x.is_multiple_of(2) {
        if s == 0 {
            plane.at(0, row)
        } else {
            (near + plane.at(s - 1, row) + 1) >> 2
        }
    } else if s == n - 1 {
        plane.at(n - 1, row)
    } else {
        (near + plane.at(s + 1, row) + 2) >> 2
    }
}

/// libjpeg-turbo `h2v2_fancy_upsample` at one output column, given the
/// current and nearest neighbouring component rows.
fn h2v2(plane: &ComponentPlane, curr: u32, near: u32, x: u32) -> u32 {
    let n = plane.geometry.width;
    let colsum = |i: u32| 3 * plane.at(i, curr) + plane.at(i, near);
    if n == 1 {
        return (4 * colsum(0) + 8) >> 4;
    }
    let s = (x / 2).min(n - 1);
    let this = colsum(s);
    if x == 0 {
        (4 * this + 8) >> 4
    } else if x == 2 * n - 1 {
        (4 * this + 7) >> 4
    } else if x.is_multiple_of(2) {
        (3 * this + colsum(s - 1) + 8) >> 4
    } else {
        (3 * this + colsum(s + 1) + 7) >> 4
    }
}

fn component_sample(plane: &ComponentPlane, x: u32, y: u32) -> u32 {
    let geometry = plane.geometry;
    let last_row = geometry.height - 1;
    match geometry.upsample {
        JpegDeviceUpsample::None => plane.at(x, y),
        JpegDeviceUpsample::FancyH2V1 => h2v1(plane, y.min(last_row), x),
        JpegDeviceUpsample::FancyH2V2 | JpegDeviceUpsample::FancyH1V2 => {
            let curr = (y / 2).min(last_row);
            let bottom = y % 2 == 1;
            let near = if bottom {
                (curr + 1).min(last_row)
            } else {
                curr.saturating_sub(1)
            };
            if geometry.upsample == JpegDeviceUpsample::FancyH2V2 {
                h2v2(plane, curr, near, x)
            } else {
                (3 * plane.at(x, curr) + plane.at(x, near) + if bottom { 2 } else { 1 }) >> 2
            }
        }
        JpegDeviceUpsample::Replicate => plane.at(
            (x / geometry.h_ratio).min(geometry.width - 1),
            (y / geometry.v_ratio).min(last_row),
        ),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "reference samples are bounded by the JPEG sample precision"
)]
fn rgb_of(color_space: ColorSpace, precision: u8, c: [u32; 4]) -> [u32; 3] {
    let narrow8 = |v: u32| v as u8;
    let narrow12 = |v: u32| v as u16;
    match (color_space, precision) {
        (ColorSpace::Grayscale, _) => [c[0]; 3],
        (ColorSpace::Rgb, _) => [c[0], c[1], c[2]],
        (ColorSpace::YCbCr, 8) => {
            let (r, g, b) = ycbcr::ycbcr_to_rgb(narrow8(c[0]), narrow8(c[1]), narrow8(c[2]));
            [r, g, b].map(u32::from)
        }
        (ColorSpace::YCbCr, _) => {
            let (r, g, b) = ycbcr::ycbcr12_to_rgb16(narrow12(c[0]), narrow12(c[1]), narrow12(c[2]));
            [r, g, b].map(u32::from)
        }
        (ColorSpace::Cmyk, 8) => {
            let (r, g, b) = cmyk::inverted_cmyk_to_rgb(
                narrow8(c[0]),
                narrow8(c[1]),
                narrow8(c[2]),
                narrow8(c[3]),
            );
            [r, g, b].map(u32::from)
        }
        (ColorSpace::Cmyk, _) => {
            let (r, g, b) = cmyk::inverted_cmyk12_to_rgb16(
                narrow12(c[0]),
                narrow12(c[1]),
                narrow12(c[2]),
                narrow12(c[3]),
            );
            [r, g, b].map(u32::from)
        }
        (ColorSpace::Ycck, 8) => {
            let (r, g, b) =
                cmyk::ycck_to_rgb(narrow8(c[0]), narrow8(c[1]), narrow8(c[2]), narrow8(c[3]));
            [r, g, b].map(u32::from)
        }
        (ColorSpace::Ycck, _) => {
            let (r, g, b) = cmyk::ycck12_to_rgb16(
                narrow12(c[0]),
                narrow12(c[1]),
                narrow12(c[2]),
                narrow12(c[3]),
            );
            [r, g, b].map(u32::from)
        }
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "reference samples are bounded by the output sample width"
)]
fn render_reference(decoder: &Decoder<'_>, request: DecodeRequest) -> Vec<u8> {
    let info = decoder.info();
    let planes = component_planes(decoder, request.scale);
    let roi = request.region.unwrap_or(Rect::full(info.dimensions));
    let denom = request.scale.denominator();
    let (x0, y0) = (roi.x / denom, roi.y / denom);
    let rect = Rect {
        x: x0,
        y: y0,
        w: (roi.x + roi.w).div_ceil(denom) - x0,
        h: (roi.y + roi.h).div_ceil(denom) - y0,
    };
    let fmt = request.fmt;
    let wide = matches!(
        fmt,
        PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16
    );
    let alpha = if wide { 0xFFFF } else { 0xFF };
    let mut out = Vec::new();
    let mut push = |value: u32| {
        if wide {
            out.extend_from_slice(&(value as u16).to_le_bytes());
        } else {
            out.push(value as u8);
        }
    };
    for oy in 0..rect.h {
        for ox in 0..rect.w {
            let (x, y) = (rect.x + ox, rect.y + oy);
            let mut c = [0u32; 4];
            for (slot, plane) in c.iter_mut().zip(&planes) {
                *slot = component_sample(plane, x, y);
            }
            let rgb = rgb_of(info.color_space, info.bit_depth, c);
            match fmt {
                PixelFormat::Gray8 | PixelFormat::Gray16 => push(match info.color_space {
                    ColorSpace::Grayscale | ColorSpace::YCbCr => c[0],
                    _ => (77 * rgb[0] + 150 * rgb[1] + 29 * rgb[2] + 128) >> 8,
                }),
                PixelFormat::Rgb8 | PixelFormat::Rgb16 => rgb.into_iter().for_each(&mut push),
                _ => {
                    rgb.into_iter().for_each(&mut push);
                    push(alpha);
                }
            }
        }
    }
    out
}

/// Returns whether CPU decode accepted the request and was compared.
fn assert_reference_matches_cpu(name: &str, jpeg: &[u8], request: DecodeRequest) -> bool {
    let decoder = Decoder::new(jpeg).expect("decoder");
    // The device contract covers exactly the requests CPU decode accepts.
    let Ok((expected, _)) = decoder.decode_request(request) else {
        return false;
    };
    let actual = render_reference(&decoder, request);
    if let Some(index) = actual.iter().zip(&expected).position(|(a, e)| a != e) {
        panic!(
            "{name} {request:?}: first mismatch at byte {index}: device {} cpu {}",
            actual[index], expected[index]
        );
    }
    assert_eq!(actual.len(), expected.len(), "{name} {request:?} length");
    true
}

const SCALES: [Downscale; 4] = [
    Downscale::None,
    Downscale::Half,
    Downscale::Quarter,
    Downscale::Eighth,
];

fn requests_for(precision: u8, dimensions: (u32, u32)) -> Vec<DecodeRequest> {
    let formats: &[PixelFormat] = if precision == 8 {
        &[PixelFormat::Gray8, PixelFormat::Rgb8, PixelFormat::Rgba8]
    } else {
        &[PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Rgba16]
    };
    let (w, h) = dimensions;
    let region = Rect {
        x: w / 3,
        y: h / 4,
        w: (w - w / 3).div_ceil(2).max(1),
        h: (h - h / 4).div_ceil(2).max(1),
    };
    let mut requests = Vec::new();
    for &fmt in formats {
        for scale in SCALES {
            requests.push(DecodeRequest {
                fmt,
                region: None,
                scale,
            });
            requests.push(DecodeRequest {
                fmt,
                region: Some(region),
                scale,
            });
        }
    }
    requests
}

#[test]
fn device_reference_matches_cpu_on_scaled_matrix() {
    let (mut compared, mut rejected) = (0, 0);
    for case in j2k_test_support::scaled_matrix_cases() {
        for request in requests_for(case.precision, (case.width, case.height)) {
            if assert_reference_matches_cpu(case.name, case.jpeg, request) {
                compared += 1;
                continue;
            }
            // Only Gray16 requests of 12-bit color images are rejected by CPU decode.
            assert!(
                request.fmt == PixelFormat::Gray16 && case.precision > 8 && !case.is_gray(),
                "{} {request:?}: CPU decode rejected the request",
                case.name
            );
            rejected += 1;
        }
    }
    assert!(
        compared > 6000 && rejected > 0,
        "compared {compared}, rejected {rejected}"
    );
}

#[test]
fn device_reference_matches_cpu_on_four_component_and_rgb_fixtures() {
    use j2k_test_support as fx;
    let cases: [(&str, Vec<u8>); 32] = [
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
        ("rgb_app14_8x8", fx::rgb_app14_8x8_jpeg()),
        ("progressive_8x8", fx::progressive_8x8_jpeg()),
        ("e12_cmyk_8x8", fx::extended_12bit_cmyk_8x8_jpeg()),
        (
            "e12_cmyk_nonconstant",
            fx::extended_12bit_cmyk_nonconstant_8x8_jpeg(),
        ),
        (
            "e12_cmyk_restart",
            fx::extended_12bit_cmyk_restart_16x8_jpeg(),
        ),
        ("e12_cmyk_422", fx::extended_12bit_cmyk_16x8_422_jpeg()),
        ("e12_cmyk_420", fx::extended_12bit_cmyk_16x16_420_jpeg()),
        ("e12_ycck_8x8", fx::extended_12bit_ycck_8x8_jpeg()),
        (
            "e12_ycck_nonconstant",
            fx::extended_12bit_ycck_nonconstant_8x8_jpeg(),
        ),
        ("e12_ycck_422", fx::extended_12bit_ycck_16x8_422_jpeg()),
        ("e12_ycck_420", fx::extended_12bit_ycck_16x16_420_jpeg()),
        (
            "p12_cmyk_nonconstant",
            fx::progressive_12bit_cmyk_nonconstant_8x8_jpeg(),
        ),
        ("p12_cmyk_422", fx::progressive_12bit_cmyk_16x8_422_jpeg()),
        ("p12_cmyk_420", fx::progressive_12bit_cmyk_16x16_420_jpeg()),
        (
            "p12_ycck_nonconstant",
            fx::progressive_12bit_ycck_nonconstant_8x8_jpeg(),
        ),
        ("p12_ycck_422", fx::progressive_12bit_ycck_16x8_422_jpeg()),
        ("p12_ycck_420", fx::progressive_12bit_ycck_16x16_420_jpeg()),
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
            "e12_gray_restart",
            fx::extended_12bit_grayscale_restart_16x8_jpeg(),
        ),
        ("p12_rgb_422", fx::progressive_12bit_rgb_422_32x8_jpeg()),
        (
            "p12_ycbcr_420",
            fx::progressive_12bit_ycbcr_420_32x32_jpeg(),
        ),
        ("p12_gray", fx::progressive_12bit_grayscale_8x8_jpeg()),
    ];
    let mut compared = 0;
    for (name, jpeg) in &cases {
        let decoder = Decoder::new(jpeg).expect("decoder");
        let info = decoder.info();
        let mut case_compared = 0;
        for request in requests_for(info.bit_depth, info.dimensions) {
            case_compared += usize::from(assert_reference_matches_cpu(name, jpeg, request));
        }
        assert!(
            case_compared > 0,
            "{name}: CPU decode rejected every request"
        );
        compared += case_compared;
    }
    assert!(compared > 32 * 8, "compared {compared}");
}
