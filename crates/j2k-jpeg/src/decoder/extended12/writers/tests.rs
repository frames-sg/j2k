// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::planes::Extended12Plane;
use super::*;

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn samples(seed: u32, len: usize) -> Vec<u16> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| u16::try_from(xorshift(&mut state) % 4096).expect("12-bit sample"))
        .collect()
}

#[test]
fn block_output_span_matches_brute_force() {
    for extent in 1..=20u32 {
        for denom in [1u32, 2, 4, 8] {
            let outputs = extent.div_ceil(denom);
            for start in 0..outputs {
                for len in 0..=outputs - start {
                    for lo in (0..extent).step_by(8) {
                        let hi = (lo + 8).min(extent);
                        let expected: Vec<u32> = (start..start + len)
                            .filter(|&o| {
                                let source = o.saturating_mul(denom).min(extent - 1);
                                (lo..hi).contains(&source)
                            })
                            .collect();
                        let actual: Vec<u32> =
                            block_output_span(start, len, denom, extent, lo, hi).collect();
                        assert_eq!(
                            actual, expected,
                            "extent {extent} denom {denom} start {start} len {len} block {lo}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fancy_pairs_match_the_per_pixel_upsamplers() {
    for len in 1..=40usize {
        let curr = samples(u32::try_from(len).expect("len"), len);
        let near = samples(u32::try_from(len * 7).expect("len"), len);
        for first in 0..len {
            let pairs = (len - first).min(FANCY_CHUNK / 2);
            let mut h2v1 = vec![0u16; 2 * pairs];
            fancy_pairs(FANCY_H2V1, first, len - 1, &mut h2v1, |i| {
                u32::from(curr[i])
            });
            let mut h2v2 = vec![0u16; 2 * pairs];
            fancy_pairs(FANCY_H2V2, first, len - 1, &mut h2v2, |i| {
                3 * u32::from(curr[i]) + u32::from(near[i])
            });
            for (offset, (&up1, &up2)) in h2v1.iter().zip(&h2v2).enumerate() {
                let x = 2 * first + offset;
                assert_eq!(
                    up1,
                    upsample_extended12_h2v1_at(&curr, x),
                    "h2v1 len {len} x {x}"
                );
                // Single-sample rows take the dedicated path in the writers.
                if len > 1 {
                    let expected = upsample_h2v2_rows_at(&curr, &near, 2 * len, x);
                    assert_eq!(up2, expected, "h2v2 len {len} x {x}");
                }
            }
        }
    }
}

fn plane(seed: u32, width: usize, rows: usize) -> Extended12Plane {
    let stride = width.next_multiple_of(8);
    Extended12Plane {
        pixels: samples(seed, stride * rows),
        stride,
        width,
    }
}

fn region(dimensions: (u32, u32), output_rect: Rect) -> Extended12WriteRegion {
    Extended12WriteRegion {
        output_rect,
        dimensions,
        downscale: DownscaleFactor::Full,
        output: Extended12Output::Rgb16,
    }
}

fn rects(width: u32, height: u32) -> [Rect; 3] {
    [
        Rect::full((width, height)),
        Rect {
            x: 1,
            y: 1,
            w: width - 2,
            h: height - 2,
        },
        Rect {
            x: 3,
            y: 2,
            w: width - 4,
            h: 1,
        },
    ]
}

/// Per-pixel reference for the full-resolution subsampled writers.
fn expected_subsampled(
    planes: &[Extended12Plane; 3],
    dimensions: (u32, u32),
    rect: Rect,
    h2v2: bool,
) -> Vec<u8> {
    let stride = rect.w as usize * 6;
    let mut out = vec![0u8; stride * rect.h as usize];
    let chroma_height = (dimensions.1 as usize).div_ceil(2);
    for y in 0..rect.h as usize {
        let source_y = rect.y as usize + y;
        for x in 0..rect.w as usize {
            let source_x = rect.x as usize + x;
            let luma = planes[0].pixels[source_y * planes[0].stride + source_x];
            let [c1, c2] = [&planes[1], &planes[2]].map(|plane| {
                if h2v2 {
                    super::super::upsample::upsample_extended12_plane_h2v2_at(
                        plane,
                        chroma_height,
                        source_x,
                        source_y,
                    )
                } else {
                    super::super::upsample::upsample_extended12_plane_h2v1_at(
                        plane, source_x, source_y,
                    )
                }
            });
            let (r, g, b) = crate::color::ycbcr::ycbcr12_to_rgb16(luma, c1, c2);
            let dst = &mut out[y * stride + x * 6..][..6];
            dst[0..2].copy_from_slice(&r.to_le_bytes());
            dst[2..4].copy_from_slice(&g.to_le_bytes());
            dst[4..6].copy_from_slice(&b.to_le_bytes());
        }
    }
    out
}

#[test]
fn full_resolution_subsampled_writers_match_per_pixel_upsampling() {
    for (width, height) in [(5u32, 6u32), (67, 9), (130, 4), (4, 3)] {
        for h2v2 in [true, false] {
            let chroma_width = (width as usize).div_ceil(2);
            let chroma_rows = if h2v2 {
                (height as usize).div_ceil(2)
            } else {
                height as usize
            };
            let planes = [
                plane(width * 3, width as usize, height as usize),
                plane(width * 5, chroma_width, chroma_rows),
                plane(width * 11, chroma_width, chroma_rows),
            ];
            for rect in rects(width, height) {
                let stride = rect.w as usize * 6;
                let mut actual = vec![0u8; stride * rect.h as usize];
                let write = if h2v2 {
                    write_extended12_color420_planes_region
                } else {
                    write_extended12_color422_planes_region
                };
                write(
                    &mut actual,
                    stride,
                    region((width, height), rect),
                    Extended12RgbProjection::YCbCr,
                    &planes,
                );
                assert_eq!(
                    actual,
                    expected_subsampled(&planes, (width, height), rect, h2v2),
                    "{width}x{height} h2v2={h2v2} {rect:?}"
                );
            }
        }
    }
}

#[test]
fn full_resolution_block_writers_match_the_per_pixel_path() {
    let (width, height) = (21u32, 13u32);
    let pixels: [[u16; 64]; 3] = [1, 2, 3].map(|seed| {
        let values = samples(seed, 64);
        core::array::from_fn(|i| values[i])
    });
    for rect in rects(width, height) {
        for output in [Extended12Output::Gray16, Extended12Output::Rgb16] {
            let bytes = if matches!(output, Extended12Output::Gray16) {
                2
            } else {
                6
            };
            let stride = rect.w as usize * bytes;
            let write_region = Extended12WriteRegion {
                output,
                ..region((width, height), rect)
            };
            let mut gray = vec![0u8; stride * rect.h as usize];
            let mut expected_gray = gray.clone();
            let mut color = vec![0u8; rect.w as usize * 6 * rect.h as usize];
            let mut expected_color = color.clone();
            for origin in [(0, 0), (8, 0), (16, 8), (8, 8)] {
                write_extended12_block_region(&mut gray, stride, write_region, origin, &pixels[0]);
                for_each_block_output(write_region, origin, |row, col, src| {
                    let sample = pixels[0][src];
                    match output {
                        Extended12Output::Gray16 => {
                            let at = row * stride + col * 2;
                            expected_gray[at..at + 2].copy_from_slice(&sample.to_le_bytes());
                        }
                        Extended12Output::Rgb16 => {
                            write_rgb16(
                                &mut expected_gray,
                                row * stride + col * 6,
                                (sample, sample, sample),
                            );
                        }
                    }
                });
                let color_stride = rect.w as usize * 6;
                write_extended12_rgb_block_region(
                    &mut color,
                    color_stride,
                    write_region,
                    Extended12RgbProjection::YCbCr,
                    origin,
                    &pixels,
                );
                for_each_block_output(write_region, origin, |row, col, src| {
                    let rgb = crate::color::ycbcr::ycbcr12_to_rgb16(
                        pixels[0][src],
                        pixels[1][src],
                        pixels[2][src],
                    );
                    write_rgb16(&mut expected_color, row * color_stride + col * 6, rgb);
                });
            }
            assert_eq!(gray, expected_gray, "{output:?} {rect:?}");
            assert_eq!(color, expected_color, "{rect:?}");
        }
    }
}
