// SPDX-License-Identifier: MIT OR Apache-2.0

//! A serial 8-bit region decode writes rows into the caller's buffer as they
//! are finished, where a parallel one fills component planes first. Both must
//! write the same bytes, leave stride padding alone, and fail the same way.

use j2k::{CpuDecodeParallelism, J2kDecoder, J2kError, J2kScratchPool};
use j2k_core::{PixelFormat, Rect};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::{wrap_jp2_codestream, wrap_jp2_rgba_codestream};

const WIDTH: u32 = 203;
const HEIGHT: u32 = 157;
const PADDING: u8 = 0xab;

fn noisy_pixels(components: u32) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    (0..WIDTH * HEIGHT * components)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn codestream(components: u16, options: &EncodeOptions) -> Vec<u8> {
    let pixels = noisy_pixels(u32::from(components));
    encode(&pixels, WIDTH, HEIGHT, components, 8, false, options).expect("encode fixture")
}

/// The region into a padded buffer, or the error, decoded at `levels`.
fn decode(
    bytes: &[u8],
    parallelism: CpuDecodeParallelism,
    fmt: PixelFormat,
    roi: Rect,
    levels: u8,
) -> Result<Vec<u8>, String> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    let divisor = 1 << levels;
    let width = (roi.x + roi.w).div_ceil(divisor) - roi.x / divisor;
    let height = (roi.y + roi.h).div_ceil(divisor) - roi.y / divisor;
    let stride = width as usize * fmt.bytes_per_pixel() + 5;
    let mut out = vec![PADDING; stride * height as usize];
    let result = decoder.decode_region_scaled_pow2_into(
        &mut J2kScratchPool::new(),
        &mut out,
        stride,
        fmt,
        roi,
        levels,
    );
    match result {
        Ok(_) => {
            for row in out.chunks(stride) {
                assert!(
                    row[width as usize * fmt.bytes_per_pixel()..]
                        .iter()
                        .all(|&byte| byte == PADDING),
                    "stride padding untouched"
                );
            }
            Ok(out)
        }
        Err(error) => {
            assert!(
                out.iter().all(|&byte| byte == PADDING),
                "a rejected format writes nothing"
            );
            Err(error_kind(&error))
        }
    }
}

fn error_kind(error: &J2kError) -> String {
    format!("{error:?}")
}

#[test]
fn serial_region_rows_match_parallel_planes() {
    let base = EncodeOptions {
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    let rgb = codestream(
        3,
        &EncodeOptions {
            reversible: false,
            ..base.clone()
        },
    );
    let rgb_tiled = codestream(
        3,
        &EncodeOptions {
            reversible: true,
            tile_size: Some((64, 64)),
            use_ht_block_coding: true,
            ..base.clone()
        },
    );
    let gray = codestream(
        1,
        &EncodeOptions {
            reversible: true,
            ..base.clone()
        },
    );
    let rgba = codestream(
        4,
        &EncodeOptions {
            reversible: false,
            ..base
        },
    );
    let fixtures = [
        ("rgb", rgb.clone()),
        ("rgb tiled HT", rgb_tiled),
        ("gray", gray.clone()),
        (
            "gray jp2",
            wrap_jp2_codestream(&gray, WIDTH, HEIGHT, 1, 8, 17),
        ),
        (
            "rgb jp2",
            wrap_jp2_codestream(&rgb, WIDTH, HEIGHT, 3, 8, 16),
        ),
        // The channel definition keeps this one on the planes path.
        (
            "rgba jp2",
            wrap_jp2_rgba_codestream(&rgba, WIDTH, HEIGHT, 8),
        ),
    ];
    let regions = [
        Rect {
            x: 0,
            y: 0,
            w: WIDTH,
            h: HEIGHT,
        },
        Rect {
            x: 37,
            y: 21,
            w: 130,
            h: 101,
        },
        Rect {
            x: 63,
            y: 60,
            w: 9,
            h: 70,
        },
    ];
    let mut decoded = 0;
    for (label, bytes) in &fixtures {
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8] {
            for roi in regions {
                for levels in [0, 1, 2] {
                    let serial = decode(bytes, CpuDecodeParallelism::Serial, fmt, roi, levels);
                    let parallel = decode(bytes, CpuDecodeParallelism::Auto, fmt, roi, levels);
                    assert_eq!(serial, parallel, "{label} {fmt:?} {roi:?} levels={levels}");
                    decoded += usize::from(serial.is_ok());
                }
            }
        }
    }
    // Gray to RGB and RGB to gray are rejected; the rest must decode.
    assert!(decoded > 50, "{decoded} decodes succeeded");
}

/// The image decoded into a tight buffer, and its row length.
fn decode_full(
    bytes: &[u8],
    parallelism: CpuDecodeParallelism,
    fmt: PixelFormat,
) -> (Vec<u8>, usize) {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    let (width, height) = decoder.info().dimensions;
    let stride = width as usize * fmt.bytes_per_pixel();
    let mut out = vec![0; stride * height as usize];
    decoder
        .decode_into(&mut out, stride, fmt)
        .expect("full decode");
    (out, stride)
}

/// A region of a tile whose coarse resolutions are empty (a one-sample tile
/// row or column at an odd position, with two levels) decodes the full
/// image's pixels, on the streamed rows and on the planes, at both bit depths.
#[test]
fn region_of_tile_with_empty_coarse_levels_matches_full_decode() {
    let gray: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    let fixtures = [
        (
            "gray 5/3",
            gray,
            [PixelFormat::Gray8, PixelFormat::Gray16],
            [(0, 5, 8, 1), (3, 5, 1, 1), (2, 3, 5, 3), (0, 0, 8, 6)],
        ),
        (
            "rgb 9/7",
            rgb,
            [PixelFormat::Rgb8, PixelFormat::Rgb16],
            [(5, 0, 1, 6), (5, 5, 1, 1), (0, 5, 6, 1), (3, 2, 3, 4)],
        ),
    ];
    for (label, bytes, formats, regions) in fixtures {
        for fmt in formats {
            let bytes_per_pixel = fmt.bytes_per_pixel();
            for parallelism in [CpuDecodeParallelism::Auto, CpuDecodeParallelism::Serial] {
                let (full, full_stride) = decode_full(bytes, parallelism, fmt);
                for (x, y, w, h) in regions {
                    let roi = Rect { x, y, w, h };
                    let stride = w as usize * bytes_per_pixel;
                    let mut out = vec![0; stride * h as usize];
                    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
                    decoder.set_cpu_decode_parallelism(parallelism);
                    decoder
                        .decode_region_into(&mut J2kScratchPool::new(), &mut out, stride, fmt, roi)
                        .unwrap_or_else(|error| {
                            panic!("{label} {fmt:?} {parallelism:?} {roi:?}: {error:?}")
                        });
                    let crop: Vec<u8> = full
                        .chunks(full_stride)
                        .skip(y as usize)
                        .take(h as usize)
                        .flat_map(|row| &row[x as usize * bytes_per_pixel..][..stride])
                        .copied()
                        .collect();
                    assert_eq!(out, crop, "{label} {fmt:?} {parallelism:?} {roi:?}");
                }
            }
        }
    }
    // The reversible image decodes losslessly.
    let expected: Vec<u8> = (0..6_u32)
        .flat_map(|y| (0..8_u32).map(move |x| ((13 * x + 29 * y + 7) % 251) as u8))
        .collect();
    assert_eq!(
        decode_full(gray, CpuDecodeParallelism::Auto, PixelFormat::Gray8).0,
        expected
    );
}
