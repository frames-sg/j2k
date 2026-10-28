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
