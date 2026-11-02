// SPDX-License-Identifier: MIT OR Apache-2.0

//! An 8-bit full-image decode whose component planes would pass the decode
//! budget, or for which streaming is the faster route, runs as a region
//! decode of the whole image that streams its rows (several tiles at once
//! when it has them and the pool to run them). A decoder whose planes budget
//! is fixed takes either route on demand, and the streamed route must write
//! the bytes the planes route writes, at every scale, stride and
//! parallelism, and fail where it fails.

use j2k::{CpuDecodeParallelism, J2kDecoder, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::wrap_jp2_rgba_codestream;

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

/// The image at `scale` in a buffer padded past each row, or `None` when the
/// decode fails, from a decoder whose planes budget is `planes_budget` when
/// one is given.
fn decode(
    bytes: &[u8],
    planes_budget: Option<usize>,
    parallelism: CpuDecodeParallelism,
    fmt: PixelFormat,
    scale: Downscale,
    padding: usize,
) -> Option<Vec<u8>> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    if let Some(bytes) = planes_budget {
        decoder.set_full_decode_planes_budget(bytes);
    }
    let (width, height) = decoder.info().dimensions;
    let divisor = scale.denominator();
    let (width, height) = (width.div_ceil(divisor), height.div_ceil(divisor));
    let row = width as usize * fmt.bytes_per_pixel();
    let stride = row + padding;
    let mut out = vec![PADDING; stride * height as usize];
    let pool = &mut J2kScratchPool::new();
    let result = match scale {
        Downscale::None => decoder.decode_into(&mut out, stride, fmt),
        _ => decoder.decode_scaled_into(pool, &mut out, stride, fmt, scale),
    };
    result.ok()?;
    for line in out.chunks(stride) {
        assert!(
            line[row..].iter().all(|&byte| byte == PADDING),
            "stride padding untouched"
        );
    }
    Some(out)
}

#[test]
fn streamed_full_image_matches_planes() {
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
    let rgb_tiled_ht = codestream(
        3,
        &EncodeOptions {
            reversible: true,
            tile_size: Some((64, 64)),
            use_ht_block_coding: true,
            ..base.clone()
        },
    );
    // Enough tiles for every worker of a large pool: decoded tiles at once.
    let rgb_many_tiles = codestream(
        3,
        &EncodeOptions {
            reversible: false,
            tile_size: Some((16, 16)),
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
    let gray_tiled = codestream(
        1,
        &EncodeOptions {
            reversible: false,
            tile_size: Some((96, 64)),
            ..base.clone()
        },
    );
    let rgba = wrap_jp2_rgba_codestream(
        &codestream(
            4,
            &EncodeOptions {
                reversible: false,
                ..base
            },
        ),
        WIDTH,
        HEIGHT,
        8,
    );
    let (rgb_formats, gray_formats) = (
        &[PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8][..],
        &[PixelFormat::Gray8, PixelFormat::Rgb8][..],
    );
    let fixtures = [
        ("rgb", rgb, rgb_formats),
        ("rgb tiled HT", rgb_tiled_ht, rgb_formats),
        ("rgb many tiles", rgb_many_tiles, rgb_formats),
        ("gray", gray, gray_formats),
        ("gray tiled", gray_tiled, gray_formats),
        ("rgba", rgba, rgb_formats),
    ];
    let mut decoded = 0;
    for (label, bytes, formats) in &fixtures {
        for &fmt in *formats {
            for scale in [Downscale::None, Downscale::Half, Downscale::Eighth] {
                for (parallelism, padding) in [
                    (CpuDecodeParallelism::Serial, 5),
                    (CpuDecodeParallelism::Auto, 0),
                ] {
                    let planes = decode(bytes, Some(usize::MAX), parallelism, fmt, scale, padding);
                    let streamed = decode(bytes, Some(0), parallelism, fmt, scale, padding);
                    let default = decode(bytes, None, parallelism, fmt, scale, padding);
                    assert!(
                        planes == streamed && planes == default,
                        "{label} {fmt:?} {scale:?} {parallelism:?}: planes {} streamed {} default {}",
                        planes.is_some(),
                        streamed.is_some(),
                        default.is_some()
                    );
                    decoded += usize::from(planes.is_some());
                }
            }
        }
    }
    // Gray to RGB and RGB to gray are rejected; the rest must decode.
    assert!(decoded >= 9 * 3 * 2, "{decoded} decodes succeeded");
}

/// Tiles whose coarse resolutions are empty while finer ones are not (a
/// one-sample tile row or column at an odd position, with two levels)
/// stream past a lowered budget like any other, and the streamed route
/// writes what the planes route writes.
#[test]
fn tiles_with_empty_coarse_levels_stream_as_the_planes_decode() {
    let gray: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    for (label, bytes, fmt) in [
        ("gray 5/3", gray, PixelFormat::Gray8),
        ("rgb 9/7", rgb, PixelFormat::Rgb8),
    ] {
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            for scale in [Downscale::None, Downscale::Half] {
                let planes = decode(bytes, Some(usize::MAX), parallelism, fmt, scale, 0);
                let lowered = decode(bytes, Some(0), parallelism, fmt, scale, 0);
                let default = decode(bytes, None, parallelism, fmt, scale, 0);
                assert!(planes.is_some(), "{label} {parallelism:?} {scale:?}");
                assert_eq!(planes, lowered, "{label} {parallelism:?} {scale:?}");
                assert_eq!(planes, default, "{label} {parallelism:?} {scale:?}");
            }
        }
    }
}

/// A decoder that streamed one full-image decode decodes the next on the
/// planes, and the other way round, from the context it kept.
#[test]
fn routes_alternate_on_one_decoder() {
    let bytes = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 3,
            reversible: false,
            ..EncodeOptions::default()
        },
    );
    let fmt = PixelFormat::Rgb8;
    let stride = WIDTH as usize * 3;
    let expected = decode(
        &bytes,
        None,
        CpuDecodeParallelism::Auto,
        fmt,
        Downscale::None,
        0,
    )
    .expect("planes decode");
    let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
    for budget in [usize::MAX, 0, usize::MAX, 0] {
        decoder.set_full_decode_planes_budget(budget);
        let mut out = vec![0; stride * HEIGHT as usize];
        decoder
            .decode_into(&mut out, stride, fmt)
            .unwrap_or_else(|error| panic!("budget {budget}: {error:?}"));
        assert!(out == expected, "budget {budget}");
    }
}
