// SPDX-License-Identifier: MIT OR Apache-2.0

//! A 16-bit full-image decode and a decode into owned native component
//! planes whose `f32` planes would pass the decode budget run as a region
//! decode of the whole image that streams its rows, as 8-bit decodes do, and
//! 16-bit region decodes stream their rows always. A decoder whose planes
//! budget is lowered takes the streamed route for small images, and must
//! write the bytes the planes route writes for every precision and
//! signedness, at every scale, stride and parallelism, and fail where it
//! fails.

use j2k::{CpuDecodeParallelism, J2kDecodedNativeComponents, J2kDecoder, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::wrap_jp2_rgba_codestream;

const WIDTH: u32 = 203;
const HEIGHT: u32 = 157;
const PADDING: u8 = 0xab;

const RGB_FORMATS: &[PixelFormat] = &[
    PixelFormat::Rgb16,
    PixelFormat::Rgba16,
    PixelFormat::Gray16,
    PixelFormat::Rgb8,
];
const GRAY_FORMATS: &[PixelFormat] = &[PixelFormat::Gray16, PixelFormat::Rgb16, PixelFormat::Gray8];
const RGBA_FORMATS: &[PixelFormat] = &[PixelFormat::Rgba16, PixelFormat::Rgb16, PixelFormat::Rgba8];

/// Noise samples of `bit_depth` bits, one to three little-endian bytes
/// each, which the encoder masks to the precision (and sign-extends when
/// signed).
fn noisy_samples(components: u16, bit_depth: u8) -> Vec<u8> {
    let bytes = usize::from(bit_depth).div_ceil(8);
    let mut state = 0x2545_f491_u32 ^ u32::from(bit_depth);
    (0..WIDTH as usize * HEIGHT as usize * usize::from(components) * bytes)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn codestream(components: u16, bit_depth: u8, signed: bool, options: &EncodeOptions) -> Vec<u8> {
    let samples = noisy_samples(components, bit_depth);
    encode(
        &samples, WIDTH, HEIGHT, components, bit_depth, signed, options,
    )
    .expect("encode fixture")
}

/// Images of 4, 8, 12, 16, 20 and 24 bits, signed and unsigned, reversible
/// and irreversible, tiled and untiled, classic and HT, with the output
/// formats each is decoded to (8-bit ones included: they stream past the
/// decode budget too).
fn fixtures() -> Vec<(&'static str, Vec<u8>, &'static [PixelFormat])> {
    let levels = EncodeOptions {
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    let irreversible = EncodeOptions {
        reversible: false,
        ..levels.clone()
    };
    let reversible = EncodeOptions {
        reversible: true,
        ..levels.clone()
    };
    vec![
        (
            "rgb 8-bit 9/7",
            codestream(3, 8, false, &irreversible),
            RGB_FORMATS,
        ),
        (
            "rgb 8-bit tiled HT 5/3",
            codestream(
                3,
                8,
                false,
                &EncodeOptions {
                    tile_size: Some((64, 64)),
                    use_ht_block_coding: true,
                    ..reversible.clone()
                },
            ),
            RGB_FORMATS,
        ),
        (
            "rgb 12-bit signed 9/7",
            codestream(3, 12, true, &irreversible),
            RGB_FORMATS,
        ),
        (
            "rgb 16-bit 5/3",
            codestream(3, 16, false, &reversible),
            RGB_FORMATS,
        ),
        (
            "gray 12-bit 5/3",
            codestream(1, 12, false, &reversible),
            GRAY_FORMATS,
        ),
        (
            "gray 16-bit tiled 9/7",
            codestream(
                1,
                16,
                false,
                &EncodeOptions {
                    tile_size: Some((96, 64)),
                    ..irreversible.clone()
                },
            ),
            GRAY_FORMATS,
        ),
        (
            "gray 16-bit signed 9/7",
            codestream(1, 16, true, &irreversible),
            GRAY_FORMATS,
        ),
        (
            "gray 20-bit 5/3",
            codestream(1, 20, false, &reversible),
            GRAY_FORMATS,
        ),
        (
            "rgb 24-bit signed 9/7",
            codestream(3, 24, true, &irreversible),
            RGB_FORMATS,
        ),
        (
            "gray 8-bit signed 5/3",
            codestream(1, 8, true, &reversible),
            GRAY_FORMATS,
        ),
        (
            "gray 4-bit 5/3",
            codestream(1, 4, false, &reversible),
            GRAY_FORMATS,
        ),
        (
            "rgba 8-bit 9/7",
            wrap_jp2_rgba_codestream(&codestream(4, 8, false, &irreversible), WIDTH, HEIGHT, 8),
            RGBA_FORMATS,
        ),
    ]
}

fn decoder(
    bytes: &[u8],
    planes_budget: Option<usize>,
    parallelism: CpuDecodeParallelism,
) -> J2kDecoder<'_> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    if let Some(bytes) = planes_budget {
        decoder.set_full_decode_planes_budget(bytes);
    }
    decoder
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
    let mut decoder = decoder(bytes, planes_budget, parallelism);
    let divisor = scale.denominator();
    let (width, height) = (WIDTH.div_ceil(divisor), HEIGHT.div_ceil(divisor));
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
fn streamed_full_image_outputs_match_planes() {
    let mut decoded = 0;
    for (label, bytes, formats) in &fixtures() {
        for &fmt in *formats {
            for scale in [Downscale::None, Downscale::Half, Downscale::Eighth] {
                for (parallelism, padding) in [
                    (CpuDecodeParallelism::Serial, 6),
                    (CpuDecodeParallelism::Auto, 0),
                ] {
                    let planes = decode(bytes, None, parallelism, fmt, scale, padding);
                    let streamed = decode(bytes, Some(0), parallelism, fmt, scale, padding);
                    assert!(
                        planes == streamed,
                        "{label} {fmt:?} {scale:?} {parallelism:?}: planes {} streamed {}",
                        planes.is_some(),
                        streamed.is_some()
                    );
                    decoded += usize::from(planes.is_some());
                }
            }
        }
    }
    // Gray to RGB and RGB to gray are rejected; the rest must decode.
    assert!(
        decoded >= (4 * 3 + 5 * 2 + 3) * 3 * 2,
        "{decoded} decodes succeeded"
    );
}

fn native_components(
    bytes: &[u8],
    planes_budget: Option<usize>,
    parallelism: CpuDecodeParallelism,
) -> J2kDecodedNativeComponents {
    decoder(bytes, planes_budget, parallelism)
        .decode_native_components()
        .expect("native components")
}

#[test]
fn streamed_native_components_match_planes() {
    for (label, bytes, _) in &fixtures() {
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            let planes = native_components(bytes, None, parallelism);
            let streamed = native_components(bytes, Some(0), parallelism);
            assert_eq!(planes, streamed, "{label} {parallelism:?}");
        }
    }
}

/// Native component region decodes stream their rows into the planes,
/// which must hold the full image's planes cropped to the region.
#[test]
fn native_region_components_match_the_full_image() {
    for (label, bytes, _) in &fixtures() {
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            let full = native_components(bytes, None, parallelism);
            for roi in regions() {
                let region = decoder(bytes, None, parallelism)
                    .decode_native_region_components(roi)
                    .unwrap_or_else(|error| panic!("{label} {roi:?}: {error:?}"));
                assert_eq!(region.dimensions(), (roi.w, roi.h), "{label} {roi:?}");
                assert_eq!(region.color_space(), full.color_space(), "{label}");
                assert_eq!(region.planes().len(), full.planes().len(), "{label}");
                for (plane, whole) in region.planes().iter().zip(full.planes()) {
                    assert_eq!(plane.bit_depth(), whole.bit_depth(), "{label}");
                    assert_eq!(plane.signed(), whole.signed(), "{label}");
                    let bytes = usize::from(whole.bytes_per_sample());
                    let row = roi.w as usize * bytes;
                    for (y, line) in plane.data().chunks(row).enumerate() {
                        let start =
                            ((roi.y as usize + y) * WIDTH as usize + roi.x as usize) * bytes;
                        assert!(
                            line == &whole.data()[start..start + row],
                            "{label} {parallelism:?} {roi:?} row {y}"
                        );
                    }
                }
            }
        }
    }
}

fn regions() -> [Rect; 4] {
    [
        Rect::full((WIDTH, HEIGHT)),
        Rect {
            x: 17,
            y: 9,
            w: 64,
            h: 50,
        },
        Rect {
            x: WIDTH - 33,
            y: HEIGHT - 21,
            w: 33,
            h: 21,
        },
        Rect {
            x: 100,
            y: 0,
            w: 1,
            h: HEIGHT,
        },
    ]
}

/// The 16-bit region decodes stream their rows, which must be the rows of
/// the full image the planes decode.
#[test]
fn sixteen_bit_regions_match_the_full_image() {
    let pool = &mut J2kScratchPool::new();
    for (label, bytes, formats) in &fixtures() {
        for &fmt in formats.iter().filter(|fmt| fmt.bytes_per_pixel() % 2 == 0) {
            for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
                let Some(full) = decode(bytes, None, parallelism, fmt, Downscale::None, 0) else {
                    continue;
                };
                let pixel = fmt.bytes_per_pixel();
                let full_stride = WIDTH as usize * pixel;
                for roi in regions() {
                    let stride = roi.w as usize * pixel + 2;
                    let mut out = vec![PADDING; stride * roi.h as usize];
                    decoder(bytes, None, parallelism)
                        .decode_region_into(pool, &mut out, stride, fmt, roi)
                        .unwrap_or_else(|error| panic!("{label} {fmt:?} {roi:?}: {error:?}"));
                    for (row, line) in out.chunks(stride).enumerate() {
                        let start = (roi.y as usize + row) * full_stride + roi.x as usize * pixel;
                        assert!(
                            line[..roi.w as usize * pixel]
                                == full[start..start + roi.w as usize * pixel],
                            "{label} {fmt:?} {parallelism:?} {roi:?} row {row}"
                        );
                        assert_eq!(line[roi.w as usize * pixel..], [PADDING; 2]);
                    }
                }
            }
        }
    }
}

/// Samples of `bit_depth` bits, at least a quarter of them within 16 bits
/// (signed: within `i16`).
fn deep_samples(components: u16, bit_depth: u8, signed: bool) -> Vec<i64> {
    let span = 1_i64 << 18;
    let offset = if signed { span / 2 } else { 0 };
    let (min, max) = if signed {
        (-(1 << (bit_depth - 1)), (1 << (bit_depth - 1)) - 1)
    } else {
        (0, (1 << bit_depth) - 1)
    };
    let count = WIDTH as usize * HEIGHT as usize * usize::from(components);
    (0_i64..)
        .take(count)
        .map(|index| ((index * 7919 + index / 3) % span - offset).clamp(min, max))
        .collect()
}

/// A 16-bit output of an image deeper than 16 bits holds each sample's
/// value clamped to the 16-bit range (two's complement when signed), as a
/// scaled decode always wrote it, on the planes route, on the streamed
/// route and in region decodes; an 8-bit output of an unsigned one holds
/// the sample over the component's range, on either route.
#[test]
fn deep_samples_decode_to_their_values_clamped_to_sixteen_bits() {
    let reversible = EncodeOptions {
        num_decomposition_levels: 3,
        reversible: true,
        ..EncodeOptions::default()
    };
    let pool = &mut J2kScratchPool::new();
    for (components, bit_depth, signed, fmt) in [
        (1_u16, 20_u8, false, PixelFormat::Gray16),
        (1, 24, true, PixelFormat::Gray16),
        (3, 20, true, PixelFormat::Rgb16),
        (3, 17, false, PixelFormat::Rgb16),
    ] {
        let label = format!("{components} x {bit_depth} bits, signed {signed}");
        let values = deep_samples(components, bit_depth, signed);
        let samples: Vec<u8> = values
            .iter()
            .flat_map(|value| {
                let bytes = value.to_le_bytes();
                [bytes[0], bytes[1], bytes[2]]
            })
            .collect();
        let bytes = encode(
            &samples,
            WIDTH,
            HEIGHT,
            components,
            bit_depth,
            signed,
            &reversible,
        )
        .expect("encode deep fixture");
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|&value| {
                let clamped = if signed {
                    let value = i16::try_from(value.clamp(-32_768, 32_767)).expect("i16");
                    u16::from_le_bytes(value.to_le_bytes())
                } else {
                    u16::try_from(value.clamp(0, 65_535)).expect("u16")
                };
                clamped.to_le_bytes()
            })
            .collect();
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            for budget in [None, Some(0)] {
                let decoded = decode(&bytes, budget, parallelism, fmt, Downscale::None, 0)
                    .unwrap_or_else(|| panic!("{label} {budget:?}: decode failed"));
                assert!(
                    decoded == expected,
                    "{label} {parallelism:?} planes budget {budget:?}"
                );
            }
            if !signed {
                let fmt = if components == 1 {
                    PixelFormat::Gray8
                } else {
                    PixelFormat::Rgb8
                };
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "the unsigned fixtures are at most 20 bits deep, exact in f32"
                )]
                let max = ((1_u64 << bit_depth) - 1) as f32;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss,
                    reason = "unsigned samples below 2^20 are exact in f32 and scale into 0..=255"
                )]
                let expected: Vec<u8> = values
                    .iter()
                    .map(|&value| ((value as f32 / max) * 255.0).round() as u8)
                    .collect();
                for budget in [None, Some(0)] {
                    let decoded = decode(&bytes, budget, parallelism, fmt, Downscale::None, 0);
                    assert!(
                        decoded.as_ref() == Some(&expected),
                        "{label} {fmt:?} {parallelism:?} planes budget {budget:?}"
                    );
                }
            }
            let pixel = fmt.bytes_per_pixel();
            for roi in regions() {
                let stride = roi.w as usize * pixel;
                let mut out = vec![PADDING; stride * roi.h as usize];
                decoder(&bytes, None, parallelism)
                    .decode_region_into(pool, &mut out, stride, fmt, roi)
                    .unwrap_or_else(|error| panic!("{label} {roi:?}: {error:?}"));
                for (row, line) in out.chunks(stride).enumerate() {
                    let start = ((roi.y as usize + row) * WIDTH as usize + roi.x as usize) * pixel;
                    assert!(
                        line == &expected[start..start + stride],
                        "{label} {parallelism:?} {roi:?} row {row}"
                    );
                }
            }
        }
    }
}

/// Tiles whose coarse resolutions are empty while finer ones are not keep
/// the planes for 16-bit and native component output too.
#[test]
fn tiles_with_empty_coarse_levels_keep_the_planes() {
    let gray: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    for (label, bytes, fmt) in [
        ("gray 5/3", gray, PixelFormat::Gray16),
        ("rgb 9/7", rgb, PixelFormat::Rgb16),
    ] {
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            let mut planes = decoder(bytes, None, parallelism);
            let mut lowered = decoder(bytes, Some(0), parallelism);
            let (width, height) = planes.info().dimensions;
            let stride = width as usize * fmt.bytes_per_pixel();
            let mut expected = vec![0; stride * height as usize];
            let mut out = vec![0; stride * height as usize];
            planes
                .decode_into(&mut expected, stride, fmt)
                .unwrap_or_else(|error| panic!("{label}: {error:?}"));
            lowered
                .decode_into(&mut out, stride, fmt)
                .unwrap_or_else(|error| panic!("{label}: {error:?}"));
            assert_eq!(out, expected, "{label} {parallelism:?}");
            assert_eq!(
                lowered.decode_native_components().expect("native"),
                planes.decode_native_components().expect("native"),
                "{label} {parallelism:?}"
            );
        }
    }
}
