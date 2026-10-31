// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{full_image_route, FullImageRoute, PlanesBudget, RouteRequest};
use crate::backend::image_at_scale;
use crate::DecodeSettings;
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::DecoderContext;
use j2k_native::{encode, CpuDecodeParallelism, EncodeOptions};

fn codestream(components: u16, options: &EncodeOptions) -> Vec<u8> {
    let (width, height) = (96_u32, 80_u32);
    let pixels: Vec<u8> = (0..width * height * u32::from(components))
        .map(|index| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    encode(&pixels, width, height, components, 8, false, options).expect("encode fixture")
}

fn route(bytes: &[u8], scale: Downscale, fmt: PixelFormat, budget: usize) -> FullImageRoute {
    let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
    full_image_route(&image, request(fmt, budget, CpuDecodeParallelism::Serial))
}

fn request(fmt: PixelFormat, budget: usize, parallelism: CpuDecodeParallelism) -> RouteRequest {
    RouteRequest {
        fmt,
        budget: PlanesBudget::Fixed(budget),
        parallelism,
        components: 3,
        reduced: false,
    }
}

/// The floor the native decoder plans for the planes of `bytes` at `scale`.
fn floor(bytes: &[u8], scale: Downscale) -> usize {
    let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
    image
        .full_decode_planes_floor(0)
        .expect("plan")
        .expect("a streamable image has a floor")
}

#[test]
fn an_image_streams_only_when_its_planes_pass_the_budget() {
    let fixtures = [
        codestream(
            3,
            &EncodeOptions {
                num_decomposition_levels: 3,
                reversible: false,
                ..EncodeOptions::default()
            },
        ),
        codestream(
            1,
            &EncodeOptions {
                num_decomposition_levels: 3,
                tile_size: Some((32, 32)),
                use_ht_block_coding: true,
                ..EncodeOptions::default()
            },
        ),
    ];
    for bytes in &fixtures {
        for scale in [Downscale::None, Downscale::Half, Downscale::Quarter] {
            let floor = floor(bytes, scale);
            // The planes alone of the smallest output are more than this.
            assert!(floor > 20 * 24 * 4, "{scale:?}: {floor}");
            for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8] {
                assert_eq!(
                    route(bytes, scale, fmt, floor - 1),
                    FullImageRoute::StreamedRegion
                );
                assert_eq!(route(bytes, scale, fmt, floor), FullImageRoute::Planes);
                assert_eq!(
                    route(bytes, scale, fmt, j2k_native::DEFAULT_MAX_DECODE_BYTES),
                    FullImageRoute::Planes
                );
            }
        }
    }
}

#[test]
fn reduced_decodes_plan_full_resolution_coefficients() {
    let bytes = codestream(
        1,
        &EncodeOptions {
            num_decomposition_levels: 3,
            ..EncodeOptions::default()
        },
    );
    // A quarter of the planes and IDWT buffers, but every coefficient.
    let full = floor(&bytes, Downscale::None);
    let half = floor(&bytes, Downscale::Half);
    assert!(half < full, "{half} < {full}");
    assert!(half > 96 * 80 * 4, "{half}");
}

#[test]
fn sixteen_bit_output_keeps_the_planes() {
    let bytes = codestream(3, &EncodeOptions::default());
    for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16, PixelFormat::Gray16] {
        assert_eq!(
            route(&bytes, Downscale::None, fmt, 0),
            FullImageRoute::Planes
        );
    }
}

/// A tile whose coarse resolutions are empty while finer ones are not keeps
/// the planes, which reconstruct it, whatever the budget.
#[test]
fn tiles_with_empty_coarse_levels_keep_the_planes() {
    let gray: &[u8] = include_bytes!(
        "../../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    for (bytes, fmt) in [(gray, PixelFormat::Gray8), (rgb, PixelFormat::Rgb8)] {
        let image =
            image_at_scale(bytes, DecodeSettings::default(), Downscale::None).expect("image");
        assert_eq!(image.full_decode_planes_floor(0).expect("plan"), None);
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            assert_eq!(
                full_image_route(&image, request(fmt, 0, parallelism)),
                FullImageRoute::Planes
            );
        }
    }
}

/// The output rectangles of an image's tiles cover it once, at every scale.
#[test]
fn tile_rects_cover_the_output_once() {
    let bytes = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 3,
            tile_size: Some((40, 24)),
            ..EncodeOptions::default()
        },
    );
    for scale in [
        Downscale::None,
        Downscale::Half,
        Downscale::Quarter,
        Downscale::Eighth,
    ] {
        let image = image_at_scale(&bytes, DecodeSettings::default(), scale).expect("image");
        let (width, height) = (image.width(), image.height());
        let mut covered = vec![0_u8; width as usize * height as usize];
        let rects = image.output_tile_rects().expect("tile rects");
        assert!(rects.len() > 1, "{scale:?}");
        for (x, y, w, h) in rects {
            for row in y..y + h {
                for column in x..x + w {
                    covered[(row * width + column) as usize] += 1;
                }
            }
        }
        assert!(covered.iter().all(|&count| count == 1), "{scale:?}");
    }
}

/// A tiled image that streams decodes its tiles on several workers when the
/// decode may use the pool and has a busy tile for each worker, and as one
/// region otherwise.
#[test]
fn tiled_images_stream_tiles_at_once_on_the_pool() {
    let bytes = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 2,
            tile_size: Some((8, 8)),
            ..EncodeOptions::default()
        },
    );
    let image = image_at_scale(&bytes, DecodeSettings::default(), Downscale::None).expect("image");
    let rects = image.output_tile_rects().expect("tile rects");
    assert_eq!(rects.len(), 120);
    assert_eq!(rects[0], (0, 0, 8, 8));
    assert_eq!(rects[119], (88, 72, 8, 8));
    let serial = full_image_route(
        &image,
        request(PixelFormat::Rgb8, 0, CpuDecodeParallelism::Serial),
    );
    assert_eq!(serial, FullImageRoute::StreamedRegion);
    // 8 x 8 tiles are too small to decode on their own.
    let auto = full_image_route(
        &image,
        request(PixelFormat::Rgb8, 0, CpuDecodeParallelism::Auto),
    );
    assert_eq!(auto, FullImageRoute::StreamedRegion);
}

/// Tiles at once take the calling thread and every pool thread, and need a
/// tile for each that keeps them busy, of a size worth its own decode.
#[test]
fn tiles_at_once_need_a_busy_tile_for_every_worker() {
    let grid = |tiles: usize, side: u32| vec![(0, 0, side, side); tiles];
    let workers = |tiles, threads| super::tile_workers(&grid(tiles, 1024), threads);
    assert_eq!(workers(32, 18), Some(19));
    assert_eq!(workers(256, 18), Some(19));
    assert_eq!(workers(32, 6), Some(7));
    // Fewer tiles than workers, or rounds that leave a fifth of them idle.
    assert_eq!(workers(16, 18), None);
    assert_eq!(workers(16, 6), None);
    assert_eq!(workers(20, 18), None);
    // One thread decodes tiles one after another.
    assert_eq!(workers(100, 1), None);
    // Small tiles, or so many that reparsing the tile headers dominates.
    assert_eq!(super::tile_workers(&grid(32, 128), 6), None);
    assert_eq!(super::tile_workers(&grid(32, 256), 6), Some(7));
    assert_eq!(workers(2048, 6), None);
}

/// Without a fixed budget, reduced three-component classic images stream,
/// while HTJ2K, gray and small full-resolution images keep the planes.
#[test]
fn the_default_streams_where_streaming_measured_faster() {
    let options = |use_ht_block_coding| EncodeOptions {
        num_decomposition_levels: 3,
        reversible: false,
        use_ht_block_coding,
        ..EncodeOptions::default()
    };
    let classic = codestream(3, &options(false));
    let ht = codestream(3, &options(true));
    let gray = codestream(1, &options(false));
    let default = |bytes: &[u8], scale, fmt, components, parallelism| {
        let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
        full_image_route(
            &image,
            RouteRequest {
                fmt,
                budget: PlanesBudget::Default,
                parallelism,
                components,
                reduced: scale != Downscale::None,
            },
        )
    };
    for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
        for scale in [Downscale::Half, Downscale::Quarter] {
            for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8] {
                assert!(
                    default(&classic, scale, fmt, 3, parallelism).streams(),
                    "{scale:?} {fmt:?} {parallelism:?}"
                );
                assert_eq!(
                    default(&ht, scale, fmt, 3, parallelism),
                    FullImageRoute::Planes
                );
            }
            assert_eq!(
                default(&gray, scale, PixelFormat::Gray8, 1, parallelism),
                FullImageRoute::Planes
            );
        }
        // 96 x 80 samples are far below a megapixel.
        assert_eq!(
            default(&classic, Downscale::None, PixelFormat::Rgb8, 3, parallelism),
            FullImageRoute::Planes
        );
        assert_eq!(
            default(
                &classic,
                Downscale::Half,
                PixelFormat::Rgb16,
                3,
                parallelism
            ),
            FullImageRoute::Planes
        );
    }
}

/// Decoding the tiles at once on several workers writes the bytes the planes
/// write, at every scale, into padded rows, and only the image's bytes.
#[test]
fn tiles_at_once_write_the_planes_bytes() {
    let bytes = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 3,
            reversible: false,
            tile_size: Some((24, 16)),
            ..EncodeOptions::default()
        },
    );
    for scale in [Downscale::None, Downscale::Half, Downscale::Quarter] {
        let image = image_at_scale(&bytes, DecodeSettings::default(), scale).expect("image");
        let tiles: Vec<Rect> = image
            .output_tile_rects()
            .expect("tile rects")
            .into_iter()
            .map(|(x, y, w, h)| Rect { x, y, w, h })
            .collect();
        for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8] {
            let row = image.width() as usize * fmt.bytes_per_pixel();
            let stride = row + 3;
            let len = stride * image.height() as usize;
            let mut planes = vec![0xab; len];
            crate::decode::decode_full_image_into_with_native_context(
                &image,
                &mut DecoderContext::default(),
                &mut planes,
                stride,
                fmt,
                &FullImageRoute::Planes,
            )
            .expect("planes");
            for workers in [2, 3, 7] {
                let mut streamed = vec![0xab; len];
                super::tiles::decode_tiles_into(
                    &image,
                    &tiles,
                    workers,
                    (&mut streamed, stride, fmt),
                )
                .expect("tiles at once");
                assert!(streamed == planes, "{scale:?} {fmt:?} {workers}");
            }
        }
    }
}

/// A codestream cut short fails the same way however many workers decode
/// its tiles: with the error of the first tile that fails.
#[test]
fn tiles_at_once_report_the_first_failing_tile() {
    let bytes = codestream(
        3,
        &EncodeOptions {
            num_decomposition_levels: 2,
            tile_size: Some((32, 32)),
            ..EncodeOptions::default()
        },
    );
    let truncated = &bytes[..bytes.len() * 3 / 4];
    let Ok(image) = image_at_scale(truncated, DecodeSettings::default(), Downscale::None) else {
        return;
    };
    let tiles: Vec<Rect> = image
        .output_tile_rects()
        .expect("tile rects")
        .into_iter()
        .map(|(x, y, w, h)| Rect { x, y, w, h })
        .collect();
    let stride = image.width() as usize * 3;
    let mut out = vec![0; stride * image.height() as usize];
    let one =
        super::tiles::decode_tiles_into(&image, &tiles, 1, (&mut out, stride, PixelFormat::Rgb8))
            .map_err(|error| error.to_string());
    for workers in [2, 5] {
        let many = super::tiles::decode_tiles_into(
            &image,
            &tiles,
            workers,
            (&mut out, stride, PixelFormat::Rgb8),
        )
        .map_err(|error| error.to_string());
        assert_eq!(one, many, "{workers}");
    }
}
