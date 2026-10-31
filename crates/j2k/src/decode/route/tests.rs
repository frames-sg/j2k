// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{full_image_route, FullImageRoute, PlanesBudget, RouteRequest};
use crate::backend::image_at_scale;
use crate::decode::{native_components_route, NativeRoute};
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
        coded_bytes: 0,
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

/// A 16-bit decode streams when the planes decode or the pack of its native
/// samples, which holds every `f32` plane besides them, would pass the
/// budget. With small tiles and 16-bit samples the pack charges more than
/// the decode.
#[test]
fn sixteen_bit_output_streams_when_its_planes_or_their_pack_pass_the_budget() {
    let (width, height) = (96_u32, 80_u32);
    let samples: Vec<u8> = (0..width * height * 3 * 2)
        .map(|index| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    let options = EncodeOptions {
        num_decomposition_levels: 2,
        tile_size: Some((16, 16)),
        ..EncodeOptions::default()
    };
    let bytes = encode(&samples, width, height, 3, 16, false, &options).expect("encode fixture");
    let floor = floor(&bytes, Downscale::None);
    let pack = 96 * 80 * 3 * (4 + 2);
    assert!(pack > floor, "{pack} > {floor}");
    for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16] {
        assert_eq!(
            route(&bytes, Downscale::None, fmt, pack - 1),
            FullImageRoute::StreamedRegion
        );
        assert_eq!(
            route(&bytes, Downscale::None, fmt, pack),
            FullImageRoute::Planes
        );
    }
    // The 8-bit planes write the output directly, without a pack.
    assert_eq!(
        route(&bytes, Downscale::None, PixelFormat::Rgb8, pack - 1),
        FullImageRoute::Planes
    );
    // An RGB image has no 16-bit gray layout; the planes report that.
    assert_eq!(
        route(&bytes, Downscale::None, PixelFormat::Gray16, 0),
        FullImageRoute::Planes
    );
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

/// Without a fixed budget, tiny reduced images, gray, reduced tiled and
/// small full-resolution images keep the planes.
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
    let tiled = codestream(
        3,
        &EncodeOptions {
            tile_size: Some((48, 40)),
            ..options(false)
        },
    );
    let default = |bytes: &[u8], scale, fmt, components, parallelism| {
        let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
        full_image_route(
            &image,
            RouteRequest {
                fmt,
                budget: PlanesBudget::Default,
                parallelism,
                components,
                coded_bytes: bytes.len(),
                reduced: scale != Downscale::None,
            },
        )
    };
    for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
        for scale in [Downscale::Half, Downscale::Quarter] {
            for fmt in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Gray8] {
                // 48 x 40 and 24 x 20 outputs are below 256 x 256.
                assert_eq!(
                    default(&classic, scale, fmt, 3, parallelism),
                    FullImageRoute::Planes,
                    "{scale:?} {fmt:?} {parallelism:?}"
                );
                assert_eq!(
                    default(&ht, scale, fmt, 3, parallelism),
                    FullImageRoute::Planes,
                    "{scale:?} {fmt:?} {parallelism:?}"
                );
            }
            assert_eq!(
                default(&gray, scale, PixelFormat::Gray8, 1, parallelism),
                FullImageRoute::Planes
            );
            assert_eq!(
                default(&tiled, scale, PixelFormat::Rgb8, 3, parallelism),
                FullImageRoute::Planes
            );
        }
        // 96 x 80 samples are far below a megapixel, tiled or not, and below
        // the smallest gray image that streams.
        for bytes in [classic.as_slice(), ht.as_slice(), tiled.as_slice()] {
            assert_eq!(
                default(bytes, Downscale::None, PixelFormat::Rgb8, 3, parallelism),
                FullImageRoute::Planes
            );
        }
        assert_eq!(
            default(&gray, Downscale::None, PixelFormat::Gray8, 1, parallelism),
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

/// Full-resolution classic images stream from 4 coded bits per pixel on 2
/// to 11 pool threads from 512 x 512 samples, on larger pools too below 2^19
/// samples, and on one thread from a megapixel.
#[test]
fn full_resolution_images_stream_when_tier1_bound() {
    use super::full_resolution_streams;
    let mosaic = 4096 * 2048;
    // 11 MB lossless and 2.2 MB at 2 bits per pixel.
    assert!(full_resolution_streams(mosaic, 11_104_128, 6));
    assert!(!full_resolution_streams(mosaic, 2_246_803, 6));
    for threads in [12, 18] {
        assert!(
            !full_resolution_streams(mosaic, 11_104_128, threads),
            "{threads}"
        );
    }
    // One thread: from a megapixel.
    assert!(full_resolution_streams(mosaic, 11_104_128, 1));
    assert!(!full_resolution_streams(mosaic, 2_246_803, 1));
    assert!(full_resolution_streams(mosaic, 11_104_128, 2));
    assert!(full_resolution_streams(mosaic, 11_104_128, 11));
    // 768 x 512 at 9.6 and 512 x 512 at 6.6 bits per pixel stream on any
    // pool, 4096 x 1024 at 5.9 only below 12 threads.
    for threads in [2, 6, 12, 18] {
        assert!(
            full_resolution_streams(768 * 512, 472_804, threads),
            "{threads}"
        );
        assert!(
            full_resolution_streams(512 * 512, 217_619, threads),
            "{threads}"
        );
    }
    assert!(full_resolution_streams(4096 * 1024, 3_114_650, 6));
    assert!(!full_resolution_streams(4096 * 1024, 3_114_650, 12));
    assert!(!full_resolution_streams(768 * 512, 472_804, 1));
    // Below 512 x 512, and 512 x 512 tiles at 2 bits per pixel.
    assert!(!full_resolution_streams(500 * 500, 250_000, 6));
    assert!(!full_resolution_streams(512 * 512, 64_947, 6));
}

/// Reduced decodes stream from a 1024 x 512 output on any pool (HTJ2K on
/// two threads or more); smaller classic outputs from 256 x 256 on 2 to 11
/// pool threads.
#[test]
fn reduced_decodes_stream_when_large_or_on_a_mid_sized_pool() {
    use super::reduced_streams;
    let (large, small) = (1024 * 512, 384 * 256);
    for threads in [1, 6, 18] {
        assert!(reduced_streams(large, false, threads), "{threads}");
        assert!(!reduced_streams(small, true, threads), "{threads}");
    }
    assert!(!reduced_streams(large, true, 1));
    for threads in [2, 6, 11] {
        assert!(reduced_streams(large, true, threads), "{threads}");
        assert!(reduced_streams(small, false, threads), "{threads}");
    }
    for threads in [1, 12, 18] {
        assert!(!reduced_streams(small, false, threads), "{threads}");
    }
    // 256 x 256 streams on 6 threads, 192 x 128 keeps the planes.
    assert!(reduced_streams(256 * 256, false, 6));
    assert!(!reduced_streams(192 * 128, false, 6));
}

/// Gray images stream from 2896 x 2896 below 12 pool threads, and on larger
/// pools from 5792 x 5792.
#[test]
fn large_gray_images_stream() {
    use super::gray_streams;
    for threads in [1, 6, 11] {
        assert!(gray_streams(2896 * 2896, threads), "{threads}");
        assert!(!gray_streams(2048 * 2048, threads), "{threads}");
    }
    for threads in [12, 18] {
        assert!(gray_streams(5792 * 5792, threads), "{threads}");
        assert!(!gray_streams(4096 * 4096, threads), "{threads}");
    }
}

/// Images of more than 8 bits and signed images stream at 16 bits and into
/// native component planes once their planes pass the budget.
#[test]
fn deep_and_signed_images_stream_past_the_budget() {
    let (width, height) = (64_u32, 48_u32);
    for (components, bit_depth, signed, reversible) in [
        (3, 12, true, false),
        (3, 16, false, true),
        (1, 12, false, true),
        (1, 16, true, false),
        (1, 4, false, true),
        (1, 20, false, true),
        (3, 24, true, false),
    ] {
        let bytes_per_sample = u32::from(bit_depth).div_ceil(8);
        let samples: Vec<u8> = (0..width * height * u32::from(components) * bytes_per_sample)
            .map(|index| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let options = EncodeOptions {
            num_decomposition_levels: 3,
            reversible,
            ..EncodeOptions::default()
        };
        let bytes = encode(
            &samples, width, height, components, bit_depth, signed, &options,
        )
        .expect("encode fixture");
        let image =
            image_at_scale(&bytes, DecodeSettings::default(), Downscale::None).expect("image");
        assert_eq!(image.streamed_rows_native_precision(), Some(bit_depth));
        let fmt = if components == 1 {
            PixelFormat::Gray16
        } else {
            PixelFormat::Rgb16
        };
        let label = format!("{components} x {bit_depth} bits, signed {signed}");
        assert_eq!(
            full_image_route(&image, request(fmt, 0, CpuDecodeParallelism::Serial)),
            FullImageRoute::StreamedRegion,
            "{label}"
        );
        assert_eq!(
            native_components_route(&image, 0),
            NativeRoute::Rows,
            "{label}"
        );
        let budget = j2k_native::DEFAULT_MAX_DECODE_BYTES;
        assert_eq!(
            full_image_route(&image, request(fmt, budget, CpuDecodeParallelism::Serial)),
            FullImageRoute::Planes
        );
        assert_eq!(native_components_route(&image, budget), NativeRoute::Planes);
    }
}
