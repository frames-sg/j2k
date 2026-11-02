// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{full_image_route, FullImageRoute, PlanesBudget, RouteRequest};
use crate::backend::{image_at_scale, Image};
use crate::decode::{
    decode_full_image_into_with_native_context, decode_full_image_on_route,
    decode_native_components_with_native_context, native_components_route,
    native_region_components_route, NativeRoute,
};
use crate::{DecodeSettings, J2kDecoder, J2kError, J2kScratchPool};
use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_native::{encode, CpuDecodeParallelism, EncodeOptions};
use j2k_native::{DecodeError, DecoderContext, DecodingError};

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
        scale: Downscale::None,
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

/// A native decoder context for `parallelism`.
fn context<'a>(parallelism: CpuDecodeParallelism) -> DecoderContext<'a> {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    context
}

/// `image` decoded on `route` into a tight `fmt` buffer.
fn decode_on(
    image: &Image<'_>,
    parallelism: CpuDecodeParallelism,
    fmt: PixelFormat,
    route: &FullImageRoute,
) -> Vec<u8> {
    let stride = image.width() as usize * fmt.bytes_per_pixel();
    let mut out = vec![0; stride * image.height() as usize];
    decode_full_image_into_with_native_context(
        image,
        &mut context(parallelism),
        &mut out,
        stride,
        fmt,
        route,
    )
    .unwrap_or_else(|error| panic!("{fmt:?} {parallelism:?} {route:?}: {error:?}"));
    out
}

/// A tile whose coarse resolutions are empty while finer ones are not (a
/// one-sample tile row or column at an odd position, with two levels)
/// streams like any other, as the region IDWTs reconstruct it from its first
/// level with samples: the planes floor plans it, a budget of zero puts
/// every output on the streamed route, and that route writes what the planes
/// route writes under a raised budget. (These images are too small for any
/// budget but zero to stream them: their header estimate stays far below.)
#[test]
fn tiles_with_empty_coarse_levels_stream_past_the_budget() {
    let gray: &[u8] = include_bytes!(
        "../../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_gray_8x6_tile8x5_53.j2k"
    );
    let rgb: &[u8] = include_bytes!(
        "../../../../j2k-test-support/fixtures/empty_coarse_levels/openjpeg_rgb_6x6_tile5x5_97.j2k"
    );
    for (bytes, formats) in [
        (gray, [PixelFormat::Gray8, PixelFormat::Gray16]),
        (rgb, [PixelFormat::Rgb8, PixelFormat::Rgb16]),
    ] {
        for scale in [Downscale::None, Downscale::Half] {
            let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
            assert!(floor(bytes, scale) > 0, "{scale:?}");
            for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
                let label = format!("{:?} {scale:?} {parallelism:?}", image.color_space());
                for fmt in formats {
                    let streamed = full_image_route(&image, request(fmt, 0, parallelism));
                    assert_eq!(streamed, FullImageRoute::StreamedRegion, "{label} {fmt:?}");
                    let planes = full_image_route(&image, request(fmt, usize::MAX, parallelism));
                    assert_eq!(planes, FullImageRoute::Planes, "{label} {fmt:?}");
                    assert_eq!(
                        decode_on(&image, parallelism, fmt, &streamed),
                        decode_on(&image, parallelism, fmt, &planes),
                        "{label} {fmt:?}"
                    );
                }
                let native = |route| {
                    let decoded = decode_native_components_with_native_context(
                        &image,
                        &mut context(parallelism),
                        None,
                        route,
                    )
                    .unwrap_or_else(|error| panic!("{label} {route:?}: {error:?}"));
                    let planes = decoded.into_parts().planes;
                    planes
                        .into_iter()
                        .map(|plane| {
                            let plane = plane.into_parts();
                            (
                                plane.data,
                                plane.dimensions,
                                plane.bit_depth,
                                plane.signed,
                                plane.bytes_per_sample,
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    native_components_route(&image, 0),
                    NativeRoute::Rows,
                    "{label}"
                );
                assert_eq!(
                    native_components_route(&image, usize::MAX),
                    NativeRoute::Planes,
                    "{label}"
                );
                assert_eq!(
                    native(NativeRoute::Rows),
                    native(NativeRoute::Planes),
                    "{label}"
                );
            }
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

/// Eight 512 x 512 tiles whose tile-parts take turns in the codestream, one
/// RPCL packet each, stream tile by tile at once at half resolution on four
/// threads: the layout the forced-route tests check that route with.
#[test]
fn interleaved_tile_parts_stream_tiles_at_once() {
    let options = EncodeOptions {
        num_decomposition_levels: 2,
        progression_order: j2k_native::EncodeProgressionOrder::Rpcl,
        tile_size: Some((512, 512)),
        tile_part_packet_limit: Some(1),
        write_plt: true,
        ..EncodeOptions::default()
    };
    let bytes = j2k_test_support::with_tile_parts_interleaved(
        &encode(&vec![77; 2048 * 1024], 2048, 1024, 1, 8, false, &options).expect("encode fixture"),
    );
    let image = image_at_scale(&bytes, DecodeSettings::default(), Downscale::Half).expect("image");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("pool");
    let route = pool.install(|| {
        full_image_route(
            &image,
            RouteRequest {
                components: 1,
                scale: Downscale::Half,
                ..request(PixelFormat::Gray8, 0, CpuDecodeParallelism::Auto)
            },
        )
    });
    assert!(
        matches!(route, FullImageRoute::StreamedTiles { ref tiles, .. } if tiles.len() == 8),
        "{route:?}"
    );
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

/// Without a fixed budget, tiny reduced, gray and tiled images and small
/// full-resolution images keep the planes.
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
                scale,
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

/// Full-resolution classic images stream from 4 coded bits per pixel on 1
/// to 11 pool threads from 512 x 512 samples and on larger pools too below
/// 2^19 samples; at fewer bits per pixel from 4096 x 2048 samples on pools
/// of up to 6 threads and of 12 or more.
#[test]
fn full_resolution_images_stream_when_tier1_bound() {
    use super::full_resolution_streams;
    let mosaic = 4096 * 2048;
    // 11 MB lossless and 2.2 MB at 2 bits per pixel.
    assert!(full_resolution_streams(mosaic, 11_104_128, 6));
    // At 2 bits per pixel on one thread and on small and large pools, but
    // not on the 7 to 11 threads where one pool reader holds the decode back.
    for threads in [1, 2, 6, 12, 18] {
        assert!(
            full_resolution_streams(mosaic, 2_246_803, threads),
            "{threads}"
        );
    }
    for threads in [7, 10, 11] {
        assert!(
            !full_resolution_streams(mosaic, 2_246_803, threads),
            "{threads}"
        );
    }
    for threads in [12, 18] {
        assert!(
            !full_resolution_streams(mosaic, 11_104_128, threads),
            "{threads}"
        );
    }
    // 5428 x 4460 at 1.5 bits per pixel streams on one thread, 2896 x 2896
    // at 1.5 keeps the planes on any pool.
    assert!(full_resolution_streams(5428 * 4460, 4_553_929, 1));
    for threads in [1, 6, 18] {
        assert!(
            !full_resolution_streams(2896 * 2896, 1_579_087, threads),
            "{threads}"
        );
    }
    for threads in [1, 2, 11] {
        assert!(
            full_resolution_streams(mosaic, 11_104_128, threads),
            "{threads}"
        );
    }
    // 768 x 512 at 9.6 and 512 x 512 at 6.6 bits per pixel stream on any
    // pool, 4096 x 1024 at 5.9 only below 12 threads.
    for threads in [1, 2, 6, 12, 18] {
        assert!(
            full_resolution_streams(768 * 512, 472_804, threads),
            "{threads}"
        );
        assert!(
            full_resolution_streams(512 * 512, 217_619, threads),
            "{threads}"
        );
    }
    assert!(full_resolution_streams(4096 * 1024, 3_114_650, 1));
    assert!(full_resolution_streams(4096 * 1024, 3_114_650, 6));
    assert!(!full_resolution_streams(4096 * 1024, 3_114_650, 12));
    // No pool threads at all decide as before: from a megapixel at 4 bits per
    // pixel or more, never at fewer.
    assert!(full_resolution_streams(mosaic, 11_104_128, 0));
    assert!(!full_resolution_streams(768 * 512, 472_804, 0));
    assert!(!full_resolution_streams(5428 * 4460, 4_553_929, 0));
    // Below 512 x 512, and 512 x 512 tiles at 2 bits per pixel.
    for threads in [1, 6] {
        assert!(!full_resolution_streams(500 * 500, 250_000, threads));
        assert!(!full_resolution_streams(512 * 384, 118_125, threads));
        assert!(!full_resolution_streams(512 * 512, 64_947, threads));
    }
}

/// Reduced decodes stream from a 1024 x 512 output on any pool, HTJ2K on two
/// threads or more and on one from 1/4; smaller classic outputs from
/// 256 x 256 on 2 to 11 pool threads and from 512 x 256 on one.
#[test]
fn reduced_decodes_stream_when_large_or_on_a_mid_sized_pool() {
    use super::reduced_streams;
    let (large, small) = (1024 * 512, 384 * 256);
    let (half, quarter) = (Downscale::Half, Downscale::Quarter);
    for threads in [1, 6, 18] {
        for scale in [half, quarter, Downscale::Eighth] {
            assert!(reduced_streams(large, scale, false, threads), "{threads}");
            assert!(!reduced_streams(small, scale, true, threads), "{threads}");
        }
    }
    // On one thread HTJ2K streams at 1/4 and 1/8, not at 1/2.
    assert!(!reduced_streams(4 * large, half, true, 1));
    assert!(reduced_streams(large, quarter, true, 1));
    assert!(reduced_streams(large, Downscale::Eighth, true, 1));
    for threads in [2, 6, 11] {
        assert!(reduced_streams(4 * large, half, true, threads), "{threads}");
        assert!(reduced_streams(small, half, false, threads), "{threads}");
    }
    for threads in [12, 18] {
        assert!(!reduced_streams(small, half, false, threads), "{threads}");
    }
    // 256 x 256 streams on 6 threads, 192 x 128 keeps the planes; on one
    // thread from 512 x 256.
    assert!(reduced_streams(256 * 256, half, false, 6));
    assert!(!reduced_streams(192 * 128, half, false, 6));
    assert!(reduced_streams(512 * 256, Downscale::Eighth, false, 1));
    assert!(!reduced_streams(small, half, false, 1));
    assert!(!reduced_streams(256 * 256, half, false, 1));
    // No pool threads at all decide as before.
    assert!(reduced_streams(large, half, false, 0));
    assert!(!reduced_streams(large, quarter, true, 0));
    assert!(!reduced_streams(512 * 256, Downscale::Eighth, false, 0));
}

/// Gray images stream at full resolution from 2896 x 2896 on 2 to 11 pool
/// threads and on larger pools from 5792 x 5792, untiled at full resolution
/// only; on one thread untiled from 1024 x 1024 and reduced from 512 x 256,
/// tiled or not.
#[test]
fn large_gray_images_stream() {
    use super::gray_streams;
    for threads in [0, 1, 6, 11] {
        assert!(
            gray_streams(2896 * 2896, false, false, threads),
            "{threads}"
        );
    }
    for threads in [0, 6, 11] {
        assert!(
            !gray_streams(2048 * 2048, false, false, threads),
            "{threads}"
        );
        assert!(
            !gray_streams(2896 * 2896, true, false, threads),
            "{threads}"
        );
        assert!(
            !gray_streams(5792 * 5792, false, true, threads),
            "{threads}"
        );
    }
    for threads in [12, 18] {
        assert!(
            gray_streams(5792 * 5792, false, false, threads),
            "{threads}"
        );
        assert!(
            !gray_streams(4096 * 4096, false, false, threads),
            "{threads}"
        );
    }
    // One thread: 1024 x 1024 and larger, and reduced outputs from 512 x 256.
    assert!(gray_streams(1024 * 1024, false, false, 1));
    assert!(!gray_streams(1024 * 512, false, false, 1));
    for tiled in [false, true] {
        assert!(gray_streams(1275 * 1650, true, tiled, 1));
        assert!(gray_streams(512 * 256, true, tiled, 1));
        assert!(!gray_streams(256 * 256, true, tiled, 1));
    }
}

/// `bytes`, a one-tile codestream, claiming a `width` x `height` image and
/// tile. Headers only: the route reads them and plans the planes from them.
fn claiming(bytes: &[u8], (width, height): (u32, u32)) -> Vec<u8> {
    let bytes = rewrite_siz(bytes, &[6, 22], |_| width);
    rewrite_siz(&bytes, &[10, 26], |_| height)
}

/// The default route of `bytes`, an image of `components`, at `scale` into
/// RGB8 or Gray8 on one thread, as if the codestream were `coded_bytes` long.
fn one_thread_route(
    bytes: &[u8],
    components: u16,
    scale: Downscale,
    coded_bytes: usize,
) -> FullImageRoute {
    let fmt = if components == 1 {
        PixelFormat::Gray8
    } else {
        PixelFormat::Rgb8
    };
    one_thread_route_to(bytes, components, scale, coded_bytes, fmt)
}

/// [`one_thread_route`] into `fmt`: the same with
/// `CpuDecodeParallelism::Serial` and on a pool of one thread.
fn one_thread_route_to(
    bytes: &[u8],
    components: u16,
    scale: Downscale,
    coded_bytes: usize,
    fmt: PixelFormat,
) -> FullImageRoute {
    let image = image_at_scale(bytes, DecodeSettings::default(), scale).expect("image");
    let route = |parallelism| {
        full_image_route(
            &image,
            RouteRequest {
                fmt,
                budget: PlanesBudget::Default,
                parallelism,
                components,
                coded_bytes,
                scale,
            },
        )
    };
    let serial = route(CpuDecodeParallelism::Serial);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    let one_thread_pool = pool.install(|| route(CpuDecodeParallelism::Auto));
    assert_eq!(serial, one_thread_pool, "{fmt:?} {scale:?}");
    serial
}

/// On one thread the default streams large low-rate and mid-sized classic
/// RGB images, gray images from a megapixel, reduced and tiled images and
/// HTJ2K images from 1/4, and keeps HTJ2K images at full resolution and 1/2,
/// HTJ2K gray images and small images on the planes.
#[test]
fn one_thread_streams_wherever_streaming_is_nearly_as_fast() {
    use FullImageRoute::{Planes, StreamedRegion};
    let options = |use_ht_block_coding| EncodeOptions {
        num_decomposition_levels: 3,
        reversible: false,
        use_ht_block_coding,
        ..EncodeOptions::default()
    };
    let (classic, ht) = (
        codestream(3, &options(false)),
        codestream(3, &options(true)),
    );
    let (gray, gray_ht) = (
        codestream(1, &options(false)),
        codestream(1, &options(true)),
    );
    let full = Downscale::None;
    // RGB at 2 and 1.5 bits per pixel, and at 6.
    let mosaic = claiming(&classic, (4096, 2048));
    assert_eq!(
        one_thread_route_to(&mosaic, 3, full, 2_246_803, PixelFormat::Rgba8),
        StreamedRegion
    );
    assert_eq!(
        one_thread_route(&mosaic, 3, full, 2_246_803),
        StreamedRegion
    );
    let square = claiming(&classic, (2896, 2896));
    assert_eq!(one_thread_route(&square, 3, full, 1_579_087), Planes);
    let mid = claiming(&classic, (1024, 768));
    assert_eq!(one_thread_route(&mid, 3, full, 589_824), StreamedRegion);
    let small = claiming(&classic, (512, 384));
    assert_eq!(one_thread_route(&small, 3, full, 147_456), Planes);
    // HTJ2K keeps the planes at full resolution and at 1/2, and streams
    // from 1/4.
    let ht_mosaic = claiming(&ht, (4096, 2048));
    assert_eq!(one_thread_route(&ht_mosaic, 3, full, 12_236_468), Planes);
    assert_eq!(
        one_thread_route(&ht_mosaic, 3, Downscale::Half, 12_236_468),
        Planes
    );
    assert_eq!(
        one_thread_route(&ht_mosaic, 3, Downscale::Quarter, 2_249_731),
        StreamedRegion
    );
    // Reduced classic outputs from 512 x 256.
    for scale in [Downscale::Quarter, Downscale::Eighth] {
        assert_eq!(
            one_thread_route(&mosaic, 3, scale, 2_246_803),
            StreamedRegion,
            "{scale:?}"
        );
    }
    assert_eq!(
        one_thread_route(&mid, 3, Downscale::Eighth, 589_824),
        Planes
    );
    // Gray: classic from a megapixel and reduced, HTJ2K never.
    let gray_square = claiming(&gray, (1024, 1024));
    assert_eq!(
        one_thread_route(&gray_square, 1, full, 196_608),
        StreamedRegion
    );
    let gray_page = claiming(&gray, (5100, 6600));
    assert_eq!(
        one_thread_route(&gray_page, 1, Downscale::Quarter, 6_319_456),
        StreamedRegion
    );
    let gray_strip = claiming(&gray, (1024, 512));
    assert_eq!(one_thread_route(&gray_strip, 1, full, 98_304), Planes);
    let gray_ht_square = claiming(&gray_ht, (2048, 2048));
    assert_eq!(
        one_thread_route(&gray_ht_square, 1, full, 1_114_149),
        Planes
    );
}

/// On one thread a tiled classic image streams its tiles one after another
/// at full resolution from 256 x 256 tiles, gray or RGB, and an HTJ2K one
/// from 512 x 512 tiles, or from 256 x 256 into 16 bits. A reduced classic
/// decode streams by its output's size whatever the tiles.
#[test]
fn one_thread_streams_tiles_from_256_x_256() {
    let encoded = |components: u16, (width, height): (u32, u32), tile: u32, ht: bool| {
        let pixels = vec![0x80_u8; (width * height) as usize * usize::from(components)];
        let options = EncodeOptions {
            num_decomposition_levels: 3,
            tile_size: Some((tile, tile)),
            use_ht_block_coding: ht,
            ..EncodeOptions::default()
        };
        encode(&pixels, width, height, components, 8, false, &options).expect("encode fixture")
    };
    let flat = |components, size, tile| encoded(components, size, tile, false);
    for components in [3, 1] {
        let bytes = flat(components, (1024, 512), 256);
        let coded = bytes.len();
        for scale in [Downscale::None, Downscale::Half] {
            assert_eq!(
                one_thread_route(&bytes, components, scale, coded),
                FullImageRoute::StreamedRegion,
                "{components} {scale:?}"
            );
        }
        let bytes = flat(components, (512, 256), 128);
        assert_eq!(
            one_thread_route(&bytes, components, Downscale::None, bytes.len()),
            FullImageRoute::Planes,
            "{components}"
        );
    }
    let ht = encoded(3, (1024, 512), 256, true);
    assert_eq!(
        one_thread_route(&ht, 3, Downscale::None, ht.len()),
        FullImageRoute::Planes
    );
    let ht = encoded(3, (1024, 512), 512, true);
    assert_eq!(
        one_thread_route(&ht, 3, Downscale::None, ht.len()),
        FullImageRoute::StreamedRegion
    );
    // 16-bit outputs: HTJ2K tiles from 256 x 256.
    let ht = encoded(3, (1024, 512), 256, true);
    for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16] {
        assert_eq!(
            one_thread_route_to(&ht, 3, Downscale::None, ht.len(), fmt),
            FullImageRoute::StreamedRegion,
            "{fmt:?}"
        );
    }
    let ht = encoded(3, (1024, 512), 128, true);
    assert_eq!(
        one_thread_route_to(&ht, 3, Downscale::None, ht.len(), PixelFormat::Rgb16),
        FullImageRoute::Planes
    );
    // A 512 x 256 output of 128 x 128 tiles streams, a full-resolution
    // decode of them keeps the planes.
    let bytes = flat(3, (1024, 512), 128);
    assert_eq!(
        one_thread_route(&bytes, 3, Downscale::Half, bytes.len()),
        FullImageRoute::StreamedRegion
    );
    assert_eq!(
        one_thread_route(&bytes, 3, Downscale::None, bytes.len()),
        FullImageRoute::Planes
    );
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

/// `bytes` with the SIZ fields at `fields` (byte offsets from the marker)
/// rewritten by `rewrite`.
fn rewrite_siz(bytes: &[u8], fields: &[usize], rewrite: impl Fn(u32) -> u32) -> Vec<u8> {
    let mut bytes = bytes.to_vec();
    let siz = bytes
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    for &field in fields {
        let field = &mut bytes[siz + field..siz + field + 4];
        let value = u32::from_be_bytes(field.try_into().expect("four-byte SIZ field"));
        field.copy_from_slice(&rewrite(value).to_be_bytes());
    }
    bytes
}

/// A region decode into native component planes packs its rows into planes
/// it allocates before decoding only when the region decode streams them,
/// by the check the full-image floor makes: an image of more than four
/// components, or with image and tile offsets, keeps the planes route
/// however large it is, whose decode meets the budget before it allocates
/// image-sized buffers, and the rows decode refuses it.
#[test]
fn native_region_rows_only_for_images_whose_region_decodes_stream() {
    let options = EncodeOptions {
        num_decomposition_levels: 2,
        ..EncodeOptions::default()
    };
    let rgb = codestream(3, &options);
    // The whole image and its tile one tile further on.
    let translated = rewrite_siz(&rgb, &[6, 10, 14, 18, 30, 34], |value| value + 128);
    // One 8192 x 8192 tile, whose native planes alone pass the budget: an
    // image of these components prefers its streamed rows. Headers only;
    // nothing here decodes its packets.
    let claimed = |components| {
        let bytes = codestream(components, &options);
        rewrite_siz(&bytes, &[6, 10, 22, 26], |_| 8192)
    };
    for (label, bytes, streams) in [
        ("rgb", rgb.clone(), true),
        ("rgb with offsets", translated, false),
        ("four components", claimed(4), true),
        ("five components", claimed(5), false),
    ] {
        let image =
            image_at_scale(&bytes, DecodeSettings::default(), Downscale::None).expect("image");
        let (width, height) = (image.width(), image.height());
        assert_eq!(image.streamed_rows_native_precision(), Some(8), "{label}");
        assert_eq!(image.region_decode_streams_rows(), streams, "{label}");
        assert_eq!(
            image.full_decode_planes_floor(0).expect("plan").is_some(),
            streams,
            "{label}"
        );
        let expected = if streams {
            NativeRoute::Rows
        } else {
            NativeRoute::Planes
        };
        assert_eq!(
            native_region_components_route(&image, Rect::full((width, height))),
            expected,
            "{label}"
        );
        if !streams {
            let Err(error) = image.decode_native_region_components_rows_with_context(
                (0, 0, width, height),
                &mut DecoderContext::default(),
            ) else {
                panic!("{label}: the rows decode allocated its planes and decoded");
            };
            assert!(
                matches!(
                    error,
                    DecodeError::Decoding(DecodingError::UnsupportedFeature(_))
                ),
                "{label}: {error:?}"
            );
        }
    }
}

/// On one thread a 16-bit output streams from 512 x 512 samples whatever the
/// coding, untiled or in tiles of 256 x 256 or more, where an 8-bit one of
/// a full-resolution HTJ2K image keeps the planes.
#[test]
fn one_thread_streams_sixteen_bit_outputs_from_512_x_512() {
    use FullImageRoute::{Planes, StreamedRegion};
    let options = |use_ht_block_coding| EncodeOptions {
        num_decomposition_levels: 3,
        reversible: false,
        use_ht_block_coding,
        ..EncodeOptions::default()
    };
    let (classic, ht) = (
        codestream(3, &options(false)),
        codestream(3, &options(true)),
    );
    let gray_ht = codestream(1, &options(true));
    let full = Downscale::None;
    let ht_mosaic = claiming(&ht, (4096, 2048));
    for fmt in [PixelFormat::Rgb16, PixelFormat::Rgba16] {
        assert_eq!(
            one_thread_route_to(&ht_mosaic, 3, full, 2_249_731, fmt),
            StreamedRegion,
            "{fmt:?}"
        );
    }
    assert_eq!(one_thread_route(&ht_mosaic, 3, full, 2_249_731), Planes);
    // 1.5 bits per pixel below 4096 x 2048, and 512 x 512 at 6.
    let square = claiming(&classic, (2896, 2896));
    assert_eq!(
        one_thread_route_to(&square, 3, full, 1_579_087, PixelFormat::Rgb16),
        StreamedRegion
    );
    let tile = claiming(&classic, (512, 512));
    assert_eq!(
        one_thread_route_to(&tile, 3, full, 196_608, PixelFormat::Rgb16),
        StreamedRegion
    );
    let small = claiming(&classic, (512, 384));
    assert_eq!(
        one_thread_route_to(&small, 3, full, 147_456, PixelFormat::Rgb16),
        Planes
    );
    let gray_ht_square = claiming(&gray_ht, (2048, 2048));
    assert_eq!(
        one_thread_route_to(&gray_ht_square, 1, full, 1_114_149, PixelFormat::Gray16),
        StreamedRegion
    );
    // The reduced output counts: 1/2 of 1024 x 512 is below 512 x 512.
    let strip = claiming(&ht, (1024, 512));
    assert_eq!(
        one_thread_route_to(&strip, 3, Downscale::Half, 1_000_000, PixelFormat::Rgb16),
        Planes
    );
}

/// `bytes`, a one-tile codestream of one coding style with the largest
/// precincts, as a `width` x `height` image whose packets are all empty: its
/// main header with the image and tile resized, then one tile-part of a zero
/// byte per packet. It decodes to mid-level samples whatever its size,
/// without a code-block to decode.
fn with_empty_packets(bytes: &[u8], (width, height): (u32, u32)) -> Vec<u8> {
    let (mut end, mut levels, mut components) = (2, None, None);
    while bytes[end..end + 2] != [0xff, 0x90] {
        let length = usize::from(u16::from_be_bytes([bytes[end + 2], bytes[end + 3]]));
        match bytes[end + 1] {
            0x51 => components = Some(u16::from_be_bytes([bytes[end + 38], bytes[end + 39]])),
            0x52 => {
                assert_eq!(bytes[end + 4] & 1, 0, "the largest precincts");
                levels = Some(u32::from(bytes[end + 9]));
            }
            0x53 => panic!("one coding style"),
            _ => {}
        }
        end += 2 + length;
    }
    let (levels, components) = (levels.expect("COD"), components.expect("SIZ"));
    let precincts = |extent: u32| u64::from(extent.div_ceil(1 << 15));
    let per_component: u64 = (0..=levels)
        .map(|shift| precincts(width.div_ceil(1 << shift)) * precincts(height.div_ceil(1 << shift)))
        .sum();
    let packets = usize::try_from(per_component * u64::from(components)).expect("packets");
    let mut out = claiming(&bytes[..end], (width, height));
    let tile_part = u32::try_from(14 + packets).expect("tile-part length");
    out.extend_from_slice(&[0xff, 0x90, 0, 10, 0, 0]);
    out.extend_from_slice(&tile_part.to_be_bytes());
    out.extend_from_slice(&[0, 1, 0xff, 0x93]);
    out.resize(out.len() + packets, 0);
    out.extend_from_slice(&[0xff, 0xd9]);
    out
}

/// The default request for a full-resolution decode of `bytes`, an image of
/// `components`, into `fmt` with `parallelism`.
fn default_request(
    bytes: &[u8],
    components: u16,
    fmt: PixelFormat,
    parallelism: CpuDecodeParallelism,
) -> RouteRequest {
    RouteRequest {
        fmt,
        budget: PlanesBudget::Default,
        parallelism,
        components,
        coded_bytes: bytes.len(),
        scale: Downscale::None,
    }
}

/// A wide, shallow RGB image of one 5/3 level in 1024 x 4 code-blocks
/// without a multiple-component transform, of `width` x `height`.
fn wide_and_shallow(width: u32, height: u32) -> Vec<u8> {
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        code_block_width_exp: 8,
        code_block_height_exp: 0,
        use_mct: false,
        ..EncodeOptions::default()
    };
    with_empty_packets(&codestream(3, &options), (width, height))
}

/// A wide, shallow image whose planes fit the decode budget while its
/// streamed rows' line buffers do not (8,388,608 x 1 RGB) is preferred for
/// streaming on one thread and on a pool of six threads. Its stream is
/// refused by the budget before it decodes, and the decode then runs on the
/// planes and writes their bytes.
#[test]
fn a_preferred_stream_past_the_budget_decodes_on_the_planes() {
    use CpuDecodeParallelism::{Auto, Serial};
    let width = 1_u32 << 23;
    let bytes = wide_and_shallow(width, 1);
    let image = image_at_scale(&bytes, DecodeSettings::default(), Downscale::None).expect("image");
    let planes_floor = image.full_decode_planes_floor(0).expect("plan");
    assert!(
        planes_floor.is_some_and(|floor| floor <= j2k_native::DEFAULT_MAX_DECODE_BYTES),
        "{planes_floor:?}"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(6)
        .build()
        .expect("pool");
    for fmt in [PixelFormat::Rgb8, PixelFormat::Rgb16] {
        let request = default_request(&bytes, 3, fmt, Serial);
        let route = full_image_route(&image, request);
        assert_eq!(route, FullImageRoute::StreamedRegion, "{fmt:?}");
        let on_the_pool =
            pool.install(|| full_image_route(&image, default_request(&bytes, 3, fmt, Auto)));
        assert_eq!(on_the_pool, FullImageRoute::StreamedRegion, "{fmt:?}");
        let stride = width as usize * fmt.bytes_per_pixel();
        let mut out = vec![0; stride];
        let refused = decode_full_image_into_with_native_context(
            &image,
            &mut context(Serial),
            &mut out,
            stride,
            fmt,
            &route,
        );
        assert!(
            matches!(refused, Err(J2kError::ResourceLimit { .. })),
            "{fmt:?}: {refused:?}"
        );
        let planes = decode_on(&image, Serial, fmt, &FullImageRoute::Planes);
        decode_full_image_on_route(
            &image,
            &mut context(Serial),
            &mut out,
            stride,
            request,
            &route,
        )
        .unwrap_or_else(|error| panic!("{fmt:?}: {error:?}"));
        assert!(out == planes, "{fmt:?}");
        // The decoder's own full-image decode, on one thread and on the pool.
        for parallelism in [Some(crate::CpuDecodeParallelism::Serial), None] {
            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            if let Some(parallelism) = parallelism {
                decoder.set_cpu_decode_parallelism(parallelism);
            }
            let mut out = vec![0; stride];
            pool.install(|| decoder.decode_into(&mut out, stride, fmt))
                .unwrap_or_else(|error| panic!("{fmt:?} {parallelism:?}: {error:?}"));
            assert!(out == planes, "{fmt:?} {parallelism:?}");
        }
    }
}

/// A wide, shallow image whose streamed rows fit the budget keeps its
/// preferred stream on one thread, and the stream writes the planes' bytes:
/// 65,536 x 4 RGB into 16 bits, and 1,048,576 x 8 RGB at a low rate.
#[test]
fn a_wide_image_whose_stream_fits_streams() {
    use CpuDecodeParallelism::Serial;
    for ((width, height), fmt) in [
        ((1 << 16, 4), PixelFormat::Rgb16),
        ((1 << 20, 8), PixelFormat::Rgb8),
    ] {
        let bytes = wide_and_shallow(width, height);
        let image =
            image_at_scale(&bytes, DecodeSettings::default(), Downscale::None).expect("image");
        let route = full_image_route(&image, default_request(&bytes, 3, fmt, Serial));
        assert_eq!(route, FullImageRoute::StreamedRegion, "{width} {fmt:?}");
        let streamed = decode_on(&image, Serial, fmt, &route);
        let planes = decode_on(&image, Serial, fmt, &FullImageRoute::Planes);
        assert!(streamed == planes, "{width} {fmt:?}");
    }
}

/// Rows of `components` 8-bit samples of a `width` x `height` image with
/// edges and texture.
#[expect(
    clippy::cast_possible_truncation,
    reason = "each sample keeps the low byte of its pattern on purpose"
)]
fn textured(components: u16, (width, height): (u32, u32)) -> Vec<u8> {
    let components = u32::from(components);
    (0..width * height * components)
        .map(|index| {
            let (pixel, component) = (index / components, index % components);
            let (x, y) = (pixel % width, pixel / width);
            ((x * 3 + y * 5) ^ ((x * y) >> 6)).wrapping_add(component * 85) as u8
        })
        .collect()
}

/// On one thread the decoder's default full-image and reduced decodes of
/// each kind of image that streams there write the planes' bytes: mid-sized
/// RGB, gray from a megapixel, RGB and gray in tiles of 256 x 256, reduced
/// RGB untiled and in tiles, reduced gray, HTJ2K RGB at 1/4, and HTJ2K RGB
/// into 16 bits untiled and in tiles of 256 x 256 (large low-rate RGB in
/// `a_wide_image_whose_stream_fits_streams`).
#[test]
fn one_thread_default_streams_write_the_planes_bytes() {
    let encoded = |components: u16, size: (u32, u32), tile: Option<u32>, ht: bool| {
        let options = EncodeOptions {
            num_decomposition_levels: 3,
            reversible: false,
            tile_size: tile.map(|tile| (tile, tile)),
            use_ht_block_coding: ht,
            ..EncodeOptions::default()
        };
        let pixels = textured(components, size);
        encode(&pixels, size.0, size.1, components, 8, false, &options).expect("encode fixture")
    };
    let rgb = |size, tile, ht| (encoded(3, size, tile, ht), 3);
    let gray = |size, tile| (encoded(1, size, tile, false), 1);
    let ht_mosaic = with_empty_packets(&encoded(3, (96, 80), None, true), (4096, 2048));
    let (full, half, quarter) = (Downscale::None, Downscale::Half, Downscale::Quarter);
    let (rgb8, rgb16, gray8) = (PixelFormat::Rgb8, PixelFormat::Rgb16, PixelFormat::Gray8);
    let cases = [
        ("RGB", rgb((512, 512), None, false), full, rgb8),
        ("gray", gray((1024, 1024), None), full, gray8),
        ("RGB tiles", rgb((512, 512), Some(256), false), full, rgb8),
        ("gray tiles", gray((512, 512), Some(256)), full, gray8),
        ("reduced RGB", rgb((1024, 512), None, false), half, rgb8),
        ("1/2 tiles", rgb((1024, 512), Some(128), false), half, rgb8),
        ("reduced gray", gray((1024, 512), None), half, gray8),
        ("HTJ2K at 1/4", (ht_mosaic, 3), quarter, rgb8),
        ("HTJ2K 16-bit", rgb((512, 512), None, true), full, rgb16),
        (
            "16-bit tiles",
            rgb((1024, 512), Some(256), true),
            full,
            rgb16,
        ),
    ];
    for (label, (bytes, components), scale, fmt) in cases {
        assert_eq!(
            one_thread_route_to(&bytes, components, scale, bytes.len(), fmt),
            FullImageRoute::StreamedRegion,
            "{label}"
        );
        let decode = |planes_budget: Option<usize>| {
            let mut decoder = J2kDecoder::new(&bytes).expect("decoder");
            decoder.set_cpu_decode_parallelism(crate::CpuDecodeParallelism::Serial);
            if let Some(bytes) = planes_budget {
                decoder.set_full_decode_planes_budget(bytes);
            }
            let (width, height) = decoder.info().dimensions;
            let denominator = scale.denominator();
            let (width, height) = (width.div_ceil(denominator), height.div_ceil(denominator));
            let stride = width as usize * fmt.bytes_per_pixel();
            let mut out = vec![0; stride * height as usize];
            decoder
                .decode_scaled_into(&mut J2kScratchPool::new(), &mut out, stride, fmt, scale)
                .unwrap_or_else(|error| panic!("{label}: {error:?}"));
            out
        };
        assert!(decode(None) == decode(Some(usize::MAX)), "{label}");
    }
}
