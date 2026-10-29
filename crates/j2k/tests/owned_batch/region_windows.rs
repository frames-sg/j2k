// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region requests decode through prepared plans sized to the region rather
//! than to whole resolution levels.

use std::{num::NonZeroUsize, sync::Arc};

use j2k::{
    BatchDecodeOptions, BatchLayout, CpuBatchDecoder, DecodeRequest, Downscale, EncodedImage,
    PreparationDepth, Rect,
};
use j2k_native::{encode, encode_htj2k, EncodeOptions};
use j2k_test_support::Htj2kRgbaAlpha;

use super::fixtures::{wrap_rgba_jp2, wrap_rgba_jph};
use super::oracles::{decoded_samples_for_source, native_request_oracle};

/// Neutral RGB codestream re-declared as one 18000 x 12000 tile. Every packet
/// is empty, so it is cheap to build and decode, but a whole level-0 plane is
/// far larger than the default decode allocation limit.
fn empty_large_rgb_codestream(ht: bool) -> Vec<u8> {
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 5,
        ..EncodeOptions::default()
    };
    let codec = if ht { encode_htj2k } else { encode };
    let mut bytes =
        codec(&[128; 32 * 32 * 3], 32, 32, 3, 8, false, &options).expect("encode neutral RGB8");
    let siz = bytes
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ");
    for (offset, value) in [(6, 18_000_u32), (10, 12_000), (22, 18_000), (26, 12_000)] {
        bytes[siz + offset..siz + offset + 4].copy_from_slice(&value.to_be_bytes());
    }
    bytes
}

/// Deterministic samples with smooth gradients, edges, and per-component
/// variation, so every sub-band carries coefficients.
fn textured_samples(width: u32, height: u32, components: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).flat_map(move |x| (0..components).map(move |c| (x, y, c))))
        .map(|(x, y, c)| {
            let gradient = x * 3 + y * 2 + c * 61;
            let edge = if (x / 23 + y / 17 + c) % 2 == 0 {
                0
            } else {
                97
            };
            let noise = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ c) >> 7;
            u8::try_from((gradient + edge + noise % 29) % 256).expect("byte sample")
        })
        .collect()
}

/// One worker runs each image's plan on its own; several workers share
/// entropy decoding across the batch before each image's IDWT.
const EXECUTION_MODES: [(usize, BatchLayout); 2] = [(1, BatchLayout::Nchw), (4, BatchLayout::Nhwc)];

fn decode_and_compare(encoded: &Arc<[u8]>, requests: &[DecodeRequest], case: &str) {
    for (workers, layout) in EXECUTION_MODES {
        let case = format!("{case} workers={workers} {layout:?}");
        decode_and_compare_with(encoded, requests, &case, NonZeroUsize::new(workers), layout);
    }
}

fn decode_and_compare_with(
    encoded: &Arc<[u8]>,
    requests: &[DecodeRequest],
    case: &str,
    workers: Option<NonZeroUsize>,
    layout: BatchLayout,
) {
    let mut options = BatchDecodeOptions::default();
    options.workers = workers;
    options.layout = layout;
    let mut decoder = CpuBatchDecoder::new(options);
    let prepared = decoder
        .prepare(
            requests
                .iter()
                .map(|&request| EncodedImage::new(Arc::clone(encoded), request))
                .collect(),
        )
        .unwrap_or_else(|error| panic!("{case}: prepare: {error}"));
    assert!(
        prepared.errors().is_empty(),
        "{case}: {:?}",
        prepared.errors()
    );
    let images = prepared
        .groups()
        .iter()
        .flat_map(j2k::PreparedBatchGroup::images)
        .collect::<Vec<_>>();
    assert!(
        images
            .iter()
            .all(|image| image.preparation_depth() != PreparationDepth::MetadataOnly),
        "{case}: every request must keep a prepared offset plan"
    );

    let result = decoder
        .decode_prepared(&prepared)
        .unwrap_or_else(|error| panic!("{case}: decode: {error}"));
    assert!(result.errors().is_empty(), "{case}: {:?}", result.errors());
    for image in images {
        let source_index = image.source_index();
        assert_eq!(
            decoded_samples_for_source(&result, source_index),
            native_request_oracle(image, layout),
            "{case} request={:?}",
            requests[source_index]
        );
    }
}

#[test]
fn cpu_batch_decodes_small_regions_of_very_large_images() {
    // A whole level-0 plane of this image needs 864 MB per component; the
    // region needs a few code-blocks.
    let roi = Rect {
        x: 8_501,
        y: 5_503,
        w: 16,
        h: 16,
    };
    let requests = [
        DecodeRequest::Region { roi },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];
    for ht in [false, true] {
        let encoded = Arc::<[u8]>::from(empty_large_rgb_codestream(ht));
        decode_and_compare(&encoded, &requests, &format!("large ht={ht}"));
    }
}

#[test]
fn region_plans_keep_whole_level_geometry_for_every_backend() {
    // Metal and CUDA execute the prepared geometry as whole levels; only the
    // CPU executors run the region-sized copy.
    let roi = Rect {
        x: 8_501,
        y: 5_503,
        w: 16,
        h: 16,
    };
    for ht in [false, true] {
        let encoded = Arc::<[u8]>::from(empty_large_rgb_codestream(ht));
        let mut options = BatchDecodeOptions::default();
        options.workers = NonZeroUsize::new(1);
        let prepared = CpuBatchDecoder::new(options)
            .prepare(vec![EncodedImage::new(
                encoded,
                DecodeRequest::Region { roi },
            )])
            .expect("prepare large region");
        let image = &prepared.groups()[0].images()[0];
        let tiles = image.htj2k_plan().map_or_else(
            || {
                image
                    .classic_plan()
                    .expect("classic offset plan")
                    .geometry()
                    .tiles()
            },
            |plan| plan.geometry().tiles(),
        );
        let components = &tiles[0].color_geometry().expect("RGB tile").component_plans;
        for component in components {
            let last_idwt = component
                .steps
                .iter()
                .rev()
                .find_map(|step| match step {
                    j2k_native::J2kDirectGrayscaleStep::Idwt(idwt) => Some(idwt.rect),
                    _ => None,
                })
                .expect("level-0 IDWT");
            assert_eq!(
                (last_idwt.width(), last_idwt.height()),
                (18_000, 12_000),
                "ht={ht}: shared geometry must keep whole levels"
            );
        }
    }
}

#[test]
fn cpu_batch_region_windows_match_single_image_decode() {
    const WIDTH: u32 = 389;
    const HEIGHT: u32 = 263;
    let regions = [
        (0, 0, 1, 1),
        (WIDTH - 1, HEIGHT - 1, 1, 1),
        (37, 19, 64, 41),
        (1, 1, WIDTH - 2, HEIGHT - 2),
        (150, 0, 17, HEIGHT),
        (0, 131, WIDTH, 3),
        (201, 97, 63, 65),
        (0, 0, WIDTH, HEIGHT),
    ];
    let requests = regions
        .iter()
        .flat_map(|&(x, y, w, h)| {
            let roi = Rect { x, y, w, h };
            [
                DecodeRequest::Region { roi },
                DecodeRequest::RegionReduced {
                    roi,
                    scale: Downscale::Half,
                },
                DecodeRequest::RegionReduced {
                    roi,
                    scale: Downscale::Quarter,
                },
            ]
        })
        .collect::<Vec<_>>();

    for components in [1_u16, 3] {
        let samples = textured_samples(WIDTH, HEIGHT, u32::from(components));
        for ht in [false, true] {
            for reversible in [true, false] {
                // Exponents are stored minus 2: 4 is 64 x 64, 2 is 16 x 16.
                for (tile_size, code_block_exp) in [(None, 4), (None, 2), (Some((128, 96)), 2)] {
                    let options = EncodeOptions {
                        reversible,
                        num_decomposition_levels: 5,
                        code_block_width_exp: code_block_exp,
                        code_block_height_exp: code_block_exp,
                        tile_size,
                        use_mct: components == 3,
                        ..EncodeOptions::default()
                    };
                    let codec = if ht { encode_htj2k } else { encode };
                    let encoded = Arc::<[u8]>::from(
                        codec(&samples, WIDTH, HEIGHT, components, 8, false, &options)
                            .expect("encode textured fixture"),
                    );
                    let case = format!(
                        "components={components} ht={ht} reversible={reversible} \
                         tiles={tile_size:?} code_block_exp={code_block_exp}"
                    );
                    decode_and_compare(&encoded, &requests, &case);
                }
            }
        }
    }
}

#[test]
fn cpu_batch_region_windows_match_single_image_decode_with_three_levels() {
    // The fixture and regions j2k-metal checks its region batches against.
    let (width, height) = (300_u32, 260_u32);
    let regions = [
        ((0, 0, 40, 36), Downscale::None),
        ((131, 97, 23, 41), Downscale::None),
        ((251, 213, 49, 47), Downscale::None),
        ((100, 64, 96, 80), Downscale::Half),
        ((37, 150, 120, 110), Downscale::Quarter),
    ];
    let requests = regions
        .iter()
        .map(|&((x, y, w, h), scale)| {
            let roi = Rect { x, y, w, h };
            if scale == Downscale::None {
                DecodeRequest::Region { roi }
            } else {
                DecodeRequest::RegionReduced { roi, scale }
            }
        })
        .collect::<Vec<_>>();
    // Four components exercise the RGBA plan, with explicit alpha.
    for components in [1_u16, 3, 4] {
        let pixels: Vec<u8> = (0..width * height * u32::from(components))
            .map(|index| {
                let pixel = index / u32::from(components);
                let (x, y) = (pixel % width, pixel / width);
                u8::try_from((x * 7 + y * 13 + (x * y) / 5 + index % 3 * 41) & 0xff)
                    .expect("masked")
            })
            .collect();
        for ht in [false, true] {
            for reversible in [true, false] {
                let options = EncodeOptions {
                    reversible,
                    use_mct: components >= 3,
                    num_decomposition_levels: 3,
                    code_block_width_exp: 4,
                    code_block_height_exp: 4,
                    ..EncodeOptions::default()
                };
                let codec = if ht { encode_htj2k } else { encode };
                let codestream = codec(&pixels, width, height, components, 8, false, &options)
                    .expect("encode three-level fixture");
                let encoded = Arc::<[u8]>::from(match (components, ht) {
                    (4, true) => wrap_rgba_jph(&codestream, Htj2kRgbaAlpha::Straight),
                    (4, false) => wrap_rgba_jp2(&codestream, Htj2kRgbaAlpha::Straight),
                    _ => codestream,
                });
                let case = format!("components={components} ht={ht} reversible={reversible}");
                decode_and_compare(&encoded, &requests, &case);
            }
        }
    }
}

#[test]
#[ignore = "release large-region CPU batch characterization; requires J2K_LARGE_REGION_INPUT"]
fn local_large_region_cpu_batch_characterization() {
    use j2k::{CpuBatchSamples, CpuDecodeParallelism, J2kDecoder, J2kScratchPool, PixelFormat};

    let path = std::env::var_os("J2K_LARGE_REGION_INPUT").expect("trusted large codestream");
    let bytes: Arc<[u8]> = std::fs::read(path).expect("large codestream").into();
    for (reduction, scale) in [Downscale::None, Downscale::Half, Downscale::Quarter]
        .into_iter()
        .enumerate()
    {
        let side = 1024 * scale.denominator();
        let rois: Vec<_> = (0..4)
            .map(|index| Rect {
                x: 2048 + (index % 2) * side,
                y: 1024 + (index / 2) * side,
                w: side,
                h: side,
            })
            .collect();
        let inputs: Vec<_> = rois
            .iter()
            .map(|&roi| {
                let request = if scale == Downscale::None {
                    DecodeRequest::Region { roi }
                } else {
                    DecodeRequest::RegionReduced { roi, scale }
                };
                EncodedImage::new(bytes.clone(), request)
            })
            .collect();
        for sample in 0..7 {
            let mut options = BatchDecodeOptions::default();
            options.layout = BatchLayout::Nhwc;
            let mut decoder = CpuBatchDecoder::new(options);
            let started = std::time::Instant::now();
            let prepared = decoder.prepare(inputs.clone()).expect("prepare");
            let result = decoder.decode_prepared(&prepared).expect("decode");
            let batch_ms = started.elapsed().as_secs_f64() * 1000.0;
            // Report rather than fail, so the same run characterizes builds
            // that cannot decode a level.
            if let Some(error) = result.errors().first() {
                eprintln!("large_region level={reduction} cpu_batch_error={error}");
                break;
            }

            let started = std::time::Instant::now();
            let tiles = std::thread::scope(|scope| {
                let handles: Vec<_> = rois
                    .iter()
                    .map(|&roi| {
                        let bytes = &bytes;
                        scope.spawn(move || {
                            let mut decoder = J2kDecoder::new(bytes).expect("CPU decoder");
                            decoder.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
                            let mut rgb = vec![0; 1024 * 1024 * 3];
                            decoder
                                .decode_region_scaled_pow2_into(
                                    &mut J2kScratchPool::default(),
                                    &mut rgb,
                                    1024 * 3,
                                    PixelFormat::Rgb8,
                                    roi,
                                    u8::try_from(reduction).expect("reduction"),
                                )
                                .expect("CPU region decode");
                            rgb
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("tile thread"))
                    .collect::<Vec<_>>()
            });
            let tiles_ms = started.elapsed().as_secs_f64() * 1000.0;
            if sample == 0 {
                for (index, tile) in tiles.iter().enumerate() {
                    let CpuBatchSamples::U8(samples) = decoded_samples_for_source(&result, index)
                    else {
                        panic!("RGB8 batch samples");
                    };
                    assert!(samples == *tile, "level {reduction} tile {index} differs");
                }
            }
            eprintln!(
                "large_region level={reduction} sample={sample} tiles=4 cpu_batch_ms={batch_ms:.3} cpu_tiles_ms={tiles_ms:.3}"
            );
        }
    }
}
