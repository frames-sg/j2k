// SPDX-License-Identifier: MIT OR Apache-2.0

//! A region decode on the streamed IDWT hands its rows to a row sink as they
//! are finished, with every component decoded together and no component
//! planes, its bands decoded serially or on the worker pool. Its samples must
//! match the planes bit for bit, every sample must arrive once, and it must
//! hold less.

use alloc::vec::Vec;

use super::banded_tier1::{cases, noisy_pixels, Case, HEIGHT, TILE, WIDTH};
use crate::{
    encode, ComponentPlane, ComponentRowSink, CpuDecodeParallelism, DecodeSettings, DecoderContext,
    EncodeOptions, Image,
};

/// Collects what a sink receives into region-sized planes of sample bits,
/// counting how often each sample arrives.
struct CollectingSink {
    width: usize,
    planes: Vec<Vec<u32>>,
    arrivals: Vec<u8>,
    depths: Vec<(u8, bool)>,
}

impl CollectingSink {
    fn new(width: u32, height: u32) -> Self {
        let samples = (width * height) as usize;
        Self {
            width: width as usize,
            planes: Vec::new(),
            arrivals: alloc::vec![0; samples],
            depths: Vec::new(),
        }
    }
}

impl ComponentRowSink for CollectingSink {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        let samples = self.arrivals.len();
        if self.planes.is_empty() {
            self.planes = alloc::vec![alloc::vec![0; samples]; planes.len()];
            self.depths = planes
                .iter()
                .map(|plane| (plane.bit_depth(), plane.signed()))
                .collect();
        }
        assert_eq!(planes.len(), self.planes.len());
        let (width, height) = planes[0].dimensions();
        for (component, plane) in planes.iter().enumerate() {
            assert_eq!(plane.dimensions(), (width, height));
            assert_eq!((plane.bit_depth(), plane.signed()), self.depths[component]);
            for row in 0..height as usize {
                for column in 0..width as usize {
                    let at = (y as usize + row) * self.width + x as usize + column;
                    self.planes[component][at] =
                        plane.samples()[row * width as usize + column].to_bits();
                    if component == 0 {
                        self.arrivals[at] += 1;
                    }
                }
            }
        }
    }
}

struct Decoded {
    planes: Vec<Vec<u32>>,
    depths: Vec<(u8, bool)>,
    streamed: bool,
    channel_bytes: usize,
    peak_bytes: usize,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(dead_code, reason = "only the pool band tests compare the counters")
    )]
    pool_components: usize,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(dead_code, reason = "only the pool band tests compare the counters")
    )]
    decoded_blocks: usize,
    #[cfg_attr(
        not(feature = "parallel"),
        expect(dead_code, reason = "only the pool band tests compare the counters")
    )]
    skipped_blocks: usize,
}

fn decode_planes(image: &Image<'_>, roi: (u32, u32, u32, u32)) -> Decoded {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    let components = image
        .decode_region_components_for_integer_output_with_context(roi, &mut context)
        .expect("decode region planes");
    let planes = components
        .planes()
        .iter()
        .map(|plane| {
            plane.samples()[..(roi.2 * roi.3) as usize]
                .iter()
                .map(|sample| sample.to_bits())
                .collect()
        })
        .collect();
    let depths = components
        .planes()
        .iter()
        .map(|plane| (plane.bit_depth(), plane.signed()))
        .collect();
    drop(components);
    let peak = context
        .tile_decode_context
        .debug_counters
        .workspace_peak
        .bytes();
    if std::env::var_os("ROW_SINK_DEBUG").is_some() {
        std::eprintln!("planes {roi:?} {peak:?}");
    }
    let counters = context.tile_decode_context.debug_counters;
    Decoded {
        planes,
        depths,
        streamed: false,
        channel_bytes: peak.channels,
        peak_bytes: peak.total(),
        pool_components: counters.pool_band_components,
        decoded_blocks: counters.decoded_code_blocks,
        skipped_blocks: counters.skipped_code_blocks,
    }
}

fn decode_rows(
    image: &Image<'_>,
    roi: (u32, u32, u32, u32),
    parallelism: CpuDecodeParallelism,
) -> Decoded {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    let mut sink = CollectingSink::new(roi.2, roi.3);
    image
        .decode_region_rows_for_integer_output_with_context(roi, &mut context, &mut sink)
        .expect("decode region rows");
    assert!(
        sink.arrivals.iter().all(|&count| count == 1),
        "every sample arrives once"
    );
    let peak = context
        .tile_decode_context
        .debug_counters
        .workspace_peak
        .bytes();
    if std::env::var_os("ROW_SINK_DEBUG").is_some() {
        std::eprintln!("rows {roi:?} {peak:?}");
    }
    let counters = context.tile_decode_context.debug_counters;
    Decoded {
        planes: sink.planes,
        depths: sink.depths,
        streamed: context.tile_decode_context.stream_rows,
        channel_bytes: peak.channels,
        peak_bytes: peak.total(),
        pool_components: counters.pool_band_components,
        decoded_blocks: counters.decoded_code_blocks,
        skipped_blocks: counters.skipped_code_blocks,
    }
}

fn row_sink_cases() -> Vec<Case> {
    let mut cases = cases();
    let base = EncodeOptions {
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    cases.push(Case {
        label: "rgba 9/7",
        components: 4,
        options: EncodeOptions {
            reversible: false,
            ..base.clone()
        },
    });
    cases.push(Case {
        label: "rgb 5/3 without MCT, tiled",
        components: 3,
        options: EncodeOptions {
            reversible: true,
            use_mct: false,
            tile_size: Some((TILE, TILE)),
            ..base.clone()
        },
    });
    cases.push(Case {
        label: "gray 9/7, 2 layers",
        components: 1,
        options: EncodeOptions {
            reversible: false,
            num_layers: 2,
            ..base
        },
    });
    cases
}

#[test]
fn row_sink_matches_component_planes_bit_for_bit() {
    for case in row_sink_cases() {
        let pixels = noisy_pixels(case.components);
        let bytes = encode(
            &pixels,
            WIDTH,
            HEIGHT,
            u16::from(case.components),
            8,
            false,
            &case.options,
        )
        .expect("encode row sink fixture");
        for reduction in [0_u8, 1, 2] {
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                .expect("parse row sink fixture");
            let (width, height) = (image.width(), image.height());
            let tile = TILE >> reduction;
            let large = (
                width / 20,
                height / 20,
                width - width / 10,
                height - height / 10,
            );
            for roi in [
                (0, 0, width, height),
                large,
                (width / 3, height / 3, width / 7, height / 9),
                (tile + 1, tile + 2, tile - 3, tile - 4),
                (tile - 5, tile - 3, 11, 7),
                (0, 0, width, 9),
                (0, height - 9, width, 9),
                (width - 1, height - 1, 1, 1),
            ] {
                let label = format!("{} reduction={reduction} roi={roi:?}", case.label);
                let planes = decode_planes(&image, roi);
                let rows = decode_rows(&image, roi, CpuDecodeParallelism::Serial);
                assert!(rows.streamed, "{label}: streamed");
                assert_eq!(rows.depths, planes.depths, "{label}");
                assert_eq!(rows.planes.len(), planes.planes.len(), "{label}");
                for (component, (rows, planes)) in
                    rows.planes.iter().zip(&planes.planes).enumerate()
                {
                    let first = rows.iter().zip(planes).position(|(a, b)| a != b);
                    assert_eq!(first, None, "{label}: component {component}");
                }
                // Only the component records, with no samples.
                assert!(rows.channel_bytes <= 256, "{label}: no planes");
                // Every component's bands are live together, which costs more
                // than small planes save; on a large region the planes win.
                assert!(
                    roi != large || reduction != 0 || rows.peak_bytes < planes.peak_bytes,
                    "{label}: peak {} vs {}",
                    rows.peak_bytes,
                    planes.peak_bytes
                );
                // A parallel decode streams its rows when its bands can
                // decode on the pool, and otherwise hands over its planes at
                // the end (see `pool_band_rows_match_component_planes_bit_for_bit`).
                let parallel = decode_rows(&image, roi, CpuDecodeParallelism::Auto);
                assert_eq!(parallel.planes, planes.planes, "{label}: parallel");
            }
        }
    }
}

/// With two threads or more, a region decode whose bands the route can
/// decode streams its rows too, with every component's bands on the pool at
/// once. Its rows must match the serial planes bit for bit, decode and skip
/// the same code-blocks, and hold no planes.
#[cfg(feature = "parallel")]
#[test]
fn pool_band_rows_match_component_planes_bit_for_bit() {
    for threads in [2, 4, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        for case in row_sink_cases() {
            let pixels = noisy_pixels(case.components);
            let bytes = encode(
                &pixels,
                WIDTH,
                HEIGHT,
                u16::from(case.components),
                8,
                false,
                &case.options,
            )
            .expect("encode pool row fixture");
            let mut pool_components = 0;
            for reduction in [0_u8, 1, 2] {
                let image =
                    Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                        .expect("parse pool row fixture");
                let (width, height) = (image.width(), image.height());
                let tile = TILE >> reduction;
                for roi in [
                    (0, 0, width, height),
                    (
                        width / 20,
                        height / 20,
                        width - width / 10,
                        height - height / 10,
                    ),
                    (width / 3, height / 3, width / 7, height / 9),
                    (tile - 5, tile - 3, 11, 7),
                    (0, height - 9, width, 9),
                    (width - 1, height - 1, 1, 1),
                ] {
                    let label = format!(
                        "{} threads={threads} reduction={reduction} roi={roi:?}",
                        case.label
                    );
                    let planes = decode_planes(&image, roi);
                    let rows =
                        pool.install(|| decode_rows(&image, roi, CpuDecodeParallelism::Auto));
                    assert_eq!(rows.planes, planes.planes, "{label}");
                    assert_eq!(rows.depths, planes.depths, "{label}");
                    if rows.pool_components > 0 {
                        assert!(rows.streamed, "{label}: streamed");
                        assert!(rows.channel_bytes <= 256, "{label}: no planes");
                        assert_eq!(rows.decoded_blocks, planes.decoded_blocks, "{label}");
                        assert_eq!(rows.skipped_blocks, planes.skipped_blocks, "{label}");
                    }
                    pool_components += rows.pool_components;
                }
            }
            // As in `pool_bands_match_whole_window_tier1_bit_for_bit`: the
            // route takes HT blocks and classic blocks of one arithmetic
            // segment, and two threads reach every such case.
            let routed = case.options.use_ht_block_coding || case.options.num_layers <= 1;
            if !routed {
                assert_eq!(pool_components, 0, "{}", case.label);
            } else if threads == 2 {
                assert!(pool_components > 0, "{} never used the pool", case.label);
            }
        }
    }
}

/// Each component's route plans its own clipped code-blocks, while a pool
/// worker or reader decodes the strips of every component with one scratch.
/// Components with detail only along the right edge code only the narrow
/// last blocks of each band, and a noisy one full-size blocks, so a worker
/// that served an edge component first must still fit the noisy component's
/// clipped blocks.
#[cfg(feature = "parallel")]
#[test]
fn pool_band_rows_fit_every_components_clipped_blocks() {
    let noise = noisy_pixels(1);
    let mut pixels = Vec::with_capacity(noise.len() * 3);
    for (at, &sample) in noise.iter().enumerate() {
        let edge = at % WIDTH as usize >= WIDTH as usize - 6;
        let detail = if edge { sample } else { 128 };
        pixels.extend_from_slice(&[sample, detail, detail ^ 0x55]);
    }
    for (label, use_ht_block_coding) in [("classic", false), ("ht", true)] {
        let bytes = encode(
            &pixels,
            WIDTH,
            HEIGHT,
            3,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 4,
                code_block_width_exp: 4,
                code_block_height_exp: 0,
                use_ht_block_coding,
                use_mct: false,
                ..EncodeOptions::default()
            },
        )
        .expect("encode clipped block fixture");
        let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), 0)
            .expect("parse clipped block fixture");
        for threads in [2, 3, 6, 12] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build pool");
            for roi in [(100, 17, 103, 120), (37, 3, 166, 9), (60, 41, 143, 101)] {
                let planes = decode_planes(&image, roi);
                for attempt in 0..4 {
                    let rows =
                        pool.install(|| decode_rows(&image, roi, CpuDecodeParallelism::Auto));
                    assert_eq!(
                        rows.planes, planes.planes,
                        "{label} threads={threads} roi={roi:?} attempt={attempt}"
                    );
                }
            }
        }
    }
}

/// A sink with an RGB8 destination, which [`ComponentRowSink::write_rows`]
/// fills the way the destination's contract says: each 8-bit sample rounded
/// and saturated.
#[cfg_attr(
    not(feature = "parallel"),
    expect(
        dead_code,
        reason = "only the parallel band reader offers an RGB8 destination"
    )
)]
struct Rgb8Sink {
    width: usize,
    rgb: Vec<u8>,
    offers_destination: bool,
    rectangles: usize,
}

impl ComponentRowSink for Rgb8Sink {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "samples are rounded and saturated to 8 bits, as the destination contract says"
    )]
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        self.rectangles += 1;
        let (width, height) = planes[0].dimensions();
        for row in 0..height as usize {
            for column in 0..width as usize {
                let pixel = (y as usize + row) * self.width + x as usize + column;
                for (channel, plane) in planes.iter().enumerate() {
                    let sample = plane.samples()[row * width as usize + column];
                    self.rgb[pixel * 3 + channel] = sample.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    fn rgb8_region(&mut self, width: u32, height: u32) -> Option<&mut [u8]> {
        let len = (width as usize * 3).checked_mul(height as usize)?;
        self.offers_destination.then(|| &mut self.rgb[..len])
    }
}

/// A streamed decode may pack three 8-bit components straight into a sink's
/// RGB8 destination. The bytes must be those the sink packs from the rows,
/// on one thread and on the pool, where each component's reconstruction runs
/// as its own task when the pool is large enough.
#[cfg(feature = "parallel")]
#[test]
fn rgb8_destination_matches_the_sink_pack_byte_for_byte() {
    let pools: Vec<rayon::ThreadPool> = [2, 4, 12]
        .into_iter()
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build pool")
        })
        .collect();
    for case in row_sink_cases()
        .into_iter()
        .filter(|case| case.components == 3)
    {
        let pixels = noisy_pixels(case.components);
        let bytes = encode(&pixels, WIDTH, HEIGHT, 3, 8, false, &case.options)
            .expect("encode RGB8 destination fixture");
        let mut packed_into_destination = 0;
        for reduction in [0_u8, 1] {
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                .expect("parse RGB8 destination fixture");
            let (width, height) = (image.width(), image.height());
            for roi in [
                (0, 0, width, height),
                (
                    width / 20,
                    height / 20,
                    width - width / 10,
                    height - height / 10,
                ),
                (width / 3, height / 3, width / 7, height / 9),
            ] {
                let decode = |parallelism, offers_destination| {
                    let mut context = DecoderContext::default();
                    context.set_cpu_decode_parallelism(parallelism);
                    let mut sink = Rgb8Sink {
                        width: roi.2 as usize,
                        rgb: alloc::vec![0; (roi.2 * roi.3 * 3) as usize],
                        offers_destination,
                        rectangles: 0,
                    };
                    image
                        .decode_region_rows_for_integer_output_with_context(
                            roi,
                            &mut context,
                            &mut sink,
                        )
                        .expect("decode RGB8 destination rows");
                    sink
                };
                let label = format!("{} reduction={reduction} roi={roi:?}", case.label);
                let expected = decode(CpuDecodeParallelism::Serial, false);
                for (pool, threads) in pools.iter().zip([2, 4, 12]) {
                    let packed = pool.install(|| decode(CpuDecodeParallelism::Auto, true));
                    assert_eq!(packed.rgb, expected.rgb, "{label} threads={threads}");
                    packed_into_destination += usize::from(packed.rectangles == 0);
                    let rows = pool.install(|| decode(CpuDecodeParallelism::Auto, false));
                    assert_eq!(rows.rgb, expected.rgb, "{label} threads={threads} rows");
                }
            }
        }
        assert!(
            packed_into_destination > 0,
            "{}: never packed into the destination",
            case.label
        );
    }
}

/// Outcome of a region decode, through the planes or the row sink: a hash
/// of the samples, or the error.
fn outcome(
    bytes: &[u8],
    strict: bool,
    roi: (u32, u32, u32, u32),
    rows: bool,
    parallelism: CpuDecodeParallelism,
) -> String {
    let settings = DecodeSettings {
        strict,
        ..DecodeSettings::default()
    };
    let image = match Image::new(bytes, &settings) {
        Ok(image) => image,
        Err(error) => return format!("parse {error:?}"),
    };
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    let hash = |planes: &[Vec<u32>]| {
        planes
            .iter()
            .flatten()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &bits| {
                (hash ^ u64::from(bits)).wrapping_mul(0x0100_0000_01b3)
            })
    };
    if rows {
        let mut sink = CollectingSink::new(roi.2, roi.3);
        match image.decode_region_rows_for_integer_output_with_context(roi, &mut context, &mut sink)
        {
            Ok(()) => format!("ok {:016x}", hash(&sink.planes)),
            Err(error) => format!("decode {error:?}"),
        }
    } else {
        match image.decode_region_components_for_integer_output_with_context(roi, &mut context) {
            Ok(components) => {
                let planes: Vec<Vec<u32>> = components
                    .planes()
                    .iter()
                    .map(|plane| {
                        plane.samples()[..(roi.2 * roi.3) as usize]
                            .iter()
                            .map(|sample| sample.to_bits())
                            .collect()
                    })
                    .collect();
                format!("ok {:016x}", hash(&planes))
            }
            Err(error) => format!("decode {error:?}"),
        }
    }
}

#[test]
fn row_sink_rejects_corrupt_code_blocks_as_component_planes_do() {
    // Small code-blocks give each component many bands, and the components
    // decode together, so two corrupt blocks often fail in a different order
    // than the planes path decodes them. On the pool they also fail in any
    // order across components.
    const SIZE: u32 = 48;
    #[cfg(feature = "parallel")]
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("build pool");
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let pixels: Vec<u8> = (0..SIZE * SIZE * 3).map(|_| (next() >> 56) as u8).collect();
    for use_ht_block_coding in [false, true] {
        let bytes = encode(
            &pixels,
            SIZE,
            SIZE,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                code_block_width_exp: 0,
                code_block_height_exp: 0,
                num_layers: 2,
                use_ht_block_coding,
                ..EncodeOptions::default()
            },
        )
        .expect("encode corrupt-block fixture");
        let mut failures = 0;
        for iteration in 0..300 {
            let mut mutant = bytes.clone();
            for _ in 0..=next() % 4 {
                let span = u64::try_from(mutant.len() - 120).expect("fixture length");
                let at = 120 + usize::try_from(next() % span).expect("offset in the fixture");
                mutant[at] ^= 1 << (next() % 8);
            }
            for strict in [false, true] {
                let roi = [(0, 0, SIZE, SIZE), (30, 30, 18, 18), (5, 9, 13, 7)]
                    [usize::try_from(next() % 3).expect("region index")];
                let label = format!(
                    "ht={use_ht_block_coding} iteration={iteration} strict={strict} roi={roi:?}"
                );
                let serial = CpuDecodeParallelism::Serial;
                let planes = outcome(&mutant, strict, roi, false, serial);
                let rows = outcome(&mutant, strict, roi, true, serial);
                assert_eq!(rows, planes, "{label}");
                #[cfg(feature = "parallel")]
                {
                    let auto = CpuDecodeParallelism::Auto;
                    let pool_rows = pool.install(|| outcome(&mutant, strict, roi, true, auto));
                    assert_eq!(pool_rows, planes, "pool {label}");
                }
                failures += usize::from(planes.starts_with("decode"));
            }
        }
        // The mutants must reach Tier-1 failures for the check to mean much.
        assert!(
            failures > 20,
            "ht={use_ht_block_coding}: {failures} failures"
        );
    }
}

/// The rows of a streamed decode on `context`, and its starting budget.
fn streamed_rows<'a>(
    image: &Image<'a>,
    roi: (u32, u32, u32, u32),
    context: &mut DecoderContext<'a>,
) -> (Vec<Vec<u32>>, usize) {
    let mut sink = CollectingSink::new(roi.2, roi.3);
    image
        .decode_region_rows_for_integer_output_with_context(roi, context, &mut sink)
        .expect("decode region rows");
    assert!(context.tile_decode_context.stream_rows);
    assert!(sink.arrivals.iter().all(|&count| count == 1));
    (
        sink.planes,
        context
            .tile_decode_context
            .debug_counters
            .decode_baseline_bytes,
    )
}

/// A decode that streams rows can't be rerun once rows reach the sink, so a
/// reused context must start it from a fresh context's budget: no retained
/// scratch or component planes charged, and the same rows.
#[test]
fn reused_context_streams_rows_from_a_fresh_budget() {
    let pixels = noisy_pixels(3);
    let bytes = encode(
        &pixels,
        WIDTH,
        HEIGHT,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 4,
            tile_size: Some((TILE, TILE)),
            ..EncodeOptions::default()
        },
    )
    .expect("encode reuse fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse reuse fixture");
    let roi = (37, 21, 130, 101);

    let mut fresh = DecoderContext::default();
    fresh.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    let (fresh_planes, fresh_baseline) = streamed_rows(&image, roi, &mut fresh);

    let mut reused = DecoderContext::default();
    reused.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
    drop(
        image
            .decode_region_components_for_integer_output_with_context(
                (0, 0, WIDTH, HEIGHT),
                &mut reused,
            )
            .expect("decode full planes"),
    );
    assert!(reused.tile_decode_context.retained_channel_bytes().unwrap() > 0);
    assert!(reused.tile_decode_context.retained_scratch_bytes().unwrap() > 0);
    let (reused_planes, reused_baseline) = streamed_rows(&image, roi, &mut reused);

    assert_eq!(reused_planes, fresh_planes);
    assert_eq!(reused_baseline, fresh_baseline);
    assert_eq!(
        reused.tile_decode_context.retained_channel_bytes().unwrap(),
        fresh.tile_decode_context.retained_channel_bytes().unwrap()
    );
}

/// The ROI plan's IDWT charge must cover the ring buffer each streamed tile
/// takes: every component's rings, and a scratch row each. `take_line_buffer`
/// asserts it for every tile of every streamed decode; this runs it over the
/// row-sink cases, multi-tile ones included, and three regions.
#[test]
fn streamed_rows_charge_every_components_rings() {
    for case in row_sink_cases() {
        let pixels = noisy_pixels(case.components);
        let bytes = encode(
            &pixels,
            WIDTH,
            HEIGHT,
            u16::from(case.components),
            8,
            false,
            &case.options,
        )
        .expect("encode ring charge fixture");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse ring fixture");
        for roi in [(0, 0, WIDTH, HEIGHT), (37, 21, 130, 101), (63, 60, 9, 70)] {
            let mut context = DecoderContext::default();
            context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
            streamed_rows(&image, roi, &mut context);
        }
    }
}

/// A streamed decode that holds the image's packet directory and runs out of
/// budget gives the directory up and reruns only while no row has reached the
/// sink, since a rerun after that would send those rows again. This decode
/// peaks before its first row, so it reruns, and every sample still arrives
/// once.
#[cfg(feature = "std")]
#[test]
fn streamed_rows_evict_the_packet_directory_before_the_first_row() {
    let pixels = noisy_pixels(3);
    let bytes = encode(
        &pixels,
        WIDTH,
        HEIGHT,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 4,
            // A packet directory needs one RPCL tile.
            progression_order: crate::EncodeProgressionOrder::Rpcl,
            ..EncodeOptions::default()
        },
    )
    .expect("encode directory fixture");
    let roi = (37, 21, 130, 101);
    let decode = |image: &Image<'_>, baseline: usize| {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
        let mut sink = CollectingSink::new(roi.2, roi.3);
        let result = image.run_decode_rows(
            &mut context,
            Some(roi),
            None,
            true,
            baseline,
            Some(&mut sink),
        );
        (result, sink, context.tile_decode_context.stream_rows)
    };
    // The largest baseline a decode without a directory fits in.
    let (mut fits, mut fails) = (0, crate::DEFAULT_MAX_DECODE_BYTES);
    while fails - fits > 1 {
        let candidate = fits + (fails - fits) / 2;
        let fresh = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        match decode(&fresh, candidate).0 {
            Ok(_) => fits = candidate,
            Err(error) => {
                assert!(crate::j2c::is_capacity_error(&error), "{error:?}");
                fails = candidate;
            }
        }
    }
    let fresh = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
    let (result, expected, streamed) = decode(&fresh, fits);
    result.expect("a fresh image fits");
    assert!(streamed);

    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
    let metadata = image.retained_allocation_bytes().expect("metadata bytes");
    for _ in 0..2 {
        decode(&image, metadata).0.expect("decode region rows");
    }
    assert!(
        image.retained_allocation_bytes().expect("retained bytes") > metadata,
        "a repeated region builds a directory"
    );
    let (result, sink, streamed) = decode(&image, fits);
    result.expect("evicting the directory restores a fitting request");
    assert!(streamed);
    assert!(sink.arrivals.iter().all(|&count| count == 1));
    assert_eq!(sink.planes, expected.planes);
    assert_eq!(
        image.retained_allocation_bytes().expect("retained bytes"),
        metadata
    );
}
