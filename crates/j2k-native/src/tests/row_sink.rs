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

#[expect(
    clippy::struct_field_names,
    reason = "decoded_blocks and skipped_blocks mirror the decoder's decoded and skipped code-block counters"
)]
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
    #[cfg_attr(
        not(feature = "parallel"),
        expect(dead_code, reason = "only the pool band tests compare the counters")
    )]
    parted_bands: usize,
    /// The deepest finest-level and coarser-level rings of the pool bands
    /// that held no whole window.
    #[cfg(feature = "parallel")]
    ring_slots: (usize, usize),
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
        parted_bands: counters.pool_band_parts,
        #[cfg(feature = "parallel")]
        ring_slots: (
            counters.pool_band_finest_slots,
            counters.pool_band_other_slots,
        ),
    }
}

fn decode_rows(
    image: &Image<'_>,
    roi: (u32, u32, u32, u32),
    parallelism: CpuDecodeParallelism,
) -> Decoded {
    decode_rows_with(image, roi, parallelism, |_| {})
}

fn decode_rows_with(
    image: &Image<'_>,
    roi: (u32, u32, u32, u32),
    parallelism: CpuDecodeParallelism,
    configure: impl FnOnce(&mut DecoderContext),
) -> Decoded {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    configure(&mut context);
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
        parted_bands: counters.pool_band_parts,
        #[cfg(feature = "parallel")]
        ring_slots: (
            counters.pool_band_finest_slots,
            counters.pool_band_other_slots,
        ),
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
            // route takes HT and classic blocks, and two threads reach every
            // case.
            if threads == 2 {
                assert!(pool_components > 0, "{} never used the pool", case.label);
            }
        }
    }
}

/// A streamed decode whose classic bands split into parts, which threads
/// decode at once into each band's slot, must give the serial planes' rows
/// bit for bit, decode and skip the same code-blocks, and hold no planes.
#[cfg(feature = "parallel")]
#[test]
fn pool_band_part_rows_match_component_planes_bit_for_bit() {
    let mut parted = 0;
    for threads in [2, 13] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        for case in row_sink_cases()
            .into_iter()
            .filter(|case| !case.options.use_ht_block_coding && case.options.num_layers <= 1)
        {
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
            .expect("encode band part row fixture");
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), 0)
                .expect("parse band part row fixture");
            let (width, height) = (image.width(), image.height());
            for roi in [
                (0, 0, width, height),
                (width / 3, height / 3, width / 7, height / 9),
                (0, height - 9, width, 9),
            ] {
                let planes = decode_planes(&image, roi);
                for parts in [None, Some(2), Some(5)] {
                    let label = format!(
                        "{} threads={threads} roi={roi:?} parts={parts:?}",
                        case.label
                    );
                    let rows = pool.install(|| {
                        decode_rows_with(&image, roi, CpuDecodeParallelism::Auto, |context| {
                            context.tile_decode_context.band_parts = parts;
                        })
                    });
                    assert_eq!(rows.planes, planes.planes, "{label}");
                    if rows.pool_components > 0 {
                        assert!(rows.streamed, "{label}: streamed");
                        assert!(rows.channel_bytes <= 256, "{label}: no planes");
                        assert_eq!(rows.decoded_blocks, planes.decoded_blocks, "{label}");
                        assert_eq!(rows.skipped_blocks, planes.skipped_blocks, "{label}");
                    }
                    parted += rows.parted_bands;
                }
            }
        }
    }
    assert!(parted > 0, "no band decoded in parts");
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
pub(super) struct Rgb8Sink {
    pub(super) width: usize,
    pub(super) rgb: Vec<u8>,
    pub(super) offers_destination: bool,
    pub(super) rectangles: usize,
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

/// The serial bands pack three 8-bit components into a sink's RGB8
/// destination too: a 9/7 image with the MCT straight from the IDWT's rows
/// through the fused pack, others after transforming a copy of the rows. The
/// bytes must be those the sink packs from the rows.
#[test]
fn serial_rgb8_destination_matches_the_sink_pack_byte_for_byte() {
    for case in row_sink_cases()
        .into_iter()
        .filter(|case| case.components == 3)
    {
        let pixels = noisy_pixels(case.components);
        let bytes = encode(&pixels, WIDTH, HEIGHT, 3, 8, false, &case.options)
            .expect("encode serial RGB8 destination fixture");
        let mut packed_into_destination = 0;
        for reduction in [0_u8, 1] {
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                .expect("parse serial RGB8 destination fixture");
            let (width, height) = (image.width(), image.height());
            for roi in [
                (0, 0, width, height),
                (width / 3, height / 3, width / 7, height / 9),
                (0, height - 9, width, 9),
            ] {
                let decode = |offers_destination| {
                    let mut context = DecoderContext::default();
                    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
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
                        .expect("decode serial RGB8 destination rows");
                    sink
                };
                let label = format!("{} reduction={reduction} roi={roi:?}", case.label);
                let expected = decode(false);
                let packed = decode(true);
                assert_eq!(packed.rgb, expected.rgb, "{label}");
                packed_into_destination += usize::from(packed.rectangles == 0);
            }
        }
        assert!(
            packed_into_destination > 0,
            "{}: never packed into the destination",
            case.label
        );
    }
}

/// A streamed decode has no per-sub-band path, so a classic component with
/// fewer rows of code-blocks than the pool has threads must still decode its
/// bands on the pool, in parts, rather than serially, with the planes' rows
/// bit for bit.
#[cfg(feature = "parallel")]
#[test]
fn pool_band_rows_use_a_pool_larger_than_the_code_block_rows() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(48)
        .build()
        .expect("build pool");
    for case in row_sink_cases().into_iter().filter(|case| {
        case.components == 3 && !case.options.use_ht_block_coding && case.options.num_layers <= 1
    }) {
        let pixels = noisy_pixels(case.components);
        let bytes = encode(&pixels, WIDTH, HEIGHT, 3, 8, false, &case.options)
            .expect("encode large pool fixture");
        let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), 0)
            .expect("parse large pool fixture");
        let roi = (0, 0, image.width(), image.height());
        let planes = decode_planes(&image, roi);
        let rows = pool.install(|| decode_rows(&image, roi, CpuDecodeParallelism::Auto));
        assert_eq!(rows.planes, planes.planes, "{}", case.label);
        assert!(rows.streamed, "{}: streamed", case.label);
        assert!(rows.pool_components > 0, "{}: decoded serially", case.label);
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

/// A classic tile of several components reconstructs each on a reader of
/// its own from 12 pool threads, and on 10 and 11 only from a size; the
/// readers then hold a batch of rows each, where one reader finishes its
/// rows in the IDWT's rings. The ROI plan must charge that batch on every
/// such pool: the
/// line buffer each streamed tile takes stays within the IDWT charge, and
/// the rows match the serial planes. The unit tests' size gate lies between
/// this full image and its half, so both sides of it run on 10 and 11
/// threads.
#[cfg(feature = "parallel")]
#[test]
fn classic_streamed_rows_charge_the_batches_of_their_readers() {
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
            // 16-row blocks give every component more rows of blocks than
            // the pool threads, so its bands decode on the pool.
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode reader charge fixture");
    for threads in [9, 10, 11, 12] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        for reduction in [0_u8, 1] {
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                .expect("parse reader charge fixture");
            let roi = (0, 0, image.width(), image.height());
            let label = format!("threads={threads} reduction={reduction}");
            let planes = decode_planes(&image, roi);
            let (rows, counters) = pool.install(|| {
                let mut context = DecoderContext::default();
                context.set_cpu_decode_parallelism(CpuDecodeParallelism::Auto);
                let mut sink = CollectingSink::new(roi.2, roi.3);
                image
                    .decode_region_rows_for_integer_output_with_context(
                        roi,
                        &mut context,
                        &mut sink,
                    )
                    .expect("decode region rows");
                assert!(context.tile_decode_context.stream_rows, "{label}");
                assert!(sink.arrivals.iter().all(|&count| count == 1), "{label}");
                (sink.planes, context.tile_decode_context.debug_counters)
            });
            assert_eq!(rows, planes.planes, "{label}");
            assert_eq!(counters.pool_band_components, 3, "{label}: pool bands");
            let readers = if threads >= 12 || (threads >= 10 && reduction == 0) {
                3
            } else {
                1
            };
            assert_eq!(counters.pool_band_readers, readers, "{label}: readers");
            assert_eq!(
                counters.uncharged_line_samples, 0,
                "{label}: line buffer beyond the charge"
            );
            // One reader finishes its rows in the IDWT's rings; a reader per
            // component holds a batch of them.
            assert_eq!(
                counters.line_batch_samples == 0,
                readers == 1,
                "{label}: row batch {}",
                counters.line_batch_samples
            );
        }
    }
}

/// The fixture's noise repeated `across` times side by side and `down`
/// times one under the other, with `components` components, encoded with
/// four levels and 16-row code-blocks: every component has more rows of
/// blocks than the pool threads, so its bands decode on the pool.
#[cfg(feature = "parallel")]
fn pool_band_fixture(components: u8, across: usize, down: usize) -> Vec<u8> {
    let row = usize::from(components) * WIDTH as usize;
    let rows: Vec<u8> = noisy_pixels(components)
        .chunks_exact(row)
        .flat_map(|samples| core::iter::repeat_n(samples, across).flatten())
        .copied()
        .collect();
    let width = WIDTH * u32::try_from(across).expect("fixture width");
    let height = HEIGHT * u32::try_from(down).expect("fixture height");
    encode(
        &rows.repeat(down),
        width,
        height,
        u16::from(components),
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 4,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode pool band fixture")
}

/// A classic tile that reconstructs each component on its own reader without
/// deep band rings holds two bands in each finest ring when its pool has
/// more threads than the tile has finest rings, and its bands then split
/// into parts so every thread has one: three components on 10 and 11
/// threads and at half resolution from 12, four components at half
/// resolution on 13 threads but not on 12. One reader keeps three bands,
/// and the deep rings of full resolution from 12 threads four, both in
/// whole bands; the coarser levels keep theirs. The rows match the serial
/// planes on every pool.
#[cfg(feature = "parallel")]
#[test]
fn shallow_finest_rings_split_their_bands_into_parts() {
    // Three fixture widths side by side give finest-level bands of several
    // code-blocks at both resolutions, each above the unit tests' reader
    // size gate, and two heights give the coarser levels rings that hold no
    // whole window.
    for (components, threads) in [(3_u8, &[9_usize, 10, 11, 12][..]), (4, &[12, 13][..])] {
        let bytes = pool_band_fixture(components, 3, 2);
        for reduction in [0_u8, 1] {
            let image = Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                .expect("parse shallow ring fixture");
            let roi = (0, 0, image.width(), image.height());
            let planes = decode_planes(&image, roi);
            for &threads in threads {
                let label =
                    format!("components={components} threads={threads} reduction={reduction}");
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("build pool");
                let rows = pool.install(|| decode_rows(&image, roi, CpuDecodeParallelism::Auto));
                assert_eq!(rows.planes, planes.planes, "{label}");
                assert_eq!(
                    rows.pool_components,
                    usize::from(components),
                    "{label}: pool bands"
                );
                let deep = threads >= 12 && reduction == 0;
                let shallow = threads >= 10 && !deep && 3 * usize::from(components) < threads;
                let slots = if deep {
                    (4, 3)
                } else if shallow {
                    (2, 2)
                } else {
                    (3, 2)
                };
                assert_eq!(rows.ring_slots, slots, "{label}: ring slots");
                assert_eq!(rows.parted_bands > 0, shallow, "{label}: parted bands");
            }
        }
    }
}

/// A decode that shares its pool with other images in flight, as the images
/// of an explicit-worker batch that fill its pool do, holds two bands in
/// every ring below 12 threads, in place of three at the finest level, and
/// keeps its bands whole, where on 10 threads a lone decode's two-band rings
/// split them; with fewer bands per ring it peaks lower than a lone decode.
/// From 12 threads it holds the lone decode's deep rings. Its rows match the serial planes. A context
/// recovered from the workspace of a marked decode decodes as a fresh one
/// does.
#[cfg(feature = "parallel")]
#[test]
fn shared_pool_rows_hold_two_bands_per_ring() {
    // Two fixture heights give the coarser levels rings that hold no whole
    // window, deep ones too.
    let bytes = pool_band_fixture(3, 1, 2);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse shared pool fixture");
    let roi = (0, 0, image.width(), image.height());
    let planes = decode_planes(&image, roi);
    for (threads, lone_slots, shared_slots) in [
        (4, (3, 2), (2, 2)),
        (10, (2, 2), (2, 2)),
        (12, (4, 3), (4, 3)),
    ] {
        let label = format!("threads={threads}");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        let (lone, shared, recovered) = pool.install(|| {
            let lone = decode_rows(&image, roi, CpuDecodeParallelism::Auto);
            let shared = decode_rows_with(&image, roi, CpuDecodeParallelism::Auto, |context| {
                context.set_shared_pool(true);
            });
            let mut marked = DecoderContext::default();
            marked.set_shared_pool(true);
            let mut sink = CollectingSink::new(roi.2, roi.3);
            image
                .decode_region_rows_for_integer_output_with_context(roi, &mut marked, &mut sink)
                .expect("decode marked rows");
            let workspace = marked.into_workspace();
            let recovered = decode_rows_with(&image, roi, CpuDecodeParallelism::Auto, |context| {
                *context = DecoderContext::from_workspace(workspace);
            });
            (lone, shared, recovered)
        });
        for (decoded, name) in [
            (&lone, "lone"),
            (&shared, "shared"),
            (&recovered, "recovered"),
        ] {
            assert_eq!(decoded.planes, planes.planes, "{label} {name}");
            assert_eq!(decoded.pool_components, 3, "{label} {name}: pool bands");
        }
        assert_eq!(lone.ring_slots, lone_slots, "{label}: lone ring slots");
        assert_eq!(
            shared.ring_slots, shared_slots,
            "{label}: shared ring slots"
        );
        assert_eq!(
            recovered.ring_slots, lone_slots,
            "{label}: recovered ring slots"
        );
        assert_eq!(
            (lone.parted_bands > 0, shared.parted_bands),
            (threads == 10, 0),
            "{label}: bands in parts"
        );
        if shared_slots != lone_slots {
            assert!(
                shared.peak_bytes < lone.peak_bytes,
                "{label}: shared {} lone {}",
                shared.peak_bytes,
                lone.peak_bytes
            );
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

/// A classic tile whose components reconstruct on readers of their own
/// holds one batch of rows, which the ROI plan charges by the tile's width.
/// In an 8,200-wide image of 8,192-wide tiles, a batch sized by the image's
/// width keeps its 255 rows whole (half would be under 128), where the
/// tile's 8,192-wide rows are charged 128 (half of 256): the batch must keep
/// to the tile's charge, or the first tile holds twice it.
#[cfg(feature = "parallel")]
#[test]
fn reader_batches_of_tiles_narrower_than_the_image_stay_within_their_charge() {
    let (width, height) = (8200_u32, 1024_u32);
    // Smooth content keeps the large fixture's Tier-1 cheap.
    let pixels: Vec<u8> = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [
                    x.to_le_bytes()[0],
                    y.to_le_bytes()[0],
                    (x + y).to_le_bytes()[0],
                ]
            })
        })
        .collect();
    let bytes = encode(
        &pixels,
        width,
        height,
        3,
        8,
        false,
        &EncodeOptions {
            num_decomposition_levels: 2,
            // 16-row blocks give each component more rows of blocks than
            // the pool threads, so its bands decode on the pool.
            code_block_height_exp: 2,
            tile_size: Some((8192, height)),
            ..EncodeOptions::default()
        },
    )
    .expect("encode wide tile fixture");
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse wide tile fixture");
    let roi = (0, 0, width, height);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(12)
        .build()
        .expect("build pool");
    let counters = pool.install(|| {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(CpuDecodeParallelism::Auto);
        let mut sink = CollectingSink::new(roi.2, roi.3);
        image
            .decode_region_rows_for_integer_output_with_context(roi, &mut context, &mut sink)
            .expect("decode region rows");
        assert!(context.tile_decode_context.stream_rows);
        assert!(sink.arrivals.iter().all(|&count| count == 1));
        context.tile_decode_context.debug_counters
    });
    assert_eq!(counters.pool_band_readers, 3, "a reader per component");
    assert_eq!(
        counters.line_batch_samples,
        3 * 8192 * 128,
        "one batch of 128 rows of each component of the wide tile"
    );
    assert_eq!(
        counters.uncharged_line_samples, 0,
        "line buffer beyond the charge"
    );
}

/// A serial region decode that runs out of budget must do so before its
/// first row, whatever the budget, so that the retries without retained
/// scratch or the packet directory can still rerun it. A region that clips
/// the code-blocks of its last rows decodes them only after its first rows
/// went out, so the scratch an HTJ2K block clipped this way decodes into must
/// be held, and charged, from the start. Budgets just past the largest one
/// that fits can fit again where they leave out the optional row index, so
/// each one either decodes every sample once or fails before any.
#[test]
fn streamed_rows_run_out_of_budget_before_their_first_row() {
    for use_ht_block_coding in [false, true] {
        let pixels = noisy_pixels(3);
        let bytes = encode(
            &pixels,
            WIDTH,
            HEIGHT,
            3,
            8,
            false,
            &EncodeOptions {
                use_ht_block_coding,
                num_decomposition_levels: 3,
                code_block_height_exp: 1,
                ..EncodeOptions::default()
            },
        )
        .expect("encode budget fixture");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        for roi in [
            (37, 21, 130, 101),
            (0, 0, WIDTH, 61),
            (0, 0, WIDTH, HEIGHT - 11),
        ] {
            let label = format!("ht={use_ht_block_coding} {roi:?}");
            let decode = |baseline: usize| {
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
                (result, sink)
            };
            let (mut fits, mut fails) = (0, crate::DEFAULT_MAX_DECODE_BYTES);
            while fails - fits > 1 {
                let candidate = fits + (fails - fits) / 2;
                match decode(candidate).0 {
                    Ok(_) => fits = candidate,
                    Err(_) => fails = candidate,
                }
            }
            let mut failures = 0;
            for over in [
                0,
                1,
                2,
                3,
                5,
                9,
                17,
                65,
                257,
                1 << 10,
                1 << 12,
                1 << 14,
                1 << 16,
            ] {
                let (result, sink) = decode(fits + over);
                match result {
                    Ok(_) => assert!(
                        sink.arrivals.iter().all(|&count| count == 1),
                        "{label} +{over}: every sample once"
                    ),
                    Err(error) => {
                        assert!(
                            crate::j2c::is_capacity_error(&error),
                            "{label} +{over}: {error:?}"
                        );
                        assert!(
                            sink.arrivals.iter().all(|&count| count == 0),
                            "{label} +{over}: rows before the failure"
                        );
                        failures += 1;
                    }
                }
            }
            assert!(failures > 0, "{label}: some budget fails");
        }
    }
}

/// Stitched windows of a reduced resolution, as a tiled viewer or an IIIF
/// server requests them, hold few rows of code-blocks per component: fewer
/// than a pool of 18 threads has workers. Such a window's streamed rows must
/// still decode its classic bands on the pool, in parts, and not fall back to
/// one serial thread for the whole tile, with the planes' rows bit for bit.
#[cfg(feature = "parallel")]
#[test]
fn reduced_windows_stream_on_a_pool_with_more_threads_than_their_rows() {
    for threads in [18, 48] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool");
        for case in row_sink_cases().into_iter().filter(|case| {
            case.components == 3
                && !case.options.use_ht_block_coding
                && case.options.num_layers <= 1
                && case.options.tile_size.is_none()
        }) {
            let pixels = noisy_pixels(case.components);
            let bytes = encode(&pixels, WIDTH, HEIGHT, 3, 8, false, &case.options)
                .expect("encode reduced window fixture");
            for reduction in [1_u8, 2] {
                let image =
                    Image::new_with_reduction(&bytes, &DecodeSettings::default(), reduction)
                        .expect("parse reduced window fixture");
                let (width, height) = (image.width(), image.height());
                let (half_width, half_height) = (width.div_ceil(2), height.div_ceil(2));
                for roi in [
                    (0, 0, half_width, half_height),
                    (half_width, 0, width - half_width, half_height),
                    (0, height - 3, half_width, 3),
                ] {
                    let label = format!(
                        "{} threads={threads} reduction={reduction} roi={roi:?}",
                        case.label
                    );
                    let planes = decode_planes(&image, roi);
                    let rows =
                        pool.install(|| decode_rows(&image, roi, CpuDecodeParallelism::Auto));
                    assert_eq!(rows.planes, planes.planes, "{label}");
                    assert!(rows.streamed, "{label}: streamed");
                    assert_eq!(rows.pool_components, 3, "{label}: decoded serially");
                }
            }
        }
    }
}

/// A sink that stops after it has taken `limit` hand-overs, counting every
/// hand-over it is given.
struct StoppingSink {
    limit: usize,
    calls: usize,
}

impl ComponentRowSink for StoppingSink {
    fn write_rows(&mut self, _x: u32, _y: u32, _planes: &[ComponentPlane<'_>]) {
        self.calls += 1;
    }

    fn stopped(&self) -> bool {
        self.calls >= self.limit
    }
}

/// A sink that stops ends the region decode at that hand-over, serially and
/// on a pool, with `SinkStopped`: the rest of the region is not decoded for
/// rows that would go nowhere.
#[cfg(feature = "parallel")]
#[test]
fn a_stopped_sink_ends_the_decode_at_its_hand_over() {
    let bytes = pool_band_fixture(3, 2, 2);
    let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
    let roi = (0, 0, image.width(), image.height());
    let mut full = StoppingSink {
        limit: usize::MAX,
        calls: 0,
    };
    image
        .decode_region_rows_for_integer_output_with_context(
            roi,
            &mut DecoderContext::default(),
            &mut full,
        )
        .expect("whole decode");
    assert!(
        full.calls > 4,
        "the fixture hands over {} times",
        full.calls
    );

    let mut parallelisms = alloc::vec![(CpuDecodeParallelism::Serial, 1)];
    if cfg!(feature = "parallel") {
        parallelisms.push((CpuDecodeParallelism::Auto, 4));
    }
    for (parallelism, threads) in parallelisms {
        let run = || {
            let mut context = DecoderContext::default();
            context.set_cpu_decode_parallelism(parallelism);
            let mut sink = StoppingSink { limit: 1, calls: 0 };
            let decoded = image.decode_region_rows_for_integer_output_with_context(
                roi,
                &mut context,
                &mut sink,
            );
            (decoded, sink.calls)
        };
        #[cfg(feature = "parallel")]
        let (decoded, calls) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(run);
        #[cfg(not(feature = "parallel"))]
        let (decoded, calls) = {
            let _ = threads;
            run()
        };
        assert_eq!(
            decoded,
            Err(crate::DecodingError::SinkStopped.into()),
            "{parallelism:?}"
        );
        assert_eq!(calls, 1, "{parallelism:?}: hand-overs after the stop");
    }
}

#[test]
fn a_stopped_sink_is_reported_after_a_component_plane_fallback() {
    let bytes = include_bytes!(
        "../../../j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c"
    );
    let image = Image::new(bytes, &DecodeSettings::default()).expect("origin image");
    assert!(!image.region_decode_streams_rows());
    let mut sink = StoppingSink { limit: 1, calls: 0 };
    let decoded = image.decode_region_rows_for_integer_output_with_context(
        (0, 0, image.width(), image.height()),
        &mut DecoderContext::default(),
        &mut sink,
    );
    assert_eq!(decoded, Err(crate::DecodingError::SinkStopped.into()));
    assert_eq!(sink.calls, 1);
}

#[cfg(feature = "parallel")]
#[test]
fn operation_cancellation_stops_rows_and_leaves_the_context_reusable() {
    use alloc::sync::Arc;
    use core::sync::atomic::{AtomicBool, Ordering};
    struct CancelAfterFirst {
        flag: Arc<AtomicBool>,
        calls: usize,
    }
    impl ComponentRowSink for CancelAfterFirst {
        fn write_rows(&mut self, _: u32, _: u32, _: &[ComponentPlane<'_>]) {
            self.calls += 1;
            self.flag.store(true, Ordering::Relaxed);
        }
    }
    let bytes = pool_band_fixture(3, 2, 2);
    let image = Image::new(&bytes, &DecodeSettings::default()).unwrap();
    let roi = (0, 0, image.width(), image.height());
    let mut context = DecoderContext::default();
    let flag = Arc::new(AtomicBool::new(false));
    let mut sink = CancelAfterFirst {
        flag: flag.clone(),
        calls: 0,
    };
    let result = context.with_operation_control(
        crate::DecodeOperationControl::new(None, Some(flag)),
        |context| image.decode_region_rows_for_integer_output_with_context(roi, context, &mut sink),
    );
    assert_eq!(result, Err(crate::DecodingError::Cancelled.into()));
    assert_eq!(
        sink.calls, 1,
        "cancellation must prevent another published row"
    );
    assert_eq!(
        context
            .tile_decode_context
            .debug_counters
            .whole_window_reruns,
        0,
        "cancellation must not trigger a diagnostic decode"
    );
    let mut full = StoppingSink {
        limit: usize::MAX,
        calls: 0,
    };
    image
        .decode_region_rows_for_integer_output_with_context(roi, &mut context, &mut full)
        .unwrap();
    assert!(
        full.calls > 1,
        "cancelled policy must not survive the operation"
    );
}

#[test]
fn operation_working_limit_applies_to_existing_images_and_restores_context() {
    let bytes = include_bytes!(
        "../../../j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c"
    );
    let image = Image::new(bytes, &DecodeSettings::default()).unwrap();
    let mut context = DecoderContext::default();
    let policy = crate::DecodeOperationControl::new(Some(1), None);
    let error = context
        .with_operation_control(policy.clone(), |context| image.decode_with_context(context))
        .err()
        .expect("working limit");
    assert!(matches!(
        error,
        crate::DecodeError::AllocationTooLarge {
            cap: crate::DEFAULT_MAX_DECODE_BYTES,
            ..
        }
    ));
    let decoded = image.decode_with_context(&mut context).unwrap();
    assert_eq!((decoded.width, decoded.height), (23, 17));
}

#[test]
fn operation_working_limit_includes_borrowed_plane_metadata_handoff() {
    let bytes = include_bytes!(
        "../../../j2k-test-support/fixtures/openjpeg_origin/rgb_23x17_origin_3_5_ht.j2c"
    );
    let mut image = Image::new(bytes, &DecodeSettings::default()).unwrap();
    // Native component output carries this metadata without applying a CMS.
    // Its contents do not affect the copy's allocation size.
    image.color_space = crate::ColorSpace::Icc {
        profile: alloc::vec![0; 64 * 1024],
        num_channels: 3,
    };
    let mut context = DecoderContext::default();
    let control = crate::DecodeOperationControl::new(Some(96 * 1024), None);
    let result = context.with_operation_control(control, |context| {
        image
            .decode_region_components_for_integer_output_with_context((0, 0, 1, 1), context)
            .map(|_| ())
    });
    assert!(
        matches!(result, Err(crate::DecodeError::AllocationTooLarge { .. })),
        "metadata clone must fit beside decoded channels and the live image: {result:?}"
    );
    assert!(image
        .decode_region_components_for_integer_output_with_context((0, 0, 1, 1), &mut context)
        .is_ok());
}

/// Native region samples as interleaved bytes, or concatenated component planes.
fn decode_native_region_bytes<'a>(
    image: &Image<'a>,
    context: &mut DecoderContext<'a>,
    rect: (u32, u32, u32, u32),
    component_planes: bool,
) -> crate::Result<Vec<u8>> {
    if component_planes {
        image
            .decode_native_region_components_with_context(rect, context)
            .map(|decoded| {
                decoded
                    .planes
                    .into_iter()
                    .flat_map(|plane| plane.data)
                    .collect()
            })
    } else {
        image
            .decode_native_region_with_context(rect, context)
            .map(|decoded| decoded.data)
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "raw and component region routes share one sampled 25-bit fixture and budget search"
)]
fn high_precision_sampled_regions_decode_without_the_full_output() {
    let (width, height) = (128_u32, 128_u32);
    let pixels = (0..width * height)
        .flat_map(|index| ((index * 104_729) & ((1 << 25) - 1)).to_le_bytes())
        .collect::<Vec<_>>();
    // Mixed sampling stays outside the shared window planner, but the integer
    // store keeps only the region's replicated samples, as the float store
    // does, so no full output is decoded and cropped.
    let sampled = (0..width.div_ceil(2) * height.div_ceil(2))
        .flat_map(|index| ((index * 500_009) & ((1 << 25) - 1)).to_le_bytes())
        .collect::<Vec<_>>();
    let expanded = (0..width * height)
        .flat_map(|index| {
            let offset =
                (((index / width) / 2) * width.div_ceil(2) + (index % width) / 2) as usize * 4;
            sampled[offset..offset + 4].iter().copied()
        })
        .collect::<Vec<_>>();
    let interleaved = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(expanded.as_chunks::<4>().0)
        .flat_map(|(first, second)| first.iter().chain(second).copied())
        .collect::<Vec<_>>();
    let planes = [
        crate::EncodeTypedComponentPlane {
            data: &pixels,
            x_rsiz: 1,
            y_rsiz: 1,
            bit_depth: 25,
            signed: false,
        },
        crate::EncodeTypedComponentPlane {
            data: &sampled,
            x_rsiz: 2,
            y_rsiz: 2,
            bit_depth: 25,
            signed: false,
        },
    ];
    let bytes = crate::encode_typed_component_planes_53(
        &planes,
        width,
        height,
        &crate::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            tile_size: Some((32, 32)),
            use_mct: false,
            ..crate::EncodeOptions::default()
        },
    )
    .expect("small tiled high-precision source");
    let image = Image::new(&bytes, &DecodeSettings::default()).unwrap();
    let roi = (1, 1, width - 2, height - 2);
    let crop_bytes = (roi.2 * roi.3 * 8) as usize;
    let crop_rect = |data: &[u8], pixel_bytes: usize, (x, y, w, h): (u32, u32, u32, u32)| {
        data.chunks_exact(width as usize * pixel_bytes)
            .skip(y as usize)
            .take(h as usize)
            .flat_map(|row| {
                row[x as usize * pixel_bytes..(x + w) as usize * pixel_bytes]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>()
    };
    let crop = |data: &[u8], pixel_bytes: usize| crop_rect(data, pixel_bytes, roi);

    for component_planes in [false, true] {
        let expected = if component_planes {
            [crop(&pixels, 4), crop(&expanded, 4)].concat()
        } else {
            crop(&interleaved, 8)
        };
        let mut context = DecoderContext::default();
        let packed_bytes = if component_planes {
            let full = image
                .decode_native_components_with_context(&mut context)
                .unwrap();
            assert_eq!(full.planes[0].data, pixels);
            assert_eq!(full.planes[1].data, expanded);
            full.planes.capacity() * core::mem::size_of::<crate::NativeComponentPlane>()
                + full
                    .planes
                    .iter()
                    .map(|plane| plane.data.capacity())
                    .sum::<usize>()
        } else {
            let full = image.decode_native_with_context(&mut context).unwrap();
            assert_eq!(full.data, interleaved);
            full.data.capacity() + full.component_signed.capacity().div_ceil(8)
        };
        // The full output fits with headroom, but a second nearly full-sized
        // crop could not coexist with it: the region must decode without it.
        let cap = image.retained_allocation_bytes().unwrap()
            + context.retained_allocation_bytes().unwrap()
            + packed_bytes
            + crop_bytes / 2;
        assert!(
            cap < 2 * 1024 * 1024,
            "representative small working set: {cap}"
        );
        drop(context);

        for reserved_owner in [false, true] {
            let control = if reserved_owner {
                crate::DecodeOperationControl::default()
                    .beside_retained_bytes(crate::DEFAULT_MAX_DECODE_BYTES - cap)
                    .unwrap()
            } else {
                crate::DecodeOperationControl::new(Some(cap), None)
            };
            let mut context = DecoderContext::default();
            context
                .with_operation_control(control.clone(), |context| {
                    if component_planes {
                        image
                            .decode_native_components_with_context(context)
                            .map(|_| ())
                    } else {
                        image.decode_native_with_context(context).map(|_| ())
                    }
                })
                .expect("the full native output fits the same operation budget");
            let result = context.with_operation_control(control, |context| {
                decode_native_region_bytes(&image, context, roi, component_planes)
            });
            assert_eq!(
                result.unwrap_or_else(|error| panic!(
                    "bounded sampled region: planes={component_planes}, reserved={reserved_owner}, cap={cap}: {error:?}"
                )),
                expected
            );
            let actual = if component_planes {
                image
                    .decode_native_region_components_with_context(roi, &mut context)
                    .unwrap()
                    .planes
                    .into_iter()
                    .flat_map(|plane| plane.data)
                    .collect::<Vec<_>>()
            } else {
                image
                    .decode_native_region_with_context(roi, &mut context)
                    .unwrap()
                    .data
            };
            assert_eq!(
                actual, expected,
                "uncontrolled reuse preserves native pixels"
            );
        }

        // A small region fits an allowance the full output cannot.
        let small = (5, 7, 17, 13);
        let small_expected = if component_planes {
            [crop_rect(&pixels, 4, small), crop_rect(&expanded, 4, small)].concat()
        } else {
            crop_rect(&interleaved, 8, small)
        };
        let fits = |limit: usize| {
            DecoderContext::default().with_operation_control(
                crate::DecodeOperationControl::new(Some(limit), None),
                |context| decode_native_region_bytes(&image, context, small, component_planes),
            )
        };
        let (mut low, mut high) = (1, cap);
        assert_eq!(fits(high).expect("small region within cap"), small_expected);
        while low < high {
            let middle = low + (high - low) / 2;
            if fits(middle).is_ok() {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        assert_eq!(fits(high).expect("smallest allowance"), small_expected);
        let full = DecoderContext::default().with_operation_control(
            crate::DecodeOperationControl::new(Some(high), None),
            |context| {
                if component_planes {
                    image
                        .decode_native_components_with_context(context)
                        .map(|_| ())
                } else {
                    image.decode_native_with_context(context).map(|_| ())
                }
            },
        );
        assert!(
            matches!(full, Err(crate::DecodeError::AllocationTooLarge { .. })),
            "planes={component_planes}: the full output exceeds the small region's {high}-byte allowance: {full:?}"
        );
    }
}
