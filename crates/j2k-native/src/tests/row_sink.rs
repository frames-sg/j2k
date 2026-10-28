// SPDX-License-Identifier: MIT OR Apache-2.0

//! A serial region decode on the streamed IDWT hands its rows to a row sink
//! as they are finished, with every component decoded together and no
//! component planes. Its samples must match the planes bit for bit, every
//! sample must arrive once, and it must hold less.

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
    Decoded {
        planes,
        depths,
        streamed: false,
        channel_bytes: peak.channels,
        peak_bytes: peak.total(),
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
    Decoded {
        planes: sink.planes,
        depths: sink.depths,
        streamed: context.tile_decode_context.stream_rows,
        channel_bytes: peak.channels,
        peak_bytes: peak.total(),
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
                // A parallel decode keeps its planes and hands them over at
                // the end.
                let parallel = decode_rows(&image, roi, CpuDecodeParallelism::Auto);
                #[cfg(feature = "parallel")]
                if rayon::current_num_threads() > 1 {
                    assert!(!parallel.streamed, "{label}: parallel planes");
                }
                assert_eq!(parallel.planes, planes.planes, "{label}: parallel");
            }
        }
    }
}

/// Outcome of a serial region decode, through the planes or the row sink:
/// a hash of the samples, or the error.
fn outcome(bytes: &[u8], strict: bool, roi: (u32, u32, u32, u32), rows: bool) -> String {
    let settings = DecodeSettings {
        strict,
        ..DecodeSettings::default()
    };
    let image = match Image::new(bytes, &settings) {
        Ok(image) => image,
        Err(error) => return format!("parse {error:?}"),
    };
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(CpuDecodeParallelism::Serial);
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
    // than the planes path decodes them.
    const SIZE: u32 = 48;
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
                let planes = outcome(&mutant, strict, roi, false);
                let rows = outcome(&mutant, strict, roi, true);
                assert_eq!(
                    rows, planes,
                    "ht={use_ht_block_coding} iteration={iteration} strict={strict} roi={roi:?}"
                );
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
