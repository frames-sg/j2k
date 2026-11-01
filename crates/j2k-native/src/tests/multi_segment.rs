// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic code-blocks of several codeword segments (selective bypass,
//! termination on each pass, or both) and the other code-block styles
//! (context reset, segmentation symbols, vertically causal context,
//! predictable termination) decode on the worker pool as on one thread: in
//! place on the tile route, and band by band ahead of the streamed IDWT,
//! whose rows then need no component planes.
//!
//! The fixtures in `fixtures/multi_segment/` were encoded with `OpenJPEG`
//! 2.5.4's `opj_compress` from synthetic images of 64 x 64 and 128 x 128
//! samples. The unsigned 8-bit red, green and blue samples at `(x, y)` are
//!
//! ```text
//! r = (7x² + 13y² + 5xy + 31x) mod 256
//! g = (3x² + 11y² + 9xy + 29y + 64) mod 256
//! b = (13x² + 5y² + 3xy + 7(x xor y)) mod 256
//! ```
//!
//! and the gray sample is `(r + 2g + b) / 4`, rounded down. Written as a
//! binary PPM (`P6`) and PGM (`P5`) of maximum 255, `rgb64.ppm`,
//! `rgb128.ppm`, `gray64.pgm` and `gray128.pgm`, they encode with two
//! decomposition levels, 16 x 16 code-blocks, RPCL order and three or four
//! quality layers, the last of most at a low compression ratio, so that
//! blocks code bit-planes far below their first four:
//!
//! ```sh
//! c="-n 3 -b 16,16 -p RPCL"
//! opj_compress -i gray128.pgm -o bypass_gray_53.j2k $c -M 1 -r 4,2,1
//! opj_compress -i rgb64.ppm -o bypass_rgb_97.j2k $c -I -M 1 -r 8,4,2,1
//! opj_compress -i gray128.pgm -o termall_gray_97.j2k $c -I -M 4 -r 16,8,3
//! opj_compress -i rgb128.ppm -o termall_rgb_53.j2k $c -M 4 -r 32,16,8
//! opj_compress -i rgb128.ppm -o reset_rgb_97.j2k $c -I -M 2 -r 32,16,8
//! opj_compress -i gray128.pgm -o segmark_gray_53.j2k $c -M 32 -r 16,8,3
//! opj_compress -i gray64.pgm -o bypass_termall_gray_97.j2k $c -I -M 5 -r 4,2,1
//! opj_compress -i rgb64.ppm -o bypass_termall_rgb_53.j2k $c -M 5 -r 8,4,1
//! opj_compress -i gray64.pgm -o all_modes_gray_97.j2k $c -I -M 63 -r 4,2,1
//! ```
//!
//! As archival images are often coded, one has selective bypass, four
//! decomposition levels, 64 x 64 code-blocks, precincts, packet lengths and
//! seven layers:
//!
//! ```sh
//! opj_compress -i rgb128.ppm -o bypass_rgb_97_64x64.j2k -n 5 -b 64,64 -p RPCL -PLT -I -M 1 \
//!     -c "[128,128],[128,128],[64,64]" -r 24,16,10,6,3,2,1.5
//! ```
//!
//! And one codes its first component in 32 x 64 code-blocks and the others
//! in 64 x 64 ones. Two encodes differ only in their code-block size:
//!
//! ```sh
//! c="-n 2 -p RPCL -PLT -mct 0 -M 1 -r 6,3,1.6"
//! opj_compress -i rgb128.ppm -o first.j2k $c -b 32,64
//! opj_compress -i rgb128.ppm -o rest.j2k $c -b 64,64
//! ```
//!
//! `bypass_rgb_53_blocks_per_component.j2k` is the main header of
//! `rest.j2k`, without its comment and with a COC marker after its COD that
//! gives component 0 the first encode's code-block size (`Scoc` 0, then
//! `SPcoc` as in its COD); then one tile-part without packet lengths, whose
//! packets, in their RPCL order, are the first encode's for component 0
//! and the second's for the others, as their packet lengths split them.
//! `opj_decompress` decodes its first component as the first encode's and
//! the others as the second's.
//!
//! `-M` adds 1 for selective bypass, 2 for context reset, 4 for termination
//! on each pass, 8 for the vertically causal context, 16 for predictable
//! termination and 32 for segmentation symbols. The samples contain no
//! externally sourced image content.

use alloc::vec::Vec;
use core::cell::Cell;

use crate::{
    ComponentPlane, ComponentRowSink, CpuDecodeParallelism, DecodeSettings, DecoderContext, Image,
};

const FIXTURES: [(&str, &[u8]); 11] = [
    (
        "bypass gray 5/3",
        include_bytes!("../../fixtures/multi_segment/bypass_gray_53.j2k"),
    ),
    (
        "bypass rgb 9/7",
        include_bytes!("../../fixtures/multi_segment/bypass_rgb_97.j2k"),
    ),
    (
        "termall gray 9/7",
        include_bytes!("../../fixtures/multi_segment/termall_gray_97.j2k"),
    ),
    (
        "termall rgb 5/3",
        include_bytes!("../../fixtures/multi_segment/termall_rgb_53.j2k"),
    ),
    (
        "reset rgb 9/7",
        include_bytes!("../../fixtures/multi_segment/reset_rgb_97.j2k"),
    ),
    (
        "segmark gray 5/3",
        include_bytes!("../../fixtures/multi_segment/segmark_gray_53.j2k"),
    ),
    (
        "bypass and termall gray 9/7",
        include_bytes!("../../fixtures/multi_segment/bypass_termall_gray_97.j2k"),
    ),
    (
        "bypass and termall rgb 5/3",
        include_bytes!("../../fixtures/multi_segment/bypass_termall_rgb_53.j2k"),
    ),
    (
        "all modes gray 9/7",
        include_bytes!("../../fixtures/multi_segment/all_modes_gray_97.j2k"),
    ),
    (
        "bypass rgb 9/7, 64 x 64 blocks",
        include_bytes!("../../fixtures/multi_segment/bypass_rgb_97_64x64.j2k"),
    ),
    (
        "bypass rgb 5/3, code-block size per component",
        include_bytes!("../../fixtures/multi_segment/bypass_rgb_53_blocks_per_component.j2k"),
    ),
];

/// Collects the sample bits a sink receives into region-sized planes, and
/// checks that each sample arrives once.
struct Rows {
    width: usize,
    planes: Vec<Vec<u32>>,
    arrivals: Vec<u8>,
}

impl ComponentRowSink for Rows {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        if self.planes.is_empty() {
            self.planes = alloc::vec![alloc::vec![0; self.arrivals.len()]; planes.len()];
        }
        let (width, height) = planes[0].dimensions();
        for (component, plane) in planes.iter().enumerate() {
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

/// What a decode wrote, and where it ran.
struct Decoded {
    samples: Vec<Vec<u32>>,
    streamed: bool,
    pool_components: usize,
    parted_bands: usize,
    uncharged_line_samples: usize,
    channel_bytes: usize,
    /// The bytes the decode's budget held after `reset`.
    baseline_bytes: usize,
}

fn context(parallelism: CpuDecodeParallelism) -> DecoderContext<'static> {
    let mut context = DecoderContext::default();
    context.set_cpu_decode_parallelism(parallelism);
    context
}

/// The region's rows, streamed to a sink where the decode can.
fn rows(image: &Image<'_>, roi: (u32, u32, u32, u32)) -> Decoded {
    rows_with(image, roi, |_| {})
}

/// [`rows`] from a context `configure` sets up.
fn rows_with(
    image: &Image<'_>,
    roi: (u32, u32, u32, u32),
    configure: impl FnOnce(&mut DecoderContext<'_>),
) -> Decoded {
    let mut context = context(CpuDecodeParallelism::Auto);
    configure(&mut context);
    let mut sink = Rows {
        width: roi.2 as usize,
        planes: Vec::new(),
        arrivals: alloc::vec![0; (roi.2 * roi.3) as usize],
    };
    image
        .decode_region_rows_for_integer_output_with_context(roi, &mut context, &mut sink)
        .expect("decode region rows");
    assert!(
        sink.arrivals.iter().all(|&count| count == 1),
        "every sample arrives once"
    );
    let counters = context.tile_decode_context.debug_counters;
    Decoded {
        samples: sink.planes,
        streamed: context.tile_decode_context.stream_rows,
        pool_components: counters.pool_band_components,
        parted_bands: counters.pool_band_parts,
        uncharged_line_samples: counters.uncharged_line_samples,
        channel_bytes: counters.workspace_peak.bytes().channels,
        baseline_bytes: counters.decode_baseline_bytes,
    }
}

/// The region's component planes from a serial decode.
fn planes(image: &Image<'_>, roi: (u32, u32, u32, u32)) -> Vec<Vec<u32>> {
    let mut context = context(CpuDecodeParallelism::Serial);
    let components = image
        .decode_region_components_for_integer_output_with_context(roi, &mut context)
        .expect("decode region planes");
    components
        .planes()
        .iter()
        .map(|plane| {
            plane.samples()[..(roi.2 * roi.3) as usize]
                .iter()
                .map(|sample| sample.to_bits())
                .collect()
        })
        .collect()
}

/// The image's native samples, whole or of `roi`, with Auto parallelism.
fn native(image: &Image<'_>, roi: Option<(u32, u32, u32, u32)>) -> Vec<u8> {
    let mut context = context(CpuDecodeParallelism::Auto);
    match roi {
        Some(roi) => image.decode_native_region_with_context(roi, &mut context),
        None => image.decode_native_with_context(&mut context),
    }
    .expect("decode native samples")
    .data
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("build pool")
}

fn regions(image: &Image<'_>) -> [(u32, u32, u32, u32); 3] {
    let (width, height) = (image.width(), image.height());
    [
        (0, 0, width, height),
        (width / 5, height / 7, width / 2, height / 3),
        (width / 3, 1, width / 4, height - 2),
    ]
}

#[test]
fn multi_segment_blocks_decode_on_a_pool_as_on_one_thread() {
    let one = pool(1);
    for (label, bytes) in FIXTURES {
        for reduction in [0_u8, 1] {
            let image = Image::new_with_reduction(bytes, &DecodeSettings::default(), reduction)
                .expect("parse fixture");
            let regions = regions(&image);
            let expected_full = one.install(|| native(&image, None));
            let expected: Vec<_> = regions
                .iter()
                .map(|&roi| one.install(|| (native(&image, Some(roi)), rows(&image, roi))))
                .collect();
            for (&roi, (_, rows)) in regions.iter().zip(&expected) {
                assert_eq!(
                    rows.samples,
                    planes(&image, roi),
                    "{label} 1/{} {roi:?}: rows on one thread against the planes",
                    1 << reduction
                );
            }
            for threads in [2, 4, 12] {
                pool(threads).install(|| {
                    assert!(
                        native(&image, None) == expected_full,
                        "{label} 1/{} on {threads} threads: full image",
                        1 << reduction
                    );
                    for (&roi, (expected_native, expected_rows)) in regions.iter().zip(&expected) {
                        let context = format!("{label} 1/{} {roi:?}", 1 << reduction);
                        assert!(
                            native(&image, Some(roi)) == *expected_native,
                            "{context} on {threads} threads: region"
                        );
                        assert_eq!(
                            rows(&image, roi).samples,
                            expected_rows.samples,
                            "{context} on {threads} threads: rows"
                        );
                    }
                });
            }
        }
    }
}

/// On a pool, these tiles decode on the pool's own routes: the tile route
/// for the planes, and pool bands for streamed rows, which then hold no
/// planes and leave no line-buffer samples uncharged.
#[test]
fn multi_segment_tiles_take_the_pool_routes() {
    let mut parted = 0;
    for threads in [4, 12] {
        pool(threads).install(|| {
            for (label, bytes) in FIXTURES {
                let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
                let mut context = context(CpuDecodeParallelism::Auto);
                image
                    .decode_native_with_context(&mut context)
                    .expect("decode fixture");
                assert!(
                    context
                        .tile_decode_context
                        .debug_counters
                        .classic_task_workspace_growths
                        > 0,
                    "{label} on {threads} threads: the planes decode on the tile route"
                );
                for roi in regions(&image) {
                    let decoded = rows(&image, roi);
                    let components = decoded.samples.len();
                    assert!(
                        decoded.streamed
                            && decoded.pool_components == components
                            && decoded.channel_bytes <= 256,
                        "{label} {roi:?} on {threads} threads: rows from pool bands, no \
                         planes: streamed {}, pool components {} of {components}, plane bytes {}",
                        decoded.streamed,
                        decoded.pool_components,
                        decoded.channel_bytes,
                    );
                    assert_eq!(decoded.uncharged_line_samples, 0, "{label} {roi:?}");
                    parted += decoded.parted_bands;
                }
            }
        });
    }
    assert!(parted > 0, "some pool bands decode in parts");
}

/// A full-image rows decode on a pool whose tiles only serial bands can
/// stream (here the pool's routes are switched off, as for a tile they
/// can't decode) streams from serial bands when its planes would pass the
/// decode budget, rather than fail on them, and keeps the planes within it.
/// With the pool's routes on, its bands stream these tiles within the
/// budget.
#[test]
fn rows_stream_from_serial_bands_when_the_planes_pass_the_budget() {
    for (label, bytes) in FIXTURES {
        let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
        let roi = (0, 0, image.width(), image.height());
        let expected = pool(1).install(|| rows(&image, roi));
        pool(4).install(|| {
            let planes = rows_with(&image, roi, |context| {
                context.tile_decode_context.force_per_sub_band = true;
            });
            assert!(!planes.streamed, "{label}: planes within the budget");
            assert_eq!(planes.samples, expected.samples, "{label}: planes");
            // `reset` charged the structural bytes and the planes: the
            // planes pass this budget, which leaves room for a packet cache
            // of up to half their size, which a decode past the budget
            // drops to try again.
            let budget = planes.baseline_bytes - planes.channel_bytes / 2;
            let streamed = rows_with(&image, roi, |context| {
                context.tile_decode_context.force_per_sub_band = true;
                context.tile_decode_context.planes_budget = Some(budget);
            });
            assert!(
                streamed.streamed && streamed.pool_components == 0 && streamed.channel_bytes <= 256,
                "{label}: serial bands past the budget: streamed {}, pool components {}, plane \
                 bytes {}",
                streamed.streamed,
                streamed.pool_components,
                streamed.channel_bytes,
            );
            assert_eq!(streamed.samples, expected.samples, "{label}: serial bands");
            // The pool's bands take these tiles, within the budget too.
            let pooled = rows_with(&image, roi, |context| {
                context.tile_decode_context.planes_budget = Some(budget);
            });
            let components = pooled.samples.len();
            assert!(
                pooled.streamed && pooled.pool_components == components,
                "{label}: pool bands within the budget"
            );
            assert_eq!(pooled.samples, expected.samples, "{label}: pool bands");
        });
    }
}

/// What the codeword segments of an image's classic code-blocks hold, as
/// its packets give them.
#[derive(Clone, Copy, Debug, Default)]
struct Segments {
    /// Blocks of several segments.
    segmented: usize,
    /// Raw segments, and raw segments an arithmetic one follows.
    raw: usize,
    raw_then_arithmetic: usize,
    /// Segments that hold coding passes and no bytes.
    empty: usize,
}

std::thread_local! {
    /// Whether the image [`segments`] reads codes with selective bypass, and
    /// in one codeword segment per block, and what it found in its tile.
    static STYLE: Cell<(bool, bool)> = const { Cell::new((false, true)) };
    static FOUND: Cell<Option<Segments>> = const { Cell::new(None) };
}

/// The codeword segments of `image`'s classic code-blocks, read from its
/// tile's records once the tile route on a pool has decoded them.
fn segments(image: &Image<'_>) -> Segments {
    let style = image
        .header
        .global_coding_style
        .component_parameters
        .parameters
        .code_block_style;
    pool(4).install(|| {
        STYLE.with(|cell| {
            cell.set((
                style.selective_arithmetic_coding_bypass,
                style.single_codeword_segment(),
            ));
        });
        let mut context = context(CpuDecodeParallelism::Auto);
        context.tile_decode_context.tile_route_post_decode = Some(|storage| {
            // Past the first ten passes, the passes but the cleanup ones
            // are raw. A block of one segment numbers its layers'
            // contributions instead.
            let (bypass, single) = STYLE.with(Cell::get);
            let raw = |pass: u8| bypass && pass >= 10 && !pass.is_multiple_of(3);
            let mut found = Segments::default();
            for block in &storage.code_blocks {
                // Each segment's first pass, passes and bytes.
                let mut segments: Vec<(u8, u8, usize)> = Vec::new();
                let (mut index, mut passes) = (None, 0_u8);
                for layer in &storage.layers[block.layers.clone()] {
                    let Some(range) = layer.segments() else {
                        continue;
                    };
                    for record in &storage.segments[range] {
                        if index != Some(record.idx) && !(single && index.is_some()) {
                            index = Some(record.idx);
                            segments.push((passes, 0, 0));
                        }
                        if let Some(segment) = segments.last_mut() {
                            segment.1 += record.coding_pases;
                            segment.2 += record.data.len();
                        }
                        passes += record.coding_pases;
                    }
                }
                found.segmented += usize::from(segments.len() > 1);
                for (at, &(first, passes, bytes)) in segments.iter().enumerate() {
                    if raw(first) {
                        found.raw += 1;
                        found.raw_then_arithmetic += usize::from(
                            segments.get(at + 1).is_some_and(|&(next, _, _)| !raw(next)),
                        );
                    }
                    found.empty += usize::from(passes > 0 && bytes == 0);
                }
            }
            FOUND.with(|cell| cell.set(Some(found)));
            Ok(())
        });
        image
            .decode_native_with_context(&mut context)
            .expect("decode fixture");
        FOUND
            .with(Cell::take)
            .expect("the tile route decoded the tile")
    })
}

/// The fixtures' blocks hold every kind of codeword segment: blocks of
/// several in every fixture whose style makes them, raw segments and an
/// arithmetic segment after them in every one with selective bypass, and
/// some segments with coding passes and no bytes.
#[test]
fn fixtures_hold_every_kind_of_codeword_segment() {
    let mut empty = 0;
    for (label, bytes) in FIXTURES {
        let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
        let style = image
            .header
            .global_coding_style
            .component_parameters
            .parameters
            .code_block_style;
        let found = segments(&image);
        assert_eq!(
            found.segmented > 0,
            !style.single_codeword_segment(),
            "{label}: {found:?}"
        );
        if style.selective_arithmetic_coding_bypass {
            assert!(
                found.raw > 0 && found.raw_then_arithmetic > 0,
                "{label}: {found:?}"
            );
        }
        empty += found.empty;
    }
    assert!(empty > 0, "some segments hold coding passes and no bytes");
}
