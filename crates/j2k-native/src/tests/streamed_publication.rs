// SPDX-License-Identifier: MIT OR Apache-2.0

//! A streamed decode hands its rows to the sink as it finishes them, so a
//! decode that fails partway leaves some rows written. Which rows that is
//! depends on the route: rows one at a time on serial bands and with one
//! pool reader, whole batches with a reader per component. Whatever the
//! route, every row the sink receives must be the row a clean decode
//! produces, received once and in order from the top, with nothing after a
//! row that is missing; the error must be the one the failure raises; and a
//! decode that runs out of budget must fail before its first row, so that
//! the retries that rerun it (without the retained scratch, or without the
//! image's packet directory) still can.

use alloc::vec;
use alloc::vec::Vec;

use super::banded_tier1::{noisy_pixels, HEIGHT, WIDTH};
use crate::error::{DecodeError, DecodingError};
use crate::{
    encode, ComponentPlane, ComponentRowSink, CpuDecodeParallelism, DecodeSettings, DecoderContext,
    EncodeOptions, Image,
};

/// What an RGB8 destination holds where nothing was written.
const UNWRITTEN: u8 = 0xa5;

/// The rows a sink received, as sample bits per component, with the number
/// of times each pixel arrived; or, with a destination, the RGB8 bytes the
/// decoder packed into it. Each time the decoder asks for the destination,
/// the rows written since the last time are counted, moved out and the
/// destination blanked again, so a row written twice counts twice even with
/// the same bytes.
struct Published {
    width: usize,
    height: usize,
    planes: Vec<Vec<u32>>,
    arrivals: Vec<u8>,
    rgb: Option<Vec<u8>>,
    received: Vec<u8>,
}

impl Published {
    fn new((width, height): (u32, u32), destination: bool) -> Self {
        let samples = (width * height) as usize;
        Self {
            width: width as usize,
            height: height as usize,
            planes: vec![vec![0; samples]; 3],
            arrivals: vec![0; samples],
            rgb: destination.then(|| vec![UNWRITTEN; samples * 3]),
            received: vec![UNWRITTEN; if destination { samples * 3 } else { 0 }],
        }
    }

    /// Count and move out the destination's pixels written since the last
    /// call: of every row whose first or last pixel was written (the
    /// decoder writes whole rows of a tile this wide), or with `all`, of
    /// every row.
    fn collect(&mut self, all: bool) {
        let Some(rgb) = self.rgb.as_mut() else {
            return;
        };
        let blank = |pixel: &[u8]| pixel.iter().all(|&byte| byte == UNWRITTEN);
        for (y, row) in rgb.chunks_exact_mut(self.width * 3).enumerate() {
            if !all && blank(&row[..3]) && blank(&row[row.len() - 3..]) {
                continue;
            }
            for (x, pixel) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                if !blank(pixel) {
                    let at = y * self.width + x;
                    self.arrivals[at] += 1;
                    self.received[at * 3..][..3].copy_from_slice(pixel);
                    pixel.fill(UNWRITTEN);
                }
            }
        }
    }

    /// The rows from the top that hold the clean decode's samples, checking
    /// that every other row holds nothing and that no pixel arrived twice.
    fn rows(&self, clean: &Self, label: &str) -> usize {
        let mut published = 0;
        for y in 0..self.height {
            let label = format!("{label}: row {y}");
            if self.row_written(clean, y, &label) {
                assert_eq!(published, y, "{label} after a missing row");
                published += 1;
            }
        }
        published
    }

    /// Whether row `y` arrived, checking that all of it arrived at most once
    /// and that, if it arrived, it holds the clean row's samples or bytes.
    fn row_written(&self, clean: &Self, y: usize, label: &str) -> bool {
        let row = y * self.width..(y + 1) * self.width;
        let arrivals = &self.arrivals[row.clone()];
        assert!(
            arrivals.iter().all(|&count| count == arrivals[0]) && arrivals[0] <= 1,
            "{label} arrived {arrivals:?}"
        );
        let written = arrivals[0] == 1;
        if written && self.rgb.is_some() {
            let bytes = row.start * 3..row.end * 3;
            assert_eq!(
                self.received[bytes.clone()],
                clean.received[bytes],
                "{label}"
            );
        } else if written {
            for (plane, clean) in self.planes.iter().zip(&clean.planes) {
                assert_eq!(plane[row.clone()], clean[row.clone()], "{label}");
            }
        }
        written
    }

    fn untouched(&self) -> bool {
        self.arrivals.iter().all(|&count| count == 0)
    }
}

impl ComponentRowSink for Published {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        let (width, height) = planes[0].dimensions();
        for row in 0..height as usize {
            for column in 0..width as usize {
                let at = (y as usize + row) * self.width + x as usize + column;
                for (component, plane) in planes.iter().enumerate() {
                    self.planes[component][at] =
                        plane.samples()[row * width as usize + column].to_bits();
                }
                self.arrivals[at] += 1;
            }
        }
    }

    fn rgb8_region(&mut self, width: u32, height: u32) -> Option<&mut [u8]> {
        self.collect(false);
        let len = (width as usize * 3).checked_mul(height as usize)?;
        self.rgb.as_mut().map(|rgb| &mut rgb[..len])
    }
}

/// A fixture with many bands per component: 8-row code-blocks over three
/// levels of three noisy components.
fn fixture(reversible: bool, use_ht_block_coding: bool, rpcl: bool) -> Vec<u8> {
    encode(
        &noisy_pixels(3),
        WIDTH,
        HEIGHT,
        3,
        8,
        false,
        &EncodeOptions {
            reversible,
            use_ht_block_coding,
            num_decomposition_levels: 3,
            code_block_height_exp: 1,
            progression_order: if rpcl {
                crate::EncodeProgressionOrder::Rpcl
            } else {
                EncodeOptions::default().progression_order
            },
            ..EncodeOptions::default()
        },
    )
    .expect("encode publication fixture")
}

/// How a streamed decode runs: on serial bands, or on a pool of this many
/// threads (one reader below 12 for classic tiles, a reader per component
/// from 12 or for HTJ2K ones).
#[derive(Clone, Copy, Debug)]
enum Route {
    Serial,
    #[cfg(feature = "parallel")]
    Pool(usize),
}

fn routes() -> Vec<Route> {
    #[cfg(feature = "parallel")]
    return vec![Route::Serial, Route::Pool(4), Route::Pool(12)];
    #[cfg(not(feature = "parallel"))]
    return vec![Route::Serial];
}

struct Outcome {
    result: crate::Result<()>,
    sink: Published,
    streamed_samples: usize,
    #[cfg_attr(not(feature = "parallel"), expect(dead_code, reason = "pool counters"))]
    readers: usize,
    #[cfg_attr(not(feature = "parallel"), expect(dead_code, reason = "pool counters"))]
    fallbacks: usize,
}

/// Stream the whole image along `route`, into a destination or as rows, on
/// a context `configure` sets up, from `baseline` bytes.
fn stream<'a>(
    image: &Image<'a>,
    route: Route,
    destination: bool,
    baseline: usize,
    configure: impl FnOnce(&mut DecoderContext<'a>) + Send,
) -> Outcome {
    stream_region(image, route, FULL, destination, baseline, configure)
}

/// The whole image, as a region.
const FULL: (u32, u32, u32, u32) = (0, 0, WIDTH, HEIGHT);

/// Stream `region` of the image along `route`, as [`stream`] does.
fn stream_region<'a>(
    image: &Image<'a>,
    route: Route,
    region: (u32, u32, u32, u32),
    destination: bool,
    baseline: usize,
    configure: impl FnOnce(&mut DecoderContext<'a>) + Send,
) -> Outcome {
    let run = || {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(match route {
            Route::Serial => CpuDecodeParallelism::Serial,
            #[cfg(feature = "parallel")]
            Route::Pool(_) => CpuDecodeParallelism::Auto,
        });
        configure(&mut context);
        let mut sink = Published::new((region.2, region.3), destination);
        let result = image
            .run_decode_rows(
                &mut context,
                Some(region),
                None,
                true,
                baseline,
                Some(&mut sink),
            )
            .map(drop);
        sink.collect(true);
        let tile_ctx = &context.tile_decode_context;
        // A decode past its budget may fail before it plans its rows.
        assert!(
            tile_ctx.stream_rows || result.is_err(),
            "{route:?}: the decode streams"
        );
        #[cfg(feature = "parallel")]
        let (readers, fallbacks) = (
            tile_ctx.debug_counters.pool_band_readers,
            tile_ctx.debug_counters.pool_band_fallbacks,
        );
        #[cfg(not(feature = "parallel"))]
        let (readers, fallbacks) = (0, 0);
        Outcome {
            result,
            sink,
            streamed_samples: tile_ctx.streamed_samples,
            readers,
            fallbacks,
        }
    };
    match route {
        Route::Serial => run(),
        #[cfg(feature = "parallel")]
        Route::Pool(threads) => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("build pool")
            .install(run),
    }
}

fn is_band_failure(result: &crate::Result<()>) -> bool {
    matches!(
        result,
        Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
    )
}

/// A band decode that fails partway through a streamed decode leaves only
/// clean rows, once and from the top, and the failure's error: none before
/// the first band, rows one at a time on serial bands and with one pool
/// reader, whole batches with a reader per component. On serial bands a
/// later failure leaves no fewer rows; on the pool, which band fails is the
/// one a worker happens to take at that count.
#[test]
fn a_failing_band_leaves_only_clean_rows_from_the_top() {
    for (reversible, ht) in [(false, false), (true, false), (false, true)] {
        let bytes = fixture(reversible, ht, false);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        for route in routes() {
            for destination in [false, true] {
                let clean = stream(&image, route, destination, 0, |_| {});
                clean.result.expect("clean decode");
                assert_eq!(clean.sink.rows(&clean.sink, "clean"), HEIGHT as usize);
                #[cfg(feature = "parallel")]
                if let Route::Pool(threads) = route {
                    let readers = if ht || threads >= 12 { 3 } else { 1 };
                    assert_eq!(clean.readers, readers, "{route:?} ht={ht}: readers");
                }
                let mut published = Vec::new();
                for after in [0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 10_000] {
                    let label = format!(
                        "reversible={reversible} ht={ht} {route:?} destination={destination} \
                         failing band {after}"
                    );
                    let outcome = stream(&image, route, destination, 0, |context| {
                        context.tile_decode_context.fail_band_after = Some(after);
                    });
                    let rows = outcome.sink.rows(&clean.sink, &label);
                    if outcome.result.is_ok() {
                        // Past the tile's last band nothing fails.
                        assert_eq!(rows, HEIGHT as usize, "{label}");
                        continue;
                    }
                    assert!(
                        is_band_failure(&outcome.result),
                        "{label}: {:?}",
                        outcome.result
                    );
                    published.push(rows);
                }
                let label = format!("reversible={reversible} ht={ht} {route:?} {destination}");
                assert_eq!(
                    published.first(),
                    Some(&0),
                    "{label}: the first band fails first"
                );
                if matches!(route, Route::Serial) {
                    assert!(
                        published.windows(2).all(|pair| pair[0] <= pair[1]),
                        "{label}: later failures leave no fewer rows: {published:?}"
                    );
                    assert!(
                        published
                            .iter()
                            .any(|&rows| rows > 0 && rows < HEIGHT as usize),
                        "{label}: some failure comes partway: {published:?}"
                    );
                }
            }
        }
    }
}

/// A band the pool leaves running with no worker to finish it is a
/// scheduler fault. Before any row reaches the sink, the tile decodes again
/// on serial bands, as fast decodes do; after, the rows can't be taken
/// back, so the decode fails, having handed over only clean rows, once.
/// Which rows precede a strand depends on how far the workers ran ahead, so
/// each kind of tile must show both outcomes over its pools and outputs.
#[cfg(feature = "parallel")]
#[test]
fn a_band_stranded_after_published_rows_ends_the_decode() {
    for ht in [false, true] {
        let bytes = fixture(false, ht, false);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        let (mut fell_back, mut failed_after_rows) = (0, 0);
        for threads in [4, 12] {
            let route = Route::Pool(threads);
            for destination in [false, true] {
                let clean = stream(&image, route, destination, 0, |_| {});
                clean.result.expect("clean decode");
                for after in [
                    0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
                ] {
                    let label = format!(
                        "ht={ht} threads={threads} destination={destination} stranded task {after}"
                    );
                    let outcome = stream(&image, route, destination, 0, |context| {
                        context.tile_decode_context.strand_pool_band = true;
                        context.tile_decode_context.strand_pool_band_after = after;
                    });
                    let rows = outcome.sink.rows(&clean.sink, &label);
                    if outcome.result.is_ok() {
                        assert_eq!(rows, HEIGHT as usize, "{label}");
                        fell_back += usize::from(outcome.fallbacks == 1);
                    } else {
                        assert!(is_band_failure(&outcome.result), "{label}");
                        assert_eq!(outcome.fallbacks, 1, "{label}");
                        assert!(rows > 0, "{label}: stranded before any row, yet failed");
                        failed_after_rows += 1;
                    }
                }
            }
        }
        assert!(
            fell_back > 0,
            "ht={ht}: a strand before the first row falls back"
        );
        assert!(
            failed_after_rows > 0,
            "ht={ht}: a strand after published rows fails"
        );
    }
}

/// The largest baseline at which a fresh decode of `region` along `route`
/// fits its budget.
fn fitting_baseline(
    image: &Image<'_>,
    route: Route,
    region: (u32, u32, u32, u32),
    destination: bool,
) -> usize {
    let (mut fits, mut fails) = (0, crate::DEFAULT_MAX_DECODE_BYTES);
    while fails - fits > 1 {
        let candidate = fits + (fails - fits) / 2;
        let outcome = stream_region(image, route, region, destination, candidate, |_| {});
        match outcome.result {
            Ok(()) => fits = candidate,
            Err(_) => fails = candidate,
        }
    }
    fits
}

/// A streamed decode charges everything it holds before its first row, so
/// whatever budget it starts from, a capacity failure comes before any row
/// reaches the sink: the retained-scratch and packet-directory retries,
/// which rerun only a decode that has handed nothing over, stay open on
/// every route. A decode that succeeds through either retry still does,
/// with every row once. Besides the whole image, a region that clips the
/// code-blocks of its last rows, which decode only after its first rows
/// went out.
#[test]
fn capacity_failures_come_before_the_first_row() {
    for ht in [false, true] {
        let bytes = fixture(false, ht, true);
        for region in [FULL, (0, 0, WIDTH, HEIGHT - 11), (0, 0, WIDTH, 61)] {
            for route in routes() {
                for destination in [false, true] {
                    let label = format!("ht={ht} {region:?} {route:?} destination={destination}");
                    capacity_failures_leave_the_sink_untouched(
                        &bytes,
                        route,
                        region,
                        destination,
                        &label,
                    );
                }
            }
        }
    }
}

fn capacity_failures_leave_the_sink_untouched(
    bytes: &[u8],
    route: Route,
    region: (u32, u32, u32, u32),
    destination: bool,
    label: &str,
) {
    let rows = region.3 as usize;
    let fresh = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
    let clean = stream_region(&fresh, route, region, destination, 0, |_| {});
    clean.result.expect("clean decode");
    let fits = fitting_baseline(&fresh, route, region, destination);
    // Past the largest budget that fits, one that leaves out the optional
    // row index can fit again: each decodes every row once or fails first.
    let mut failures = 0;
    for over in [
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
        1 << 18,
    ] {
        let outcome = stream_region(&fresh, route, region, destination, fits + over, |_| {});
        let label = format!("{label} +{over}");
        match outcome.result {
            Ok(()) => assert_eq!(outcome.sink.rows(&clean.sink, &label), rows),
            Err(error) => {
                assert!(crate::j2c::is_capacity_error(&error), "{label}: {error:?}");
                assert_eq!(outcome.streamed_samples, 0, "{label}");
                assert!(outcome.sink.untouched(), "{label}: rows before the failure");
                failures += 1;
            }
        }
    }
    assert!(failures > 0, "{label}: some budget fails");
    for under in [0, 1, 1 << 10, 1 << 16] {
        let outcome = stream_region(&fresh, route, region, destination, fits - under, |_| {});
        outcome.result.expect("within the budget");
        assert_eq!(outcome.sink.rows(&clean.sink, label), rows);
    }

    // A context holding the scratch of a planes decode starts over the
    // budget; the retry without it fits.
    let mut reused = DecoderContext::default();
    drop(
        fresh
            .decode_region_components_for_integer_output_with_context(region, &mut reused)
            .expect("planes decode"),
    );
    assert!(reused.tile_decode_context.retained_scratch_bytes().unwrap() > 0);
    let outcome = stream_region(&fresh, route, region, destination, fits, |context| {
        core::mem::swap(context, &mut reused);
        context.set_cpu_decode_parallelism(match route {
            Route::Serial => CpuDecodeParallelism::Serial,
            #[cfg(feature = "parallel")]
            Route::Pool(_) => CpuDecodeParallelism::Auto,
        });
    });
    outcome
        .result
        .expect("a reused context fits as a fresh one");
    assert_eq!(outcome.sink.rows(&clean.sink, label), rows);

    // An image whose repeated regions built a packet directory gives it up
    // for a request that fits without it.
    #[cfg(feature = "std")]
    {
        let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
        let metadata = image.retained_allocation_bytes().expect("metadata bytes");
        for _ in 0..2 {
            stream_region(&image, route, region, destination, metadata, |_| {})
                .result
                .expect("decode with the directory");
        }
        assert!(
            image.retained_allocation_bytes().expect("retained bytes") > metadata,
            "{label}: repeated decodes build a directory"
        );
        let outcome = stream_region(&image, route, region, destination, fits, |_| {});
        outcome
            .result
            .expect("evicting the directory fits the request");
        assert_eq!(outcome.sink.rows(&clean.sink, label), rows);
        assert_eq!(
            image.retained_allocation_bytes().expect("retained bytes"),
            metadata,
            "{label}: the directory is evicted"
        );
    }
}
