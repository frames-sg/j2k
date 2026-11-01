// SPDX-License-Identifier: MIT OR Apache-2.0

//! A bounded row decode of an image whose tiles span its width runs as one
//! region decode of the whole image that streams its rows and hands them
//! over a stripe at a time; other images decode each stripe as a region of
//! its own. Either way the rows must be the bytes `decode_into` writes, in
//! order and once each, for every stripe height, pool and sample depth, and
//! a sink that fails must stop the rows at the row it refused and be
//! reported as the sink's error.

use core::fmt;

use j2k::{CpuDecodeParallelism, DecodeRowsError, J2kDecoder, J2kRowDecodeOptions, RowSink};
use j2k_core::PixelFormat;
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::wrap_jp2_rgba_codestream;
use rayon::ThreadPoolBuilder;

const WIDTH: u32 = 203;
const HEIGHT: u32 = 157;

fn noisy_samples(components: u32, bit_depth: u8) -> Vec<u8> {
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let samples = WIDTH * HEIGHT * components;
    if bit_depth <= 8 {
        return (0..samples).map(|_| (next() >> 24) as u8).collect();
    }
    let max = (1_u32 << bit_depth) - 1;
    (0..samples)
        .flat_map(|_| {
            u16::try_from(next() % (max + 1))
                .expect("sample")
                .to_le_bytes()
        })
        .collect()
}

fn codestream(components: u16, bit_depth: u8, options: &EncodeOptions) -> Vec<u8> {
    encoded(components, bit_depth, false, options)
}

fn encoded(components: u16, bit_depth: u8, signed: bool, options: &EncodeOptions) -> Vec<u8> {
    let samples = noisy_samples(u32::from(components), bit_depth);
    encode(
        &samples, WIDTH, HEIGHT, components, bit_depth, signed, options,
    )
    .expect("encode fixture")
}

/// A sink error carrying the row the sink refused.
#[derive(Debug, PartialEq, Eq)]
struct Refused(u32);

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "row {} refused", self.0)
    }
}

impl std::error::Error for Refused {}

/// The rows a sink took, as little-endian bytes, and every row index it
/// was given, refusing the row `refuse` when it is set.
#[derive(Default)]
struct Rows {
    bytes: Vec<u8>,
    order: Vec<u32>,
    refuse: Option<u32>,
}

impl Rows {
    fn take(&mut self, y: u32) -> Result<(), Refused> {
        self.order.push(y);
        if self.refuse == Some(y) {
            return Err(Refused(y));
        }
        Ok(())
    }
}

impl RowSink<u8> for Rows {
    type Error = Refused;

    fn write_row(&mut self, y: u32, row: &[u8]) -> Result<(), Refused> {
        self.take(y)?;
        self.bytes.extend_from_slice(row);
        Ok(())
    }
}

impl RowSink<u16> for Rows {
    type Error = Refused;

    fn write_row(&mut self, y: u32, row: &[u16]) -> Result<(), Refused> {
        self.take(y)?;
        self.bytes
            .extend(row.iter().flat_map(|sample| sample.to_le_bytes()));
        Ok(())
    }
}

fn fmt_for(components: u16, wide: bool) -> PixelFormat {
    match (components, wide) {
        (1, false) => PixelFormat::Gray8,
        (1, true) => PixelFormat::Gray16,
        (4, false) => PixelFormat::Rgba8,
        (4, true) => PixelFormat::Rgba16,
        (_, false) => PixelFormat::Rgb8,
        (_, true) => PixelFormat::Rgb16,
    }
}

fn decode_into(bytes: &[u8], parallelism: CpuDecodeParallelism, fmt: PixelFormat) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    let stride = WIDTH as usize * fmt.bytes_per_pixel();
    let mut out = vec![0_u8; stride * HEIGHT as usize];
    decoder
        .decode_into(&mut out, stride, fmt)
        .expect("full decode");
    out
}

fn decode_rows(
    bytes: &[u8],
    parallelism: CpuDecodeParallelism,
    wide: bool,
    stripe_rows: u32,
    sink: &mut Rows,
) -> Result<(), DecodeRowsError<j2k::J2kError, Refused>> {
    let mut decoder = J2kDecoder::new(bytes).expect("decoder");
    decoder.set_cpu_decode_parallelism(parallelism);
    decode_rows_with(&mut decoder, wide, stripe_rows, sink)
}

fn decode_rows_with(
    decoder: &mut J2kDecoder<'_>,
    wide: bool,
    stripe_rows: u32,
    sink: &mut Rows,
) -> Result<(), DecodeRowsError<j2k::J2kError, Refused>> {
    let options = J2kRowDecodeOptions::new(stripe_rows);
    let outcome = if wide {
        decoder.decode_rows_u16_bounded(sink, options)
    } else {
        decoder.decode_rows_u8_bounded(sink, options)
    }?;
    assert_eq!(outcome.decoded, j2k_core::Rect::full((WIDTH, HEIGHT)));
    Ok(())
}

struct Fixture {
    name: &'static str,
    bytes: Vec<u8>,
    components: u16,
    /// Whether 16-bit rows are compared too.
    wide: bool,
    /// Whether the rows decode in one streamed region decode, rather than
    /// a stripe at a time.
    streams: bool,
}

fn fixtures() -> Vec<Fixture> {
    let base = EncodeOptions {
        num_decomposition_levels: 4,
        ..EncodeOptions::default()
    };
    let lossy = EncodeOptions {
        reversible: false,
        ..base.clone()
    };
    let rgba = codestream(4, 8, &base);
    vec![
        Fixture {
            name: "rgb 9/7",
            bytes: codestream(3, 8, &lossy),
            components: 3,
            wide: true,
            streams: true,
        },
        Fixture {
            name: "rgb 5/3 HT",
            bytes: codestream(
                3,
                8,
                &EncodeOptions {
                    use_ht_block_coding: true,
                    ..base.clone()
                },
            ),
            components: 3,
            wide: true,
            streams: true,
        },
        Fixture {
            name: "gray 12-bit 9/7",
            bytes: codestream(1, 12, &lossy),
            components: 1,
            wide: true,
            streams: true,
        },
        Fixture {
            name: "gray signed 12-bit 5/3",
            bytes: encoded(1, 12, true, &base),
            components: 1,
            wide: true,
            streams: true,
        },
        // Tiles as wide as the image: one region decode, a tile after another.
        Fixture {
            name: "rgb tiles spanning the width",
            bytes: codestream(
                3,
                8,
                &EncodeOptions {
                    tile_size: Some((WIDTH, 48)),
                    ..lossy.clone()
                },
            ),
            components: 3,
            wide: true,
            streams: true,
        },
        // Rows of one tile arrive before the next tile's: a stripe at a time.
        Fixture {
            name: "rgb tiles side by side",
            bytes: codestream(
                3,
                8,
                &EncodeOptions {
                    tile_size: Some((64, 64)),
                    ..lossy.clone()
                },
            ),
            components: 3,
            wide: true,
            streams: false,
        },
        // A channel definition keeps region decodes on component planes.
        Fixture {
            name: "rgba with channel definition",
            bytes: wrap_jp2_rgba_codestream(&rgba, WIDTH, HEIGHT, 8),
            components: 4,
            wide: true,
            streams: false,
        },
        // No wavelet levels to stream through.
        Fixture {
            name: "rgb without decomposition levels",
            bytes: codestream(
                3,
                8,
                &EncodeOptions {
                    num_decomposition_levels: 0,
                    ..base.clone()
                },
            ),
            components: 3,
            wide: false,
            streams: false,
        },
    ]
}

fn parallelisms() -> [(&'static str, CpuDecodeParallelism, usize); 4] {
    [
        ("serial", CpuDecodeParallelism::Serial, 1),
        ("auto on 1 thread", CpuDecodeParallelism::Auto, 1),
        ("auto on 4 threads", CpuDecodeParallelism::Auto, 4),
        ("auto on 11 threads", CpuDecodeParallelism::Auto, 11),
    ]
}

#[test]
fn bounded_rows_match_full_decode_at_every_stripe_height_pool_and_depth() {
    let fixtures = fixtures();
    for (pool_name, parallelism, threads) in parallelisms() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            for fixture in &fixtures {
                for wide in [false, true].into_iter().filter(|&wide| !wide || fixture.wide) {
                    let fmt = fmt_for(fixture.components, wide);
                    let expected = decode_into(&fixture.bytes, parallelism, fmt);
                    // One-row stripes of a stripe-at-a-time decode take a
                    // region decode per row: serially only.
                    let one_row = fixture.streams || parallelism == CpuDecodeParallelism::Serial;
                    let heights = [1, 7, 64, HEIGHT, 4 * HEIGHT];
                    for stripe_rows in heights.into_iter().filter(|&rows| rows > 1 || one_row) {
                        let mut sink = Rows::default();
                        decode_rows(&fixture.bytes, parallelism, wide, stripe_rows, &mut sink)
                            .unwrap_or_else(|error| {
                                panic!(
                                    "{} {fmt:?} {stripe_rows} rows, {pool_name}: {error}",
                                    fixture.name
                                )
                            });
                        assert_eq!(
                            sink.order,
                            (0..HEIGHT).collect::<Vec<_>>(),
                            "{} {fmt:?} {stripe_rows} rows, {pool_name}: rows in order, once",
                            fixture.name
                        );
                        assert!(
                            sink.bytes == expected,
                            "{} {fmt:?} {stripe_rows} rows, {pool_name}: rows differ from decode_into",
                            fixture.name
                        );
                    }
                }
            }
        });
    }
}

#[test]
fn a_refused_row_stops_the_rows_and_is_reported_as_the_sinks_error() {
    let fixtures = fixtures();
    for (pool_name, parallelism, threads) in parallelisms() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            for fixture in &fixtures {
                for wide in [false, true].into_iter().filter(|&wide| !wide || fixture.wide) {
                    let fmt = fmt_for(fixture.components, wide);
                    let expected = decode_into(&fixture.bytes, parallelism, fmt);
                    let row_bytes = WIDTH as usize * fmt.bytes_per_pixel();
                    for (stripe_rows, refuse) in [(1, 0), (7, 50), (64, 100), (HEIGHT, HEIGHT - 1)]
                    {
                        let mut decoder = J2kDecoder::new(&fixture.bytes).expect("decoder");
                        decoder.set_cpu_decode_parallelism(parallelism);
                        let mut sink = Rows {
                            refuse: Some(refuse),
                            ..Rows::default()
                        };
                        let result = decode_rows_with(&mut decoder, wide, stripe_rows, &mut sink);
                        let context =
                            format!("{} {fmt:?} {stripe_rows} rows, {pool_name}", fixture.name);
                        assert!(
                            matches!(result, Err(DecodeRowsError::Sink(Refused(row))) if row == refuse),
                            "{context}: {result:?}"
                        );
                        // The rows before the refused one, then the refused
                        // one, and nothing after it.
                        assert_eq!(
                            sink.order,
                            (0..=refuse).collect::<Vec<_>>(),
                            "{context}: rows handed over"
                        );
                        assert!(
                            sink.bytes == expected[..refuse as usize * row_bytes],
                            "{context}: rows before the refused one"
                        );
                        // The decoder decodes every row again afterwards.
                        if stripe_rows == 7 {
                            let mut again = Rows::default();
                            decode_rows_with(&mut decoder, wide, stripe_rows, &mut again)
                                .unwrap_or_else(|error| panic!("{context}: again: {error}"));
                            assert!(again.bytes == expected, "{context}: rows after a refusal");
                        }
                    }
                }
            }
        });
    }
}

#[test]
fn a_codestream_that_fails_before_its_rows_hands_over_none_and_reports_the_decode_error() {
    let mut bytes = codestream(
        3,
        8,
        &EncodeOptions {
            num_decomposition_levels: 4,
            reversible: false,
            ..EncodeOptions::default()
        },
    );
    // Without its end-of-codestream marker the tile data is rejected.
    assert_eq!(bytes.pop(), Some(0xd9));
    assert_eq!(bytes.pop(), Some(0xff));
    for (pool_name, parallelism, threads) in parallelisms() {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        pool.install(|| {
            for wide in [false, true] {
                let mut sink = Rows::default();
                let result = decode_rows(&bytes, parallelism, wide, 7, &mut sink);
                assert!(
                    matches!(result, Err(DecodeRowsError::Decode(_))),
                    "{pool_name} wide={wide}: {result:?}"
                );
                assert!(sink.order.is_empty(), "{pool_name} wide={wide}: no rows");
            }
        });
    }
}
