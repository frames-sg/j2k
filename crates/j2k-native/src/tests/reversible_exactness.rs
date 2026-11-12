// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reversible coding returns every sample exactly, at every precision.
//!
//! The reversible colour transform and 5/3 wavelet run in `f32` only while
//! every value they compute is a whole number `f32` holds, at most 2^24 in
//! magnitude. The largest such values are the sums the lifting steps take of
//! two neighbours: through any number of decomposition levels they stay
//! below 16 times the largest input magnitude, and a chroma sample of
//! `P`-bit input reaches 2^P - 1. So `f32` is exact through 20 bits, and
//! deeper samples must be coded with integers
//! ([`crate::j2c::MAX_F32_EXACT_REVERSIBLE_PRECISION`]). One bit deeper, the
//! predict step's sums pass 2^24 from the third level and lose their low
//! bit.
//!
//! These tests hold that limit in place from both sides, on the inputs that
//! drive the sums highest, and check that every decode route returns the
//! source samples at every precision on either side of it.

use alloc::vec;
use alloc::vec::Vec;

use crate::j2c::fdwt::{
    packed_dwt_scratch_len, try_forward_dwt_packed_f32, try_forward_dwt_packed_i64,
};
use crate::j2c::MAX_F32_EXACT_REVERSIBLE_PRECISION;
use crate::{
    encode, encode_htj2k, ComponentPlane, ComponentRowSink, CpuDecodeParallelism, DecodeSettings,
    DecoderContext, EncodeOptions, Image,
};

const WIDTH: u32 = 96;
const HEIGHT: u32 = 80;
const LEVELS: u8 = 4;

/// The neighbour sum a lifting step takes that a worst-case input drives
/// highest.
#[derive(Clone, Copy, Debug)]
enum Sum {
    /// Of two samples, by the predict step: the first past 2^24, and the one
    /// whose halving then goes wrong.
    Predict,
    /// Of two high-pass values, by the update step: the largest.
    Update,
}

const SUMS: [Sum; 2] = [Sum::Predict, Sum::Update];

/// The signs of the inputs that drive three values of a 1-D 5/3 analysis
/// highest at decomposition level `level`, each taken near the middle of a
/// signal of `len` samples.
struct WorstCaseSigns {
    /// For the high-pass output.
    high: Vec<i64>,
    /// For the sum the predict step takes of two samples.
    predict_sum: Vec<i64>,
    /// For the sum the update step takes of two high-pass values.
    update_sum: Vec<i64>,
}

impl WorstCaseSigns {
    /// Every sample is followed as a linear function of the input, without
    /// the rounding of the lifting steps: a linear function is largest on
    /// the input whose signs match its coefficients.
    fn new(len: usize, level: u8) -> Self {
        let mut signal: Vec<Vec<f64>> = (0..len)
            .map(|index| {
                let mut unit = vec![0.0; len];
                unit[index] = 1.0;
                unit
            })
            .collect();
        let (mut high, mut predict_sum, mut update_sum) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..level {
            let count = signal.len();
            assert!(count >= 4, "the signal is too short for this level");
            // Whole-sample symmetric extension (T.800 F.4.7).
            let mirror = |index: usize, step: isize| -> usize {
                let at = index.cast_signed() + step;
                if at < 0 {
                    at.unsigned_abs()
                } else if at.unsigned_abs() >= count {
                    2 * (count - 1) - at.unsigned_abs()
                } else {
                    at.unsigned_abs()
                }
            };
            let add = |left: &[f64], right: &[f64]| -> Vec<f64> {
                left.iter().zip(right).map(|(a, b)| a + b).collect()
            };
            let mut detail: Vec<Vec<f64>> = vec![Vec::new(); count];
            let mut sample_sums: Vec<Vec<f64>> = vec![Vec::new(); count];
            for index in (1..count).step_by(2) {
                let sum = add(&signal[mirror(index, -1)], &signal[mirror(index, 1)]);
                detail[index] = signal[index]
                    .iter()
                    .zip(&sum)
                    .map(|(sample, sum)| sample - 0.5 * sum)
                    .collect();
                sample_sums[index] = sum;
            }
            let mut low = Vec::new();
            let mut detail_sums = Vec::new();
            for index in (0..count).step_by(2) {
                let sum = add(&detail[mirror(index, -1)], &detail[mirror(index, 1)]);
                low.push(
                    signal[index]
                        .iter()
                        .zip(&sum)
                        .map(|(sample, sum)| sample + 0.25 * sum)
                        .collect(),
                );
                detail_sums.push(sum);
            }
            let middle = (count / 2) | 1;
            high.clone_from(&detail[middle]);
            predict_sum.clone_from(&sample_sums[middle]);
            update_sum.clone_from(&detail_sums[detail_sums.len() / 2]);
            signal = low;
        }
        let signs = |function: &[f64]| -> Vec<i64> {
            function
                .iter()
                .map(|&coefficient| if coefficient < 0.0 { -1 } else { 1 })
                .collect()
        };
        Self {
            high: signs(&high),
            predict_sum: signs(&predict_sum),
            update_sum: signs(&update_sum),
        }
    }

    fn sum(self, sum: Sum) -> Vec<i64> {
        match sum {
            Sum::Predict => self.predict_sum,
            Sum::Update => self.update_sum,
        }
    }
}

/// A `width` x `height` grid of signs under which a 2-D 5/3 analysis takes
/// its largest `sum` at `level`: the step of one direction over the
/// high-pass output of the other. `transposed` swaps the two directions.
fn worst_case_sign_grid(
    width: u32,
    height: u32,
    level: u8,
    sum: Sum,
    transposed: bool,
) -> Vec<i64> {
    let columns = WorstCaseSigns::new(width as usize, level);
    let rows = WorstCaseSigns::new(height as usize, level);
    let (columns, rows) = if transposed {
        (columns.sum(sum), rows.high)
    } else {
        (columns.high, rows.sum(sum))
    };
    rows.iter()
        .flat_map(|row| columns.iter().map(move |column| row * column))
        .collect()
}

/// A small step back from the extreme at each sample, so that neighbour sums
/// are odd as often as even: an `f32` past 2^24 drops exactly the odd ones.
fn jitter(index: usize) -> i64 {
    let mixed = (index as u64)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .rotate_left(29);
    (mixed % 4).cast_signed()
}

/// One transform input plane at magnitude `amplitude` under the worst-case
/// signs of `sum` at `level`.
fn worst_case_plane(amplitude: i64, level: u8, sum: Sum, transposed: bool) -> Vec<i64> {
    worst_case_sign_grid(WIDTH, HEIGHT, level, sum, transposed)
        .iter()
        .enumerate()
        .map(|(index, sign)| sign * (amplitude - jitter(index)))
        .collect()
}

/// Whether the `f32` and the integer 5/3 analysis of `plane` write the same
/// coefficients.
#[expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "the comparison is of the float transform against whole numbers it may not hold"
)]
fn float_analysis_matches_integers(plane: &[i64]) -> bool {
    let scratch = packed_dwt_scratch_len(WIDTH, HEIGHT).expect("scratch length");
    let mut integers = plane.to_vec();
    let mut floats: Vec<f32> = plane.iter().map(|&sample| sample as f32).collect();
    let integer_shape = try_forward_dwt_packed_i64(
        &mut integers,
        WIDTH,
        HEIGHT,
        LEVELS,
        &mut vec![0; scratch],
        (0, 0),
    )
    .expect("integer analysis");
    let float_shape = try_forward_dwt_packed_f32(
        &mut floats,
        WIDTH,
        HEIGHT,
        LEVELS,
        true,
        &mut vec![0.0; scratch],
        None,
        (0, 0),
    )
    .expect("float analysis");
    assert_eq!(integer_shape, float_shape);
    integers
        .iter()
        .zip(&floats)
        .all(|(&integer, &float)| float.fract() == 0.0 && float as i64 == integer)
}

fn every_worst_case_plane(amplitude: i64) -> impl Iterator<Item = Vec<i64>> {
    (1..=LEVELS).flat_map(move |level| {
        SUMS.into_iter().flat_map(move |sum| {
            [false, true]
                .into_iter()
                .map(move |transposed| worst_case_plane(amplitude, level, sum, transposed))
        })
    })
}

#[test]
fn float_wavelet_matches_integers_on_worst_case_input_at_its_precision_limit() {
    // A chroma sample of P-bit input lies within 2^P - 1 of zero.
    let amplitude = (1_i64 << MAX_F32_EXACT_REVERSIBLE_PRECISION) - 1;
    for plane in every_worst_case_plane(amplitude) {
        assert!(float_analysis_matches_integers(&plane));
    }
}

#[test]
fn float_wavelet_is_inexact_one_bit_past_its_precision_limit() {
    let amplitude = (1_i64 << (MAX_F32_EXACT_REVERSIBLE_PRECISION + 1)) - 1;
    assert!(
        every_worst_case_plane(amplitude).any(|plane| !float_analysis_matches_integers(&plane)),
        "raising the limit needs a proof that the float transform stays exact"
    );
}

/// One encoded image and the samples it must decode to.
struct Source {
    depth: u8,
    signed: bool,
    components: u16,
    /// Interleaved samples.
    samples: Vec<i64>,
}

impl Source {
    /// Samples at the extremes of the `depth`-bit range under the worst-case
    /// signs of `sum` at `level`. With three components, green opposes red and blue,
    /// which puts both chroma planes of the reversible colour transform at
    /// their extremes.
    fn worst_case(depth: u8, signed: bool, components: u16, level: u8, sum: Sum) -> Self {
        let (low, high) = if signed {
            (-(1_i64 << (depth - 1)), (1_i64 << (depth - 1)) - 1)
        } else {
            (0, (1_i64 << depth) - 1)
        };
        let span = (high - low).min(3);
        let mut samples = Vec::new();
        for (index, sign) in worst_case_sign_grid(WIDTH, HEIGHT, level, sum, false)
            .into_iter()
            .enumerate()
        {
            for component in 0..components {
                // A step of its own per component: chroma, the difference
                // of two, is then odd as often as even.
                let step =
                    jitter(index * usize::from(components) + usize::from(component)).min(span);
                let opposed = components == 3 && component == 1;
                samples.push(if (sign > 0) == opposed {
                    low + step
                } else {
                    high - step
                });
            }
        }
        Self {
            depth,
            signed,
            components,
            samples,
        }
    }

    fn bytes_per_sample(&self) -> usize {
        usize::from(self.depth).div_ceil(8)
    }

    fn pixels(&self) -> Vec<u8> {
        let width = self.bytes_per_sample();
        self.samples
            .iter()
            .flat_map(|sample| sample.to_le_bytes().into_iter().take(width))
            .collect()
    }

    fn encode(&self, high_throughput: bool, tile_size: Option<(u32, u32)>) -> Vec<u8> {
        let options = EncodeOptions {
            num_decomposition_levels: LEVELS,
            reversible: true,
            use_mct: self.components == 3,
            tile_size,
            ..EncodeOptions::default()
        };
        let pixels = self.pixels();
        let (components, depth, signed) = (self.components, self.depth, self.signed);
        if high_throughput {
            encode_htj2k(&pixels, WIDTH, HEIGHT, components, depth, signed, &options)
        } else {
            encode(&pixels, WIDTH, HEIGHT, components, depth, signed, &options)
        }
        .unwrap_or_else(|error| panic!("{} encode: {error}", self.label(high_throughput)))
    }

    fn label(&self, high_throughput: bool) -> alloc::string::String {
        alloc::format!(
            "{}-bit {} x{} {}",
            self.depth,
            if self.signed { "signed" } else { "unsigned" },
            self.components,
            if high_throughput { "HT" } else { "classic" },
        )
    }

    /// The samples packed bytes hold, read as the decoder writes them.
    fn unpack(&self, bytes: &[u8]) -> Vec<i64> {
        let width = self.bytes_per_sample();
        let shift = 64 - u32::from(self.depth);
        bytes
            .chunks_exact(width)
            .map(|chunk| {
                let mut wide = [0_u8; 8];
                wide[..width].copy_from_slice(chunk);
                let raw = u64::from_le_bytes(wide) << shift;
                if self.signed {
                    raw.cast_signed() >> shift
                } else {
                    (raw >> shift).cast_signed()
                }
            })
            .collect()
    }

    /// The source samples of `roi`, interleaved.
    fn region(&self, (x, y, width, height): (u32, u32, u32, u32)) -> Vec<i64> {
        let components = usize::from(self.components);
        (y..y + height)
            .flat_map(|row| {
                let start = (row * WIDTH + x) as usize * components;
                self.samples[start..start + width as usize * components]
                    .iter()
                    .copied()
            })
            .collect()
    }
}

/// Collects the rectangles a row sink receives into interleaved samples.
struct RegionSink {
    width: usize,
    components: usize,
    samples: Vec<i64>,
    arrivals: Vec<u8>,
}

impl ComponentRowSink for RegionSink {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "planes of at most 24 bits hold whole numbers"
    )]
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        assert_eq!(planes.len(), self.components);
        let (width, height) = planes[0].dimensions();
        for (component, plane) in planes.iter().enumerate() {
            for row in 0..height as usize {
                for column in 0..width as usize {
                    let at = (y as usize + row) * self.width + x as usize + column;
                    let sample = plane.samples()[row * width as usize + column];
                    assert_eq!(sample.fract().to_bits() << 1, 0, "a whole number");
                    self.samples[at * self.components + component] = sample as i64;
                    if component == 0 {
                        self.arrivals[at] += 1;
                    }
                }
            }
        }
    }
}

const REGIONS: [(u32, u32, u32, u32); 3] =
    [(0, 0, WIDTH, HEIGHT), (17, 9, 40, 33), (64, 48, 32, 32)];

/// Every decode route of `codestream` returns the samples of `source`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "planes of at most 24 bits hold whole numbers"
)]
fn assert_every_route_decodes_exactly(source: &Source, codestream: &[u8], label: &str) {
    let image = Image::new(codestream, &DecodeSettings::default()).expect("parse");
    let components = usize::from(source.components);

    let native = image.decode_native().expect("native decode");
    assert_eq!(native.bit_depth, source.depth, "{label}");
    assert!(
        source.unpack(&native.data) == source.samples,
        "{label}: native decode"
    );

    for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
        for roi in REGIONS {
            let expected = source.region(roi);

            let mut context = DecoderContext::default();
            context.set_cpu_decode_parallelism(parallelism);
            let region = image
                .decode_native_region_with_context(roi, &mut context)
                .expect("native region decode");
            assert!(
                source.unpack(&region.data) == expected,
                "{label}: native region {roi:?} {parallelism:?}"
            );

            let mut context = DecoderContext::default();
            context.set_cpu_decode_parallelism(parallelism);
            let decoded = image
                .decode_region_components_for_integer_output_with_context(roi, &mut context)
                .expect("component planes");
            let planes = decoded.planes();
            assert_eq!(planes.len(), components);
            let samples = (roi.2 * roi.3) as usize;
            let interleaved: Vec<i64> = (0..samples)
                .flat_map(|at| planes.iter().map(move |plane| plane.samples()[at] as i64))
                .collect();
            assert!(
                interleaved == expected,
                "{label}: component planes {roi:?} {parallelism:?}"
            );
            drop(decoded);

            let mut context = DecoderContext::default();
            context.set_cpu_decode_parallelism(parallelism);
            let mut sink = RegionSink {
                width: roi.2 as usize,
                components,
                samples: vec![0; samples * components],
                arrivals: vec![0; samples],
            };
            image
                .decode_region_rows_for_integer_output_with_context(roi, &mut context, &mut sink)
                .expect("row sink");
            assert!(sink.arrivals.iter().all(|&count| count == 1), "{label}");
            assert!(
                sink.samples == expected,
                "{label}: row sink {roi:?} {parallelism:?}"
            );
        }
    }
}

/// Precisions on both sides of the float limit, through the deepest the raw
/// pixel encoder takes.
const PRECISIONS: [u8; 11] = [8, 12, 16, 17, 18, 19, 20, 21, 22, 23, 24];

#[test]
fn lossless_round_trips_worst_case_samples_at_every_precision() {
    for depth in PRECISIONS {
        for signed in [false, true] {
            for components in [1, 3] {
                for level in 1..=LEVELS {
                    for sum in SUMS {
                        let source = Source::worst_case(depth, signed, components, level, sum);
                        for high_throughput in [false, true] {
                            let label = source.label(high_throughput);
                            let codestream = source.encode(high_throughput, None);
                            let image =
                                Image::new(&codestream, &DecodeSettings::default()).expect("parse");
                            let decoded = image.decode_native().expect("native decode");
                            assert!(
                                source.unpack(&decoded.data) == source.samples,
                                "{label}, level {level}, {sum:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn every_decode_route_is_exact_on_both_sides_of_the_float_limit() {
    for depth in [16, 20, 21, 23, 24] {
        for components in [1, 3] {
            let source = Source::worst_case(depth, false, components, 3, Sum::Predict);
            for high_throughput in [false, true] {
                let label = source.label(high_throughput);
                assert_every_route_decodes_exactly(
                    &source,
                    &source.encode(high_throughput, None),
                    &label,
                );
            }
            let label = alloc::format!("{} tiled", source.label(false));
            assert_every_route_decodes_exactly(
                &source,
                &source.encode(false, Some((64, 48))),
                &label,
            );
        }
    }
}
