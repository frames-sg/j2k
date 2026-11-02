// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::codestream::WaveletTransform;
use super::super::rect::IntRect;
use super::direct::apply_single_decomposition_idwt_job;
use super::horizontal::{filter_horizontal, filter_horizontal_i64};
use super::model::{idwt_buffer_size, CoefficientSource};
use super::roi::interleave_samples_roi;
use super::vertical::{filter_vertical, filter_vertical_i64};
use crate::error::{DecodeError, DecodingError};
use crate::{J2kIdwtBand, J2kRect, J2kSingleDecompositionIdwtJob, J2kWaveletTransform, Result};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn assert_f32_bits_within_ulps(actual: &[f32], expected_bits: &[u32], max_ulps: u32) {
    const SIGN_MASK: u32 = 0x8000_0000;

    assert_eq!(actual.len(), expected_bits.len());
    for (index, (&actual, &expected_bits)) in actual.iter().zip(expected_bits).enumerate() {
        let ordered = |bits: u32| {
            if bits & SIGN_MASK == 0 {
                bits | SIGN_MASK
            } else {
                !bits
            }
        };
        let distance = ordered(actual.to_bits()).abs_diff(ordered(expected_bits));
        assert!(
            distance <= max_ulps,
            "sample {index} differs by {distance} ULPs: actual={actual:?}, expected={:?}",
            f32::from_bits(expected_bits)
        );
    }
}

#[test]
fn idwt_workspace_includes_shift_padding_for_degenerate_rectangles() {
    let rect = IntRect::from_xywh(0, 0, 1, 7);

    assert_eq!(idwt_buffer_size(rect).expect("bounded geometry"), (7, 16));
}

fn direct_job<'a>(
    rect: J2kRect,
    transform: J2kWaveletTransform,
    ll: J2kIdwtBand<'a>,
    hl: J2kIdwtBand<'a>,
    lh: J2kIdwtBand<'a>,
    hh: J2kIdwtBand<'a>,
) -> J2kSingleDecompositionIdwtJob<'a> {
    J2kSingleDecompositionIdwtJob {
        rect,
        transform,
        ll,
        hl,
        lh,
        hh,
    }
}

fn band(rect: J2kRect, coefficients: &[f32]) -> J2kIdwtBand<'_> {
    J2kIdwtBand { rect, coefficients }
}

#[test]
fn reversible_53_even_direct_job_is_bit_exact() {
    const EXPECTED: [u32; 16] = [
        3_225_419_776,
        3_231_711_232,
        3_225_419_776,
        3_229_614_080,
        3_212_836_864,
        1_093_664_768,
        0,
        1_096_810_496,
        3_221_225_472,
        3_221_225_472,
        3_221_225_472,
        3_212_836_864,
        1_065_353_216,
        1_098_907_648,
        1_073_741_824,
        1_100_480_512,
    ];
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 4,
        y1: 4,
    };
    let band_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let ll = [1.0, 2.0, 3.0, 4.0];
    let hl = [5.0, 6.0, 7.0, 8.0];
    let lh = [9.0, 10.0, 11.0, 12.0];
    let hh = [13.0, 14.0, 15.0, 16.0];
    let mut output = Vec::with_capacity(32);
    let capacity = output.capacity();

    apply_single_decomposition_idwt_job(
        direct_job(
            rect,
            J2kWaveletTransform::Reversible53,
            band(band_rect, &ll),
            band(band_rect, &hl),
            band(band_rect, &lh),
            band(band_rect, &hh),
        ),
        &mut output,
    )
    .expect("valid reversible direct IDWT job");

    assert_eq!(bits(&output).as_slice(), &EXPECTED);
    assert_eq!(
        output.capacity(),
        capacity,
        "target capacity must be reused"
    );
}

#[test]
#[expect(
    clippy::similar_names,
    reason = "LL, HL, LH, and HH test-band names follow JPEG 2000 specification notation"
)]
fn irreversible_97_odd_direct_job_matches_scalar_reference_within_eight_ulps() {
    const EXPECTED: [u32; 15] = [
        3_243_516_307,
        1_088_535_889,
        1_086_781_832,
        3_237_068_116,
        1_072_899_987,
        1_090_811_480,
        3_205_449_568,
        3_240_528_041,
        1_057_965_920,
        1_095_801_596,
        3_240_069_383,
        1_090_703_406,
        1_072_530_022,
        3_238_711_342,
        1_091_758_982,
    ];
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 5,
        y1: 3,
    };
    let ll_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 3,
        y1: 2,
    };
    let hl_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let lh_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 3,
        y1: 1,
    };
    let hh_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 1,
    };
    let ll = [0.5, -1.25, 2.0, 3.5, -4.25, 5.75];
    let hl = [6.5, -7.0, 8.25, -9.5];
    let lh = [10.0, -11.5, 12.75];
    let hh = [-13.0, 14.25];
    let mut output = Vec::new();

    apply_single_decomposition_idwt_job(
        direct_job(
            rect,
            J2kWaveletTransform::Irreversible97,
            band(ll_rect, &ll),
            band(hl_rect, &hl),
            band(lh_rect, &lh),
            band(hh_rect, &hh),
        ),
        &mut output,
    )
    .expect("valid irreversible direct IDWT job");

    assert_f32_bits_within_ulps(&output, &EXPECTED, 8);
}

#[test]
fn reversible_53_two_levels_are_bit_exact() {
    const EXPECTED: [u32; 16] = [
        3_242_196_992,
        1_073_741_824,
        3_233_808_384,
        3_247_964_160,
        1_088_421_888,
        1_073_741_824,
        1_086_324_736,
        1_092_616_192,
        3_225_419_776,
        1_082_130_432,
        3_221_225_472,
        3_239_051_264,
        3_249_537_024,
        1_097_859_072,
        1_092_616_192,
        3_243_245_568,
    ];
    let unit_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    let two_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let four_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 4,
        y1: 4,
    };
    let mut level_one = Vec::new();
    apply_single_decomposition_idwt_job(
        direct_job(
            two_rect,
            J2kWaveletTransform::Reversible53,
            band(unit_rect, &[1.0]),
            band(unit_rect, &[2.0]),
            band(unit_rect, &[3.0]),
            band(unit_rect, &[4.0]),
        ),
        &mut level_one,
    )
    .expect("valid first IDWT level");

    let high_a = [5.0, -6.0, 7.0, -8.0];
    let high_b = [9.0, 10.0, -11.0, 12.0];
    let high_c = [-13.0, 14.0, 15.0, -16.0];
    let mut output = Vec::new();
    apply_single_decomposition_idwt_job(
        direct_job(
            four_rect,
            J2kWaveletTransform::Reversible53,
            band(two_rect, &level_one),
            band(two_rect, &high_a),
            band(two_rect, &high_b),
            band(two_rect, &high_c),
        ),
        &mut output,
    )
    .expect("valid second IDWT level");

    assert_eq!(bits(&output).as_slice(), &EXPECTED);
}

#[test]
fn reversible_i64_filters_preserve_even_and_odd_goldens() {
    const EVEN_EXPECTED: [i64; 16] = [-1, -2, -1, -1, 2, 10, 3, 14, 2, 7, 2, 8, 8, 27, 9, 31];
    const ODD_EXPECTED: [i64; 15] = [
        2_473_901_162_495,
        824_633_720_829,
        2_473_901_162_495,
        824_633_720_831,
        2_473_901_162_496,
        824_633_720_835,
        274_877_906_958,
        824_633_720_837,
        274_877_906_963,
        824_633_720_839,
        2_473_901_162_500,
        824_633_720_844,
        2_473_901_162_500,
        824_633_720_846,
        2_473_901_162_501,
    ];
    let mut even: Vec<i64> = (1..=16).collect();
    let even_rect = IntRect::from_xywh(0, 0, 4, 4);
    filter_horizontal_i64(&mut even, even_rect);
    filter_vertical_i64(&mut even, even_rect);
    assert_eq!(even.as_slice(), &EVEN_EXPECTED);

    let base = 1_i64 << 40;
    let mut odd = vec![
        base + 1,
        -base + 2,
        base + 3,
        -base + 4,
        base + 5,
        -base + 6,
        base + 7,
        -base + 8,
        base + 9,
        -base + 10,
        base + 11,
        -base + 12,
        base + 13,
        -base + 14,
        base + 15,
    ];
    let odd_rect = IntRect::from_xywh(0, 0, 5, 3);
    filter_horizontal_i64(&mut odd, odd_rect);
    filter_vertical_i64(&mut odd, odd_rect);
    assert_eq!(odd.as_slice(), &ODD_EXPECTED);
}

#[test]
fn irreversible_97_nonzero_origin_roi_matches_scalar_reference_within_eight_ulps() {
    const EXPECTED: [u32; 12] = [
        1_098_644_014,
        1_084_670_382,
        3_239_209_924,
        1_072_899_987,
        3_241_441_280,
        3_239_628_158,
        1_073_078_272,
        1_095_801_596,
        1_094_877_886,
        1_066_907_502,
        3_239_826_100,
        1_091_758_982,
    ];
    let ll = [0.5, -1.25, 2.0, 3.5, -4.25, 5.75];
    let hl = [6.5, -7.0, 8.25, -9.5];
    let lh = [10.0, -11.5, 12.75];
    let hh = [-13.0, 14.25];
    let output_window = IntRect::from_ltrb(1, 0, 5, 3);
    let mut output = vec![0.0; 12];

    interleave_samples_roi(
        CoefficientSource::new(&ll, IntRect::from_ltrb(0, 0, 3, 2), 3),
        CoefficientSource::new(&hl, IntRect::from_ltrb(0, 0, 2, 2), 2),
        CoefficientSource::new(&lh, IntRect::from_ltrb(0, 0, 3, 1), 3),
        CoefficientSource::new(&hh, IntRect::from_ltrb(0, 0, 2, 1), 2),
        &mut output,
        output_window,
    );
    filter_horizontal(&mut output, output_window, WaveletTransform::Irreversible97);
    filter_vertical(&mut output, output_window, WaveletTransform::Irreversible97);

    assert_f32_bits_within_ulps(&output, &EXPECTED, 8);
}

#[test]
fn invalid_direct_band_preserves_target_and_error_class() {
    let rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 2,
        y1: 2,
    };
    let band_rect = J2kRect {
        x0: 0,
        y0: 0,
        x1: 1,
        y1: 1,
    };
    let empty = [];
    let valid = [1.0];
    let mut target = vec![42.0, -7.0];

    let result = apply_single_decomposition_idwt_job(
        direct_job(
            rect,
            J2kWaveletTransform::Reversible53,
            band(band_rect, &empty),
            band(band_rect, &valid),
            band(band_rect, &valid),
            band(band_rect, &valid),
        ),
        &mut target,
    );

    assert!(matches!(
        result,
        Err(DecodeError::Decoding(DecodingError::CodeBlockDecodeFailure))
    ));
    assert_eq!(target, [42.0, -7.0]);
}

#[test]
fn public_idwt_function_signatures_stay_stable() {
    let _: fn(J2kSingleDecompositionIdwtJob<'_>, &mut Vec<f32>) -> Result<()> =
        apply_single_decomposition_idwt_job;
}

/// Deterministic xorshift values: small integers for 5/3, fractions for 9/7.
#[expect(
    clippy::cast_precision_loss,
    reason = "the drawn value is below 20001 and exact in f32"
)]
fn pseudo_random_coefficients(count: usize, seed: u64, integral: bool) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let value = (state % 20_001) as f32 - 10_000.0;
            if integral {
                value
            } else {
                value / 7.0
            }
        })
        .collect()
}

/// Deterministic xorshift integers in `-magnitude..=magnitude`.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the drawn value is below 2 * magnitude + 1, which fits i64"
)]
fn pseudo_random_integers(count: usize, seed: u64, magnitude: i64) -> Vec<i64> {
    let span = magnitude.unsigned_abs() * 2 + 1;
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % span) as i64 - magnitude
        })
        .collect()
}

#[test]
#[expect(
    clippy::cast_precision_loss,
    reason = "every value is below 2^23 in magnitude and exact in f32"
)]
fn reversible_53_lifting_terms_are_exact_floor_divisions() {
    use super::filter_common::{floor_div_i64, reversible_53_predict, reversible_53_update};

    let mut values = vec![
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        -5,
        1 << 20,
        -(1 << 20) - 1,
        (1 << 22) - 1,
        -(1 << 22),
    ];
    values.extend(pseudo_random_integers(96, 9, 1 << 22));
    for s in [0_i64, 7, -9, 1 << 21] {
        for &left in &values {
            for &right in &values {
                let (sf, lf, rf) = (s as f32, left as f32, right as f32);
                assert_eq!(
                    reversible_53_predict(sf, lf, rf),
                    (s - floor_div_i64(left + right + 2, 4)) as f32,
                    "(F-5) s={s} left={left} right={right}"
                );
                assert_eq!(
                    reversible_53_update(sf, lf, rf),
                    (s + floor_div_i64(left + right, 2)) as f32,
                    "(F-6) s={s} left={left} right={right}"
                );
            }
        }
    }
}

/// The `f32` path holds integer-valued 5-3 coefficients. Where `f32` holds
/// them exactly, its horizontal and vertical lifting must equal the exact
/// integer reference sample for sample, at every origin parity.
///
/// A lone sample at an odd origin is not lifted but halved (F.3.7). The
/// `f32` path keeps that half as a fraction and the integer reference
/// floors it, so those one-sample dimensions are left out.
#[test]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture sizes fit u32; coefficients stay below 2^23 in magnitude, exact in f32"
)]
fn reversible_53_f32_filters_match_the_exact_integer_reference() {
    let shapes = [
        (1, 1),
        (1, 5),
        (2, 3),
        (5, 1),
        (7, 6),
        (8, 9),
        (13, 4),
        (16, 16),
        (33, 7),
    ];
    for (width, height) in shapes {
        for (x0, y0) in [(0, 0), (1, 0), (0, 1), (3, 5)] {
            if (width == 1 && x0 % 2 == 1) || (height == 1 && y0 % 2 == 1) {
                continue;
            }
            for (seed, magnitude) in [(1, 255), (2, 1 << 16), (3, 1 << 20)] {
                let rect = IntRect::from_xywh(x0, y0, width, height);
                let count = (width * height) as usize;
                let mut exact = pseudo_random_integers(count, seed, magnitude);
                let mut samples: Vec<f32> = exact.iter().map(|&value| value as f32).collect();
                filter_horizontal(&mut samples, rect, WaveletTransform::Reversible53);
                filter_vertical(&mut samples, rect, WaveletTransform::Reversible53);
                filter_horizontal_i64(&mut exact, rect);
                filter_vertical_i64(&mut exact, rect);
                let lifted: Vec<i64> = samples.iter().map(|&value| value as i64).collect();
                assert_eq!(
                    lifted, exact,
                    "{width}x{height} at ({x0}, {y0}), magnitude {magnitude}"
                );
            }
        }
    }
}

#[test]
#[expect(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    reason = "one table-driven equivalence sweep; fixture band sizes fit u32"
)]
fn banded_2d_matches_plain_2d_bit_for_bit() {
    use super::super::build::{Decomposition, SubBand, SubBandType};
    use super::super::decode::DecompositionStorage;
    #[cfg(feature = "parallel")]
    use super::banded::banded_scratch_elements_for_execution;
    use super::banded::{filter_2d_banded, use_banded};
    use super::direct::filter_2d;
    use super::model::IDWTInput;

    #[cfg(feature = "parallel")]
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test pool");
    let mut seed = 1_u64;
    for transform in [
        WaveletTransform::Reversible53,
        WaveletTransform::Irreversible97,
    ] {
        for (x0, y0) in [(0, 0), (1, 0), (0, 1), (1, 1), (3, 6)] {
            for (width, height) in [
                (64, 65),
                (65, 71),
                (100, 300),
                (257, 263),
                (600, 200),
                (1000, 33),
                (8, 600),
                (513, 17),
                (16, 300),
                (17, 300),
                (18, 260),
                // Around the first row width whose 9-7I horizontal synthesis
                // starts with a register-window pass, and its tail.
                (47, 100),
                (48, 100),
                (49, 100),
                (81, 60),
                // High-first width whose longer high band ends on a SIMD
                // boundary in the fused final horizontal lifting step.
                (31, 300),
                // Tall enough for several parallel strips with halos on both
                // sides, at both row parities.
                (300, 1000),
                (1031, 133),
            ] {
                assert!(use_banded(width as usize, height as usize));
                let rect = IntRect::from_xywh(x0, y0, width, height);
                let (u0, u1, v0, v1) = (rect.x0, rect.x1, rect.y0, rect.y1);
                let low_w = (u1.div_ceil(2) - u0.div_ceil(2)) as usize;
                let high_w = (u1 / 2 - u0 / 2) as usize;
                let low_h = (v1.div_ceil(2) - v0.div_ceil(2)) as usize;
                let high_h = (v1 / 2 - v0 / 2) as usize;
                let integral = transform == WaveletTransform::Reversible53;
                seed += 1;
                let ll = pseudo_random_coefficients(low_w * low_h, seed, integral);
                let mut storage = DecompositionStorage::default();
                let mut start = 0;
                for (index, len) in [high_w * low_h, low_w * high_h, high_w * high_h]
                    .into_iter()
                    .enumerate()
                {
                    seed += 1;
                    storage
                        .coefficients
                        .extend(pseudo_random_coefficients(len, seed, integral));
                    storage.sub_bands.push(SubBand {
                        sub_band_type: [
                            SubBandType::HighLow,
                            SubBandType::LowHigh,
                            SubBandType::HighHigh,
                        ][index],
                        rect,
                        storage_rect: rect,
                        precincts: 0..0,
                        coefficients: start..start + len,
                    });
                    start += len;
                }
                let decomposition = Decomposition {
                    sub_bands: [0, 1, 2],
                    rect,
                };
                let input = IDWTInput {
                    rect: IntRect::from_xywh(0, 0, low_w as u32, low_h as u32),
                    coefficients: &ll,
                };

                let mut plain = Vec::new();
                filter_2d(input, &mut plain, &decomposition, transform, &storage)
                    .expect("plain level");
                let label = format!("{transform:?} origin=({x0},{y0}) size={width}x{height}");

                let mut serial = Vec::new();
                let mut scratch = Vec::new();
                filter_2d_banded(
                    input,
                    &decomposition,
                    &mut serial,
                    &mut scratch,
                    transform,
                    &storage,
                    false,
                )
                .expect("serial banded level");
                assert_eq!(bits(&serial), bits(&plain), "serial {label}");

                // The direct single-level entry point takes the banded path
                // for packed bands; both normalizations must match the
                // windowed interleave and plain filters.
                let packed = super::banded::packed_band_rects(rect);
                let band_data = [
                    ll.as_slice(),
                    &storage.coefficients[storage.sub_bands[0].coefficients.clone()],
                    &storage.coefficients[storage.sub_bands[1].coefficients.clone()],
                    &storage.coefficients[storage.sub_bands[2].coefficients.clone()],
                ];
                let job_band = |index: usize| band(J2kRect::from(packed[index]), band_data[index]);
                let job = direct_job(
                    J2kRect::from(rect),
                    J2kWaveletTransform::from(transform),
                    job_band(0),
                    job_band(1),
                    job_band(2),
                    job_band(3),
                );
                let mut direct_codestream = Vec::new();
                super::direct::apply_codestream_single_decomposition_idwt_job(
                    job,
                    &mut direct_codestream,
                    false,
                )
                .expect("direct codestream level");
                assert_eq!(
                    bits(&direct_codestream),
                    bits(&plain),
                    "direct codestream {label}"
                );

                let sources = band_data
                    .iter()
                    .zip(packed)
                    .map(|(data, packed)| CoefficientSource::new(data, packed, packed.width()))
                    .collect::<Vec<_>>();
                let mut windowed = vec![0.0; (width * height) as usize];
                interleave_samples_roi(
                    sources[0],
                    sources[1],
                    sources[2],
                    sources[3],
                    &mut windowed,
                    rect,
                );
                filter_horizontal(&mut windowed, rect, transform);
                filter_vertical(&mut windowed, rect, transform);
                let mut direct_standard = Vec::new();
                apply_single_decomposition_idwt_job(job, &mut direct_standard)
                    .expect("direct standard level");
                assert_eq!(
                    bits(&direct_standard),
                    bits(&windowed),
                    "direct standard {label}"
                );

                #[cfg(feature = "parallel")]
                {
                    let mut parallel = vec![f32::NAN; 3];
                    let expected_scratch = pool
                        .install(|| {
                            banded_scratch_elements_for_execution(
                                width as usize,
                                height as usize,
                                true,
                            )
                        })
                        .expect("parallel scratch size");
                    pool.install(|| {
                        filter_2d_banded(
                            input,
                            &decomposition,
                            &mut parallel,
                            &mut scratch,
                            transform,
                            &storage,
                            true,
                        )
                    })
                    .expect("parallel banded level");
                    assert_eq!(scratch.len(), expected_scratch, "parallel scratch {label}");
                    assert_eq!(bits(&parallel), bits(&plain), "parallel {label}");
                }
            }
        }
    }
}

/// Sub-bands for the streamed IDWT, each stored in its own rectangle.
struct StoredBands {
    bands: Vec<(Vec<f32>, IntRect)>,
}

impl super::lines::SubBandRows for StoredBands {
    fn prepare_row(&mut self, _sub_band: usize, _y: u32) -> Result<()> {
        Ok(())
    }

    fn samples(&self, sub_band: usize) -> (&[f32], IntRect, u32) {
        let (samples, rect) = &self.bands[sub_band];
        (samples, *rect, rect.width())
    }
}

/// `rect` grown by `[left, top, right, bottom]` samples; negative values
/// shrink it.
#[expect(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "the fixture rectangles stay at small non-negative coordinates"
)]
fn adjusted(rect: IntRect, [left, top, right, bottom]: [i64; 4]) -> IntRect {
    let at = |value: u32, delta: i64| (i64::from(value) + delta).max(0) as u32;
    IntRect::from_ltrb(
        at(rect.x0, -left),
        at(rect.y0, -top),
        at(rect.x1, right).max(at(rect.x0, -left)),
        at(rect.y1, bottom).max(at(rect.y0, -top)),
    )
}

/// The rows `copy` of the streamed IDWT of `windows`, each taken through
/// `LineReconstruction::next_row_mut` and overwritten with NaN as soon as it
/// is read, so a row a later step still read would poison the rows after it.
fn streamed_rows_overwritten(
    rows: &mut StoredBands,
    sub_bands: &[[usize; 3]],
    windows: &[IntRect],
    transform: WaveletTransform,
    copy: IntRect,
) -> Vec<(u32, Vec<u32>)> {
    let mut ring = vec![
        f32::NAN;
        super::lines::ring_samples(windows.iter().copied(), transform)
            .expect("ring")
    ];
    let mut lines =
        super::lines::LineReconstruction::new(sub_bands, windows, 0, transform, &mut ring, copy)
            .expect("plan the streamed rows");
    let mut out = Vec::new();
    while let Some((y, row)) = lines.next_row_mut(rows).expect("streamed row") {
        out.push((y, bits(row)));
        row.fill(f32::NAN);
    }
    out
}

/// Band rows a reconstruction of one component asks for, logged in the
/// order every component asks.
struct LoggedBands<'a> {
    bands: &'a StoredBands,
    component: usize,
    log: &'a core::cell::RefCell<Vec<(usize, usize, u32)>>,
}

impl super::lines::SubBandRows for LoggedBands<'_> {
    fn prepare_row(&mut self, sub_band: usize, y: u32) -> Result<()> {
        self.log.borrow_mut().push((self.component, sub_band, y));
        Ok(())
    }

    fn samples(&self, sub_band: usize) -> (&[f32], IntRect, u32) {
        let (samples, rect) = &self.bands.bands[sub_band];
        (samples, *rect, rect.width())
    }
}

/// A streamed tile finishes each row of its components in place, which
/// needs one arrival more than a final row; taking every component's row
/// final first must keep the band rows they ask for, and so the bands they
/// decode, in the order that handing the rows over final asks for them:
/// over three components, both transforms, two levels, up to four skipped
/// first rows, and the window's bottom edge or rows short of it.
#[test]
fn rows_finished_in_place_ask_for_bands_as_final_rows_do() {
    use super::banded::packed_band_rects;
    use super::lines::{ring_samples, LineReconstruction};

    for transform in [
        WaveletTransform::Reversible53,
        WaveletTransform::Irreversible97,
    ] {
        for (x0, y0, width, height) in [(0, 0, 9, 7), (1, 1, 20, 13), (3, 2, 33, 21)] {
            let first = IntRect::from_xywh(x0, y0, width, height);
            let next = IntRect::from_xywh(2 * x0 + 1, 2 * y0, 2 * width - 1, 2 * height);
            let mut seed = 11_u64;
            let bands = StoredBands {
                bands: packed_band_rects(first)
                    .into_iter()
                    .chain(packed_band_rects(next).into_iter().skip(1))
                    .map(|rect| {
                        seed += 1;
                        let len = (rect.width() * rect.height()) as usize;
                        (pseudo_random_coefficients(len, seed, false), rect)
                    })
                    .collect(),
            };
            let windows = [first, next];
            for (skipped, cut) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (0, 5), (1, 6)] {
                let copy = IntRect::from_ltrb(next.x0, next.y0 + skipped, next.x1, next.y1 - cut);
                let order = |in_place: bool| {
                    let log = core::cell::RefCell::new(Vec::new());
                    let ring = ring_samples(windows, transform).expect("ring");
                    let mut rings = vec![vec![f32::NAN; ring]; 3];
                    let mut lines: Vec<_> = rings
                        .iter_mut()
                        .map(|ring| {
                            LineReconstruction::new(
                                &[[1, 2, 3], [4, 5, 6]],
                                &windows,
                                0,
                                transform,
                                ring,
                                copy,
                            )
                            .expect("plan the streamed rows")
                        })
                        .collect();
                    let mut sources: Vec<_> = (0..3)
                        .map(|component| LoggedBands {
                            bands: &bands,
                            component,
                            log: &log,
                        })
                        .collect();
                    for y in copy.y0..copy.y1 {
                        if in_place {
                            for (line, source) in lines.iter_mut().zip(sources.iter_mut()) {
                                line.finish_next_row(source).expect("final row");
                            }
                            for (line, source) in lines.iter_mut().zip(sources.iter_mut()) {
                                let (row_y, _) =
                                    line.next_row_mut(source).expect("row").expect("a row");
                                assert_eq!(row_y, y);
                            }
                        } else {
                            for (line, source) in lines.iter_mut().zip(sources.iter_mut()) {
                                let (row_y, _) =
                                    line.next_row(source).expect("row").expect("a row");
                                assert_eq!(row_y, y);
                            }
                        }
                    }
                    log.into_inner()
                };
                let label = format!("{transform:?} {first:?} skipped {skipped} cut {cut}");
                let final_order = order(false);
                assert!(!final_order.is_empty(), "{label}");
                assert_eq!(order(true), final_order, "{label}");
            }
        }
    }
}

/// The whole-window region IDWT of one level, as `roi::fill_level_roi` runs it.
fn whole_window_level(
    sources: [CoefficientSource<'_>; 4],
    window: IntRect,
    transform: WaveletTransform,
) -> Vec<f32> {
    let mut output = vec![0.0; (window.width() * window.height()) as usize];
    interleave_samples_roi(
        sources[0],
        sources[1],
        sources[2],
        sources[3],
        &mut output,
        window,
    );
    super::horizontal::filter_horizontal_codestream(&mut output, window, transform);
    super::vertical::filter_vertical_codestream(&mut output, window, transform);
    output
}

/// The streamed IDWT synthesizes a row from its two band rows in place with
/// the banded SIMD kernel when their stored rectangles hold them whole, and
/// otherwise through the windowed interleave. Every row must match the
/// whole-window region IDWT, which reads every sample through
/// `CoefficientSource::get` and filters with the scalar `filter_row`, bit for
/// bit: both transforms, both column and row parities, every height up to a
/// few wraps of the level's ring, stored rectangles wider than the window's
/// bands and clipped on either side, sub-bands with nothing stored, and a
/// second level reading the first one's rows.
#[test]
#[expect(clippy::too_many_lines, reason = "one table-driven equivalence sweep")]
fn streamed_rows_match_the_whole_window_levels_bit_for_bit() {
    use super::banded::packed_band_rects;
    use super::lines::{reconstruct_rows, ring_samples};

    let mut seed = 7_u64;
    for transform in [
        WaveletTransform::Reversible53,
        WaveletTransform::Irreversible97,
    ] {
        let integral = transform == WaveletTransform::Reversible53;
        for (x0, y0) in [(0, 0), (1, 0), (0, 1), (1, 1), (5, 2), (8, 3)] {
            for (width, height) in [
                (16, 9),
                (17, 20),
                (31, 2),
                (40, 33),
                (64, 12),
                (161, 40),
                // Every height up to a few ring wraps, around the SIMD width.
                (15, 1),
                (16, 3),
                (17, 4),
                (24, 5),
                (32, 6),
                (33, 7),
                (8, 8),
                (47, 10),
                (49, 11),
            ] {
                for clip in [
                    [0, 0, 0, 0],
                    [2, 1, 3, 2],
                    [-2, 0, 0, 0],
                    [0, 0, -3, 0],
                    [-1, -1, -2, -1],
                    [3, 0, -1, 1],
                    [1, 0, 1, 0],
                ] {
                    let label = format!(
                        "{transform:?} origin=({x0},{y0}) size={width}x{height} clip={clip:?}"
                    );
                    let window = IntRect::from_xywh(x0, y0, width, height);
                    // A sub-band without code-blocks stores nothing and
                    // reads as zero: here HH, or LH with a clipped HL.
                    let empty = match clip {
                        [1, 0, 1, 0] => Some(3),
                        [-2, 0, 0, 0] => Some(2),
                        _ => None,
                    };
                    let mut bands = Vec::new();
                    for (index, rect) in packed_band_rects(window).into_iter().enumerate() {
                        let stored = if empty == Some(index) {
                            IntRect::from_xywh(0, 0, 0, 0)
                        } else {
                            adjusted(rect, clip)
                        };
                        seed += 1;
                        let samples = pseudo_random_coefficients(
                            (stored.width() * stored.height()) as usize,
                            seed,
                            integral,
                        );
                        bands.push((samples, stored));
                    }
                    let sources: [CoefficientSource<'_>; 4] = core::array::from_fn(|index| {
                        let (samples, rect) = &bands[index];
                        CoefficientSource::new(samples, *rect, rect.width())
                    });
                    let expected = whole_window_level(sources, window, transform);

                    let mut rows = StoredBands {
                        bands: bands.clone(),
                    };
                    let mut ring = vec![f32::NAN; ring_samples([window], transform).expect("ring")];
                    let mut streamed = vec![f32::NAN; expected.len()];
                    let mut output_samples = 0;
                    reconstruct_rows(
                        &mut rows,
                        &[[1, 2, 3]],
                        &[window],
                        0,
                        transform,
                        &mut ring,
                        window,
                        &mut |y, samples| {
                            let start = ((y - window.y0) * width) as usize;
                            streamed[start..][..samples.len()].copy_from_slice(samples);
                            Ok(())
                        },
                        &mut output_samples,
                    )
                    .expect("streamed level");
                    assert_eq!(bits(&streamed), bits(&expected), "one level {label}");
                    let overwritten = streamed_rows_overwritten(
                        &mut StoredBands {
                            bands: bands.clone(),
                        },
                        &[[1, 2, 3]],
                        &[window],
                        transform,
                        window,
                    );
                    assert_eq!(overwritten.len(), height as usize, "rows {label}");
                    for (y, row) in overwritten {
                        let start = ((y - window.y0) * width) as usize;
                        assert_eq!(
                            row,
                            bits(&expected[start..][..width as usize]),
                            "one level, rows overwritten as read, {label} row {y}"
                        );
                    }

                    // A second level whose LL is the first level's window,
                    // clipped or grown the same way.
                    let next = IntRect::from_xywh(2 * x0 + 1, 2 * y0, 2 * width - 3, 2 * height);
                    let next_bands = packed_band_rects(next);
                    let first = adjusted(next_bands[0], clip);
                    if first.width() < 2 || first.height() < 2 {
                        continue;
                    }
                    let stored = packed_band_rects(first)
                        .into_iter()
                        .chain(next_bands[1..].iter().map(|rect| adjusted(*rect, clip)));
                    let levels: Vec<(Vec<f32>, IntRect)> = stored
                        .map(|rect| {
                            seed += 1;
                            let len = (rect.width() * rect.height()) as usize;
                            (pseudo_random_coefficients(len, seed, integral), rect)
                        })
                        .collect();
                    let source = |index: usize| {
                        let (samples, rect) = &levels[index];
                        CoefficientSource::new(samples, *rect, rect.width())
                    };
                    let coarse = whole_window_level(
                        [source(0), source(1), source(2), source(3)],
                        first,
                        transform,
                    );
                    let expected = whole_window_level(
                        [
                            CoefficientSource::new(&coarse, first, first.width()),
                            source(4),
                            source(5),
                            source(6),
                        ],
                        next,
                        transform,
                    );
                    let mut rows = StoredBands {
                        bands: levels.clone(),
                    };
                    let windows = [first, next];
                    let mut ring =
                        vec![f32::NAN; ring_samples(windows, transform).expect("two-level ring")];
                    let copy = adjusted(next, [-1, -2, -2, -1]);
                    let mut streamed = vec![f32::NAN; expected.len()];
                    reconstruct_rows(
                        &mut rows,
                        &[[1, 2, 3], [4, 5, 6]],
                        &windows,
                        0,
                        transform,
                        &mut ring,
                        copy,
                        &mut |y, samples| {
                            let start = ((y - next.y0) * next.width() + copy.x0 - next.x0) as usize;
                            streamed[start..][..samples.len()].copy_from_slice(samples);
                            Ok(())
                        },
                        &mut output_samples,
                    )
                    .expect("streamed levels");
                    for y in copy.y0..copy.y1 {
                        let start = ((y - next.y0) * next.width() + copy.x0 - next.x0) as usize;
                        let range = start..start + copy.width() as usize;
                        assert_eq!(
                            bits(&streamed[range.clone()]),
                            bits(&expected[range]),
                            "two levels {label} row {y}"
                        );
                    }
                    let overwritten = streamed_rows_overwritten(
                        &mut StoredBands { bands: levels },
                        &[[1, 2, 3], [4, 5, 6]],
                        &windows,
                        transform,
                        copy,
                    );
                    assert_eq!(overwritten.len(), copy.height() as usize, "rows {label}");
                    for (y, row) in overwritten {
                        let start = ((y - next.y0) * next.width() + copy.x0 - next.x0) as usize;
                        assert_eq!(
                            row,
                            bits(&expected[start..][..copy.width() as usize]),
                            "two levels, rows overwritten as read, {label} row {y}"
                        );
                    }
                }
            }
        }
    }
}

/// Band rows wider than 4096 samples that hold no code-block, as in the
/// chroma of an 18000-wide image, read as zero rows through the banded SIMD
/// kernel like narrower ones; every row still matches the whole-window region
/// IDWT bit for bit, with either high band empty.
#[test]
fn wide_empty_band_rows_match_the_whole_window_level_bit_for_bit() {
    use super::lines::{reconstruct_rows, ring_samples};

    for transform in [
        WaveletTransform::Reversible53,
        WaveletTransform::Irreversible97,
    ] {
        let integral = transform == WaveletTransform::Reversible53;
        for (x0, empty) in [(0, 3), (1, 2), (1, 3)] {
            let window = IntRect::from_xywh(x0, 2, 16_400, 6);
            let mut seed = 31_u64;
            let bands: Vec<(Vec<f32>, IntRect)> = super::banded::packed_band_rects(window)
                .into_iter()
                .enumerate()
                .map(|(index, rect)| {
                    let stored = if index == empty {
                        IntRect::from_xywh(0, 0, 0, 0)
                    } else {
                        rect
                    };
                    seed += 1;
                    let len = (stored.width() * stored.height()) as usize;
                    (pseudo_random_coefficients(len, seed, integral), stored)
                })
                .collect();
            let sources: [CoefficientSource<'_>; 4] = core::array::from_fn(|index| {
                let (samples, rect) = &bands[index];
                CoefficientSource::new(samples, *rect, rect.width())
            });
            let expected = whole_window_level(sources, window, transform);
            let mut rows = StoredBands { bands };
            let mut ring = vec![f32::NAN; ring_samples([window], transform).expect("ring")];
            let mut streamed = vec![f32::NAN; expected.len()];
            let mut output_samples = 0;
            reconstruct_rows(
                &mut rows,
                &[[1, 2, 3]],
                &[window],
                0,
                transform,
                &mut ring,
                window,
                &mut |y, samples| {
                    let start = ((y - window.y0) * window.width()) as usize;
                    streamed[start..][..samples.len()].copy_from_slice(samples);
                    Ok(())
                },
                &mut output_samples,
            )
            .expect("streamed level");
            assert_eq!(
                bits(&streamed),
                bits(&expected),
                "{transform:?} origin x {x0}, empty band {empty}"
            );
        }
    }
}

/// A tile streaming from pool bands with deep band rings holds one batch of
/// rows instead of two, so its batch takes half the rows unless that leaves
/// fewer than 128; two pipelined batches keep to 256 rows each; tiles too
/// short for full batches still split their rows into at least four
/// batches, and a row wider than a batch is a batch of its own.
#[test]
fn deep_band_rings_halve_the_streamed_row_batch() {
    use super::roi::streamed_batch_rows;

    // 5661 x 8287 (one tile): 256 rows of each component in each of two
    // pipelined batches (370 fit their samples), 185 rows in the one batch
    // of deep rings.
    assert_eq!(streamed_batch_rows(5661, 8287, false), 256);
    assert_eq!(streamed_batch_rows(5661, 8287, true), 185);
    // 4096 wide: 256 rows either way (512 fit, or half of them).
    assert_eq!(streamed_batch_rows(4096, 4096, false), 256);
    assert_eq!(streamed_batch_rows(4096, 4096, true), 256);
    // 1024 wide: 256 rows in each of two batches, 512 in one.
    assert_eq!(streamed_batch_rows(1024, 1 << 16, false), 256);
    assert_eq!(streamed_batch_rows(1024, 1 << 16, true), 512);
    for deep_band_rings in [false, true] {
        // 18000 wide: 116 rows, too few to halve.
        assert_eq!(streamed_batch_rows(18000, 12000, deep_band_rings), 116);
        // 8200 wide: 255 rows, too few to halve.
        assert_eq!(streamed_batch_rows(8200, 12000, deep_band_rings), 255);
        assert_eq!(streamed_batch_rows(256, 200, deep_band_rings), 50);
        assert_eq!(streamed_batch_rows(256, 3, deep_band_rings), 1);
        assert_eq!(streamed_batch_rows(3 << 20, 4096, deep_band_rings), 1);
    }
}
