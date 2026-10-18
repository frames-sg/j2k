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
