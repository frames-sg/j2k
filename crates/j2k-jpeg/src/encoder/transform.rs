// SPDX-License-Identifier: MIT OR Apache-2.0

use core::f64::consts::PI;
use std::sync::OnceLock;

use j2k_codec_math::jpeg::ZIGZAG;

#[cfg(target_arch = "aarch64")]
#[path = "transform_neon.rs"]
mod neon;

pub(super) struct QuantizationDivisors {
    values: [f64; 64],
    reciprocals: [f64; 64],
    #[cfg(target_arch = "aarch64")]
    f32_reciprocals: [f32; 64],
    #[cfg(target_arch = "aarch64")]
    f32_boundary_thresholds: [f32; 64],
    #[cfg(target_arch = "aarch64")]
    q31_quarter_reciprocals: [i32; 64],
    #[cfg(target_arch = "aarch64")]
    ac_zero_energy_threshold: u32,
    #[cfg(target_arch = "aarch64")]
    neon: Option<fearless_simd::Neon>,
}

pub(super) struct QuantizedBlock {
    pub(super) dc: i16,
    /// Raw quantized AC coefficients in zigzag order; slot zero is unused.
    pub(super) ac_coefficients: [i16; 64],
    /// AC nonzero map with coefficient 1 in the most-significant bit.
    pub(super) nonzero_ac: u64,
}

#[cfg(target_arch = "aarch64")]
impl QuantizedBlock {
    pub(super) const fn zeroed() -> Self {
        Self {
            dc: 0,
            ac_coefficients: [0; 64],
            nonzero_ac: 0,
        }
    }
}

const NATURAL_TO_ZIGZAG: [usize; 64] = natural_to_zigzag();

#[cfg(target_arch = "aarch64")]
const EXACT_INTEGER_AC_MASK: u64 = (1u64 << 4) | (1u64 << 32) | (1u64 << 36);

const fn natural_to_zigzag() -> [usize; 64] {
    let mut inverse = [0usize; 64];
    let mut zigzag = 0;
    while zigzag < 64 {
        inverse[ZIGZAG[zigzag] as usize] = zigzag;
        zigzag += 1;
    }
    inverse
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the asserted positive scaled bound is below 2^24 and converted exactly"
)]
fn f32_boundary_threshold(error_bound: f32) -> f32 {
    debug_assert!((0.0..0.25).contains(&error_bound));
    // Values in [0.25, 0.5) are spaced 2^-25 apart. Scaling by that power of
    // two is exact. Rounding the number of steps upward produces the greatest
    // f32 no larger than the exact 0.5-error_bound threshold. Comparing the
    // nearest-integer fraction against it can add a fallback but cannot omit
    // one selected by the existing guard.
    let steps = (error_bound * 33_554_432.0).ceil() as u32;
    f32::from_bits(0.5_f32.to_bits() - steps)
}

impl QuantizationDivisors {
    #[cfg_attr(
        target_arch = "aarch64",
        expect(
            clippy::cast_possible_truncation,
            clippy::manual_div_ceil,
            reason = "the guarded SIMD paths prove the narrowed representations and retain the explicit directed-reciprocal quotient"
        )
    )]
    pub(super) fn new(quant: &[u8; 64]) -> Self {
        let values = core::array::from_fn(|natural| {
            let u = natural % 8;
            let v = natural / 8;
            8.0 * AAN_SCALE[u] * AAN_SCALE[v] * f64::from(quant[natural])
        });
        let reciprocals = values.map(f64::recip);
        #[cfg(target_arch = "aarch64")]
        let ac_zero_energy_threshold = {
            let min_ac_quantizer = quant[1..]
                .iter()
                .copied()
                .min()
                .expect("JPEG quantization table has AC entries");
            // For D=64*sum(x^2)-sum(x)^2, every normalized AC coefficient is
            // bounded by M*sqrt(D)/8. Exact-rational expansion of the binary64
            // AAN constants and scales proves that every AC row sums to zero
            // and every two-dimensional row has M<1.001. Therefore
            // D<16*(q_min-1)^2 leaves at least
            // 0.5-0.001*254/2=0.373 unquantized units before the zero-bin
            // boundary. After twice the largest f32/f64 error-table entry and
            // threshold rounding, the worst clearance remains above 0.359.
            // The minimum of 16 is a performance gate; smaller quantizers are
            // correct under the same proof but unlikely to amortize the scan.
            if min_ac_quantizer < 16 {
                0
            } else {
                let zero_margin = u32::from(min_ac_quantizer - 1);
                16 * zero_margin * zero_margin
            }
        };
        Self {
            values,
            reciprocals,
            #[cfg(target_arch = "aarch64")]
            f32_reciprocals: {
                let mut reciprocals = reciprocals.map(|reciprocal| reciprocal as f32);
                // The DC butterfly is an exact integer sum in [-8192, 8128].
                // Round its reciprocal upward so exact half-integer results
                // stay on the away-from-zero side of the rounding boundary.
                // Other values are at least 1/(8*q) from a half-integer, much
                // farther than the combined f32 reciprocal/multiply error.
                // This product is exact in f64: the divisor is the integer
                // 8*q <= 2040 and the reciprocal has a 24-bit significand.
                if f64::from(reciprocals[0]) * values[0] < 1.0 {
                    reciprocals[0] = f32::from_bits(reciprocals[0].to_bits() + 1);
                }
                reciprocals
            },
            #[cfg(target_arch = "aarch64")]
            f32_boundary_thresholds: core::array::from_fn(|natural| {
                let error_bound = if natural == 0 {
                    // The directed DC reciprocal above makes rounding exact.
                    0.0
                } else {
                    F32_QUANTIZATION_ERROR_BOUNDS_Q1[natural] / f32::from(quant[natural])
                };
                f32_boundary_threshold(error_bound)
            }),
            #[cfg(target_arch = "aarch64")]
            q31_quarter_reciprocals: {
                // Mirror the proved runtime binary64 sequence exactly: use
                // the stored divisor reciprocal, quarter it, scale to Q31,
                // then round ties to even. Natural DC instead uses the proved
                // upward reciprocal needed for exact signed tie handling.
                let mut q31 = reciprocals.map(|reciprocal| {
                    ((reciprocal * 0.25) * 2_147_483_648.0).round_ties_even() as i32
                });
                let dc_quantizer = u64::from(quant[0]);
                let dc_reciprocal = ((1u64 << 26) + dc_quantizer - 1) / dc_quantizer;
                q31[0] =
                    i32::try_from(dc_reciprocal).expect("baseline JPEG DC reciprocal fits i32");
                q31
            },
            #[cfg(target_arch = "aarch64")]
            ac_zero_energy_threshold,
            #[cfg(target_arch = "aarch64")]
            neon: if cfg!(feature = "scalar-only") {
                None
            } else {
                fearless_simd::Level::new().as_neon()
            },
        }
    }
}

/// Prepare the ordinary standard-table entropy path's private AC view.
///
/// `fdct_quantize` always returns raw coefficients and a raw bitmap. Only this
/// explicit post-correction step may replace the coefficient prefix with
/// packed-Huffman-slot tokens and return the corresponding tagged state.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(super) fn prepare_entropy_ac(
    ac_coefficients: &mut [i16; 64],
    nonzero_ac: u64,
    divisors: &QuantizationDivisors,
) -> u64 {
    const MIN_TOKENIZED_NONZERO_AC: u32 = 24;

    if nonzero_ac == 0
        || nonzero_ac == u64::MAX - 1
        || nonzero_ac.count_ones() < MIN_TOKENIZED_NONZERO_AC
    {
        return nonzero_ac;
    }
    let Some(neon) = divisors.neon else {
        return nonzero_ac;
    };
    let token_count = neon::compact_entropy_ac(neon, ac_coefficients, nonzero_ac);
    debug_assert!((1..=63).contains(&token_count));
    (u64::from(token_count) << 1) | 1
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn prepare_entropy_ac_for_test(
    ac_coefficients: &mut [i16; 64],
    nonzero_ac: u64,
) -> Option<u64> {
    let divisors = QuantizationDivisors::new(&[1; 64]);
    divisors.neon?;
    Some(prepare_entropy_ac(ac_coefficients, nonzero_ac, &divisors))
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "validated eight-bit input bounds every narrowing and measured block dispatch requires this transform in its caller"
)]
#[inline(always)]
pub(super) fn fdct_quantize(
    block: &[u8; 64],
    quant: &[u8; 64],
    divisors: &QuantizationDivisors,
    cosine: &[[f64; 8]; 8],
) -> QuantizedBlock {
    let prefix = u64::from_ne_bytes(
        block[..8]
            .try_into()
            .expect("fixed JPEG block has an eight-sample prefix"),
    );
    // An eight-byte word equals its one-byte rotation iff all bytes are equal.
    let is_constant = prefix == prefix.rotate_right(8) && {
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(neon) = divisors.neon {
                neon::is_constant_block(neon, block)
            } else {
                let first = block[0];
                block[8..].iter().all(|&sample| sample == first)
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let first = block[0];
            block[8..].iter().all(|&sample| sample == first)
        }
    };
    if is_constant {
        let first = block[0];
        let value = f64::from((i32::from(first) - 128) * 64);
        let dc = quantize_exact_boundary(block, quant, cosine, 0, value, divisors.values[0]);
        return QuantizedBlock {
            dc,
            ac_coefficients: [0; 64],
            nonzero_ac: 0,
        };
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = divisors.neon {
        if divisors.ac_zero_energy_threshold != 0 {
            if let Some(centered_sum) =
                neon::ac_zero_energy_centered_sum(neon, block, divisors.ac_zero_energy_threshold)
            {
                // The directed DC reciprocal is exhaustively proven to match
                // the original quantizer for every possible centered sum.
                let dc = (f32::from(centered_sum) * divisors.f32_reciprocals[0]).round() as i16;
                return QuantizedBlock {
                    dc,
                    ac_coefficients: [0; 64],
                    nonzero_ac: 0,
                };
            }
        }

        let mut output = QuantizedBlock {
            dc: 0,
            ac_coefficients: [0; 64],
            nonzero_ac: 0,
        };
        let ambiguous =
            neon::fdct_quantize_q16(neon, block, &divisors.q31_quarter_reciprocals, &mut output);
        if ambiguous != 0 {
            correct_ambiguous_coefficients(block, quant, divisors, cosine, ambiguous, &mut output);
        }
        return output;
    }

    let values = fdct_values(block, divisors);
    let mut dc = 0i16;
    let mut ac_coefficients = [0i16; 64];
    let mut nonzero_ac = 0u64;
    for (natural, value) in values.into_iter().enumerate() {
        let fast_quantized = value * divisors.reciprocals[natural];
        // A reciprocal multiply is substantially cheaper than a division. Its
        // worst-case error over an eight-bit JPEG block is far below this
        // guard band, so only values close enough to change integer rounding
        // repeat the original division.
        let quantized = if (fast_quantized.abs().fract() - 0.5).abs() < 1e-7 {
            value / divisors.values[natural]
        } else {
            fast_quantized
        };
        // Reassociation can move an exact half-integer across the rounding
        // boundary. Preserve the original codestream on those rare ties.
        let coefficient = if (quantized.abs().fract() - 0.5).abs() < 1e-9 {
            quantize_direct(block, quant, cosine, natural)
        } else {
            quantized.round() as i16
        };
        let zigzag_index = NATURAL_TO_ZIGZAG[natural];
        if zigzag_index == 0 {
            dc = coefficient;
        } else {
            ac_coefficients[zigzag_index] = coefficient;
        }
        update_nonzero_ac(&mut nonzero_ac, zigzag_index, coefficient);
    }
    QuantizedBlock {
        dc,
        ac_coefficients,
        nonzero_ac,
    }
}

/// Quantize a block whose rows (`AXIS == 1`) or columns (`AXIS == 2`) repeat.
/// The caller selects the image geometry once, outside the per-block loop.
#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the fixed sample reference preserves the shared scalar and NEON repeated-axis ABI"
)]
pub(super) fn fdct_quantize_repeated_axis<const AXIS: u8>(
    samples: &[u8; 8],
    quant: &[u8; 64],
    divisors: &QuantizationDivisors,
    cosine: &[[f64; 8]; 8],
) -> QuantizedBlock {
    let mut output = QuantizedBlock::zeroed();
    fdct_quantize_repeated_axis_core::<AXIS>(samples, quant, divisors, cosine, &mut output);
    output
}

/// Write a compact repeated edge into a zero-initialized caller-owned block.
#[cfg(all(target_arch = "aarch64", not(feature = "scalar-only")))]
#[inline(never)]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the fixed sample reference preserves the caller-owned repeated-axis ABI"
)]
pub(super) fn fdct_quantize_repeated_axis_into<const AXIS: u8>(
    samples: &[u8; 8],
    quant: &[u8; 64],
    divisors: &QuantizationDivisors,
    cosine: &[[f64; 8]; 8],
    output: &mut QuantizedBlock,
) {
    fdct_quantize_repeated_axis_core::<AXIS>(samples, quant, divisors, cosine, output);
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::cast_possible_truncation,
    clippy::inline_always,
    reason = "validated eight-bit coefficients fit i16 and the measured axis dispatch keeps the guarded core in its callers"
)]
#[inline(always)]
fn fdct_quantize_repeated_axis_core<const AXIS: u8>(
    samples: &[u8; 8],
    quant: &[u8; 64],
    divisors: &QuantizationDivisors,
    cosine: &[[f64; 8]; 8],
    output: &mut QuantizedBlock,
) {
    debug_assert!(AXIS == 1 || AXIS == 2);
    if let Some(neon) = divisors.neon {
        let first = samples[0];
        if *samples == [first; 8] {
            // Match the established directed-reciprocal path: the repeated
            // DC butterfly is the exact centered sample times 64, followed
            // by the stored f32 reciprocal and ties-away rounding.
            let value = f32::from((i16::from(first) - 128) * 64);
            output.dc = (value * divisors.f32_reciprocals[0]).round() as i16;
            return;
        }

        let ambiguous = neon::fdct_quantize_repeated_axis_f32::<AXIS>(
            neon,
            samples,
            &divisors.f32_reciprocals,
            &divisors.f32_boundary_thresholds,
            output,
        );
        if !ambiguous {
            return;
        }
    }
    // Scalar-only builds and the rare guarded coefficient retain the complete
    // established transform and exact-boundary correction path.
    fdct_quantize_repeated_axis_fallback_into::<AXIS>(samples, quant, divisors, cosine, output);
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the fallback shares the fixed sample-reference ABI with the guarded repeated-axis path"
)]
fn fdct_quantize_repeated_axis_fallback_into<const AXIS: u8>(
    samples: &[u8; 8],
    quant: &[u8; 64],
    divisors: &QuantizationDivisors,
    cosine: &[[f64; 8]; 8],
    output: &mut QuantizedBlock,
) {
    let block: [u8; 64] = match AXIS {
        1 => core::array::from_fn(|index| samples[index % 8]),
        2 => core::array::from_fn(|index| samples[index / 8]),
        _ => unreachable!("repeated JPEG FDCT axis is rows or columns"),
    };
    *output = fdct_quantize(&block, quant, divisors, cosine);
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn correct_ambiguous_coefficients(
    block: &[u8; 64],
    quant: &[u8; 64],
    divisors: &QuantizationDivisors,
    cosine: &[[f64; 8]; 8],
    mut ambiguous: u64,
    output: &mut QuantizedBlock,
) {
    // Natural frequencies 4, 32, and 36 use only the u/v=0 or 4
    // add/subtract butterflies, whose AAN scales are one. Their transform
    // value K and every intermediate are exactly representable integers;
    // K is in [-8160, 8160], and the divisor is the exact integer 8*q. For
    // every q in 1..=255, a value K/(8*q) that is not a tie is at least
    // 1/(8*q) from a half-integer. The former f32 path's conservative error
    // bound plus its rounded threshold is far below that gap. For the Q16
    // path, these frequencies have at most about 0.0000381 quantized units of
    // error and use a zero-unit guard that selects only exact Q14
    // half-integers. That error remains below the minimum non-tie gap of
    // 1/(8*255) > 0.00049. Either ambiguity guard can therefore select these
    // bits only for true half-integers. Those ties are exactly representable
    // in f64, so the former exact-boundary path necessarily selected the same
    // direct quantizer. Call it without first computing the complete f64 AAN
    // transform.
    let exact_values =
        ((ambiguous & !EXACT_INTEGER_AC_MASK) != 0).then(|| fdct_values(block, divisors));
    while ambiguous != 0 {
        let natural = ambiguous.trailing_zeros() as usize;
        ambiguous &= ambiguous - 1;
        let coefficient = if EXACT_INTEGER_AC_MASK & (1u64 << natural) != 0 {
            quantize_direct(block, quant, cosine, natural)
        } else {
            quantize_exact_boundary(
                block,
                quant,
                cosine,
                natural,
                exact_values
                    .as_ref()
                    .expect("ambiguous SIMD coefficient has exact transform values")[natural],
                divisors.values[natural],
            )
        };
        let zigzag_index = NATURAL_TO_ZIGZAG[natural];
        if zigzag_index == 0 {
            output.dc = coefficient;
        } else {
            output.ac_coefficients[zigzag_index] = coefficient;
        }
        update_nonzero_ac(&mut output.nonzero_ac, zigzag_index, coefficient);
    }
}

#[expect(
    clippy::inline_always,
    reason = "bitmap maintenance stays folded into the measured quantization loops"
)]
#[inline(always)]
fn update_nonzero_ac(bitmap: &mut u64, zigzag: usize, coefficient: i16) {
    if zigzag == 0 {
        return;
    }
    let mask = 1u64 << (64 - zigzag);
    if coefficient == 0 {
        *bitmap &= !mask;
    } else {
        *bitmap |= mask;
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "validated baseline samples and quantizers bound the rounded coefficient to i16"
)]
fn quantize_exact_boundary(
    block: &[u8; 64],
    quant: &[u8; 64],
    cosine: &[[f64; 8]; 8],
    natural: usize,
    value: f64,
    divisor: f64,
) -> i16 {
    let quantized = value / divisor;
    if (quantized.abs().fract() - 0.5).abs() < 1e-9 {
        return quantize_direct(block, quant, cosine, natural);
    }
    quantized.round() as i16
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the direct baseline DCT proof bounds the rounded coefficient to i16"
)]
fn quantize_direct(
    block: &[u8; 64],
    quant: &[u8; 64],
    cosine: &[[f64; 8]; 8],
    natural: usize,
) -> i16 {
    let v = natural / 8;
    let u = natural % 8;
    let direct = if natural == 0 {
        // cosine[0] is exactly one. Centering and summing eight-bit samples is
        // exact in both i32 and f64, so this preserves the original operation
        // result while avoiding 128 multiplications by one.
        f64::from(
            block
                .iter()
                .map(|&sample| i32::from(sample) - 128)
                .sum::<i32>(),
        )
    } else {
        let mut direct = 0.0;
        for y in 0..8 {
            for x in 0..8 {
                direct += (f64::from(block[y * 8 + x]) - 128.0) * cosine[u][x] * cosine[v][y];
            }
        }
        direct
    };
    let cu = if u == 0 {
        core::f64::consts::FRAC_1_SQRT_2
    } else {
        1.0
    };
    let cv = if v == 0 {
        core::f64::consts::FRAC_1_SQRT_2
    } else {
        1.0
    };
    (0.25 * cu * cv * direct / f64::from(quant[natural])).round() as i16
}

fn fdct_values(block: &[u8; 64], divisors: &QuantizationDivisors) -> [f64; 64] {
    #[cfg(target_arch = "aarch64")]
    if let Some(neon) = divisors.neon {
        return neon::fdct_values(neon, block);
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = divisors;
    fdct_values_scalar(block)
}

fn fdct_values_scalar(block: &[u8; 64]) -> [f64; 64] {
    let mut columns = [[0.0; 8]; 8];
    for y in 0..8 {
        let row = fdct_8(core::array::from_fn(|x| {
            f64::from(block[y * 8 + x]) - 128.0
        }));
        for u in 0..8 {
            columns[u][y] = row[u];
        }
    }
    let mut values = [0.0; 64];
    for (u, column) in columns.into_iter().enumerate() {
        for (v, value) in fdct_8(column).into_iter().enumerate() {
            values[v * 8 + u] = value;
        }
    }
    values
}

// Arai–Agui–Nakajima scaling: the five-multiply 1-D transform defers
// normalization to quantization. F64 plus the tie fallback above preserves
// the existing encoder's rounding, including its byte-level golden fixtures.
const AAN_SCALE: [f64; 8] = [
    1.0,
    1.387_039_845_322_147_5,
    1.306_562_964_876_376_6,
    1.175_875_602_419_358_8,
    1.0,
    0.785_694_958_387_102_2,
    0.541_196_100_146_197_1,
    0.275_899_379_282_943_1,
];

// Conservative absolute error, in quantization-bin units, for the f32 AAN
// transform plus reciprocal quantization at q=1. Each output is tracked as an
// exact-real linear form over centered samples, with
// B=128*sum(abs(coefficients)); this preserves cancellation instead of
// repeatedly adding operand magnitude bounds. Integer butterflies are exact
// while B<=2^24. Otherwise an add/subtract propagates
// E'=Ea+Eb+u(B+Ea+Eb), and multiplication by c propagates
// E'=|f32(c)|E+|f32(c)-c|B+u|f32(c)|(B+E). F32 and f64 errors are tracked
// independently through the exact operation order of both 1-D passes, then
// reciprocal and final-product rounding are added. The resulting table has a
// maximum 0.006955415 bins after 12.5% plus 1e-6 margin. The reciprocal scales
// as 1/q, and every absolute transform/error term is multiplied by it, so the
// constructor divides by the actual quantizer without weakening the bound.
#[cfg(target_arch = "aarch64")]
#[rustfmt::skip]
#[expect(
    clippy::excessive_precision,
    reason = "the generated decimal bounds intentionally retain their established f32 rounding"
)]
const F32_QUANTIZATION_ERROR_BOUNDS_Q1: [f32; 64] = [
    0.000_138_330, 0.000_538_303, 0.000_429_458, 0.000_522_305, 0.000_138_330, 0.000_595_798, 0.000_497_102, 0.001_198_742,
    0.000_351_646, 0.000_739_154, 0.000_612_516, 0.000_720_245, 0.000_351_646, 0.000_807_116, 0.000_692_472, 0.001_519_811,
    0.000_239_145, 0.000_582_835, 0.000_483_949, 0.000_568_055, 0.000_239_145, 0.000_635_954, 0.000_546_444, 0.001_193_001,
    0.000_335_648, 0.000_686_042, 0.000_571_874, 0.000_668_943, 0.000_335_648, 0.000_747_497, 0.000_644_174, 0.001_391_953,
    0.000_138_330, 0.000_538_303, 0.000_429_458, 0.000_522_305, 0.000_138_330, 0.000_595_798, 0.000_497_102, 0.001_198_742,
    0.000_409_142, 0.000_942_245, 0.000_769_734, 0.000_916_654, 0.000_409_142, 0.001_034_218, 0.000_877_939, 0.001_998_715,
    0.000_306_788, 0.001_161_794, 0.000_918_286, 0.001_126_112, 0.000_306_788, 0.001_290_035, 0.001_069_160, 0.002_634_866,
    0.001_012_085, 0.003_030_787, 0.002_380_790, 0.002_935_723, 0.001_012_085, 0.003_372_453, 0.002_782_757, 0.006_955_415,
];

#[inline]
fn fdct_8(input: [f64; 8]) -> [f64; 8] {
    let sum: [f64; 4] = core::array::from_fn(|i| input[i] + input[7 - i]);
    let diff: [f64; 4] = core::array::from_fn(|i| input[i] - input[7 - i]);
    let outer = sum[0] + sum[3];
    let inner = sum[1] + sum[2];
    let even = sum[0] - sum[3];
    let rotation = (even + sum[1] - sum[2]) * core::f64::consts::FRAC_1_SQRT_2;
    let left = diff[3] + diff[2];
    let right = diff[1] + diff[0];
    let cross = (left - right) * 0.382_683_432_365_089_8;
    let low = left * 0.541_196_100_146_197 + cross;
    let high = right * 1.306_562_964_876_376_6 + cross;
    let middle = (diff[2] + diff[1]) * core::f64::consts::FRAC_1_SQRT_2;
    [
        outer + inner,
        diff[0] + middle + high,
        even + rotation,
        diff[0] - middle - low,
        outer - inner,
        diff[0] - middle + low,
        even - rotation,
        diff[0] + middle - high,
    ]
}

/// Forward DCT and quantization of one block of level-shifted samples at a
/// precision up to 12 bits, in natural order: the 8-bit path's AAN
/// transform, rounding half away from zero, with the same exact fallback for
/// results within rounding error of a tie.
#[expect(
    clippy::cast_possible_truncation,
    reason = "12-bit DCT coefficients are bounded by 8 * 2048 and fit i16"
)]
pub(super) fn fdct_quantize_centered(
    centered: &[i32; 64],
    quant: &[u16; 64],
    cosine: &[[f64; 8]; 8],
) -> [i16; 64] {
    let mut columns = [[0.0; 8]; 8];
    for y in 0..8 {
        let row = fdct_8(core::array::from_fn(|x| f64::from(centered[y * 8 + x])));
        for u in 0..8 {
            columns[u][y] = row[u];
        }
    }
    let mut values = [0.0; 64];
    for (u, column) in columns.into_iter().enumerate() {
        for (v, value) in fdct_8(column).into_iter().enumerate() {
            values[v * 8 + u] = value;
        }
    }
    core::array::from_fn(|natural| {
        let divisor =
            8.0 * AAN_SCALE[natural % 8] * AAN_SCALE[natural / 8] * f64::from(quant[natural]);
        let quantized = values[natural] / divisor;
        if (quantized.abs().fract() - 0.5).abs() < 1e-9 {
            quantize_direct_centered(centered, quant, cosine, natural)
        } else {
            quantized.round() as i16
        }
    })
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "12-bit DCT coefficients are bounded by 8 * 2048 and fit i16"
)]
fn quantize_direct_centered(
    centered: &[i32; 64],
    quant: &[u16; 64],
    cosine: &[[f64; 8]; 8],
    natural: usize,
) -> i16 {
    let (u, v) = (natural % 8, natural / 8);
    let mut direct = 0.0;
    for y in 0..8 {
        for x in 0..8 {
            direct += f64::from(centered[y * 8 + x]) * cosine[u][x] * cosine[v][y];
        }
    }
    let scale = |index: usize| {
        if index == 0 {
            core::f64::consts::FRAC_1_SQRT_2
        } else {
            1.0
        }
    };
    (0.25 * scale(u) * scale(v) * direct / f64::from(quant[natural])).round() as i16
}

pub(super) fn cosine_table() -> &'static [[f64; 8]; 8] {
    static TABLE: OnceLock<[[f64; 8]; 8]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0; 8]; 8];
        for (u, row) in (0u32..8).zip(&mut table) {
            for (x, value) in (0u32..8).zip(row) {
                *value = ((f64::from(2 * x + 1) * f64::from(u) * PI) / 16.0).cos();
            }
        }
        table
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "aarch64")]
    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "eight-bit block sums fit exactly in f32 and rounded DC coefficients fit i32"
    )]
    fn directed_dc_reciprocal_matches_direct_rounding_for_every_sum_and_quantizer() {
        for q in 1u8..=255 {
            let divisors = QuantizationDivisors::new(&[q; 64]);
            for sum in -8192i32..=8128 {
                let actual = (sum as f32 * divisors.f32_reciprocals[0]).round() as i32;
                let expected = (0.25
                    * core::f64::consts::FRAC_1_SQRT_2
                    * core::f64::consts::FRAC_1_SQRT_2
                    * f64::from(sum)
                    / f64::from(q))
                .round() as i32;
                assert_eq!(actual, expected, "sum {sum}, quantizer {q}");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn nearest_guard_thresholds_conservatively_include_the_previous_guard() {
        fn previous_guard(value: f32, error_bound: f32) -> bool {
            let fraction = (value - value.trunc()).abs();
            (fraction - 0.5).abs() < error_bound
        }

        fn nearest_guard(value: f32, threshold: f32) -> bool {
            let nearest_fraction = (value - value.round_ties_even()).abs();
            nearest_fraction > threshold
        }

        for q in 1u8..=255 {
            let divisors = QuantizationDivisors::new(&[q; 64]);
            for (natural, &base_error_bound) in F32_QUANTIZATION_ERROR_BOUNDS_Q1.iter().enumerate()
            {
                let error_bound = if natural == 0 {
                    0.0
                } else {
                    base_error_bound / f32::from(q)
                };
                let threshold = divisors.f32_boundary_thresholds[natural];
                let exact_threshold = 0.5_f64 - f64::from(error_bound);
                assert!(f64::from(threshold) <= exact_threshold);
                assert!(f64::from(f32::from_bits(threshold.to_bits() + 1)) > exact_threshold);

                // Probe both fractional sides and signs at the representable
                // values that straddle the conservative threshold. The floor
                // property above supplies the proof for all other values.
                for nearest_fraction in [threshold, f32::from_bits(threshold.to_bits() + 1)] {
                    for value in [nearest_fraction, 1.0 - nearest_fraction] {
                        assert!(
                            !previous_guard(value, error_bound) || nearest_guard(value, threshold)
                        );
                        assert!(
                            !previous_guard(-value, error_bound)
                                || nearest_guard(-value, threshold)
                        );
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn repeated_axis_fdct_matches_the_complete_transform() {
        let cosine = cosine_table();
        let mut seed = 29u32;
        for case in 0..384 {
            let samples: [u8; 8] = core::array::from_fn(|index| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match case {
                    0..=255 => u8::try_from(case).expect("bounded case fits u8"),
                    256 => u8::from(index == 0) * 255,
                    257 => u8::from(index & 1 == 0) * 255,
                    _ => (seed >> 24) as u8,
                }
            });
            let repeated_rows: [u8; 64] = core::array::from_fn(|index| samples[index % 8]);
            let repeated_columns: [u8; 64] = core::array::from_fn(|index| samples[index / 8]);
            let quantizers = [
                [1u8; 64],
                [3u8; 64],
                [16u8; 64],
                [255u8; 64],
                core::array::from_fn(|index| {
                    u8::try_from((index * 37) % 255 + 1).expect("quantizer fits u8")
                }),
            ];
            for quant in quantizers {
                let divisors = QuantizationDivisors::new(&quant);
                let expected_rows = fdct_quantize(&repeated_rows, &quant, &divisors, cosine);
                let actual_rows =
                    fdct_quantize_repeated_axis::<1>(&samples, &quant, &divisors, cosine);
                assert_eq!(actual_rows.dc, expected_rows.dc, "row DC case {case}");
                assert_eq!(
                    actual_rows.ac_coefficients, expected_rows.ac_coefficients,
                    "row AC case {case}"
                );
                assert_eq!(
                    actual_rows.nonzero_ac, expected_rows.nonzero_ac,
                    "row bitmap case {case}"
                );

                let expected_columns = fdct_quantize(&repeated_columns, &quant, &divisors, cosine);
                let actual_columns =
                    fdct_quantize_repeated_axis::<2>(&samples, &quant, &divisors, cosine);
                assert_eq!(
                    actual_columns.dc, expected_columns.dc,
                    "column DC case {case}"
                );
                assert_eq!(
                    actual_columns.ac_coefficients, expected_columns.ac_coefficients,
                    "column AC case {case}"
                );
                assert_eq!(
                    actual_columns.nonzero_ac, expected_columns.nonzero_ac,
                    "column bitmap case {case}"
                );
            }
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::too_many_lines,
        reason = "the reference transform covers bounded eight-bit samples and quantized coefficients"
    )]
    fn separable_dct_preserves_quantization_including_rounding_ties() {
        let cosine = cosine_table();
        let mut seed = 17u32;
        for case in 0..390 {
            let block = core::array::from_fn(|i| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                match case {
                    0..=255 => u8::try_from(case).expect("bounded case fits u8"),
                    256 => {
                        if i == 0 {
                            255
                        } else {
                            0
                        }
                    }
                    257 => {
                        if i % 2 == 0 {
                            255
                        } else {
                            0
                        }
                    }
                    // D=63*7^2=3,087 is below the q=16 threshold 3,600.
                    384 => 128 + u8::from(i == 0) * 7,
                    // D=63*8^2=4,032 is above the q=16 threshold.
                    385 => 128 + u8::from(i == 0) * 8,
                    // D=128*89^2=1,013,888 is below the q=255 threshold
                    // 1,032,256.
                    386 => match i {
                        0 => 217,
                        1 => 39,
                        _ => 128,
                    },
                    // D=128*90^2=1,036,800 is above the q=255 threshold.
                    387 => match i {
                        0 => 218,
                        1 => 38,
                        _ => 128,
                    },
                    // Nonconstant q=1 blocks whose centered sums put DC on
                    // each signed half boundary in the Q16 SIMD path.
                    388 => 128 + u8::from(i == 0) * 4,
                    389 => 128 - u8::from(i == 0) * 4,
                    _ => (seed >> 24) as u8,
                }
            });
            let mixed_quant = core::array::from_fn(|natural| match natural {
                0 => 1,
                1 => 16,
                _ => 255,
            });
            for (quant_case, quant) in [[1u8; 64], [3u8; 64], [16u8; 64], [255u8; 64]]
                .into_iter()
                .chain(matches!(case, 384 | 385).then_some(mixed_quant))
                .enumerate()
            {
                let divisors = QuantizationDivisors::new(&quant);
                #[cfg(target_arch = "aarch64")]
                if let Some(neon) = divisors.neon.filter(|_| {
                    (384..=387).contains(&case) && divisors.ac_zero_energy_threshold != 0
                }) {
                    let sum = block.iter().copied().map(u32::from).sum::<u32>();
                    let square_sum = block
                        .iter()
                        .copied()
                        .map(u32::from)
                        .map(|sample| sample * sample)
                        .sum::<u32>();
                    let energy = 64 * square_sum - sum * sum;
                    let expected =
                        (energy < divisors.ac_zero_energy_threshold).then_some(sum as i16 - 8_192);
                    assert_eq!(
                        neon::ac_zero_energy_centered_sum(
                            neon,
                            &block,
                            divisors.ac_zero_energy_threshold,
                        ),
                        expected,
                        "energy gate case {case}, quantizer case {quant_case}"
                    );
                }
                let actual = fdct_quantize(&block, &quant, &divisors, cosine);
                let reference: [i32; 64] = core::array::from_fn(|k| {
                    let (v, u) = (k / 8, k % 8);
                    let mut sum = 0.0;
                    for y in 0..8 {
                        for x in 0..8 {
                            sum +=
                                (f64::from(block[y * 8 + x]) - 128.0) * cosine[u][x] * cosine[v][y];
                        }
                    }
                    let cu = if u == 0 {
                        core::f64::consts::FRAC_1_SQRT_2
                    } else {
                        1.0
                    };
                    let cv = if v == 0 {
                        core::f64::consts::FRAC_1_SQRT_2
                    } else {
                        1.0
                    };
                    (0.25 * cu * cv * sum / f64::from(quant[k])).round() as i32
                });
                let actual_natural: [i32; 64] = core::array::from_fn(|natural| {
                    let zigzag = NATURAL_TO_ZIGZAG[natural];
                    if zigzag == 0 {
                        return i32::from(actual.dc);
                    }
                    i32::from(actual.ac_coefficients[zigzag])
                });
                assert_eq!(
                    actual_natural, reference,
                    "case {case}, quantizer case {quant_case}"
                );
                let expected_nonzero = (1..64).fold(0u64, |bitmap, zigzag| {
                    bitmap | u64::from(actual.ac_coefficients[zigzag] != 0) << (64 - zigzag)
                });
                assert_eq!(actual.nonzero_ac, expected_nonzero);
            }
        }
    }
}
