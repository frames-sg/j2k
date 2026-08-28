//! Forward quantization for JPEG 2000 encoding.
//!
//! - Lossless (reversible 5-3): No quantization, just sign/magnitude conversion
//! - Lossy (irreversible 9-7): Scalar deadzone quantization with step sizes
//!   derived from the DWT subband gain norms.

use alloc::vec;
use alloc::vec::Vec;

use crate::math::{floor_f32, log2_f32, pow2i, round_f32};
use crate::{
    EncodeError, EncodeResult, IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales,
    J2kSubBandType,
};

/// Quantization parameters for a single subband.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct QuantStepSize {
    pub(crate) exponent: u16,
    pub(crate) mantissa: u16,
}

pub(crate) fn subband_scales_all_valid(scales: IrreversibleQuantizationSubbandScales) -> bool {
    [
        scales.low_low,
        scales.high_low,
        scales.low_high,
        scales.high_high,
    ]
    .iter()
    .all(|scale| scale.is_finite() && *scale > 0.0)
}

fn subband_scale_for_step_index(
    scales: IrreversibleQuantizationSubbandScales,
    index: usize,
) -> f32 {
    if index == 0 {
        return scales.low_low;
    }
    match (index - 1) % 3 {
        0 => scales.high_low,
        1 => scales.low_high,
        _ => scales.high_high,
    }
}

fn subband_scale_for_subband(
    scales: IrreversibleQuantizationSubbandScales,
    subband: J2kSubBandType,
) -> f32 {
    match subband {
        J2kSubBandType::LowLow => scales.low_low,
        J2kSubBandType::HighLow => scales.high_low,
        J2kSubBandType::LowHigh => scales.low_high,
        J2kSubBandType::HighHigh => scales.high_high,
    }
}

impl QuantStepSize {
    /// Compute the JPEG 2000 irreversible step size:
    /// `Δ = 2^(R_b - exponent) × (1 + mantissa / 2048)`.
    #[expect(
        clippy::trivially_copy_pass_by_ref,
        reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
    )]
    fn delta(&self, range_bits: u8) -> f32 {
        let rb = i32::from(range_bits) - i32::from(self.exponent);
        let base = pow2i(rb);
        base * (1.0 + f32::from(self.mantissa) / 2048.0)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "finite quantization values are explicitly rounded and clamped immediately after conversion"
    )]
    fn from_delta(range_bits: u8, delta: f32) -> Self {
        debug_assert!(delta.is_finite() && delta > 0.0);

        let floor_log2 = floor_f32(log2_f32(delta)) as i32;
        let mut exponent = i32::from(range_bits) - floor_log2;
        let normalized = delta / pow2i(floor_log2);
        let mut mantissa = round_f32((normalized - 1.0) * 2048.0) as i32;

        if mantissa >= 2048 {
            exponent -= 1;
            mantissa = 0;
        }

        Self {
            exponent: u16::try_from(exponent.clamp(0, 31)).unwrap_or_default(),
            mantissa: u16::try_from(mantissa.clamp(0, 2047)).unwrap_or_default(),
        }
    }
}

/// Compute the exact irreversible 9/7 quantization step tuple the native encoder
/// writes for one subband under a global plus per-subband profile.
///
/// # Panics
///
/// Panics if the internal quantization step exponent is not clamped to the
/// JPEG 2000 exponent range before conversion.
#[must_use]
pub fn irreversible_quantization_step_for_subband(
    bit_depth: u8,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
    subband: J2kSubBandType,
) -> IrreversibleQuantizationStep {
    let base_step = QuantStepSize {
        exponent: u16::from(bit_depth) + u16::from(guard_bits),
        mantissa: 0,
    };
    let scale =
        if irreversible_quantization_scale.is_finite() && irreversible_quantization_scale > 0.0 {
            irreversible_quantization_scale
        } else {
            1.0
        };
    let subband_scales = if subband_scales_all_valid(irreversible_quantization_subband_scales) {
        irreversible_quantization_subband_scales
    } else {
        IrreversibleQuantizationSubbandScales::default()
    };
    let step_size = QuantStepSize::from_delta(
        bit_depth,
        base_step.delta(bit_depth) * scale * subband_scale_for_subband(subband_scales, subband),
    );
    IrreversibleQuantizationStep {
        exponent: u8::try_from(step_size.exponent).unwrap_or(u8::MAX),
        mantissa: step_size.mantissa,
    }
}

/// Compute default quantization step sizes for the irreversible 9-7 transform.
///
/// The step sizes are derived from the DWT 9-7 subband gain norms (Table E.1 in T.800).
/// For lossless mode, step sizes are not used (exponents store bit depth info only).
#[cfg(test)]
pub(crate) fn compute_step_sizes(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
) -> Vec<QuantStepSize> {
    compute_step_sizes_with_irreversible_scale(
        bit_depth,
        num_decompositions,
        reversible,
        guard_bits,
        1.0,
    )
}

/// Compute quantization step sizes with an irreversible 9-7 scale multiplier.
///
/// A scale of 1.0 preserves the quality-first default. Larger scales coarsen
/// the irreversible quantizer while keeping the same subband gain relationship.
#[cfg(test)]
pub(crate) fn compute_step_sizes_with_irreversible_scale(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
) -> Vec<QuantStepSize> {
    compute_step_sizes_with_irreversible_profile(
        bit_depth,
        num_decompositions,
        reversible,
        guard_bits,
        irreversible_quantization_scale,
        IrreversibleQuantizationSubbandScales::default(),
    )
}

/// Compute quantization step sizes with global and per-subband irreversible
/// 9/7 scale multipliers.
#[cfg(test)]
pub(crate) fn compute_step_sizes_with_irreversible_profile(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
) -> Vec<QuantStepSize> {
    let mut step_sizes = Vec::new();
    append_step_sizes_with_irreversible_profile(
        &mut step_sizes,
        bit_depth,
        num_decompositions,
        reversible,
        guard_bits,
        irreversible_quantization_scale,
        irreversible_quantization_subband_scales,
    );
    step_sizes
}

pub(crate) fn append_step_sizes_with_irreversible_profile(
    step_sizes: &mut Vec<QuantStepSize>,
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
) {
    if reversible {
        // For reversible 5-3, QCD stores the subband exponent only.
        // The decoder reconstructs the number of bitplanes as:
        //   Mb = guard_bits + exponent - 1
        // For lossless coding we therefore need exponents that reproduce the
        // reversible subband dynamic range:
        //   LL => bit_depth + 0
        //   HL/LH => bit_depth + 1
        //   HH => bit_depth + 2
        // This gain depends on subband orientation, not decomposition level.
        step_sizes.push(QuantStepSize {
            exponent: u16::from(bit_depth),
            mantissa: 0,
        });

        for _ in 0..num_decompositions {
            step_sizes.push(QuantStepSize {
                exponent: u16::from(bit_depth) + 1,
                mantissa: 0,
            });
            step_sizes.push(QuantStepSize {
                exponent: u16::from(bit_depth) + 1,
                mantissa: 0,
            });
            step_sizes.push(QuantStepSize {
                exponent: u16::from(bit_depth) + 2,
                mantissa: 0,
            });
        }
    } else {
        // Quality-first irreversible 9-7 default. Use one exponent/mantissa for all
        // subbands and let R_b = bit_depth + log_gain make LL finest and HH
        // coarsest under the decoder's QCD formula.
        let base_step = QuantStepSize {
            exponent: u16::from(bit_depth) + u16::from(guard_bits),
            mantissa: 0,
        };
        let scale = if irreversible_quantization_scale.is_finite()
            && irreversible_quantization_scale > 0.0
        {
            irreversible_quantization_scale
        } else {
            1.0
        };
        let subband_scales = if subband_scales_all_valid(irreversible_quantization_subband_scales) {
            irreversible_quantization_subband_scales
        } else {
            IrreversibleQuantizationSubbandScales::default()
        };
        let step_count = 1usize + usize::from(num_decompositions) * 3;

        for index in 0..step_count {
            let subband_scale = subband_scale_for_step_index(subband_scales, index);
            step_sizes.push(QuantStepSize::from_delta(
                bit_depth,
                base_step.delta(bit_depth) * scale * subband_scale,
            ));
        }
    }
}

const OPENHTJ2K_D97_ENERGY_NORMS: [(f64, f64); 16] = [
    (1.965_907_314_575_295_7, 2.080_871_927_589_849),
    (4.122_409_873_969_023, 3.868_863_224_131_922),
    (8.416_744_177_952_724, 8.317_022_299_806_517),
    (16.935_572_073_021_724, 17.201_929_112_787_134),
    (33.924_926_802_207_46, 34.746_895_711_342_454),
    (67.877_165_259_517_36, 69.675_395_886_752_16),
    (135.768_047_117_209_76, 139.443_143_900_563_17),
    (271.542_960_998_980_6, 278.932_688_221_656_86),
    (543.089_356_530_109_5, 557.888_607_932_094_2),
    (1_086.180_430_479_691_7, 1_115.788_835_859_533_3),
    (2_172.361_719_689_337, 2_231.583_482_284_095_7),
    (4_344.723_868_746_226, 4_463.169_869_925_348),
    (8_689.447_952_176_557, 8_926.341_192_539_1),
    (17_378.896_011_694_746, 17_852.683_111_423_37),
    (34_757.792_077_060_57, 35_705.366_586_020_52),
    (69_515.584_180_957_42, 71_410.733_353_619_97),
];

const OPENHTJ2K_VISUAL_WEIGHTS_444: [[f64; 15]; 3] = [
    [
        0.0901, 0.2758, 0.2758, 0.7018, 0.8378, 0.8378, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
    [
        0.0263, 0.0863, 0.0863, 0.1362, 0.2564, 0.2564, 0.3346, 0.4691, 0.4691, 0.5444, 0.6523,
        0.6523, 0.7078, 0.7797, 0.7797,
    ],
    [
        0.0773, 0.1835, 0.1835, 0.2598, 0.4130, 0.4130, 0.5040, 0.6464, 0.6464, 0.7220, 0.8254,
        0.8254, 0.8769, 0.9424, 0.9424,
    ],
];

const OPENHTJ2K_COLOR_GAINS: [f64; 3] = [1.7321, 1.8051, 1.5734];

/// Append the expounded 9/7 QCD or QCC tuples used by `OpenHTJ2K`'s Qfactor
/// profile for grayscale or 4:4:4 YCbCr content.
pub(crate) fn append_openhtj2k_qfactor_step_sizes(
    step_sizes: &mut Vec<QuantStepSize>,
    bit_depth: u8,
    num_decompositions: u8,
    qfactor: u8,
    component: usize,
) -> EncodeResult<()> {
    if !(1..=100).contains(&qfactor) {
        return Err(EncodeError::InvalidInput {
            what: "OpenHTJ2K Qfactor must be in 1..=100",
        });
    }
    let Some(visual_weights) = OPENHTJ2K_VISUAL_WEIGHTS_444.get(component) else {
        return Err(EncodeError::InvalidInput {
            what: "OpenHTJ2K Qfactor supports grayscale or three-component 4:4:4 input",
        });
    };
    let level_count = usize::from(num_decompositions);
    if level_count > OPENHTJ2K_D97_ENERGY_NORMS.len() {
        return Err(EncodeError::Unsupported {
            what: "OpenHTJ2K Qfactor supports at most 16 decomposition levels",
        });
    }

    let qfactor = f64::from(qfactor);
    let magnitude = if qfactor < 50.0 {
        50.0 / qfactor
    } else {
        2.0 * (1.0 - qfactor / 100.0)
    };
    let mut power = 1.0;
    let mut alpha = 0.04;
    if qfactor >= 97.0 {
        power = 0.0;
        alpha = 0.10;
    } else if qfactor > 65.0 {
        let magnitude_t0 = 2.0 * (1.0 - 65.0 / 100.0);
        let magnitude_t1 = 2.0 * (1.0 - 97.0 / 100.0);
        power = (libm::log(magnitude_t1) - libm::log(magnitude))
            / (libm::log(magnitude_t1) - libm::log(magnitude_t0));
        alpha = 0.10 * libm::pow(0.04 / 0.10, power);
    }

    let epsilon = libm::sqrt(0.5) / libm::scalbn(1.0, i32::from(bit_depth));
    let delta_reference = alpha * magnitude * OPENHTJ2K_COLOR_GAINS[0] + epsilon;
    let component_gain = OPENHTJ2K_COLOR_GAINS[component];
    let start = step_sizes.len();
    for (level, &(low, high)) in OPENHTJ2K_D97_ENERGY_NORMS[..level_count].iter().enumerate() {
        let base = level * 3;
        for (offset, wmse) in [high * high, low * high, high * low]
            .into_iter()
            .enumerate()
        {
            let weight = visual_weights.get(base + offset).copied().unwrap_or(1.0);
            let delta =
                delta_reference / (libm::sqrt(wmse) * libm::pow(weight, power) * component_gain);
            step_sizes.push(openhtj2k_step_from_normalized_delta(delta));
        }
    }
    let low_energy = if level_count == 0 {
        1.0
    } else {
        let low = OPENHTJ2K_D97_ENERGY_NORMS[level_count - 1].0;
        low * low
    };
    let low_delta = delta_reference / (libm::sqrt(low_energy) * component_gain);
    step_sizes.push(openhtj2k_step_from_normalized_delta(low_delta));
    step_sizes[start..].reverse();
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the OpenHTJ2K Qfactor conversion clamps exponent and mantissa to their marker widths"
)]
fn openhtj2k_step_from_normalized_delta(mut delta: f64) -> QuantStepSize {
    let mut exponent = 0_i32;
    while delta < 1.0 {
        delta *= 2.0;
        exponent += 1;
    }
    let mut mantissa = libm::floor((delta - 1.0) * 2048.0 + 0.5) as i32;
    if mantissa >= 2048 {
        mantissa = 0;
        exponent -= 1;
    }
    if exponent > 31 {
        exponent = 31;
        mantissa = 0;
    } else if exponent < 0 {
        exponent = 0;
        mantissa = 2047;
    }
    QuantStepSize {
        exponent: u16::try_from(exponent).expect("Qfactor exponent was clamped to 0..=31"),
        mantissa: u16::try_from(mantissa).expect("Qfactor mantissa was clamped to 0..=2047"),
    }
}

/// Quantize wavelet coefficients for a single subband.
///
/// For lossless: converts f32 to i32 (round to nearest integer).
/// For lossy: applies scalar deadzone quantization.
///
/// Returns (magnitude, sign) pairs packed as i32 values.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "finite wavelet coefficients are intentionally rounded at the codec quantization boundary"
)]
#[allow(
    clippy::disallowed_macros,
    reason = "the infallible scalar parity API preserves its established Vec-returning contract"
)]
pub(crate) fn quantize_subband(
    coefficients: &[f32],
    step_size: &QuantStepSize,
    range_bits: u8,
    reversible: bool,
) -> Vec<i32> {
    if reversible {
        // No quantization: round to nearest integer
        coefficients.iter().map(|&c| round_f32(c) as i32).collect()
    } else {
        let delta = step_size.delta(range_bits);
        if delta <= 0.0 {
            return vec![0i32; coefficients.len()];
        }
        coefficients
            .iter()
            .map(|&c| {
                // Deadzone quantization: q = sign(c) * floor(|c| / Δ)
                let sign = if c < 0.0 { -1 } else { 1 };
                let magnitude = floor_f32(c.abs() / delta) as i32;
                sign * magnitude
            })
            .collect()
    }
}

/// Fallible counterpart used by the allocation-bounded native encode path.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
#[expect(
    clippy::cast_possible_truncation,
    reason = "finite wavelet coefficients are intentionally rounded at the codec quantization boundary"
)]
pub(crate) fn try_quantize_subband(
    coefficients: &[f32],
    step_size: &QuantStepSize,
    range_bits: u8,
    reversible: bool,
) -> EncodeResult<Vec<i32>> {
    let requested_bytes = coefficients
        .len()
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "quantized subband coefficients",
        })?;
    let mut quantized = Vec::new();
    quantized
        .try_reserve_exact(coefficients.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "quantized subband coefficients",
            bytes: requested_bytes,
        })?;
    if reversible {
        for &coefficient in coefficients {
            quantized.push(round_f32(coefficient) as i32);
        }
        return Ok(quantized);
    }

    let delta = step_size.delta(range_bits);
    if delta <= 0.0 {
        quantized.resize(coefficients.len(), 0);
        return Ok(quantized);
    }
    for &coefficient in coefficients {
        let sign = if coefficient < 0.0 { -1 } else { 1 };
        let magnitude = floor_f32(coefficient.abs() / delta) as i32;
        quantized.push(sign * magnitude);
    }
    Ok(quantized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openhtj2k_qfactor_90_matches_reference_qcd_and_qcc() {
        let expected = [
            [
                0x65df, 0x65b4, 0x65b4, 0x658b, 0x5dc9, 0x5dc9, 0x5dad, 0x560f, 0x560f, 0x5625,
                0x4008, 0x4008, 0x410b, 0x3dab, 0x3dab, 0x337e,
            ],
            [
                0x654f, 0x66db, 0x66db, 0x6764, 0x5027, 0x5027, 0x50d7, 0x49c6, 0x49c6, 0x4b9b,
                0x45c4, 0x45c4, 0x39b0, 0x3396, 0x3396, 0x2a15,
            ],
            [
                0x6745, 0x6788, 0x6788, 0x67e6, 0x5056, 0x5056, 0x50d5, 0x4995, 0x4995, 0x4ae4,
                0x4481, 0x4481, 0x3819, 0x3130, 0x3130, 0x35a3,
            ],
        ];

        for (component, expected) in expected.iter().enumerate() {
            let mut actual = Vec::new();
            append_openhtj2k_qfactor_step_sizes(&mut actual, 8, 5, 90, component)
                .expect("valid OpenHTJ2K Qfactor profile");
            let actual = actual
                .iter()
                .map(|step| (step.exponent << 11) | step.mantissa)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_lossless_quantize() {
        let coeffs = vec![10.0, -5.0, 3.7, -8.2, 0.0];
        let step = QuantStepSize {
            exponent: 12,
            mantissa: 0,
        };
        let result = quantize_subband(&coeffs, &step, 1, true);
        assert_eq!(result, vec![10, -5, 4, -8, 0]);
    }

    #[test]
    fn test_lossy_quantize() {
        let coeffs = vec![10.0, -5.0, 0.3, -0.1];
        let step = QuantStepSize {
            exponent: 8,
            mantissa: 0,
        };
        let delta = step.delta(8);
        assert!((delta - 1.0).abs() < 0.01);

        let result = quantize_subband(&coeffs, &step, 8, false);
        assert_eq!(result[0], 10);
        assert_eq!(result[1], -5);
        assert_eq!(result[2], 0); // Below deadzone
        assert_eq!(result[3], 0); // Below deadzone
    }

    #[test]
    fn irreversible_quantization_uses_direct_division_at_integer_boundary() {
        let coefficient = f32::from_bits(0x41fa_c8ff);
        let step = QuantStepSize {
            exponent: 8,
            mantissa: 23,
        };

        let result = quantize_subband(&[coefficient, -coefficient], &step, 8, false);

        assert_eq!(result, vec![30, -30]);
    }

    #[test]
    fn test_compute_step_sizes_reversible() {
        let steps = compute_step_sizes(8, 3, true, 1);
        // 1 LL + 3 levels × 3 subbands = 10
        assert_eq!(steps.len(), 10);
        // All mantissas should be 0 for reversible
        assert!(steps.iter().all(|s| s.mantissa == 0));
        let exponents: Vec<u16> = steps.iter().map(|s| s.exponent).collect();
        assert_eq!(exponents, vec![8, 9, 9, 10, 9, 9, 10, 9, 9, 10]);
    }

    #[test]
    fn test_compute_step_sizes_irreversible() {
        let steps = compute_step_sizes(8, 3, false, 1);
        assert_eq!(steps.len(), 10);
    }

    #[test]
    fn irreversible_steps_match_decoder_qcd_contract() {
        let steps = compute_step_sizes(8, 1, false, 2);
        let exponents: Vec<u16> = steps.iter().map(|step| step.exponent).collect();
        let mantissas: Vec<u16> = steps.iter().map(|step| step.mantissa).collect();
        assert_eq!(exponents, vec![10, 10, 10, 10]);
        assert_eq!(mantissas, vec![0, 0, 0, 0]);

        let deltas: Vec<f32> = [8u8, 9, 9, 10]
            .iter()
            .zip(&steps)
            .map(|(&range_bits, step)| step.delta(range_bits))
            .collect();
        assert!((deltas[0] - 0.25).abs() < 0.001);
        assert!((deltas[1] - 0.5).abs() < 0.001);
        assert!((deltas[2] - 0.5).abs() < 0.001);
        assert!((deltas[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn irreversible_quantization_scale_coarsens_qcd_deltas() {
        let steps = compute_step_sizes_with_irreversible_scale(8, 1, false, 2, 4.0);
        let exponents: Vec<u16> = steps.iter().map(|step| step.exponent).collect();
        let mantissas: Vec<u16> = steps.iter().map(|step| step.mantissa).collect();
        assert_eq!(exponents, vec![8, 8, 8, 8]);
        assert_eq!(mantissas, vec![0, 0, 0, 0]);

        let deltas: Vec<f32> = [8u8, 9, 9, 10]
            .iter()
            .zip(&steps)
            .map(|(&range_bits, step)| step.delta(range_bits))
            .collect();
        assert!((deltas[0] - 1.0).abs() < 0.001);
        assert!((deltas[1] - 2.0).abs() < 0.001);
        assert!((deltas[2] - 2.0).abs() < 0.001);
        assert!((deltas[3] - 4.0).abs() < 0.001);
    }

    #[test]
    fn irreversible_quantization_scale_uses_mantissa_for_fractional_steps() {
        let steps = compute_step_sizes_with_irreversible_scale(8, 1, false, 2, 5.0);
        let exponents: Vec<u16> = steps.iter().map(|step| step.exponent).collect();
        let mantissas: Vec<u16> = steps.iter().map(|step| step.mantissa).collect();
        assert_eq!(exponents, vec![8, 8, 8, 8]);
        assert_eq!(mantissas, vec![512, 512, 512, 512]);

        let deltas: Vec<f32> = [8u8, 9, 9, 10]
            .iter()
            .zip(&steps)
            .map(|(&range_bits, step)| step.delta(range_bits))
            .collect();
        assert!((deltas[0] - 1.25).abs() < 0.001);
        assert!((deltas[1] - 2.5).abs() < 0.001);
        assert!((deltas[2] - 2.5).abs() < 0.001);
        assert!((deltas[3] - 5.0).abs() < 0.001);
    }

    #[test]
    fn irreversible_subband_scales_change_only_selected_97_steps() {
        let subband_scales = IrreversibleQuantizationSubbandScales {
            low_low: 1.0,
            high_low: 1.0,
            low_high: 1.0,
            high_high: 1.5,
        };

        let default_steps = compute_step_sizes_with_irreversible_profile(
            8,
            1,
            false,
            2,
            1.9,
            IrreversibleQuantizationSubbandScales::default(),
        );
        let shaped_steps =
            compute_step_sizes_with_irreversible_profile(8, 1, false, 2, 1.9, subband_scales);

        assert_eq!(shaped_steps[0], default_steps[0]);
        assert_eq!(shaped_steps[1], default_steps[1]);
        assert_eq!(shaped_steps[2], default_steps[2]);
        assert!(shaped_steps[3].delta(10) > default_steps[3].delta(10));
    }

    #[test]
    fn saturated_irreversible_coefficients_fit_declared_bitplanes() {
        let guard_bits = 2;
        let steps = compute_step_sizes(8, 1, false, guard_bits);
        let range_bits = [8u8, 9, 9, 10];

        for (&range_bits, step) in range_bits.iter().zip(&steps) {
            let quantized = quantize_subband(&[-128.0, 127.0], step, range_bits, false);
            let total_bitplanes = u16::from(guard_bits) + step.exponent - 1;
            let max_abs = quantized
                .iter()
                .map(|coefficient| coefficient.unsigned_abs())
                .max()
                .unwrap();
            assert!(
                max_abs < (1u32 << total_bitplanes),
                "range_bits={range_bits} step={step:?} quantized={quantized:?} total_bitplanes={total_bitplanes}"
            );
        }
    }
}
