//! Forward quantization for JPEG 2000 encoding.
//!
//! - Lossless (reversible 5-3): No quantization, just sign/magnitude conversion
//! - Lossy (irreversible 9-7): Scalar deadzone quantization with step sizes
//!   derived from the DWT subband gain norms.

use alloc::vec;
use alloc::vec::Vec;

use crate::math::{floor_f32, log2_f32, pow2i, round_f32};
use crate::{IrreversibleQuantizationStep, IrreversibleQuantizationSubbandScales, J2kSubBandType};

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
    /// Δ = 2^(R_b - exponent) × (1 + mantissa/2048).
    fn delta(&self, range_bits: u8) -> f32 {
        let rb = range_bits as i32 - self.exponent as i32;
        let base = pow2i(rb);
        base * (1.0 + self.mantissa as f32 / 2048.0)
    }

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
            exponent: u16::try_from(exponent.clamp(0, 31)).expect("clamped exponent fits u16"),
            mantissa: u16::try_from(mantissa.clamp(0, 2047)).expect("clamped mantissa fits u16"),
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
        exponent: bit_depth as u16 + guard_bits as u16,
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
        exponent: u8::try_from(step_size.exponent).expect("step exponent is clamped to u8 range"),
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
pub(crate) fn compute_step_sizes_with_irreversible_profile(
    bit_depth: u8,
    num_decompositions: u8,
    reversible: bool,
    guard_bits: u8,
    irreversible_quantization_scale: f32,
    irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales,
) -> Vec<QuantStepSize> {
    let mut step_sizes = Vec::new();

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
            exponent: bit_depth as u16,
            mantissa: 0,
        });

        for _ in 0..num_decompositions {
            step_sizes.push(QuantStepSize {
                exponent: bit_depth as u16 + 1,
                mantissa: 0,
            });
            step_sizes.push(QuantStepSize {
                exponent: bit_depth as u16 + 1,
                mantissa: 0,
            });
            step_sizes.push(QuantStepSize {
                exponent: bit_depth as u16 + 2,
                mantissa: 0,
            });
        }
    } else {
        // Quality-first irreversible 9-7 default. Use one exponent/mantissa for all
        // subbands and let R_b = bit_depth + log_gain make LL finest and HH
        // coarsest under the decoder's QCD formula.
        let base_step = QuantStepSize {
            exponent: bit_depth as u16 + guard_bits as u16,
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

    step_sizes
}

/// Quantize wavelet coefficients for a single subband.
///
/// For lossless: converts f32 to i32 (round to nearest integer).
/// For lossy: applies scalar deadzone quantization.
///
/// Returns (magnitude, sign) pairs packed as i32 values.
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
        let inv_delta = 1.0 / delta;

        coefficients
            .iter()
            .map(|&c| {
                // Deadzone quantization: q = sign(c) * floor(|c| / Δ)
                let sign = if c < 0.0 { -1 } else { 1 };
                let magnitude = floor_f32(c.abs() * inv_delta) as i32;
                sign * magnitude
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let total_bitplanes = guard_bits as u16 + step.exponent - 1;
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
