// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared HTJ2K 9/7 code-block option validation and quantization metadata.

use crate::{Htj2k97CodeBlockAxis, Htj2k97CodeBlockOptions, Htj2k97CodeBlockOptionsError};
use j2k::J2kSubBandType;
use j2k_native::irreversible_quantization_step_for_subband;

/// Deadzone quantization step size `Δ` for a subband.
///
/// `Δ = 2^(range_bits − exponent) · (1 + mantissa/2048)`, with
/// `range_bits = bit_depth + {LL:0, HL:1, LH:1, HH:2}` and the shared
/// `(exponent, mantissa)` derived by this module's quantizer.
#[must_use]
pub fn htj2k97_subband_delta(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> f64 {
    let log_gain = match sub_band_type {
        J2kSubBandType::LowLow => 0,
        J2kSubBandType::HighLow | J2kSubBandType::LowHigh => 1,
        J2kSubBandType::HighHigh => 2,
    };
    let range_bits = i32::from(options.bit_depth) + log_gain;
    let (exponent, mantissa) = htj2k97_step(options, sub_band_type);
    pow2i_f64(range_bits - i32::from(exponent)) * (1.0 + f64::from(mantissa) / 2048.0)
}

/// Total declared bitplanes for every code-block in a subband.
///
/// `saturating(guard_bits + exponent - 1)`. The exponent is derived from the
/// effective global plus per-subband quantization profile, so callers must pass
/// the actual subband kind.
#[must_use]
pub fn htj2k97_subband_total_bitplanes(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> u8 {
    let (exponent, _) = htj2k97_step(options, sub_band_type);
    options
        .guard_bits
        .saturating_add(exponent)
        .saturating_sub(1)
}

/// Validate 9/7 code-block options against the numeric limits both GPU
/// backends must agree on, returning the decoded `(cb_width, cb_height)`.
///
/// One shared implementation keeps Metal and CUDA from drifting: the same
/// options must be accepted or rejected identically by every backend. Errors
/// use a backend-neutral typed taxonomy so adapters never need to parse prose.
///
/// # Errors
/// Rejects zero/oversized bit depths and guard bits, non-finite or
/// non-positive quantization scales, code-block dimensions beyond the HTJ2K
/// limits (sides ≤ 1024, area ≤ 4096), and subband deltas or total bitplane
/// counts outside the supported range.
pub fn validate_htj2k97_codeblock_options(
    options: Htj2k97CodeBlockOptions,
) -> Result<(usize, usize), Htj2k97CodeBlockOptionsError> {
    if options.bit_depth == 0
        || options.bit_depth > 30
        || options.guard_bits > 30
        || !options.irreversible_quantization_scale.is_finite()
        || options.irreversible_quantization_scale <= 0.0
    {
        return Err(Htj2k97CodeBlockOptionsError::NumericOptionsOutOfRange);
    }
    let subband_scales = options.irreversible_quantization_subband_scales;
    if [
        subband_scales.low_low,
        subband_scales.high_low,
        subband_scales.low_high,
        subband_scales.high_high,
    ]
    .iter()
    .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange);
    }

    let cb_width =
        checked_code_block_dim(options.code_block_width_exp, Htj2k97CodeBlockAxis::Width)?;
    let cb_height =
        checked_code_block_dim(options.code_block_height_exp, Htj2k97CodeBlockAxis::Height)?;
    if cb_width > 1024
        || cb_height > 1024
        || cb_width
            .checked_mul(cb_height)
            .is_none_or(|area| area > 4096)
    {
        return Err(Htj2k97CodeBlockOptionsError::DimensionsExceedLimits {
            width: cb_width,
            height: cb_height,
        });
    }

    for subband in [
        J2kSubBandType::LowLow,
        J2kSubBandType::HighLow,
        J2kSubBandType::LowHigh,
        J2kSubBandType::HighHigh,
    ] {
        let delta = htj2k97_subband_delta(options, subband);
        if !delta.is_finite()
            || delta <= 0.0
            || htj2k97_subband_total_bitplanes(options, subband) > 30
        {
            return Err(Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange);
        }
    }

    Ok((cb_width, cb_height))
}

fn checked_code_block_dim(
    exp_minus_two: u8,
    axis: Htj2k97CodeBlockAxis,
) -> Result<usize, Htj2k97CodeBlockOptionsError> {
    1usize.checked_shl(u32::from(exp_minus_two) + 2).ok_or(
        Htj2k97CodeBlockOptionsError::DimensionExponentUnsupported {
            axis,
            exponent_minus_two: exp_minus_two,
        },
    )
}

/// Shared `(exponent, mantissa)` for the irreversible 9/7 quantizer.
fn htj2k97_step(options: Htj2k97CodeBlockOptions, sub_band_type: J2kSubBandType) -> (u8, u16) {
    let step = irreversible_quantization_step_for_subband(
        options.bit_depth,
        options.guard_bits,
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales,
        sub_band_type,
    );
    (step.exponent, step.mantissa)
}

fn pow2i_f64(exp: i32) -> f64 {
    2.0f64.powi(exp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k::IrreversibleQuantizationSubbandScales;

    #[test]
    fn shared_validator_accepts_standard_options_and_returns_dims() {
        let options = Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 2,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            irreversible_quantization_scale: 1.0,
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
        };
        assert_eq!(validate_htj2k97_codeblock_options(options), Ok((64, 64)));
    }

    #[test]
    fn shared_validator_rejects_out_of_spec_options_on_every_backend() {
        let valid = Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 2,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            irreversible_quantization_scale: 1.0,
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
        };

        // Each case was accepted by the old Metal-only validator.
        let oversized_bit_depth = Htj2k97CodeBlockOptions {
            bit_depth: 31,
            ..valid
        };
        let oversized_guard_bits = Htj2k97CodeBlockOptions {
            guard_bits: 31,
            ..valid
        };
        // 1024x1024: each side passes the per-side cap, area breaks the
        // HTJ2K 4096 limit.
        let oversized_area = Htj2k97CodeBlockOptions {
            code_block_width_exp: 8,
            code_block_height_exp: 8,
            ..valid
        };
        for options in [oversized_bit_depth, oversized_guard_bits, oversized_area] {
            assert!(
                validate_htj2k97_codeblock_options(options).is_err(),
                "options must be rejected: {options:?}"
            );
        }

        // guard_bits == 0 stays accepted (the old Metal validator rejected it,
        // CUDA and the native encoder accept it).
        let zero_guard_bits = Htj2k97CodeBlockOptions {
            guard_bits: 0,
            ..valid
        };
        assert!(validate_htj2k97_codeblock_options(zero_guard_bits).is_ok());
    }

    #[test]
    fn shared_validator_returns_each_typed_failure_variant() {
        let valid = Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 2,
            code_block_width_exp: 4,
            code_block_height_exp: 4,
            irreversible_quantization_scale: 1.0,
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
        };

        let numeric = Htj2k97CodeBlockOptions {
            bit_depth: 31,
            ..valid
        };
        assert_eq!(
            validate_htj2k97_codeblock_options(numeric),
            Err(Htj2k97CodeBlockOptionsError::NumericOptionsOutOfRange)
        );

        let mut quantization = valid;
        quantization
            .irreversible_quantization_subband_scales
            .low_low = 0.0;
        assert_eq!(
            validate_htj2k97_codeblock_options(quantization),
            Err(Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange)
        );

        let exponent = Htj2k97CodeBlockOptions {
            code_block_width_exp: u8::MAX,
            ..valid
        };
        assert_eq!(
            validate_htj2k97_codeblock_options(exponent),
            Err(Htj2k97CodeBlockOptionsError::DimensionExponentUnsupported {
                axis: Htj2k97CodeBlockAxis::Width,
                exponent_minus_two: u8::MAX,
            })
        );

        let dimensions = Htj2k97CodeBlockOptions {
            code_block_width_exp: 8,
            code_block_height_exp: 8,
            ..valid
        };
        assert_eq!(
            validate_htj2k97_codeblock_options(dimensions),
            Err(Htj2k97CodeBlockOptionsError::DimensionsExceedLimits {
                width: 1024,
                height: 1024,
            })
        );
    }

    #[test]
    fn oracle_subband_profile_changes_only_selected_delta_and_bitplanes() {
        let mut options = Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            irreversible_quantization_scale: 1.9,
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
        };
        let high_low_delta = htj2k97_subband_delta(options, J2kSubBandType::HighLow);
        let high_high_delta = htj2k97_subband_delta(options, J2kSubBandType::HighHigh);
        let default_hh_bitplanes =
            htj2k97_subband_total_bitplanes(options, J2kSubBandType::HighHigh);

        options.irreversible_quantization_subband_scales.high_high = 1.5;

        assert_eq!(
            htj2k97_subband_delta(options, J2kSubBandType::HighLow).to_bits(),
            high_low_delta.to_bits()
        );
        assert!(htj2k97_subband_delta(options, J2kSubBandType::HighHigh) > high_high_delta);
        assert_ne!(
            htj2k97_subband_total_bitplanes(options, J2kSubBandType::HighHigh),
            default_hh_bitplanes
        );
    }
}
