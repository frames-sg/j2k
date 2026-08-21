// SPDX-License-Identifier: MIT OR Apache-2.0

//! Irreversible quantization values and per-subband job contracts.

/// Subband quantization job.
#[derive(Debug, Clone, Copy)]
pub struct J2kQuantizeSubbandJob<'a> {
    /// Source subband coefficients in row-major order.
    pub coefficients: &'a [f32],
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this subband.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
}

/// Multipliers applied to irreversible 9/7 quantization step sizes by subband.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrreversibleQuantizationSubbandScales {
    /// Multiplier for the LL subband.
    pub low_low: f32,
    /// Multiplier for HL subbands.
    pub high_low: f32,
    /// Multiplier for LH subbands.
    pub low_high: f32,
    /// Multiplier for HH subbands.
    pub high_high: f32,
}

impl Default for IrreversibleQuantizationSubbandScales {
    fn default() -> Self {
        Self {
            low_low: 1.0,
            high_low: 1.0,
            low_high: 1.0,
            high_high: 1.0,
        }
    }
}

/// Public JPEG 2000 irreversible quantization step-size tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrreversibleQuantizationStep {
    /// Quantization step-size exponent.
    pub exponent: u8,
    /// Quantization step-size mantissa.
    pub mantissa: u16,
}
