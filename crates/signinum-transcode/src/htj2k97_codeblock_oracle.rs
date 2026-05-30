// SPDX-License-Identifier: Apache-2.0

//! Shared scalar oracle: float 9/7 bands into prequantized HTJ2K code-blocks.
//!
//! This module re-derives the native encoder's irreversible 9/7 scalar
//! quantization and code-block layout (see
//! `signinum-j2k-native/src/j2c/quantize.rs` and the `prepare_subband` /
//! `subband_range_bits` helpers in `.../j2c/encode.rs`) in `f64`, so both GPU
//! backends can compare their fused code-block kernels against one
//! authoritative CPU reference instead of each re-deriving the math.
//!
//! The re-derivation is anchored to native truth by a codestream pin test (see
//! the module tests): encoding the oracle's prequantized output reproduces the
//! native precomputed-DWT codestream byte-for-byte.

use crate::accelerator::Htj2k97CodeBlockOptions;
use crate::dct97_2d::Dwt97TwoDimensional;
use signinum_j2k_native::{
    J2kSubBandType, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

/// Quantize one level of float 9/7 bands into a prequantized HTJ2K component.
///
/// Resolution nesting matches the native encoder for a single decomposition
/// level: resolution 0 holds `[LL]`, resolution 1 holds `[HL, LH, HH]`.
#[must_use]
pub fn prequantized_component_from_dwt97(
    dwt: &Dwt97TwoDimensional<f64>,
    options: Htj2k97CodeBlockOptions,
    x_rsiz: u8,
    y_rsiz: u8,
) -> PrequantizedHtj2k97Component {
    PrequantizedHtj2k97Component {
        x_rsiz,
        y_rsiz,
        resolutions: vec![
            PrequantizedHtj2k97Resolution {
                subbands: vec![quantize_codeblock_subband(
                    &dwt.ll,
                    dwt.low_width,
                    dwt.low_height,
                    J2kSubBandType::LowLow,
                    options,
                )],
            },
            PrequantizedHtj2k97Resolution {
                subbands: vec![
                    quantize_codeblock_subband(
                        &dwt.hl,
                        dwt.high_width,
                        dwt.low_height,
                        J2kSubBandType::HighLow,
                        options,
                    ),
                    quantize_codeblock_subband(
                        &dwt.lh,
                        dwt.low_width,
                        dwt.high_height,
                        J2kSubBandType::LowHigh,
                        options,
                    ),
                    quantize_codeblock_subband(
                        &dwt.hh,
                        dwt.high_width,
                        dwt.high_height,
                        J2kSubBandType::HighHigh,
                        options,
                    ),
                ],
            },
        ],
    }
}

/// Quantize a single float subband and slice it into code-block-major layout.
///
/// Code-blocks are emitted outer `cby`, inner `cbx`; each block's coefficients
/// are row-major, matching the native encoder's `copy_code_block_coefficients`.
#[must_use]
pub fn quantize_codeblock_subband(
    coefficients: &[f64],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> PrequantizedHtj2k97Subband {
    let quantized = quantize_subband_coefficients(coefficients, sub_band_type, options);
    let cb_width = 1usize << (options.code_block_width_exp + 2);
    let cb_height = 1usize << (options.code_block_height_exp + 2);
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let mut code_blocks = Vec::with_capacity(num_cbs_x * num_cbs_y);

    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx * cb_width;
            let y0 = cby * cb_height;
            let block_width = (width - x0).min(cb_width);
            let block_height = (height - y0).min(cb_height);
            let mut block_coefficients = Vec::with_capacity(block_width * block_height);
            for y in 0..block_height {
                let row_start = (y0 + y) * width + x0;
                block_coefficients.extend_from_slice(&quantized[row_start..row_start + block_width]);
            }
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: block_coefficients,
                width: block_width as u32,
                height: block_height as u32,
            });
        }
    }

    PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: num_cbs_x as u32,
        num_cbs_y: num_cbs_y as u32,
        total_bitplanes: htj2k97_subband_total_bitplanes(options, sub_band_type),
        code_blocks,
    }
}

/// Deadzone quantization step size `Δ` for a subband.
///
/// `Δ = 2^(range_bits − exponent) · (1 + mantissa/2048)`, with
/// `range_bits = bit_depth + {LL:0, HL:1, LH:1, HH:2}` and the shared
/// `(exponent, mantissa)` from [`htj2k97_step`].
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
    let (exponent, mantissa) = htj2k97_step(options);
    pow2i_f64(range_bits - i32::from(exponent)) * (1.0 + f64::from(mantissa) / 2048.0)
}

/// Total declared bitplanes for every code-block in a subband.
///
/// `saturating(guard_bits + exponent − 1)`. The exponent is subband-independent
/// for the quality-first single-step quantizer, so the result does not vary by
/// `sub_band_type`; the parameter is kept for call-site symmetry with
/// [`htj2k97_subband_delta`].
#[must_use]
pub fn htj2k97_subband_total_bitplanes(
    options: Htj2k97CodeBlockOptions,
    sub_band_type: J2kSubBandType,
) -> u8 {
    let _ = sub_band_type;
    let (exponent, _) = htj2k97_step(options);
    options.guard_bits.saturating_add(exponent).saturating_sub(1)
}

fn quantize_subband_coefficients(
    coefficients: &[f64],
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Vec<i32> {
    let delta = htj2k97_subband_delta(options, sub_band_type);
    let inv_delta = 1.0 / delta;

    coefficients
        .iter()
        .map(|&coefficient| {
            // Deadzone quantization: q = sign(c) · floor(|c| · (1/Δ)), sign(0) = +1.
            let sign = if coefficient < 0.0 { -1 } else { 1 };
            sign * (coefficient.abs() * inv_delta).floor() as i32
        })
        .collect()
}

/// Shared `(exponent, mantissa)` for the irreversible 9/7 quantizer.
///
/// Mirrors native `QuantStepSize::from_delta` applied to
/// `base_step.delta · scale`, where `base_step.delta = 2^(−guard_bits)`.
fn htj2k97_step(options: Htj2k97CodeBlockOptions) -> (u8, u16) {
    let base_delta =
        pow2i_f64(-i32::from(options.guard_bits)) * f64::from(options.irreversible_quantization_scale);
    let floor_log2 = base_delta.log2().floor() as i32;
    let mut exponent = i32::from(options.bit_depth) - floor_log2;
    let normalized = base_delta / pow2i_f64(floor_log2);
    let mut mantissa = ((normalized - 1.0) * 2048.0).round() as i32;

    if mantissa >= 2048 {
        exponent -= 1;
        mantissa = 0;
    }

    (
        u8::try_from(exponent.clamp(0, 31)).expect("clamped exponent fits u8"),
        u16::try_from(mantissa.clamp(0, 2047)).expect("clamped mantissa fits u16"),
    )
}

fn pow2i_f64(exp: i32) -> f64 {
    if exp >= 0 {
        f64::from(1u32 << exp.cast_unsigned())
    } else {
        1.0 / f64::from(1u32 << (-exp).cast_unsigned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use signinum_j2k_native::{
        encode_precomputed_htj2k_97, encode_prequantized_htj2k_97, EncodeOptions,
        J2kForwardDwt97Level, J2kForwardDwt97Output, PrecomputedHtj2k97Component,
        PrecomputedHtj2k97Image, PrequantizedHtj2k97Image,
    };

    // Boundary-free coefficients on a 0.25 grid: exact in both f32 and f64, and
    // every product with the scale-1.0 inverse deltas (4, 2, 1) lands on an exact
    // integer/half-integer. So the f64 oracle and native's f32 quantizer agree
    // bit-for-bit here and the codestream pin is exact, not merely close.
    fn sample_band(len: usize, offset: f64) -> Vec<f64> {
        (0..len)
            .map(|idx| ((idx % 17) as f64 - 8.0) * 0.5 + offset)
            .collect()
    }

    #[test]
    fn oracle_prequantized_component_matches_native_precomputed_codestream() {
        let width = 17u32;
        let height = 13u32;
        let low_width = width.div_ceil(2) as usize;
        let low_height = height.div_ceil(2) as usize;
        let high_width = (width / 2) as usize;
        let high_height = (height / 2) as usize;

        let ll = sample_band(low_width * low_height, 0.25);
        let hl = sample_band(high_width * low_height, -0.75);
        let lh = sample_band(low_width * high_height, 1.25);
        let hh = sample_band(high_width * high_height, -1.5);

        let options = EncodeOptions {
            num_decomposition_levels: 1,
            reversible: false,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            ..EncodeOptions::default()
        };

        // Native precomputed-DWT path quantizes the f32 bands internally.
        let precomputed_image = PrecomputedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components: vec![PrecomputedHtj2k97Component {
                x_rsiz: 1,
                y_rsiz: 1,
                dwt: J2kForwardDwt97Output {
                    ll: ll.iter().map(|&v| v as f32).collect(),
                    ll_width: low_width as u32,
                    ll_height: low_height as u32,
                    levels: vec![J2kForwardDwt97Level {
                        hl: hl.iter().map(|&v| v as f32).collect(),
                        lh: lh.iter().map(|&v| v as f32).collect(),
                        hh: hh.iter().map(|&v| v as f32).collect(),
                        width,
                        height,
                        low_width: low_width as u32,
                        low_height: low_height as u32,
                        high_width: high_width as u32,
                        high_height: high_height as u32,
                    }],
                },
            }],
        };

        // Oracle prequantized path (f64) over the same bands.
        let dwt = Dwt97TwoDimensional {
            ll,
            hl,
            lh,
            hh,
            low_width,
            low_height,
            high_width,
            high_height,
        };
        let codeblock_options = Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 2,
            code_block_width_exp: 2,
            code_block_height_exp: 2,
            irreversible_quantization_scale: 1.0,
        };
        let component = prequantized_component_from_dwt97(&dwt, codeblock_options, 1, 1);
        let prequantized_image = PrequantizedHtj2k97Image {
            width,
            height,
            bit_depth: 8,
            signed: false,
            components: vec![component],
        };

        let expected = encode_precomputed_htj2k_97(&precomputed_image, &options)
            .expect("native precomputed 9/7 encode");
        let actual = encode_prequantized_htj2k_97(&prequantized_image, &options)
            .expect("oracle prequantized 9/7 encode");

        assert_eq!(
            actual, expected,
            "oracle prequantized component must reproduce the native precomputed-DWT codestream"
        );
    }
}
