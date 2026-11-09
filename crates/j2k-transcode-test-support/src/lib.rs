// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dev-only oracle helpers for transcode adapter parity tests.

#![forbid(unsafe_code)]

mod canonical97;
pub mod corpus_validation;
pub mod dct53_1d;
pub mod dct53_multilevel;
mod dwt_diff;
mod move_only;
pub mod transcode97_matrix;

pub use canonical97::{canonical_dwt97_from_blocks, canonical_prequantized_component_from_dwt97};

use j2k_transcode::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions,
};
#[cfg(test)]
use j2k_types::encode_geometry::encode_dwt_level_dimensions_for_input;
use j2k_types::{
    J2kSubBandType, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

pub use dwt_diff::{max_abs_diff_53, max_abs_diff_97};

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
    let quantize = |values: &[f64], sub_band_type| {
        quantize_subband_coefficients(values, sub_band_type, options)
    };
    codeblock_component_from_quantized(
        &QuantizedBands {
            ll: quantize(&dwt.ll, J2kSubBandType::LowLow),
            hl: quantize(&dwt.hl, J2kSubBandType::HighLow),
            lh: quantize(&dwt.lh, J2kSubBandType::LowHigh),
            hh: quantize(&dwt.hh, J2kSubBandType::HighHigh),
        },
        dwt,
        options,
        x_rsiz,
        y_rsiz,
    )
}

/// Quantized coefficients of one decomposition level, row-major per band.
struct QuantizedBands {
    ll: Vec<i32>,
    hl: Vec<i32>,
    lh: Vec<i32>,
    hh: Vec<i32>,
}

/// Nest quantized single-level bands into resolutions: resolution 0 holds
/// `[LL]`, resolution 1 holds `[HL, LH, HH]`.
fn codeblock_component_from_quantized(
    bands: &QuantizedBands,
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
                subbands: vec![slice_codeblock_subband(
                    &bands.ll,
                    dwt.low_width,
                    dwt.low_height,
                    J2kSubBandType::LowLow,
                    options,
                )],
            },
            PrequantizedHtj2k97Resolution {
                subbands: vec![
                    slice_codeblock_subband(
                        &bands.hl,
                        dwt.high_width,
                        dwt.low_height,
                        J2kSubBandType::HighLow,
                        options,
                    ),
                    slice_codeblock_subband(
                        &bands.lh,
                        dwt.low_width,
                        dwt.high_height,
                        J2kSubBandType::LowHigh,
                        options,
                    ),
                    slice_codeblock_subband(
                        &bands.hh,
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
///
/// # Panics
///
/// Panics if a code-block dimension or count does not fit its codestream `u32`
/// field.
#[must_use]
pub fn quantize_codeblock_subband(
    coefficients: &[f64],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> PrequantizedHtj2k97Subband {
    let quantized = quantize_subband_coefficients(coefficients, sub_band_type, options);
    slice_codeblock_subband(&quantized, width, height, sub_band_type, options)
}

fn slice_codeblock_subband(
    quantized: &[i32],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> PrequantizedHtj2k97Subband {
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp);
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp);
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
                block_coefficients
                    .extend_from_slice(&quantized[row_start..row_start + block_width]);
            }
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: block_coefficients,
                width: u32::try_from(block_width)
                    .expect("code-block width must fit the codestream u32 field"),
                height: u32::try_from(block_height)
                    .expect("code-block height must fit the codestream u32 field"),
            });
        }
    }

    PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: u32::try_from(num_cbs_x)
            .expect("horizontal code-block count must fit the codestream u32 field"),
        num_cbs_y: u32::try_from(num_cbs_y)
            .expect("vertical code-block count must fit the codestream u32 field"),
        total_bitplanes: htj2k97_subband_total_bitplanes(options, sub_band_type),
        code_blocks,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the float-to-integer cast intentionally applies Rust's saturating quantizer semantics"
)]
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
            let sign = if coefficient < 0.0 { -1 } else { 1 };
            sign * (coefficient.abs() * inv_delta).floor() as i32
        })
        .collect()
}

fn htj2k97_code_block_dim(exp_minus_two: u8) -> usize {
    1usize
        .checked_shl(u32::from(exp_minus_two) + 2)
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_native::{
        encode_precomputed_htj2k_97, encode_prequantized_htj2k_97, EncodeOptions,
        J2kForwardDwt97Level, J2kForwardDwt97Output, PrecomputedHtj2k97Component,
        PrecomputedHtj2k97Image,
    };
    use j2k_transcode::Dwt97TwoDimensional;
    use j2k_types::{IrreversibleQuantizationSubbandScales, PrequantizedHtj2k97Image};

    #[expect(
        clippy::cast_precision_loss,
        reason = "the synthetic pattern index is reduced modulo 17 before conversion"
    )]
    fn sample_band(len: usize, offset: f64) -> Vec<f64> {
        (0..len)
            .map(|idx| ((idx % 17) as f64 - 8.0) * 0.5 + offset)
            .collect()
    }

    #[test]
    fn prequantized_oracle_matches_native_precomputed_codestream() {
        let width = 17u32;
        let height = 13u32;
        let level = encode_dwt_level_dimensions_for_input(width, height);
        let low_width = usize::try_from(level.low_width).expect("fixture width fits usize");
        let low_height = usize::try_from(level.low_height).expect("fixture height fits usize");
        let high_width = usize::try_from(level.high_width).expect("fixture width fits usize");
        let high_height = usize::try_from(level.high_height).expect("fixture height fits usize");

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

        #[expect(
            clippy::cast_possible_truncation,
            reason = "the bounded synthetic fixture values are intentionally narrowed to the f32 native oracle ABI"
        )]
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
                    ll_width: u32::try_from(low_width).expect("fixture width fits u32"),
                    ll_height: u32::try_from(low_height).expect("fixture height fits u32"),
                    levels: vec![J2kForwardDwt97Level {
                        hl: hl.iter().map(|&v| v as f32).collect(),
                        lh: lh.iter().map(|&v| v as f32).collect(),
                        hh: hh.iter().map(|&v| v as f32).collect(),
                        width,
                        height,
                        low_width: u32::try_from(low_width).expect("fixture width fits u32"),
                        low_height: u32::try_from(low_height).expect("fixture height fits u32"),
                        high_width: u32::try_from(high_width).expect("fixture width fits u32"),
                        high_height: u32::try_from(high_height).expect("fixture height fits u32"),
                    }],
                },
            }],
        };

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
            irreversible_quantization_subband_scales:
                IrreversibleQuantizationSubbandScales::default(),
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
            "prequantized oracle must reproduce the native precomputed-DWT codestream"
        );
    }
}
