// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dev-only oracle helpers for transcode adapter parity tests.

#![forbid(unsafe_code)]

use j2k_transcode::{
    htj2k97_subband_delta, htj2k97_subband_total_bitplanes, Dwt97TwoDimensional,
    Htj2k97CodeBlockOptions,
};
use j2k_types::{
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

    fn sample_band(len: usize, offset: f64) -> Vec<f64> {
        (0..len)
            .map(|idx| ((idx % 17) as f64 - 8.0) * 0.5 + offset)
            .collect()
    }

    #[test]
    fn prequantized_oracle_matches_native_precomputed_codestream() {
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
        let actual =
            encode_prequantized_htj2k_97(&native_prequantized_image(prequantized_image), &options)
                .expect("oracle prequantized 9/7 encode");

        assert_eq!(
            actual, expected,
            "prequantized oracle must reproduce the native precomputed-DWT codestream"
        );
    }

    fn native_prequantized_image(
        image: PrequantizedHtj2k97Image,
    ) -> j2k_native::PrequantizedHtj2k97Image {
        j2k_native::PrequantizedHtj2k97Image {
            width: image.width,
            height: image.height,
            bit_depth: image.bit_depth,
            signed: image.signed,
            components: image
                .components
                .into_iter()
                .map(|component| j2k_native::PrequantizedHtj2k97Component {
                    x_rsiz: component.x_rsiz,
                    y_rsiz: component.y_rsiz,
                    resolutions: component
                        .resolutions
                        .into_iter()
                        .map(|resolution| j2k_native::PrequantizedHtj2k97Resolution {
                            subbands: resolution
                                .subbands
                                .into_iter()
                                .map(|subband| j2k_native::PrequantizedHtj2k97Subband {
                                    sub_band_type: subband.sub_band_type,
                                    num_cbs_x: subband.num_cbs_x,
                                    num_cbs_y: subband.num_cbs_y,
                                    total_bitplanes: subband.total_bitplanes,
                                    code_blocks: subband
                                        .code_blocks
                                        .into_iter()
                                        .map(|block| j2k_native::PrequantizedHtj2k97CodeBlock {
                                            coefficients: block.coefficients,
                                            width: block.width,
                                            height: block.height,
                                        })
                                        .collect(),
                                })
                                .collect(),
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}
