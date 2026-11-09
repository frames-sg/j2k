// SPDX-License-Identifier: MIT OR Apache-2.0

//! Independent reference for the canonical binary32 9/7 transcode.
//!
//! Accelerator outputs are compared bit for bit with these helpers. They are
//! a deliberately plain second implementation of
//! [`j2k_codec_math::transcode97`]: per-block [`idct8x8`] into a sample plane,
//! interleaved [`forward_lift97`] over every row and then every column, and
//! [`quantize97`] with the host-rounded reciprocal step.

use j2k_codec_math::transcode97::{
    forward_lift97, idct8x8, quantize97, quantize97_inv_delta, Transcode97Arithmetic,
};
use j2k_transcode::{htj2k97_subband_delta, Dwt97TwoDimensional, Htj2k97CodeBlockOptions};
use j2k_types::{J2kSubBandType, PrequantizedHtj2k97Component};

use crate::{codeblock_component_from_quantized, QuantizedBands};

/// Host binary32 primitives: Rust never contracts `*` and `+`, and
/// `f32::mul_add` rounds once.
struct ReferenceArithmetic;

impl Transcode97Arithmetic for ReferenceArithmetic {
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
}

/// Canonical one-level 9/7 bands of a DCT block grid.
///
/// Each `f64` coefficient is first rounded to `f32`, which is exact for the
/// integer-valued dequantized coefficients a JPEG carries. Band values are
/// returned widened to `f64`, so they compare bit for bit with accelerator
/// outputs that read back binary32 bands.
///
/// # Panics
///
/// Panics if the block grid does not cover `width` x `height`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the canonical transform consumes binary32 coefficients"
)]
pub fn canonical_dwt97_from_blocks(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Dwt97TwoDimensional<f64> {
    assert!(block_cols * 8 >= width && block_rows * 8 >= height);
    assert_eq!(blocks.len(), block_cols * block_rows);
    let mut plane = vec![0.0f32; width * height];
    for (block_index, block) in blocks.iter().enumerate() {
        let (block_x, block_y) = (block_index % block_cols, block_index / block_cols);
        let mut coefficients = [0.0f32; 64];
        for (index, value) in block.iter().flatten().enumerate() {
            coefficients[index] = *value as f32;
        }
        let samples = idct8x8::<ReferenceArithmetic>(&coefficients);
        for local_y in 0..8 {
            for local_x in 0..8 {
                let (x, y) = (block_x * 8 + local_x, block_y * 8 + local_y);
                if x < width && y < height {
                    plane[y * width + x] = samples[local_y * 8 + local_x];
                }
            }
        }
    }
    for row in plane.chunks_mut(width) {
        forward_lift97::<ReferenceArithmetic>(row);
    }
    let mut column = vec![0.0f32; height];
    for x in 0..width {
        for (y, value) in column.iter_mut().enumerate() {
            *value = plane[y * width + x];
        }
        forward_lift97::<ReferenceArithmetic>(&mut column);
        for (y, value) in column.iter().enumerate() {
            plane[y * width + x] = *value;
        }
    }
    let (low_width, high_width) = (width.div_ceil(2), width / 2);
    let (low_height, high_height) = (height.div_ceil(2), height / 2);
    let band = |x_parity: usize, y_parity: usize, band_width: usize, band_height: usize| {
        let mut band = Vec::with_capacity(band_width * band_height);
        for y in 0..band_height {
            for x in 0..band_width {
                band.push(f64::from(
                    plane[(2 * y + y_parity) * width + 2 * x + x_parity],
                ));
            }
        }
        band
    };
    Dwt97TwoDimensional {
        ll: band(0, 0, low_width, low_height),
        hl: band(1, 0, high_width, low_height),
        lh: band(0, 1, low_width, high_height),
        hh: band(1, 1, high_width, high_height),
        low_width,
        low_height,
        high_width,
        high_height,
    }
}

/// Canonically quantize one level of binary32 9/7 bands into the
/// prequantized code-block layout of
/// [`prequantized_component_from_dwt97`](crate::prequantized_component_from_dwt97).
///
/// # Panics
///
/// Panics if a band value is not exactly representable in `f32`, which would
/// mean the bands did not come from the canonical binary32 transform.
#[must_use]
pub fn canonical_prequantized_component_from_dwt97(
    dwt: &Dwt97TwoDimensional<f64>,
    options: Htj2k97CodeBlockOptions,
    x_rsiz: u8,
    y_rsiz: u8,
) -> PrequantizedHtj2k97Component {
    let quantize = |values: &[f64], sub_band_type: J2kSubBandType| {
        let inv_delta = quantize97_inv_delta(htj2k97_subband_delta(options, sub_band_type));
        values
            .iter()
            .map(|&value| quantize97::<ReferenceArithmetic>(exact_f32(value), inv_delta))
            .collect::<Vec<_>>()
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

#[expect(
    clippy::cast_possible_truncation,
    reason = "the narrowing is checked to be exact"
)]
fn exact_f32(value: f64) -> f32 {
    let narrowed = value as f32;
    assert_eq!(
        f64::from(narrowed).to_bits(),
        value.to_bits(),
        "canonical 9/7 band value {value} is not a binary32 value"
    );
    narrowed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(
        clippy::cast_precision_loss,
        reason = "fixture indices are small and exact in f64"
    )]
    fn blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
        (0..block_cols * block_rows)
            .map(|block| {
                let mut values = [[0.0; 8]; 8];
                for (index, value) in values.iter_mut().flatten().enumerate() {
                    *value = ((block * 31 + index * 7) % 61) as f64 - 30.0;
                }
                values
            })
            .collect()
    }

    #[test]
    fn constant_block_grid_lifts_to_a_flat_low_band() {
        let mut flat = vec![[[0.0; 8]; 8]; 4];
        for block in &mut flat {
            block[0][0] = 160.0;
        }
        let dwt = canonical_dwt97_from_blocks(&flat, 2, 2, 13, 11);
        let first = dwt.ll[0];
        assert!(dwt.ll.iter().all(|value| (value - first).abs() < 1e-3));
        assert!(dwt.hh.iter().all(|value| value.abs() < 1e-3));
    }

    #[test]
    fn canonical_quantization_uses_the_shared_layout() {
        let blocks = blocks(3, 2);
        let dwt = canonical_dwt97_from_blocks(&blocks, 3, 2, 21, 13);
        let options = Htj2k97CodeBlockOptions {
            bit_depth: 8,
            guard_bits: 2,
            code_block_width_exp: 1,
            code_block_height_exp: 1,
            irreversible_quantization_scale: 1.0,
            irreversible_quantization_subband_scales:
                j2k_types::IrreversibleQuantizationSubbandScales::default(),
        };
        let canonical = canonical_prequantized_component_from_dwt97(&dwt, options, 1, 1);
        let reference = crate::prequantized_component_from_dwt97(&dwt, options, 1, 1);
        assert_eq!(canonical.resolutions.len(), reference.resolutions.len());
        for (canonical, reference) in canonical.resolutions.iter().zip(&reference.resolutions) {
            for (canonical, reference) in canonical.subbands.iter().zip(&reference.subbands) {
                assert_eq!(canonical.num_cbs_x, reference.num_cbs_x);
                assert_eq!(canonical.num_cbs_y, reference.num_cbs_y);
                assert_eq!(canonical.total_bitplanes, reference.total_bitplanes);
                for (canonical, reference) in
                    canonical.code_blocks.iter().zip(&reference.code_blocks)
                {
                    assert_eq!(canonical.width, reference.width);
                    assert_eq!(canonical.height, reference.height);
                    for (&a, &b) in canonical.coefficients.iter().zip(&reference.coefficients) {
                        assert!((a - b).abs() <= 1, "{a} vs {b}");
                    }
                }
            }
        }
    }
}
