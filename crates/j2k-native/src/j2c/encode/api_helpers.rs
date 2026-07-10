// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::samples::{raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample};
use super::SubBandType;
use crate::J2kSubBandType;

pub(super) fn public_sub_band_type(sub_band_type: SubBandType) -> J2kSubBandType {
    match sub_band_type {
        SubBandType::LowLow => J2kSubBandType::LowLow,
        SubBandType::HighLow => J2kSubBandType::HighLow,
        SubBandType::LowHigh => J2kSubBandType::LowHigh,
        SubBandType::HighHigh => J2kSubBandType::HighHigh,
    }
}

pub(super) fn internal_sub_band_type(sub_band_type: J2kSubBandType) -> SubBandType {
    match sub_band_type {
        J2kSubBandType::LowLow => SubBandType::LowLow,
        J2kSubBandType::HighLow => SubBandType::HighLow,
        J2kSubBandType::LowHigh => SubBandType::LowHigh,
        J2kSubBandType::HighHigh => SubBandType::HighHigh,
    }
}

pub(super) fn default_public_code_block_style() -> crate::J2kCodeBlockStyle {
    crate::J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: false,
        reset_context_probabilities: false,
        termination_on_each_pass: false,
        vertically_causal_context: false,
        segmentation_symbols: false,
    }
}

/// Convert interleaved pixel bytes to per-component f32 arrays.
#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples or metadata at this rounding boundary"
)]
pub(crate) fn deinterleave_to_f32(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> Vec<Vec<f32>> {
    if num_components == 3 && bit_depth == 8 && !signed {
        return deinterleave_rgb8_unsigned_to_f32(pixels, num_pixels);
    }

    let nc = num_components as usize;
    let mut components = vec![vec![0.0f32; num_pixels]; nc];
    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };

    let bytes_per_sample = raw_pixel_bytes_per_sample(bit_depth).unwrap_or(2);
    for (i, pixel) in pixels
        .chunks_exact(nc * bytes_per_sample)
        .take(num_pixels)
        .enumerate()
    {
        for (c, component) in components.iter_mut().enumerate().take(nc) {
            let offset = c * bytes_per_sample;
            let raw = read_le_sample_value(&pixel[offset..offset + bytes_per_sample], bit_depth);
            component[i] = if signed {
                sign_extend_sample(raw, bit_depth) as f32
            } else {
                raw as f32 - unsigned_offset
            };
        }
    }

    components
}

pub(super) fn deinterleave_rgb8_unsigned_to_f32(pixels: &[u8], num_pixels: usize) -> Vec<Vec<f32>> {
    let mut r = Vec::with_capacity(num_pixels);
    let mut g = Vec::with_capacity(num_pixels);
    let mut b = Vec::with_capacity(num_pixels);

    for pixel in pixels.chunks_exact(3).take(num_pixels) {
        r.push(f32::from(pixel[0]) - 128.0);
        g.push(f32::from(pixel[1]) - 128.0);
        b.push(f32::from(pixel[2]) - 128.0);
    }

    vec![r, g, b]
}

/// Calculate the maximum number of decomposition levels for given dimensions.
pub(super) fn max_decomposition_levels(width: u32, height: u32) -> u8 {
    let min_dim = width.min(height);
    if min_dim <= 1 {
        return 0;
    }
    u8::try_from(min_dim.ilog2()).expect("a u32 logarithm fits in u8")
}
