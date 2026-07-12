// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use alloc::vec;
use alloc::vec::Vec;

use super::samples::{raw_pixel_bytes_per_sample, read_le_sample_value, sign_extend_sample};
use super::SubBandType;
use crate::{EncodeError, EncodeResult, J2kSubBandType};

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
#[cfg(test)]
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

/// Fallible production deinterleave used by the phase-bounded encode path.
#[expect(
    clippy::cast_precision_loss,
    reason = "the codec float domain intentionally receives bounded integer samples at this rounding boundary"
)]
pub(crate) fn try_deinterleave_to_f32(
    pixels: &[u8],
    num_pixels: usize,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
) -> EncodeResult<Vec<Vec<f32>>> {
    if num_components == 3 && bit_depth == 8 && !signed {
        return try_deinterleave_rgb8_unsigned_to_f32(pixels, num_pixels);
    }

    let component_count = usize::from(num_components);
    let outer_bytes = component_count
        .checked_mul(core::mem::size_of::<Vec<f32>>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "deinterleaved component owners",
        })?;
    let plane_bytes = num_pixels.checked_mul(core::mem::size_of::<f32>()).ok_or(
        EncodeError::ArithmeticOverflow {
            what: "deinterleaved component samples",
        },
    )?;
    let mut components = Vec::new();
    components.try_reserve_exact(component_count).map_err(|_| {
        EncodeError::HostAllocationFailed {
            what: "deinterleaved component owners",
            bytes: outer_bytes,
        }
    })?;
    for _ in 0..component_count {
        let mut component = Vec::new();
        component
            .try_reserve_exact(num_pixels)
            .map_err(|_| EncodeError::HostAllocationFailed {
                what: "deinterleaved component samples",
                bytes: plane_bytes,
            })?;
        component.resize(num_pixels, 0.0);
        components.push(component);
    }

    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };
    let bytes_per_sample =
        raw_pixel_bytes_per_sample(bit_depth).map_err(|what| EncodeError::InvalidInput { what })?;
    for (sample_index, pixel) in pixels
        .chunks_exact(component_count * bytes_per_sample)
        .take(num_pixels)
        .enumerate()
    {
        for (component_index, component) in components.iter_mut().enumerate() {
            let offset = component_index * bytes_per_sample;
            let raw = read_le_sample_value(&pixel[offset..offset + bytes_per_sample], bit_depth);
            component[sample_index] = if signed {
                sign_extend_sample(raw, bit_depth) as f32
            } else {
                raw as f32 - unsigned_offset
            };
        }
    }
    Ok(components)
}

fn try_deinterleave_rgb8_unsigned_to_f32(
    pixels: &[u8],
    num_pixels: usize,
) -> EncodeResult<Vec<Vec<f32>>> {
    let plane_bytes = num_pixels.checked_mul(core::mem::size_of::<f32>()).ok_or(
        EncodeError::ArithmeticOverflow {
            what: "RGB deinterleave samples",
        },
    )?;
    let mut r = try_reserve_f32_plane(num_pixels, plane_bytes)?;
    let mut g = try_reserve_f32_plane(num_pixels, plane_bytes)?;
    let mut b = try_reserve_f32_plane(num_pixels, plane_bytes)?;
    for pixel in pixels.chunks_exact(3).take(num_pixels) {
        r.push(f32::from(pixel[0]) - 128.0);
        g.push(f32::from(pixel[1]) - 128.0);
        b.push(f32::from(pixel[2]) - 128.0);
    }
    let mut components = Vec::new();
    components
        .try_reserve_exact(3)
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "RGB deinterleave component owners",
            bytes: 3 * core::mem::size_of::<Vec<f32>>(),
        })?;
    components.push(r);
    components.push(g);
    components.push(b);
    Ok(components)
}

fn try_reserve_f32_plane(count: usize, bytes: usize) -> EncodeResult<Vec<f32>> {
    let mut plane = Vec::new();
    plane
        .try_reserve_exact(count)
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "RGB deinterleave samples",
            bytes,
        })?;
    Ok(plane)
}

#[cfg(test)]
pub(crate) fn deinterleave_rgb8_unsigned_to_f32(pixels: &[u8], num_pixels: usize) -> Vec<Vec<f32>> {
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

#[cfg(test)]
#[path = "api_helpers/tests.rs"]
mod tests;
