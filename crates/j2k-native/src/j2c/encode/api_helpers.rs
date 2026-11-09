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
/// `parallel` lets the planes of a large image fill on the Rayon pool.
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
    parallel: bool,
) -> EncodeResult<Vec<Vec<f32>>> {
    if num_components == 3 && bit_depth == 8 && !signed {
        return try_deinterleave_rgb8_unsigned_to_f32(pixels, num_pixels, parallel);
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
        components.push(component);
    }

    let unsigned_offset = if signed {
        0.0
    } else {
        (1_u64 << (u32::from(bit_depth) - 1)) as f32
    };
    let bytes_per_sample =
        raw_pixel_bytes_per_sample(bit_depth).map_err(|what| EncodeError::InvalidInput { what })?;
    let sample = move |bytes: &[u8]| {
        let raw = read_le_sample_value(bytes, bit_depth);
        if signed {
            sign_extend_sample(raw, bit_depth) as f32
        } else {
            raw as f32 - unsigned_offset
        }
    };
    let pixel_bytes = component_count * bytes_per_sample;
    if pixel_bytes == 0 {
        return Err(EncodeError::InvalidInput {
            what: "deinterleave needs at least one component",
        });
    }
    let whole_pixels = (pixels.len() / pixel_bytes).min(num_pixels);
    let pixels = &pixels[..whole_pixels * pixel_bytes];
    if !fill_planes_on_pool(pixels, &mut components, bytes_per_sample, parallel, sample) {
        fill_planes_serially(pixels, &mut components, bytes_per_sample, sample);
    }
    // A short pixel buffer leaves the rest of each plane zero.
    for component in &mut components {
        component.resize(num_pixels, 0.0);
    }
    Ok(components)
}

/// Fill each plane on this thread. Samples of one to four bytes take loops
/// specialized to their width; the function stays out of line, so its loops
/// compile the same for every caller (inlined, their speed moved by up to
/// half between builds).
#[inline(never)]
fn fill_planes_serially(
    pixels: &[u8],
    planes: &mut [Vec<f32>],
    bytes_per_sample: usize,
    sample: impl Fn(&[u8]) -> f32,
) {
    match bytes_per_sample {
        1 => fill_planes_of::<1>(pixels, planes, sample),
        2 => fill_planes_of::<2>(pixels, planes, sample),
        3 => fill_planes_of::<3>(pixels, planes, sample),
        4 => fill_planes_of::<4>(pixels, planes, sample),
        _ => {
            let pixel_bytes = planes.len() * bytes_per_sample;
            for (index, plane) in planes.iter_mut().enumerate() {
                let offset = index * bytes_per_sample;
                plane.extend(
                    pixels
                        .chunks_exact(pixel_bytes)
                        .map(|pixel| sample(&pixel[offset..offset + bytes_per_sample])),
                );
            }
        }
    }
}

/// Fill each plane from samples of `BYTES` bytes, one plane at a time.
fn fill_planes_of<const BYTES: usize>(
    pixels: &[u8],
    planes: &mut [Vec<f32>],
    sample: impl Fn(&[u8]) -> f32,
) {
    let pixel_bytes = planes.len() * BYTES;
    for (index, plane) in planes.iter_mut().enumerate() {
        let offset = index * BYTES;
        plane.extend(
            pixels
                .chunks_exact(pixel_bytes)
                .map(|pixel| sample(&pixel[offset..offset + BYTES])),
        );
    }
}

/// Images of fewer samples fill their planes on one thread: the floor of the
/// 8-bit RGB path, counted in samples.
#[cfg(feature = "parallel")]
const PARALLEL_DEINTERLEAVE_SAMPLES: usize = 3 * PARALLEL_DEINTERLEAVE_PIXELS;

/// Fill each plane on the Rayon pool when that is allowed and worth it: one
/// thread converting every sample of a large image dominates its encode.
#[cfg(feature = "parallel")]
fn fill_planes_on_pool(
    pixels: &[u8],
    planes: &mut [Vec<f32>],
    bytes_per_sample: usize,
    parallel: bool,
    sample: impl Fn(&[u8]) -> f32 + Sync,
) -> bool {
    use rayon::prelude::*;
    let pixel_bytes = planes.len() * bytes_per_sample;
    if !parallel
        || pixel_bytes == 0
        || pixels.len() / bytes_per_sample < PARALLEL_DEINTERLEAVE_SAMPLES
        || rayon::current_num_threads() < 2
    {
        return false;
    }
    for (index, plane) in planes.iter_mut().enumerate() {
        let offset = index * bytes_per_sample;
        plane.par_extend(
            pixels
                .par_chunks_exact(pixel_bytes)
                .with_min_len(1 << 14)
                .map(|pixel| sample(&pixel[offset..offset + bytes_per_sample])),
        );
    }
    true
}

#[cfg(not(feature = "parallel"))]
fn fill_planes_on_pool(
    _: &[u8],
    _: &mut [Vec<f32>],
    _: usize,
    _: bool,
    _: impl Fn(&[u8]) -> f32,
) -> bool {
    false
}

fn try_deinterleave_rgb8_unsigned_to_f32(
    pixels: &[u8],
    num_pixels: usize,
    parallel: bool,
) -> EncodeResult<Vec<Vec<f32>>> {
    let plane_bytes = num_pixels.checked_mul(core::mem::size_of::<f32>()).ok_or(
        EncodeError::ArithmeticOverflow {
            what: "RGB deinterleave samples",
        },
    )?;
    let mut r = try_reserve_f32_plane(num_pixels, plane_bytes)?;
    let mut g = try_reserve_f32_plane(num_pixels, plane_bytes)?;
    let mut b = try_reserve_f32_plane(num_pixels, plane_bytes)?;
    let pixels = pixels.as_chunks::<3>().0;
    let pixels = &pixels[..num_pixels.min(pixels.len())];
    if !fill_rgb_planes_on_pool(pixels, [&mut r, &mut g, &mut b], parallel) {
        for pixel in pixels {
            r.push(f32::from(pixel[0]) - 128.0);
            g.push(f32::from(pixel[1]) - 128.0);
            b.push(f32::from(pixel[2]) - 128.0);
        }
    }
    rgb_components(r, g, b)
}

fn rgb_components(r: Vec<f32>, g: Vec<f32>, b: Vec<f32>) -> EncodeResult<Vec<Vec<f32>>> {
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

/// Smaller images fill their planes on one thread.
#[cfg(feature = "parallel")]
const PARALLEL_DEINTERLEAVE_PIXELS: usize = 1 << 16;

/// Fill each plane on the Rayon pool when that is allowed and worth it; one
/// thread is faster filling all three in one pass.
#[cfg(feature = "parallel")]
fn fill_rgb_planes_on_pool(pixels: &[[u8; 3]], planes: [&mut Vec<f32>; 3], parallel: bool) -> bool {
    use rayon::prelude::*;
    if !parallel || pixels.len() < PARALLEL_DEINTERLEAVE_PIXELS || rayon::current_num_threads() < 2
    {
        return false;
    }
    for (channel, plane) in planes.into_iter().enumerate() {
        let sample = move |pixel: &[u8; 3]| f32::from(pixel[channel]) - 128.0;
        plane.par_extend(pixels.par_iter().with_min_len(1 << 14).map(sample));
    }
    true
}

#[cfg(not(feature = "parallel"))]
fn fill_rgb_planes_on_pool(_: &[[u8; 3]], _: [&mut Vec<f32>; 3], _: bool) -> bool {
    false
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

    for pixel in pixels.as_chunks::<3>().0.iter().take(num_pixels) {
        r.push(f32::from(pixel[0]) - 128.0);
        g.push(f32::from(pixel[1]) - 128.0);
        b.push(f32::from(pixel[2]) - 128.0);
    }

    vec![r, g, b]
}

#[cfg(test)]
#[path = "api_helpers/tests.rs"]
mod tests;
