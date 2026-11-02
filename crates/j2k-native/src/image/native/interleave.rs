// SPDX-License-Identifier: MIT OR Apache-2.0

//! Interleaved native samples of several components, packed a block of
//! pixels at a time.

use alloc::vec::Vec;

use crate::j2c::ComponentData;
use crate::math::{dispatch, Level, Simd};

use super::Image;

/// Pixels packed per block, and so per task when the pack is split across
/// threads.
const BLOCK_PIXELS: usize = 16 * 1024;

impl Image<'_> {
    /// Append samples `0..pixel_count` of every component, pixel by pixel,
    /// as packed little-endian native samples of `bit_depth`. Produces the
    /// same bytes as [`Self::push_component_native_sample_bytes`] for every
    /// index and component, but two to four `f32` components of one
    /// signedness and at most 16 bits are converted as
    /// [`Self::extend_component_native_sample_bytes`] converts one, a block
    /// of pixels at a time, the blocks split across threads when `parallel`.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "samples are rounded and clamped to the component's 8- or 16-bit range before packing"
    )]
    pub(crate) fn extend_interleaved_native_sample_bytes(
        out: &mut Vec<u8>,
        components: &[ComponentData],
        pixel_count: usize,
        bit_depth: u8,
        parallel: bool,
    ) {
        let push_each = |out: &mut Vec<u8>| {
            for index in 0..pixel_count {
                for component in components {
                    Self::push_component_native_sample_bytes(out, component, index, bit_depth);
                }
            }
        };
        let Some(planes) = float_planes(components, pixel_count) else {
            push_each(out);
            return;
        };
        let planes = &planes[..components.len()];
        with_native_converter!(
            bit_depth,
            components[0].signed,
            interleave_converted(out, planes, parallel),
            _ => push_each(out),
        );
    }
}

/// The first `pixel_count` samples of each component, when there are two
/// to four `f32` components of one signedness.
fn float_planes(components: &[ComponentData], pixel_count: usize) -> Option<[&[f32]; 4]> {
    let first = components.first()?;
    if !(2..=4).contains(&components.len()) {
        return None;
    }
    let mut planes: [&[f32]; 4] = [&[]; 4];
    for (plane, component) in planes.iter_mut().zip(components) {
        if component.integer_container.is_some() || component.signed != first.signed {
            return None;
        }
        *plane = component.container.truncated().get(..pixel_count)?;
    }
    Some(planes)
}

/// Append `convert(sample)` of every plane's sample, pixel by pixel, to
/// `out`, a block of pixels at a time, the blocks split across threads when
/// `parallel` and the planes are large.
fn interleave_converted<const N: usize>(
    out: &mut Vec<u8>,
    planes: &[&[f32]],
    parallel: bool,
    convert: impl Fn(f32) -> [u8; N] + Send + Sync,
) {
    let pixel_bytes = planes.len() * N;
    let start = out.len();
    // Callers reserve the full output first, so this does not reallocate.
    out.resize(start + planes[0].len() * pixel_bytes, 0);
    let destination = &mut out[start..];
    #[cfg(feature = "parallel")]
    if parallel && planes[0].len() >= super::PARALLEL_PACK_SAMPLES {
        use rayon::prelude::*;
        destination
            .par_chunks_mut(pixel_bytes * BLOCK_PIXELS)
            .enumerate()
            .for_each(|(index, destination)| {
                dispatch!(Level::new(), simd => interleave_block(
                    simd,
                    destination,
                    planes,
                    index * BLOCK_PIXELS,
                    &convert,
                ));
            });
        return;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    dispatch!(Level::new(), simd => {
        for (index, destination) in destination.chunks_mut(pixel_bytes * BLOCK_PIXELS).enumerate() {
            interleave_block(simd, destination, planes, index * BLOCK_PIXELS, &convert);
        }
    });
}

/// Write `convert(sample)` of every plane's samples from `first` on to
/// `destination`, pixel by pixel, at the dispatched SIMD level (see `math`).
#[expect(
    clippy::inline_always,
    reason = "inlined into each SIMD dispatch target"
)]
#[inline(always)]
fn interleave_block<S: Simd, const N: usize>(
    _simd: S,
    destination: &mut [u8],
    planes: &[&[f32]],
    first: usize,
    convert: &impl Fn(f32) -> [u8; N],
) {
    let pixel_bytes = planes.len() * N;
    for (offset, pixel) in destination.chunks_exact_mut(pixel_bytes).enumerate() {
        for (samples, bytes) in planes.iter().zip(pixel.as_chunks_mut::<N>().0) {
            *bytes = convert(samples[first + offset]);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::image::Image;
    use crate::j2c::ComponentData;
    use crate::math::{SimdBuffer, SIMD_WIDTH};

    /// Samples around the bounds and rounding ties of 1 to 16 bits.
    #[expect(clippy::cast_precision_loss, reason = "test values fit f32 exactly")]
    fn samples(count: usize) -> Vec<f32> {
        let mut samples = Vec::from([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0]);
        let mut value = -70_000_i32;
        while samples.len() < count {
            samples.extend([value as f32, value as f32 + 0.5, value as f32 - 0.5]);
            value += 97;
        }
        samples.truncate(count);
        samples
    }

    #[test]
    fn interleaved_pack_matches_the_per_sample_pack() {
        let count = 5000;
        let base = samples(count);
        for components in 2..=5 {
            for bit_depth in [1, 4, 8, 9, 12, 16, 17] {
                for signed in [false, true] {
                    for parallel in [false, true] {
                        let planes: Vec<ComponentData> = (0..components)
                            .map(|component| {
                                let mut samples = base.clone();
                                samples.rotate_left(component * 7);
                                ComponentData {
                                    container: SimdBuffer::<SIMD_WIDTH>::new(samples),
                                    integer_container: None,
                                    bit_depth,
                                    signed,
                                }
                            })
                            .collect();
                        let mut expected = Vec::new();
                        for index in 0..count {
                            for component in &planes {
                                Image::push_component_native_sample_bytes(
                                    &mut expected,
                                    component,
                                    index,
                                    bit_depth,
                                );
                            }
                        }
                        let mut packed = Vec::from([7_u8]);
                        packed.reserve(expected.len());
                        Image::extend_interleaved_native_sample_bytes(
                            &mut packed,
                            &planes,
                            count,
                            bit_depth,
                            parallel,
                        );
                        assert_eq!(
                            packed[1..],
                            expected[..],
                            "{components} x {bit_depth} bits, signed {signed}, parallel {parallel}"
                        );
                        assert_eq!(packed[0], 7);
                    }
                }
            }
        }
    }
}
