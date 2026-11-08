// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::color::{
    native_component_plane_dimensions, DecodedNativeComponents, NativeComponentPlane,
};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::ComponentData;
use crate::math::{dispatch, Level, Simd};
use crate::{checked_decode_byte_len2, native_bytes_per_sample, try_reserve_decode_elements};

use super::Image;

/// Calls `$function($args..., convert)` with `convert` packing one `f32`
/// sample of `$bit_depth` bits, 1 to 16, and `$signed` into its
/// little-endian native bytes: clamped to the component's range, rounded
/// half away from zero, `[u8; 1]` up to 8 bits and `[u8; 2]` above. Other
/// depths evaluate `$wider`. Every native packer converts through this one
/// match; a macro because the bytes' width differs between arms.
macro_rules! with_native_converter {
    (
        $bit_depth:expr,
        $signed:expr,
        $function:ident($($arg:expr),* $(,)?),
        _ => $wider:expr $(,)?
    ) => {
        match ($bit_depth, $signed) {
            (bit_depth @ 1..=8, true) => {
                let (min, max) = $crate::math::signed_native_bounds(bit_depth);
                $function(
                    $($arg,)*
                    #[inline(always)]
                    move |sample: f32| {
                        [$crate::math::round_clamped_native(sample, min, max) as u8]
                    },
                )
            }
            (bit_depth @ 1..=8, false) => {
                let max = $crate::math::unsigned_native_max(bit_depth);
                $function(
                    $($arg,)*
                    #[inline(always)]
                    move |sample: f32| {
                        [$crate::math::round_clamped_native(sample, 0.0, max) as u8]
                    },
                )
            }
            (bit_depth @ 9..=16, true) => {
                let (min, max) = $crate::math::signed_native_bounds(bit_depth);
                $function(
                    $($arg,)*
                    #[inline(always)]
                    move |sample: f32| {
                        ($crate::math::round_clamped_native(sample, min, max) as u16).to_le_bytes()
                    },
                )
            }
            (bit_depth @ 9..=16, false) => {
                let max = $crate::math::unsigned_native_max(bit_depth);
                $function(
                    $($arg,)*
                    #[inline(always)]
                    move |sample: f32| {
                        ($crate::math::round_clamped_native(sample, 0.0, max) as u16).to_le_bytes()
                    },
                )
            }
            _ => $wider,
        }
    };
}

mod allocation;
mod interleave;
mod rows;
pub(super) use allocation::{try_clone_color_space, NativeOutputBudget};

impl Image<'_> {
    pub(super) fn pack_native_component_planes(
        &self,
        components: &[ComponentData],
        component_owner_capacity: usize,
        dimensions: (u32, u32),
        retained_baseline_bytes: usize,
        parallel: bool,
    ) -> Result<DecodedNativeComponents> {
        // The decode budget holds the `f32` planes the pack reads; the packed
        // planes are the output and have their own allowance.
        NativeOutputBudget::for_decoded_channels(
            retained_baseline_bytes,
            components,
            component_owner_capacity,
        )?;
        let mut budget = NativeOutputBudget::for_output();
        budget.include_elements::<NativeComponentPlane>(components.len())?;
        for (component_idx, component) in components.iter().enumerate() {
            let sampling = self.component_plane_sampling_at(component_idx);
            let bytes_per_sample = native_bytes_per_sample(component.bit_depth)?;
            let sample_count = component
                .integer_container
                .as_ref()
                .map_or(component.container.truncated().len(), Vec::len);
            let capacity = checked_decode_byte_len2(sample_count, bytes_per_sample)?;
            native_component_plane_dimensions(dimensions, sampling, sample_count)?;
            budget.include_elements::<u8>(capacity)?;
        }
        budget.include_color_space_clone(&self.color_space)?;

        let color_space = try_clone_color_space(&self.color_space)?;
        budget.include_color_space_clone_overage(&self.color_space, &color_space)?;
        let mut planes = Vec::new();
        try_reserve_decode_elements(&mut planes, components.len())?;
        budget.include_capacity_overage::<NativeComponentPlane>(
            components.len(),
            planes.capacity(),
        )?;
        for (component_idx, component) in components.iter().enumerate() {
            let sampling = self.component_plane_sampling_at(component_idx);
            let bytes_per_sample = native_bytes_per_sample(component.bit_depth)?;
            let sample_count = component
                .integer_container
                .as_ref()
                .map_or(component.container.truncated().len(), Vec::len);
            let plane_dimensions =
                native_component_plane_dimensions(dimensions, sampling, sample_count)?;
            let capacity = checked_decode_byte_len2(sample_count, bytes_per_sample)?;
            let mut data = Vec::new();
            try_reserve_decode_elements(&mut data, capacity)?;
            budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
            Self::extend_component_native_sample_bytes(
                &mut data,
                component,
                0..sample_count,
                component.bit_depth,
                parallel,
            );
            if data.len() != capacity {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            planes.push(NativeComponentPlane {
                data,
                dimensions: plane_dimensions,
                bit_depth: component.bit_depth,
                signed: component.signed,
                sampling,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            });
        }

        let packed = DecodedNativeComponents {
            dimensions,
            color_space,
            has_alpha: self.has_alpha,
            planes,
        };
        NativeOutputBudget::validate_component_pack(&packed)?;
        Ok(packed)
    }

    /// Validate bounded bitmap reconstruction using effective tile styles.
    /// `retained_capacity` counts additional live owners beside this image and
    /// its retained planning baseline, such as reusable decoder scratch.
    ///
    /// # Errors
    /// Returns unsupported exact-integer sampled-region reconstruction, cancellation,
    /// tile metadata or aggregate allocation errors.
    #[doc(hidden)]
    pub fn validate_region_bitmap_layout_with_operation_control(
        &self,
        retained_capacity: usize,
        control: &crate::DecodeOperationControl,
    ) -> Result<()> {
        control.check()?;
        let retained = super::allocation::combine_retained_bytes(
            self.retained_planning_bytes()?,
            retained_capacity,
        )?;
        let baseline = super::allocation::combine_retained_bytes(control.baseline(retained)?, 0)?;
        crate::j2c::validate_region_bitmap_layout(self.codestream, &self.header, baseline, control)
    }

    /// Append the samples in `range` of one component as packed little-endian
    /// native samples. Produces the same bytes as calling
    /// [`Self::push_component_native_sample_bytes`] for every index, but packs
    /// eight- and sixteen-bit float planes in chunks the compiler vectorizes.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "samples are rounded and clamped to the component's 8- or 16-bit range before packing"
    )]
    pub(crate) fn extend_component_native_sample_bytes(
        out: &mut Vec<u8>,
        component: &ComponentData,
        range: core::ops::Range<usize>,
        bit_depth: u8,
        parallel: bool,
    ) {
        if let Some(samples) = component.integer_container.as_ref() {
            for &sample in &samples[range] {
                Self::push_native_i64_sample_bytes(out, sample, bit_depth, component.signed);
            }
            return;
        }
        let samples = &component.container.truncated()[range];
        with_native_converter!(
            bit_depth,
            component.signed,
            extend_converted(out, samples, parallel),
            _ => {
                for &sample in samples {
                    Self::push_native_sample_bytes(out, sample, bit_depth, component.signed);
                }
            },
        );
    }

    pub(crate) fn push_component_native_sample_bytes(
        out: &mut Vec<u8>,
        component: &ComponentData,
        index: usize,
        bit_depth: u8,
    ) {
        if let Some(samples) = component.integer_container.as_ref() {
            Self::push_native_i64_sample_bytes(out, samples[index], bit_depth, component.signed);
        } else {
            Self::push_native_sample_bytes(
                out,
                component.container.truncated()[index],
                bit_depth,
                component.signed,
            );
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "samples are clamped to the declared signed or unsigned component range before packing"
    )]
    fn push_native_i64_sample_bytes(out: &mut Vec<u8>, sample: i64, bit_depth: u8, signed: bool) {
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let clamped = sample.clamp(min, max);
            if bit_depth <= 8 {
                out.push((clamped as i8) as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as i16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        } else {
            let max = (1u64 << u32::from(bit_depth)) - 1;
            let clamped = if sample <= 0 {
                0
            } else {
                (sample as u64).min(max)
            };
            if bit_depth <= 8 {
                out.push(clamped as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as u16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "rounded samples are range-checked before stable native-width packing"
    )]
    pub(crate) fn push_native_sample_bytes(
        out: &mut Vec<u8>,
        sample: f32,
        bit_depth: u8,
        signed: bool,
    ) {
        let sample = f64::from(sample);
        if signed {
            let magnitude_bits = u32::from(bit_depth.saturating_sub(1));
            let min = -(1_i64 << magnitude_bits);
            let max = (1_i64 << magnitude_bits) - 1;
            let clamped = if sample.is_nan() {
                0
            } else if sample <= min as f64 {
                min
            } else if sample >= max as f64 {
                max
            } else if sample >= 0.0 {
                (sample + 0.5) as i64
            } else {
                (sample - 0.5) as i64
            };
            if bit_depth <= 8 {
                out.push((clamped as i8) as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as i16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        } else {
            let max = (1u64 << u32::from(bit_depth)) - 1;
            let clamped = if sample.is_nan() || sample <= 0.0 {
                0
            } else if sample >= max as f64 {
                max
            } else {
                (sample + 0.5) as u64
            };
            if bit_depth <= 8 {
                out.push(clamped as u8);
            } else if bit_depth <= 16 {
                out.extend_from_slice(&(clamped as u16).to_le_bytes());
            } else {
                let bytes = clamped.to_le_bytes();
                let byte_count = native_bytes_per_sample(bit_depth).unwrap_or(8);
                out.extend_from_slice(&bytes[..byte_count]);
            }
        }
    }
}

/// Planes with at least this many samples pack in parallel. Unit tests lower
/// it so the parallel packer runs on small fixtures.
#[cfg(all(feature = "parallel", not(test)))]
const PARALLEL_PACK_SAMPLES: usize = 1 << 20;
#[cfg(all(feature = "parallel", test))]
const PARALLEL_PACK_SAMPLES: usize = 1024;

/// Append `convert(sample)` for every sample, in chunks the compiler
/// vectorizes, splitting large planes across threads when `parallel`.
fn extend_converted<const N: usize>(
    out: &mut Vec<u8>,
    samples: &[f32],
    parallel: bool,
    convert: impl Fn(f32) -> [u8; N] + Send + Sync,
) {
    const CHUNK: usize = 64;
    #[cfg(feature = "parallel")]
    if parallel && samples.len() >= PARALLEL_PACK_SAMPLES {
        use rayon::prelude::*;
        // Callers reserve the full output first, so this does not reallocate.
        let start = out.len();
        out.resize(start + samples.len() * N, 0);
        out[start..]
            .as_chunks_mut::<N>()
            .0
            .par_chunks_mut(16 * 1024)
            .zip(samples.par_chunks(16 * 1024))
            .for_each(|(destination, source)| {
                dispatch!(Level::new(), simd => convert_samples(simd, destination, source, &convert));
            });
        return;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;

    dispatch!(Level::new(), simd => {
        let mut staged = [[0_u8; N]; CHUNK];
        for chunk in samples.chunks(CHUNK) {
            let staged = &mut staged[..chunk.len()];
            convert_samples(simd, staged, chunk, &convert);
            out.extend_from_slice(staged.as_flattened());
        }
    });
}

/// Write `convert(sample)` of every `source` sample to `destination`, at the
/// dispatched SIMD level (see `math`), where its rounding also vectorizes.
#[expect(
    clippy::inline_always,
    reason = "inlined into each SIMD dispatch target"
)]
#[inline(always)]
fn convert_samples<S: Simd, const N: usize>(
    _simd: S,
    destination: &mut [[u8; N]],
    source: &[f32],
    convert: impl Fn(f32) -> [u8; N],
) {
    for (destination, &sample) in destination.iter_mut().zip(source) {
        *destination = convert(sample);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::Image;

    #[test]
    fn reversible_integer_packing_keeps_exact_clamping_semantics() {
        let mut out = Vec::new();
        for sample in [-129, -1, 127, 128] {
            Image::push_native_i64_sample_bytes(&mut out, sample, 8, true);
        }
        assert_eq!(out, [128, 255, 127, 127]);

        out.clear();
        for sample in [-1, 0, 255, 256] {
            Image::push_native_i64_sample_bytes(&mut out, sample, 8, false);
        }
        assert_eq!(out, [0, 0, 255, 255]);
    }
}
