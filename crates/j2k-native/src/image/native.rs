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

pub(super) struct FusedGrayLastLevel<'a> {
    pub(super) level: crate::j2c::idwt::LastLevel,
    pub(super) shift: crate::j2c::mct::SampleShift,
    pub(super) ll: &'a [f32],
    pub(super) scratch: [&'a mut [f32]; 2],
}

impl Image<'_> {
    pub(super) fn pack_native_component_planes(
        &self,
        components: &[ComponentData],
        component_owner_capacity: usize,
        dimensions: (u32, u32),
        retained_baseline_bytes: usize,
        parallel: bool,
        mut fused_gray: Option<FusedGrayLastLevel<'_>>,
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
            let fused = (component_idx == 0 && components.len() == 1)
                .then(|| fused_gray.take())
                .flatten();
            let data = Self::pack_native_component_bytes(
                component,
                sample_count,
                capacity,
                parallel,
                &mut budget,
                fused,
            )?;
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

    fn pack_native_component_bytes(
        component: &ComponentData,
        sample_count: usize,
        capacity: usize,
        parallel: bool,
        budget: &mut NativeOutputBudget,
        fused_gray: Option<FusedGrayLastLevel<'_>>,
    ) -> Result<Vec<u8>> {
        let mut data = Vec::new();
        if let Some(fused) = fused_gray {
            crate::try_zero_fill_decode_elements(&mut data, capacity)?;
            budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
            pack_last_level_gray_native(
                fused.level,
                fused.shift,
                component.container.truncated(),
                fused.ll,
                fused.scratch,
                &mut data,
                component.bit_depth,
                component.signed,
                parallel,
            )?;
            return Ok(data);
        }
        #[cfg(feature = "parallel")]
        let zeroed_parallel = parallel
            && sample_count >= PARALLEL_PACK_SAMPLES
            && component.integer_container.is_none()
            && (1..=16).contains(&component.bit_depth)
            && crate::fresh_decode_buffers_come_zeroed();
        #[cfg(not(feature = "parallel"))]
        let zeroed_parallel = false;
        if zeroed_parallel {
            crate::try_zero_fill_decode_elements(&mut data, capacity)?;
        } else {
            try_reserve_decode_elements(&mut data, capacity)?;
        }
        budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
        if zeroed_parallel {
            #[cfg(feature = "parallel")]
            Self::write_component_native_sample_bytes_parallel(
                &mut data,
                component,
                0..sample_count,
                component.bit_depth,
            );
        } else {
            Self::extend_component_native_sample_bytes(
                &mut data,
                component,
                0..sample_count,
                component.bit_depth,
                parallel,
            );
        }
        Ok(data)
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

    /// Write the samples in `range` of one component into a pre-sized native
    /// buffer in parallel chunks.
    #[cfg(feature = "parallel")]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "samples are rounded and clamped to the component's 8- or 16-bit range before packing"
    )]
    fn write_component_native_sample_bytes_parallel(
        out: &mut [u8],
        component: &ComponentData,
        range: core::ops::Range<usize>,
        bit_depth: u8,
    ) {
        let samples = &component.container.truncated()[range];
        with_native_converter!(
            bit_depth,
            component.signed,
            write_converted_parallel(out, samples),
            _ => {},
        );
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

/// Reconstruct a single-component grayscale image's final IDWT level and pack
/// each finished row straight into `output` while it stays in the L1/L2 ring.
#[expect(
    clippy::too_many_arguments,
    reason = "single-component fused last-level reconstruction keeps geometry, buffers, and packer explicit"
)]
fn pack_last_level_gray_native(
    level: crate::j2c::idwt::LastLevel,
    shift: crate::j2c::mct::SampleShift,
    plane: &[f32],
    ll: &[f32],
    scratch: [&mut [f32]; 2],
    output: &mut [u8],
    bit_depth: u8,
    signed: bool,
    parallel: bool,
) -> Result<()> {
    match (bit_depth, signed) {
        (bit_depth @ 1..=8, true) => {
            let (min, max) = crate::math::signed_native_bounds(bit_depth);
            crate::j2c::idwt::pack_last_level_gray(
                level,
                plane,
                ll,
                scratch,
                output,
                1,
                parallel,
                &NativeGrayU8Emitter { shift, min, max },
            )
        }
        (bit_depth @ 1..=8, false) => {
            let max = crate::math::unsigned_native_max(bit_depth);
            crate::j2c::idwt::pack_last_level_gray(
                level,
                plane,
                ll,
                scratch,
                output,
                1,
                parallel,
                &NativeGrayU8Emitter {
                    shift,
                    min: 0.0,
                    max,
                },
            )
        }
        (bit_depth @ 9..=16, true) => {
            let (min, max) = crate::math::signed_native_bounds(bit_depth);
            crate::j2c::idwt::pack_last_level_gray(
                level,
                plane,
                ll,
                scratch,
                output,
                2,
                parallel,
                &NativeGrayU16Emitter { shift, min, max },
            )
        }
        (bit_depth @ 9..=16, false) => {
            let max = crate::math::unsigned_native_max(bit_depth);
            crate::j2c::idwt::pack_last_level_gray(
                level,
                plane,
                ll,
                scratch,
                output,
                2,
                parallel,
                &NativeGrayU16Emitter {
                    shift,
                    min: 0.0,
                    max,
                },
            )
        }
        _ => Err(DecodingError::CodeBlockDecodeFailure.into()),
    }
}

struct NativeGrayU8Emitter {
    shift: crate::j2c::mct::SampleShift,
    min: f32,
    max: f32,
}

impl crate::j2c::idwt::GrayRowEmitter for NativeGrayU8Emitter {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "samples are clamped to 8-bit bounds and inlined into the dispatched SIMD target"
    )]
    #[inline(always)]
    fn emit<S: Simd>(&self, _simd: S, row: &mut [f32], out_row: &mut [u8]) {
        let (min, max) = (self.min, self.max);
        match self.shift {
            crate::j2c::mct::SampleShift::Copy => {
                for (dst, &sample) in out_row.iter_mut().zip(row.iter()) {
                    *dst = crate::math::round_clamped_native(sample, min, max) as u8;
                }
            }
            crate::j2c::mct::SampleShift::Add(addend) => {
                for (dst, &sample) in out_row.iter_mut().zip(row.iter()) {
                    *dst = crate::math::round_clamped_native(sample + addend, min, max) as u8;
                }
            }
            crate::j2c::mct::SampleShift::RoundThenAdd(addend) => {
                for (dst, &sample) in out_row.iter_mut().zip(row.iter()) {
                    *dst = crate::math::round_clamped_native(
                        crate::math::round_ties_even_then_add(sample, addend),
                        min,
                        max,
                    ) as u8;
                }
            }
        }
    }
}

struct NativeGrayU16Emitter {
    shift: crate::j2c::mct::SampleShift,
    min: f32,
    max: f32,
}

impl crate::j2c::idwt::GrayRowEmitter for NativeGrayU16Emitter {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::inline_always,
        reason = "samples are clamped to 16-bit bounds and inlined into the dispatched SIMD target"
    )]
    #[inline(always)]
    fn emit<S: Simd>(&self, _simd: S, row: &mut [f32], out_row: &mut [u8]) {
        let (min, max) = (self.min, self.max);
        let dst = out_row.as_chunks_mut::<2>().0;
        match self.shift {
            crate::j2c::mct::SampleShift::Copy => {
                for (dst, &sample) in dst.iter_mut().zip(row.iter()) {
                    *dst =
                        (crate::math::round_clamped_native(sample, min, max) as u16).to_le_bytes();
                }
            }
            crate::j2c::mct::SampleShift::Add(addend) => {
                for (dst, &sample) in dst.iter_mut().zip(row.iter()) {
                    *dst = (crate::math::round_clamped_native(sample + addend, min, max) as u16)
                        .to_le_bytes();
                }
            }
            crate::j2c::mct::SampleShift::RoundThenAdd(addend) => {
                for (dst, &sample) in dst.iter_mut().zip(row.iter()) {
                    *dst = (crate::math::round_clamped_native(
                        crate::math::round_ties_even_then_add(sample, addend),
                        min,
                        max,
                    ) as u16)
                        .to_le_bytes();
                }
            }
        }
    }
}

/// Write `convert(sample)` for every sample into `out` in parallel chunks.
#[cfg(feature = "parallel")]
fn write_converted_parallel<const N: usize>(
    out: &mut [u8],
    samples: &[f32],
    convert: impl Fn(f32) -> [u8; N] + Send + Sync,
) {
    use rayon::prelude::*;
    out.as_chunks_mut::<N>()
        .0
        .par_chunks_mut(16 * 1024)
        .zip(samples.par_chunks(16 * 1024))
        .for_each(|(destination, source)| {
            dispatch!(Level::new(), simd => convert_samples(simd, destination, source, &convert));
        });
}

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
        // Callers reserve the full output first, so this does not reallocate.
        let start = out.len();
        out.resize(start + samples.len() * N, 0);
        write_converted_parallel(&mut out[start..], samples, convert);
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

    #[test]
    fn fused_gray_last_level_native_decode_matches_separate_idwt() {
        use crate::{DecodeSettings, DecoderContext, EncodeOptions};

        let (width, height) = (131_u32, 97_u32);
        let n = (width * height) as usize;
        for (bit_depth, signed, reversible) in [
            (8_u8, false, true),
            (8, true, true),
            (12, false, true),
            (12, true, false),
            (16, false, false),
            (16, true, true),
        ] {
            let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };
            let mut pixels = Vec::with_capacity(n * bytes_per_sample);
            for i in 0..n {
                let raw = ((i * 197 + 13) as u16) & (u16::MAX >> (16 - bit_depth));
                if bytes_per_sample == 1 {
                    pixels.push(raw as u8);
                } else {
                    pixels.extend_from_slice(&raw.to_le_bytes());
                }
            }
            let opts = EncodeOptions {
                reversible,
                num_decomposition_levels: 3,
                ..EncodeOptions::default()
            };
            let encoded = crate::encode_htj2k(&pixels, width, height, 1, bit_depth, signed, &opts)
                .expect("encode HTJ2K grayscale");
            let image = Image::new(&encoded, &DecodeSettings::default()).expect("parse image");
            let mut fused_ctx = DecoderContext::default();
            let fused = image
                .decode_native_components_with_context(&mut fused_ctx)
                .expect("fused native decode");
            let mut separate_ctx = DecoderContext::default();
            separate_ctx.tile_decode_context.separate_last_idwt_level = true;
            let separate = image
                .decode_native_components_with_context(&mut separate_ctx)
                .expect("separate native decode");
            assert_eq!(
                fused.planes[0].data, separate.planes[0].data,
                "fused vs separate mismatch for b={bit_depth} signed={signed} rev={reversible}"
            );
            if reversible {
                assert_eq!(
                    fused.planes[0].data, pixels,
                    "lossless round-trip mismatch for b={bit_depth} signed={signed}"
                );
            }
        }
    }
}
