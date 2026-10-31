// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native-bit-depth component planes packed from a region decode's
//! streamed rows instead of from region-sized `f32` planes.

use alloc::vec::Vec;

use super::{
    round_clamped_native, signed_native_bounds, try_clone_color_space, unsigned_native_max,
};
use crate::color::{
    ComponentPlane, ComponentRowSink, DecodedNativeComponents, NativeComponentPlane,
};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::DecoderContext;
use crate::roi::validate_roi;
use crate::{
    checked_decode_byte_len2, checked_decode_sample_count, native_bytes_per_sample,
    try_reserve_decode_elements, try_resize_decode_elements,
};

use super::Image;

impl<'a> Image<'a> {
    /// The precision of every component when a region decode's rows (see
    /// [`Self::decode_region_rows_for_integer_output_with_context`]) carry
    /// the samples the native-bit-depth decodes pack: the image has no
    /// palette, channel definition, sYCC or `CIELab`, which only those
    /// decodes' planes resolve or convert, and its components are at full
    /// sampling with one precision of at most 24 bits. `None` otherwise.
    #[doc(hidden)]
    #[must_use]
    pub fn streamed_rows_native_precision(&self) -> Option<u8> {
        if !self.permits_line_region_idwt() {
            return None;
        }
        let precision = self.header.component_infos.first()?.size_info.precision;
        let uniform = self.header.component_infos.iter().all(|component| {
            component.size_info.precision == precision
                && component.size_info.horizontal_resolution == 1
                && component.size_info.vertical_resolution == 1
        });
        (uniform && (1..=24).contains(&precision)).then_some(precision)
    }

    /// [`Self::decode_native_region_components_with_context`] on the
    /// region decode that hands its rows to a sink (see
    /// [`Self::decode_region_rows_for_integer_output_with_context`]), which
    /// packs them into the returned planes as they are finished: the decode
    /// holds no region-sized `f32` planes. The planes hold the same bytes.
    /// Only for an image with a [`Self::streamed_rows_native_precision`]
    /// whose region decodes stream their rows (see
    /// [`Self::region_decode_streams_rows`]): the planes are allocated
    /// before the decode, so a region decode that fell back to component
    /// planes would hold both, and meet the decode budget only after
    /// allocating them.
    /// The region of the whole image gives
    /// [`Self::decode_native_components_with_context`]'s planes.
    ///
    /// The returned planes are the decode's output: like a caller's buffer
    /// they are not charged to the decode budget, and each is held to the
    /// per-allocation cap.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid, the image has no
    /// streamed rows precision or does not stream its rows, a plane cannot
    /// be allocated, or decoding fails.
    #[doc(hidden)]
    pub fn decode_native_region_components_rows_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        validate_roi((self.width(), self.height()), roi)?;
        if self.streamed_rows_native_precision().is_none() || !self.region_decode_streams_rows() {
            bail!(DecodingError::UnsupportedFeature(
                "streamed native component planes need one to four unsubsampled components of one precision up to 24 bits, no image or tile offsets, and no palette, channel definition, sYCC or CIELab"
            ));
        }
        let dimensions = (roi.2, roi.3);
        let samples = checked_decode_sample_count(dimensions.0, dimensions.1)?;
        let mut planes = Vec::new();
        try_reserve_decode_elements(&mut planes, self.header.component_infos.len())?;
        for component in &self.header.component_infos {
            let bit_depth = component.size_info.precision;
            let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
            let mut data = Vec::new();
            try_resize_decode_elements(
                &mut data,
                checked_decode_byte_len2(samples, bytes_per_sample)?,
                0,
            )?;
            planes.push(NativeComponentPlane {
                data,
                dimensions,
                bit_depth,
                signed: component.size_info.signed,
                sampling: (1, 1),
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            });
        }
        let color_space = try_clone_color_space(&self.color_space)?;
        let mut sink = NativePlanesSink {
            planes: &mut planes,
            scratch: Vec::new(),
            failed: false,
        };
        self.decode_region_rows_for_integer_output_with_context(roi, decoder_context, &mut sink)?;
        if sink.failed {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        Ok(DecodedNativeComponents {
            dimensions,
            color_space,
            has_alpha: self.has_alpha,
            planes,
        })
    }
}

/// Packs each rectangle of a region decode into the native planes it
/// covers. A rectangle that does not fit them fails the decode.
struct NativePlanesSink<'p> {
    planes: &'p mut [NativeComponentPlane],
    /// One row of samples wider than 16 bits, packed before it is copied.
    scratch: Vec<u8>,
    failed: bool,
}

impl ComponentRowSink for NativePlanesSink<'_> {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        if !self.failed && self.try_write_rows(x, y, planes).is_none() {
            self.failed = true;
        }
    }
}

impl NativePlanesSink<'_> {
    fn try_write_rows(&mut self, x: u32, y: u32, sources: &[ComponentPlane<'_>]) -> Option<()> {
        if sources.len() != self.planes.len() {
            return None;
        }
        for (source, target) in sources.iter().zip(self.planes.iter_mut()) {
            if source.bit_depth != target.bit_depth || source.signed != target.signed {
                return None;
            }
            let width = source.dimensions.0 as usize;
            let height = source.dimensions.1 as usize;
            if width == 0 || height == 0 {
                continue;
            }
            let bytes_per_sample = usize::from(target.bytes_per_sample);
            let row_bytes = width.checked_mul(bytes_per_sample)?;
            let stride = (target.dimensions.0 as usize).checked_mul(bytes_per_sample)?;
            let mut start = (y as usize)
                .checked_mul(stride)?
                .checked_add((x as usize).checked_mul(bytes_per_sample)?)?;
            let rows = source.samples.get(..width.checked_mul(height)?)?;
            for samples in rows.chunks_exact(width) {
                let end = start.checked_add(row_bytes)?;
                let destination = target.data.get_mut(start..end)?;
                pack_native_samples(
                    destination,
                    samples,
                    target.bit_depth,
                    target.signed,
                    &mut self.scratch,
                )
                .ok()?;
                start = start.checked_add(stride)?;
            }
        }
        Some(())
    }
}

/// Write `samples` to `destination` as the packed little-endian native
/// samples `Image::extend_component_native_sample_bytes` appends for an
/// `f32` component of `bit_depth`. Samples wider than 16 bits are packed
/// in `scratch` first.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "samples are rounded and clamped to the component's 8- or 16-bit range before packing"
)]
pub(super) fn pack_native_samples(
    destination: &mut [u8],
    samples: &[f32],
    bit_depth: u8,
    signed: bool,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    match (bit_depth, signed) {
        (1..=8, true) => {
            let (min, max) = signed_native_bounds(bit_depth);
            for (destination, &sample) in destination.iter_mut().zip(samples) {
                *destination = (round_clamped_native(sample, min, max) as i8) as u8;
            }
        }
        (1..=8, false) => {
            let max = unsigned_native_max(bit_depth);
            for (destination, &sample) in destination.iter_mut().zip(samples) {
                *destination = round_clamped_native(sample, 0.0, max) as u8;
            }
        }
        (9..=16, true) => {
            let (min, max) = signed_native_bounds(bit_depth);
            for (destination, &sample) in destination.as_chunks_mut::<2>().0.iter_mut().zip(samples)
            {
                *destination =
                    ((round_clamped_native(sample, min, max) as i16) as u16).to_le_bytes();
            }
        }
        (9..=16, false) => {
            let max = unsigned_native_max(bit_depth);
            for (destination, &sample) in destination.as_chunks_mut::<2>().0.iter_mut().zip(samples)
            {
                *destination = (round_clamped_native(sample, 0.0, max) as u16).to_le_bytes();
            }
        }
        _ => {
            scratch.clear();
            try_reserve_decode_elements(scratch, destination.len())?;
            for &sample in samples {
                Image::push_native_sample_bytes(scratch, sample, bit_depth, signed);
            }
            let packed = scratch
                .get(..destination.len())
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            destination.copy_from_slice(packed);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::pack_native_samples;
    use crate::image::Image;
    use crate::native_bytes_per_sample;

    /// Samples around every bound and rounding tie of each precision, plus
    /// values far outside it and NaN.
    #[expect(clippy::cast_precision_loss, reason = "bounds of at most 24 bits")]
    fn samples(bit_depth: u8) -> Vec<f32> {
        let mut samples = Vec::from([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -0.0]);
        for bound in [
            -(1_i64 << (bit_depth - 1)),
            (1_i64 << (bit_depth - 1)) - 1,
            (1_i64 << bit_depth) - 1,
            0,
            1,
        ] {
            for offset in [-1.5, -1.0, -0.5, -0.49, 0.0, 0.49, 0.5, 1.0, 1.5] {
                samples.push(bound as f32 + offset);
            }
        }
        samples.extend([1.0e9, -1.0e9, 0.499_999_97, 2.5, -2.5, 3.5]);
        samples
    }

    #[test]
    fn packed_rows_match_the_planes_pack_at_every_precision() {
        let mut scratch = Vec::new();
        for bit_depth in 1..=24 {
            for signed in [false, true] {
                let samples = samples(bit_depth);
                let mut expected = Vec::new();
                for &sample in &samples {
                    Image::push_native_sample_bytes(&mut expected, sample, bit_depth, signed);
                }
                let bytes = native_bytes_per_sample(bit_depth).expect("bytes per sample");
                let mut packed = alloc::vec![0xa5; samples.len() * bytes];
                pack_native_samples(&mut packed, &samples, bit_depth, signed, &mut scratch)
                    .expect("pack");
                assert_eq!(packed, expected, "{bit_depth} bits, signed {signed}");
            }
        }
    }
}
