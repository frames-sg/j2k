// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native-bit-depth component planes packed from a region decode's
//! streamed rows instead of from region-sized `f32` planes.

use alloc::vec::Vec;

use super::{convert_samples, try_clone_color_space, NativeOutputBudget};
use crate::color::{
    ColorSpace, ComponentPlane, ComponentRowSink, DecodedNativeComponents, NativeComponentPlane,
};
use crate::error::{bail, DecodeError, DecodingError, Result, ValidationError};
use crate::j2c::codestream::ComponentInfo;
use crate::j2c::DecoderContext;
use crate::math::{dispatch, Level, Simd};
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
    /// [`Self::region_decode_streams_rows`]).
    /// The region of the whole image gives
    /// [`Self::decode_native_components_with_context`]'s planes.
    ///
    /// The returned planes are the decode's output: like a caller's buffer,
    /// and like the planes the planes route packs, they are not charged to
    /// the decode budget but to an allowance of their own (see
    /// `NativeOutputBudget::for_output`), checked before the decode, and
    /// each is held to the per-allocation cap. They are allocated when the
    /// first rows arrive, so a region decode that falls back to component
    /// planes meets the decode budget before they exist.
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
        let dimensions = (roi.2, roi.3);
        let mut sink = NativePlanesSink::new(self, dimensions);
        self.decode_native_rows(roi, decoder_context, &mut sink)?;
        let planes = sink.finish()?;
        let color_space = try_clone_color_space(&self.color_space)?;
        Ok(DecodedNativeComponents {
            dimensions,
            color_space,
            has_alpha: self.has_alpha,
            planes,
        })
    }

    /// The region decode of [`Self::decode_native_region_components_rows_with_context`]
    /// into `sink`.
    fn decode_native_rows(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
        sink: &mut NativePlanesSink<'_>,
    ) -> Result<()> {
        validate_roi((self.width(), self.height()), roi)?;
        if self.streamed_rows_native_precision().is_none() || !self.region_decode_streams_rows() {
            bail!(DecodingError::UnsupportedFeature(
                "streamed native component planes need one to four unsubsampled components of one precision up to 24 bits, no image or tile offsets, and no palette, channel definition, sYCC or CIELab"
            ));
        }
        // The planes are allocated when the first rows arrive, but their
        // allowance is known now: an output that cannot exist fails before
        // the decode.
        sink.output_allowance()?;
        self.decode_region_rows_for_integer_output_with_context(roi, decoder_context, sink)
    }
}

/// Packs each rectangle of a region decode into the native planes it
/// covers. The planes are allocated when the first rectangle arrives, once
/// per decode. The first error, an allocation or a rectangle that does not
/// fit the planes, stops every later write and fails the decode.
struct NativePlanesSink<'i> {
    components: &'i [ComponentInfo],
    color_space: &'i ColorSpace,
    dimensions: (u32, u32),
    planes: Vec<NativeComponentPlane>,
    /// One row of samples wider than 16 bits, packed before it is copied.
    scratch: Vec<u8>,
    error: Option<DecodeError>,
    /// Times the planes were allocated.
    #[cfg(test)]
    allocations: usize,
    /// Rectangles handed to the sink.
    #[cfg(test)]
    writes: usize,
}

impl ComponentRowSink for NativePlanesSink<'_> {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        #[cfg(test)]
        {
            self.writes += 1;
        }
        if self.error.is_some() {
            return;
        }
        if self.planes.is_empty() {
            if let Err(error) = self.allocate() {
                self.error = Some(error);
                return;
            }
        }
        if self.try_write_rows(x, y, planes).is_none() {
            self.error = Some(DecodingError::CodeBlockDecodeFailure.into());
        }
    }
}

impl<'i> NativePlanesSink<'i> {
    /// A sink for the `dimensions` of a region of `image`, with no planes
    /// yet.
    fn new(image: &'i Image<'_>, dimensions: (u32, u32)) -> Self {
        Self {
            components: &image.header.component_infos,
            color_space: &image.color_space,
            dimensions,
            planes: Vec::new(),
            scratch: Vec::new(),
            error: None,
            #[cfg(test)]
            allocations: 0,
            #[cfg(test)]
            writes: 0,
        }
    }

    /// The planes held, as the planes route holds its packed planes, to the
    /// output allowance, and each to the per-allocation cap.
    fn output_allowance(&self) -> Result<NativeOutputBudget> {
        let samples = checked_decode_sample_count(self.dimensions.0, self.dimensions.1)?;
        let mut allowance = NativeOutputBudget::for_output();
        allowance.include_elements::<NativeComponentPlane>(self.components.len())?;
        for component in self.components {
            let bytes_per_sample = native_bytes_per_sample(component.size_info.precision)?;
            let bytes = checked_decode_byte_len2(samples, bytes_per_sample)?;
            allowance.include_elements::<u8>(bytes)?;
        }
        allowance.include_color_space_clone(self.color_space)?;
        Ok(allowance)
    }

    /// Allocate every component's plane within the output allowance.
    fn allocate(&mut self) -> Result<()> {
        let samples = checked_decode_sample_count(self.dimensions.0, self.dimensions.1)?;
        let mut allowance = self.output_allowance()?;
        let mut planes = Vec::new();
        try_reserve_decode_elements(&mut planes, self.components.len())?;
        allowance.include_capacity_overage::<NativeComponentPlane>(
            self.components.len(),
            planes.capacity(),
        )?;
        for component in self.components {
            let bit_depth = component.size_info.precision;
            let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
            let bytes = checked_decode_byte_len2(samples, bytes_per_sample)?;
            let mut data = Vec::new();
            try_resize_decode_elements(&mut data, bytes, 0)?;
            allowance.include_capacity_overage::<u8>(bytes, data.capacity())?;
            planes.push(NativeComponentPlane {
                data,
                dimensions: self.dimensions,
                bit_depth,
                signed: component.size_info.signed,
                sampling: (1, 1),
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            });
        }
        self.planes = planes;
        #[cfg(test)]
        {
            self.allocations += 1;
        }
        Ok(())
    }

    /// The packed planes, or the first error a rectangle met. A decode that
    /// handed over no rows has no planes.
    fn finish(self) -> Result<Vec<NativeComponentPlane>> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.planes.is_empty() {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        Ok(self.planes)
    }

    fn try_write_rows(&mut self, x: u32, y: u32, sources: &[ComponentPlane<'_>]) -> Option<()> {
        dispatch!(Level::new(), simd => self.try_write_rows_at(simd, x, y, sources))
    }

    /// [`Self::try_write_rows`] at the dispatched SIMD level (see `math`).
    #[expect(
        clippy::inline_always,
        reason = "inlined into each SIMD dispatch target"
    )]
    #[inline(always)]
    fn try_write_rows_at<S: Simd>(
        &mut self,
        simd: S,
        x: u32,
        y: u32,
        sources: &[ComponentPlane<'_>],
    ) -> Option<()> {
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
                pack_native_samples_at(
                    simd,
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
#[cfg(test)]
pub(super) fn pack_native_samples(
    destination: &mut [u8],
    samples: &[f32],
    bit_depth: u8,
    signed: bool,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    dispatch!(Level::new(), simd => pack_native_samples_at(
        simd,
        destination,
        samples,
        bit_depth,
        signed,
        scratch,
    ))
}

/// [`pack_native_samples`] at the dispatched SIMD level (see `math`).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::inline_always,
    reason = "samples are rounded and clamped to the component's 8- or 16-bit range before packing; inlined into each SIMD dispatch target"
)]
#[inline(always)]
fn pack_native_samples_at<S: Simd>(
    simd: S,
    destination: &mut [u8],
    samples: &[f32],
    bit_depth: u8,
    signed: bool,
    scratch: &mut Vec<u8>,
) -> Result<()> {
    with_native_converter!(
        bit_depth,
        signed,
        convert_samples(simd, destination.as_chunks_mut().0, samples),
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
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::{pack_native_samples, NativePlanesSink};
    use crate::image::Image;
    use crate::j2c::DecoderContext;
    use crate::{
        encode, native_bytes_per_sample, CpuDecodeParallelism, DecodeErrorClass, DecodeSettings,
        EncodeOptions,
    };

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

    fn context<'a>(parallelism: CpuDecodeParallelism) -> DecoderContext<'a> {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(parallelism);
        context
    }

    /// The rows route allocates its planes when the first rows arrive, once
    /// per decode. A region decode that falls back to component planes (here
    /// an RGB image without decomposition levels, which has no Tier-1 bands
    /// to stream) meets the decode budget before any plane exists: past it,
    /// the decode fails with a resource limit and the sink allocated
    /// nothing. Headers only: its SIZ claims 12000 x 12000 samples, whose
    /// `f32` planes pass the budget before any packet is read.
    #[test]
    fn native_rows_allocate_the_planes_when_the_first_rows_arrive() {
        let options = EncodeOptions {
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        let mut bytes = encode(&[0; 16 * 16 * 3], 16, 16, 3, 8, false, &options).expect("encode");
        let siz = bytes
            .windows(2)
            .position(|marker| marker == [0xff, 0x51])
            .expect("SIZ marker");
        for field in [6, 10, 22, 26] {
            bytes[siz + field..siz + field + 4].copy_from_slice(&12_000_u32.to_be_bytes());
        }
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("headers parse");
        assert!(image.region_decode_streams_rows());
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            let mut sink = NativePlanesSink::new(&image, (12_000, 12_000));
            let error = image
                .decode_native_rows((0, 0, 12_000, 12_000), &mut context(parallelism), &mut sink)
                .expect_err("the planes pass the budget");
            assert!(
                matches!(error.classify(), DecodeErrorClass::ResourceLimit { .. }),
                "{parallelism:?}: {error:?}"
            );
            assert_eq!((sink.allocations, sink.writes), (0, 0), "{parallelism:?}");
            assert!(sink.planes.is_empty(), "{parallelism:?}");
        }
    }

    /// A streamed region decode hands its rows over in many rectangles, one
    /// or more per tile, and the sink allocates its planes for the first of
    /// them only. The planes hold the planes route's bytes.
    #[test]
    fn streamed_native_rows_allocate_the_planes_once() {
        let (width, height) = (96_u32, 80_u32);
        let samples: Vec<u8> = (0..width * height * 3)
            .map(|index| (index.wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect();
        let options = EncodeOptions {
            num_decomposition_levels: 2,
            tile_size: Some((32, 32)),
            ..EncodeOptions::default()
        };
        let bytes = encode(&samples, width, height, 3, 8, false, &options).expect("encode");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let roi = (0, 0, width, height);
        for parallelism in [CpuDecodeParallelism::Serial, CpuDecodeParallelism::Auto] {
            let mut sink = NativePlanesSink::new(&image, (width, height));
            image
                .decode_native_rows(roi, &mut context(parallelism), &mut sink)
                .expect("streamed decode");
            assert_eq!(sink.allocations, 1, "{parallelism:?}");
            let writes = sink.writes;
            assert!(writes >= 9, "{parallelism:?}: {writes} rectangles");
            let streamed = sink.finish().expect("planes");
            let expected = image
                .decode_native_region_components_with_context(roi, &mut context(parallelism))
                .expect("planes route");
            assert_eq!(streamed.len(), expected.planes.len(), "{parallelism:?}");
            for (streamed, expected) in streamed.iter().zip(&expected.planes) {
                assert!(streamed.data == expected.data, "{parallelism:?}");
            }
        }
    }
}
