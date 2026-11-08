// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native, coefficient, and borrowed-component decode output surfaces.

use alloc::vec::Vec;

use crate::color::{
    ComponentPlane, ComponentRowSink, DecodedComponents, DecodedNativeComponents, RawBitmap,
};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::{self, ComponentData, DecoderContext, Reversible53CoefficientImage};
use crate::jp2::DecodedImage;
use crate::CpuDecodeParallelism;
use crate::{
    checked_decode_byte_len3, checked_decode_byte_len4, checked_decode_sample_count,
    native_bytes_per_sample, try_reserve_decode_elements, validate_roi, HtCodeBlockDecoder,
};

use super::native::{try_clone_color_space, NativeOutputBudget};
use super::Image;

impl<'a> Image<'a> {
    /// Extract selected native wavelet bands without inverse DWT or color conversion.
    ///
    /// See [`crate::WaveletSelection`] for selection and coefficient semantics.
    ///
    /// # Errors
    /// Returns an error for invalid selectors or unsupported coding contracts.
    pub fn decode_wavelets_with_context(
        &self,
        context: &mut DecoderContext<'a>,
        selection: &crate::WaveletSelection,
    ) -> Result<crate::WaveletImage> {
        self.with_packet_cache_retry(|| {
            j2c::wavelets::extract(
                self.codestream,
                &self.header,
                self.retained_metadata_bytes()?,
                context,
                selection,
            )
        })
    }

    /// Decode the image at native bit depth without scaling to 8-bit.
    ///
    /// For images with bit depth ≤ 8, returns pixel data as `Vec<u8>`.
    /// For images with bit depth > 8 (e.g., 12-bit or 16-bit), returns
    /// pixel data as little-endian `u16` values packed into `Vec<u8>`.
    ///
    /// This is essential for medical imaging (DICOM) where 12-bit and 16-bit
    /// images must preserve their full dynamic range.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding or native sample packing fails.
    pub fn decode_native(&self) -> Result<RawBitmap> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_with_context(&mut decoder_context)
    }

    /// Decode at native bit depth while accounting an already-live external
    /// allocation, such as the encoded `Vec` being round-trip validated.
    ///
    /// `retained_capacity` must be the allocator capacity of that external
    /// owner, not merely its logical length.
    ///
    /// # Errors
    ///
    /// Returns an error when aggregate retained allocation accounting,
    /// decoding, sizing, or native sample packing fails.
    #[doc(hidden)]
    pub fn decode_native_with_retained_capacity(
        &self,
        retained_capacity: usize,
    ) -> Result<RawBitmap> {
        let retained_baseline_bytes = super::allocation::combine_retained_bytes(
            retained_capacity,
            self.retained_header_metadata_bytes()?,
        )?;
        let mut decoder_context = DecoderContext::default();
        self.decode_native_with_context_and_retained_baseline(
            &mut decoder_context,
            retained_baseline_bytes,
        )
    }

    /// Extract reversible 5/3 wavelet coefficients for coefficient-domain
    /// classic JPEG 2000 to HTJ2K recoding.
    ///
    /// This decodes classic Tier-1 code-blocks into dequantized reversible
    /// wavelet coefficients, but does not run inverse DWT or color conversion.
    #[doc(hidden)]
    pub fn decode_reversible_53_coefficients(&self) -> Result<Reversible53CoefficientImage> {
        let mut decoder_context = DecoderContext::default();
        self.decode_reversible_53_coefficients_with_context(&mut decoder_context)
    }

    /// Extract reversible 5/3 wavelet coefficients using a caller-provided
    /// decoder context.
    #[doc(hidden)]
    pub fn decode_reversible_53_coefficients_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<Reversible53CoefficientImage> {
        self.with_packet_cache_retry(|| {
            j2c::recode::extract_reversible_53_coefficients(
                self.codestream,
                &self.header,
                self.retained_metadata_bytes()?,
                decoder_context,
            )
        })
    }

    /// Decode a region of the image at native bit depth.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid or decoding and packing fail.
    pub fn decode_native_region(&self, roi: (u32, u32, u32, u32)) -> Result<RawBitmap> {
        self.decode_native_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid or decoding and packing fail.
    pub fn decode_native_region_components(
        &self,
        roi: (u32, u32, u32, u32),
    ) -> Result<DecodedNativeComponents> {
        self.decode_native_region_components_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode the image at native bit depth using a caller-provided decoder
    /// context so allocations can be reused across repeated decodes.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding, sizing, or native sample packing fails.
    pub fn decode_native_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        let retained_baseline_bytes = self.retained_header_metadata_bytes()?;
        self.decode_native_with_context_and_retained_baseline(
            decoder_context,
            retained_baseline_bytes,
        )
    }

    pub(crate) fn decode_native_with_context_and_retained_baseline(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        retained_baseline_bytes: usize,
    ) -> Result<RawBitmap> {
        self.with_packet_cache_retry(|| {
            let bit_depth = self.uniform_header_bit_depth()?;
            let parallel = decoder_context.cpu_decode_parallelism() == CpuDecodeParallelism::Auto;
            let retained_baseline_bytes =
                self.run_decode(decoder_context, None, None, true, retained_baseline_bytes)?;
            self.drop_ignored_components(&mut decoder_context.tile_decode_context.channel_data);

            let components = &decoder_context.tile_decode_context.channel_data;
            let component_owner_capacity = components.capacity();
            let num_components =
                u16::try_from(components.len()).map_err(|_| ValidationError::TooManyChannels)?;
            let width = self.width();
            let height = self.height();
            let pixel_count = checked_decode_sample_count(width, height)?;
            let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
            let capacity = checked_decode_byte_len3(
                pixel_count,
                usize::from(num_components),
                bytes_per_sample,
            )?;
            let mut budget = NativeOutputBudget::for_decoded_channels(
                retained_baseline_bytes,
                components,
                component_owner_capacity,
            )?;
            budget.include_bit_capacity(components.len())?;
            budget.include_elements::<u8>(capacity)?;
            let component_signed = Self::try_component_signedness(components)?;
            budget.include_bit_capacity_overage(components.len(), component_signed.capacity())?;
            let signed = component_signed.iter().all(|signed| *signed);
            let mut data = Vec::new();
            try_reserve_decode_elements(&mut data, capacity)?;
            budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
            if let [component] = components.as_slice() {
                Self::extend_component_native_sample_bytes(
                    &mut data,
                    component,
                    0..pixel_count,
                    bit_depth,
                    parallel,
                );
            } else {
                Self::extend_interleaved_native_sample_bytes(
                    &mut data,
                    components,
                    pixel_count,
                    bit_depth,
                    parallel,
                );
            }
            if data.len() != capacity {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            let bitmap = RawBitmap {
                data,
                width,
                height,
                bit_depth,
                signed,
                component_signed,
                num_components,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            };
            NativeOutputBudget::validate_raw_pack(
                retained_baseline_bytes,
                components,
                component_owner_capacity,
                &bitmap,
            )?;
            Ok(bitmap)
        })
    }

    /// Decode a region of the image at native bit depth using a caller-provided
    /// decoder context.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid or decoding, sizing, and packing fail.
    pub fn decode_native_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        self.with_packet_cache_retry(|| {
            validate_roi((self.width(), self.height()), roi)?;
            if self.requires_exact_integer_decode() {
                return self.decode_native_region_via_full_decode(roi, decoder_context);
            }
            let bit_depth = self.uniform_header_bit_depth()?;
            let retained_image_bytes = self.retained_header_metadata_bytes()?;
            let parallel = decoder_context.cpu_decode_parallelism() == CpuDecodeParallelism::Auto;
            let retained_image_bytes =
                self.run_decode(decoder_context, Some(roi), None, true, retained_image_bytes)?;
            self.drop_ignored_components(&mut decoder_context.tile_decode_context.channel_data);

            let components = &decoder_context.tile_decode_context.channel_data;
            let component_owner_capacity = components.capacity();
            let num_components =
                u16::try_from(components.len()).map_err(|_| ValidationError::TooManyChannels)?;
            let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
            let (_x, _y, width, height) = roi;
            let capacity = checked_decode_byte_len4(
                width as usize,
                height as usize,
                usize::from(num_components),
                bytes_per_sample,
            )?;
            let mut budget = NativeOutputBudget::for_decoded_channels(
                retained_image_bytes,
                components,
                component_owner_capacity,
            )?;
            budget.include_bit_capacity(components.len())?;
            budget.include_elements::<u8>(capacity)?;
            let mut data = Vec::new();
            let component_signed = Self::try_component_signedness(components)?;
            budget.include_bit_capacity_overage(components.len(), component_signed.capacity())?;
            try_reserve_decode_elements(&mut data, capacity)?;
            budget.include_capacity_overage::<u8>(capacity, data.capacity())?;
            let signed = component_signed.iter().all(|signed| *signed);

            let pixel_count = width as usize * height as usize;
            if let [component] = components.as_slice() {
                Self::extend_component_native_sample_bytes(
                    &mut data,
                    component,
                    0..pixel_count,
                    bit_depth,
                    parallel,
                );
            } else {
                Self::extend_interleaved_native_sample_bytes(
                    &mut data,
                    components,
                    pixel_count,
                    bit_depth,
                    parallel,
                );
            }
            if data.len() != capacity {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }

            let bitmap = RawBitmap {
                data,
                width,
                height,
                bit_depth,
                signed,
                component_signed,
                num_components,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            };
            NativeOutputBudget::validate_raw_pack(
                retained_image_bytes,
                components,
                component_owner_capacity,
                &bitmap,
            )?;
            Ok(bitmap)
        })
    }

    fn try_component_signedness(components: &[ComponentData]) -> Result<Vec<bool>> {
        let mut signedness = Vec::new();
        try_reserve_decode_elements(&mut signedness, components.len())?;
        signedness.extend(components.iter().map(|component| component.signed));
        Ok(signedness)
    }

    pub(super) fn component_plane_sampling_at(&self, component_idx: usize) -> (u8, u8) {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return (1, 1);
        }
        self.header
            .component_infos
            .get(component_idx)
            .map_or((1, 1), |component| {
                (
                    component.size_info.horizontal_resolution,
                    component.size_info.vertical_resolution,
                )
            })
    }

    pub(super) fn try_borrow_component_planes_with_retained_baseline<'ctx>(
        &self,
        components: &'ctx [ComponentData],
        component_owner_capacity: usize,
        dimensions: (u32, u32),
        retained_image_bytes: usize,
    ) -> Result<DecodedComponents<'ctx>> {
        self.with_packet_cache_retry(|| {
            #[cfg(feature = "std")]
            let retained_image_bytes = super::allocation::combine_retained_bytes(
                retained_image_bytes,
                self.packet_cache.retained_bytes(),
            )?;
            let mut budget = NativeOutputBudget::for_decoded_channels(
                retained_image_bytes,
                components,
                component_owner_capacity,
            )?;
            budget.include_elements::<ComponentPlane<'_>>(components.len())?;
            budget.include_color_space_clone(&self.color_space)?;
            let color_space = try_clone_color_space(&self.color_space)?;
            budget.include_color_space_clone_overage(&self.color_space, &color_space)?;
            let mut planes = Vec::new();
            try_reserve_decode_elements(&mut planes, components.len())?;
            budget.include_capacity_overage::<ComponentPlane<'_>>(
                components.len(),
                planes.capacity(),
            )?;
            for (component_idx, component) in components.iter().enumerate() {
                planes.push(ComponentPlane {
                    samples: component.container.truncated(),
                    dimensions,
                    bit_depth: component.bit_depth,
                    signed: component.signed,
                    sampling: self.component_plane_sampling_at(component_idx),
                });
            }

            let mut packed = DecodedComponents {
                dimensions,
                color_space,
                has_alpha: self.has_alpha,
                planes,
                live_bytes: 0,
            };
            packed.live_bytes = NativeOutputBudget::validate_borrowed_pack(
                retained_image_bytes,
                components,
                component_owner_capacity,
                &packed,
            )?;
            Ok(packed)
        })
    }

    fn uniform_header_bit_depth(&self) -> Result<u8> {
        let Some(first) = self.header.component_infos.first() else {
            bail!(DecodingError::CodeBlockDecodeFailure);
        };
        if self
            .header
            .component_infos
            .iter()
            .any(|component| component.size_info.precision != first.size_info.precision)
        {
            bail!(DecodingError::UnsupportedFeature(
                "decode_native requires uniform component bit depths; use decode_components for mixed-depth images"
            ));
        }
        if first.size_info.precision > 38 {
            bail!(DecodingError::UnsupportedFeature(
                "decode_native supports JPEG 2000 Part 1 component precision up to 38 bits"
            ));
        }
        Ok(first.size_info.precision)
    }

    pub(super) fn validate_component_plane_precision(&self) -> Result<()> {
        if self
            .header
            .component_infos
            .iter()
            .any(|component| component.size_info.precision > 24)
        {
            bail!(DecodingError::UnsupportedFeature(
                "decode_components currently supports component planes up to 24 bits per component"
            ));
        }
        Ok(())
    }
}

/// Borrowed component planes rounded for integer output.
///
/// The plain component-plane methods return unrounded irreversible (9/7)
/// samples with the unsigned level shift already added. Rounding those planes
/// afterwards differs from [`Image::decode`] on exact ties, because adding the
/// shift costs one bit of f32 precision. These methods round each centered
/// sample ties-to-even before the shift, as the interleaved decode paths do, so
/// converting their planes to integer samples reproduces that output.
/// Reversible (5/3) samples are unchanged.
impl<'a> Image<'a> {
    /// Rounded-for-integer-output counterpart of
    /// [`Self::decode_components_with_context`].
    ///
    /// # Errors
    ///
    /// Returns an error when component precision is unsupported or decoding fails.
    #[doc(hidden)]
    pub fn decode_components_for_integer_output_with_context<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        self.decode_borrowed_component_planes(decoder_context, None, None, true)
    }

    /// Rounded-for-integer-output counterpart of
    /// [`Self::decode_region_components_with_context`].
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid, precision is unsupported, or decoding fails.
    #[doc(hidden)]
    pub fn decode_region_components_for_integer_output_with_context<'ctx>(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        self.decode_borrowed_component_planes(decoder_context, Some(roi), None, true)
    }

    /// [`Self::decode_region_components_for_integer_output_with_context`]
    /// that hands the region's samples to `sink` instead of returning
    /// component planes. A plain region decode streams them as they are
    /// finished and never holds the whole planes, its Tier-1 bands decoded
    /// serially or on the worker pool; any other decode hands over its planes
    /// as one rectangle at the end.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid, precision is unsupported,
    /// or decoding fails. A decode that fails may have handed `sink` some of
    /// the region already.
    #[doc(hidden)]
    pub fn decode_region_rows_for_integer_output_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
        sink: &mut dyn ComponentRowSink,
    ) -> Result<()> {
        validate_roi((self.width(), self.height()), roi)?;
        self.validate_component_plane_precision()?;
        let mut kept_sink = KeptComponentsSink {
            sink: &mut *sink,
            ignored_components: self.ignored_components,
        };
        let charged_image_bytes = self.run_decode_rows(
            decoder_context,
            Some(roi),
            None,
            true,
            self.retained_header_metadata_bytes()?,
            Some(&mut kept_sink),
        )?;
        if decoder_context.tile_decode_context.stream_rows {
            return Ok(());
        }
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = self.finish_decoded_image(
            &mut decoder_context.tile_decode_context.channel_data,
            charged_image_bytes,
        )?;
        let components = self.try_borrow_component_planes_with_retained_baseline(
            decoded_components.as_slice(),
            decoded_components.capacity(),
            (roi.2, roi.3),
            self.retained_header_metadata_bytes()?,
        )?;
        sink.write_rows(0, 0, &components.planes);
        Ok(())
    }

    /// Whether [`Self::decode_region_rows_for_integer_output_with_context`]
    /// can stream this image's rows, as far as its main header and boxes
    /// say: the image has no palette, channel definition, sYCC or `CIELab`
    /// (see `permits_line_region_idwt`), its components are at full sampling
    /// with at most 24 bits, and the image and its tiles have no offsets and
    /// one to four components (see `j2c::image_streams_rows`, which the
    /// decode checks before it streams). Any other region decode falls back
    /// to component planes of the region and hands them over at the end.
    /// A tile's own coding can still keep it there (see
    /// [`Self::full_decode_planes_floor`], which plans the full image only
    /// when this holds).
    #[doc(hidden)]
    #[must_use]
    pub fn region_decode_streams_rows(&self) -> bool {
        self.permits_line_region_idwt()
            && self.header.component_infos.iter().all(|component| {
                component.size_info.precision <= 24
                    && component.size_info.horizontal_resolution == 1
                    && component.size_info.vertical_resolution == 1
            })
            && j2c::image_streams_rows(&self.header)
    }

    /// The bytes a full-image decode into component planes charges the
    /// decode budget at least, planned from the codestream's tiles without
    /// decoding, when [`Self::decode_region_rows_for_integer_output_with_context`]
    /// over the whole image could stream its rows instead and so need no
    /// planes. A caller holding the planes to `budget` bytes runs that
    /// region decode when the floor passes it: the planes decode would be
    /// refused, and an image it can finish never has a floor past the
    /// budget.
    ///
    /// `None` when the region decode could not stream (see
    /// [`Self::region_decode_streams_rows`], or a tile with an inverse MCT
    /// over fewer than three components or over wavelets that differ), or
    /// when the header alone puts the planes and one tile's coefficients and
    /// IDWT buffers within half of `budget`.
    ///
    /// # Errors
    ///
    /// Returns the error parsing the tiles meets, which the planes decode
    /// meets as well.
    #[doc(hidden)]
    pub fn full_decode_planes_floor(&self, budget: usize) -> Result<Option<usize>> {
        if !self.region_decode_streams_rows() {
            return Ok(None);
        }
        j2c::full_decode_planes_floor(
            self.codestream,
            &self.header,
            self.retained_header_metadata_bytes()?,
            budget,
        )
    }

    /// The output rectangles `(x, y, width, height)` of the image's tiles at
    /// this image's resolution, in raster order, leaving out tiles with no
    /// output samples. A region decode of each rectangle decodes that tile
    /// alone, and together they cover the image once.
    ///
    /// # Errors
    ///
    /// Returns an error when the header's sampling factors are invalid.
    #[doc(hidden)]
    pub fn output_tile_rects(&self) -> Result<Vec<(u32, u32, u32, u32)>> {
        let size = &self.header.size_data;
        let (x_shrink, y_shrink) = (
            size.checked_x_shrink_factor()?,
            size.checked_y_shrink_factor()?,
        );
        let (x_origin, y_origin) = j2c::output_grid_offset(size);
        let mut rects = Vec::new();
        for ty in 0..size.num_y_tiles() {
            for tx in 0..size.num_x_tiles() {
                let rect = j2c::tile_rect(size, tx, ty);
                let x = rect.x0.div_ceil(x_shrink).saturating_sub(x_origin);
                let y = rect.y0.div_ceil(y_shrink).saturating_sub(y_origin);
                let x1 = rect.x1.div_ceil(x_shrink).saturating_sub(x_origin);
                let y1 = rect.y1.div_ceil(y_shrink).saturating_sub(y_origin);
                let width = x1.saturating_sub(x);
                let height = y1.saturating_sub(y);
                if width > 0 && height > 0 {
                    rects.push((x, y, width, height));
                }
            }
        }
        Ok(rects)
    }

    /// Rounded-for-integer-output counterpart of
    /// [`Self::decode_components_with_ht_decoder`].
    ///
    /// # Errors
    ///
    /// Returns an error when component precision is unsupported or decoding fails.
    #[doc(hidden)]
    pub fn decode_components_for_integer_output_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        self.decode_borrowed_component_planes(decoder_context, None, Some(ht_decoder), true)
    }

    /// Rounded-for-integer-output counterpart of
    /// [`Self::decode_region_components_with_ht_decoder`].
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid, precision is unsupported, or decoding fails.
    #[doc(hidden)]
    pub fn decode_region_components_for_integer_output_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        roi: (u32, u32, u32, u32),
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        self.decode_borrowed_component_planes(decoder_context, Some(roi), Some(ht_decoder), true)
    }

    pub(super) fn decode_borrowed_component_planes<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        roi: Option<(u32, u32, u32, u32)>,
        ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
        round_irreversible_output: bool,
    ) -> Result<DecodedComponents<'ctx>> {
        let dimensions = match roi {
            Some(roi) => {
                validate_roi((self.width(), self.height()), roi)?;
                (roi.2, roi.3)
            }
            None => (self.width(), self.height()),
        };
        self.validate_component_plane_precision()?;
        let (decoded_image, _charged_image_bytes) = self.decode_image(
            decoder_context,
            roi,
            ht_decoder,
            round_irreversible_output,
            self.retained_header_metadata_bytes()?,
        )?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        self.try_borrow_component_planes_with_retained_baseline(
            decoded_components.as_slice(),
            decoded_components.capacity(),
            dimensions,
            self.retained_header_metadata_bytes()?,
        )
    }
}

/// Hands a sink the streamed rows of every component but the image's
/// [`Image::ignored_components`].
struct KeptComponentsSink<'s> {
    sink: &'s mut dyn ComponentRowSink,
    ignored_components: usize,
}

impl ComponentRowSink for KeptComponentsSink<'_> {
    fn write_rows(&mut self, x: u32, y: u32, planes: &[ComponentPlane<'_>]) {
        let kept = planes.len().saturating_sub(self.ignored_components);
        self.sink.write_rows(x, y, &planes[..kept]);
    }

    fn rgb8_region(&mut self, width: u32, height: u32) -> Option<&mut [u8]> {
        self.sink.rgb8_region(width, height)
    }
}
