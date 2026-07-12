// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native, coefficient, and borrowed-component decode output surfaces.

use alloc::vec::Vec;

use crate::color::{ComponentPlane, DecodedComponents, DecodedNativeComponents, RawBitmap};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::{self, ComponentData, DecoderContext, Reversible53CoefficientImage};
use crate::{
    checked_decode_byte_len3, checked_decode_byte_len4, checked_decode_sample_count,
    native_bytes_per_sample, try_reserve_decode_elements, validate_roi,
};

use super::native::{try_clone_color_space, NativeOutputBudget};
use super::Image;

impl<'a> Image<'a> {
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
            self.retained_metadata_bytes()?,
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
        j2c::recode::extract_reversible_53_coefficients(
            self.codestream,
            &self.header,
            self.retained_metadata_bytes()?,
            decoder_context,
        )
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
        let retained_baseline_bytes = self.retained_metadata_bytes()?;
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
        let bit_depth = self.uniform_header_bit_depth()?;
        self.decode_with_output_region_and_ht_decoder_with_retained_baseline(
            decoder_context,
            None,
            None,
            retained_baseline_bytes,
        )?;

        let components = &decoder_context.tile_decode_context.channel_data;
        let component_owner_capacity = components.capacity();
        let num_components =
            u16::try_from(components.len()).map_err(|_| ValidationError::TooManyChannels)?;
        let width = self.width();
        let height = self.height();
        let pixel_count = checked_decode_sample_count(width, height)?;
        let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
        let capacity =
            checked_decode_byte_len3(pixel_count, usize::from(num_components), bytes_per_sample)?;
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
        for index in 0..pixel_count {
            for component in components {
                Self::push_component_native_sample_bytes(&mut data, component, index, bit_depth);
            }
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
        validate_roi((self.width(), self.height()), roi)?;
        if self.requires_exact_integer_decode() {
            return self.decode_native_region_via_full_decode(roi, decoder_context);
        }
        let bit_depth = self.uniform_header_bit_depth()?;
        self.decode_with_output_region(decoder_context, Some(roi))?;

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
        let retained_image_bytes = self.retained_metadata_bytes()?;
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

        for row in 0..height as usize {
            for col in 0..width as usize {
                let index = row * width as usize + col;
                for component in components {
                    Self::push_component_native_sample_bytes(
                        &mut data, component, index, bit_depth,
                    );
                }
            }
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

    pub(super) fn try_borrow_component_planes<'ctx>(
        &self,
        components: &'ctx [ComponentData],
        component_owner_capacity: usize,
        dimensions: (u32, u32),
    ) -> Result<DecodedComponents<'ctx>> {
        let retained_image_bytes = self.retained_metadata_bytes()?;
        let mut budget = NativeOutputBudget::for_component_pack(
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
        budget
            .include_capacity_overage::<ComponentPlane<'_>>(components.len(), planes.capacity())?;
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
