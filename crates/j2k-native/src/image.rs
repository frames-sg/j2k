// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::error::err;
use crate::j2c::{self, Header};
use crate::jp2::colr::EnumeratedColorspace;
use crate::jp2::{self, DecodedImage, ImageBoxes};
use crate::{
    checked_decode_byte_len3, convert_color_space, interleave_and_convert,
    interleave_and_convert_region, resolve_palette_indices, try_resize_decode_elements,
    validate_and_reorder_channels, validate_interleaved_output_buffer, validate_roi, Bitmap,
    ColorSpace, DecodedComponents, DecodedNativeComponents, DecoderContext, DecodingError,
    FormatError, HtCodeBlockDecoder, Result, CODESTREAM_MAGIC, JP2_MAGIC,
};

mod allocation;
mod compare;
#[cfg(test)]
mod contract_tests;
mod direct_api;
mod native;
mod output_api;
use self::allocation::retained_metadata_bytes;
pub(crate) use self::allocation::{retained_container_metadata_bytes, DecodeOwnerBudget};
use self::native::{try_clone_color_space, NativeOutputBudget};

/// Settings to apply during decoding.
#[derive(Debug, Copy, Clone)]
pub struct DecodeSettings {
    /// Whether palette indices should be resolved.
    ///
    /// JPEG2000 images can be stored in two different ways. First, by storing
    /// RGB values (depending on the color space) for each pixel. Secondly, by
    /// only storing a single index for each channel, and then resolving the
    /// actual color using the index.
    ///
    /// If you disable this option, in case you have an image with palette
    /// indices, they will not be resolved, but instead a grayscale image
    /// will be returned, with each pixel value corresponding to the palette
    /// index of the location.
    pub resolve_palette_indices: bool,
    /// Whether strict mode should be enabled when decoding.
    ///
    /// The default is lenient for compatibility with older releases. Lenient
    /// mode may tolerate malformed optional container metadata that strict mode
    /// rejects. Use [`DecodeSettings::strict`] for fail-closed validation of
    /// public or adversarial inputs.
    pub strict: bool,
    /// A hint for the target resolution that the image should be decoded at.
    pub target_resolution: Option<(u32, u32)>,
}

impl DecodeSettings {
    /// Compatibility decode settings.
    ///
    /// Lenient mode keeps the historical behavior of accepting recoverable
    /// optional metadata problems where possible.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: false,
            target_resolution: None,
        }
    }

    /// Strict decode settings for fail-closed validation.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        }
    }

    /// Whether the settings permit lenient tolerance of malformed optional
    /// metadata.
    #[must_use]
    pub const fn lenient_tolerance_enabled(&self) -> bool {
        !self.strict
    }
}

impl Default for DecodeSettings {
    fn default() -> Self {
        Self::lenient()
    }
}

/// A JPEG2000 image or codestream.
pub struct Image<'a> {
    /// Complete encoded input retained by the caller. Referenced execution
    /// plans express compressed payload ranges relative to this owner.
    pub(crate) encoded_input: &'a [u8],
    /// The tile-part payload used by the legacy JPEG 2000 decoder.
    pub(crate) codestream: &'a [u8],
    /// The header of the J2C codestream.
    pub(crate) header: Header<'a>,
    /// The JP2 boxes of the image. In the case of a raw codestream, we
    /// will synthesize the necessary boxes.
    pub(crate) boxes: ImageBoxes,
    /// Settings that should be applied during decoding.
    pub(crate) settings: DecodeSettings,
    /// Whether the image has an alpha channel.
    pub(crate) has_alpha: bool,
    /// The color space of the image.
    pub(crate) color_space: ColorSpace,
}

#[derive(Clone, Copy)]
pub(crate) struct ImageSource<'a> {
    encoded_input: &'a [u8],
    codestream: &'a [u8],
}

impl<'a> ImageSource<'a> {
    pub(crate) const fn new(encoded_input: &'a [u8], codestream: &'a [u8]) -> Self {
        Self {
            encoded_input,
            codestream,
        }
    }
}

impl<'a> Image<'a> {
    pub(crate) fn from_parsed_parts(
        source: ImageSource<'a>,
        header: Header<'a>,
        boxes: ImageBoxes,
        settings: DecodeSettings,
        color_space: ColorSpace,
        has_alpha: bool,
    ) -> Result<Self> {
        Self::from_parsed_parts_with_retained_baseline(
            source,
            header,
            boxes,
            settings,
            color_space,
            has_alpha,
            0,
        )
    }

    pub(crate) fn from_parsed_parts_with_retained_baseline(
        source: ImageSource<'a>,
        header: Header<'a>,
        boxes: ImageBoxes,
        settings: DecodeSettings,
        color_space: ColorSpace,
        has_alpha: bool,
        retained_baseline_bytes: usize,
    ) -> Result<Self> {
        let metadata_bytes = retained_metadata_bytes(&header, &boxes, &color_space)?;
        allocation::combine_retained_bytes(retained_baseline_bytes, metadata_bytes)?;
        Ok(Self {
            encoded_input: source.encoded_input,
            codestream: source.codestream,
            header,
            boxes,
            settings,
            has_alpha,
            color_space,
        })
    }

    pub(crate) fn retained_metadata_bytes(&self) -> Result<usize> {
        retained_metadata_bytes(&self.header, &self.boxes, &self.color_space)
    }

    /// Return the allocator capacities retained by this parsed image.
    ///
    /// # Errors
    ///
    /// Returns an error if nested metadata capacity arithmetic overflows or
    /// exceeds the native decode cap.
    #[doc(hidden)]
    pub fn retained_allocation_bytes(&self) -> Result<usize> {
        self.retained_metadata_bytes()
    }

    /// Try to create a new JPEG2000 image from the given data.
    ///
    /// # Errors
    ///
    /// Returns an error when the input signature, container, or codestream is invalid.
    pub fn new(data: &'a [u8], settings: &DecodeSettings) -> Result<Self> {
        if data.starts_with(JP2_MAGIC) {
            jp2::parse(data, *settings)
        } else if data.starts_with(CODESTREAM_MAGIC) {
            j2c::parse(data, settings)
        } else {
            err!(FormatError::InvalidSignature)
        }
    }

    /// Parse an image while accounting already-live codec-owned allocations.
    ///
    /// This adapter is used when validation parses another image while an
    /// encoded output and earlier parsed metadata remain live.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is invalid or aggregate parser-owned
    /// allocations exceed the native decode cap.
    #[doc(hidden)]
    pub fn new_with_retained_baseline(
        data: &'a [u8],
        settings: &DecodeSettings,
        retained_baseline_bytes: usize,
    ) -> Result<Self> {
        if retained_baseline_bytes == 0 {
            return Self::new(data, settings);
        }
        if data.starts_with(JP2_MAGIC) {
            jp2::parse_with_retained_baseline(data, *settings, retained_baseline_bytes)
        } else if data.starts_with(CODESTREAM_MAGIC) {
            j2c::parse_with_retained_baseline(data, settings, retained_baseline_bytes)
        } else {
            err!(FormatError::InvalidSignature)
        }
    }

    /// Whether the image has an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// The color space of the image.
    #[must_use]
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// The width of the image.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.header.size_data.image_width()
    }

    /// The height of the image.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.header.size_data.image_height()
    }

    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    #[must_use]
    pub fn original_bit_depth(&self) -> u8 {
        // Note that this only works if all components have the same precision.
        self.header.component_infos[0].size_info.precision
    }

    /// Whether decode finishes with additional host-side component mutation or reordering.
    #[doc(hidden)]
    #[must_use]
    pub fn supports_direct_device_plane_reuse(&self) -> bool {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return false;
        }
        if self.boxes.channel_definition.is_some() {
            return false;
        }
        !matches!(
            self.boxes
                .primary_color_specification()
                .map(|spec| &spec.color_space),
            Some(jp2::colr::ColorSpace::Enumerated(
                EnumeratedColorspace::Sycc | EnumeratedColorspace::CieLab(_)
            ))
        )
    }

    /// Decode the image and return its decoded result as a `Vec<u8>`, with each
    /// channel interleaved.
    ///
    /// # Errors
    ///
    /// Returns an error when image validation, decoding, or output allocation fails.
    pub fn decode(&self) -> Result<Vec<u8>> {
        let bitmap = self.decode_with_context(&mut DecoderContext::default())?;
        Ok(bitmap.data)
    }

    /// Decode the image and return its decoded result using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    ///
    /// # Errors
    ///
    /// Returns an error when image validation, decoding, or output allocation fails.
    pub fn decode_with_context(&self, decoder_context: &mut DecoderContext<'a>) -> Result<Bitmap> {
        (|| {
            let retained_image_bytes = self.retained_metadata_bytes()?;
            let mut decoded_image =
                self.decode_image(decoder_context, None, None, retained_image_bytes)?;
            let component_owner_capacity = decoded_image.decoded_components.capacity();
            let buffer_size = checked_decode_byte_len3(
                self.width() as usize,
                self.height() as usize,
                decoded_image.decoded_components.len(),
            )?;
            let mut budget = NativeOutputBudget::for_decoded_channels(
                retained_image_bytes,
                decoded_image.decoded_components,
                component_owner_capacity,
            )?;
            budget.include_elements::<u8>(buffer_size)?;
            budget.include_color_space_clone(&self.color_space)?;

            let color_space = try_clone_color_space(&self.color_space)?;
            budget.include_color_space_clone_overage(&self.color_space, &color_space)?;
            let mut data = Vec::new();
            try_resize_decode_elements(&mut data, buffer_size, 0_u8)?;
            budget.include_capacity_overage::<u8>(buffer_size, data.capacity())?;
            validate_interleaved_output_buffer(&decoded_image, &data)?;
            interleave_and_convert(&mut decoded_image, &mut data)?;
            let bitmap = Bitmap {
                color_space,
                data,
                has_alpha: self.has_alpha,
                width: self.width(),
                height: self.height(),
                original_bit_depth: self.original_bit_depth(),
            };
            NativeOutputBudget::validate_bitmap_pack(
                retained_image_bytes,
                decoded_image.decoded_components,
                component_owner_capacity,
                &bitmap,
            )?;
            Ok(bitmap)
        })()
    }

    /// Decode the image into borrowed component planes using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    ///
    /// # Errors
    ///
    /// Returns an error when component precision is unsupported or decoding fails.
    pub fn decode_components_with_context<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        self.validate_component_plane_precision()?;
        let decoded_image =
            self.decode_image(decoder_context, None, None, self.retained_metadata_bytes()?)?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        self.try_borrow_component_planes(
            decoded_components.as_slice(),
            decoded_components.capacity(),
            (self.width(), self.height()),
        )
    }

    /// Decode the image into owned native-bit-depth component planes.
    ///
    /// Unlike [`Self::decode_native`], this preserves per-component bit depth
    /// and signedness metadata and does not require all components to share a
    /// single packed interleaved representation.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, decoding, or native sample packing fails.
    pub fn decode_native_components(&self) -> Result<DecodedNativeComponents> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_components_with_context(&mut decoder_context)
    }

    /// Decode owned native component planes while accounting an already-live
    /// external allocation, such as the encoded `Vec` being validated.
    ///
    /// `retained_capacity` must be the allocator capacity of that external
    /// owner, not merely its logical length.
    ///
    /// # Errors
    ///
    /// Returns an error when aggregate retained allocation accounting,
    /// decoding, or native component packing fails.
    #[doc(hidden)]
    pub fn decode_native_components_with_retained_capacity(
        &self,
        retained_capacity: usize,
    ) -> Result<DecodedNativeComponents> {
        let retained_baseline_bytes =
            allocation::combine_retained_bytes(retained_capacity, self.retained_metadata_bytes()?)?;
        let mut decoder_context = DecoderContext::default();
        self.decode_native_components_with_context_and_retained_baseline(
            &mut decoder_context,
            retained_baseline_bytes,
        )
    }

    /// Decode the image into owned native-bit-depth component planes using a
    /// caller-provided decoder context.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, decoding, or native sample packing fails.
    pub fn decode_native_components_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        let retained_baseline_bytes = self.retained_metadata_bytes()?;
        self.decode_native_components_with_context_and_retained_baseline(
            decoder_context,
            retained_baseline_bytes,
        )
    }

    fn decode_native_components_with_context_and_retained_baseline(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        retained_baseline_bytes: usize,
    ) -> Result<DecodedNativeComponents> {
        let decoded_image =
            self.decode_image(decoder_context, None, None, retained_baseline_bytes)?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        let component_owner_capacity = decoded_components.capacity();
        self.pack_native_component_planes(
            decoded_components,
            component_owner_capacity,
            (self.width(), self.height()),
            retained_baseline_bytes,
        )
    }

    /// Decode borrowed component planes while delegating HTJ2K code-block decode.
    #[doc(hidden)]
    pub fn decode_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        self.validate_component_plane_precision()?;
        let decoded_image = self.decode_image(
            decoder_context,
            None,
            Some(ht_decoder),
            self.retained_metadata_bytes()?,
        )?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        self.try_borrow_component_planes(
            decoded_components.as_slice(),
            decoded_components.capacity(),
            (self.width(), self.height()),
        )
    }

    /// Decode borrowed component planes for a requested region using a
    /// caller-provided decoder context.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid, precision is unsupported, or decoding fails.
    pub fn decode_region_components_with_context<'ctx>(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        validate_roi((self.width(), self.height()), roi)?;
        self.validate_component_plane_precision()?;
        let (_x, _y, width, height) = roi;
        let decoded_image = self.decode_image(
            decoder_context,
            Some(roi),
            None,
            self.retained_metadata_bytes()?,
        )?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        self.try_borrow_component_planes(
            decoded_components.as_slice(),
            decoded_components.capacity(),
            (width, height),
        )
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes using a caller-provided decoder context.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid or decoding and packing fail.
    pub fn decode_native_region_components_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        validate_roi((self.width(), self.height()), roi)?;
        if self.requires_exact_integer_decode() {
            return self.decode_native_region_components_via_full_decode(roi, decoder_context);
        }
        let (_x, _y, width, height) = roi;
        let retained_image_bytes = self.retained_metadata_bytes()?;
        let decoded_image =
            self.decode_image(decoder_context, Some(roi), None, retained_image_bytes)?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        let component_owner_capacity = decoded_components.capacity();
        self.pack_native_component_planes(
            decoded_components,
            component_owner_capacity,
            (width, height),
            retained_image_bytes,
        )
    }

    /// Decode borrowed component planes for a requested region while
    /// delegating code-block/transform stages through the adapter backend hook.
    #[doc(hidden)]
    pub fn decode_region_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        roi: (u32, u32, u32, u32),
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        validate_roi((self.width(), self.height()), roi)?;
        self.validate_component_plane_precision()?;
        let (_x, _y, width, height) = roi;
        let decoded_image = self.decode_image(
            decoder_context,
            Some(roi),
            Some(ht_decoder),
            self.retained_metadata_bytes()?,
        )?;
        let DecodedImage {
            decoded_components,
            boxes: _,
        } = decoded_image;
        self.try_borrow_component_planes(
            decoded_components.as_slice(),
            decoded_components.capacity(),
            (width, height),
        )
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid or decoding fails.
    pub fn decode_region(&self, roi: (u32, u32, u32, u32)) -> Result<Bitmap> {
        self.decode_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap
    /// using a caller-provided decoder context.
    ///
    /// # Errors
    ///
    /// Returns an error when the region is invalid, decoding fails, or output sizing overflows.
    pub fn decode_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<Bitmap> {
        validate_roi((self.width(), self.height()), roi)?;
        (|| {
            let retained_image_bytes = self.retained_metadata_bytes()?;
            let mut decoded_image =
                self.decode_image(decoder_context, Some(roi), None, retained_image_bytes)?;
            let component_owner_capacity = decoded_image.decoded_components.capacity();
            let (_x, _y, width, height) = roi;
            let data_len = checked_decode_byte_len3(
                width as usize,
                height as usize,
                decoded_image.decoded_components.len(),
            )?;
            let mut budget = NativeOutputBudget::for_decoded_channels(
                retained_image_bytes,
                decoded_image.decoded_components,
                component_owner_capacity,
            )?;
            budget.include_elements::<u8>(data_len)?;
            budget.include_color_space_clone(&self.color_space)?;

            let color_space = try_clone_color_space(&self.color_space)?;
            budget.include_color_space_clone_overage(&self.color_space, &color_space)?;
            let mut data = Vec::new();
            try_resize_decode_elements(&mut data, data_len, 0_u8)?;
            budget.include_capacity_overage::<u8>(data_len, data.capacity())?;
            interleave_and_convert_region(
                &mut decoded_image,
                width as usize,
                (0, 0, width, height),
                &mut data,
            );
            let bitmap = Bitmap {
                color_space,
                data,
                has_alpha: self.has_alpha,
                width,
                height,
                original_bit_depth: self.original_bit_depth(),
            };
            NativeOutputBudget::validate_bitmap_pack(
                retained_image_bytes,
                decoded_image.decoded_components,
                component_owner_capacity,
                &bitmap,
            )?;
            Ok(bitmap)
        })()
    }

    /// Decode the image into the given buffer.
    ///
    /// This method does the same as [`Image::decode`], but you can provide
    /// a custom buffer for the output, as well as a decoder context. Doing
    /// so allows the internal decode engine to reuse memory allocations, so
    /// this is especially recommended if you plan on converting multiple
    /// images in the same session.
    ///
    /// The buffer must have the correct size.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding fails or `buf` is too small for the image.
    pub fn decode_into(
        &self,
        buf: &mut [u8],
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<()> {
        let mut decoded_image =
            self.decode_image(decoder_context, None, None, self.retained_metadata_bytes()?)?;
        validate_interleaved_output_buffer(&decoded_image, buf)?;
        interleave_and_convert(&mut decoded_image, buf)?;

        Ok(())
    }

    fn decode_image<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
        retained_baseline_bytes: usize,
    ) -> Result<DecodedImage<'ctx, '_>> {
        let settings = &self.settings;
        let mut ht_decoder = ht_decoder;
        decoder_context.set_output_region(output_region);
        let decode_result = j2c::decode(
            self.codestream,
            &self.header,
            retained_baseline_bytes,
            decoder_context,
            &mut ht_decoder,
        );
        decoder_context.set_output_region(None);
        decode_result?;
        let mut decoded_image = DecodedImage {
            decoded_components: &mut decoder_context.tile_decode_context.channel_data,
            boxes: &self.boxes,
        };

        if settings.resolve_palette_indices {
            let components = core::mem::take(decoded_image.decoded_components);
            *decoded_image.decoded_components =
                resolve_palette_indices(components, decoded_image.boxes, retained_baseline_bytes)?;
        }

        if let Some(cdef) = &decoded_image.boxes.channel_definition {
            validate_and_reorder_channels(
                cdef,
                decoded_image.decoded_components,
                retained_baseline_bytes,
            )?;
        }

        let bit_depth = decoded_image
            .decoded_components
            .first()
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
            .bit_depth;
        convert_color_space(&mut decoded_image, bit_depth)?;
        Ok(decoded_image)
    }
}
