// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use crate::error::{bail, err};
use crate::j2c::{self, ComponentData, Header};
use crate::jp2::cdef::ChannelAssociation;
use crate::jp2::colr::EnumeratedColorspace;
use crate::jp2::{self, DecodedImage, ImageBoxes};
use crate::{
    checked_decode_byte_len2, checked_decode_byte_len3, checked_decode_byte_len4,
    checked_decode_sample_count, convert_color_space, interleave_and_convert,
    interleave_and_convert_region, math, native_bytes_per_sample,
    native_component_plane_dimensions, resolve_palette_indices, validate_channel_definition,
    validate_interleaved_output_buffer, validate_roi, Bitmap, ColorSpace, ComponentPlane,
    DecodedComponents, DecodedNativeComponents, DecoderContext, DecodingError,
    DirectPlanUnsupportedReason, FormatError, HtCodeBlockDecoder, J2kDirectColorPlan,
    J2kDirectGrayscalePlan, NativeComponentPlane, RawBitmap, Result, Reversible53CoefficientImage,
    ValidationError, CODESTREAM_MAGIC, JP2_MAGIC,
};

mod native;

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

impl<'a> Image<'a> {
    /// Try to create a new JPEG2000 image from the given data.
    pub fn new(data: &'a [u8], settings: &DecodeSettings) -> Result<Self> {
        if data.starts_with(JP2_MAGIC) {
            jp2::parse(data, *settings)
        } else if data.starts_with(CODESTREAM_MAGIC) {
            j2c::parse(data, settings)
        } else {
            err!(FormatError::InvalidSignature)
        }
    }

    /// Whether the image has an alpha channel.
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// The color space of the image.
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// The width of the image.
    pub fn width(&self) -> u32 {
        self.header.size_data.image_width()
    }

    /// The height of the image.
    pub fn height(&self) -> u32 {
        self.header.size_data.image_height()
    }

    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub fn original_bit_depth(&self) -> u8 {
        // Note that this only works if all components have the same precision.
        self.header.component_infos[0].size_info.precision
    }

    /// Whether decode finishes with additional host-side component mutation or reordering.
    #[doc(hidden)]
    pub fn supports_direct_device_plane_reuse(&self) -> bool {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return false;
        }
        if self.boxes.channel_definition.is_some() {
            return false;
        }
        !matches!(
            self.boxes
                .color_specification
                .as_ref()
                .map(|spec| &spec.color_space),
            Some(jp2::colr::ColorSpace::Enumerated(
                EnumeratedColorspace::Sycc | EnumeratedColorspace::CieLab(_)
            ))
        )
    }

    /// Decode the image and return its decoded result as a `Vec<u8>`, with each
    /// channel interleaved.
    pub fn decode(&self) -> Result<Vec<u8>> {
        let bitmap = self.decode_with_context(&mut DecoderContext::default())?;
        Ok(bitmap.data)
    }

    /// Decode the image and return its decoded result using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    pub fn decode_with_context(&self, decoder_context: &mut DecoderContext<'a>) -> Result<Bitmap> {
        let buffer_size = checked_decode_byte_len3(
            self.width() as usize,
            self.height() as usize,
            self.color_space.num_channels() as usize + if self.has_alpha { 1 } else { 0 },
        )?;
        let mut buf = vec![0; buffer_size];
        self.decode_into(&mut buf, decoder_context)?;

        Ok(Bitmap {
            color_space: self.color_space.clone(),
            data: buf,
            has_alpha: self.has_alpha,
            width: self.width(),
            height: self.height(),
            original_bit_depth: self.original_bit_depth(),
        })
    }

    /// Decode the image into borrowed component planes using a caller-provided
    /// decoder context so allocations can be reused across repeated decodes.
    pub fn decode_components_with_context<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        self.validate_component_plane_precision()?;
        let decoded_image = self.prepare_decoded_image(decoder_context)?;
        Ok(self.borrow_component_planes(
            decoded_image.decoded_components.as_slice(),
            (self.width(), self.height()),
        ))
    }

    /// Decode the image into owned native-bit-depth component planes.
    ///
    /// Unlike [`Self::decode_native`], this preserves per-component bit depth
    /// and signedness metadata and does not require all components to share a
    /// single packed interleaved representation.
    pub fn decode_native_components(&self) -> Result<DecodedNativeComponents> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_components_with_context(&mut decoder_context)
    }

    /// Decode the image into owned native-bit-depth component planes using a
    /// caller-provided decoder context.
    pub fn decode_native_components_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<DecodedNativeComponents> {
        let decoded_image = self.prepare_decoded_image(decoder_context)?;
        self.pack_native_component_planes(
            decoded_image.decoded_components,
            (self.width(), self.height()),
        )
    }

    /// Build a adapter grayscale direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a adapter grayscale direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_grayscale_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectGrayscalePlan> {
        if !matches!(self.color_space, ColorSpace::Gray) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::GrayscaleImageWithoutAlpha
            ));
        }

        decoder_context.set_output_region(Some(output_region));
        let result =
            j2c::build_direct_grayscale_plan(self.codestream, &self.header, decoder_context);
        decoder_context.set_output_region(None);
        result
    }

    /// Build a adapter RGB direct device plan without materializing host component planes.
    #[doc(hidden)]
    pub fn build_direct_color_plan_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context)
    }

    /// Build a adapter RGB direct device plan for an output-space region.
    #[doc(hidden)]
    pub fn build_direct_color_plan_region_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: (u32, u32, u32, u32),
    ) -> Result<J2kDirectColorPlan> {
        if !matches!(self.color_space, ColorSpace::RGB) || self.has_alpha {
            bail!(DecodingError::DirectPlanUnsupported(
                DirectPlanUnsupportedReason::ColorRgbImageWithoutAlpha
            ));
        }

        decoder_context.set_output_region(Some(output_region));
        let result = j2c::build_direct_color_plan(self.codestream, &self.header, decoder_context);
        decoder_context.set_output_region(None);
        result
    }

    /// Decode borrowed component planes while delegating HTJ2K code-block decode.
    #[doc(hidden)]
    pub fn decode_components_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedComponents<'ctx>> {
        self.validate_component_plane_precision()?;
        let decoded_image =
            self.prepare_decoded_image_with_ht_decoder(decoder_context, ht_decoder)?;
        Ok(self.borrow_component_planes(
            decoded_image.decoded_components.as_slice(),
            (self.width(), self.height()),
        ))
    }

    /// Decode borrowed component planes for a requested region using a
    /// caller-provided decoder context.
    pub fn decode_region_components_with_context<'ctx>(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedComponents<'ctx>> {
        validate_roi((self.width(), self.height()), roi)?;
        self.validate_component_plane_precision()?;
        let (_x, _y, width, height) = roi;
        let decoded_image = self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        Ok(self
            .borrow_component_planes(decoded_image.decoded_components.as_slice(), (width, height)))
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes using a caller-provided decoder context.
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
        let decoded_image = self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        self.pack_native_component_planes(decoded_image.decoded_components, (width, height))
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
        let decoded_image = self.prepare_decoded_image_with_region_and_ht_decoder(
            decoder_context,
            Some(roi),
            Some(ht_decoder),
        )?;
        Ok(self
            .borrow_component_planes(decoded_image.decoded_components.as_slice(), (width, height)))
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap.
    pub fn decode_region(&self, roi: (u32, u32, u32, u32)) -> Result<Bitmap> {
        self.decode_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a region of the image and return it as an 8-bit interleaved bitmap
    /// using a caller-provided decoder context.
    pub fn decode_region_with_context(
        &self,
        roi: (u32, u32, u32, u32),
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<Bitmap> {
        validate_roi((self.width(), self.height()), roi)?;
        let mut decoded_image =
            self.prepare_decoded_image_with_region(decoder_context, Some(roi))?;
        let (_x, _y, width, height) = roi;
        let channels =
            self.color_space.num_channels() as usize + if self.has_alpha { 1 } else { 0 };
        let data_len = checked_decode_byte_len3(width as usize, height as usize, channels)?;
        let mut data = vec![0; data_len];
        interleave_and_convert_region(
            &mut decoded_image,
            width as usize,
            (0, 0, width, height),
            &mut data,
        );
        Ok(Bitmap {
            color_space: self.color_space.clone(),
            data,
            has_alpha: self.has_alpha,
            width,
            height,
            original_bit_depth: self.original_bit_depth(),
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
    pub fn decode_native(&self) -> Result<RawBitmap> {
        let mut decoder_context = DecoderContext::default();
        self.decode_native_with_context(&mut decoder_context)
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
            decoder_context,
        )
    }

    /// Decode a region of the image at native bit depth.
    pub fn decode_native_region(&self, roi: (u32, u32, u32, u32)) -> Result<RawBitmap> {
        self.decode_native_region_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode a source-coordinate region into owned native-bit-depth component
    /// planes.
    pub fn decode_native_region_components(
        &self,
        roi: (u32, u32, u32, u32),
    ) -> Result<DecodedNativeComponents> {
        self.decode_native_region_components_with_context(roi, &mut DecoderContext::default())
    }

    /// Decode the image at native bit depth using a caller-provided decoder
    /// context so allocations can be reused across repeated decodes.
    pub fn decode_native_with_context(
        &self,
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<RawBitmap> {
        let bit_depth = self.uniform_header_bit_depth()?;
        self.decode_with_output_region(decoder_context, None)?;

        let components = &decoder_context.tile_decode_context.channel_data;
        let num_components =
            u16::try_from(components.len()).map_err(|_| ValidationError::TooManyChannels)?;
        let width = self.width();
        let height = self.height();
        let pixel_count = checked_decode_sample_count(width, height)?;
        let component_signed = Self::component_signedness(components);
        let signed = component_signed.iter().all(|signed| *signed);

        let bytes_per_sample = native_bytes_per_sample(bit_depth)?;
        if bytes_per_sample == 1 {
            let capacity = checked_decode_byte_len2(pixel_count, usize::from(num_components))?;
            let mut data = Vec::with_capacity(capacity);
            for i in 0..pixel_count {
                for component in components.iter() {
                    Self::push_component_native_sample_bytes(&mut data, component, i, bit_depth);
                }
            }
            Ok(RawBitmap {
                data,
                width,
                height,
                bit_depth,
                signed,
                component_signed,
                num_components,
                bytes_per_sample: 1,
            })
        } else {
            let capacity = checked_decode_byte_len3(
                pixel_count,
                usize::from(num_components),
                bytes_per_sample,
            )?;
            let mut data = Vec::with_capacity(capacity);
            for i in 0..pixel_count {
                for component in components.iter() {
                    Self::push_component_native_sample_bytes(&mut data, component, i, bit_depth);
                }
            }
            Ok(RawBitmap {
                data,
                width,
                height,
                bit_depth,
                signed,
                component_signed,
                num_components,
                bytes_per_sample: u8::try_from(bytes_per_sample)
                    .map_err(|_| ValidationError::ImageTooLarge)?,
            })
        }
    }

    /// Decode a region of the image at native bit depth using a caller-provided
    /// decoder context.
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
        let mut data = Vec::with_capacity(capacity);
        let component_signed = Self::component_signedness(components);
        let signed = component_signed.iter().all(|signed| *signed);

        for row in 0..height as usize {
            for col in 0..width as usize {
                let idx = row * width as usize + col;
                for component in components {
                    Self::push_component_native_sample_bytes(&mut data, component, idx, bit_depth);
                }
            }
        }

        Ok(RawBitmap {
            data,
            width,
            height,
            bit_depth,
            signed,
            component_signed,
            num_components,
            bytes_per_sample: u8::try_from(bytes_per_sample)
                .map_err(|_| ValidationError::ImageTooLarge)?,
        })
    }

    fn component_signedness(components: &[ComponentData]) -> Vec<bool> {
        components
            .iter()
            .map(|component| component.signed)
            .collect()
    }

    fn component_plane_sampling(&self, plane_count: usize) -> Vec<(u8, u8)> {
        if self.settings.resolve_palette_indices && self.boxes.palette.is_some() {
            return vec![(1, 1); plane_count];
        }

        let mut sampling = self
            .header
            .component_infos
            .iter()
            .take(plane_count)
            .map(|component| {
                (
                    component.size_info.horizontal_resolution,
                    component.size_info.vertical_resolution,
                )
            })
            .collect::<Vec<_>>();
        sampling.resize(plane_count, (1, 1));
        sampling
    }

    fn borrow_component_planes<'ctx>(
        &self,
        components: &'ctx [ComponentData],
        dimensions: (u32, u32),
    ) -> DecodedComponents<'ctx> {
        let sampling = self.component_plane_sampling(components.len());
        let planes = components
            .iter()
            .zip(sampling)
            .map(|(component, sampling)| ComponentPlane {
                samples: component.container.truncated(),
                dimensions,
                bit_depth: component.bit_depth,
                signed: component.signed,
                sampling,
            })
            .collect();

        DecodedComponents {
            dimensions,
            color_space: self.color_space.clone(),
            has_alpha: self.has_alpha,
            planes,
        }
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

    fn validate_component_plane_precision(&self) -> Result<()> {
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

    /// Decode the image into the given buffer.
    ///
    /// This method does the same as [`Image::decode`], but you can provide
    /// a custom buffer for the output, as well as a decoder context. Doing
    /// so allows the internal decode engine to reuse memory allocations, so
    /// this is especially recommended if you plan on converting multiple
    /// images in the same session.
    ///
    /// The buffer must have the correct size.
    pub fn decode_into(
        &self,
        buf: &mut [u8],
        decoder_context: &mut DecoderContext<'a>,
    ) -> Result<()> {
        let mut decoded_image = self.prepare_decoded_image(decoder_context)?;
        validate_interleaved_output_buffer(&decoded_image, buf)?;
        interleave_and_convert(&mut decoded_image, buf)?;

        Ok(())
    }

    fn prepare_decoded_image<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region(decoder_context, None)
    }

    fn prepare_decoded_image_with_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        ht_decoder: &mut dyn HtCodeBlockDecoder,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region_and_ht_decoder(
            decoder_context,
            None,
            Some(ht_decoder),
        )
    }

    fn prepare_decoded_image_with_region<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
    ) -> Result<DecodedImage<'ctx>> {
        self.prepare_decoded_image_with_region_and_ht_decoder(decoder_context, output_region, None)
    }

    fn prepare_decoded_image_with_region_and_ht_decoder<'ctx>(
        &self,
        decoder_context: &'ctx mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
    ) -> Result<DecodedImage<'ctx>> {
        let settings = &self.settings;
        self.decode_with_output_region_and_ht_decoder(decoder_context, output_region, ht_decoder)?;
        let mut decoded_image = DecodedImage {
            decoded_components: &mut decoder_context.tile_decode_context.channel_data,
            boxes: self.boxes.clone(),
        };

        if settings.resolve_palette_indices {
            let components = core::mem::take(decoded_image.decoded_components);
            *decoded_image.decoded_components =
                resolve_palette_indices(components, &decoded_image.boxes)?;
        }

        if let Some(cdef) = &decoded_image.boxes.channel_definition {
            validate_channel_definition(cdef, decoded_image.decoded_components.len())?;
            let mut components = decoded_image
                .decoded_components
                .iter()
                .cloned()
                .zip(
                    cdef.channel_definitions
                        .iter()
                        .map(|c| match c._association {
                            ChannelAssociation::WholeImage => u16::MAX,
                            ChannelAssociation::Colour(c) => c,
                            ChannelAssociation::Unspecified => u16::MAX,
                        }),
                )
                .collect::<Vec<_>>();
            components.sort_by_key(|component| component.1);
            *decoded_image.decoded_components = components.into_iter().map(|c| c.0).collect();
        }

        let bit_depth = decoded_image.decoded_components[0].bit_depth;
        convert_color_space(&mut decoded_image, bit_depth)?;
        Ok(decoded_image)
    }

    fn decode_with_output_region(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
    ) -> Result<()> {
        self.decode_with_output_region_and_ht_decoder(decoder_context, output_region, None)
    }

    fn decode_with_output_region_and_ht_decoder(
        &self,
        decoder_context: &mut DecoderContext<'a>,
        output_region: Option<(u32, u32, u32, u32)>,
        mut ht_decoder: Option<&mut dyn HtCodeBlockDecoder>,
    ) -> Result<()> {
        decoder_context.set_output_region(output_region);
        let decode_result = j2c::decode(
            self.codestream,
            &self.header,
            decoder_context,
            &mut ht_decoder,
        );
        decoder_context.set_output_region(None);
        decode_result
    }
}
