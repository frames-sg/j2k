// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::error::bail;
use crate::j2c::{ComponentData, Header};
use crate::jp2::cdef::ChannelType;
use crate::jp2::colr::{CieLab, EnumeratedColorspace};
use crate::jp2::icc::ICCMetadata;
use crate::jp2::{self, DecodedImage, ImageBoxes};
use crate::math::{self, dispatch, f32x8, Level, Simd, SIMD_WIDTH};
use crate::{
    checked_decode_sample_count, try_reserve_decode_elements, ColorError, DecodeSettings,
    DecodingError, FormatError, Result, ValidationError, DEFAULT_MAX_DECODE_BYTES,
};

mod allocation;
mod postprocess;
pub(crate) use postprocess::{resolve_palette_indices, validate_and_reorder_channels};

pub(crate) fn resolve_alpha_and_color_space(
    boxes: &ImageBoxes,
    header: &Header<'_>,
    settings: &DecodeSettings,
    retained_baseline_bytes: usize,
) -> Result<(ColorSpace, bool)> {
    let mut num_components = header.component_infos.len();

    // Override number of components with what is actually in the palette box
    // in case we resolve them.
    if settings.resolve_palette_indices {
        if let Some(palette_box) = &boxes.palette {
            num_components = palette_box.columns.len();
        }
    }

    let mut has_alpha = false;

    if let Some(cdef) = &boxes.channel_definition {
        has_alpha = cdef.channel_definitions.iter().any(|definition| {
            matches!(
                definition.channel_type,
                ChannelType::Opacity | ChannelType::PremultipliedOpacity
            )
        });
    }

    // If palette indices remain unresolved, the exposed samples are indices;
    // avoid cloning an ICC profile that would be discarded immediately.
    let mut color_space = if !settings.resolve_palette_indices && boxes.palette.is_some() {
        has_alpha = false;
        ColorSpace::Gray
    } else {
        let retained_container_bytes =
            crate::image::retained_container_metadata_bytes(header, boxes)?
                .checked_add(retained_baseline_bytes)
                .ok_or(ValidationError::ImageTooLarge)?;
        if retained_container_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        get_color_space(boxes, num_components, retained_container_bytes)?
    };

    let actual_num_components = header.component_infos.len();

    // Validate the number of channels.
    if boxes.palette.is_none()
        && actual_num_components != usize::from(color_space.num_channels() + u16::from(has_alpha))
    {
        if !settings.strict
            && actual_num_components == usize::from(color_space.num_channels()) + 1
            && !has_alpha
        {
            // See OPENJPEG test case orb-blue10-lin-j2k. Assume that we have an
            // alpha channel in this case.
            has_alpha = true;
        } else {
            // Color space is invalid, attempt to repair.
            if actual_num_components == 1 || (actual_num_components == 2 && has_alpha) {
                color_space = ColorSpace::Gray;
            } else if actual_num_components == 3 {
                color_space = ColorSpace::RGB;
            } else if actual_num_components == 4 {
                if has_alpha {
                    color_space = ColorSpace::RGB;
                } else {
                    color_space = ColorSpace::CMYK;
                }
            } else {
                color_space = ColorSpace::Unknown {
                    num_channels: u16::try_from(actual_num_components)
                        .map_err(|_| ValidationError::TooManyChannels)?,
                };
            }
        }
    }

    Ok((color_space, has_alpha))
}

/// The color space of the image.
#[derive(Debug)]
pub enum ColorSpace {
    /// A grayscale image.
    Gray,
    /// An RGB image.
    RGB,
    /// A CMYK image.
    CMYK,
    /// An unknown color space.
    Unknown {
        /// The number of channels of the color space.
        num_channels: u16,
    },
    /// An image based on an ICC profile.
    Icc {
        /// The raw data of the ICC profile.
        profile: Vec<u8>,
        /// The number of channels used by the ICC profile.
        num_channels: u16,
    },
}

impl ColorSpace {
    /// Return the number of expected channels for the color space.
    #[must_use]
    pub fn num_channels(&self) -> u16 {
        match self {
            Self::Gray => 1,
            Self::RGB => 3,
            Self::CMYK => 4,
            Self::Unknown { num_channels } => *num_channels,
            Self::Icc {
                num_channels: num_components,
                ..
            } => *num_components,
        }
    }
}

/// A bitmap storing the decoded result of the image.
pub struct Bitmap {
    /// The color space of the image.
    pub color_space: ColorSpace,
    /// The raw pixel data of the image. The result will always be in
    /// 8-bit (in case the original image had a different bit-depth, this
    /// decode path scales it to 8-bit).
    ///
    /// The size is guaranteed to equal
    /// `width * height * (num_channels + (if has_alpha { 1 } else { 0 })`.
    /// Pixels are interleaved on a per-channel basis, the alpha channel always
    /// appearing as the last channel, if available.
    pub data: Vec<u8>,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The original bit depth of the image. You usually don't need to do anything
    /// with this parameter, it just exists for informational purposes.
    pub original_bit_depth: u8,
}

/// Raw decoded pixel data at native bit depth (no 8-bit scaling).
///
/// For bit depths ≤ 8, `data` contains one byte per sample.
/// For bit depths > 8 (e.g., 12 or 16), `data` contains two bytes per sample
/// in little-endian byte order (`u16` LE).
///
/// Samples are interleaved: for a 3-component image, the layout is
/// `[R0, G0, B0, R1, G1, B1, ...]`.
pub struct RawBitmap {
    /// The raw pixel data at native bit depth.
    pub data: Vec<u8>,
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
    /// The original bit depth per sample (e.g., 8, 12, 16).
    pub bit_depth: u8,
    /// Whether every component in this packed bitmap is signed.
    ///
    /// Use [`Self::component_signed`] for per-component signedness when
    /// handling arbitrary JPEG 2000 component metadata.
    pub signed: bool,
    /// Per-component signedness in codestream/component order.
    pub component_signed: Vec<bool>,
    /// The number of components (e.g., 1 for grayscale, 3 for RGB).
    pub num_components: u16,
    /// Bytes per sample in the packed little-endian native representation.
    pub bytes_per_sample: u8,
}

/// One owned decoded component plane at native bit depth.
pub struct NativeComponentPlane {
    pub(crate) data: Vec<u8>,
    pub(crate) dimensions: (u32, u32),
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) sampling: (u8, u8),
    pub(crate) bytes_per_sample: u8,
}

/// Allocation-free facade adapter representation of an owned native component plane.
#[doc(hidden)]
pub type NativeComponentPlaneParts = (Vec<u8>, (u32, u32), u8, bool, (u8, u8), u8);

impl NativeComponentPlane {
    /// Packed little-endian sample bytes for this component in row-major order.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    crate::__j2k_component_plane_metadata_accessors!();

    /// Bytes used for each packed little-endian sample in [`Self::data`].
    #[must_use]
    pub fn bytes_per_sample(&self) -> u8 {
        self.bytes_per_sample
    }

    /// Return the byte capacity owned by this plane.
    #[doc(hidden)]
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        self.data.capacity()
    }

    /// Consume this plane into allocation-free facade adapter parts.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> NativeComponentPlaneParts {
        (
            self.data,
            self.dimensions,
            self.bit_depth,
            self.signed,
            self.sampling,
            self.bytes_per_sample,
        )
    }
}

/// Owned decoded native-bit-depth component planes for an image.
pub struct DecodedNativeComponents {
    pub(crate) dimensions: (u32, u32),
    pub(crate) color_space: ColorSpace,
    pub(crate) has_alpha: bool,
    pub(crate) planes: Vec<NativeComponentPlane>,
}

impl DecodedNativeComponents {
    /// Dimensions of the decoded image represented by these planes.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    #[must_use]
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[NativeComponentPlane] {
        &self.planes
    }

    /// Return the actual heap capacity retained by this owned result.
    #[doc(hidden)]
    #[must_use]
    pub fn allocated_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .planes
            .capacity()
            .checked_mul(core::mem::size_of::<NativeComponentPlane>())?;
        for plane in &self.planes {
            bytes = bytes.checked_add(plane.allocated_bytes())?;
        }
        if let ColorSpace::Icc { profile, .. } = &self.color_space {
            bytes = bytes.checked_add(profile.capacity())?;
        }
        Some(bytes)
    }

    /// Consume this result into allocation-free facade adapter parts.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> ((u32, u32), ColorSpace, bool, Vec<NativeComponentPlane>) {
        (
            self.dimensions,
            self.color_space,
            self.has_alpha,
            self.planes,
        )
    }
}

/// A borrowed decoded component plane.
pub struct ComponentPlane<'a> {
    pub(crate) samples: &'a [f32],
    pub(crate) dimensions: (u32, u32),
    pub(crate) bit_depth: u8,
    pub(crate) signed: bool,
    pub(crate) sampling: (u8, u8),
}

/// Allocation-free facade adapter representation of a borrowed component plane.
#[doc(hidden)]
pub type ComponentPlaneParts<'a> = (&'a [f32], (u32, u32), u8, bool, (u8, u8));

impl<'a> ComponentPlane<'a> {
    /// Component samples in row-major order.
    #[must_use]
    pub fn samples(&self) -> &'a [f32] {
        self.samples
    }

    crate::__j2k_component_plane_metadata_accessors!();

    /// Consume this borrowed plane into allocation-free facade adapter parts.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> ComponentPlaneParts<'a> {
        (
            self.samples,
            self.dimensions,
            self.bit_depth,
            self.signed,
            self.sampling,
        )
    }
}

/// Borrowed decoded component planes for an image.
pub struct DecodedComponents<'a> {
    pub(crate) dimensions: (u32, u32),
    pub(crate) color_space: ColorSpace,
    pub(crate) has_alpha: bool,
    pub(crate) planes: Vec<ComponentPlane<'a>>,
    pub(crate) live_bytes: usize,
}

impl<'a> DecodedComponents<'a> {
    /// Dimensions of the decoded image represented by these planes.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Color space after JPEG 2000 color conversion has been applied.
    #[must_use]
    pub fn color_space(&self) -> &ColorSpace {
        &self.color_space
    }

    /// Whether the decoded image has an alpha channel.
    #[must_use]
    pub fn has_alpha(&self) -> bool {
        self.has_alpha
    }

    /// Borrowed decoded component planes in display order.
    #[must_use]
    pub fn planes(&self) -> &[ComponentPlane<'a>] {
        &self.planes
    }

    /// Return the retained heap capacity that remains live with this borrowed result.
    ///
    /// This includes the decoder-context component owners backing the borrowed
    /// sample slices, SIMD padding, exact-integer shadow buffers, the component
    /// metadata vector, and an owned ICC profile when present.
    #[doc(hidden)]
    #[must_use]
    pub fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    /// Consume this result into allocation-free facade adapter parts.
    #[doc(hidden)]
    #[must_use]
    pub fn into_parts(self) -> ((u32, u32), ColorSpace, bool, Vec<ComponentPlane<'a>>) {
        (
            self.dimensions,
            self.color_space,
            self.has_alpha,
            self.planes,
        )
    }
}

pub(crate) fn validate_interleaved_output_buffer(
    image: &DecodedImage<'_, '_>,
    buf: &[u8],
) -> Result<()> {
    let required_len = interleaved_output_len(image)?;
    if buf.len() < required_len {
        bail!(DecodingError::OutputBufferTooSmall);
    }
    Ok(())
}

fn interleaved_output_len(image: &DecodedImage<'_, '_>) -> Result<usize> {
    let Some(first) = image.decoded_components.first() else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    first
        .container
        .truncated()
        .len()
        .checked_mul(image.decoded_components.len())
        .ok_or(ValidationError::ImageTooLarge.into())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel samples are rounded and intentionally quantized to the stable 8-bit output format"
)]
pub(crate) fn interleave_and_convert(
    image: &mut DecodedImage<'_, '_>,
    buf: &mut [u8],
) -> Result<()> {
    let components = &mut *image.decoded_components;
    let num_components = components.len();

    let mut all_same_bit_depth = Some(components[0].bit_depth);

    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    let max_len = components[0].container.truncated().len();

    let mut output_iter = buf.iter_mut();

    if all_same_bit_depth == Some(8) && num_components <= 4 {
        // Fast path for the common case.
        match num_components {
            // Gray-scale.
            1 => {
                for (output, input) in output_iter.zip(
                    components[0]
                        .container
                        .iter()
                        .map(|v| math::round_f32(*v) as u8),
                ) {
                    *output = input;
                }
            }
            // Gray-scale with alpha.
            2 => {
                let c0 = &components[0];
                let c1 = &components[1];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                }
            }
            // RGB
            3 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                }
            }
            // RGBA or CMYK.
            4 => {
                let c0 = &components[0];
                let c1 = &components[1];
                let c2 = &components[2];
                let c3 = &components[3];

                let c0 = &c0.container[..max_len];
                let c1 = &c1.container[..max_len];
                let c2 = &c2.container[..max_len];
                let c3 = &c3.container[..max_len];

                for i in 0..max_len {
                    *output_iter.next().unwrap() = math::round_f32(c0[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c1[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c2[i]) as u8;
                    *output_iter.next().unwrap() = math::round_f32(c3[i]) as u8;
                }
            }
            _ => bail!(ValidationError::TooManyChannels),
        }
    } else {
        // Slow path that also requires us to scale to 8 bit.
        let mul_factor = ((1 << 8) - 1) as f32;

        for sample in 0..max_len {
            for channel in components.iter() {
                *output_iter.next().unwrap() = math::round_f32(
                    (channel.container[sample]
                        / ((1_u64 << u32::from(channel.bit_depth)) - 1) as f32)
                        * mul_factor,
                ) as u8;
            }
        }
    }

    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "region samples use the same stable rounded 8-bit quantization as full-image decode"
)]
pub(crate) fn interleave_and_convert_region(
    image: &mut DecodedImage<'_, '_>,
    image_width: usize,
    roi: (u32, u32, u32, u32),
    buf: &mut [u8],
) {
    let components = &mut *image.decoded_components;
    let num_components = components.len();
    let (x, y, width, height) = roi;
    let mut output_iter = buf.iter_mut();

    let mut all_same_bit_depth = Some(components[0].bit_depth);
    for component in components.iter().skip(1) {
        if Some(component.bit_depth) != all_same_bit_depth {
            all_same_bit_depth = None;
        }
    }

    if all_same_bit_depth == Some(8) && num_components <= 4 {
        for row in y as usize..(y + height) as usize {
            let row_base = row * image_width;
            for col in x as usize..(x + width) as usize {
                let idx = row_base + col;
                for component in components.iter() {
                    *output_iter.next().unwrap() = math::round_f32(component.container[idx]) as u8;
                }
            }
        }
    } else {
        let mul_factor = ((1 << 8) - 1) as f32;
        for row in y as usize..(y + height) as usize {
            let row_base = row * image_width;
            for col in x as usize..(x + width) as usize {
                let idx = row_base + col;
                for component in components.iter() {
                    *output_iter.next().unwrap() = math::round_f32(
                        (component.container[idx]
                            / ((1_u64 << u32::from(component.bit_depth)) - 1) as f32)
                            * mul_factor,
                    ) as u8;
                }
            }
        }
    }
}

pub(crate) fn native_component_plane_dimensions(
    reference_dimensions: (u32, u32),
    sampling: (u8, u8),
    sample_count: usize,
) -> Result<(u32, u32)> {
    let reference_sample_count =
        checked_decode_sample_count(reference_dimensions.0, reference_dimensions.1)?;
    if sample_count == reference_sample_count {
        return Ok(reference_dimensions);
    }

    let (x_rsiz, y_rsiz) = sampling;
    if x_rsiz == 0 || y_rsiz == 0 {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    let sampled_dimensions = (
        reference_dimensions.0.div_ceil(u32::from(x_rsiz)),
        reference_dimensions.1.div_ceil(u32::from(y_rsiz)),
    );
    let sampled_sample_count =
        checked_decode_sample_count(sampled_dimensions.0, sampled_dimensions.1)?;
    if sample_count == sampled_sample_count {
        return Ok(sampled_dimensions);
    }

    bail!(DecodingError::CodeBlockDecodeFailure)
}

pub(crate) fn convert_color_space(image: &mut DecodedImage<'_, '_>, bit_depth: u8) -> Result<()> {
    if let Some(jp2::colr::ColorSpace::Enumerated(e)) = &image
        .boxes
        .primary_color_specification()
        .map(|i| &i.color_space)
    {
        match e {
            EnumeratedColorspace::Sycc => {
                dispatch!(Level::new(), simd => {
                    sycc_to_rgb(simd, image.decoded_components, bit_depth)
                })?;
            }
            EnumeratedColorspace::CieLab(cielab) => {
                dispatch!(Level::new(), simd => {
                    cielab_to_rgb(simd, image.decoded_components, bit_depth, cielab)
                })?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn get_color_space(
    boxes: &ImageBoxes,
    num_components: usize,
    retained_container_bytes: usize,
) -> Result<ColorSpace> {
    let cs = match boxes
        .primary_color_specification()
        .map_or(&jp2::colr::ColorSpace::Unknown, |specification| {
            &specification.color_space
        }) {
        jp2::colr::ColorSpace::Enumerated(e) => {
            match e {
                EnumeratedColorspace::Cmyk => ColorSpace::CMYK,
                EnumeratedColorspace::Srgb
                | EnumeratedColorspace::EsRgb
                | EnumeratedColorspace::Sycc => ColorSpace::RGB,
                EnumeratedColorspace::RommRgb => {
                    // Use an ICC profile to process the RommRGB color space.
                    ColorSpace::Icc {
                        profile: try_clone_color_profile(
                            include_bytes!("../assets/ProPhoto-v2-micro.icc"),
                            retained_container_bytes,
                        )?,
                        num_channels: 3,
                    }
                }
                EnumeratedColorspace::Greyscale => ColorSpace::Gray,
                EnumeratedColorspace::CieLab(_) => ColorSpace::Icc {
                    profile: try_clone_color_profile(
                        include_bytes!("../assets/LAB.icc"),
                        retained_container_bytes,
                    )?,
                    num_channels: 3,
                },
                _ => bail!(FormatError::Unsupported),
            }
        }
        jp2::colr::ColorSpace::Icc(icc) => {
            if let Some(metadata) = ICCMetadata::from_data(icc) {
                ColorSpace::Icc {
                    profile: try_clone_color_profile(icc, retained_container_bytes)?,
                    num_channels: u16::from(metadata.color_space.num_components()),
                }
            } else {
                // See OPENJPEG test orb-blue10-lin-jp2.jp2. They seem to
                // assume RGB in this case (even though the image has 4
                // components with no opacity channel, they assume RGBA instead
                // of CMYK).
                ColorSpace::RGB
            }
        }
        jp2::colr::ColorSpace::Unknown => match num_components {
            1 => ColorSpace::Gray,
            3 => ColorSpace::RGB,
            4 => ColorSpace::CMYK,
            _ => ColorSpace::Unknown {
                num_channels: u16::try_from(num_components).unwrap_or(u16::MAX),
            },
        },
    };

    Ok(cs)
}

fn try_clone_color_profile(profile: &[u8], retained_bytes: usize) -> Result<Vec<u8>> {
    checked_color_profile_peak(retained_bytes, profile.len(), DEFAULT_MAX_DECODE_BYTES)?;
    let mut cloned = Vec::new();
    try_reserve_decode_elements(&mut cloned, profile.len())?;
    checked_color_profile_peak(retained_bytes, cloned.capacity(), DEFAULT_MAX_DECODE_BYTES)?;
    cloned.extend_from_slice(profile);
    Ok(cloned)
}

fn checked_color_profile_peak(
    retained_bytes: usize,
    profile_bytes: usize,
    cap: usize,
) -> Result<usize> {
    let peak = retained_bytes
        .checked_add(profile_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    if peak > cap {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(peak)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Rust's saturating float-to-integer conversion is retained before rejecting negative indices"
)]
fn palette_index(sample: f32) -> Result<usize> {
    let rounded = math::round_f32(sample) as i64;
    usize::try_from(rounded).map_err(|_| ColorError::PaletteResolutionFailed.into())
}

fn sign_extend_palette_value(raw: u64, bit_depth: u8) -> i64 {
    if bit_depth == 0 {
        return raw.cast_signed();
    }
    if bit_depth >= 64 {
        return raw.cast_signed();
    }

    let mask = (1_u64 << bit_depth) - 1;
    let value = raw & mask;
    let shift = 64 - u32::from(bit_depth);
    (value << shift).cast_signed() >> shift
}

fn clamped_power_of_two_u32(exponent: u8) -> u32 {
    if u32::from(exponent) >= u32::BITS {
        u32::MAX
    } else {
        1_u32 << exponent
    }
}

fn clamped_add_u32(left: u32, right: u32) -> u32 {
    if right > u32::MAX - left {
        u32::MAX
    } else {
        left + right
    }
}

fn max_value_for_bit_depth(bit_depth: u8) -> u32 {
    if u32::from(bit_depth) >= u32::BITS {
        u32::MAX
    } else {
        (1_u32 << bit_depth) - 1
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "OpenJPEG-compatible CIE Lab scaling intentionally uses f32 arithmetic"
)]
#[inline]
pub(crate) fn cielab_to_rgb<S: Simd>(
    simd: S,
    components: &mut [ComponentData],
    bit_depth: u8,
    lab: &CieLab,
) -> Result<()> {
    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::LabConversionFailed)?;

    let [l, a, b] = head else {
        bail!(ColorError::LabConversionFailed);
    };

    let prec0 = l.bit_depth;
    let prec1 = a.bit_depth;
    let prec2 = b.bit_depth;

    // Prevent underflows/divisions by zero further below.
    if prec0 < 4 || prec1 < 4 || prec2 < 4 {
        bail!(ColorError::LabConversionFailed);
    }

    let rl = lab.rl.unwrap_or(100);
    let ra = lab.ra.unwrap_or(170);
    let rb = lab.rb.unwrap_or(200);
    let ol = lab.ol.unwrap_or(0);
    let a_shift = bit_depth
        .checked_sub(1)
        .ok_or(ColorError::LabConversionFailed)?;
    let b_high_shift = bit_depth
        .checked_sub(2)
        .ok_or(ColorError::LabConversionFailed)?;
    let b_low_shift = bit_depth
        .checked_sub(3)
        .ok_or(ColorError::LabConversionFailed)?;
    let default_a_offset = clamped_power_of_two_u32(a_shift);
    let default_b_offset = clamped_add_u32(
        clamped_power_of_two_u32(b_high_shift),
        clamped_power_of_two_u32(b_low_shift),
    );
    let oa = lab.oa.unwrap_or(default_a_offset);
    let ob = lab.ob.unwrap_or(default_b_offset);

    // Copied from OpenJPEG.
    let min_l = -(rl as f32 * ol as f32) / ((1_u64 << u32::from(prec0)) - 1) as f32;
    let max_l = min_l + rl as f32;
    let min_a = -(ra as f32 * oa as f32) / ((1_u64 << u32::from(prec1)) - 1) as f32;
    let max_a = min_a + ra as f32;
    let min_b = -(rb as f32 * ob as f32) / ((1_u64 << u32::from(prec2)) - 1) as f32;
    let max_b = min_b + rb as f32;

    let bit_max = max_value_for_bit_depth(bit_depth);

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    let divisor_l = ((1_u64 << u32::from(prec0)) - 1) as f32;
    let divisor_a = ((1_u64 << u32::from(prec1)) - 1) as f32;
    let divisor_b = ((1_u64 << u32::from(prec2)) - 1) as f32;

    let scale_l_final = bit_max as f32 / 100.0;
    let scale_ab_final = bit_max as f32 / 255.0;

    let l_offset = min_l * scale_l_final;
    let l_scale = (max_l - min_l) / divisor_l * scale_l_final;
    let a_offset = (min_a + 128.0) * scale_ab_final;
    let a_scale = (max_a - min_a) / divisor_a * scale_ab_final;
    let b_offset = (min_b + 128.0) * scale_ab_final;
    let b_scale = (max_b - min_b) / divisor_b * scale_ab_final;

    let l_offset_v = f32x8::splat(simd, l_offset);
    let l_scale_v = f32x8::splat(simd, l_scale);
    let a_offset_v = f32x8::splat(simd, a_offset);
    let a_scale_v = f32x8::splat(simd, a_scale);
    let b_offset_v = f32x8::splat(simd, b_offset);
    let b_scale_v = f32x8::splat(simd, b_scale);

    // Note that we are not doing the actual conversion with the ICC profile yet,
    // just decoding the raw LAB values.
    // We leave applying the ICC profile to the user.
    for ((l_chunk, a_chunk), b_chunk) in l
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(a.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(b.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let l_v = f32x8::from_slice(simd, l_chunk);
        let a_v = f32x8::from_slice(simd, a_chunk);
        let b_v = f32x8::from_slice(simd, b_chunk);

        l_v.mul_add(l_scale_v, l_offset_v).store(l_chunk);
        a_v.mul_add(a_scale_v, a_offset_v).store(a_chunk);
        b_v.mul_add(b_scale_v, b_offset_v).store(b_chunk);
    }

    // The color transform replaced the source samples. Any exact-integer
    // shadow describes the pre-transform component values and must not win
    // over the converted f32 planes during native output packing.
    l.integer_container = None;
    a.integer_container = None;
    b.integer_container = None;

    Ok(())
}

#[expect(
    clippy::cast_precision_loss,
    reason = "JPEG 2000 sYCC conversion intentionally uses f32 SIMD arithmetic"
)]
#[inline]
fn sycc_to_rgb<S: Simd>(simd: S, components: &mut [ComponentData], bit_depth: u8) -> Result<()> {
    let offset = (1_u64 << (u32::from(bit_depth) - 1)) as f32;
    let max_value = ((1_u64 << u32::from(bit_depth)) - 1) as f32;

    let (head, _) = components
        .split_at_mut_checked(3)
        .ok_or(ColorError::SyccConversionFailed)?;

    let [luma, blue_chroma, red_chroma] = head else {
        bail!(ColorError::SyccConversionFailed);
    };

    let offset_v = f32x8::splat(simd, offset);
    let max_v = f32x8::splat(simd, max_value);
    let zero_v = f32x8::splat(simd, 0.0);
    let red_chroma_to_red = f32x8::splat(simd, 1.402);
    let blue_chroma_to_green = f32x8::splat(simd, -0.344_136);
    let red_chroma_to_green = f32x8::splat(simd, -0.714_136);
    let blue_chroma_to_blue = f32x8::splat(simd, 1.772);

    for ((luma_chunk, blue_chroma_chunk), red_chroma_chunk) in luma
        .container
        .chunks_exact_mut(SIMD_WIDTH)
        .zip(blue_chroma.container.chunks_exact_mut(SIMD_WIDTH))
        .zip(red_chroma.container.chunks_exact_mut(SIMD_WIDTH))
    {
        let luma_values = f32x8::from_slice(simd, luma_chunk);
        let blue_chroma_values = f32x8::from_slice(simd, blue_chroma_chunk) - offset_v;
        let red_chroma_values = f32x8::from_slice(simd, red_chroma_chunk) - offset_v;

        // r = y + 1.402 * cr
        let red = red_chroma_values.mul_add(red_chroma_to_red, luma_values);
        // g = y - 0.344136 * cb - 0.714136 * cr
        let green = red_chroma_values.mul_add(
            red_chroma_to_green,
            blue_chroma_values.mul_add(blue_chroma_to_green, luma_values),
        );
        // b = y + 1.772 * cb
        let blue = blue_chroma_values.mul_add(blue_chroma_to_blue, luma_values);

        red.min(max_v).max(zero_v).store(luma_chunk);
        green.min(max_v).max(zero_v).store(blue_chroma_chunk);
        blue.min(max_v).max(zero_v).store(red_chroma_chunk);
    }

    luma.integer_container = None;
    blue_chroma.integer_container = None;
    red_chroma.integer_container = None;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        checked_color_profile_peak, clamped_add_u32, clamped_power_of_two_u32,
        max_value_for_bit_depth, palette_index, sycc_to_rgb, ColorSpace, ComponentPlane,
        DecodedComponents, DecodedNativeComponents, NativeComponentPlane,
    };
    use crate::j2c::ComponentData;
    use crate::math::{dispatch, Level, SimdBuffer, SIMD_WIDTH};
    use alloc::{vec, vec::Vec};
    use core::mem::size_of;

    #[test]
    fn lab_integer_scaling_preserves_clamped_boundaries() {
        assert_eq!(clamped_power_of_two_u32(31), 1_u32 << 31);
        assert_eq!(clamped_power_of_two_u32(32), u32::MAX);
        assert_eq!(clamped_add_u32(u32::MAX, 1), u32::MAX);
        assert_eq!(max_value_for_bit_depth(31), (1_u32 << 31) - 1);
        assert_eq!(max_value_for_bit_depth(32), u32::MAX);
    }

    #[test]
    fn sycc_conversion_discards_pretransform_integer_shadows() {
        let component = |value: u8| ComponentData {
            container: SimdBuffer::<SIMD_WIDTH>::new(vec![f32::from(value); SIMD_WIDTH]),
            integer_container: Some(vec![i64::from(value); SIMD_WIDTH]),
            bit_depth: 8,
            signed: false,
        };
        let mut components = vec![component(128), component(128), component(128)];

        dispatch!(Level::new(), simd => sycc_to_rgb(simd, &mut components, 8))
            .expect("sYCC conversion");

        assert!(
            components
                .iter()
                .all(|component| component.integer_container.is_none()),
            "native packing must not reuse pre-transform exact samples"
        );
    }

    #[test]
    fn retained_color_profile_peak_accepts_exact_cap_and_rejects_one_over() {
        assert_eq!(
            checked_color_profile_peak(7, 5, 12).expect("exact ICC clone peak"),
            12
        );
        assert!(checked_color_profile_peak(8, 5, 12).is_err());
    }

    #[test]
    fn palette_indices_reject_negative_samples_without_wrapping() {
        assert!(palette_index(-1.0).is_err());
        assert_eq!(palette_index(2.4).expect("valid palette index"), 2);
    }

    #[test]
    fn native_component_handoff_preserves_owned_capacities() {
        let mut data = Vec::with_capacity(9);
        data.push(3);
        let mut planes = Vec::with_capacity(4);
        planes.push(NativeComponentPlane {
            data,
            dimensions: (1, 1),
            bit_depth: 8,
            signed: false,
            sampling: (1, 1),
            bytes_per_sample: 1,
        });
        let mut profile = Vec::with_capacity(7);
        profile.push(1);
        let decoded = DecodedNativeComponents {
            dimensions: (1, 1),
            color_space: ColorSpace::Icc {
                profile,
                num_channels: 1,
            },
            has_alpha: false,
            planes,
        };
        let expected = decoded.planes.capacity() * size_of::<NativeComponentPlane>()
            + decoded.planes[0].data.capacity()
            + match &decoded.color_space {
                ColorSpace::Icc { profile, .. } => profile.capacity(),
                _ => 0,
            };
        let plane_owner_capacity = decoded.planes.capacity();
        let data_capacity = decoded.planes[0].data.capacity();
        let profile_capacity = match &decoded.color_space {
            ColorSpace::Icc { profile, .. } => profile.capacity(),
            _ => 0,
        };
        assert_eq!(decoded.allocated_bytes(), Some(expected));

        let (_, color_space, _, planes) = decoded.into_parts();
        assert_eq!(planes.capacity(), plane_owner_capacity);
        assert_eq!(planes[0].allocated_bytes(), data_capacity);
        assert!(matches!(
            color_space,
            ColorSpace::Icc { profile, .. } if profile.capacity() == profile_capacity
        ));
    }

    #[test]
    fn borrowed_component_handoff_preserves_metadata_capacities() {
        let samples = [2.0_f32];
        let mut planes = Vec::with_capacity(3);
        planes.push(ComponentPlane {
            samples: &samples,
            dimensions: (1, 1),
            bit_depth: 8,
            signed: false,
            sampling: (1, 1),
        });
        let mut profile = Vec::with_capacity(5);
        profile.push(1);
        let decoded = DecodedComponents {
            dimensions: (1, 1),
            color_space: ColorSpace::Icc {
                profile,
                num_channels: 1,
            },
            has_alpha: false,
            planes,
            live_bytes: 123,
        };
        let plane_owner_capacity = decoded.planes.capacity();
        let profile_capacity = match &decoded.color_space {
            ColorSpace::Icc { profile, .. } => profile.capacity(),
            _ => 0,
        };

        assert_eq!(decoded.live_bytes(), 123);
        let (_, color_space, _, planes) = decoded.into_parts();
        assert_eq!(planes.capacity(), plane_owner_capacity);
        assert!(core::ptr::eq(
            planes[0].samples().as_ptr(),
            samples.as_ptr()
        ));
        assert!(matches!(
            color_space,
            ColorSpace::Icc { profile, .. } if profile.capacity() == profile_capacity
        ));
    }
}
