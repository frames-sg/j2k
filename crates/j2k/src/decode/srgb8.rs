// SPDX-License-Identifier: MIT OR Apache-2.0

//! Explicit 8-bit sRGB output normalization.

use crate::{backend::ColorSpace, backend::Image, J2kError};
use alloc::vec::Vec;
use j2k_core::{
    ensure_allocation_within_cap, try_host_vec_filled, BufferError, HostAllocationError,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
use moxcms::{CmsError, ColorProfile, DataColorSpace, Layout, ParsingOptions, TransformOptions};

const SRGB8_OUTPUT_WHAT: &str = "J2K sRGB8 normalized output";

/// Interleaved sample layout returned by [`J2kSrgb8Image`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum J2kSrgb8Layout {
    /// One 8-bit sRGB-grey sample per pixel.
    Gray,
    /// Three interleaved 8-bit sRGB samples per pixel.
    Rgb,
    /// Three interleaved 8-bit sRGB samples followed by alpha per pixel.
    Rgba,
}

impl J2kSrgb8Layout {
    const fn channels(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// Owned, tightly packed 8-bit sRGB decode result.
#[derive(Debug, PartialEq, Eq)]
pub struct J2kSrgb8Image {
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
    data: Vec<u8>,
}

impl J2kSrgb8Image {
    /// Decoded image dimensions.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Interleaved sample layout.
    #[must_use]
    pub const fn layout(&self) -> J2kSrgb8Layout {
        self.layout
    }

    /// Tightly packed 8-bit image samples.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume the image and return its tightly packed sample storage.
    #[must_use]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

pub(crate) fn decode_image_srgb8<'a>(
    image: &Image<'a>,
    native_context: &mut j2k_native::DecoderContext<'a>,
) -> Result<J2kSrgb8Image, J2kError> {
    let retained_image_bytes = image
        .retained_allocation_bytes()
        .map_err(J2kError::from_native_decode_error)?;
    let primary_icc = image.primary_icc_profile();
    let bitmap = image
        .decode_with_context(native_context)
        .map_err(J2kError::from_native_decode_error)?;
    let dimensions = (bitmap.width, bitmap.height);
    let has_alpha = bitmap.has_alpha;
    let bitmap_profile_bytes = match &bitmap.color_space {
        ColorSpace::Icc { profile, .. } => profile.capacity(),
        _ => 0,
    };
    let retained_bytes = checked_peak_bytes(
        retained_image_bytes,
        bitmap.data.capacity(),
        bitmap_profile_bytes,
    )?;

    if let Some(profile) = primary_icc {
        return convert_icc_data(&bitmap.data, profile, dimensions, has_alpha, retained_bytes);
    }

    match bitmap.color_space {
        ColorSpace::Gray if has_alpha => {
            expand_gray_alpha(&bitmap.data, dimensions, retained_bytes)
        }
        ColorSpace::Gray => finish(bitmap.data, dimensions, J2kSrgb8Layout::Gray),
        ColorSpace::RGB => finish(
            bitmap.data,
            dimensions,
            if has_alpha {
                J2kSrgb8Layout::Rgba
            } else {
                J2kSrgb8Layout::Rgb
            },
        ),
        ColorSpace::Icc { profile, .. } => convert_icc_data(
            &bitmap.data,
            &profile,
            dimensions,
            has_alpha,
            retained_bytes,
        ),
        ColorSpace::CMYK | ColorSpace::Unknown { .. } => Err(j2k_core::Unsupported {
            what: "decode_srgb8 requires grayscale, RGB, or a restricted ICC input profile",
        }
        .into()),
    }
}

fn convert_icc_data(
    source: &[u8],
    profile: &[u8],
    dimensions: (u32, u32),
    has_alpha: bool,
    retained_bytes: usize,
) -> Result<J2kSrgb8Image, J2kError> {
    let max_profile_size = profile
        .len()
        .checked_add(1)
        .ok_or_else(output_size_overflow)?;
    let source_profile = ColorProfile::new_from_slice_with_options(
        profile,
        ParsingOptions {
            max_profile_size,
            max_allowed_clut_size: 0,
            max_allowed_trc_size: 65_536,
        },
    )
    .map_err(|error| map_profile_error(&error))?;
    reject_non_restricted_profile(&source_profile)?;

    match source_profile.color_space {
        DataColorSpace::Rgb => transform_rgb(
            source,
            dimensions,
            has_alpha,
            retained_bytes,
            &source_profile,
        ),
        DataColorSpace::Gray => transform_gray(
            source,
            dimensions,
            has_alpha,
            retained_bytes,
            &source_profile,
        ),
        _ => Err(j2k_core::Unsupported {
            what: "decode_srgb8 supports restricted ICC RGB and monochrome input profiles",
        }
        .into()),
    }
}

fn reject_non_restricted_profile(profile: &ColorProfile) -> Result<(), J2kError> {
    if profile.lut_a_to_b_perceptual.is_some()
        || profile.lut_a_to_b_colorimetric.is_some()
        || profile.lut_a_to_b_saturation.is_some()
        || profile.lut_b_to_a_perceptual.is_some()
        || profile.lut_b_to_a_colorimetric.is_some()
        || profile.lut_b_to_a_saturation.is_some()
        || profile.gamut.is_some()
    {
        return Err(j2k_core::Unsupported {
            what: "decode_srgb8 supports restricted matrix/TRC ICC profiles only",
        }
        .into());
    }
    Ok(())
}

fn transform_rgb(
    source: &[u8],
    dimensions: (u32, u32),
    has_alpha: bool,
    retained_bytes: usize,
    source_profile: &ColorProfile,
) -> Result<J2kSrgb8Image, J2kError> {
    let cms_layout = if has_alpha { Layout::Rgba } else { Layout::Rgb };
    let destination = ColorProfile::new_srgb();
    transform_icc(
        source,
        dimensions,
        retained_bytes,
        source_profile,
        cms_layout,
        &destination,
        cms_layout,
    )
}

fn transform_gray(
    source: &[u8],
    dimensions: (u32, u32),
    has_alpha: bool,
    retained_bytes: usize,
    source_profile: &ColorProfile,
) -> Result<J2kSrgb8Image, J2kError> {
    if has_alpha {
        let destination = ColorProfile::new_srgb();
        return transform_icc(
            source,
            dimensions,
            retained_bytes,
            source_profile,
            Layout::GrayAlpha,
            &destination,
            Layout::Rgba,
        );
    }

    let mut destination = ColorProfile::new_gray_with_gamma(1.0);
    destination.gray_trc = ColorProfile::new_srgb().red_trc;
    transform_icc(
        source,
        dimensions,
        retained_bytes,
        source_profile,
        Layout::Gray,
        &destination,
        Layout::Gray,
    )
}

fn transform_icc(
    source: &[u8],
    dimensions: (u32, u32),
    retained_bytes: usize,
    source_profile: &ColorProfile,
    source_layout: Layout,
    destination_profile: &ColorProfile,
    destination_layout: Layout,
) -> Result<J2kSrgb8Image, J2kError> {
    let output_layout = match destination_layout {
        Layout::Gray => J2kSrgb8Layout::Gray,
        Layout::Rgb => J2kSrgb8Layout::Rgb,
        Layout::Rgba => J2kSrgb8Layout::Rgba,
        _ => {
            return Err(J2kError::InternalInvariant {
                what: "restricted ICC transform has a non-sRGB destination layout",
            });
        }
    };
    let mut output = allocate_output(dimensions, output_layout, retained_bytes)?;
    let transform = source_profile
        .create_transform_8bit(
            source_layout,
            destination_profile,
            destination_layout,
            TransformOptions::default(),
        )
        .map_err(|error| map_transform_error(&error))?;
    transform
        .transform(source, &mut output)
        .map_err(|error| map_transform_error(&error))?;
    finish(output, dimensions, output_layout)
}

fn expand_gray_alpha(
    source: &[u8],
    dimensions: (u32, u32),
    retained_bytes: usize,
) -> Result<J2kSrgb8Image, J2kError> {
    let mut output = allocate_output(dimensions, J2kSrgb8Layout::Rgba, retained_bytes)?;
    for (input, pixel) in source.chunks_exact(2).zip(output.chunks_exact_mut(4)) {
        pixel.copy_from_slice(&[input[0], input[0], input[0], input[1]]);
    }
    finish(output, dimensions, J2kSrgb8Layout::Rgba)
}

fn allocate_output(
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
    retained_bytes: usize,
) -> Result<Vec<u8>, J2kError> {
    let len = expected_len(dimensions, layout)?;
    checked_peak_bytes(retained_bytes, len, 0)?;
    let output = try_host_vec_filled(len, 0_u8).map_err(host_allocation_error)?;
    checked_peak_bytes(retained_bytes, output.capacity(), 0)?;
    Ok(output)
}

fn finish(
    data: Vec<u8>,
    dimensions: (u32, u32),
    layout: J2kSrgb8Layout,
) -> Result<J2kSrgb8Image, J2kError> {
    let expected = expected_len(dimensions, layout)?;
    if data.len() != expected {
        return Err(J2kError::InternalInvariant {
            what: "normalized sRGB output length does not match its layout",
        });
    }
    Ok(J2kSrgb8Image {
        dimensions,
        layout,
        data,
    })
}

fn expected_len(dimensions: (u32, u32), layout: J2kSrgb8Layout) -> Result<usize, J2kError> {
    (dimensions.0 as usize)
        .checked_mul(dimensions.1 as usize)
        .and_then(|pixels| pixels.checked_mul(layout.channels()))
        .ok_or_else(output_size_overflow)
}

fn checked_peak_bytes(first: usize, second: usize, third: usize) -> Result<usize, J2kError> {
    let requested = first
        .checked_add(second)
        .and_then(|bytes| bytes.checked_add(third))
        .ok_or_else(output_size_overflow)?;
    ensure_allocation_within_cap(
        requested,
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        SRGB8_OUTPUT_WHAT,
    )
    .map_err(Into::into)
}

fn output_size_overflow() -> J2kError {
    BufferError::SizeOverflow {
        what: SRGB8_OUTPUT_WHAT,
    }
    .into()
}

fn host_allocation_error(error: HostAllocationError) -> J2kError {
    BufferError::HostAllocationFailed {
        bytes: error.requested_bytes(),
        what: SRGB8_OUTPUT_WHAT,
    }
    .into()
}

fn map_profile_error(error: &CmsError) -> J2kError {
    match error {
        CmsError::OutOfMemory(bytes) => BufferError::HostAllocationFailed {
            bytes: *bytes,
            what: "restricted ICC profile",
        }
        .into(),
        _ => J2kError::InvalidIccProfile,
    }
}

fn map_transform_error(error: &CmsError) -> J2kError {
    match error {
        CmsError::OutOfMemory(bytes) => BufferError::HostAllocationFailed {
            bytes: *bytes,
            what: "restricted ICC transform",
        }
        .into(),
        _ => j2k_core::Unsupported {
            what: "restricted ICC profile cannot be transformed to sRGB",
        }
        .into(),
    }
}
