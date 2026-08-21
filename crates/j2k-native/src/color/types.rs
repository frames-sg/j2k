// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public decoded color and packed bitmap value types.

use alloc::vec::Vec;

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
    /// Interleaved 8-bit pixel data, with alpha last when present.
    pub data: Vec<u8>,
    /// Whether the image has an alpha channel.
    pub has_alpha: bool,
    /// The width of the image.
    pub width: u32,
    /// The height of the image.
    pub height: u32,
    /// The original bit depth of the image.
    pub original_bit_depth: u8,
}

/// Raw decoded pixel data at native bit depth without 8-bit scaling.
///
/// Samples are interleaved. Samples above eight bits use little-endian packed
/// storage with [`Self::bytes_per_sample`] bytes per sample.
pub struct RawBitmap {
    /// The raw pixel data at native bit depth.
    pub data: Vec<u8>,
    /// The width of the image in pixels.
    pub width: u32,
    /// The height of the image in pixels.
    pub height: u32,
    /// The original bit depth per sample.
    pub bit_depth: u8,
    /// Whether every component in this packed bitmap is signed.
    pub signed: bool,
    /// Per-component signedness in codestream/component order.
    pub component_signed: Vec<bool>,
    /// The number of components.
    pub num_components: u16,
    /// Bytes per sample in the packed little-endian native representation.
    pub bytes_per_sample: u8,
}
