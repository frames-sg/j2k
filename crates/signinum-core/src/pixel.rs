// SPDX-License-Identifier: Apache-2.0

use crate::sample::SampleType;

/// Logical channel layout for a pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelLayout {
    /// Red, green, blue channels.
    Rgb,
    /// Red, green, blue, alpha channels.
    Rgba,
    /// Single luminance channel.
    Gray,
}

/// Pixel storage format used by public decode APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// Interleaved 8-bit RGB.
    Rgb8,
    /// Interleaved 8-bit RGBA.
    Rgba8,
    /// 8-bit grayscale.
    Gray8,
    /// Interleaved 16-bit RGB, native endian in caller-owned buffers.
    Rgb16,
    /// Interleaved 16-bit RGBA, native endian in caller-owned buffers.
    Rgba16,
    /// 16-bit grayscale, native endian in caller-owned buffers.
    Gray16,
}

impl PixelFormat {
    /// Return the logical channel layout.
    pub const fn layout(self) -> PixelLayout {
        match self {
            Self::Rgb8 | Self::Rgb16 => PixelLayout::Rgb,
            Self::Rgba8 | Self::Rgba16 => PixelLayout::Rgba,
            Self::Gray8 | Self::Gray16 => PixelLayout::Gray,
        }
    }

    /// Return the sample type used by each channel.
    pub const fn sample(self) -> SampleType {
        match self {
            Self::Rgb8 | Self::Rgba8 | Self::Gray8 => SampleType::U8,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 => SampleType::U16,
        }
    }

    /// Return the number of channels per pixel.
    pub const fn channels(self) -> usize {
        match self.layout() {
            PixelLayout::Rgb => 3,
            PixelLayout::Rgba => 4,
            PixelLayout::Gray => 1,
        }
    }

    /// Return bytes occupied by one channel sample.
    pub const fn bytes_per_sample(self) -> usize {
        match self.sample() {
            SampleType::U8 => 1,
            SampleType::U16 => 2,
        }
    }

    /// Return bytes occupied by one interleaved pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_sample()
    }
}
