// SPDX-License-Identifier: Apache-2.0

use crate::sample::SampleType;

/// Channel layout independent of sample width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelLayout {
    /// Three-channel red, green, blue layout.
    Rgb,
    /// Four-channel red, green, blue, alpha layout.
    Rgba,
    /// Single-channel grayscale layout.
    Gray,
}

/// Concrete interleaved pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// Interleaved 8-bit RGB.
    Rgb8,
    /// Interleaved 8-bit RGBA.
    Rgba8,
    /// 8-bit grayscale.
    Gray8,
    /// Interleaved 16-bit RGB.
    Rgb16,
    /// Interleaved 16-bit RGBA.
    Rgba16,
    /// 16-bit grayscale.
    Gray16,
}

impl PixelFormat {
    /// Return the channel layout for this pixel format.
    pub const fn layout(self) -> PixelLayout {
        match self {
            Self::Rgb8 | Self::Rgb16 => PixelLayout::Rgb,
            Self::Rgba8 | Self::Rgba16 => PixelLayout::Rgba,
            Self::Gray8 | Self::Gray16 => PixelLayout::Gray,
        }
    }

    /// Return the integer sample type for this pixel format.
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

    /// Return the number of bytes in one channel sample.
    pub const fn bytes_per_sample(self) -> usize {
        match self.sample() {
            SampleType::U8 => 1,
            SampleType::U16 => 2,
        }
    }

    /// Return the number of bytes in one interleaved pixel.
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_sample()
    }
}
