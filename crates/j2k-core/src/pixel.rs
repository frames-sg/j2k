// SPDX-License-Identifier: MIT OR Apache-2.0

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

impl PixelLayout {
    /// Return the number of channels in this layout.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Rgb => 3,
            Self::Rgba => 4,
            Self::Gray => 1,
        }
    }
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
    /// Interleaved signed 16-bit RGB.
    RgbI16,
    /// Interleaved signed 16-bit RGBA.
    RgbaI16,
    /// Signed 16-bit grayscale.
    GrayI16,
}

impl PixelFormat {
    /// Return the channel layout for this pixel format.
    #[must_use]
    pub const fn layout(self) -> PixelLayout {
        match self {
            Self::Rgb8 | Self::Rgb16 | Self::RgbI16 => PixelLayout::Rgb,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaI16 => PixelLayout::Rgba,
            Self::Gray8 | Self::Gray16 | Self::GrayI16 => PixelLayout::Gray,
        }
    }

    /// Return the integer sample type for this pixel format.
    #[must_use]
    pub const fn sample(self) -> SampleType {
        match self {
            Self::Rgb8 | Self::Rgba8 | Self::Gray8 => SampleType::U8,
            Self::Rgb16 | Self::Rgba16 | Self::Gray16 => SampleType::U16,
            Self::RgbI16 | Self::RgbaI16 | Self::GrayI16 => SampleType::I16,
        }
    }

    /// Return the number of channels per pixel.
    #[must_use]
    pub const fn channels(self) -> usize {
        self.layout().channels()
    }

    /// Return the number of bytes in one channel sample.
    #[must_use]
    pub const fn bytes_per_sample(self) -> usize {
        match self.sample() {
            SampleType::U8 => 1,
            SampleType::U16 | SampleType::I16 => 2,
        }
    }

    /// Return the number of bytes in one interleaved pixel.
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::{PixelFormat, PixelLayout};
    use crate::SampleType;

    #[test]
    fn signed_sixteen_bit_formats_preserve_layout_and_size() {
        for (format, layout, channels) in [
            (PixelFormat::RgbI16, PixelLayout::Rgb, 3),
            (PixelFormat::RgbaI16, PixelLayout::Rgba, 4),
            (PixelFormat::GrayI16, PixelLayout::Gray, 1),
        ] {
            assert_eq!(format.layout(), layout);
            assert_eq!(format.sample(), SampleType::I16);
            assert_eq!(format.channels(), channels);
            assert_eq!(format.bytes_per_sample(), 2);
            assert_eq!(format.bytes_per_pixel(), channels * 2);
        }
    }
}
