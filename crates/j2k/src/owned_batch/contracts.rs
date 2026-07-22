// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::sync::Arc;
use core::num::NonZeroUsize;

use j2k_core::{
    Colorspace, CompressedPayloadKind, CompressedTransferSyntax, Downscale, PixelFormat,
    PixelLayout, Rect, SampleType,
};

use super::{BatchExecutionShape, BatchItemError, PreparedImage};
use crate::{DecodeSettings, DeviceDecodeRequest};

/// Amount of native execution planning retained by a prepared image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PreparationDepth {
    /// Only inspection and geometry are retained. Decoding reparses native
    /// packet structure because the current general parser is borrowing.
    MetadataOnly,
    /// Per-tile Gray/RGB/RGBA HTJ2K packet and code-block geometry is retained;
    /// compressed payloads remain `(offset, length)` references into the
    /// original [`EncodedImage::bytes`] allocation.
    Htj2kOffsetPlan,
    /// Per-tile Gray/RGB/RGBA classic JPEG 2000 packet and code-block
    /// geometry is retained; compressed fragments remain byte ranges into the
    /// original [`EncodedImage::bytes`] allocation.
    ClassicOffsetPlan,
}

/// Requested pixels from one encoded image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DecodeRequest {
    /// Decode the complete image at its native resolution.
    #[default]
    Full,
    /// Decode a full-resolution source region.
    Region {
        /// Source-coordinate region of interest.
        roi: Rect,
    },
    /// Decode the complete image at a reduced resolution.
    Reduced {
        /// Power-of-two reduction.
        scale: Downscale,
    },
    /// Decode a source region at a reduced resolution.
    RegionReduced {
        /// Source-coordinate region of interest.
        roi: Rect,
        /// Power-of-two reduction.
        scale: Downscale,
    },
}

impl DecodeRequest {
    pub(super) fn device_request(self) -> DeviceDecodeRequest {
        match self {
            Self::Full => DeviceDecodeRequest::Full,
            Self::Region { roi } => DeviceDecodeRequest::Region { roi },
            Self::Reduced { scale } => DeviceDecodeRequest::Scaled { scale },
            Self::RegionReduced { roi, scale } => DeviceDecodeRequest::RegionScaled { roi, scale },
        }
    }
}

/// Tensor-compatible channel ordering requested from a batch decoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BatchLayout {
    /// Batch, channel, height, width.
    #[default]
    Nchw,
    /// Batch, height, width, channel.
    Nhwc,
}

/// One owned encoded image submitted to a batch decoder.
#[derive(Debug, Clone)]
pub struct EncodedImage {
    /// Complete J2K, JP2, or JPH bytes.
    pub bytes: Arc<[u8]>,
    /// Pixels requested from the image.
    pub request: DecodeRequest,
}

pub(super) struct PrepareJob {
    pub(super) source_index: usize,
    pub(super) input: Option<EncodedImage>,
}

pub(super) type PrepareImageResult =
    Result<(PreparedImage, BatchGroupInfo, BatchExecutionShape), BatchItemError>;

impl EncodedImage {
    /// Construct an owned image request without copying `bytes`.
    #[must_use]
    pub fn new(bytes: Arc<[u8]>, request: DecodeRequest) -> Self {
        Self { bytes, request }
    }

    /// Construct a full-resolution, full-image request.
    #[must_use]
    pub fn full(bytes: Arc<[u8]>) -> Self {
        Self::new(bytes, DecodeRequest::Full)
    }
}

/// Options shared by preparation and CPU batch decode.
#[derive(Debug, Clone, Copy)]
pub struct BatchDecodeOptions {
    /// Output channel ordering.
    pub layout: BatchLayout,
    /// Native decoder validation settings. The request controls target resolution.
    pub settings: DecodeSettings,
    /// CPU worker count. `None` uses available parallelism.
    pub workers: Option<NonZeroUsize>,
}

impl Default for BatchDecodeOptions {
    fn default() -> Self {
        Self {
            layout: BatchLayout::Nchw,
            settings: DecodeSettings::strict(),
            workers: None,
        }
    }
}

/// Interpretation of an output alpha channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BatchAlpha {
    /// The group has no alpha channel.
    None,
    /// Color samples are independent of straight opacity.
    Straight,
    /// Color samples are premultiplied by opacity.
    Premultiplied,
}

/// Compressed block-coding route selected during preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BatchCodecRoute {
    /// JPEG 2000 Part 1 block coding.
    Classic,
    /// JPEG 2000 Part 15 high-throughput block coding.
    Htj2k,
}

/// Wavelet transform declared by the compressed syntax.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BatchWaveletTransform {
    /// Reversible integer 5/3 transform.
    Reversible53,
    /// Irreversible floating-point 9/7 transform.
    Irreversible97,
}

/// Metadata shared by every image in one homogeneous output group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchGroupInfo {
    /// Decoded width and height.
    pub dimensions: (u32, u32),
    /// Number and interpretation of output channels.
    pub color: PixelLayout,
    /// Alpha interpretation declared by JP2 channel definitions.
    pub alpha: BatchAlpha,
    /// Exact significant sample precision.
    pub precision: u8,
    /// Whether the codestream samples are signed.
    pub signed: bool,
    /// Native Rust/tensor integer type.
    pub sample_type: SampleType,
    /// Requested output channel ordering.
    pub layout: BatchLayout,
    /// Parsed color-space declaration.
    pub colorspace: Colorspace,
    /// Classic or high-throughput block coding.
    pub route: BatchCodecRoute,
    /// Reversible or irreversible wavelet transform.
    pub transform: BatchWaveletTransform,
    /// Compressed transfer syntax.
    pub transfer_syntax: CompressedTransferSyntax,
    /// Raw codestream or still-image wrapper shape.
    pub payload_kind: CompressedPayloadKind,
}

impl BatchGroupInfo {
    /// Number of samples in one image in this group.
    #[must_use]
    pub fn samples_per_image(&self) -> Option<usize> {
        (self.dimensions.0 as usize)
            .checked_mul(self.dimensions.1 as usize)?
            .checked_mul(self.color.channels())
    }

    /// Exact native integer pixel format represented by this group.
    ///
    /// `None` reports a newer color/sample contract or metadata that does not
    /// follow the fast-batch width and signedness rules.
    #[doc(hidden)]
    #[must_use]
    pub const fn native_pixel_format(&self) -> Option<PixelFormat> {
        if !matches!(
            (self.sample_type, self.precision, self.signed),
            (SampleType::U8, 1..=8, false)
                | (SampleType::U16, 9..=16, false)
                | (SampleType::I16, 1..=16, true)
        ) {
            return None;
        }
        match (self.color, self.sample_type) {
            (PixelLayout::Gray, SampleType::U8) => Some(PixelFormat::Gray8),
            (PixelLayout::Gray, SampleType::U16) => Some(PixelFormat::Gray16),
            (PixelLayout::Gray, SampleType::I16) => Some(PixelFormat::GrayI16),
            (PixelLayout::Rgb, SampleType::U8) => Some(PixelFormat::Rgb8),
            (PixelLayout::Rgb, SampleType::U16) => Some(PixelFormat::Rgb16),
            (PixelLayout::Rgb, SampleType::I16) => Some(PixelFormat::RgbI16),
            (PixelLayout::Rgba, SampleType::U8) => Some(PixelFormat::Rgba8),
            (PixelLayout::Rgba, SampleType::U16) => Some(PixelFormat::Rgba16),
            (PixelLayout::Rgba, SampleType::I16) => Some(PixelFormat::RgbaI16),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use j2k_core::{
        Colorspace, CompressedPayloadKind, CompressedTransferSyntax, PixelFormat, PixelLayout,
        SampleType,
    };

    use super::{BatchAlpha, BatchCodecRoute, BatchGroupInfo, BatchLayout, BatchWaveletTransform};

    fn group_info(
        color: PixelLayout,
        sample_type: SampleType,
        precision: u8,
        signed: bool,
    ) -> BatchGroupInfo {
        BatchGroupInfo {
            dimensions: (8, 8),
            color,
            alpha: if color == PixelLayout::Rgba {
                BatchAlpha::Straight
            } else {
                BatchAlpha::None
            },
            precision,
            signed,
            sample_type,
            layout: BatchLayout::Nchw,
            colorspace: if color == PixelLayout::Gray {
                Colorspace::Grayscale
            } else {
                Colorspace::SRgb
            },
            route: BatchCodecRoute::Htj2k,
            transform: BatchWaveletTransform::Reversible53,
            transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
            payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
        }
    }

    #[test]
    fn native_pixel_format_maps_every_representable_color_and_sample_type() {
        for (color, sample_type, precision, signed, expected) in [
            (
                PixelLayout::Gray,
                SampleType::U8,
                8,
                false,
                PixelFormat::Gray8,
            ),
            (
                PixelLayout::Gray,
                SampleType::U16,
                12,
                false,
                PixelFormat::Gray16,
            ),
            (
                PixelLayout::Gray,
                SampleType::I16,
                12,
                true,
                PixelFormat::GrayI16,
            ),
            (
                PixelLayout::Rgb,
                SampleType::U8,
                8,
                false,
                PixelFormat::Rgb8,
            ),
            (
                PixelLayout::Rgb,
                SampleType::U16,
                12,
                false,
                PixelFormat::Rgb16,
            ),
            (
                PixelLayout::Rgb,
                SampleType::I16,
                12,
                true,
                PixelFormat::RgbI16,
            ),
            (
                PixelLayout::Rgba,
                SampleType::U8,
                8,
                false,
                PixelFormat::Rgba8,
            ),
            (
                PixelLayout::Rgba,
                SampleType::U16,
                12,
                false,
                PixelFormat::Rgba16,
            ),
            (
                PixelLayout::Rgba,
                SampleType::I16,
                12,
                true,
                PixelFormat::RgbaI16,
            ),
        ] {
            assert_eq!(
                group_info(color, sample_type, precision, signed).native_pixel_format(),
                Some(expected)
            );
        }
    }

    #[test]
    fn native_pixel_format_rejects_inconsistent_sample_metadata() {
        for info in [
            group_info(PixelLayout::Gray, SampleType::U8, 0, false),
            group_info(PixelLayout::Gray, SampleType::U8, 9, false),
            group_info(PixelLayout::Gray, SampleType::U8, 8, true),
            group_info(PixelLayout::Rgb, SampleType::U16, 8, false),
            group_info(PixelLayout::Rgb, SampleType::U16, 17, false),
            group_info(PixelLayout::Rgba, SampleType::I16, 12, false),
            group_info(PixelLayout::Rgba, SampleType::I16, 17, true),
        ] {
            assert_eq!(info.native_pixel_format(), None);
        }
    }
}
