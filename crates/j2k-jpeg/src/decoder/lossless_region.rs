// SPDX-License-Identifier: MIT OR Apache-2.0

use super::color_convert::{
    copy_rgb16_scaled_rect, copy_rgb16_to_rgba16, copy_rgb8_scaled_rect, copy_rgb8_to_rgba8,
};
use super::output_format::{allocate_output_buffer, scaled_rect_covering};
use super::scratch::checked_scratch_len;
use super::{DecodeOutcome, Decoder};
use crate::error::JpegError;
use crate::info::{ColorSpace, DownscaleFactor, Rect};

#[derive(Clone, Copy)]
pub(super) enum LosslessRgbRegionFallback {
    Rgb8,
    YCbCr8,
    Rgb16,
    YCbCr16,
}

#[derive(Clone, Copy)]
pub(super) enum LosslessRgbaAlpha {
    U8(u8),
    U16(u16),
}

impl LosslessRgbRegionFallback {
    pub(super) fn for_color_space_8(color_space: ColorSpace) -> Self {
        match color_space {
            ColorSpace::YCbCr => Self::YCbCr8,
            _ => Self::Rgb8,
        }
    }

    pub(super) fn for_color_space_16(color_space: ColorSpace) -> Self {
        match color_space {
            ColorSpace::YCbCr => Self::YCbCr16,
            _ => Self::Rgb16,
        }
    }

    pub(super) const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::YCbCr8 => 3,
            Self::Rgb16 | Self::YCbCr16 => 6,
        }
    }

    pub(super) fn decode_full(
        self,
        decoder: &Decoder<'_>,
        out: &mut [u8],
        stride: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        match self {
            Self::Rgb8 => decoder.decode_lossless_rgb8_into(out, stride),
            Self::YCbCr8 => decoder.decode_lossless_ycbcr8_into(out, stride),
            Self::Rgb16 => decoder.decode_lossless_rgb16_into(out, stride),
            Self::YCbCr16 => decoder.decode_lossless_ycbcr16_into(out, stride),
        }
    }

    pub(super) fn copy_scaled(
        self,
        full: &[u8],
        dimensions: (u32, u32),
        output_rect: Rect,
        downscale: u32,
        out: &mut [u8],
        stride: usize,
    ) {
        match self {
            Self::Rgb8 | Self::YCbCr8 => {
                copy_rgb8_scaled_rect(full, dimensions, output_rect, downscale, out, stride);
            }
            Self::Rgb16 | Self::YCbCr16 => {
                copy_rgb16_scaled_rect(full, dimensions, output_rect, downscale, out, stride);
            }
        }
    }

    pub(super) fn decode_rgb_region_scaled_into(
        self,
        decoder: &Decoder<'_>,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
    ) -> Result<DecodeOutcome, JpegError> {
        if roi == Rect::full(decoder.info.dimensions) && downscale == DownscaleFactor::Full {
            return self.decode_full(decoder, out, stride);
        }

        let (width, height) = decoder.info.dimensions;
        let full_stride = checked_scratch_len(&[width as usize, self.bytes_per_pixel()])?;
        let mut full =
            allocate_output_buffer(checked_scratch_len(&[full_stride, height as usize])?);
        let mut outcome = self.decode_full(decoder, &mut full, full_stride)?;
        let output_rect = scaled_rect_covering(roi, downscale)?;
        self.copy_scaled(
            &full,
            (width, height),
            output_rect,
            downscale.denominator(),
            out,
            stride,
        );
        outcome.decoded = roi;
        Ok(outcome)
    }

    pub(super) fn decode_rgba_region_scaled_into(
        self,
        decoder: &Decoder<'_>,
        out: &mut [u8],
        stride: usize,
        roi: Rect,
        downscale: DownscaleFactor,
        alpha: LosslessRgbaAlpha,
    ) -> Result<DecodeOutcome, JpegError> {
        let output_rect = scaled_rect_covering(roi, downscale)?;
        let rgb_stride = checked_scratch_len(&[output_rect.w as usize, self.bytes_per_pixel()])?;
        let mut rgb =
            allocate_output_buffer(checked_scratch_len(&[rgb_stride, output_rect.h as usize])?);
        let outcome =
            self.decode_rgb_region_scaled_into(decoder, &mut rgb, rgb_stride, roi, downscale)?;
        match (self, alpha) {
            (Self::Rgb8 | Self::YCbCr8, LosslessRgbaAlpha::U8(alpha)) => {
                copy_rgb8_to_rgba8(
                    &rgb,
                    rgb_stride,
                    output_rect.w,
                    output_rect.h,
                    out,
                    stride,
                    alpha,
                );
            }
            (Self::Rgb16 | Self::YCbCr16, LosslessRgbaAlpha::U16(alpha)) => {
                copy_rgb16_to_rgba16(
                    &rgb,
                    rgb_stride,
                    output_rect.w,
                    output_rect.h,
                    out,
                    stride,
                    alpha,
                );
            }
            _ => {
                return Err(JpegError::InternalInvariant {
                    reason: "lossless RGBA fallback bit depth mismatch",
                });
            }
        }
        Ok(outcome)
    }
}
