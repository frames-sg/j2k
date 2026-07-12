// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public output-buffer routing for full, scaled, and region decodes.

use super::{
    additional_decode_scratch_bytes, jpeg_profile_stages_enabled, output_format_from_parts,
    scaled_dimensions, scaled_rect_covering, validate_buffer, DecodeOutcome, Decoder, Downscale,
    DownscaleFactor, Instant, JpegError, LosslessRegionRequest, LosslessRgbRegionFallback,
    LosslessRgbaAlpha, OutputFormat, PixelFormat, Rect, ScratchPool, DEFAULT_SCRATCH,
};

mod dispatch;
mod owned_output;
mod profile;

#[cfg(test)]
mod tests;

use self::dispatch::OutputRoute;
use self::profile::DecodeProfileRecord;

impl Decoder<'_> {
    /// Decode the full image into the caller's buffer.
    ///
    /// # Errors
    /// - [`JpegError::OutputBufferTooSmall`] or [`JpegError::InvalidStride`]
    ///   if the provided buffer/stride cannot hold the image at `fmt`.
    /// - [`JpegError::NotImplemented`] if `fmt` requests a raw output the
    ///   current release does not emit (e.g. `RawYCbCr8`).
    /// - Any entropy- or structural-decode error from the scan walker.
    pub fn decode_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt))
    }

    /// Decode the full image into the caller's buffer using the core
    /// `PixelFormat` + `Downscale` contract.
    ///
    /// # Errors
    ///
    /// Returns an output-buffer, unsupported-format, or scan decode error.
    pub fn decode_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_scaled_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt, scale)
        })
    }

    /// Decode the full image into the caller's buffer, reusing the supplied
    /// [`ScratchPool`]. On a long-running tile batch this eliminates the
    /// per-tile allocation of stripe buffers, the DC predictor, and the
    /// chroma upsample rows — the realistic WSI reader shape. The first
    /// call against a fresh pool does the allocation; subsequent calls at
    /// the same-or-smaller shape reuse the underlying `Vec`s.
    ///
    /// # Errors
    /// Identical to [`Self::decode_into`].
    pub fn decode_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_scaled_into_with_scratch(pool, out, stride, fmt, Downscale::None)
    }

    pub(super) fn decode_lossless_output_format_region_scaled(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        roi: Rect,
        downscale: DownscaleFactor,
        external_live_bytes: usize,
    ) -> Option<Result<DecodeOutcome, JpegError>> {
        self.lossless_plan.as_ref()?;
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                LosslessRgbRegionFallback::for_color_space_8(self.info.color_space)
                    .decode_rgb_region_scaled_into(
                        self,
                        LosslessRegionRequest {
                            out,
                            stride,
                            roi,
                            downscale,
                            external_live_bytes,
                        },
                    )
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                LosslessRgbRegionFallback::for_color_space_8(self.info.color_space)
                    .decode_rgba_region_scaled_into(
                        self,
                        LosslessRegionRequest {
                            out,
                            stride,
                            roi,
                            downscale,
                            external_live_bytes,
                        },
                        LosslessRgbaAlpha::U8(alpha),
                    )
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => self
                .decode_lossless_gray8_region_scaled_into(
                    out,
                    stride,
                    roi,
                    downscale,
                    external_live_bytes,
                ),
            OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. } => self
                .decode_lossless_gray16_region_scaled_into(
                    out,
                    stride,
                    roi,
                    downscale,
                    external_live_bytes,
                ),
            OutputFormat::Rgb16 | OutputFormat::Rgb16Scaled { .. } => {
                LosslessRgbRegionFallback::for_color_space_16(self.info.color_space)
                    .decode_rgb_region_scaled_into(
                        self,
                        LosslessRegionRequest {
                            out,
                            stride,
                            roi,
                            downscale,
                            external_live_bytes,
                        },
                    )
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                LosslessRgbRegionFallback::for_color_space_16(self.info.color_space)
                    .decode_rgba_region_scaled_into(
                        self,
                        LosslessRegionRequest {
                            out,
                            stride,
                            roi,
                            downscale,
                            external_live_bytes,
                        },
                        LosslessRgbaAlpha::U16(alpha),
                    )
            }
        };
        Some(result)
    }

    pub(super) fn decode_into_output_format_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_into_output_format_with_scratch_and_external(pool, out, stride, fmt, 0)
    }

    fn decode_into_output_format_with_scratch_and_external(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        external_live_bytes: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        let downscale = fmt.downscale();
        let (w, h) = scaled_dimensions(self.info.dimensions, downscale);
        let full_roi = Rect::full(self.info.dimensions);
        let additional_scratch = additional_decode_scratch_bytes(
            self.info.sof_kind,
            self.info.dimensions,
            fmt,
            full_roi,
            Rect::full((w, h)),
            downscale,
        )?;
        let scratch_bytes = self.prepare_decode_workspace_with_additional(
            pool,
            external_live_bytes,
            additional_scratch,
        )?;
        let bpp = fmt.bytes_per_pixel();
        validate_buffer(out, stride, w, h, bpp)?;
        if let Some(result) = self.decode_lossless_output_format_region_scaled(
            out,
            stride,
            fmt,
            full_roi,
            downscale,
            external_live_bytes,
        ) {
            return result;
        }
        let decode_start = profile_enabled.then(Instant::now);
        let output_rect = Rect::full((w, h));
        let result = self.dispatch_full_output(OutputRoute {
            pool,
            out,
            stride,
            fmt,
            source_roi: full_roi,
            output_rect,
            downscale,
            external_live_bytes,
        });
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            DecodeProfileRecord {
                total_start,
                decode_start,
                source_dimensions: self.info.dimensions,
                output_rect,
                stride,
                bytes_per_pixel: bpp,
                scratch_bytes,
                fmt,
                downscale,
                source_roi: None,
            }
            .emit(outcome);
        }
        result
    }

    /// [`Self::decode_scaled_into`] with caller-owned scratch.
    ///
    /// # Errors
    ///
    /// Returns an output-buffer, unsupported-format, or scan decode error.
    pub fn decode_scaled_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_into_output_format_with_scratch(
            pool,
            out,
            stride,
            output_format_from_parts(self.info.sof_kind, fmt, scale)?,
        )
    }
    /// Decode a rectangular region of the image into the caller's buffer.
    ///
    /// `roi` is expressed in source-image coordinates. If `fmt` requests a
    /// downscaled output, the written pixels cover the corresponding bounding
    /// box in the scaled image grid.
    ///
    /// # Errors
    ///
    /// Returns an invalid-region, output-buffer, unsupported-format, or scan
    /// decode error.
    pub fn decode_region_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_into_with_scratch(&mut pool.borrow_mut(), out, stride, fmt, roi)
        })
    }

    /// [`Self::decode_region_into`] with caller-owned scratch.
    pub(crate) fn decode_region_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi, Downscale::None)
    }

    pub(super) fn decode_region_into_output_format_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_into_output_format_with_scratch_and_external(
            pool, out, stride, fmt, roi, 0,
        )
    }

    fn decode_region_into_output_format_with_scratch_and_external(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: OutputFormat,
        roi: Rect,
        external_live_bytes: usize,
    ) -> Result<DecodeOutcome, JpegError> {
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        if !roi.is_within(self.info.dimensions) {
            return Err(JpegError::RectOutOfBounds {
                rect: roi,
                width: self.info.dimensions.0,
                height: self.info.dimensions.1,
            });
        }

        if roi == Rect::full(self.info.dimensions) {
            return self.decode_into_output_format_with_scratch_and_external(
                pool,
                out,
                stride,
                fmt,
                external_live_bytes,
            );
        }

        let downscale = fmt.downscale();
        let scaled_roi = scaled_rect_covering(roi, downscale)?;
        let additional_scratch = additional_decode_scratch_bytes(
            self.info.sof_kind,
            self.info.dimensions,
            fmt,
            roi,
            scaled_roi,
            downscale,
        )?;
        let scratch_bytes = self.prepare_decode_workspace_with_additional(
            pool,
            external_live_bytes,
            additional_scratch,
        )?;
        validate_buffer(
            out,
            stride,
            scaled_roi.w,
            scaled_roi.h,
            fmt.bytes_per_pixel(),
        )?;
        if let Some(result) = self.decode_lossless_output_format_region_scaled(
            out,
            stride,
            fmt,
            roi,
            downscale,
            external_live_bytes,
        ) {
            return result;
        }

        let decode_start = profile_enabled.then(Instant::now);
        let result = self.dispatch_region_output(OutputRoute {
            pool,
            out,
            stride,
            fmt,
            source_roi: roi,
            output_rect: scaled_roi,
            downscale,
            external_live_bytes,
        });
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            DecodeProfileRecord {
                total_start,
                decode_start,
                source_dimensions: self.info.dimensions,
                output_rect: scaled_roi,
                stride,
                bytes_per_pixel: fmt.bytes_per_pixel(),
                scratch_bytes,
                fmt,
                downscale,
                source_roi: Some(roi),
            }
            .emit(outcome);
        }
        result
    }

    /// Decode `roi` into the caller's buffer using the core `PixelFormat` +
    /// `Downscale` contract.
    ///
    /// # Errors
    ///
    /// Returns an invalid-region, output-buffer, unsupported-format, or scan
    /// decode error.
    pub fn decode_region_scaled_into(
        &self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        DEFAULT_SCRATCH.with(|pool| {
            self.decode_region_scaled_into_with_scratch(
                &mut pool.borrow_mut(),
                out,
                stride,
                fmt,
                roi,
                scale,
            )
        })
    }

    /// [`Self::decode_region_scaled_into`] with caller-owned scratch.
    ///
    /// # Errors
    ///
    /// Returns an invalid-region, output-buffer, unsupported-format, or scan
    /// decode error.
    pub fn decode_region_scaled_into_with_scratch(
        &self,
        pool: &mut ScratchPool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome, JpegError> {
        self.decode_region_into_output_format_with_scratch(
            pool,
            out,
            stride,
            output_format_from_parts(self.info.sof_kind, fmt, scale)?,
            roi,
        )
    }
}
