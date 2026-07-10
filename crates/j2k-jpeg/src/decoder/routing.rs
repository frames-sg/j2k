// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public output-buffer routing for full, scaled, and region decodes.

use super::duration_us_string;
use super::{
    allocate_output_buffer, checked_output_geometry, decode_scan_fast_tile_rgb_region,
    decode_scan_fast_tile_rgb_region_scaled, downscale_profile_name, emit_jpeg_profile_row,
    fast_tile_region_first_decode_mcu, jpeg_profile_stages_enabled, merged_warnings,
    output_format_from_parts, output_format_profile_name, scaled_dimensions, scaled_rect_covering,
    validate_buffer, CroppedWriter, DecodeOutcome, DecodeRequest, Decoder, Downscale,
    DownscaleFactor, FastTileRegionScaledRequest, Gray8Writer, Instant, JpegError,
    LosslessRgbRegionFallback, LosslessRgbaAlpha, OutputFormat, PixelFormat, Rect, Rgb8Writer,
    Rgba8Writer, ScratchPool, SofKind, Vec, DEFAULT_MAX_DECODE_BYTES, DEFAULT_SCRATCH,
};

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

    /// Decode into a freshly allocated tightly packed buffer using a request
    /// object instead of a method-name cross-product.
    pub fn decode_request(
        &self,
        request: DecodeRequest,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        DEFAULT_SCRATCH
            .with(|pool| self.decode_request_with_scratch(&mut pool.borrow_mut(), request))
    }

    /// Decode the full image into the caller's buffer using the core
    /// `PixelFormat` + `Downscale` contract.
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

    pub(super) fn decode_request_with_scratch(
        &self,
        pool: &mut ScratchPool,
        request: DecodeRequest,
    ) -> Result<(Vec<u8>, DecodeOutcome), JpegError> {
        let legacy = output_format_from_parts(self.info.sof_kind, request.fmt, request.scale)?;
        let (stride, len) = if let Some(roi) = request.region {
            let scaled_roi = scaled_rect_covering(roi, legacy.downscale())?;
            checked_output_geometry(scaled_roi.w, scaled_roi.h, legacy.bytes_per_pixel())?
        } else {
            let (width, height) = scaled_dimensions(self.info.dimensions, legacy.downscale());
            checked_output_geometry(width, height, legacy.bytes_per_pixel())?
        };
        let mut out = allocate_output_buffer(len);
        let outcome = if let Some(roi) = request.region {
            self.decode_region_scaled_into_with_scratch(
                pool,
                &mut out,
                stride,
                request.fmt,
                roi,
                request.scale,
            )?
        } else {
            self.decode_scaled_into_with_scratch(
                pool,
                &mut out,
                stride,
                request.fmt,
                request.scale,
            )?
        };
        Ok((out, outcome))
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
    ) -> Option<Result<DecodeOutcome, JpegError>> {
        self.lossless_plan.as_ref()?;
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                LosslessRgbRegionFallback::for_color_space_8(self.info.color_space)
                    .decode_rgb_region_scaled_into(self, out, stride, roi, downscale)
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                LosslessRgbRegionFallback::for_color_space_8(self.info.color_space)
                    .decode_rgba_region_scaled_into(
                        self,
                        out,
                        stride,
                        roi,
                        downscale,
                        LosslessRgbaAlpha::U8(alpha),
                    )
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                self.decode_lossless_gray8_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. } => {
                self.decode_lossless_gray16_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Rgb16 | OutputFormat::Rgb16Scaled { .. } => {
                LosslessRgbRegionFallback::for_color_space_16(self.info.color_space)
                    .decode_rgb_region_scaled_into(self, out, stride, roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                LosslessRgbRegionFallback::for_color_space_16(self.info.color_space)
                    .decode_rgba_region_scaled_into(
                        self,
                        out,
                        stride,
                        roi,
                        downscale,
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
        let profile_enabled = jpeg_profile_stages_enabled();
        let total_start = profile_enabled.then(Instant::now);
        let downscale = fmt.downscale();
        let (w, h) = scaled_dimensions(self.info.dimensions, downscale);
        let scratch_bytes = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let bpp = fmt.bytes_per_pixel();
        validate_buffer(out, stride, w, h, bpp)?;
        let full_roi = Rect::full(self.info.dimensions);
        if let Some(result) =
            self.decode_lossless_output_format_region_scaled(out, stride, fmt, full_roi, downscale)
        {
            return result;
        }
        let decode_start = profile_enabled.then(Instant::now);
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                let mut writer = Rgb8Writer::new_with_backend(out, stride, w, self.backend);
                self.decode_rgb_with_writer(pool, &mut writer, downscale, full_roi)
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                let mut writer = Rgba8Writer::new_with_backend(out, stride, w, alpha, self.backend);
                self.decode_with_writer(pool, &mut writer, downscale, full_roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let mut writer = Gray8Writer::new(out, stride, w);
                self.decode_with_writer(pool, &mut writer, downscale, full_roi)
            }
            OutputFormat::Gray16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_gray16_into(out, stride)
            }
            OutputFormat::Gray16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_gray16_region_scaled_into(out, stride, full_roi, downscale)
            }
            OutputFormat::Rgb16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_into(out, stride)
            }
            OutputFormat::Rgb16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, full_roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_region_scaled_into(out, stride, full_roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                if matches!(
                    self.info.sof_kind,
                    SofKind::Extended12 | SofKind::Progressive12
                ) {
                    return self.decode_12bit_rgba16_region_scaled_into(
                        out, stride, full_roi, downscale, alpha,
                    );
                }
                Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        };
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            let source_width_s = self.info.dimensions.0.to_string();
            let source_height_s = self.info.dimensions.1.to_string();
            let output_width_s = w.to_string();
            let output_height_s = h.to_string();
            let stride_s = stride.to_string();
            let bpp_s = bpp.to_string();
            let output_bytes_s = stride.saturating_mul(h as usize).to_string();
            let scratch_bytes_s = scratch_bytes.to_string();
            let warning_count_s = outcome.warnings.len().to_string();
            let decode_us = duration_us_string(decode_start.elapsed());
            let total_us = duration_us_string(total_start.elapsed());
            emit_jpeg_profile_row(
                "decode",
                "cpu",
                &[
                    ("mode", "full"),
                    ("fmt", output_format_profile_name(fmt)),
                    ("downscale", downscale_profile_name(downscale)),
                    ("source_width", source_width_s.as_str()),
                    ("source_height", source_height_s.as_str()),
                    ("output_width", output_width_s.as_str()),
                    ("output_height", output_height_s.as_str()),
                    ("stride", stride_s.as_str()),
                    ("bpp", bpp_s.as_str()),
                    ("scratch_bytes", scratch_bytes_s.as_str()),
                    ("output_bytes", output_bytes_s.as_str()),
                    ("decode_us", decode_us.as_str()),
                    ("total_us", total_us.as_str()),
                    ("warnings", warning_count_s.as_str()),
                ],
            );
        }
        result
    }

    /// [`Self::decode_scaled_into`] with caller-owned scratch.
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
            return self.decode_into_output_format_with_scratch(pool, out, stride, fmt);
        }

        let downscale = fmt.downscale();
        let scaled_roi = scaled_rect_covering(roi, downscale)?;
        let scratch_bytes = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        validate_buffer(
            out,
            stride,
            scaled_roi.w,
            scaled_roi.h,
            fmt.bytes_per_pixel(),
        )?;
        if let Some(result) =
            self.decode_lossless_output_format_region_scaled(out, stride, fmt, roi, downscale)
        {
            return result;
        }

        let decode_start = profile_enabled.then(Instant::now);
        let result = match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                if fmt == OutputFormat::Rgb8
                    && downscale == DownscaleFactor::Full
                    && self.progressive_plan.is_none()
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer =
                        Rgb8Writer::new_with_backend(out, stride, scaled_roi.w, self.backend);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let checkpoint = self.checkpoint_for_mcu(
                        scan_bytes,
                        fast_tile_region_first_decode_mcu(&self.plan, roi, DownscaleFactor::Full),
                    )?;
                    let scan_warnings = decode_scan_fast_tile_rgb_region(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        roi,
                        checkpoint.as_ref(),
                    )?;
                    Ok(DecodeOutcome {
                        decoded: roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else if matches!(fmt, OutputFormat::Rgb8Scaled { .. })
                    && self.progressive_plan.is_none()
                    && self.plan.matches_fast_tile_shape()
                {
                    let mut writer =
                        Rgb8Writer::new_with_backend(out, stride, scaled_roi.w, self.backend);
                    let scan_bytes = &self.bytes[self.plan.scan_offset..];
                    let checkpoint = self.checkpoint_for_mcu(
                        scan_bytes,
                        fast_tile_region_first_decode_mcu(&self.plan, scaled_roi, downscale),
                    )?;
                    let scan_warnings = decode_scan_fast_tile_rgb_region_scaled(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        &mut writer,
                        FastTileRegionScaledRequest {
                            roi: scaled_roi,
                            downscale,
                            checkpoint: checkpoint.as_ref(),
                        },
                    )?;
                    Ok(DecodeOutcome {
                        decoded: scaled_roi,
                        warnings: merged_warnings(&self.warnings, scan_warnings),
                    })
                } else {
                    let base =
                        Rgb8Writer::new_with_backend(out, stride, scaled_roi.w, self.backend);
                    let (source_x0, source_width) =
                        self.source_window_for_output_rect(downscale, scaled_roi);
                    let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                    self.decode_rgb_with_writer(pool, &mut writer, downscale, roi)
                }
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                let base =
                    Rgba8Writer::new_with_backend(out, stride, scaled_roi.w, alpha, self.backend);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, scaled_roi);
                let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let base = Gray8Writer::new(out, stride, scaled_roi.w);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, scaled_roi);
                let mut writer = CroppedWriter::new(base, scaled_roi, source_x0, source_width);
                self.decode_with_writer(pool, &mut writer, downscale, roi)
            }
            OutputFormat::Gray16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_gray16_region_into(out, stride, roi)
            }
            OutputFormat::Gray16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_gray16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_gray16_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Rgb16 => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_region_into(out, stride, roi)
            }
            OutputFormat::Rgb16Scaled { .. } => {
                if self.info.sof_kind == SofKind::Progressive12 {
                    return self.decode_progressive12_rgb16_region_scaled_into(
                        out, stride, roi, downscale,
                    );
                }
                self.decode_extended12_rgb16_region_scaled_into(out, stride, roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. } => {
                if matches!(
                    self.info.sof_kind,
                    SofKind::Extended12 | SofKind::Progressive12
                ) {
                    return self.decode_12bit_rgba16_region_scaled_into(
                        out, stride, roi, downscale, alpha,
                    );
                }
                Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        };
        if let (Some(total_start), Some(decode_start), Ok(outcome)) =
            (total_start, decode_start, &result)
        {
            let source_width_s = self.info.dimensions.0.to_string();
            let source_height_s = self.info.dimensions.1.to_string();
            let roi_x_s = roi.x.to_string();
            let roi_y_s = roi.y.to_string();
            let roi_w_s = roi.w.to_string();
            let roi_h_s = roi.h.to_string();
            let output_width_s = scaled_roi.w.to_string();
            let output_height_s = scaled_roi.h.to_string();
            let stride_s = stride.to_string();
            let bpp_s = fmt.bytes_per_pixel().to_string();
            let output_bytes_s = stride.saturating_mul(scaled_roi.h as usize).to_string();
            let scratch_bytes_s = scratch_bytes.to_string();
            let warning_count_s = outcome.warnings.len().to_string();
            let decode_us = duration_us_string(decode_start.elapsed());
            let total_us = duration_us_string(total_start.elapsed());
            let mode = if downscale == DownscaleFactor::Full {
                "region"
            } else {
                "region_scaled"
            };
            emit_jpeg_profile_row(
                "decode",
                "cpu",
                &[
                    ("mode", mode),
                    ("fmt", output_format_profile_name(fmt)),
                    ("downscale", downscale_profile_name(downscale)),
                    ("source_width", source_width_s.as_str()),
                    ("source_height", source_height_s.as_str()),
                    ("roi_x", roi_x_s.as_str()),
                    ("roi_y", roi_y_s.as_str()),
                    ("roi_w", roi_w_s.as_str()),
                    ("roi_h", roi_h_s.as_str()),
                    ("output_width", output_width_s.as_str()),
                    ("output_height", output_height_s.as_str()),
                    ("stride", stride_s.as_str()),
                    ("bpp", bpp_s.as_str()),
                    ("scratch_bytes", scratch_bytes_s.as_str()),
                    ("output_bytes", output_bytes_s.as_str()),
                    ("decode_us", decode_us.as_str()),
                    ("total_us", total_us.as_str()),
                    ("warnings", warning_count_s.as_str()),
                ],
            );
        }
        result
    }

    /// Decode `roi` into the caller's buffer using the core `PixelFormat` +
    /// `Downscale` contract.
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
