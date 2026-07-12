// SPDX-License-Identifier: MIT OR Apache-2.0

//! Output-format dispatch after routing has validated geometry and scratch.

use super::super::{
    decode_scan_fast_tile_rgb_region, decode_scan_fast_tile_rgb_region_scaled,
    fast_tile_region_first_decode_mcu, merged_warnings, CroppedWriter, DecodeOutcome, Decoder,
    DownscaleFactor, FastTileRegionScaledRequest, Gray8Writer, JpegError, OutputFormat, Rect,
    Rgb8Writer, Rgba8Writer, ScratchPool, SofKind,
};
use crate::allocation::checked_add_allocation_bytes;

pub(super) struct OutputRoute<'a> {
    pub(super) pool: &'a mut ScratchPool,
    pub(super) out: &'a mut [u8],
    pub(super) stride: usize,
    pub(super) fmt: OutputFormat,
    pub(super) source_roi: Rect,
    pub(super) output_rect: Rect,
    pub(super) downscale: DownscaleFactor,
    pub(super) external_live_bytes: usize,
}

impl Decoder<'_> {
    pub(super) fn dispatch_full_output(
        &self,
        request: OutputRoute<'_>,
    ) -> Result<DecodeOutcome, JpegError> {
        let OutputRoute {
            pool,
            out,
            stride,
            fmt,
            source_roi,
            output_rect,
            downscale,
            external_live_bytes,
        } = request;
        match fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                let mut writer =
                    Rgb8Writer::new_with_backend(out, stride, output_rect.w, self.backend);
                self.decode_rgb_with_writer(pool, &mut writer, downscale, source_roi)
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                let mut writer =
                    Rgba8Writer::new_with_backend(out, stride, output_rect.w, alpha, self.backend);
                self.decode_with_writer(pool, &mut writer, downscale, source_roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let mut writer = Gray8Writer::new(out, stride, output_rect.w);
                self.decode_with_writer(pool, &mut writer, downscale, source_roi)
            }
            OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. }
                if self.info.sof_kind == SofKind::Progressive12 =>
            {
                self.decode_progressive12_gray16_region_scaled_into(
                    out, stride, source_roi, downscale,
                )
            }
            OutputFormat::Gray16 => self.decode_extended12_gray16_into(out, stride),
            OutputFormat::Gray16Scaled { .. } => {
                self.decode_extended12_gray16_region_scaled_into(out, stride, source_roi, downscale)
            }
            OutputFormat::Rgb16 | OutputFormat::Rgb16Scaled { .. }
                if self.info.sof_kind == SofKind::Progressive12 =>
            {
                self.decode_progressive12_rgb16_region_scaled_into(
                    out, stride, source_roi, downscale,
                )
            }
            OutputFormat::Rgb16 => self.decode_extended12_rgb16_into(out, stride),
            OutputFormat::Rgb16Scaled { .. } => {
                self.decode_extended12_rgb16_region_scaled_into(out, stride, source_roi, downscale)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. }
                if matches!(
                    self.info.sof_kind,
                    SofKind::Extended12 | SofKind::Progressive12
                ) =>
            {
                self.decode_12bit_rgba16_region_scaled_into(
                    out,
                    stride,
                    source_roi,
                    downscale,
                    alpha,
                    external_live_bytes,
                )
            }
            OutputFormat::Rgba16 { .. } | OutputFormat::Rgba16Scaled { .. } => {
                Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        }
    }

    pub(super) fn dispatch_region_output(
        &self,
        request: OutputRoute<'_>,
    ) -> Result<DecodeOutcome, JpegError> {
        match request.fmt {
            OutputFormat::Rgb8 | OutputFormat::Rgb8Scaled { .. } => {
                self.decode_region_rgb8(request)
            }
            OutputFormat::Rgba8 { alpha } | OutputFormat::Rgba8Scaled { alpha, .. } => {
                let OutputRoute {
                    pool,
                    out,
                    stride,
                    source_roi,
                    output_rect,
                    downscale,
                    ..
                } = request;
                let base =
                    Rgba8Writer::new_with_backend(out, stride, output_rect.w, alpha, self.backend);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, output_rect);
                let mut writer = CroppedWriter::new(base, output_rect, source_x0, source_width)?;
                self.decode_with_writer(pool, &mut writer, downscale, source_roi)
            }
            OutputFormat::Gray8 | OutputFormat::Gray8Scaled { .. } => {
                let OutputRoute {
                    pool,
                    out,
                    stride,
                    source_roi,
                    output_rect,
                    downscale,
                    ..
                } = request;
                let base = Gray8Writer::new(out, stride, output_rect.w);
                let (source_x0, source_width) =
                    self.source_window_for_output_rect(downscale, output_rect);
                let mut writer = CroppedWriter::new(base, output_rect, source_x0, source_width)?;
                self.decode_with_writer(pool, &mut writer, downscale, source_roi)
            }
            OutputFormat::Gray16 | OutputFormat::Gray16Scaled { .. } => {
                self.decode_region_gray16(request)
            }
            OutputFormat::Rgb16 | OutputFormat::Rgb16Scaled { .. } => {
                self.decode_region_rgb16(request)
            }
            OutputFormat::Rgba16 { alpha } | OutputFormat::Rgba16Scaled { alpha, .. }
                if matches!(
                    self.info.sof_kind,
                    SofKind::Extended12 | SofKind::Progressive12
                ) =>
            {
                self.decode_12bit_rgba16_region_scaled_into(
                    request.out,
                    request.stride,
                    request.source_roi,
                    request.downscale,
                    alpha,
                    request.external_live_bytes,
                )
            }
            OutputFormat::Rgba16 { .. } | OutputFormat::Rgba16Scaled { .. } => {
                Err(JpegError::NotImplemented {
                    sof: self.info.sof_kind,
                })
            }
        }
    }

    fn decode_region_gray16(&self, request: OutputRoute<'_>) -> Result<DecodeOutcome, JpegError> {
        let OutputRoute {
            out,
            stride,
            fmt,
            source_roi,
            downscale,
            ..
        } = request;
        if self.info.sof_kind == SofKind::Progressive12 {
            return self.decode_progressive12_gray16_region_scaled_into(
                out, stride, source_roi, downscale,
            );
        }
        match fmt {
            OutputFormat::Gray16 => {
                self.decode_extended12_gray16_region_into(out, stride, source_roi)
            }
            OutputFormat::Gray16Scaled { .. } => {
                self.decode_extended12_gray16_region_scaled_into(out, stride, source_roi, downscale)
            }
            _ => Err(JpegError::InternalInvariant {
                reason: "non-gray output reached gray region dispatch",
            }),
        }
    }

    fn decode_region_rgb16(&self, request: OutputRoute<'_>) -> Result<DecodeOutcome, JpegError> {
        let OutputRoute {
            out,
            stride,
            fmt,
            source_roi,
            downscale,
            ..
        } = request;
        if self.info.sof_kind == SofKind::Progressive12 {
            return self
                .decode_progressive12_rgb16_region_scaled_into(out, stride, source_roi, downscale);
        }
        match fmt {
            OutputFormat::Rgb16 => {
                self.decode_extended12_rgb16_region_into(out, stride, source_roi)
            }
            OutputFormat::Rgb16Scaled { .. } => {
                self.decode_extended12_rgb16_region_scaled_into(out, stride, source_roi, downscale)
            }
            _ => Err(JpegError::InternalInvariant {
                reason: "non-RGB output reached RGB region dispatch",
            }),
        }
    }

    fn decode_region_rgb8(&self, request: OutputRoute<'_>) -> Result<DecodeOutcome, JpegError> {
        let OutputRoute {
            pool,
            out,
            stride,
            fmt,
            source_roi,
            output_rect,
            downscale,
            external_live_bytes,
        } = request;
        if fmt == OutputFormat::Rgb8
            && downscale == DownscaleFactor::Full
            && self.progressive_plan.is_none()
            && self.plan.matches_fast_tile_shape()
        {
            let mut writer = Rgb8Writer::new_with_backend(out, stride, output_rect.w, self.backend);
            let scan_bytes = &self.bytes[self.plan.scan_offset..];
            let checkpoint = self.checkpoint_for_mcu(
                scan_bytes,
                fast_tile_region_first_decode_mcu(&self.plan, source_roi, DownscaleFactor::Full),
                checked_add_allocation_bytes(external_live_bytes, pool.retained_bytes())?,
            )?;
            let scan_warnings = decode_scan_fast_tile_rgb_region(
                &self.plan,
                self.backend,
                scan_bytes,
                pool,
                &mut writer,
                source_roi,
                checkpoint.as_ref(),
            )?;
            return Ok(DecodeOutcome {
                decoded: source_roi,
                warnings: merged_warnings(&self.warnings, scan_warnings)?,
            });
        }
        if matches!(fmt, OutputFormat::Rgb8Scaled { .. })
            && self.progressive_plan.is_none()
            && self.plan.matches_fast_tile_shape()
        {
            let mut writer = Rgb8Writer::new_with_backend(out, stride, output_rect.w, self.backend);
            let scan_bytes = &self.bytes[self.plan.scan_offset..];
            let checkpoint = self.checkpoint_for_mcu(
                scan_bytes,
                fast_tile_region_first_decode_mcu(&self.plan, output_rect, downscale),
                checked_add_allocation_bytes(external_live_bytes, pool.retained_bytes())?,
            )?;
            let scan_warnings = decode_scan_fast_tile_rgb_region_scaled(
                &self.plan,
                self.backend,
                scan_bytes,
                pool,
                &mut writer,
                FastTileRegionScaledRequest {
                    roi: output_rect,
                    downscale,
                    checkpoint: checkpoint.as_ref(),
                },
            )?;
            return Ok(DecodeOutcome {
                decoded: output_rect,
                warnings: merged_warnings(&self.warnings, scan_warnings)?,
            });
        }
        let base = Rgb8Writer::new_with_backend(out, stride, output_rect.w, self.backend);
        let (source_x0, source_width) = self.source_window_for_output_rect(downscale, output_rect);
        let mut writer = CroppedWriter::new(base, output_rect, source_x0, source_width)?;
        self.decode_rgb_with_writer(pool, &mut writer, downscale, source_roi)
    }
}
