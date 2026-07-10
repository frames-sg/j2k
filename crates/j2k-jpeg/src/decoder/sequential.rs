// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential and progressive writer-based decode paths.

use super::{
    checkpoint_before_mcu, decode_progressive, decode_scan_baseline, decode_scan_baseline_rgb,
    decode_scan_fast_rgb_444, decode_scan_fast_tile_rgb, emit_decode_scan_profile,
    jpeg_profile_stages_enabled, merged_warnings, scaled_dimensions, scaled_rect_covering,
    stripe_region_layout, DecodeOutcome, Decoder, DeviceCheckpoint, DownscaleFactor, Instant,
    InterleavedRgbWriter, JpegError, OutputWriter, ProgressiveDownscaleWriter, Rect, ScratchPool,
    CPU_ROI_CHECKPOINT_CADENCE_MCUS, CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS, DEFAULT_MAX_DECODE_BYTES,
};

impl Decoder<'_> {
    pub(super) fn decode_scratch_bytes(&self, cap: usize) -> Result<usize, JpegError> {
        let scratch_bytes = self
            .progressive_plan
            .as_ref()
            .map_or(self.plan.scratch_bytes, |plan| plan.scratch_bytes);
        if scratch_bytes > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: scratch_bytes,
                cap,
            });
        }
        Ok(scratch_bytes)
    }

    pub(super) fn checkpoint_for_mcu(
        &self,
        scan_bytes: &[u8],
        target_mcu: u32,
    ) -> Result<Option<DeviceCheckpoint>, JpegError> {
        if self.plan.restart_interval.is_some() || target_mcu < CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS {
            return Ok(None);
        }

        let mut cache =
            self.cpu_entropy_checkpoints
                .lock()
                .map_err(|_| JpegError::InternalInvariant {
                    reason: "CPU entropy checkpoint cache mutex poisoned",
                })?;
        checkpoint_before_mcu(
            &self.plan,
            scan_bytes,
            CPU_ROI_CHECKPOINT_CADENCE_MCUS,
            target_mcu,
            &mut cache,
        )
    }

    pub(super) fn source_window_for_output_rect(
        &self,
        downscale: DownscaleFactor,
        output_rect: Rect,
    ) -> (u32, u32) {
        if self.progressive_plan.is_some() {
            return (0, scaled_dimensions(self.info.dimensions, downscale).0);
        }
        let layout = stripe_region_layout(&self.plan, downscale, output_rect);
        (layout.source_x0, layout.source_width)
    }

    pub(super) fn decode_with_writer<W: OutputWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            let scan_start = profile_enabled.then(Instant::now);
            let scan_warnings = if downscale == DownscaleFactor::Full {
                decode_progressive(plan, self.backend, self.bytes, writer)?
            } else {
                let mut scaled =
                    ProgressiveDownscaleWriter::new(writer, downscale, self.info.dimensions);
                decode_progressive(plan, self.backend, self.bytes, &mut scaled)?
            };
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return Ok(DecodeOutcome {
                decoded,
                warnings: merged_warnings(&self.warnings, scan_warnings),
            });
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        let scan_warnings = decode_scan_baseline(
            &self.plan,
            self.backend,
            scan_bytes,
            pool,
            writer,
            downscale,
            output_rect,
        )?;
        if let Some(start) = scan_start {
            emit_decode_scan_profile(
                "baseline",
                self.info.dimensions,
                decoded,
                downscale,
                start.elapsed(),
            );
        }
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }

    pub(super) fn decode_rgb_with_writer<W: OutputWriter + InterleavedRgbWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(DEFAULT_MAX_DECODE_BYTES)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            let scan_start = profile_enabled.then(Instant::now);
            let scan_warnings = if downscale == DownscaleFactor::Full {
                decode_progressive(plan, self.backend, self.bytes, writer)?
            } else {
                let mut scaled =
                    ProgressiveDownscaleWriter::new(writer, downscale, self.info.dimensions);
                decode_progressive(plan, self.backend, self.bytes, &mut scaled)?
            };
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive_rgb",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return Ok(DecodeOutcome {
                decoded,
                warnings: merged_warnings(&self.warnings, scan_warnings),
            });
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        let (scan_path, scan_warnings) =
            if downscale == DownscaleFactor::Full && self.plan.matches_fast_tile_shape() {
                (
                    "fast420_rgb",
                    decode_scan_fast_tile_rgb(&self.plan, self.backend, scan_bytes, pool, writer)?,
                )
            } else if downscale == DownscaleFactor::Full
                && decoded == Rect::full(self.info.dimensions)
                && self.plan.matches_fast_rgb444_shape()
            {
                (
                    "fast444_rgb",
                    decode_scan_fast_rgb_444(&self.plan, self.backend, scan_bytes, pool, writer)?,
                )
            } else {
                (
                    "baseline_rgb",
                    decode_scan_baseline_rgb(
                        &self.plan,
                        self.backend,
                        scan_bytes,
                        pool,
                        writer,
                        downscale,
                        output_rect,
                    )?,
                )
            };
        if let Some(start) = scan_start {
            emit_decode_scan_profile(
                scan_path,
                self.info.dimensions,
                decoded,
                downscale,
                start.elapsed(),
            );
        }
        Ok(DecodeOutcome {
            decoded,
            warnings: merged_warnings(&self.warnings, scan_warnings),
        })
    }
}
