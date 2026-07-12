// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential and progressive writer-based decode paths.

use crate::allocation::checked_add_allocation_bytes;

use super::{
    checkpoint_before_mcu, decode_progressive, decode_scan_baseline, decode_scan_baseline_rgb,
    decode_scan_fast_rgb_444, decode_scan_fast_tile_rgb, emit_decode_scan_profile,
    jpeg_profile_stages_enabled, merged_warnings, scaled_dimensions, scaled_rect_covering,
    stripe_region_layout, DecodeOutcome, Decoder, DeviceCheckpoint, DownscaleFactor, Instant,
    InterleavedRgbWriter, JpegError, OutputWriter, ProgressiveDownscaleWriter, Rect, ScratchPool,
    CPU_ROI_CHECKPOINT_CADENCE_MCUS, CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS,
};

fn checked_live_workspace_bytes(
    owned_output_bytes: usize,
    scratch_bytes: usize,
    cap: usize,
) -> Result<usize, JpegError> {
    let requested =
        owned_output_bytes
            .checked_add(scratch_bytes)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(requested)
}

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

    pub(super) fn prepare_decode_workspace(
        &self,
        pool: &mut ScratchPool,
        owned_output_bytes: usize,
    ) -> Result<usize, JpegError> {
        self.prepare_decode_workspace_with_additional(pool, owned_output_bytes, 0)
    }

    pub(super) fn prepare_decode_workspace_with_additional(
        &self,
        pool: &mut ScratchPool,
        owned_output_bytes: usize,
        additional_scratch_bytes: usize,
    ) -> Result<usize, JpegError> {
        let workspace_cap = self.decode_workspace_cap()?;
        let base_scratch_bytes = self.decode_scratch_bytes(workspace_cap)?;
        let scratch_bytes = base_scratch_bytes
            .checked_add(additional_scratch_bytes)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: workspace_cap,
            })?;
        let requested =
            checked_live_workspace_bytes(owned_output_bytes, scratch_bytes, workspace_cap)?;
        let pool_backed = self.progressive_plan.is_none()
            && !matches!(
                self.info.sof_kind,
                super::SofKind::Lossless | super::SofKind::Extended12
            );
        if pool_backed {
            pool.reconcile_external_workspace(owned_output_bytes, workspace_cap)?;
        } else {
            pool.release_for_external_workspace(requested, workspace_cap)?;
        }
        Ok(scratch_bytes)
    }

    pub(super) fn checkpoint_for_mcu(
        &self,
        scan_bytes: &[u8],
        target_mcu: u32,
        external_decode_phase_bytes: usize,
    ) -> Result<Option<DeviceCheckpoint>, JpegError> {
        if self.plan.restart_interval.is_some() || target_mcu < CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS {
            return Ok(None);
        }

        let retained_decoder_baseline_bytes = checked_add_allocation_bytes(
            self.retained_allocation_bytes_excluding_cpu_checkpoint_cache()?,
            external_decode_phase_bytes,
        )?;
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
            retained_decoder_baseline_bytes,
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
        let _ = self.decode_scratch_bytes(self.decode_workspace_cap()?)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            let scan_start = profile_enabled.then(Instant::now);
            let detached_sink_bytes = pool.detached_sink_bytes();
            let scan_warnings = if downscale == DownscaleFactor::Full {
                let external_live_bytes =
                    checked_live_workspace_bytes(detached_sink_bytes, 0, plan.scratch_bytes)?;
                decode_progressive(plan, self.backend, self.bytes, writer, external_live_bytes)?
            } else {
                let mut scaled =
                    ProgressiveDownscaleWriter::new(writer, downscale, self.info.dimensions)?;
                let external_live_bytes = checked_live_workspace_bytes(
                    detached_sink_bytes,
                    scaled.capacity_bytes()?,
                    plan.scratch_bytes,
                )?;
                decode_progressive(
                    plan,
                    self.backend,
                    self.bytes,
                    &mut scaled,
                    external_live_bytes,
                )?
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
                warnings: merged_warnings(&self.warnings, scan_warnings)?,
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
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }

    pub(super) fn decode_rgb_with_writer<W: OutputWriter + InterleavedRgbWriter>(
        &self,
        pool: &mut ScratchPool,
        writer: &mut W,
        downscale: DownscaleFactor,
        decoded: Rect,
    ) -> Result<DecodeOutcome, JpegError> {
        let _ = self.decode_scratch_bytes(self.decode_workspace_cap()?)?;
        let profile_enabled = jpeg_profile_stages_enabled();
        if let Some(plan) = &self.progressive_plan {
            let scan_start = profile_enabled.then(Instant::now);
            let detached_sink_bytes = pool.detached_sink_bytes();
            let scan_warnings = if downscale == DownscaleFactor::Full {
                let external_live_bytes =
                    checked_live_workspace_bytes(detached_sink_bytes, 0, plan.scratch_bytes)?;
                decode_progressive(plan, self.backend, self.bytes, writer, external_live_bytes)?
            } else {
                let mut scaled =
                    ProgressiveDownscaleWriter::new(writer, downscale, self.info.dimensions)?;
                let external_live_bytes = checked_live_workspace_bytes(
                    detached_sink_bytes,
                    scaled.capacity_bytes()?,
                    plan.scratch_bytes,
                )?;
                decode_progressive(
                    plan,
                    self.backend,
                    self.bytes,
                    &mut scaled,
                    external_live_bytes,
                )?
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
                warnings: merged_warnings(&self.warnings, scan_warnings)?,
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
            warnings: merged_warnings(&self.warnings, scan_warnings)?,
        })
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::{checked_live_workspace_bytes, JpegError};

    #[test]
    fn owned_output_and_scratch_share_an_exact_cap_boundary() {
        assert_eq!(checked_live_workspace_bytes(37, 63, 100).unwrap(), 100);
        assert!(matches!(
            checked_live_workspace_bytes(38, 63, 100),
            Err(JpegError::MemoryCapExceeded {
                requested: 101,
                cap: 100
            })
        ));
    }

    #[test]
    fn live_workspace_overflow_stays_a_cap_error() {
        assert!(matches!(
            checked_live_workspace_bytes(usize::MAX, 1, 100),
            Err(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: 100
            })
        ));
    }
}
