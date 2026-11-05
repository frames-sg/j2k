// SPDX-License-Identifier: MIT OR Apache-2.0

//! Sequential and progressive writer-based decode paths.

use crate::adapter::JpegFastPacketFamily;
use crate::allocation::checked_add_allocation_bytes;
use crate::context::MAX_DECODER_CONTEXT_ALLOCATION_BYTES;
use crate::entropy::sequential::restart_checkpoint_from_index;

use super::{
    checkpoint_before_mcu, decode_progressive, decode_scan_baseline, decode_scan_baseline_rgb,
    emit_decode_scan_profile, jpeg_profile_stages_enabled, merged_warnings, scaled_dimensions,
    scaled_rect_covering, stripe_region_layout, DecodeOutcome, Decoder, DeviceCheckpoint,
    DownscaleFactor, FastRgbRequest, Instant, InterleavedRgbWriter, JpegError, OutputWriter, Rect,
    ScratchPool, CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS,
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

    /// The decoder's resume point for `target_mcu`, which a route hands to
    /// `scan_position_before_mcu`. Restart-coded scans get the
    /// restart-boundary checkpoint from this decoder's restart index (see
    /// [`Self::restart_checkpoint`]); other scans get the cached entropy
    /// checkpoint at or before the target, for targets deep enough to be
    /// worth extending the cache.
    pub(crate) fn checkpoint_for_mcu(
        &self,
        scan_bytes: &[u8],
        target_mcu: u32,
        external_decode_phase_bytes: usize,
    ) -> Result<Option<DeviceCheckpoint>, JpegError> {
        if self.plan.restart_interval.is_some() {
            return self.restart_checkpoint(target_mcu);
        }
        if target_mcu < CPU_ROI_CHECKPOINT_MIN_TARGET_MCUS {
            return Ok(None);
        }
        checkpoint_before_mcu(
            &self.plan,
            scan_bytes,
            target_mcu,
            self.checkpoint_baseline_bytes(external_decode_phase_bytes)?,
            &self.cpu_entropy_checkpoints,
        )
    }

    /// Bytes the checkpoint cache's allocations must fit beside: the decoder
    /// without that cache, plus `external_decode_phase_bytes`. The shared
    /// decoder context stays live beside the decode, charged once as in
    /// `decode_workspace_cap`.
    fn checkpoint_baseline_bytes(
        &self,
        external_decode_phase_bytes: usize,
    ) -> Result<usize, JpegError> {
        checked_add_allocation_bytes(
            checked_add_allocation_bytes(
                MAX_DECODER_CONTEXT_ALLOCATION_BYTES,
                self.retained_allocation_bytes_excluding_cpu_checkpoint_cache()?,
            )?,
            external_decode_phase_bytes,
        )
    }

    /// Restart-boundary checkpoint for `target_mcu` from this decoder's
    /// restart index. The first seek returns `None`, so one-shot decodes scan
    /// only up to their marker; the second builds and keeps the validated
    /// index. A stream whose markers do not validate as an index, or whose
    /// index the host budget cannot retain, keeps returning `None`, so its
    /// routes keep the lenient marker scan with its exact results and errors.
    fn restart_checkpoint(&self, target_mcu: u32) -> Result<Option<DeviceCheckpoint>, JpegError> {
        let build = match self
            .cpu_entropy_checkpoints
            .with_restart_index(|index| restart_checkpoint_from_index(index, target_mcu))?
        {
            Ok(checkpoint) => return Ok(checkpoint),
            Err(build) => build,
        };
        if !build {
            return Ok(None);
        }
        let Ok(Some(index)) = self.restart_index() else {
            return Ok(None);
        };
        let checkpoint = restart_checkpoint_from_index(&index, target_mcu);
        self.cpu_entropy_checkpoints
            .retain_restart_index(index, self.checkpoint_baseline_bytes(0)?)?;
        Ok(checkpoint)
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
            let external_live_bytes =
                checked_live_workspace_bytes(detached_sink_bytes, 0, plan.scratch_bytes)?;
            let scan_warnings = decode_progressive(
                plan,
                self.backend,
                self.bytes,
                writer,
                external_live_bytes,
                downscale,
            )?;
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return self.finish_outcome(decoded, merged_warnings(&self.warnings, scan_warnings)?);
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
        self.finish_outcome(decoded, merged_warnings(&self.warnings, scan_warnings)?)
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
            let external_live_bytes =
                checked_live_workspace_bytes(detached_sink_bytes, 0, plan.scratch_bytes)?;
            let scan_warnings = decode_progressive(
                plan,
                self.backend,
                self.bytes,
                writer,
                external_live_bytes,
                downscale,
            )?;
            if let Some(start) = scan_start {
                emit_decode_scan_profile(
                    "progressive_rgb",
                    self.info.dimensions,
                    decoded,
                    downscale,
                    start.elapsed(),
                );
            }
            return self.finish_outcome(decoded, merged_warnings(&self.warnings, scan_warnings)?);
        }
        let output_rect = scaled_rect_covering(decoded, downscale)?;
        let scan_bytes = &self.bytes[self.plan.scan_offset..];
        let scan_start = profile_enabled.then(Instant::now);
        // A partial 4:2:0 decode streams the whole image through the cropping
        // writer; the other fast shapes take it only for whole images.
        let fast_family = self.plan.fast_shape_family().filter(|&family| {
            downscale == DownscaleFactor::Full
                && (family == JpegFastPacketFamily::Fast420
                    || decoded == Rect::full(self.info.dimensions))
        });
        let (scan_path, scan_warnings) = if let Some(family) = fast_family {
            let request = FastRgbRequest {
                roi: Rect::full(self.plan.dimensions),
                checkpoint: None,
            };
            (
                match family {
                    JpegFastPacketFamily::Fast420 => "fast420_rgb",
                    JpegFastPacketFamily::Fast422 => "fast422_rgb",
                    JpegFastPacketFamily::Fast444 => "fast444_rgb",
                },
                family.decode(&self.plan, self.backend, scan_bytes, pool, writer, request)?,
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
        self.finish_outcome(decoded, merged_warnings(&self.warnings, scan_warnings)?)
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
