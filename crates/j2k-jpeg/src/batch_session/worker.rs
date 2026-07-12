// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;
use j2k_core::{
    BatchInfrastructureError, CodecContext, PixelFormat, ScratchPool as CoreScratchPool,
};

use super::allocation::PlannedJob;
use super::BatchResultSlot;
use crate::context::DecoderContext;
use crate::decoder::{
    decode_prepared_jpeg_tile_rgb8_in_context, decode_tile_into_in_context_with_options,
    decode_tile_region_scaled_into_in_context_with_options,
    decode_tile_scaled_into_in_context_with_options, DecodeOutcome, DecodedTile,
    PreparedJpegTileJob, TileDecodeJob, TileDecodeOutput, TileRegionScaledDecodeJob,
    TileScaledDecodeJob,
};
use crate::info::DecodeOptions;
use crate::internal::scratch::ScratchPool;
use crate::{JpegError, Warning};

#[derive(Debug, Default)]
pub(super) struct WorkerSlot {
    ctx: DecoderContext,
    pool: ScratchPool,
}

impl WorkerSlot {
    pub(super) fn decode_tile_job_chunk(
        &mut self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        results: &mut [BatchResultSlot<DecodeOutcome>],
        plans: &[PlannedJob],
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_result_slots(jobs.len(), results.len())?;
        ensure_result_slots(jobs.len(), plans.len())?;
        for ((job, plan), slot) in jobs.iter_mut().zip(plans).zip(results) {
            let outcome = retain_within_planned_warning_claim(
                decode_tile_into_in_context_with_options(
                    job.input,
                    &mut self.ctx,
                    &mut self.pool,
                    job.out,
                    job.stride,
                    fmt,
                    options,
                ),
                plan,
                |outcome| outcome.warnings.capacity(),
            )?;
            *slot = Some(outcome);
        }
        Ok(())
    }

    pub(super) fn decode_prepared_tile_job_chunk(
        &mut self,
        jobs: &mut [PreparedJpegTileJob<'_, '_>],
        results: &mut [BatchResultSlot<DecodedTile>],
        plans: &[PlannedJob],
    ) -> Result<(), BatchInfrastructureError> {
        ensure_result_slots(jobs.len(), results.len())?;
        ensure_result_slots(jobs.len(), plans.len())?;
        for ((job, plan), slot) in jobs.iter_mut().zip(plans).zip(results) {
            let outcome = match plan {
                PlannedJob::Decode { .. } => decode_prepared_jpeg_tile_rgb8_in_context(
                    &job.input,
                    &mut self.ctx,
                    &mut self.pool,
                    job.out,
                    job.stride,
                    job.options,
                ),
                PlannedJob::Reject(error) => Err(error.clone()),
            };
            let outcome = retain_within_planned_warning_claim(outcome, plan, |tile| {
                tile.warnings.capacity()
            })?;
            *slot = Some(outcome);
        }
        Ok(())
    }

    pub(super) fn decode_tile_scaled_job_chunk(
        &mut self,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        results: &mut [BatchResultSlot<DecodeOutcome>],
        plans: &[PlannedJob],
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_result_slots(jobs.len(), results.len())?;
        ensure_result_slots(jobs.len(), plans.len())?;
        for ((job, plan), slot) in jobs.iter_mut().zip(plans).zip(results) {
            let outcome = retain_within_planned_warning_claim(
                decode_tile_scaled_into_in_context_with_options(
                    job.input,
                    &mut self.ctx,
                    &mut self.pool,
                    TileDecodeOutput {
                        out: job.out,
                        stride: job.stride,
                        fmt,
                    },
                    job.scale,
                    options,
                ),
                plan,
                |outcome| outcome.warnings.capacity(),
            )?;
            *slot = Some(outcome);
        }
        Ok(())
    }

    pub(super) fn decode_tile_region_scaled_job_chunk(
        &mut self,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        results: &mut [BatchResultSlot<DecodeOutcome>],
        plans: &[PlannedJob],
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Result<(), BatchInfrastructureError> {
        ensure_result_slots(jobs.len(), results.len())?;
        ensure_result_slots(jobs.len(), plans.len())?;
        for ((job, plan), slot) in jobs.iter_mut().zip(plans).zip(results) {
            let outcome = retain_within_planned_warning_claim(
                decode_tile_region_scaled_into_in_context_with_options(
                    job.input,
                    &mut self.ctx,
                    &mut self.pool,
                    TileDecodeOutput {
                        out: job.out,
                        stride: job.stride,
                        fmt,
                    },
                    job.roi.into(),
                    job.scale,
                    options,
                ),
                plan,
                |outcome| outcome.warnings.capacity(),
            )?;
            *slot = Some(outcome);
        }
        Ok(())
    }

    pub(super) fn reset(&mut self) {
        self.ctx.clear();
        self.pool.reset();
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.ctx
            .retained_allocation_bytes()
            .saturating_add(self.pool.retained_bytes())
    }

    pub(super) fn release_allocations(&mut self) {
        self.ctx.clear();
        self.pool = ScratchPool::default();
    }

    pub(super) fn prepare_for_planning(&mut self) {
        // Planning reuses one bounded decoder context, but no stale decode
        // scratch may coexist with the planning decoder's full codec claim.
        self.pool = ScratchPool::default();
    }

    pub(super) fn planning_context(&mut self) -> &mut DecoderContext {
        &mut self.ctx
    }
}

fn retain_within_planned_warning_claim<T>(
    outcome: Result<T, JpegError>,
    plan: &PlannedJob,
    warning_capacity: impl FnOnce(&T) -> usize,
) -> Result<Result<T, JpegError>, BatchInfrastructureError> {
    // The freshly decoded warning vector remains part of this worker's
    // in-flight ownership until its allocator-reported capacity is checked.
    // Only a claim-valid outcome may cross into the retained result metadata
    // domain by being written to an ordered slot.
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(error) => return Ok(Err(error)),
    };
    let actual = warning_capacity(&outcome).saturating_mul(size_of::<Warning>());
    ensure_retained_result_claim(plan, actual)?;
    Ok(Ok(outcome))
}

fn ensure_retained_result_claim(
    plan: &PlannedJob,
    actual: usize,
) -> Result<(), BatchInfrastructureError> {
    let planned = plan.retained_result_bytes();
    if actual > planned {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG retained warning metadata",
            requested: actual,
            cap: planned,
        });
    }
    Ok(())
}

fn ensure_result_slots(jobs: usize, results: usize) -> Result<(), BatchInfrastructureError> {
    if results < jobs {
        return Err(BatchInfrastructureError::MissingResult { index: results });
    }
    if results > jobs {
        return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
            index: jobs,
            job_count: jobs,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_warning_owner_cannot_exceed_planned_metadata_claim() {
        let plan = PlannedJob::Decode {
            worker_live_bytes: 0,
            retained_result_bytes: 8,
        };
        ensure_retained_result_claim(&plan, 8).expect("exact warning claim");
        assert!(matches!(
            ensure_retained_result_claim(&plan, 9),
            Err(BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG retained warning metadata",
                requested: 9,
                cap: 8,
            })
        ));
    }
}
