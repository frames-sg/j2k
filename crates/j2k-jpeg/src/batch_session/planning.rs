// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use j2k_core::{
    BatchDecodeError, BatchInfrastructureError, TileBatchError as IndexedTileBatchError,
};

use super::allocation::{
    ensure_metadata_bytes, ensure_planning_phase, try_vec_with_retained_metadata,
    vec_capacity_bytes, PlannedJob,
};
use crate::context::DecoderContext;
use crate::decoder::{
    PlannedJpegTileDecode, PreparedJpegTileJob, TileBatchError, TileDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use crate::error::JpegError;

pub(super) trait BatchJobOutput {
    fn out_len(&self) -> usize;
}

impl BatchJobOutput for TileDecodeJob<'_, '_> {
    fn out_len(&self) -> usize {
        self.out.len()
    }
}

impl BatchJobOutput for PreparedJpegTileJob<'_, '_> {
    fn out_len(&self) -> usize {
        self.out.len()
    }
}

impl BatchJobOutput for TileScaledDecodeJob<'_, '_> {
    fn out_len(&self) -> usize {
        self.out.len()
    }
}

impl BatchJobOutput for TileRegionScaledDecodeJob<'_, '_> {
    fn out_len(&self) -> usize {
        self.out.len()
    }
}

pub(super) fn min_output_len<T: BatchJobOutput>(jobs: &[T]) -> usize {
    jobs.iter().map(BatchJobOutput::out_len).min().unwrap_or(0)
}

pub(super) fn planned_job_chunk(
    plans: &[PlannedJob],
    start_index: usize,
    chunk_len: usize,
) -> Result<&[PlannedJob], BatchInfrastructureError> {
    let end = start_index
        .checked_add(chunk_len)
        .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
    plans
        .get(start_index..end)
        .ok_or(BatchInfrastructureError::SchedulerPoisoned)
}

pub(super) fn plan_regular_jobs<T>(
    jobs: &[T],
    retained_metadata_bytes: usize,
    context: Option<&mut DecoderContext>,
    mut planner: impl FnMut(&T, &mut DecoderContext) -> Result<PlannedJpegTileDecode, JpegError>,
) -> Result<Vec<PlannedJob>, TileBatchError> {
    let mut plans = try_vec_with_retained_metadata(
        jobs.len(),
        retained_metadata_bytes,
        "JPEG batch job plans",
    )?;
    let planning_metadata = ensure_metadata_bytes(
        retained_metadata_bytes,
        vec_capacity_bytes(&plans)?,
        "JPEG planning metadata",
    )?;
    ensure_planning_phase(planning_metadata)?;
    let mut owned_context = DecoderContext::new();
    let context = context.unwrap_or(&mut owned_context);
    for (index, job) in jobs.iter().enumerate() {
        match planner(job, context) {
            Ok(plan) => plans.push(planned_job(plan)),
            Err(source) => {
                return Err(BatchDecodeError::Tile(IndexedTileBatchError {
                    index,
                    source,
                }));
            }
        }
    }
    Ok(plans)
}

pub(super) fn plan_per_tile_jobs<T>(
    jobs: &[T],
    retained_metadata_bytes: usize,
    context: Option<&mut DecoderContext>,
    mut planner: impl FnMut(&T, &mut DecoderContext) -> Result<PlannedJpegTileDecode, JpegError>,
) -> Result<Vec<PlannedJob>, BatchInfrastructureError> {
    let mut plans = try_vec_with_retained_metadata(
        jobs.len(),
        retained_metadata_bytes,
        "JPEG prepared batch job plans",
    )?;
    let planning_metadata = ensure_metadata_bytes(
        retained_metadata_bytes,
        vec_capacity_bytes(&plans)?,
        "JPEG prepared planning metadata",
    )?;
    ensure_planning_phase(planning_metadata)?;
    let mut owned_context = DecoderContext::new();
    let context = context.unwrap_or(&mut owned_context);
    for job in jobs {
        match planner(job, context) {
            Ok(plan) => plans.push(planned_job(plan)),
            Err(error) => plans.push(PlannedJob::Reject(error)),
        }
    }
    Ok(plans)
}

const fn planned_job(plan: PlannedJpegTileDecode) -> PlannedJob {
    PlannedJob::Decode {
        worker_live_bytes: plan.worker_live_bytes,
        retained_result_bytes: plan.retained_result_bytes,
    }
}
