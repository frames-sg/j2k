// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use std::sync::Mutex;

use j2k_core::{
    BatchDecodeError, BatchInfrastructureError, CodecContext,
    TileBatchError as IndexedTileBatchError,
};

use super::allocation::{
    ensure_metadata_bytes, ensure_planning_phase, try_vec_with_retained_metadata,
    vec_capacity_bytes, PlannedJob,
};
use super::worker::WorkerSlot;
use crate::context::DecoderContext;
use crate::decoder::{
    PlannedJpegTileDecode, PreparedJpegTileJob, TileBatchError, TileDecodeJob,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use crate::error::JpegError;

fn plan_job<T>(
    job: &T,
    workers: &mut [Mutex<WorkerSlot>],
    owned_context: &mut DecoderContext,
    planner: &mut impl FnMut(&T, &mut DecoderContext, usize) -> Result<PlannedJpegTileDecode, JpegError>,
) -> Result<Result<PlannedJpegTileDecode, JpegError>, BatchInfrastructureError> {
    let mut retained = owned_context.retained_allocation_bytes();
    for slot in &mut *workers {
        retained = retained.saturating_add(
            slot.get_mut()
                .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                .retained_bytes(),
        );
    }
    let context = match workers.first_mut() {
        Some(slot) => slot
            .get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
            .planning_context(),
        None => &mut *owned_context,
    };
    let external_live_bytes = retained - context.retained_allocation_bytes();
    let result = planner(job, context, external_live_bytes);
    if retained != 0 && matches!(result, Err(JpegError::MemoryCapExceeded { .. })) {
        // Parsing and construction charge all warm storage to the same codec
        // cap. Retry without it only when that retained storage prevents a fit.
        for slot in &mut *workers {
            slot.get_mut()
                .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                .release_allocations();
        }
        owned_context.clear();
        return Ok(match workers.first_mut() {
            Some(slot) => planner(
                job,
                slot.get_mut()
                    .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                    .planning_context(),
                0,
            ),
            None => planner(job, owned_context, 0),
        });
    }
    Ok(result)
}

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
    workers: &mut [Mutex<WorkerSlot>],
    mut planner: impl FnMut(&T, &mut DecoderContext, usize) -> Result<PlannedJpegTileDecode, JpegError>,
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
    for (index, job) in jobs.iter().enumerate() {
        match plan_job(job, workers, &mut owned_context, &mut planner)? {
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
    workers: &mut [Mutex<WorkerSlot>],
    mut planner: impl FnMut(&T, &mut DecoderContext, usize) -> Result<PlannedJpegTileDecode, JpegError>,
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
    for job in jobs {
        match plan_job(job, workers, &mut owned_context, &mut planner)? {
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
