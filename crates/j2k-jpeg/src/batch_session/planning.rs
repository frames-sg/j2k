// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use std::sync::Mutex;

use j2k_core::{
    BatchDecodeError, BatchInfrastructureError, CodecContext,
    TileBatchError as IndexedTileBatchError,
};

use super::allocation::{
    checked_add, ensure_metadata_bytes, ensure_planning_phase, try_vec_with_retained_metadata,
    vec_capacity_bytes, PlannedJob, JPEG_CODEC_HOST_CAP_BYTES,
};
use super::worker::WorkerSlot;
use crate::allocation::{try_new_vec_with_live_budget, AllocationBudgetError};
use crate::context::DecoderContext;
use crate::decoder::{
    PlannedJpegTileDecode, PreparedJpegTileDecode, PreparedJpegTileJob, TileBatchError,
    TileDecodeJob, TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use crate::error::JpegError;

fn plan_job<T, P>(
    job: &T,
    workers: &mut [Mutex<WorkerSlot>],
    owned_context: &mut DecoderContext,
    retained_codec_bytes: usize,
    planner: &mut impl FnMut(&T, &mut DecoderContext, usize) -> Result<P, JpegError>,
) -> Result<Result<P, JpegError>, BatchInfrastructureError> {
    let mut retained_context_bytes = owned_context.retained_allocation_bytes();
    for slot in &mut *workers {
        retained_context_bytes = retained_context_bytes.saturating_add(
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
    let external_live_bytes = retained_codec_bytes
        .saturating_add(retained_context_bytes)
        .saturating_sub(context.retained_allocation_bytes());
    let result = planner(job, context, external_live_bytes);
    if retained_context_bytes != 0 && matches!(result, Err(JpegError::MemoryCapExceeded { .. })) {
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
                retained_codec_bytes,
            ),
            None => planner(job, owned_context, retained_codec_bytes),
        });
    }
    Ok(result)
}

pub(super) struct PreparedRegularJobs<'a> {
    pub(super) plans: Vec<PlannedJob>,
    pub(super) prepared: Vec<PreparedJpegTileDecode<'a>>,
    pub(super) decoder_indices: Vec<usize>,
    pub(super) planning_context: Option<DecoderContext>,
    pub(super) retained_codec_bytes: usize,
}

#[expect(
    clippy::large_enum_variant,
    reason = "keeping prepared jobs inline avoids an untracked infallible box allocation"
)]
pub(super) enum PreparedRegularPlanning<'a> {
    Prepared(PreparedRegularJobs<'a>),
    Fallback,
}

const RECENT_INPUT_LOOKBACK: usize = 32;
const RECENT_INPUT_CONTENT_MAX_BYTES: usize = 8 * 1024;
const RECENT_INPUT_CONTENT_SCAN_BUDGET_BYTES: usize = 128 * 1024;

#[expect(
    clippy::too_many_lines,
    reason = "prepared decoder ownership and its live-byte ledger are planned transactionally"
)]
pub(super) fn plan_prepared_regular_jobs<'a, 'out>(
    jobs: &[TileDecodeJob<'a, 'out>],
    retained_metadata_bytes: usize,
    workers: &mut [Mutex<WorkerSlot>],
    mut planner: impl FnMut(
        &TileDecodeJob<'a, 'out>,
        &mut DecoderContext,
        usize,
    ) -> Result<PreparedJpegTileDecode<'a>, JpegError>,
) -> Result<PreparedRegularPlanning<'a>, TileBatchError> {
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
    let mut decoder_indices = match try_vec_with_retained_metadata(
        jobs.len(),
        planning_metadata,
        "JPEG prepared decoder indices",
    ) {
        Ok(indices) => indices,
        Err(
            BatchInfrastructureError::AllocationTooLarge { .. }
            | BatchInfrastructureError::HostAllocationFailed { .. },
        ) => return Ok(PreparedRegularPlanning::Fallback),
        Err(error) => return Err(error.into()),
    };
    let planning_metadata = ensure_metadata_bytes(
        planning_metadata,
        vec_capacity_bytes(&decoder_indices)?,
        "JPEG prepared planning metadata",
    )?;
    ensure_planning_phase(planning_metadata)?;

    let retain_owned_context = workers.is_empty();
    let mut owned_context = DecoderContext::new();
    let retained_context_bytes = match current_retained_codec_bytes(workers, &owned_context) {
        Ok(bytes) => bytes,
        Err(BatchInfrastructureError::AllocationTooLarge { .. }) => {
            release_planning_allocations(workers, &mut owned_context)?;
            0
        }
        Err(error) => return Err(error.into()),
    };
    let mut allocation_baseline_bytes = retained_context_bytes;
    let mut allocation_live_bytes = allocation_baseline_bytes;
    // Prepared entries are codec owners. Reserve their vector only after all
    // warm contexts and scratch pools have entered the same live-byte ledger.
    let unique_input_count = adjacent_unique_input_count(jobs);
    let mut prepared = match try_new_vec_with_live_budget(
        unique_input_count,
        &mut allocation_live_bytes,
        JPEG_CODEC_HOST_CAP_BYTES,
    ) {
        Ok(prepared) => prepared,
        Err(AllocationBudgetError::MemoryCapExceeded { .. }) if retained_context_bytes != 0 => {
            release_planning_allocations(workers, &mut owned_context)?;
            allocation_baseline_bytes = 0;
            allocation_live_bytes = 0;
            match try_new_vec_with_live_budget(
                unique_input_count,
                &mut allocation_live_bytes,
                JPEG_CODEC_HOST_CAP_BYTES,
            ) {
                Ok(prepared) => prepared,
                Err(_) => return Ok(PreparedRegularPlanning::Fallback),
            }
        }
        Err(_) => return Ok(PreparedRegularPlanning::Fallback),
    };
    let prepared_storage_bytes = allocation_live_bytes
        .checked_sub(allocation_baseline_bytes)
        .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
    let mut retained_prepared_bytes = prepared_storage_bytes;
    let mut content_scan_budget = RECENT_INPUT_CONTENT_SCAN_BUDGET_BYTES;

    for (index, job) in jobs.iter().enumerate() {
        // Inputs stay immutably borrowed for this call, and format/options are
        // fixed for the whole batch. Output buffers and strides remain per-job
        // and are validated independently when the shared decoder executes.
        let decoder_index = if let Some(decoder_index) =
            recent_decoder_index(index, jobs, &decoder_indices, &mut content_scan_budget)?
        {
            decoder_index
        } else {
            let prepared_job = match plan_job(
                job,
                workers,
                &mut owned_context,
                retained_prepared_bytes,
                &mut planner,
            )? {
                Ok(prepared) => prepared,
                Err(
                    JpegError::MemoryCapExceeded { .. } | JpegError::HostAllocationFailed { .. },
                ) => return Ok(PreparedRegularPlanning::Fallback),
                Err(source) => {
                    return Err(BatchDecodeError::Tile(IndexedTileBatchError {
                        index,
                        source,
                    }));
                }
            };
            retained_prepared_bytes =
                match retained_prepared_bytes.checked_add(prepared_job.retained_decoder_bytes) {
                    Some(bytes) if bytes <= JPEG_CODEC_HOST_CAP_BYTES => bytes,
                    _ => return Ok(PreparedRegularPlanning::Fallback),
                };
            let decoder_index = prepared.len();
            prepared.push(prepared_job);
            decoder_index
        };
        let prepared_job = prepared
            .get(decoder_index)
            .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
        let plan = match planned_prepared_job(prepared_job) {
            Ok(plan) => plan,
            Err(source) => {
                return Err(BatchDecodeError::Tile(IndexedTileBatchError {
                    index,
                    source,
                }));
            }
        };
        plans.push(plan);
        decoder_indices.push(decoder_index);
    }
    let planning_context_bytes = if retain_owned_context {
        owned_context.retained_allocation_bytes()
    } else {
        0
    };
    retained_prepared_bytes = match retained_prepared_bytes.checked_add(planning_context_bytes) {
        Some(bytes) if bytes <= JPEG_CODEC_HOST_CAP_BYTES => bytes,
        _ => return Ok(PreparedRegularPlanning::Fallback),
    };
    let planning_context = if retain_owned_context {
        Some(owned_context)
    } else {
        None
    };
    Ok(PreparedRegularPlanning::Prepared(PreparedRegularJobs {
        plans,
        prepared,
        decoder_indices,
        planning_context,
        retained_codec_bytes: retained_prepared_bytes,
    }))
}

pub(super) fn restore_regular_plans(
    plans: &mut [PlannedJob],
    prepared: &[PreparedJpegTileDecode<'_>],
    decoder_indices: &[usize],
) -> Result<(), BatchInfrastructureError> {
    if plans.len() != decoder_indices.len() {
        return Err(BatchInfrastructureError::SchedulerPoisoned);
    }
    for (plan, &decoder_index) in plans.iter_mut().zip(decoder_indices) {
        let prepared = prepared
            .get(decoder_index)
            .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
        *plan = planned_job(prepared.plan);
    }
    Ok(())
}

fn adjacent_unique_input_count(jobs: &[TileDecodeJob<'_, '_>]) -> usize {
    jobs.first().map_or(0, |_| {
        1 + jobs
            .windows(2)
            .filter(|pair| !same_input_storage(pair[0].input, pair[1].input))
            .count()
    })
}

fn recent_decoder_index(
    index: usize,
    jobs: &[TileDecodeJob<'_, '_>],
    decoder_indices: &[usize],
    content_scan_budget: &mut usize,
) -> Result<Option<usize>, BatchInfrastructureError> {
    if index == 0 {
        return Ok(None);
    }

    let previous = index - 1;
    if reusable_input(jobs[index].input, jobs[previous].input, content_scan_budget) {
        return decoder_indices
            .get(previous)
            .copied()
            .map(Some)
            .ok_or(BatchInfrastructureError::SchedulerPoisoned);
    }

    // Bound each job's prepared-decoder reuse search to 32 preceding jobs.
    let start = index.saturating_sub(RECENT_INPUT_LOOKBACK);
    for previous in (start..previous).rev() {
        if reusable_input(jobs[index].input, jobs[previous].input, content_scan_budget) {
            return decoder_indices
                .get(previous)
                .copied()
                .map(Some)
                .ok_or(BatchInfrastructureError::SchedulerPoisoned);
        }
    }
    Ok(None)
}

fn reusable_input(left: &[u8], right: &[u8], content_scan_budget: &mut usize) -> bool {
    if left.len() != right.len() {
        return false;
    }
    if core::ptr::eq(left.as_ptr(), right.as_ptr()) {
        return true;
    }
    if left.len() > RECENT_INPUT_CONTENT_MAX_BYTES || left.len() > *content_scan_budget {
        return false;
    }
    *content_scan_budget -= left.len();
    left == right
}

fn same_input_storage(left: &[u8], right: &[u8]) -> bool {
    left.len() == right.len() && core::ptr::eq(left.as_ptr(), right.as_ptr())
}

fn current_retained_codec_bytes(
    workers: &mut [Mutex<WorkerSlot>],
    owned_context: &DecoderContext,
) -> Result<usize, BatchInfrastructureError> {
    let mut retained = owned_context.retained_allocation_bytes();
    for slot in workers {
        retained = checked_add(
            retained,
            slot.get_mut()
                .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                .retained_bytes(),
            "JPEG retained planning codec bytes",
            JPEG_CODEC_HOST_CAP_BYTES,
        )?;
    }
    Ok(retained)
}

fn release_planning_allocations(
    workers: &mut [Mutex<WorkerSlot>],
    owned_context: &mut DecoderContext,
) -> Result<(), BatchInfrastructureError> {
    for slot in workers {
        slot.get_mut()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
            .release_allocations();
    }
    owned_context.clear();
    Ok(())
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
        match plan_job(job, workers, &mut owned_context, 0, &mut planner)? {
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
        match plan_job(job, workers, &mut owned_context, 0, &mut planner)? {
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

fn planned_prepared_job(prepared: &PreparedJpegTileDecode<'_>) -> Result<PlannedJob, JpegError> {
    let worker_live_bytes = prepared
        .plan
        .worker_live_bytes
        .checked_sub(prepared.retained_decoder_bytes)
        .ok_or(JpegError::InternalInvariant {
            reason: "prepared decoder exceeds its full worker live-byte claim",
        })?;
    Ok(PlannedJob::Decode {
        worker_live_bytes,
        retained_result_bytes: prepared.plan.retained_result_bytes,
    })
}
