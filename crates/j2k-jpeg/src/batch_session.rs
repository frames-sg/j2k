// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent JPEG batch decode session.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use j2k_core::{BatchInfrastructureError, PixelFormat, TileBatchOptions};
use std::sync::Mutex;

use crate::decoder::{
    planned_jpeg_tile_decode_live_bytes, prepare_jpeg_tile_decode, DecodeOutcome, DecodedTile,
    PreparedJpegTileJob, TileBatchError, TileDecodeJob, TileRegionScaledDecodeJob,
    TileScaledDecodeJob,
};
use crate::error::JpegError;
use crate::info::DecodeOptions;

mod allocation;
mod collection;
mod planning;
mod runtime;
mod scheduler;
mod worker;

use allocation::{accept_prepared_batch_plan, ensure_metadata_bytes, vec_capacity_bytes};
use collection::{
    collect_per_tile_results, collect_results, decode_outcome_retained_bytes,
    decoded_tile_retained_bytes, retained_live_bytes,
};
use planning::{
    min_output_len, plan_per_tile_jobs, plan_prepared_regular_jobs, plan_regular_jobs,
    planned_job_chunk, restore_regular_plans, PreparedRegularPlanning,
};
use scheduler::{run_chunks_rayon, run_chunks_scoped};
use worker::WorkerSlot;

const SMALL_OUTPUT_DEFAULT_WORKER_CAP: usize = 4;
const SMALL_OUTPUT_BYTES: usize = 32 * 1024;

type BatchResultSlot<T> = j2k_core::BatchResultSlot<T, JpegError>;

/// Reusable JPEG tile-batch runtime for WSI viewport loops.
///
/// The session keeps one decoder context and scratch pool per active worker
/// across calls. Planning counts retained worker storage against the codec
/// memory allowance and evicts it only when a new request would exceed that
/// allowance. Callers continue to own compressed inputs and
/// decoded output buffers.
#[derive(Debug)]
pub struct JpegBatchSession {
    options: TileBatchOptions,
    workers: Vec<Mutex<WorkerSlot>>,
    active_workers: usize,
    cap_small_output_default_workers: bool,
}

impl Default for JpegBatchSession {
    fn default() -> Self {
        Self::new(TileBatchOptions::default())
    }
}

impl JpegBatchSession {
    /// Create a session using the given CPU batch worker options.
    #[must_use]
    pub fn new(options: TileBatchOptions) -> Self {
        Self {
            options,
            workers: Vec::new(),
            active_workers: 0,
            cap_small_output_default_workers: false,
        }
    }

    pub(crate) fn new_one_shot(options: TileBatchOptions) -> Self {
        Self {
            cap_small_output_default_workers: true,
            ..Self::new(options)
        }
    }

    /// Return current worker options.
    #[must_use]
    pub fn options(&self) -> TileBatchOptions {
        self.options
    }

    /// Replace worker options for future decode calls.
    pub fn set_options(&mut self, options: TileBatchOptions) {
        self.options = options;
    }

    /// Number of active workers used by the most recent non-empty decode call.
    #[must_use]
    #[doc(hidden)]
    pub fn worker_count(&self) -> usize {
        self.active_workers
    }

    /// Number of worker slots retained by the session.
    #[must_use]
    #[doc(hidden)]
    pub fn retained_worker_slots(&self) -> usize {
        self.workers.len()
    }

    /// Clear worker-local decode contexts and scratch pools while retaining slots.
    pub fn reset(&mut self) {
        for slot in &self.workers {
            match slot.lock() {
                Ok(mut worker) => worker.reset(),
                Err(poisoned) => {
                    poisoned.into_inner().reset();
                    slot.clear_poison();
                }
            }
        }
        self.active_workers = 0;
    }

    /// Decode full JPEG tiles into caller-owned output buffers.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first codec failure in input order,
    /// or a typed infrastructure failure when no tile index applies.
    pub fn decode_tiles_into(
        &mut self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        fmt: PixelFormat,
    ) -> Result<Vec<DecodeOutcome>, TileBatchError> {
        self.decode_tiles_into_with_options(jobs, fmt, DecodeOptions::default())
    }

    /// Decode full JPEG tiles with explicit JPEG decode options.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first codec failure in input order,
    /// or a typed infrastructure failure when no tile index applies.
    #[expect(
        clippy::too_many_lines,
        reason = "prepared execution and its allocation-safe fallback share one batch transaction"
    )]
    pub fn decode_tiles_into_with_options(
        &mut self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        fmt: PixelFormat,
        decode_options: DecodeOptions,
    ) -> Result<Vec<DecodeOutcome>, TileBatchError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let job_count = jobs.len();
        let planning_metadata = self.prepare_job_planning(job_count)?;
        let prepared_planning = plan_prepared_regular_jobs(
            jobs,
            planning_metadata,
            &mut self.workers,
            |job, ctx, external_live_bytes| {
                prepare_jpeg_tile_decode(job.input, ctx, fmt, decode_options, external_live_bytes)
            },
        )?;
        let (
            mut plans,
            mut prepared,
            mut decoder_indices,
            mut planning_context,
            retained_prepared_bytes,
        ) = match prepared_planning {
            PreparedRegularPlanning::Prepared(prepared_jobs) => (
                prepared_jobs.plans,
                Some(prepared_jobs.prepared),
                Some(prepared_jobs.decoder_indices),
                prepared_jobs.planning_context,
                prepared_jobs.retained_codec_bytes,
            ),
            PreparedRegularPlanning::Fallback => {
                let mut previous_plan = None;
                let plans = plan_regular_jobs(
                    jobs,
                    planning_metadata,
                    &mut self.workers,
                    |job, ctx, external_live_bytes| {
                        if let Some((input, plan)) = previous_plan {
                            if core::ptr::eq(job.input, input) {
                                return Ok(plan);
                            }
                        }
                        let plan = planned_jpeg_tile_decode_live_bytes(
                            job.input,
                            ctx,
                            fmt,
                            None,
                            j2k_core::Downscale::None,
                            decode_options,
                            external_live_bytes,
                        )?;
                        // The input stays immutably borrowed for this call. Identical
                        // slices and fixed output options have the same codec claim;
                        // the claim already reserves the maximum context footprint.
                        // Reuse only within this call, without retaining input pointers
                        // after the caller can mutate or replace their storage.
                        previous_plan = Some((job.input, plan));
                        Ok(plan)
                    },
                )?;
                (plans, None, None, None, 0)
            }
        };
        let min_output_len = min_output_len(jobs);
        let desired_workers = self.desired_batch_worker_count(job_count, min_output_len);
        let candidate_workers = desired_workers.max(1).min(job_count);
        let candidate_chunk_size = job_count.div_ceil(candidate_workers);
        let expected_worker_count = job_count.div_ceil(candidate_chunk_size);
        let plan_capacity_bytes = vec_capacity_bytes(&plans)?;
        let prepared_metadata_bytes = match &decoder_indices {
            Some(indices) => ensure_metadata_bytes(
                plan_capacity_bytes,
                vec_capacity_bytes(indices)?,
                "JPEG prepared batch metadata",
            )?,
            None => plan_capacity_bytes,
        };
        let attempt = self.prepare_batch_with_shared_codec_bytes::<DecodeOutcome, DecodeOutcome>(
            &plans,
            prepared_metadata_bytes,
            min_output_len,
            retained_prepared_bytes,
        );
        let batch = if prepared.is_some() {
            accept_prepared_batch_plan(attempt, expected_worker_count)?
        } else {
            Some(attempt?)
        };
        let batch = if let Some(batch) = batch {
            batch
        } else {
            let prepared_jobs = prepared
                .take()
                .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
            let prepared_indices = decoder_indices
                .take()
                .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
            restore_regular_plans(&mut plans, &prepared_jobs, &prepared_indices)?;
            drop((prepared_jobs, prepared_indices, planning_context.take()));
            self.prepare_batch::<DecodeOutcome, DecodeOutcome>(
                &plans,
                plan_capacity_bytes,
                min_output_len,
            )?
        };
        if let Some(context) = planning_context.take() {
            let available = self.workers.len();
            self.workers
                .first_mut()
                .ok_or(BatchInfrastructureError::WorkerSlotMissing {
                    worker: 0,
                    available,
                })?
                .get_mut()
                .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?
                .install_planning_context(context);
        }
        let planned_jobs = &plans;
        let results = if let (Some(prepared), Some(decoder_indices)) = (&prepared, &decoder_indices)
        {
            let decode_chunk =
                |worker: &mut WorkerSlot,
                 start_index: usize,
                 chunk: &mut [_],
                 results: &mut [BatchResultSlot<DecodeOutcome>]| {
                    let end = start_index
                        .checked_add(chunk.len())
                        .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
                    let chunk_indices = decoder_indices
                        .get(start_index..end)
                        .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
                    let chunk_plans = planned_job_chunk(planned_jobs, start_index, chunk.len())?;
                    worker.decode_prepared_decoder_job_chunk(
                        chunk,
                        prepared,
                        chunk_indices,
                        results,
                        chunk_plans,
                        fmt,
                    )
                };
            if self.options.workers.is_some() {
                run_chunks_scoped(&self.workers, jobs, batch, decode_chunk)?
            } else {
                run_chunks_rayon(&self.workers, jobs, batch, decode_chunk)?
            }
        } else if prepared.is_none() && decoder_indices.is_none() {
            let decode_chunk =
                |worker: &mut WorkerSlot,
                 start_index: usize,
                 chunk: &mut [_],
                 results: &mut [BatchResultSlot<DecodeOutcome>]| {
                    let chunk_plans = planned_job_chunk(planned_jobs, start_index, chunk.len())?;
                    worker.decode_tile_job_chunk(chunk, results, chunk_plans, fmt, decode_options)
                };
            if self.options.workers.is_some() {
                run_chunks_scoped(&self.workers, jobs, batch, decode_chunk)?
            } else {
                run_chunks_rayon(&self.workers, jobs, batch, decode_chunk)?
            }
        } else {
            return Err(BatchInfrastructureError::SchedulerPoisoned.into());
        };
        drop((prepared, decoder_indices));
        let retained = retained_live_bytes(
            &self.workers,
            vec_capacity_bytes(&self.workers)?,
            vec_capacity_bytes(&plans)?,
            &results,
            decode_outcome_retained_bytes,
        )?;
        collect_results(job_count, results, retained)
    }

    /// Decode prepared TIFF/WSI JPEG tiles into caller-owned RGB8 buffers.
    ///
    /// Results preserve the caller's input order and retain each tile's error
    /// independently instead of collapsing the batch to the first failure.
    ///
    /// # Errors
    ///
    /// Returns a typed batch infrastructure error for planning, allocation,
    /// scheduling, or ordered-collection failures.
    pub fn decode_prepared_jpeg_tiles_rgb8(
        &mut self,
        jobs: &mut [PreparedJpegTileJob<'_, '_>],
    ) -> Result<Vec<Result<DecodedTile, JpegError>>, BatchInfrastructureError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let job_count = jobs.len();
        let planning_metadata = self.prepare_job_planning(job_count)?;
        let plans = plan_per_tile_jobs(
            jobs,
            planning_metadata,
            &mut self.workers,
            |job, ctx, external_live_bytes| {
                planned_jpeg_tile_decode_live_bytes(
                    job.input.as_bytes(),
                    ctx,
                    PixelFormat::Rgb8,
                    None,
                    j2k_core::Downscale::None,
                    job.options,
                    external_live_bytes,
                )
            },
        )?;
        let batch = self.prepare_batch::<DecodedTile, Result<DecodedTile, JpegError>>(
            &plans,
            vec_capacity_bytes(&plans)?,
            min_output_len(jobs),
        )?;
        let planned_jobs = &plans;
        let decode_chunk =
            |worker: &mut WorkerSlot,
             start_index: usize,
             chunk: &mut [_],
             results: &mut [BatchResultSlot<DecodedTile>]| {
                let chunk_plans = planned_job_chunk(planned_jobs, start_index, chunk.len())?;
                worker.decode_prepared_tile_job_chunk(chunk, results, chunk_plans)
            };
        let results = if self.options.workers.is_some() {
            run_chunks_scoped(&self.workers, jobs, batch, decode_chunk)?
        } else {
            run_chunks_rayon(&self.workers, jobs, batch, decode_chunk)?
        };
        let retained = retained_live_bytes(
            &self.workers,
            vec_capacity_bytes(&self.workers)?,
            vec_capacity_bytes(&plans)?,
            &results,
            decoded_tile_retained_bytes,
        )?;
        collect_per_tile_results(job_count, results, retained)
    }

    /// Decode scaled JPEG tiles into caller-owned output buffers.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first codec failure in input order,
    /// or a typed infrastructure failure when no tile index applies.
    pub fn decode_tiles_scaled_into(
        &mut self,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        fmt: PixelFormat,
    ) -> Result<Vec<DecodeOutcome>, TileBatchError> {
        self.decode_tiles_scaled_into_with_options(jobs, fmt, DecodeOptions::default())
    }

    /// Decode scaled JPEG tiles with explicit JPEG decode options.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first codec failure in input order,
    /// or a typed infrastructure failure when no tile index applies.
    pub fn decode_tiles_scaled_into_with_options(
        &mut self,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        fmt: PixelFormat,
        decode_options: DecodeOptions,
    ) -> Result<Vec<DecodeOutcome>, TileBatchError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let job_count = jobs.len();
        let planning_metadata = self.prepare_job_planning(job_count)?;
        let plans = plan_regular_jobs(
            jobs,
            planning_metadata,
            &mut self.workers,
            |job, ctx, external_live_bytes| {
                planned_jpeg_tile_decode_live_bytes(
                    job.input,
                    ctx,
                    fmt,
                    None,
                    job.scale,
                    decode_options,
                    external_live_bytes,
                )
            },
        )?;
        let batch = self.prepare_batch::<DecodeOutcome, DecodeOutcome>(
            &plans,
            vec_capacity_bytes(&plans)?,
            min_output_len(jobs),
        )?;
        let planned_jobs = &plans;
        let decode_chunk =
            |worker: &mut WorkerSlot,
             start_index: usize,
             chunk: &mut [_],
             results: &mut [BatchResultSlot<DecodeOutcome>]| {
                let chunk_plans = planned_job_chunk(planned_jobs, start_index, chunk.len())?;
                worker.decode_tile_scaled_job_chunk(
                    chunk,
                    results,
                    chunk_plans,
                    fmt,
                    decode_options,
                )
            };
        let results = if self.options.workers.is_some() {
            run_chunks_scoped(&self.workers, jobs, batch, decode_chunk)?
        } else {
            run_chunks_rayon(&self.workers, jobs, batch, decode_chunk)?
        };
        let retained = retained_live_bytes(
            &self.workers,
            vec_capacity_bytes(&self.workers)?,
            vec_capacity_bytes(&plans)?,
            &results,
            decode_outcome_retained_bytes,
        )?;
        collect_results(job_count, results, retained)
    }

    /// Decode scaled JPEG tile regions into caller-owned output buffers.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first codec failure in input order,
    /// or a typed infrastructure failure when no tile index applies.
    pub fn decode_tiles_region_scaled_into(
        &mut self,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        fmt: PixelFormat,
    ) -> Result<Vec<DecodeOutcome>, TileBatchError> {
        self.decode_tiles_region_scaled_into_with_options(jobs, fmt, DecodeOptions::default())
    }

    /// Decode scaled JPEG tile regions with explicit JPEG decode options.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first codec failure in input order,
    /// or a typed infrastructure failure when no tile index applies.
    pub fn decode_tiles_region_scaled_into_with_options(
        &mut self,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        fmt: PixelFormat,
        decode_options: DecodeOptions,
    ) -> Result<Vec<DecodeOutcome>, TileBatchError> {
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let job_count = jobs.len();
        let planning_metadata = self.prepare_job_planning(job_count)?;
        let plans = plan_regular_jobs(
            jobs,
            planning_metadata,
            &mut self.workers,
            |job, ctx, external_live_bytes| {
                planned_jpeg_tile_decode_live_bytes(
                    job.input,
                    ctx,
                    fmt,
                    Some(job.roi.into()),
                    job.scale,
                    decode_options,
                    external_live_bytes,
                )
            },
        )?;
        let batch = self.prepare_batch::<DecodeOutcome, DecodeOutcome>(
            &plans,
            vec_capacity_bytes(&plans)?,
            min_output_len(jobs),
        )?;
        let planned_jobs = &plans;
        let decode_chunk =
            |worker: &mut WorkerSlot,
             start_index: usize,
             chunk: &mut [_],
             results: &mut [BatchResultSlot<DecodeOutcome>]| {
                let chunk_plans = planned_job_chunk(planned_jobs, start_index, chunk.len())?;
                worker.decode_tile_region_scaled_job_chunk(
                    chunk,
                    results,
                    chunk_plans,
                    fmt,
                    decode_options,
                )
            };
        let results = if self.options.workers.is_some() {
            run_chunks_scoped(&self.workers, jobs, batch, decode_chunk)?
        } else {
            run_chunks_rayon(&self.workers, jobs, batch, decode_chunk)?
        };
        let retained = retained_live_bytes(
            &self.workers,
            vec_capacity_bytes(&self.workers)?,
            vec_capacity_bytes(&plans)?,
            &results,
            decode_outcome_retained_bytes,
        )?;
        collect_results(job_count, results, retained)
    }
}

fn available_tile_batch_workers() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

#[cfg(test)]
mod tests;
