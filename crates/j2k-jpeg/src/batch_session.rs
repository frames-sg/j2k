// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent JPEG batch decode session.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use j2k_core::{BatchInfrastructureError, PixelFormat, TileBatchOptions};
use std::sync::Mutex;

use crate::decoder::{
    planned_jpeg_tile_decode_live_bytes, DecodeOutcome, DecodedTile, PreparedJpegTileJob,
    TileBatchError, TileDecodeJob, TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use crate::error::JpegError;
use crate::info::DecodeOptions;

mod allocation;
mod collection;
mod planning;
mod runtime;
mod scheduler;
mod worker;

use allocation::vec_capacity_bytes;
use collection::{
    collect_per_tile_results, collect_results, decode_outcome_retained_bytes,
    decoded_tile_retained_bytes, retained_live_bytes,
};
use planning::{min_output_len, plan_per_tile_jobs, plan_regular_jobs, planned_job_chunk};
use scheduler::{run_chunks_rayon, run_chunks_scoped};
use worker::WorkerSlot;

const SMALL_OUTPUT_DEFAULT_WORKER_CAP: usize = 4;
const SMALL_OUTPUT_BYTES: usize = 32 * 1024;

type BatchResultSlot<T> = j2k_core::BatchResultSlot<T, JpegError>;

/// Reusable JPEG tile-batch runtime for WSI viewport loops.
///
/// The session keeps one decoder context and scratch pool per active worker
/// during a batch and retains worker slots across calls. Before planning the
/// next batch it may preserve one bounded decoder context, while evicting
/// stale scratch and other contexts so the planning decoder retains the full
/// codec memory allowance. Callers continue to own compressed inputs and
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
        let planning_context = self.planning_context()?;
        let plans = plan_regular_jobs(jobs, planning_metadata, planning_context, |job, ctx| {
            planned_jpeg_tile_decode_live_bytes(
                job.input,
                ctx,
                fmt,
                None,
                j2k_core::Downscale::None,
                decode_options,
            )
        })?;
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
                worker.decode_tile_job_chunk(chunk, results, chunk_plans, fmt, decode_options)
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
        let planning_context = self.planning_context()?;
        let plans = plan_per_tile_jobs(jobs, planning_metadata, planning_context, |job, ctx| {
            planned_jpeg_tile_decode_live_bytes(
                job.input.as_bytes(),
                ctx,
                PixelFormat::Rgb8,
                None,
                j2k_core::Downscale::None,
                job.options,
            )
        })?;
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
        let planning_context = self.planning_context()?;
        let plans = plan_regular_jobs(jobs, planning_metadata, planning_context, |job, ctx| {
            planned_jpeg_tile_decode_live_bytes(
                job.input,
                ctx,
                fmt,
                None,
                job.scale,
                decode_options,
            )
        })?;
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
        let planning_context = self.planning_context()?;
        let plans = plan_regular_jobs(jobs, planning_metadata, planning_context, |job, ctx| {
            planned_jpeg_tile_decode_live_bytes(
                job.input,
                ctx,
                fmt,
                Some(job.roi.into()),
                job.scale,
                decode_options,
            )
        })?;
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
