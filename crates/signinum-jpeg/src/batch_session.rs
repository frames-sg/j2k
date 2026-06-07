// SPDX-License-Identifier: Apache-2.0

//! Persistent JPEG batch decode session.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use rayon::prelude::*;
use signinum_core::{
    collect_indexed_batch_results, tile_batch_worker_count, CodecContext, IndexedBatchResult,
    PixelFormat, ScratchPool as CoreScratchPool, TileBatchOptions,
};
use std::sync::{Mutex, MutexGuard};

use crate::context::DecoderContext;
use crate::decoder::{
    decode_prepared_jpeg_tile_rgb8_in_context, decode_tile_into_in_context_with_options,
    decode_tile_region_scaled_into_in_context_with_options,
    decode_tile_scaled_into_in_context_with_options, DecodeOutcome, DecodedTile,
    PreparedJpegTileJob, TileBatchError, TileDecodeJob, TileRegionScaledDecodeJob,
    TileScaledDecodeJob,
};
use crate::error::JpegError;
use crate::info::DecodeOptions;
use crate::internal::scratch::ScratchPool;

#[derive(Debug, Default)]
struct WorkerSlot {
    ctx: DecoderContext,
    pool: ScratchPool,
}

impl WorkerSlot {
    fn decode_tile_job_chunk(
        &mut self,
        start_index: usize,
        jobs: &mut [TileDecodeJob<'_, '_>],
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        let mut results = Vec::with_capacity(jobs.len());
        for (local_index, job) in jobs.iter_mut().enumerate() {
            let outcome = decode_tile_into_in_context_with_options(
                job.input,
                &mut self.ctx,
                &mut self.pool,
                job.out,
                job.stride,
                fmt,
                options,
            );
            results.push((start_index + local_index, outcome));
        }
        results
    }

    fn decode_prepared_tile_job_chunk(
        &mut self,
        start_index: usize,
        jobs: &mut [PreparedJpegTileJob<'_, '_>],
    ) -> Vec<IndexedBatchResult<DecodedTile, JpegError>> {
        let mut results = Vec::with_capacity(jobs.len());
        for (local_index, job) in jobs.iter_mut().enumerate() {
            let outcome = decode_prepared_jpeg_tile_rgb8_in_context(
                &job.input,
                &mut self.ctx,
                &mut self.pool,
                job.out,
                job.stride,
                job.options,
            );
            results.push((start_index + local_index, outcome));
        }
        results
    }

    fn decode_tile_scaled_job_chunk(
        &mut self,
        start_index: usize,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        let mut results = Vec::with_capacity(jobs.len());
        for (local_index, job) in jobs.iter_mut().enumerate() {
            let outcome = decode_tile_scaled_into_in_context_with_options(
                job.input,
                &mut self.ctx,
                &mut self.pool,
                job.out,
                job.stride,
                fmt,
                job.scale,
                options,
            );
            results.push((start_index + local_index, outcome));
        }
        results
    }

    fn decode_tile_region_scaled_job_chunk(
        &mut self,
        start_index: usize,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        let mut results = Vec::with_capacity(jobs.len());
        for (local_index, job) in jobs.iter_mut().enumerate() {
            let outcome = decode_tile_region_scaled_into_in_context_with_options(
                job.input,
                &mut self.ctx,
                &mut self.pool,
                job.out,
                job.stride,
                fmt,
                job.roi,
                job.scale,
                options,
            );
            results.push((start_index + local_index, outcome));
        }
        results
    }

    fn reset(&mut self) {
        self.ctx.clear();
        self.pool.reset();
    }
}

/// Reusable JPEG tile-batch runtime for WSI viewport loops.
///
/// The session keeps one decoder context and scratch pool per active worker.
/// Reusing it across calls amortizes table/plan caches and heap scratch while
/// callers continue to own compressed inputs and decoded output buffers.
#[derive(Debug)]
pub struct JpegBatchSession {
    options: TileBatchOptions,
    workers: Vec<Mutex<WorkerSlot>>,
    active_workers: usize,
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
    pub fn worker_count(&self) -> usize {
        self.active_workers
    }

    /// Number of worker slots retained by the session.
    #[must_use]
    pub fn retained_worker_slots(&self) -> usize {
        self.workers.len()
    }

    /// Clear worker-local decode contexts and scratch pools while retaining slots.
    pub fn reset(&mut self) {
        for slot in &self.workers {
            lock_worker(slot).reset();
        }
        self.active_workers = 0;
    }

    /// Decode full JPEG tiles into caller-owned output buffers.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first failing tile index in input order.
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
    /// Returns [`TileBatchError`] with the first failing tile index in input order.
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
        let chunk_size = self.prepare_chunks(job_count);
        let results = if self.options.workers.is_some() {
            self.decode_tile_job_chunks_scoped(jobs, chunk_size, fmt, decode_options)
        } else {
            self.decode_tile_job_chunks_rayon(jobs, chunk_size, fmt, decode_options)
        };
        collect_results(job_count, results)
    }

    /// Decode prepared TIFF/WSI JPEG tiles into caller-owned RGB8 buffers.
    ///
    /// Results preserve the caller's input order and retain each tile's error
    /// independently instead of collapsing the batch to the first failure.
    #[must_use]
    pub fn decode_prepared_jpeg_tiles_rgb8(
        &mut self,
        jobs: &mut [PreparedJpegTileJob<'_, '_>],
    ) -> Vec<Result<DecodedTile, JpegError>> {
        if jobs.is_empty() {
            return Vec::new();
        }
        let job_count = jobs.len();
        let chunk_size = self.prepare_chunks(job_count);
        let results = if self.options.workers.is_some() {
            self.decode_prepared_tile_job_chunks_scoped(jobs, chunk_size)
        } else {
            self.decode_prepared_tile_job_chunks_rayon(jobs, chunk_size)
        };
        collect_per_tile_results(job_count, results)
    }

    /// Decode scaled JPEG tiles into caller-owned output buffers.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first failing tile index in input order.
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
    /// Returns [`TileBatchError`] with the first failing tile index in input order.
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
        let chunk_size = self.prepare_chunks(job_count);
        let results = if self.options.workers.is_some() {
            self.decode_tile_scaled_job_chunks_scoped(jobs, chunk_size, fmt, decode_options)
        } else {
            self.decode_tile_scaled_job_chunks_rayon(jobs, chunk_size, fmt, decode_options)
        };
        collect_results(job_count, results)
    }

    /// Decode scaled JPEG tile regions into caller-owned output buffers.
    ///
    /// # Errors
    /// Returns [`TileBatchError`] with the first failing tile index in input order.
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
    /// Returns [`TileBatchError`] with the first failing tile index in input order.
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
        let chunk_size = self.prepare_chunks(job_count);
        let results = if self.options.workers.is_some() {
            self.decode_tile_region_scaled_job_chunks_scoped(jobs, chunk_size, fmt, decode_options)
        } else {
            self.decode_tile_region_scaled_job_chunks_rayon(jobs, chunk_size, fmt, decode_options)
        };
        collect_results(job_count, results)
    }

    fn prepare_chunks(&mut self, job_count: usize) -> usize {
        let worker_count =
            tile_batch_worker_count(job_count, self.options, available_tile_batch_workers());
        self.ensure_worker_slots(worker_count);
        self.active_workers = worker_count;
        job_count.div_ceil(worker_count)
    }

    fn ensure_worker_slots(&mut self, worker_count: usize) {
        while self.workers.len() < worker_count {
            self.workers.push(Mutex::new(WorkerSlot::default()));
        }
    }

    fn decode_tile_job_chunks_rayon(
        &self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        chunk_size: usize,
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        jobs.par_chunks_mut(chunk_size)
            .enumerate()
            .flat_map_iter(|(chunk_index, chunk)| {
                let start_index = chunk_index * chunk_size;
                lock_worker(&self.workers[chunk_index]).decode_tile_job_chunk(
                    start_index,
                    chunk,
                    fmt,
                    options,
                )
            })
            .collect()
    }

    fn decode_tile_job_chunks_scoped(
        &self,
        jobs: &mut [TileDecodeJob<'_, '_>],
        chunk_size: usize,
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        if jobs.len() <= chunk_size {
            return lock_worker(&self.workers[0]).decode_tile_job_chunk(0, jobs, fmt, options);
        }

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let worker = &self.workers[chunk_index];
                handles.push(scope.spawn(move || {
                    lock_worker(worker).decode_tile_job_chunk(start_index, chunk, fmt, options)
                }));
            }
            collect_joined_batch_results(handles)
        })
    }

    fn decode_prepared_tile_job_chunks_rayon(
        &self,
        jobs: &mut [PreparedJpegTileJob<'_, '_>],
        chunk_size: usize,
    ) -> Vec<IndexedBatchResult<DecodedTile, JpegError>> {
        jobs.par_chunks_mut(chunk_size)
            .enumerate()
            .flat_map_iter(|(chunk_index, chunk)| {
                let start_index = chunk_index * chunk_size;
                lock_worker(&self.workers[chunk_index])
                    .decode_prepared_tile_job_chunk(start_index, chunk)
            })
            .collect()
    }

    fn decode_prepared_tile_job_chunks_scoped(
        &self,
        jobs: &mut [PreparedJpegTileJob<'_, '_>],
        chunk_size: usize,
    ) -> Vec<IndexedBatchResult<DecodedTile, JpegError>> {
        if jobs.len() <= chunk_size {
            return lock_worker(&self.workers[0]).decode_prepared_tile_job_chunk(0, jobs);
        }

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let worker = &self.workers[chunk_index];
                handles.push(scope.spawn(move || {
                    lock_worker(worker).decode_prepared_tile_job_chunk(start_index, chunk)
                }));
            }
            collect_joined_batch_results(handles)
        })
    }

    fn decode_tile_scaled_job_chunks_rayon(
        &self,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        chunk_size: usize,
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        jobs.par_chunks_mut(chunk_size)
            .enumerate()
            .flat_map_iter(|(chunk_index, chunk)| {
                let start_index = chunk_index * chunk_size;
                lock_worker(&self.workers[chunk_index]).decode_tile_scaled_job_chunk(
                    start_index,
                    chunk,
                    fmt,
                    options,
                )
            })
            .collect()
    }

    fn decode_tile_scaled_job_chunks_scoped(
        &self,
        jobs: &mut [TileScaledDecodeJob<'_, '_>],
        chunk_size: usize,
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        if jobs.len() <= chunk_size {
            return lock_worker(&self.workers[0])
                .decode_tile_scaled_job_chunk(0, jobs, fmt, options);
        }

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let worker = &self.workers[chunk_index];
                handles.push(scope.spawn(move || {
                    lock_worker(worker).decode_tile_scaled_job_chunk(
                        start_index,
                        chunk,
                        fmt,
                        options,
                    )
                }));
            }
            collect_joined_batch_results(handles)
        })
    }

    fn decode_tile_region_scaled_job_chunks_rayon(
        &self,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        chunk_size: usize,
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        jobs.par_chunks_mut(chunk_size)
            .enumerate()
            .flat_map_iter(|(chunk_index, chunk)| {
                let start_index = chunk_index * chunk_size;
                lock_worker(&self.workers[chunk_index]).decode_tile_region_scaled_job_chunk(
                    start_index,
                    chunk,
                    fmt,
                    options,
                )
            })
            .collect()
    }

    fn decode_tile_region_scaled_job_chunks_scoped(
        &self,
        jobs: &mut [TileRegionScaledDecodeJob<'_, '_>],
        chunk_size: usize,
        fmt: PixelFormat,
        options: DecodeOptions,
    ) -> Vec<IndexedBatchResult<DecodeOutcome, JpegError>> {
        if jobs.len() <= chunk_size {
            return lock_worker(&self.workers[0])
                .decode_tile_region_scaled_job_chunk(0, jobs, fmt, options);
        }

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let worker = &self.workers[chunk_index];
                handles.push(scope.spawn(move || {
                    lock_worker(worker).decode_tile_region_scaled_job_chunk(
                        start_index,
                        chunk,
                        fmt,
                        options,
                    )
                }));
            }
            collect_joined_batch_results(handles)
        })
    }
}

fn available_tile_batch_workers() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

fn lock_worker(slot: &Mutex<WorkerSlot>) -> MutexGuard<'_, WorkerSlot> {
    slot.lock().expect("JPEG batch worker slot poisoned")
}

fn collect_results(
    job_count: usize,
    results: Vec<IndexedBatchResult<DecodeOutcome, JpegError>>,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    collect_indexed_batch_results(job_count, results, |index, source| TileBatchError {
        index,
        source,
    })
}

fn collect_per_tile_results<T>(
    job_count: usize,
    results: Vec<IndexedBatchResult<T, JpegError>>,
) -> Vec<Result<T, JpegError>> {
    let mut ordered = core::iter::repeat_with(|| None)
        .take(job_count)
        .collect::<Vec<_>>();
    for (index, result) in results {
        ordered[index] = Some(result);
    }
    ordered
        .into_iter()
        .map(|slot| slot.expect("JPEG prepared batch worker omitted a tile result"))
        .collect()
}

fn collect_joined_batch_results<T>(
    handles: Vec<std::thread::ScopedJoinHandle<'_, Vec<IndexedBatchResult<T, JpegError>>>>,
) -> Vec<IndexedBatchResult<T, JpegError>> {
    let mut results = Vec::new();
    for handle in handles {
        match handle.join() {
            Ok(chunk_results) => results.extend(chunk_results),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    results
}
