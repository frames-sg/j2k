// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persistent JPEG batch decode session.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use j2k_core::{
    collect_indexed_batch_results, tile_batch_worker_count, CodecContext, IndexedBatchResult,
    PixelFormat, ScratchPool as CoreScratchPool, TileBatchOptions,
};
use rayon::prelude::*;
use std::sync::{Mutex, MutexGuard};

use crate::context::DecoderContext;
use crate::decoder::{
    decode_prepared_jpeg_tile_rgb8_in_context, decode_tile_into_in_context_with_options,
    decode_tile_region_scaled_into_in_context_with_options,
    decode_tile_scaled_into_in_context_with_options, DecodeOutcome, DecodedTile,
    PreparedJpegTileJob, TileBatchError, TileDecodeJob, TileDecodeOutput,
    TileRegionScaledDecodeJob, TileScaledDecodeJob,
};
use crate::error::JpegError;
use crate::info::DecodeOptions;
use crate::internal::scratch::ScratchPool;

const SMALL_OUTPUT_DEFAULT_WORKER_CAP: usize = 4;
const SMALL_OUTPUT_BYTES: usize = 32 * 1024;

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
                TileDecodeOutput {
                    out: job.out,
                    stride: job.stride,
                    fmt,
                },
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
                TileDecodeOutput {
                    out: job.out,
                    stride: job.stride,
                    fmt,
                },
                job.roi.into(),
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
        let chunk_size = self.prepare_chunks(job_count, min_output_len(jobs));
        let decode_chunk = |worker: &mut WorkerSlot, start_index, chunk: &mut [_]| {
            worker.decode_tile_job_chunk(start_index, chunk, fmt, decode_options)
        };
        let results = if self.options.workers.is_some() {
            self.run_chunks_scoped(jobs, chunk_size, decode_chunk)
        } else {
            self.run_chunks_rayon(jobs, chunk_size, decode_chunk)
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
        let chunk_size = self.prepare_chunks(job_count, min_output_len(jobs));
        let decode_chunk = |worker: &mut WorkerSlot, start_index, chunk: &mut [_]| {
            worker.decode_prepared_tile_job_chunk(start_index, chunk)
        };
        let results = if self.options.workers.is_some() {
            self.run_chunks_scoped(jobs, chunk_size, decode_chunk)
        } else {
            self.run_chunks_rayon(jobs, chunk_size, decode_chunk)
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
        let chunk_size = self.prepare_chunks(job_count, min_output_len(jobs));
        let decode_chunk = |worker: &mut WorkerSlot, start_index, chunk: &mut [_]| {
            worker.decode_tile_scaled_job_chunk(start_index, chunk, fmt, decode_options)
        };
        let results = if self.options.workers.is_some() {
            self.run_chunks_scoped(jobs, chunk_size, decode_chunk)
        } else {
            self.run_chunks_rayon(jobs, chunk_size, decode_chunk)
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
        let chunk_size = self.prepare_chunks(job_count, min_output_len(jobs));
        let decode_chunk = |worker: &mut WorkerSlot, start_index, chunk: &mut [_]| {
            worker.decode_tile_region_scaled_job_chunk(start_index, chunk, fmt, decode_options)
        };
        let results = if self.options.workers.is_some() {
            self.run_chunks_scoped(jobs, chunk_size, decode_chunk)
        } else {
            self.run_chunks_rayon(jobs, chunk_size, decode_chunk)
        };
        collect_results(job_count, results)
    }

    fn prepare_chunks(&mut self, job_count: usize, min_output_len: usize) -> usize {
        let mut worker_count =
            tile_batch_worker_count(job_count, self.options, available_tile_batch_workers());
        let small_output_default_batch = self.cap_small_output_default_workers
            && self.options.workers.is_none()
            && min_output_len <= SMALL_OUTPUT_BYTES;
        if small_output_default_batch {
            worker_count = worker_count.min(SMALL_OUTPUT_DEFAULT_WORKER_CAP);
        }
        self.ensure_worker_slots(worker_count);
        self.active_workers = worker_count;
        job_count.div_ceil(worker_count)
    }

    fn ensure_worker_slots(&mut self, worker_count: usize) {
        while self.workers.len() < worker_count {
            self.workers.push(Mutex::new(WorkerSlot::default()));
        }
    }

    fn run_chunks_rayon<T, R, F>(
        &self,
        jobs: &mut [T],
        chunk_size: usize,
        decode_chunk: F,
    ) -> Vec<IndexedBatchResult<R, JpegError>>
    where
        T: Send,
        R: Send,
        F: Fn(&mut WorkerSlot, usize, &mut [T]) -> Vec<IndexedBatchResult<R, JpegError>> + Sync,
    {
        jobs.par_chunks_mut(chunk_size)
            .enumerate()
            .flat_map_iter(|(chunk_index, chunk)| {
                let start_index = chunk_index * chunk_size;
                decode_chunk(
                    &mut lock_worker(&self.workers[chunk_index]),
                    start_index,
                    chunk,
                )
            })
            .collect()
    }

    fn run_chunks_scoped<T, R, F>(
        &self,
        jobs: &mut [T],
        chunk_size: usize,
        decode_chunk: F,
    ) -> Vec<IndexedBatchResult<R, JpegError>>
    where
        T: Send,
        R: Send,
        F: Fn(&mut WorkerSlot, usize, &mut [T]) -> Vec<IndexedBatchResult<R, JpegError>> + Sync,
    {
        if jobs.len() <= chunk_size {
            return decode_chunk(&mut lock_worker(&self.workers[0]), 0, jobs);
        }

        let decode_chunk = &decode_chunk;
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for (chunk_index, chunk) in jobs.chunks_mut(chunk_size).enumerate() {
                let start_index = chunk_index * chunk_size;
                let worker = &self.workers[chunk_index];
                handles.push(
                    scope.spawn(move || decode_chunk(&mut lock_worker(worker), start_index, chunk)),
                );
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

trait BatchJobOutput {
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

fn min_output_len<T: BatchJobOutput>(jobs: &[T]) -> usize {
    jobs.iter()
        .map(BatchJobOutput::out_len)
        .min()
        .expect("non-empty batch has an output buffer")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::Decoder;
    use j2k_test_support::JPEG_BASELINE_420_16X16;

    #[test]
    fn one_shot_session_caps_default_workers_for_small_outputs() {
        const JOBS: usize = 64;
        let info = Decoder::inspect(JPEG_BASELINE_420_16X16).expect("fixture inspect");
        let stride = info.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
        let len = stride * info.dimensions.1 as usize;
        let mut outputs = (0..JOBS).map(|_| vec![0u8; len]).collect::<Vec<_>>();
        let mut session = JpegBatchSession::new_one_shot(TileBatchOptions::default());

        let outcomes = {
            let mut jobs = outputs
                .iter_mut()
                .map(|out| TileDecodeJob {
                    input: JPEG_BASELINE_420_16X16,
                    out: out.as_mut_slice(),
                    stride,
                })
                .collect::<Vec<_>>();
            session
                .decode_tiles_into(&mut jobs, PixelFormat::Rgb8)
                .expect("one-shot session decode")
        };

        let available = available_tile_batch_workers();
        assert_eq!(outcomes.len(), JOBS);
        assert_eq!(
            session.worker_count(),
            available.min(SMALL_OUTPUT_DEFAULT_WORKER_CAP).min(JOBS)
        );
    }
}
