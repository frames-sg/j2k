// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scoped scheduling over worker contexts retained by an owned batch session.

use alloc::vec::Vec;

use j2k_core::{BatchInfrastructureError, PixelFormat, TileBatchOptions};

use super::{
    allocate_result_slots, collect_results, join_batch_workers, plan_batch, ScopedWorkerHandle,
};
use crate::batch::allocation::{
    actual_warning_owner_bytes, ensure_execution_capacity, ensure_pre_execution_capacity,
    try_vec_with_capacity,
};
use crate::batch::worker::BatchWorker;
use crate::batch::{BatchOutcome, TileBatchError, TileDecodeJob};
use crate::DecodeSettings;

/// Parallelism each retained worker uses inside one image decode for a batch
/// of `job_count` jobs; see `with_inner_parallelism`.
pub(crate) fn retained_inner_parallelism(
    job_count: usize,
    options: TileBatchOptions,
) -> Result<crate::CpuDecodeParallelism, BatchInfrastructureError> {
    Ok(plan_batch(job_count, options)?.inner_parallelism)
}

/// Run disjoint jobs without destroying worker-owned scratch after each call.
pub(crate) fn run_retained_chunks<J, R, F>(
    workers: &mut [BatchWorker],
    jobs: &mut [J],
    results: &mut [Option<R>],
    options: TileBatchOptions,
    decode_chunk: F,
) -> Result<(), BatchInfrastructureError>
where
    J: Send,
    R: Send,
    F: Fn(&mut BatchWorker, &mut [J], &mut [Option<R>]) -> Result<(), BatchInfrastructureError>
        + Sync,
{
    if jobs.is_empty() {
        return Ok(());
    }
    if jobs.len() != results.len() {
        return Err(BatchInfrastructureError::MissingResult {
            index: results.len().min(jobs.len()),
        });
    }
    let plan = plan_batch(jobs.len(), options)?;
    for worker in workers.iter_mut() {
        worker.set_owned_inner_parallelism(plan.inner_parallelism);
    }
    if workers.len() < plan.worker_count {
        return Err(BatchInfrastructureError::WorkerSlotMissing {
            worker: workers.len(),
            available: workers.len(),
        });
    }
    if plan.worker_count == 1 {
        return decode_chunk(&mut workers[0], jobs, results);
    }

    std::thread::scope(|scope| {
        let mut handles = try_vec_with_capacity::<ScopedWorkerHandle<'_>>(
            plan.worker_count,
            "J2K retained scoped worker handles",
        )?;
        let mut spawn_error = None;
        for (worker_index, ((worker, job_chunk), result_chunk)) in workers
            .iter_mut()
            .take(plan.worker_count)
            .zip(jobs.chunks_mut(plan.chunk_size))
            .zip(results.chunks_mut(plan.chunk_size))
            .enumerate()
        {
            if let Ok(handle) = std::thread::Builder::new()
                .spawn_scoped(scope, || decode_chunk(worker, job_chunk, result_chunk))
            {
                handles.push((worker_index, handle));
            } else {
                spawn_error = Some(BatchInfrastructureError::WorkerSpawnFailed {
                    worker: worker_index,
                });
                break;
            }
        }
        join_batch_workers(handles, spawn_error)
    })
}

/// Decode full-tile jobs on retained workers with the one-shot path's
/// admission plan, capacity checks, and ordered result collection.
pub(crate) fn decode_retained_tiles(
    workers: &mut [BatchWorker],
    jobs: &mut [TileDecodeJob<'_, '_>],
    fmt: PixelFormat,
    options: TileBatchOptions,
    settings: DecodeSettings,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let plan = plan_batch(jobs.len(), options)?;
    let (mut results, capacity_extra) = allocate_result_slots(jobs.len())?;
    ensure_pre_execution_capacity(plan, capacity_extra)?;
    run_retained_chunks(
        workers,
        jobs,
        &mut results,
        options,
        |worker, chunk, slots| worker.decode_retained_tile_jobs(chunk, slots, fmt, settings),
    )?;
    let warning_bytes = actual_warning_owner_bytes(&results)?;
    ensure_execution_capacity(plan, capacity_extra, warning_bytes)?;
    collect_results(results)
}
