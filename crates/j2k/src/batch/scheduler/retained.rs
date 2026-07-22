// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scoped scheduling over worker contexts retained by an owned batch session.

use j2k_core::{BatchInfrastructureError, TileBatchOptions};

use super::{join_batch_workers, plan_batch, ScopedWorkerHandle};
use crate::batch::allocation::try_vec_with_capacity;
use crate::batch::worker::BatchWorker;

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
