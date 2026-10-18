// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;
use std::sync::{Mutex, MutexGuard};

use j2k_core::BatchInfrastructureError;
use rayon::prelude::*;

use super::allocation::{
    ensure_live_domains, try_vec_with_capacity, BatchPlan, JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
};
use super::worker::WorkerSlot;
use super::BatchResultSlot;

pub(super) fn run_chunks_rayon<T, R, F>(
    workers: &[Mutex<WorkerSlot>],
    jobs: &mut [T],
    plan: BatchPlan,
    decode_chunk: F,
) -> Result<Vec<BatchResultSlot<R>>, BatchInfrastructureError>
where
    T: Send,
    R: Send,
    F: Fn(
            &mut WorkerSlot,
            usize,
            &mut [T],
            &mut [BatchResultSlot<R>],
        ) -> Result<(), BatchInfrastructureError>
        + Sync,
{
    let job_count = jobs.len();
    let (mut results, capacity_extra) = allocate_result_slots(job_count, plan)?;
    ensure_plan_capacity(plan, capacity_extra)?;
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        jobs.par_chunks_mut(plan.chunk_size)
            .zip(results.par_chunks_mut(plan.chunk_size))
            .enumerate()
            .try_for_each(|(chunk_index, (chunk, result_chunk))| {
                let start_index = chunk_index.checked_mul(plan.chunk_size).ok_or(
                    BatchInfrastructureError::ResultIndexOutOfBounds {
                        index: usize::MAX,
                        job_count,
                    },
                )?;
                let mut worker = lock_worker(worker_slot(workers, chunk_index)?)?;
                decode_chunk(&mut worker, start_index, chunk, result_chunk)
            })
    }));
    match run {
        Ok(Ok(())) => Ok(results),
        Ok(Err(error)) => Err(error),
        Err(_) => Err(BatchInfrastructureError::ParallelWorkerPanicked),
    }
}

pub(super) fn run_chunks_scoped<T, R, F>(
    workers: &[Mutex<WorkerSlot>],
    jobs: &mut [T],
    plan: BatchPlan,
    decode_chunk: F,
) -> Result<Vec<BatchResultSlot<R>>, BatchInfrastructureError>
where
    T: Send,
    R: Send,
    F: Fn(
            &mut WorkerSlot,
            usize,
            &mut [T],
            &mut [BatchResultSlot<R>],
        ) -> Result<(), BatchInfrastructureError>
        + Sync,
{
    let job_count = jobs.len();
    let (mut results, result_capacity_extra) = allocate_result_slots(job_count, plan)?;
    if plan.worker_count == 1 {
        ensure_plan_capacity(plan, result_capacity_extra)?;
        let mut worker = lock_worker(worker_slot(workers, 0)?)?;
        decode_chunk(&mut worker, 0, jobs, &mut results)?;
        return Ok(results);
    }

    let decode_chunk = &decode_chunk;
    let inline_len = plan.chunk_size.min(job_count);
    let (inline_jobs, spawned_jobs) = jobs.split_at_mut(inline_len);
    let (inline_results, spawned_results) = results.split_at_mut(inline_len);
    let run = std::thread::scope(|scope| {
        let mut handles =
            try_vec_with_capacity(plan.worker_count, "JPEG scoped batch worker handles")?;
        let handle_capacity_extra = capacity_extra_bytes::<(
            usize,
            std::thread::ScopedJoinHandle<'static, Result<(), BatchInfrastructureError>>,
        )>(
            plan.worker_count,
            handles.capacity(),
            "JPEG scoped batch worker handles",
        )?;
        let capacity_extra = result_capacity_extra
            .checked_add(handle_capacity_extra)
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG scheduler capacity",
                requested: usize::MAX,
                cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            })?;
        ensure_plan_capacity(plan, capacity_extra)?;
        let mut spawn_error = None;
        // Launch the remaining chunks first so the caller becomes worker zero
        // without reducing the requested decode concurrency.
        for (spawned_index, (chunk, result_chunk)) in spawned_jobs
            .chunks_mut(plan.chunk_size)
            .zip(spawned_results.chunks_mut(plan.chunk_size))
            .enumerate()
        {
            let chunk_index = spawned_index.checked_add(1).ok_or(
                BatchInfrastructureError::ResultIndexOutOfBounds {
                    index: usize::MAX,
                    job_count,
                },
            )?;
            let start_index = chunk_index.checked_mul(plan.chunk_size).ok_or(
                BatchInfrastructureError::ResultIndexOutOfBounds {
                    index: usize::MAX,
                    job_count,
                },
            )?;
            let worker = worker_slot(workers, chunk_index)?;
            let Ok(handle) = std::thread::Builder::new().spawn_scoped(scope, move || {
                let mut worker = lock_worker(worker)?;
                decode_chunk(&mut worker, start_index, chunk, result_chunk)
            }) else {
                spawn_error = Some(BatchInfrastructureError::WorkerSpawnFailed {
                    worker: chunk_index,
                });
                break;
            };
            handles.push((chunk_index, handle));
        }
        // Worker zero logically precedes every spawned worker, so it still
        // runs after a later spawn failure; that spawn error keeps priority.
        let inline_run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut worker = lock_worker(worker_slot(workers, 0)?)?;
            decode_chunk(&mut worker, 0, inline_jobs, inline_results)
        }));
        let inline_error = match inline_run {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error),
            Err(_) => Some(BatchInfrastructureError::WorkerPanicked { worker: 0 }),
        };
        join_batch_workers(handles, spawn_error.or(inline_error))
    });
    run?;
    Ok(results)
}

pub(super) fn lock_worker(
    slot: &Mutex<WorkerSlot>,
) -> Result<MutexGuard<'_, WorkerSlot>, BatchInfrastructureError> {
    slot.lock()
        .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)
}

fn worker_slot(
    workers: &[Mutex<WorkerSlot>],
    worker: usize,
) -> Result<&Mutex<WorkerSlot>, BatchInfrastructureError> {
    workers
        .get(worker)
        .ok_or(BatchInfrastructureError::WorkerSlotMissing {
            worker,
            available: workers.len(),
        })
}

fn join_batch_workers(
    handles: Vec<(
        usize,
        std::thread::ScopedJoinHandle<'_, Result<(), BatchInfrastructureError>>,
    )>,
    mut first_error: Option<BatchInfrastructureError>,
) -> Result<(), BatchInfrastructureError> {
    for (worker, handle) in handles {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
            Err(_) => {
                if first_error.is_none() {
                    first_error = Some(BatchInfrastructureError::WorkerPanicked { worker });
                }
            }
        }
    }
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn allocate_result_slots<R>(
    job_count: usize,
    plan: BatchPlan,
) -> Result<(Vec<BatchResultSlot<R>>, usize), BatchInfrastructureError> {
    let mut results = try_vec_with_capacity(job_count, "JPEG ordered worker result slots")?;
    let capacity_extra = capacity_extra_bytes::<BatchResultSlot<R>>(
        job_count,
        results.capacity(),
        "JPEG ordered worker result slots",
    )?;
    ensure_plan_capacity(plan, capacity_extra)?;
    results.resize_with(job_count, || None);
    Ok((results, capacity_extra))
}

fn capacity_extra_bytes<T>(
    requested_capacity: usize,
    actual_capacity: usize,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    let requested = requested_capacity.checked_mul(size_of::<T>()).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    let actual = actual_capacity.checked_mul(size_of::<T>()).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    Ok(actual.saturating_sub(requested))
}

fn ensure_plan_capacity(
    plan: BatchPlan,
    capacity_extra: usize,
) -> Result<(), BatchInfrastructureError> {
    let metadata_bytes = plan.metadata_bytes.checked_add(capacity_extra).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG scheduler capacity",
            requested: usize::MAX,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    ensure_live_domains(plan.codec_bytes, metadata_bytes, "JPEG scheduler capacity")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_capacity_delta_has_an_exact_plan_boundary() {
        let plan = BatchPlan {
            worker_count: 1,
            chunk_size: 1,
            live_bytes: JPEG_BATCH_METADATA_ALLOWANCE_BYTES - 8,
            metadata_bytes: JPEG_BATCH_METADATA_ALLOWANCE_BYTES - 8,
            codec_bytes: 0,
        };
        ensure_plan_capacity(plan, 8).expect("exact scheduler capacity");
        assert!(matches!(
            ensure_plan_capacity(plan, 9),
            Err(BatchInfrastructureError::AllocationTooLarge {
                cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
                ..
            })
        ));
    }

    #[test]
    fn missing_worker_slot_is_typed_instead_of_index_panicking() {
        let workers: Vec<Mutex<WorkerSlot>> = Vec::new();
        assert!(matches!(
            worker_slot(&workers, 0),
            Err(BatchInfrastructureError::WorkerSlotMissing {
                worker: 0,
                available: 0,
            })
        ));
    }

    #[test]
    fn scoped_scheduler_keeps_inline_worker_error_precedence() {
        let workers = (0..2)
            .map(|_| Mutex::new(WorkerSlot::default()))
            .collect::<Vec<_>>();
        let mut jobs = [0usize, 1];
        let plan = BatchPlan {
            worker_count: 2,
            chunk_size: 1,
            live_bytes: 0,
            metadata_bytes: 0,
            codec_bytes: 0,
        };

        let error =
            run_chunks_scoped::<_, usize, _>(&workers, &mut jobs, plan, |_, start, _, _| {
                Err(BatchInfrastructureError::MissingResult { index: start })
            })
            .expect_err("both workers fail");

        assert_eq!(error, BatchInfrastructureError::MissingResult { index: 0 });
    }

    #[test]
    fn scoped_scheduler_maps_inline_panic_to_worker_error() {
        let workers = (0..2)
            .map(|_| Mutex::new(WorkerSlot::default()))
            .collect::<Vec<_>>();
        let mut jobs = [0usize, 1];
        let plan = BatchPlan {
            worker_count: 2,
            chunk_size: 1,
            live_bytes: 0,
            metadata_bytes: 0,
            codec_bytes: 0,
        };

        let error =
            run_chunks_scoped::<_, usize, _>(&workers, &mut jobs, plan, |_, start, _, results| {
                assert_ne!(start, 0, "inline worker panic");
                results[0] = Some(Ok(start));
                Ok(())
            })
            .expect_err("inline worker panic");

        assert_eq!(
            error,
            BatchInfrastructureError::WorkerPanicked { worker: 0 }
        );
        assert!(
            workers[0].lock().is_err(),
            "inline panic poisons worker zero"
        );
    }
}
