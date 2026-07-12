// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible scoped scheduling over one preallocated disjoint result-slot vector.

use alloc::vec::Vec;
use core::num::NonZeroUsize;
use std::sync::Arc;

use j2k_core::{
    tile_batch_worker_count, try_collect_ordered_batch_results_with_limits,
    BatchInfrastructureError, TileBatchOptions,
};

use super::admission::BatchAllocationBudget;
use super::allocation::{
    actual_warning_owner_bytes, capacity_extra_bytes, ensure_execution_capacity,
    ensure_pre_execution_capacity, try_vec_with_capacity, J2K_BATCH_HOST_CAP_BYTES,
    J2K_BATCH_METADATA_ALLOWANCE_BYTES,
};
use super::planning::{select_batch_plan, select_direct_batch_plan, BatchPlan};
use super::worker::BatchWorker;
use super::{BatchOutcome, J2kBatchResultSlot, TileBatchError};

pub(super) type ScopedWorkerHandle<'scope> = (
    usize,
    std::thread::ScopedJoinHandle<'scope, Result<(), BatchInfrastructureError>>,
);

pub(super) fn decode_batch<J, F>(
    jobs: &mut [J],
    options: TileBatchOptions,
    decode_chunk: F,
) -> Result<Vec<BatchOutcome>, TileBatchError>
where
    J: Send,
    F: Fn(
            &mut BatchWorker,
            &mut [J],
            &mut [J2kBatchResultSlot],
        ) -> Result<(), BatchInfrastructureError>
        + Sync,
{
    if jobs.is_empty() {
        return Ok(Vec::new());
    }
    let plan = plan_batch(jobs.len(), options)?;
    let results = run_chunks_scoped(jobs, plan, None, decode_chunk)?;
    collect_results(results)
}

pub(super) fn plan_direct_batch(
    job_count: usize,
    options: TileBatchOptions,
) -> Result<BatchPlan, BatchInfrastructureError> {
    let desired_workers =
        tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    select_direct_batch_plan(job_count, desired_workers)
}

pub(super) fn plan_batch(
    job_count: usize,
    options: TileBatchOptions,
) -> Result<BatchPlan, BatchInfrastructureError> {
    let desired_workers =
        tile_batch_worker_count(job_count, options, available_tile_batch_workers());
    select_batch_plan(job_count, desired_workers)
}

pub(super) fn run_chunks_scoped<J, F>(
    jobs: &mut [J],
    plan: BatchPlan,
    allocation_budget: Option<Arc<BatchAllocationBudget>>,
    decode_chunk: F,
) -> Result<Vec<J2kBatchResultSlot>, BatchInfrastructureError>
where
    J: Send,
    F: Fn(
            &mut BatchWorker,
            &mut [J],
            &mut [J2kBatchResultSlot],
        ) -> Result<(), BatchInfrastructureError>
        + Sync,
{
    let (mut results, result_capacity_extra) = allocate_result_slots(jobs.len())?;
    ensure_pre_execution_capacity(plan, result_capacity_extra)?;

    let handle_capacity_extra = if plan.worker_count == 1 {
        let mut worker = BatchWorker::new(jobs.len(), allocation_budget);
        let decode_result = decode_chunk(&mut worker, jobs, &mut results);
        // Worker contexts and both scratch owners end before collection.
        drop(worker);
        decode_result?;
        0
    } else {
        run_parallel_chunks(
            jobs,
            &mut results,
            plan,
            result_capacity_extra,
            allocation_budget.as_ref(),
            &decode_chunk,
        )?
    };

    let capacity_extra = result_capacity_extra
        .checked_add(handle_capacity_extra)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K scheduler capacity",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    let warning_bytes = actual_warning_owner_bytes(&results)?;
    ensure_execution_capacity(plan, capacity_extra, warning_bytes)?;
    Ok(results)
}

fn run_parallel_chunks<J, F>(
    jobs: &mut [J],
    results: &mut [J2kBatchResultSlot],
    plan: BatchPlan,
    result_capacity_extra: usize,
    allocation_budget: Option<&Arc<BatchAllocationBudget>>,
    decode_chunk: &F,
) -> Result<usize, BatchInfrastructureError>
where
    J: Send,
    F: Fn(
            &mut BatchWorker,
            &mut [J],
            &mut [J2kBatchResultSlot],
        ) -> Result<(), BatchInfrastructureError>
        + Sync,
{
    let job_count = jobs.len();
    std::thread::scope(|scope| {
        let mut handles = try_vec_with_capacity::<ScopedWorkerHandle<'_>>(
            plan.worker_count,
            "J2K scoped worker handles",
        )?;
        let handle_capacity_extra =
            capacity_extra_bytes(plan.worker_count, &handles, "J2K scoped worker handles")?;
        let capacity_extra = result_capacity_extra
            .checked_add(handle_capacity_extra)
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what: "J2K scheduler capacity",
                requested: usize::MAX,
                cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
            })?;
        ensure_pre_execution_capacity(plan, capacity_extra)?;

        let mut spawn_error = None;
        for (worker_index, (job_chunk, result_chunk)) in jobs
            .chunks_mut(plan.chunk_size)
            .zip(results.chunks_mut(plan.chunk_size))
            .enumerate()
        {
            if worker_index >= plan.worker_count {
                spawn_error = Some(BatchInfrastructureError::WorkerSlotMissing {
                    worker: worker_index,
                    available: plan.worker_count,
                });
                break;
            }
            let worker_budget = allocation_budget.cloned();
            let Ok(handle) = std::thread::Builder::new().spawn_scoped(scope, move || {
                let mut worker = BatchWorker::new(job_count, worker_budget);
                let result = decode_chunk(&mut worker, job_chunk, result_chunk);
                drop(worker);
                result
            }) else {
                spawn_error = Some(BatchInfrastructureError::WorkerSpawnFailed {
                    worker: worker_index,
                });
                break;
            };
            handles.push((worker_index, handle));
        }
        join_batch_workers(handles, spawn_error)?;
        Ok(handle_capacity_extra)
    })
}

pub(super) fn collect_results(
    results: Vec<J2kBatchResultSlot>,
) -> Result<Vec<BatchOutcome>, TileBatchError> {
    let job_count = results.len();
    let warning_bytes = actual_warning_owner_bytes(&results)?;
    // Native claims, direct plans, workers, and scoped handles are no longer
    // live here. The shared collector therefore receives only the actual deep
    // warning ownership retained beside the source and ordered vectors.
    try_collect_ordered_batch_results_with_limits(
        job_count,
        results,
        warning_bytes,
        J2K_BATCH_HOST_CAP_BYTES,
        warning_bytes,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )
}

fn join_batch_workers(
    handles: Vec<ScopedWorkerHandle<'_>>,
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
    first_error.map_or(Ok(()), Err)
}

fn allocate_result_slots(
    job_count: usize,
) -> Result<(Vec<J2kBatchResultSlot>, usize), BatchInfrastructureError> {
    let mut results = try_vec_with_capacity(job_count, "J2K ordered worker result slots")?;
    let capacity_extra =
        capacity_extra_bytes(job_count, &results, "J2K ordered worker result slots")?;
    results.resize_with(job_count, || None);
    Ok((results, capacity_extra))
}

fn available_tile_batch_workers() -> usize {
    std::thread::available_parallelism().map_or(1, NonZeroUsize::get)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_batch_is_allocation_free() {
        let mut jobs: [u8; 0] = [];
        let outcomes = decode_batch(&mut jobs, TileBatchOptions::default(), |_, _, _| Ok(()))
            .expect("empty batch");
        assert!(outcomes.is_empty());
    }
}
