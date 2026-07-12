// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure aggregate live-set planning and concurrency reduction.

use core::mem::size_of;

use j2k_core::BatchInfrastructureError;

use super::allocation::{
    checked_add, checked_mul, GENERIC_WORKER_CLAIM_BYTES, J2K_BATCH_HOST_CAP_BYTES,
    J2K_BATCH_METADATA_ALLOWANCE_BYTES, MAX_ADMITTED_BATCH_WORKERS, MAX_GENERIC_BATCH_WORKERS,
};
use super::scheduler::ScopedWorkerHandle;
use super::worker::BatchWorker;
use super::{BatchOutcome, J2kBatchResultSlot};
use crate::J2kDecodeWarning;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchPlan {
    pub(super) worker_count: usize,
    pub(super) chunk_size: usize,
    pub(super) live_bytes: usize,
    pub(super) metadata_bytes: usize,
    pub(super) warning_bytes: usize,
}

pub(super) fn select_batch_plan(
    job_count: usize,
    desired_workers: usize,
) -> Result<BatchPlan, BatchInfrastructureError> {
    select_batch_plan_with_worker_limit(
        job_count,
        desired_workers,
        GENERIC_WORKER_CLAIM_BYTES,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        J2K_BATCH_HOST_CAP_BYTES,
        MAX_GENERIC_BATCH_WORKERS,
    )
}

/// Plan only bounded scheduler metadata. Direct and fallback decode allocations
/// are admitted against the unchanged aggregate execution cap at runtime.
pub(super) fn select_direct_batch_plan(
    job_count: usize,
    desired_workers: usize,
) -> Result<BatchPlan, BatchInfrastructureError> {
    select_batch_plan_with_worker_limit(
        job_count,
        desired_workers,
        0,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        J2K_BATCH_HOST_CAP_BYTES,
        MAX_ADMITTED_BATCH_WORKERS,
    )
}

#[cfg(test)]
fn select_batch_plan_with_limits(
    job_count: usize,
    desired_workers: usize,
    worker_claim_bytes: usize,
    metadata_cap: usize,
    aggregate_cap: usize,
) -> Result<BatchPlan, BatchInfrastructureError> {
    select_batch_plan_with_worker_limit(
        job_count,
        desired_workers,
        worker_claim_bytes,
        metadata_cap,
        aggregate_cap,
        MAX_GENERIC_BATCH_WORKERS,
    )
}

fn select_batch_plan_with_worker_limit(
    job_count: usize,
    desired_workers: usize,
    worker_claim_bytes: usize,
    metadata_cap: usize,
    aggregate_cap: usize,
    worker_limit: usize,
) -> Result<BatchPlan, BatchInfrastructureError> {
    if job_count == 0 {
        return Err(BatchInfrastructureError::EmptyBatchPlan);
    }
    let desired_workers = desired_workers.max(1).min(job_count).min(worker_limit);
    let (minimum_metadata, _) = batch_metadata_bytes(job_count, 1)?;
    if minimum_metadata > metadata_cap {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch metadata",
            requested: minimum_metadata,
            cap: metadata_cap,
        });
    }
    let mut smallest_rejected = usize::MAX;

    for requested_workers in (1..=desired_workers).rev() {
        let chunk_size = job_count.div_ceil(requested_workers);
        let worker_count = job_count.div_ceil(chunk_size);
        let (metadata_bytes, warning_bytes) = batch_metadata_bytes(job_count, worker_count)?;
        if metadata_bytes > metadata_cap {
            smallest_rejected = smallest_rejected.min(metadata_bytes);
            continue;
        }
        let worker_bytes = match checked_mul(
            worker_count,
            worker_claim_bytes,
            "J2K generic worker claims",
            aggregate_cap,
        ) {
            Ok(bytes) => bytes,
            Err(BatchInfrastructureError::AllocationTooLarge { .. }) => {
                smallest_rejected = usize::MAX;
                continue;
            }
            Err(error) => return Err(error),
        };
        let Some(live_bytes) = worker_bytes.checked_add(metadata_bytes) else {
            smallest_rejected = usize::MAX;
            continue;
        };
        if live_bytes <= aggregate_cap {
            return Ok(BatchPlan {
                worker_count,
                chunk_size,
                live_bytes,
                metadata_bytes,
                warning_bytes,
            });
        }
        smallest_rejected = smallest_rejected.min(live_bytes);
    }

    Err(BatchInfrastructureError::AllocationTooLarge {
        what: "J2K batch live set",
        requested: smallest_rejected,
        cap: aggregate_cap,
    })
}

fn batch_metadata_bytes(
    job_count: usize,
    worker_count: usize,
) -> Result<(usize, usize), BatchInfrastructureError> {
    let warning_bytes = checked_mul(
        job_count,
        size_of::<J2kDecodeWarning>(),
        "J2K batch warning owners",
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    let mut bytes = checked_mul(
        job_count,
        size_of::<J2kBatchResultSlot>(),
        "J2K ordered worker result slots",
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    bytes = checked_add(
        bytes,
        checked_mul(
            job_count,
            size_of::<BatchOutcome>(),
            "J2K ordered batch results",
            J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        )?,
        "J2K batch metadata",
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    bytes = checked_add(
        bytes,
        warning_bytes,
        "J2K batch metadata",
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    bytes = checked_add(
        bytes,
        checked_mul(
            worker_count,
            size_of::<BatchWorker>(),
            "J2K worker contexts and scratch owners",
            J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        )?,
        "J2K batch metadata",
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    bytes = checked_add(
        bytes,
        checked_mul(
            worker_count,
            size_of::<ScopedWorkerHandle<'static>>(),
            "J2K scoped worker handles",
            J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        )?,
        "J2K batch metadata",
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    Ok((bytes, warning_bytes))
}

#[cfg(test)]
mod tests;
