// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::{try_host_vec_with_capacity, BatchInfrastructureError, HostAllocationError};

use crate::decoder::DEFAULT_MAX_DECODE_BYTES;
use crate::error::JpegError;

/// Shared allowance for every batch-owned vector and deep warning owner.
pub(super) const JPEG_BATCH_METADATA_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;

/// Authoritative codec-domain allowance used by one planning decoder or by
/// all concurrently active worker claims in aggregate.
pub(super) const JPEG_CODEC_HOST_CAP_BYTES: usize = DEFAULT_MAX_DECODE_BYTES;

/// Maximum checked JPEG batch live set: one codec domain plus metadata.
const _: [(); 1] =
    [(); (JPEG_CODEC_HOST_CAP_BYTES <= usize::MAX - JPEG_BATCH_METADATA_ALLOWANCE_BYTES) as usize];
pub(super) const JPEG_BATCH_HOST_CAP_BYTES: usize =
    JPEG_CODEC_HOST_CAP_BYTES + JPEG_BATCH_METADATA_ALLOWANCE_BYTES;

#[derive(Debug, Clone)]
pub(super) enum PlannedJob {
    Decode {
        worker_live_bytes: usize,
        retained_result_bytes: usize,
    },
    Reject(JpegError),
}

impl PlannedJob {
    pub(super) const fn live_bytes(&self) -> usize {
        match self {
            Self::Decode {
                worker_live_bytes, ..
            } => *worker_live_bytes,
            Self::Reject(_) => 0,
        }
    }

    pub(super) const fn retained_result_bytes(&self) -> usize {
        match self {
            Self::Decode {
                retained_result_bytes,
                ..
            } => *retained_result_bytes,
            Self::Reject(_) => 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BatchMetadataLayout {
    pub(super) fixed_bytes: usize,
    pub(super) worker_slot_capacity: usize,
    pub(super) worker_slot_bytes: usize,
    pub(super) worker_result_bytes: usize,
    pub(super) ordered_result_bytes: usize,
    pub(super) handle_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchPlan {
    pub(super) worker_count: usize,
    pub(super) chunk_size: usize,
    pub(super) live_bytes: usize,
    pub(super) metadata_bytes: usize,
    pub(super) codec_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PlannedLiveBytes {
    metadata: usize,
    codec: usize,
    aggregate: usize,
}

#[derive(Clone, Copy, Debug)]
struct BatchPlanLimits {
    metadata: usize,
    codec: usize,
    aggregate: usize,
}

pub(super) fn select_batch_plan(
    jobs: &[PlannedJob],
    desired_workers: usize,
    metadata: BatchMetadataLayout,
    retained_worker_bytes: impl Fn(usize) -> usize,
) -> Result<BatchPlan, BatchInfrastructureError> {
    select_batch_plan_with_limits(
        jobs,
        desired_workers,
        metadata,
        retained_worker_bytes,
        BatchPlanLimits {
            metadata: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            codec: JPEG_CODEC_HOST_CAP_BYTES,
            aggregate: JPEG_BATCH_HOST_CAP_BYTES,
        },
    )
}

fn select_batch_plan_with_limits(
    jobs: &[PlannedJob],
    desired_workers: usize,
    metadata: BatchMetadataLayout,
    retained_worker_bytes: impl Fn(usize) -> usize,
    limits: BatchPlanLimits,
) -> Result<BatchPlan, BatchInfrastructureError> {
    if jobs.is_empty() {
        return Err(BatchInfrastructureError::EmptyBatchPlan);
    }
    let desired_workers = desired_workers.max(1).min(jobs.len());
    let mut final_rejection = BatchInfrastructureError::AllocationTooLarge {
        what: "JPEG batch live set",
        requested: usize::MAX,
        cap: limits.aggregate,
    };

    for requested_workers in (1..=desired_workers).rev() {
        let chunk_size = jobs.len().div_ceil(requested_workers);
        let worker_count = jobs.len().div_ceil(chunk_size);
        let live = match planned_live_bytes(
            jobs,
            worker_count,
            chunk_size,
            metadata,
            &retained_worker_bytes,
            limits,
        ) {
            Ok(live) => live,
            Err(error @ BatchInfrastructureError::AllocationTooLarge { .. }) => {
                final_rejection = error;
                continue;
            }
            Err(error) => return Err(error),
        };
        return Ok(BatchPlan {
            worker_count,
            chunk_size,
            live_bytes: live.aggregate,
            metadata_bytes: live.metadata,
            codec_bytes: live.codec,
        });
    }

    Err(final_rejection)
}

fn planned_live_bytes(
    jobs: &[PlannedJob],
    worker_count: usize,
    chunk_size: usize,
    metadata: BatchMetadataLayout,
    retained_worker_bytes: &impl Fn(usize) -> usize,
    limits: BatchPlanLimits,
) -> Result<PlannedLiveBytes, BatchInfrastructureError> {
    let mut metadata_bytes = metadata.fixed_bytes;
    for job in jobs {
        metadata_bytes = checked_add(
            metadata_bytes,
            job.retained_result_bytes(),
            "JPEG retained batch outcomes",
            limits.metadata,
        )?;
    }
    metadata_bytes = checked_add(
        metadata_bytes,
        checked_mul(
            metadata.worker_slot_capacity.max(worker_count),
            metadata.worker_slot_bytes,
            "JPEG batch worker slots",
            limits.metadata,
        )?,
        "JPEG batch metadata",
        limits.metadata,
    )?;
    metadata_bytes = checked_add(
        metadata_bytes,
        checked_mul(
            jobs.len(),
            metadata.worker_result_bytes,
            "JPEG ordered worker result slots",
            limits.metadata,
        )?,
        "JPEG batch metadata",
        limits.metadata,
    )?;
    metadata_bytes = checked_add(
        metadata_bytes,
        checked_mul(
            jobs.len(),
            metadata.ordered_result_bytes,
            "JPEG ordered batch results",
            limits.metadata,
        )?,
        "JPEG batch metadata",
        limits.metadata,
    )?;
    metadata_bytes = checked_add(
        metadata_bytes,
        checked_mul(
            worker_count,
            metadata.handle_bytes,
            "JPEG scoped worker handles",
            limits.metadata,
        )?,
        "JPEG batch metadata",
        limits.metadata,
    )?;
    ensure_within(metadata_bytes, limits.metadata, "JPEG batch metadata")?;

    let mut codec_bytes = 0usize;
    for (worker_index, chunk) in jobs.chunks(chunk_size).enumerate() {
        if worker_index >= worker_count {
            return Err(BatchInfrastructureError::WorkerSlotMissing {
                worker: worker_index,
                available: worker_count,
            });
        }
        let operation_bytes = chunk.iter().map(PlannedJob::live_bytes).max().unwrap_or(0);
        // A stale pool can remain live while the next decoder constructs its
        // prepared metadata. Count both rather than assuming the operation
        // claim replaces every retained owner; `prepare_batch` may explicitly
        // release stale slots and retry when this conservative sum does not fit.
        let worker_bytes = checked_add(
            operation_bytes,
            retained_worker_bytes(worker_index),
            "JPEG retained worker live set",
            limits.codec,
        )?;
        codec_bytes = checked_add(
            codec_bytes,
            worker_bytes,
            "JPEG batch codec claims",
            limits.codec,
        )?;
    }
    ensure_within(codec_bytes, limits.codec, "JPEG batch codec claims")?;
    let aggregate = checked_add(
        metadata_bytes,
        codec_bytes,
        "JPEG batch live set",
        limits.aggregate,
    )?;
    ensure_within(aggregate, limits.aggregate, "JPEG batch live set")?;
    Ok(PlannedLiveBytes {
        metadata: metadata_bytes,
        codec: codec_bytes,
        aggregate,
    })
}

pub(super) fn try_vec_with_capacity<T>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, BatchInfrastructureError> {
    try_vec_with_retained_metadata(capacity, 0, what)
}

pub(super) fn try_vec_with_retained_metadata<T>(
    capacity: usize,
    retained_metadata_bytes: usize,
    what: &'static str,
) -> Result<Vec<T>, BatchInfrastructureError> {
    let requested = checked_mul(
        capacity,
        size_of::<T>(),
        what,
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    ensure_metadata_bytes(retained_metadata_bytes, requested, what)?;
    let values =
        try_host_vec_with_capacity(capacity).map_err(|error| host_allocation_error(what, error))?;
    let actual = checked_mul(
        values.capacity(),
        size_of::<T>(),
        what,
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    ensure_metadata_bytes(retained_metadata_bytes, actual, what)?;
    Ok(values)
}

pub(super) fn ensure_metadata_bytes(
    retained: usize,
    additional: usize,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    let requested = checked_add(
        retained,
        additional,
        what,
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    if requested > JPEG_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    Ok(requested)
}

pub(super) fn ensure_live_domains(
    codec_bytes: usize,
    metadata_bytes: usize,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    if codec_bytes > JPEG_CODEC_HOST_CAP_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG batch codec claims",
            requested: codec_bytes,
            cap: JPEG_CODEC_HOST_CAP_BYTES,
        });
    }
    if metadata_bytes > JPEG_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "JPEG batch metadata",
            requested: metadata_bytes,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    let requested = checked_add(codec_bytes, metadata_bytes, what, JPEG_BATCH_HOST_CAP_BYTES)?;
    if requested > JPEG_BATCH_HOST_CAP_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap: JPEG_BATCH_HOST_CAP_BYTES,
        });
    }
    Ok(requested)
}

pub(super) fn ensure_planning_phase(
    metadata_bytes: usize,
) -> Result<usize, BatchInfrastructureError> {
    ensure_live_domains(
        JPEG_CODEC_HOST_CAP_BYTES,
        metadata_bytes,
        "JPEG batch planning phase",
    )
}

pub(super) fn vec_capacity_bytes<T>(values: &Vec<T>) -> Result<usize, BatchInfrastructureError> {
    checked_mul(
        values.capacity(),
        size_of::<T>(),
        "JPEG batch vector",
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
    )
}

fn ensure_within(
    requested: usize,
    cap: usize,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    if requested > cap {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap,
        });
    }
    Ok(())
}

fn checked_add(
    left: usize,
    right: usize,
    what: &'static str,
    cap: usize,
) -> Result<usize, BatchInfrastructureError> {
    left.checked_add(right)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap,
        })
}

fn checked_mul(
    left: usize,
    right: usize,
    what: &'static str,
    cap: usize,
) -> Result<usize, BatchInfrastructureError> {
    left.checked_mul(right)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap,
        })
}

fn host_allocation_error(
    what: &'static str,
    error: HostAllocationError,
) -> BatchInfrastructureError {
    BatchInfrastructureError::HostAllocationFailed {
        what,
        bytes: error.requested_bytes(),
    }
}

#[cfg(test)]
mod tests;
