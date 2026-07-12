// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate J2K batch allocation constants and fallible vector helpers.

use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::{try_host_vec_with_capacity, BatchInfrastructureError, HostAllocationError};

use super::planning::BatchPlan;
use super::J2kBatchResultSlot;
use crate::J2kDecodeWarning;

/// Maximum number of generic native-decode claims admitted concurrently.
///
/// This is a 0.7 safety/performance policy, not an estimate of a particular
/// image. Requested concurrency above four is reduced before spawning.
pub(super) const MAX_GENERIC_BATCH_WORKERS: usize = 4;

/// Maximum number of scheduled region-scaled workers when weighted runtime
/// admission is active. At most four worst-case generic claims execute at once;
/// the wider bound permits eight independently accounted small direct ROIs
/// without creating an unbounded number of blocked operating-system threads.
pub(super) const MAX_ADMITTED_BATCH_WORKERS: usize = MAX_GENERIC_BATCH_WORKERS * 2;

/// Fixed ceiling for batch-owned slots, handles, outcomes, and warnings.
///
/// This is a facade policy allowance rather than a codec memory requirement.
pub(super) const J2K_BATCH_METADATA_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;

/// Authoritative worst-case claim for one generic native decoder worker.
pub(super) const GENERIC_WORKER_CLAIM_BYTES: usize = j2k_native::DEFAULT_MAX_DECODE_BYTES;

const _: [(); 1] = [(); GENERIC_WORKER_CLAIM_BYTES
    .checked_mul(MAX_GENERIC_BATCH_WORKERS)
    .is_some() as usize];
pub(super) const J2K_BATCH_EXECUTION_CAP_BYTES: usize =
    GENERIC_WORKER_CLAIM_BYTES * MAX_GENERIC_BATCH_WORKERS;

/// Fixed aggregate execution cap: four native claims plus bounded metadata.
///
/// A claim reserves no host memory by itself. It is conservative accounting
/// for native ownership that cannot be estimated soundly before packet parsing.
const _: [(); 1] = [(); J2K_BATCH_EXECUTION_CAP_BYTES
    .checked_add(J2K_BATCH_METADATA_ALLOWANCE_BYTES)
    .is_some() as usize];
pub(super) const J2K_BATCH_HOST_CAP_BYTES: usize =
    J2K_BATCH_EXECUTION_CAP_BYTES + J2K_BATCH_METADATA_ALLOWANCE_BYTES;

pub(super) fn ensure_execution_capacity(
    plan: BatchPlan,
    allocator_capacity_extra: usize,
    actual_warning_bytes: usize,
) -> Result<(), BatchInfrastructureError> {
    let base_metadata = plan.metadata_bytes.checked_sub(plan.warning_bytes).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch metadata",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        },
    )?;
    let actual_metadata = base_metadata
        .checked_add(allocator_capacity_extra)
        .and_then(|bytes| bytes.checked_add(actual_warning_bytes))
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch metadata",
            requested: usize::MAX,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        })?;
    if actual_metadata > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch metadata",
            requested: actual_metadata,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }

    let planned_worker_bytes = plan.live_bytes.checked_sub(plan.metadata_bytes).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch live set",
            requested: usize::MAX,
            cap: J2K_BATCH_HOST_CAP_BYTES,
        },
    )?;
    let actual_live = planned_worker_bytes.checked_add(actual_metadata).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch live set",
            requested: usize::MAX,
            cap: J2K_BATCH_HOST_CAP_BYTES,
        },
    )?;
    if actual_live > J2K_BATCH_HOST_CAP_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch live set",
            requested: actual_live,
            cap: J2K_BATCH_HOST_CAP_BYTES,
        });
    }
    Ok(())
}

pub(super) fn ensure_pre_execution_capacity(
    plan: BatchPlan,
    allocator_capacity_extra: usize,
) -> Result<(), BatchInfrastructureError> {
    ensure_execution_capacity(plan, allocator_capacity_extra, plan.warning_bytes)
}

pub(super) fn actual_warning_owner_bytes(
    results: &[J2kBatchResultSlot],
) -> Result<usize, BatchInfrastructureError> {
    let mut bytes = 0usize;
    for outcome in results.iter().filter_map(|slot| match slot {
        Some(Ok(outcome)) => Some(outcome),
        Some(Err(_)) | None => None,
    }) {
        let warning_bytes = checked_mul(
            outcome.warnings.capacity(),
            size_of::<J2kDecodeWarning>(),
            "J2K retained batch warnings",
            J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        )?;
        bytes = checked_add(
            bytes,
            warning_bytes,
            "J2K retained batch warnings",
            J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        )?;
    }
    Ok(bytes)
}

pub(super) fn try_vec_with_capacity<T>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, BatchInfrastructureError> {
    let requested = checked_mul(
        capacity,
        size_of::<T>(),
        what,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    if requested > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    let values =
        try_host_vec_with_capacity(capacity).map_err(|error| host_allocation_error(what, error))?;
    let actual = vector_capacity_bytes(&values, what)?;
    if actual > J2K_BATCH_METADATA_ALLOWANCE_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: actual.max(requested),
            cap: J2K_BATCH_METADATA_ALLOWANCE_BYTES,
        });
    }
    Ok(values)
}

pub(super) fn capacity_extra_bytes<T>(
    requested_capacity: usize,
    values: &Vec<T>,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    let requested = checked_mul(
        requested_capacity,
        size_of::<T>(),
        what,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )?;
    let actual = vector_capacity_bytes(values, what)?;
    Ok(actual.saturating_sub(requested))
}

fn vector_capacity_bytes<T>(
    values: &Vec<T>,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    checked_mul(
        values.capacity(),
        size_of::<T>(),
        what,
        J2K_BATCH_METADATA_ALLOWANCE_BYTES,
    )
}

pub(super) fn checked_add(
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

pub(super) fn checked_mul(
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
