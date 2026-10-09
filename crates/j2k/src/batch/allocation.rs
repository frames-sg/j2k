// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate J2K batch allocation constants and fallible vector helpers.

use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::{try_host_vec_with_capacity, BatchInfrastructureError, HostAllocationError};

use super::planning::BatchPlan;
use super::J2kBatchResultSlot;
use crate::J2kDecodeWarning;

/// Generic native-decode claims admitted concurrently when the host does not
/// report its memory, and the floor on every host.
pub(crate) const BASELINE_GENERIC_BATCH_WORKERS: usize = 4;

/// Worst-case native claims may together account for `1 / divisor` of the
/// memory available to the process.
const HOST_MEMORY_SHARE_DIVISOR: u64 = 2;

/// Fixed ceiling for batch-owned slots, handles, outcomes, and warnings.
///
/// This is a facade policy allowance rather than a codec memory requirement.
pub(crate) const J2K_BATCH_METADATA_ALLOWANCE_BYTES: usize = 64 * 1024 * 1024;

/// Authoritative worst-case claim for one generic native decoder worker.
pub(super) const GENERIC_WORKER_CLAIM_BYTES: usize = j2k_native::DEFAULT_MAX_DECODE_BYTES;

/// Largest claim count whose execution cap plus metadata still fits `usize`.
const MAX_SCALED_GENERIC_BATCH_WORKERS: usize =
    (usize::MAX - J2K_BATCH_METADATA_ALLOWANCE_BYTES) / GENERIC_WORKER_CLAIM_BYTES;

const _: [(); 1] =
    [(); (BASELINE_GENERIC_BATCH_WORKERS <= MAX_SCALED_GENERIC_BATCH_WORKERS) as usize];

/// Maximum number of generic native-decode claims admitted concurrently.
///
/// Each claim accounts for one worst-case native decode. Claims may together
/// account for half of the memory available to the process (physical memory,
/// lowered by a Linux cgroup limit), and never fewer than
/// [`BASELINE_GENERIC_BATCH_WORKERS`]. A claim reserves no memory by itself;
/// requested concurrency above the limit is reduced before spawning.
pub(crate) fn generic_batch_worker_limit() -> usize {
    generic_worker_limit_for_memory(super::host_memory::process_memory_limit_bytes())
}

fn generic_worker_limit_for_memory(memory_bytes: Option<u64>) -> usize {
    let Some(memory_bytes) = memory_bytes else {
        return BASELINE_GENERIC_BATCH_WORKERS;
    };
    let claim_bytes = u64::try_from(GENERIC_WORKER_CLAIM_BYTES).unwrap_or(u64::MAX);
    let claims = memory_bytes / HOST_MEMORY_SHARE_DIVISOR / claim_bytes;
    usize::try_from(claims).unwrap_or(usize::MAX).clamp(
        BASELINE_GENERIC_BATCH_WORKERS,
        MAX_SCALED_GENERIC_BATCH_WORKERS,
    )
}

/// Maximum number of scheduled region-scaled workers when weighted runtime
/// admission is active. At most [`generic_batch_worker_limit`] worst-case
/// generic claims execute at once; twice as many workers may hold
/// independently accounted small direct ROIs without creating an unbounded
/// number of blocked operating-system threads.
pub(super) fn admitted_batch_worker_limit() -> usize {
    generic_batch_worker_limit().saturating_mul(2)
}

/// Aggregate execution cap: the admitted generic native claims.
///
/// A claim is conservative accounting for native ownership that cannot be
/// estimated soundly before packet parsing.
pub(super) fn batch_execution_cap_bytes() -> usize {
    // Cannot saturate: the worker limit is at most MAX_SCALED_GENERIC_BATCH_WORKERS.
    generic_batch_worker_limit().saturating_mul(GENERIC_WORKER_CLAIM_BYTES)
}

/// Aggregate host cap: the execution cap plus bounded batch metadata.
pub(super) fn batch_host_cap_bytes() -> usize {
    batch_execution_cap_bytes().saturating_add(J2K_BATCH_METADATA_ALLOWANCE_BYTES)
}

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
            cap: batch_host_cap_bytes(),
        },
    )?;
    let actual_live = planned_worker_bytes.checked_add(actual_metadata).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch live set",
            requested: usize::MAX,
            cap: batch_host_cap_bytes(),
        },
    )?;
    if actual_live > batch_host_cap_bytes() {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what: "J2K batch live set",
            requested: actual_live,
            cap: batch_host_cap_bytes(),
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

pub(crate) fn try_vec_with_capacity<T>(
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

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn generic_worker_limit_scales_with_host_memory() {
        // Half of the memory, in 512 MiB worst-case claims.
        assert_eq!(generic_worker_limit_for_memory(Some(48 * GIB)), 48);
        assert_eq!(generic_worker_limit_for_memory(Some(16 * GIB)), 16);
        assert_eq!(generic_worker_limit_for_memory(Some(8 * GIB)), 8);
        assert_eq!(generic_worker_limit_for_memory(Some(5 * GIB)), 5);
    }

    #[test]
    fn small_or_unknown_hosts_keep_the_baseline_limit() {
        for memory in [None, Some(0), Some(GIB), Some(4 * GIB - 1)] {
            assert_eq!(
                generic_worker_limit_for_memory(memory),
                BASELINE_GENERIC_BATCH_WORKERS,
                "{memory:?}"
            );
        }
    }

    #[test]
    fn scaled_execution_cap_cannot_overflow() {
        let limit = generic_worker_limit_for_memory(Some(u64::MAX));
        assert!(limit <= MAX_SCALED_GENERIC_BATCH_WORKERS);
        let execution = limit
            .checked_mul(GENERIC_WORKER_CLAIM_BYTES)
            .expect("execution cap fits usize");
        assert!(execution
            .checked_add(J2K_BATCH_METADATA_ALLOWANCE_BYTES)
            .is_some());
    }

    #[test]
    fn host_caps_follow_the_detected_worker_limit() {
        let limit = generic_batch_worker_limit();
        assert!(limit >= BASELINE_GENERIC_BATCH_WORKERS);
        assert_eq!(
            batch_execution_cap_bytes(),
            limit * GENERIC_WORKER_CLAIM_BYTES
        );
        assert_eq!(
            batch_host_cap_bytes(),
            batch_execution_cap_bytes() + J2K_BATCH_METADATA_ALLOWANCE_BYTES
        );
        assert_eq!(admitted_batch_worker_limit(), limit * 2);
    }
}
