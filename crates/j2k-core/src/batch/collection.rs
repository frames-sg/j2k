// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::{
    BatchDecodeError, BatchInfrastructureError, BatchResultSlot, IndexedBatchResult, TileBatchError,
};
use crate::host_allocation::{try_host_vec_with_capacity, HostAllocationError};

/// Fallibly restore indexed worker results to caller input order.
///
/// `retained_live_bytes` is the batch-owned memory that stays live while the
/// indexed result vector and ordered output vector coexist. The collector
/// counts both vector capacities against `max_live_bytes`, validates every
/// worker index without panicking, and preserves the first codec failure in
/// caller input order.
///
/// # Errors
///
/// Returns [`BatchDecodeError::Tile`] for the lowest-index codec failure and
/// [`BatchDecodeError::Infrastructure`] for cap, allocator, or result-integrity
/// failures.
#[doc(hidden)]
pub fn try_collect_indexed_batch_results<T, E>(
    job_count: usize,
    mut results: Vec<IndexedBatchResult<T, E>>,
    retained_live_bytes: usize,
    max_live_bytes: usize,
) -> Result<Vec<T>, BatchDecodeError<E>> {
    results.sort_unstable_by_key(|(index, _)| *index);
    validate_indexed_batch_results(job_count, &results)?;
    let indexed_capacity = results.capacity();

    if let Some(error_position) = results.iter().position(|(_, result)| result.is_err()) {
        let (index, result) = results.remove(error_position);
        return match result {
            Err(source) => Err(BatchDecodeError::Tile(TileBatchError { index, source })),
            Ok(_) => Err(BatchInfrastructureError::ResultKindMismatch { index }.into()),
        };
    }

    let indexed_bytes = allocation_bytes::<IndexedBatchResult<T, E>>(indexed_capacity);
    let ordered_bytes = allocation_bytes::<T>(job_count);
    ensure_collection_live_bytes(
        retained_live_bytes,
        retained_live_bytes,
        indexed_bytes,
        ordered_bytes,
        max_live_bytes,
        max_live_bytes,
        "indexed batch collection",
    )?;
    let mut ordered = try_ordered_vec(job_count, "ordered batch results")?;
    ensure_collection_live_bytes(
        retained_live_bytes,
        retained_live_bytes,
        indexed_bytes,
        allocation_bytes::<T>(ordered.capacity()),
        max_live_bytes,
        max_live_bytes,
        "indexed batch collection",
    )?;
    for (index, result) in results {
        match result {
            Ok(outcome) => ordered.push(outcome),
            Err(source) => {
                return Err(BatchDecodeError::Tile(TileBatchError { index, source }));
            }
        }
    }
    Ok(ordered)
}

/// Fallibly collect ordered worker slots under separate aggregate and
/// collection-owned memory limits.
///
/// `retained_live_bytes` covers every owner that remains live during
/// collection. `retained_collection_bytes` is the subset charged to the
/// collection/metadata domain. Both limits are checked before allocation and
/// again against the allocator-returned output capacity.
#[doc(hidden)]
pub fn try_collect_ordered_batch_results_with_limits<T, E>(
    job_count: usize,
    mut results: Vec<BatchResultSlot<T, E>>,
    retained_live_bytes: usize,
    max_live_bytes: usize,
    retained_collection_bytes: usize,
    max_collection_bytes: usize,
) -> Result<Vec<T>, BatchDecodeError<E>> {
    validate_ordered_result_count(job_count, results.len())?;
    if let Some(index) = results.iter().position(|slot| matches!(slot, Some(Err(_)))) {
        return match results.get_mut(index).and_then(Option::take) {
            Some(Err(source)) => Err(BatchDecodeError::Tile(TileBatchError { index, source })),
            _ => Err(BatchInfrastructureError::ResultKindMismatch { index }.into()),
        };
    }
    if let Some(index) = results.iter().position(Option::is_none) {
        return Err(BatchInfrastructureError::MissingResult { index }.into());
    }

    let slot_bytes = allocation_bytes::<BatchResultSlot<T, E>>(results.capacity());
    ensure_collection_live_bytes(
        retained_live_bytes,
        retained_collection_bytes,
        slot_bytes,
        allocation_bytes::<T>(job_count),
        max_live_bytes,
        max_collection_bytes,
        "ordered batch collection",
    )?;
    let mut ordered = try_ordered_vec(job_count, "ordered batch results")?;
    ensure_collection_live_bytes(
        retained_live_bytes,
        retained_collection_bytes,
        slot_bytes,
        allocation_bytes::<T>(ordered.capacity()),
        max_live_bytes,
        max_collection_bytes,
        "ordered batch collection",
    )?;
    for (index, slot) in results.into_iter().enumerate() {
        match slot {
            Some(Ok(outcome)) => ordered.push(outcome),
            Some(Err(_)) | None => {
                return Err(BatchInfrastructureError::ResultKindMismatch { index }.into());
            }
        }
    }
    Ok(ordered)
}

fn try_ordered_vec<T, E>(
    capacity: usize,
    what: &'static str,
) -> Result<Vec<T>, BatchDecodeError<E>> {
    try_host_vec_with_capacity(capacity)
        .map_err(|error| BatchDecodeError::Infrastructure(batch_host_allocation_error(what, error)))
}

fn validate_ordered_result_count(
    job_count: usize,
    result_count: usize,
) -> Result<(), BatchInfrastructureError> {
    if result_count < job_count {
        return Err(BatchInfrastructureError::MissingResult {
            index: result_count,
        });
    }
    if result_count > job_count {
        return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
            index: job_count,
            job_count,
        });
    }
    Ok(())
}

fn ensure_collection_live_bytes<E>(
    retained_live_bytes: usize,
    retained_collection_bytes: usize,
    source_bytes: usize,
    ordered_bytes: usize,
    max_live_bytes: usize,
    max_collection_bytes: usize,
    what: &'static str,
) -> Result<(), BatchDecodeError<E>> {
    let requested_collection = retained_collection_bytes
        .checked_add(source_bytes)
        .and_then(|total| total.checked_add(ordered_bytes))
        .ok_or_else(|| allocation_too_large(what, max_collection_bytes))?;
    if requested_collection > max_collection_bytes {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: requested_collection,
            cap: max_collection_bytes,
        }
        .into());
    }
    // The collection-domain retained bytes are documented as a subset of the
    // aggregate retained owner. Defensively take the larger value so an
    // inconsistent internal caller cannot understate aggregate ownership.
    let aggregate_retained_bytes = retained_live_bytes.max(retained_collection_bytes);
    let requested = aggregate_retained_bytes
        .checked_add(source_bytes)
        .and_then(|total| total.checked_add(ordered_bytes))
        .ok_or_else(|| allocation_too_large(what, max_live_bytes))?;
    if requested > max_live_bytes {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested,
            cap: max_live_bytes,
        }
        .into());
    }
    Ok(())
}

fn validate_indexed_batch_results<T, E>(
    job_count: usize,
    results: &[IndexedBatchResult<T, E>],
) -> Result<(), BatchInfrastructureError> {
    let mut expected = 0usize;
    for (index, _) in results {
        if *index >= job_count {
            return Err(BatchInfrastructureError::ResultIndexOutOfBounds {
                index: *index,
                job_count,
            });
        }
        if *index < expected {
            return Err(BatchInfrastructureError::DuplicateResult { index: *index });
        }
        if *index > expected {
            return Err(BatchInfrastructureError::MissingResult { index: expected });
        }
        expected = expected.saturating_add(1);
    }
    if expected < job_count {
        return Err(BatchInfrastructureError::MissingResult { index: expected });
    }
    Ok(())
}

const fn allocation_bytes<T>(capacity: usize) -> usize {
    match capacity.checked_mul(size_of::<T>()) {
        Some(bytes) => bytes,
        None => usize::MAX,
    }
}

const fn allocation_too_large(what: &'static str, cap: usize) -> BatchInfrastructureError {
    BatchInfrastructureError::AllocationTooLarge {
        what,
        requested: usize::MAX,
        cap,
    }
}

fn batch_host_allocation_error(
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
