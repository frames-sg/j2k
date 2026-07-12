// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;
use std::sync::Mutex;

use j2k_core::{try_collect_ordered_batch_results_with_limits, BatchInfrastructureError};

use super::allocation::{
    ensure_live_domains, ensure_metadata_bytes, try_vec_with_capacity, vec_capacity_bytes,
    JPEG_BATCH_HOST_CAP_BYTES, JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
};
use super::scheduler::lock_worker;
use super::worker::WorkerSlot;
use super::BatchResultSlot;
use crate::decoder::{DecodeOutcome, DecodedTile, TileBatchError};
use crate::error::JpegError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct BatchRetainedLiveBytes {
    codec: usize,
    metadata: usize,
}

pub(super) fn collect_results(
    job_count: usize,
    results: Vec<BatchResultSlot<DecodeOutcome>>,
    retained: BatchRetainedLiveBytes,
) -> Result<Vec<DecodeOutcome>, TileBatchError> {
    let retained_live_bytes =
        ensure_live_domains(retained.codec, retained.metadata, "JPEG batch collection")?;
    try_collect_ordered_batch_results_with_limits(
        job_count,
        results,
        retained_live_bytes,
        JPEG_BATCH_HOST_CAP_BYTES,
        retained.metadata,
        JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
    )
}

pub(super) fn collect_per_tile_results<T>(
    job_count: usize,
    results: Vec<BatchResultSlot<T>>,
    retained: BatchRetainedLiveBytes,
) -> Result<Vec<Result<T, JpegError>>, BatchInfrastructureError> {
    ensure_result_count(job_count, results.len())?;
    if let Some(index) = results.iter().position(Option::is_none) {
        return Err(BatchInfrastructureError::MissingResult { index });
    }
    let result_slot_bytes = vec_capacity_bytes(&results)?;
    let requested_ordered =
        allocation_bytes::<Result<T, JpegError>>(job_count, "JPEG prepared ordered results")?;
    ensure_batch_live_bytes(
        retained,
        result_slot_bytes,
        requested_ordered,
        "JPEG prepared batch collection",
    )?;
    let mut ordered = try_vec_with_capacity(job_count, "JPEG prepared ordered results")?;
    ensure_batch_live_bytes(
        retained,
        result_slot_bytes,
        vec_capacity_bytes(&ordered)?,
        "JPEG prepared batch collection",
    )?;
    for (index, slot) in results.into_iter().enumerate() {
        match slot {
            Some(result) => ordered.push(result),
            None => return Err(BatchInfrastructureError::ResultKindMismatch { index }),
        }
    }
    Ok(ordered)
}

pub(super) fn retained_live_bytes<T>(
    workers: &[Mutex<WorkerSlot>],
    worker_capacity_bytes: usize,
    plan_capacity_bytes: usize,
    results: &[BatchResultSlot<T>],
    retained_result_bytes: impl Fn(&T) -> usize,
) -> Result<BatchRetainedLiveBytes, BatchInfrastructureError> {
    let mut metadata = ensure_metadata_bytes(
        plan_capacity_bytes,
        worker_capacity_bytes,
        "JPEG retained batch metadata",
    )?;
    let mut codec = 0usize;
    for worker in workers {
        codec = checked_batch_live_add(
            codec,
            lock_worker(worker)?.retained_bytes(),
            "JPEG retained worker codec claims",
            super::allocation::JPEG_CODEC_HOST_CAP_BYTES,
        )?;
    }
    for result in results {
        if let Some(Ok(outcome)) = result {
            metadata = ensure_metadata_bytes(
                metadata,
                retained_result_bytes(outcome),
                "JPEG retained batch outcomes",
            )?;
        }
    }
    ensure_live_domains(codec, metadata, "JPEG retained batch live set")?;
    Ok(BatchRetainedLiveBytes { codec, metadata })
}

pub(super) fn decode_outcome_retained_bytes(outcome: &DecodeOutcome) -> usize {
    outcome
        .warnings
        .capacity()
        .saturating_mul(size_of::<crate::Warning>())
}

pub(super) fn decoded_tile_retained_bytes(tile: &DecodedTile) -> usize {
    tile.warnings
        .capacity()
        .saturating_mul(size_of::<crate::Warning>())
}

fn ensure_result_count(
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

fn allocation_bytes<T>(
    count: usize,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
        })
}

fn checked_batch_live_add(
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

fn ensure_batch_live_bytes(
    retained: BatchRetainedLiveBytes,
    indexed: usize,
    ordered: usize,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    let metadata = ensure_metadata_bytes(retained.metadata, indexed, what)?;
    let metadata = ensure_metadata_bytes(metadata, ordered, what)?;
    ensure_live_domains(retained.codec, metadata, what)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_outer_collection_reports_typed_cap_failure() {
        let results = vec![Some(Ok::<u8, JpegError>(7))];
        let error = collect_per_tile_results(
            1,
            results,
            BatchRetainedLiveBytes {
                codec: 0,
                metadata: JPEG_BATCH_METADATA_ALLOWANCE_BYTES,
            },
        )
        .expect_err("prepared outer collection exceeds aggregate cap");
        assert!(matches!(
            error,
            BatchInfrastructureError::AllocationTooLarge {
                what: "JPEG prepared batch collection",
                ..
            }
        ));
    }

    #[test]
    fn prepared_outer_collection_distinguishes_excess_from_missing_slots() {
        let excess = vec![Some(Ok::<u8, JpegError>(7)), Some(Ok(9))];
        assert!(matches!(
            collect_per_tile_results(
                1,
                excess,
                BatchRetainedLiveBytes {
                    codec: 0,
                    metadata: 0,
                },
            ),
            Err(BatchInfrastructureError::ResultIndexOutOfBounds {
                index: 1,
                job_count: 1,
            })
        ));

        let missing = vec![Some(Ok::<u8, JpegError>(7))];
        assert!(matches!(
            collect_per_tile_results(
                2,
                missing,
                BatchRetainedLiveBytes {
                    codec: 0,
                    metadata: 0,
                },
            ),
            Err(BatchInfrastructureError::MissingResult { index: 1 })
        ));
    }
}
