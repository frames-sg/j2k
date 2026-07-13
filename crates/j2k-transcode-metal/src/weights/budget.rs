// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use super::{SparseWeightRow, SparseWeightRowsError, SparseWeightTap};

pub(super) fn validate_dense_weight_budget(sample_len: usize) -> Result<(), SparseWeightRowsError> {
    let rows = sample_len
        .checked_mul(size_of::<Vec<f32>>())
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    let values = sample_len
        .checked_mul(sample_len)
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    let workspace = sample_len
        .checked_mul(3)
        .and_then(|count| count.checked_mul(size_of::<f64>()))
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    let requested = rows
        .checked_add(values)
        .and_then(|bytes| bytes.checked_add(workspace))
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    validate_budget(requested)
}

pub(super) fn bounded_sparse_weight_budget(
    sample_len: usize,
    max_taps_per_row: usize,
) -> Result<usize, SparseWeightRowsError> {
    let rows = sample_len
        .checked_mul(size_of::<SparseWeightRow>())
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    let taps = sample_len
        .checked_mul(max_taps_per_row)
        .and_then(|count| count.checked_mul(size_of::<SparseWeightTap>()))
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    let requested = rows
        .checked_add(taps)
        .ok_or(SparseWeightRowsError::SizeOverflow)?;
    validate_budget(requested)?;
    Ok(requested)
}

#[cfg(target_os = "macos")]
pub(super) fn metal_sparse_weight_budget(
    sample_len: usize,
    max_taps_per_row: usize,
) -> Result<usize, SparseWeightRowsError> {
    let row_bytes = size_of::<u32>() * 2;
    let tap_bytes = size_of::<u32>() + size_of::<f32>();
    sample_len
        .checked_mul(row_bytes)
        .and_then(|rows| {
            sample_len
                .checked_mul(max_taps_per_row)
                .and_then(|count| count.checked_mul(tap_bytes))
                .and_then(|taps| rows.checked_add(taps))
        })
        .ok_or(SparseWeightRowsError::SizeOverflow)
}

fn validate_budget(requested: usize) -> Result<(), SparseWeightRowsError> {
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(SparseWeightRowsError::AllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(())
}
