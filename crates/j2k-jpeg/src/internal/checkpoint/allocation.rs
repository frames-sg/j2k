// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible checkpoint-vector allocation and aggregate live-byte checks.

use alloc::vec::Vec;

use super::DeviceCheckpoint;
use crate::error::JpegError;

pub(super) fn checked_checkpoint_phase_bytes<T>(
    initial_live_bytes: usize,
    checkpoint_count: usize,
    terminated_copy_bytes: usize,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    checked_live_bytes(
        [
            initial_live_bytes,
            element_capacity_bytes::<T>(checkpoint_count, allocation_cap)?,
            terminated_copy_bytes,
        ],
        allocation_cap,
    )
}

pub(super) fn try_checkpoint_vec_with_live_budget<T>(
    capacity: usize,
    live_bytes: &mut usize,
    allocation_cap: usize,
) -> Result<Vec<T>, JpegError> {
    let planned_bytes = element_capacity_bytes::<T>(capacity, allocation_cap)?;
    checked_live_bytes([*live_bytes, planned_bytes], allocation_cap)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| host_allocation_error(planned_bytes))?;
    *live_bytes =
        checked_actual_checkpoint_live_bytes::<T>(*live_bytes, values.capacity(), allocation_cap)?;
    Ok(values)
}

pub(super) fn checked_actual_checkpoint_live_bytes<T>(
    initial_live_bytes: usize,
    capacity: usize,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    let actual_bytes = element_capacity_bytes::<T>(capacity, allocation_cap)?;
    checked_live_bytes([initial_live_bytes, actual_bytes], allocation_cap)
}

#[cfg(test)]
pub(super) fn checked_checkpoint_workspace_bytes(
    checkpoint_count: usize,
    terminated_copy_bytes: usize,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    checked_checkpoint_phase_bytes::<DeviceCheckpoint>(
        0,
        checkpoint_count,
        terminated_copy_bytes,
        allocation_cap,
    )
}

#[cfg(test)]
pub(super) fn try_checkpoint_vec(
    capacity: usize,
    allocation_cap: usize,
) -> Result<Vec<DeviceCheckpoint>, JpegError> {
    let mut live_bytes = 0;
    try_checkpoint_vec_with_live_budget(capacity, &mut live_bytes, allocation_cap)
}

pub(super) fn checkpoint_allocation_bytes(
    checkpoint_count: usize,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    element_capacity_bytes::<DeviceCheckpoint>(checkpoint_count, allocation_cap)
}

pub(super) fn element_capacity_bytes<T>(
    capacity: usize,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    capacity
        .checked_mul(core::mem::size_of::<T>())
        .ok_or_else(|| cap_error(usize::MAX, allocation_cap))
}

pub(super) fn checked_live_bytes(
    capacities: impl IntoIterator<Item = usize>,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    let requested = capacities
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| cap_error(usize::MAX, allocation_cap))?;
    if requested > allocation_cap {
        return Err(cap_error(requested, allocation_cap));
    }
    Ok(requested)
}

pub(super) fn host_allocation_error(requested: usize) -> JpegError {
    JpegError::HostAllocationFailed { bytes: requested }
}

fn cap_error(requested: usize, cap: usize) -> JpegError {
    JpegError::MemoryCapExceeded { requested, cap }
}
