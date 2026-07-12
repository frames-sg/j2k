// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decoder-baseline-aware growth checks for the lazy checkpoint cache.

use alloc::vec::Vec;

use super::super::allocation::{
    checked_live_bytes, checkpoint_allocation_bytes, host_allocation_error,
};
use super::super::DeviceCheckpoint;
use crate::error::JpegError;

pub(in crate::internal::checkpoint) fn reserve_checkpoint_capacity(
    checkpoints: &mut Vec<DeviceCheckpoint>,
    capacity: usize,
    retained_baseline_bytes: usize,
    allocation_cap: usize,
) -> Result<(), JpegError> {
    let planned_bytes = checkpoint_allocation_bytes(capacity, allocation_cap)?;
    let retained_bytes = checkpoint_allocation_bytes(checkpoints.capacity(), allocation_cap)?;
    checked_checkpoint_reservation_peak(
        retained_baseline_bytes,
        retained_bytes,
        planned_bytes,
        allocation_cap,
    )?;
    if capacity <= checkpoints.capacity() {
        return Ok(());
    }

    checkpoints
        .try_reserve_exact(capacity.saturating_sub(checkpoints.len()))
        .map_err(|_| host_allocation_error(planned_bytes))?;
    reconcile_actual_checkpoint_capacity(
        checkpoints,
        retained_baseline_bytes,
        retained_bytes,
        allocation_cap,
    )
}

pub(in crate::internal::checkpoint) fn reconcile_actual_checkpoint_capacity(
    checkpoints: &mut Vec<DeviceCheckpoint>,
    retained_baseline_bytes: usize,
    replaced_cache_bytes: usize,
    allocation_cap: usize,
) -> Result<(), JpegError> {
    let postcheck = checkpoint_allocation_bytes(checkpoints.capacity(), allocation_cap).and_then(
        |actual_bytes| {
            checked_checkpoint_reservation_peak(
                retained_baseline_bytes,
                replaced_cache_bytes,
                actual_bytes,
                allocation_cap,
            )
        },
    );
    if let Err(error) = postcheck {
        *checkpoints = Vec::new();
        return Err(error);
    }
    Ok(())
}

pub(in crate::internal::checkpoint) fn checked_checkpoint_reservation_peak(
    retained_baseline_bytes: usize,
    retained_cache_bytes: usize,
    replacement_cache_bytes: usize,
    allocation_cap: usize,
) -> Result<usize, JpegError> {
    if replacement_cache_bytes > retained_cache_bytes {
        checked_live_bytes(
            [
                retained_baseline_bytes,
                retained_cache_bytes,
                replacement_cache_bytes,
            ],
            allocation_cap,
        )
    } else {
        checked_live_bytes(
            [retained_baseline_bytes, retained_cache_bytes],
            allocation_cap,
        )
    }
}
