// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate live-byte checks and fallible packet-vector allocation.

use alloc::vec::Vec;
use core::mem::size_of;

use super::error::FastPacketError;
use super::types::JpegEntropyCheckpointV1;
use crate::error::JpegError;

#[cfg(test)]
pub(super) fn checked_entropy_live_bytes(
    entropy_len: usize,
    restart_count: usize,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    checked_live_bytes(
        [
            checked_element_bytes::<u8>(entropy_len, allocation_cap)?,
            checked_element_bytes::<u32>(restart_count, allocation_cap)?,
        ],
        allocation_cap,
    )
}

pub(super) fn checked_color_packet_live_bytes(
    initial_live_bytes: usize,
    entropy_len: usize,
    restart_count: usize,
    checkpoint_count: usize,
    terminated_copy_bytes: usize,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    checked_live_bytes(
        [
            initial_live_bytes,
            checked_element_bytes::<u8>(entropy_len, allocation_cap)?,
            checked_element_bytes::<u32>(restart_count, allocation_cap)?,
            checked_element_bytes::<JpegEntropyCheckpointV1>(checkpoint_count, allocation_cap)?,
            terminated_copy_bytes,
        ],
        allocation_cap,
    )
}

pub(super) fn checked_color_packet_initial_live_bytes(
    external_live_bytes: usize,
    retained_decoder_bytes: usize,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    checked_live_bytes(
        [external_live_bytes, retained_decoder_bytes],
        allocation_cap,
    )
}

pub(super) fn checked_gray_packet_live_bytes(
    entropy_len: usize,
    restart_count: usize,
    terminated_copy_bytes: usize,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    checked_live_bytes(
        [
            checked_element_bytes::<u8>(entropy_len, allocation_cap)?,
            checked_element_bytes::<u32>(restart_count, allocation_cap)?,
            terminated_copy_bytes,
        ],
        allocation_cap,
    )
}

pub(super) fn checked_element_bytes<T>(
    element_count: usize,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    element_count
        .checked_mul(size_of::<T>())
        .ok_or_else(|| cap_error(usize::MAX, allocation_cap))
}

pub(super) fn try_vec_with_exact_capacity<T>(
    element_count: usize,
    live_bytes: &mut usize,
    allocation_cap: usize,
) -> Result<Vec<T>, FastPacketError> {
    let planned_bytes = checked_element_bytes::<T>(element_count, allocation_cap)?;
    checked_live_bytes([*live_bytes, planned_bytes], allocation_cap)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(element_count)
        .map_err(|_| host_allocation_error(planned_bytes))?;
    *live_bytes =
        checked_actual_vec_live_bytes::<T>(*live_bytes, values.capacity(), allocation_cap)?;
    Ok(values)
}

pub(super) fn checked_actual_vec_live_bytes<T>(
    initial_live_bytes: usize,
    capacity: usize,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    let actual_bytes = checked_element_bytes::<T>(capacity, allocation_cap)?;
    checked_live_bytes([initial_live_bytes, actual_bytes], allocation_cap)
}

pub(super) fn host_allocation_error(requested: usize) -> FastPacketError {
    FastPacketError::Decode(JpegError::HostAllocationFailed { bytes: requested })
}

pub(super) fn checked_live_bytes(
    allocations: impl IntoIterator<Item = usize>,
    allocation_cap: usize,
) -> Result<usize, FastPacketError> {
    let requested = allocations
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| cap_error(usize::MAX, allocation_cap))?;
    if requested > allocation_cap {
        return Err(cap_error(requested, allocation_cap));
    }
    Ok(requested)
}

fn cap_error(requested: usize, cap: usize) -> FastPacketError {
    FastPacketError::Decode(JpegError::MemoryCapExceeded { requested, cap })
}
