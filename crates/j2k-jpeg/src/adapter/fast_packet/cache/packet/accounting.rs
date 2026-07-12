// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact nested-vector capacity accounting for shared fast packets.

use alloc::vec::Vec;
use core::mem::size_of;

use super::super::JpegPlanCacheError;
use crate::adapter::fast_packet::JpegEntropyCheckpointV1;

pub(super) fn color_packet_capacity_bytes(
    restart_offsets: &Vec<u32>,
    entropy_checkpoints: &Vec<JpegEntropyCheckpointV1>,
    entropy_bytes: &Vec<u8>,
) -> Result<usize, JpegPlanCacheError> {
    capacity_bytes::<u32>(restart_offsets.capacity())?
        .checked_add(capacity_bytes::<JpegEntropyCheckpointV1>(
            entropy_checkpoints.capacity(),
        )?)
        .and_then(|bytes| bytes.checked_add(entropy_bytes.capacity()))
        .ok_or(JpegPlanCacheError::Invariant(
            "JPEG fast-packet nested capacity overflow",
        ))
}

fn capacity_bytes<T>(capacity: usize) -> Result<usize, JpegPlanCacheError> {
    capacity
        .checked_mul(size_of::<T>())
        .ok_or(JpegPlanCacheError::Invariant(
            "JPEG fast-packet vector capacity overflow",
        ))
}
