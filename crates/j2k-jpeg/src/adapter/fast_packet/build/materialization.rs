// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity accounting for optional terminated-scan copies.

use super::super::allocation::checked_actual_vec_live_bytes;
use super::super::error::FastPacketError;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

pub(super) fn scan_live_bytes(
    initial_live_bytes: usize,
    owned_capacity: Option<usize>,
) -> Result<usize, FastPacketError> {
    match owned_capacity {
        None => Ok(initial_live_bytes),
        Some(capacity) => checked_actual_vec_live_bytes::<u8>(
            initial_live_bytes,
            capacity,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        ),
    }
}
