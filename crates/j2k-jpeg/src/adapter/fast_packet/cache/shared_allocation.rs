// SPDX-License-Identifier: MIT OR Apache-2.0

//! Accounting for the small shared-allocation portion of cache owners.

use core::mem::size_of;

use super::JpegPlanCacheError;

// Stable Rust exposes neither `Arc::try_new` nor the allocator-rounded byte
// size of an Arc control block. Size-dependent Vec owners are therefore fully
// fallible and charged by actual capacity, while this small fixed estimate
// covers the two reference counters used by the standard Arc representation.
const ARC_COUNTER_BYTES_ESTIMATE: usize = 2 * size_of::<usize>();

pub(super) fn shared_owner_bytes<T>(nested_bytes: usize) -> Result<usize, JpegPlanCacheError> {
    size_of::<T>()
        .checked_add(ARC_COUNTER_BYTES_ESTIMATE)
        .and_then(|bytes| bytes.checked_add(nested_bytes))
        .ok_or(JpegPlanCacheError::Invariant(
            "shared JPEG owner retained-byte count overflow",
        ))
}

pub(super) fn shared_slice_owner_bytes(payload_bytes: usize) -> Result<usize, JpegPlanCacheError> {
    ARC_COUNTER_BYTES_ESTIMATE
        .checked_add(payload_bytes)
        .ok_or(JpegPlanCacheError::Invariant(
            "shared JPEG slice owner retained-byte count overflow",
        ))
}

pub(super) fn checked_live_bytes(
    what: &'static str,
    live_bytes: usize,
    additional_bytes: usize,
    cap: usize,
) -> Result<usize, JpegPlanCacheError> {
    let requested = live_bytes.saturating_add(additional_bytes);
    if requested > cap {
        return Err(JpegPlanCacheError::Limit {
            what,
            requested,
            cap,
        });
    }
    Ok(requested)
}
