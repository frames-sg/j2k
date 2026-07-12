// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate, fallible ownership accounting for inspection metadata handoffs.

use alloc::vec::Vec;
use core::mem::size_of;

use crate::J2kError;
use j2k_core::{try_host_vec_with_capacity, BufferError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

#[derive(Clone, Copy, Debug)]
pub(super) struct ParseAllocationBudget {
    live_bytes: usize,
    cap: usize,
}

impl ParseAllocationBudget {
    pub(super) fn from_live_bytes(live_bytes: usize) -> Result<Self, J2kError> {
        Self::with_cap(live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    const fn with_cap(live_bytes: usize, cap: usize) -> Result<Self, J2kError> {
        if live_bytes > cap {
            return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: live_bytes,
                cap,
                what: "JPEG 2000 inspection metadata",
            }));
        }
        Ok(Self { live_bytes, cap })
    }

    pub(super) fn try_vec<T>(
        &mut self,
        count: usize,
        what: &'static str,
    ) -> Result<Vec<T>, J2kError> {
        let requested_bytes = element_bytes::<T>(count, what)?;
        self.ensure_additional(requested_bytes, what)?;
        let values = try_host_vec_with_capacity(count).map_err(|error| {
            J2kError::Buffer(BufferError::HostAllocationFailed {
                bytes: error.requested_bytes(),
                what,
            })
        })?;
        let actual_bytes = element_bytes::<T>(values.capacity(), what)?;
        self.ensure_additional(actual_bytes, what)?;
        self.live_bytes += actual_bytes;
        Ok(values)
    }

    pub(super) fn release_capacity<T>(&mut self, capacity: usize) -> Result<(), J2kError> {
        let bytes = element_bytes::<T>(capacity, "released JPEG 2000 metadata")?;
        self.live_bytes =
            self.live_bytes
                .checked_sub(bytes)
                .ok_or(J2kError::InternalInvariant {
                    what: "inspection allocation accounting underflow",
                })?;
        Ok(())
    }

    pub(super) const fn live_bytes(self) -> usize {
        self.live_bytes
    }

    fn ensure_additional(&self, additional: usize, what: &'static str) -> Result<(), J2kError> {
        let requested = self
            .live_bytes
            .checked_add(additional)
            .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))?;
        if requested > self.cap {
            return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested,
                cap: self.cap,
                what,
            }));
        }
        Ok(())
    }
}

pub(super) fn checked_add_bytes(
    total: &mut usize,
    additional: usize,
    what: &'static str,
) -> Result<(), J2kError> {
    *total = total
        .checked_add(additional)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))?;
    if *total > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: *total,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        }));
    }
    Ok(())
}

pub(super) fn capacity_bytes<T>(capacity: usize, what: &'static str) -> Result<usize, J2kError> {
    element_bytes::<T>(capacity, what)
}

fn element_bytes<T>(count: usize, what: &'static str) -> Result<usize, J2kError> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_budget_boundary_is_accepted_and_one_over_is_rejected() {
        let exact = ParseAllocationBudget::with_cap(8, 8).expect("exact boundary");
        assert_eq!(exact.live_bytes, 8);
        assert!(matches!(
            ParseAllocationBudget::with_cap(9, 8),
            Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: 9,
                cap: 8,
                ..
            }))
        ));
    }

    #[test]
    fn aggregate_arithmetic_overflow_is_typed() {
        let budget = ParseAllocationBudget::with_cap(1, usize::MAX).expect("small baseline");
        assert!(matches!(
            budget.ensure_additional(usize::MAX, "test inspection metadata"),
            Err(J2kError::Buffer(BufferError::SizeOverflow { .. }))
        ));
    }
}
