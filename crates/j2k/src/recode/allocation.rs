// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate fallible allocations owned by facade recode paths.

use alloc::vec::Vec;
use core::mem::size_of;

use crate::J2kError;
use j2k_core::{try_host_vec_with_capacity, BufferError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

#[derive(Clone, Copy, Debug)]
pub(super) struct RecodeAllocationBudget {
    live_bytes: usize,
    cap: usize,
}

impl RecodeAllocationBudget {
    pub(super) fn from_live_bytes(live_bytes: usize) -> Result<Self, J2kError> {
        Self::with_cap(live_bytes, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    const fn with_cap(live_bytes: usize, cap: usize) -> Result<Self, J2kError> {
        if live_bytes > cap {
            return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: live_bytes,
                cap,
                what: "HTJ2K recode retained owners",
            }));
        }
        Ok(Self { live_bytes, cap })
    }

    pub(super) fn include_bytes(
        &mut self,
        bytes: usize,
        what: &'static str,
    ) -> Result<(), J2kError> {
        let requested = self
            .live_bytes
            .checked_add(bytes)
            .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))?;
        if requested > self.cap {
            return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested,
                cap: self.cap,
                what,
            }));
        }
        self.live_bytes = requested;
        Ok(())
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
        self.include_bytes(actual_bytes, what)?;
        Ok(values)
    }

    #[cfg(test)]
    pub(super) const fn live_bytes(self) -> usize {
        self.live_bytes
    }

    fn ensure_additional(&self, bytes: usize, what: &'static str) -> Result<(), J2kError> {
        let requested = self
            .live_bytes
            .checked_add(bytes)
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

pub(super) fn copy_bytes(
    source: &[u8],
    retained_bytes: usize,
    what: &'static str,
) -> Result<Vec<u8>, J2kError> {
    let mut budget = RecodeAllocationBudget::from_live_bytes(retained_bytes)?;
    let mut output = budget.try_vec(source.len(), what)?;
    output.extend_from_slice(source);
    Ok(output)
}

pub(super) fn checked_add_owned_bytes(
    first: usize,
    second: usize,
    what: &'static str,
) -> Result<usize, J2kError> {
    let total = first
        .checked_add(second)
        .ok_or(J2kError::Buffer(BufferError::SizeOverflow { what }))?;
    if total > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(J2kError::Buffer(BufferError::AllocationTooLarge {
            requested: total,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        }));
    }
    Ok(total)
}

/// `Vec<bool>::capacity()` is measured in bits rather than elements/bytes.
#[cfg(test)]
pub(super) const fn bit_vector_capacity_bytes(capacity_bits: usize) -> usize {
    capacity_bits.div_ceil(8)
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
    fn exact_recode_owner_boundary_is_accepted_and_one_over_is_rejected() {
        let exact = RecodeAllocationBudget::with_cap(8, 8).expect("exact boundary");
        assert_eq!(exact.live_bytes(), 8);
        assert!(matches!(
            RecodeAllocationBudget::with_cap(9, 8),
            Err(J2kError::Buffer(BufferError::AllocationTooLarge {
                requested: 9,
                cap: 8,
                ..
            }))
        ));
    }

    #[test]
    fn owned_source_and_output_are_aggregate_checked() {
        assert_eq!(
            checked_add_owned_bytes(
                DEFAULT_MAX_HOST_ALLOCATION_BYTES - 2,
                2,
                "test recode owners"
            )
            .expect("exact boundary"),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_add_owned_bytes(
                DEFAULT_MAX_HOST_ALLOCATION_BYTES - 2,
                3,
                "test recode owners"
            ),
            Err(J2kError::Buffer(BufferError::AllocationTooLarge { .. }))
        ));
    }

    #[test]
    fn bit_vector_capacity_is_converted_to_owner_bytes() {
        assert_eq!(bit_vector_capacity_bytes(0), 0);
        assert_eq!(bit_vector_capacity_bytes(1), 1);
        assert_eq!(bit_vector_capacity_bytes(8), 1);
        assert_eq!(bit_vector_capacity_bytes(9), 2);
    }
}
