// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use j2k_core::{BufferError, ScratchPool, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

const ROW_SCRATCH_WHAT: &str = "J2K bounded row decode scratch";
const PACKED_SCRATCH_WHAT: &str = "J2K bounded row packed-byte scratch";
const U16_ROW_SCRATCH_WHAT: &str = "J2K bounded row u16 scratch";

/// Caller-owned reusable scratch for `j2k`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct J2kScratchPool {
    packed_bytes: Vec<u8>,
    row_u16: Vec<u16>,
}

impl J2kScratchPool {
    /// Create an empty JPEG 2000 scratch pool.
    pub const fn new() -> Self {
        Self {
            packed_bytes: Vec::new(),
            row_u16: Vec::new(),
        }
    }

    pub(crate) fn packed_bytes(&mut self, len: usize) -> Result<&mut [u8], BufferError> {
        self.prepare_with_cap(len, 0, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        Ok(&mut self.packed_bytes)
    }

    pub(crate) fn packed_bytes_and_row_u16(
        &mut self,
        packed_len: usize,
        row_len: usize,
    ) -> Result<(&mut [u8], &mut [u16]), BufferError> {
        self.prepare_with_cap(packed_len, row_len, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
        Ok((&mut self.packed_bytes, &mut self.row_u16))
    }

    fn prepare_with_cap(
        &mut self,
        packed_len: usize,
        row_len: usize,
        cap: usize,
    ) -> Result<(), BufferError> {
        let requested = checked_scratch_bytes(packed_len, row_len)?;
        if requested > cap {
            return Err(BufferError::AllocationTooLarge {
                requested,
                cap,
                what: ROW_SCRATCH_WHAT,
            });
        }

        if row_len == 0 {
            self.row_u16 = Vec::new();
        }
        let retained_with_targets = checked_scratch_bytes(
            self.packed_bytes.capacity().max(packed_len),
            self.row_u16.capacity().max(row_len),
        )?;
        if retained_with_targets > cap {
            if self.packed_bytes.capacity() > packed_len {
                self.packed_bytes = Vec::new();
            }
            if self.row_u16.capacity() > row_len {
                self.row_u16 = Vec::new();
            }
        }

        reserve_replacing_before_growth(
            &mut self.packed_bytes,
            packed_len,
            packed_len,
            PACKED_SCRATCH_WHAT,
        )?;
        let planned_row_capacity = self.row_u16.capacity().max(row_len);
        if ensure_scratch_capacity_within_cap(
            self.packed_bytes.capacity(),
            planned_row_capacity,
            cap,
        )
        .is_err()
            && self.row_u16.capacity() > row_len
        {
            self.row_u16 = Vec::new();
        }
        if let Err(error) = ensure_scratch_capacity_within_cap(
            self.packed_bytes.capacity(),
            self.row_u16.capacity().max(row_len),
            cap,
        ) {
            self.packed_bytes = Vec::new();
            self.row_u16 = Vec::new();
            return Err(error);
        }

        let row_bytes =
            row_len
                .checked_mul(core::mem::size_of::<u16>())
                .ok_or(BufferError::SizeOverflow {
                    what: U16_ROW_SCRATCH_WHAT,
                })?;
        reserve_replacing_before_growth(
            &mut self.row_u16,
            row_len,
            row_bytes,
            U16_ROW_SCRATCH_WHAT,
        )?;

        if let Err(error) = ensure_scratch_capacity_within_cap(
            self.packed_bytes.capacity(),
            self.row_u16.capacity(),
            cap,
        ) {
            self.packed_bytes = Vec::new();
            self.row_u16 = Vec::new();
            return Err(error);
        }

        self.packed_bytes.resize(packed_len, 0);
        self.row_u16.resize(row_len, 0);
        Ok(())
    }
}

fn checked_scratch_bytes(packed_len: usize, row_len: usize) -> Result<usize, BufferError> {
    row_len
        .checked_mul(core::mem::size_of::<u16>())
        .and_then(|row_bytes| packed_len.checked_add(row_bytes))
        .ok_or(BufferError::SizeOverflow {
            what: ROW_SCRATCH_WHAT,
        })
}

fn ensure_scratch_capacity_within_cap(
    packed_capacity: usize,
    row_capacity: usize,
    cap: usize,
) -> Result<usize, BufferError> {
    let retained = checked_scratch_bytes(packed_capacity, row_capacity)?;
    if retained > cap {
        return Err(BufferError::AllocationTooLarge {
            requested: retained,
            cap,
            what: ROW_SCRATCH_WHAT,
        });
    }
    Ok(retained)
}

fn reserve_replacing_before_growth<T>(
    values: &mut Vec<T>,
    target_len: usize,
    requested_bytes: usize,
    what: &'static str,
) -> Result<(), BufferError> {
    if target_len <= values.capacity() {
        return Ok(());
    }

    // Scratch contents are disposable. Release the old allocation before
    // growth so a realloc implementation cannot transiently retain both the
    // old and requested capacities outside the aggregate budget.
    *values = Vec::new();
    values
        .try_reserve_exact(target_len)
        .map_err(|_| BufferError::HostAllocationFailed {
            bytes: requested_bytes,
            what,
        })
}

#[doc(hidden)]
impl ScratchPool for J2kScratchPool {
    fn bytes_allocated(&self) -> usize {
        self.packed_bytes.capacity().saturating_add(
            self.row_u16
                .capacity()
                .saturating_mul(core::mem::size_of::<u16>()),
        )
    }

    fn reset(&mut self) {
        self.packed_bytes.clear();
        self.row_u16.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_row_scratch_has_an_exact_byte_boundary() {
        assert_eq!(checked_scratch_bytes(62, 1).unwrap(), 64);
        let mut pool = J2kScratchPool::new();
        pool.prepare_with_cap(62, 1, 64)
            .expect("exact aggregate boundary");
        assert!(pool.bytes_allocated() <= 64);

        assert!(matches!(
            pool.prepare_with_cap(63, 1, 64),
            Err(BufferError::AllocationTooLarge {
                requested: 65,
                cap: 64,
                what: ROW_SCRATCH_WHAT,
            })
        ));
    }

    #[test]
    fn stale_capacity_is_released_before_a_mixed_scratch_request() {
        let mut pool = J2kScratchPool::new();
        pool.prepare_with_cap(48, 0, 64)
            .expect("initial packed scratch");
        pool.prepare_with_cap(32, 10, 64)
            .expect("stale packed capacity is replaceable");

        assert_eq!(pool.packed_bytes.len(), 32);
        assert_eq!(pool.row_u16.len(), 10);
        assert!(pool.bytes_allocated() <= 64);
    }

    #[test]
    fn allocator_capacity_overage_is_reconciled_before_row_growth() {
        assert_eq!(
            ensure_scratch_capacity_within_cap(62, 1, 64).expect("exact aggregate capacity"),
            64
        );
        assert!(matches!(
            ensure_scratch_capacity_within_cap(63, 1, 64),
            Err(BufferError::AllocationTooLarge {
                requested: 65,
                cap: 64,
                what: ROW_SCRATCH_WHAT,
            })
        ));
    }

    #[test]
    fn row_scratch_overflow_is_typed_before_mutation() {
        let mut pool = J2kScratchPool::new();
        assert!(matches!(
            pool.prepare_with_cap(0, usize::MAX, usize::MAX),
            Err(BufferError::SizeOverflow {
                what: ROW_SCRATCH_WHAT,
            })
        ));
        assert_eq!(pool.bytes_allocated(), 0);
    }
}
