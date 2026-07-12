// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate, fallible ownership accounting for JP2/JPH metadata.

use alloc::vec::Vec;
use core::mem::size_of;

use crate::{DecodeError, Result, DEFAULT_MAX_DECODE_BYTES};

#[derive(Clone, Copy, Debug)]
pub(super) struct Jp2AllocationBudget {
    live_bytes: usize,
    cap: usize,
}

impl Jp2AllocationBudget {
    pub(super) fn from_live_bytes(live_bytes: usize) -> Result<Self> {
        Self::with_cap(live_bytes, DEFAULT_MAX_DECODE_BYTES)
    }

    const fn with_cap(live_bytes: usize, cap: usize) -> Result<Self> {
        if live_bytes > cap {
            return Err(DecodeError::AllocationTooLarge {
                what: "JP2/JPH metadata",
                requested: live_bytes,
                cap,
            });
        }
        Ok(Self { live_bytes, cap })
    }

    pub(super) fn try_vec<T>(&mut self, count: usize, what: &'static str) -> Result<Vec<T>> {
        let requested_bytes = element_bytes::<T>(count, what, self.cap)?;
        self.ensure_additional(requested_bytes, what)?;

        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| DecodeError::HostAllocationFailed {
                what,
                bytes: requested_bytes,
            })?;
        let actual_bytes = element_bytes::<T>(values.capacity(), what, self.cap)?;
        self.ensure_additional(actual_bytes, what)?;
        self.live_bytes += actual_bytes;
        Ok(values)
    }

    pub(super) fn try_copy_bytes(&mut self, source: &[u8], what: &'static str) -> Result<Vec<u8>> {
        let mut copied = self.try_vec(source.len(), what)?;
        copied.extend_from_slice(source);
        Ok(copied)
    }

    pub(super) fn release_vec<T>(&mut self, values: &Vec<T>) -> Result<()> {
        self.release_capacity::<T>(values.capacity())
    }

    pub(super) fn release_capacity<T>(&mut self, capacity: usize) -> Result<()> {
        let bytes = element_bytes::<T>(capacity, "JP2/JPH released metadata", self.cap)?;
        self.live_bytes =
            self.live_bytes
                .checked_sub(bytes)
                .ok_or(DecodeError::AllocationTooLarge {
                    what: "JP2/JPH allocation accounting",
                    requested: usize::MAX,
                    cap: self.cap,
                })?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) const fn live_bytes(self) -> usize {
        self.live_bytes
    }

    fn ensure_additional(&self, additional: usize, what: &'static str) -> Result<()> {
        let requested =
            self.live_bytes
                .checked_add(additional)
                .ok_or(DecodeError::AllocationTooLarge {
                    what,
                    requested: usize::MAX,
                    cap: self.cap,
                })?;
        if requested > self.cap {
            return Err(DecodeError::AllocationTooLarge {
                what,
                requested,
                cap: self.cap,
            });
        }
        Ok(())
    }
}

pub(super) fn checked_add_bytes(
    total: &mut usize,
    additional: usize,
    what: &'static str,
) -> Result<()> {
    *total = total
        .checked_add(additional)
        .ok_or(DecodeError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: DEFAULT_MAX_DECODE_BYTES,
        })?;
    if *total > DEFAULT_MAX_DECODE_BYTES {
        return Err(DecodeError::AllocationTooLarge {
            what,
            requested: *total,
            cap: DEFAULT_MAX_DECODE_BYTES,
        });
    }
    Ok(())
}

pub(super) fn capacity_bytes<T>(capacity: usize, what: &'static str) -> Result<usize> {
    element_bytes::<T>(capacity, what, DEFAULT_MAX_DECODE_BYTES)
}

fn element_bytes<T>(count: usize, what: &'static str, cap: usize) -> Result<usize> {
    count
        .checked_mul(size_of::<T>())
        .ok_or(DecodeError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap,
        })
}

#[cfg(test)]
mod tests {
    use super::Jp2AllocationBudget;
    use crate::DecodeError;

    #[test]
    fn exact_budget_boundary_is_accepted_and_one_over_is_rejected() {
        let exact = Jp2AllocationBudget::with_cap(8, 8).expect("exact boundary");
        assert_eq!(exact.live_bytes(), 8);
        assert!(matches!(
            Jp2AllocationBudget::with_cap(9, 8),
            Err(DecodeError::AllocationTooLarge {
                requested: 9,
                cap: 8,
                ..
            })
        ));
    }

    #[test]
    fn metadata_arithmetic_overflow_is_typed() {
        let budget = Jp2AllocationBudget::with_cap(1, usize::MAX).expect("small baseline");
        assert!(matches!(
            budget.ensure_additional(usize::MAX, "test JP2 metadata"),
            Err(DecodeError::AllocationTooLarge {
                requested: usize::MAX,
                ..
            })
        ));
    }

    #[test]
    fn moved_source_capacity_is_released_before_the_next_conversion_owner() {
        let mut budget = Jp2AllocationBudget::with_cap(8, 8).expect("full conversion peak");
        assert!(matches!(
            budget.try_vec::<u8>(1, "one-over conversion owner"),
            Err(DecodeError::AllocationTooLarge { .. })
        ));

        budget
            .release_capacity::<u8>(4)
            .expect("moved source capacity is released");
        budget
            .try_vec::<u8>(1, "next converted owner")
            .expect("released source makes room for the next owner");
    }
}
