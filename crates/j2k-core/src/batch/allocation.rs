// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::BatchInfrastructureError;
use crate::{try_host_vec_filled, try_host_vec_with_capacity, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

/// Checked element-count request used to preflight a heterogeneous batch live set.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct BatchAllocationRequest {
    count: usize,
    element_bytes: usize,
}

impl BatchAllocationRequest {
    /// Describe `count` elements of `T` without allocating them.
    #[doc(hidden)]
    #[must_use]
    pub const fn of<T>(count: usize) -> Self {
        Self {
            count,
            element_bytes: size_of::<T>(),
        }
    }

    fn bytes(self, phase: &'static str, cap: usize) -> Result<usize, BatchInfrastructureError> {
        self.count.checked_mul(self.element_bytes).ok_or(
            BatchInfrastructureError::AllocationTooLarge {
                what: phase,
                requested: usize::MAX,
                cap,
            },
        )
    }
}

/// Backend-neutral accounting for simultaneously live fallible batch vectors.
#[doc(hidden)]
pub struct BatchAllocationBudget {
    phase: &'static str,
    live_bytes: usize,
    cap: usize,
}

impl BatchAllocationBudget {
    /// Start a batch budget using the workspace host-allocation ceiling.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(phase: &'static str) -> Self {
        Self::with_cap(phase, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    /// Start a batch budget while charging owners retained by the caller.
    ///
    /// `external_live_bytes` must include every host owner that remains live
    /// throughout the batch metadata operation.
    #[doc(hidden)]
    #[must_use]
    pub const fn with_external_live(phase: &'static str, external_live_bytes: usize) -> Self {
        Self {
            phase,
            live_bytes: external_live_bytes,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        }
    }

    /// Start a batch budget with an explicit ceiling, primarily for exact-boundary tests.
    #[doc(hidden)]
    #[must_use]
    pub const fn with_cap(phase: &'static str, cap: usize) -> Self {
        Self {
            phase,
            live_bytes: 0,
            cap,
        }
    }

    /// Bytes already charged to this live set.
    #[doc(hidden)]
    #[must_use]
    pub const fn live_bytes(&self) -> usize {
        self.live_bytes
    }

    /// Check a complete heterogeneous live set before allocating any owner in it.
    #[doc(hidden)]
    pub fn preflight(
        &self,
        requests: &[BatchAllocationRequest],
    ) -> Result<(), BatchInfrastructureError> {
        let requested = requests
            .iter()
            .try_fold(self.live_bytes, |total, request| {
                total
                    .checked_add(request.bytes(self.phase, self.cap)?)
                    .ok_or(BatchInfrastructureError::AllocationTooLarge {
                        what: self.phase,
                        requested: usize::MAX,
                        cap: self.cap,
                    })
            })?;
        self.ensure_within_cap(requested)
    }

    /// Reserve an exact vector and charge its allocator-returned capacity.
    #[doc(hidden)]
    pub fn try_vec<T>(
        &mut self,
        count: usize,
        what: &'static str,
    ) -> Result<Vec<T>, BatchInfrastructureError> {
        let requested_bytes = BatchAllocationRequest::of::<T>(count).bytes(self.phase, self.cap)?;
        self.ensure_additional(requested_bytes)?;
        let values = try_host_vec_with_capacity(count).map_err(|_| {
            BatchInfrastructureError::HostAllocationFailed {
                what,
                bytes: requested_bytes,
            }
        })?;
        self.account_capacity::<T>(values.capacity())?;
        Ok(values)
    }

    /// Reserve and initialize a vector, charging its allocator-returned capacity.
    #[doc(hidden)]
    pub fn try_filled<T: Clone>(
        &mut self,
        count: usize,
        value: T,
        what: &'static str,
    ) -> Result<Vec<T>, BatchInfrastructureError> {
        let requested_bytes = BatchAllocationRequest::of::<T>(count).bytes(self.phase, self.cap)?;
        self.ensure_additional(requested_bytes)?;
        let values = try_host_vec_filled(count, value).map_err(|_| {
            BatchInfrastructureError::HostAllocationFailed {
                what,
                bytes: requested_bytes,
            }
        })?;
        self.account_capacity::<T>(values.capacity())?;
        Ok(values)
    }

    /// Charge an already allocated vector's actual capacity to this live set.
    #[doc(hidden)]
    pub fn account_capacity<T>(&mut self, capacity: usize) -> Result<(), BatchInfrastructureError> {
        let actual_bytes = BatchAllocationRequest::of::<T>(capacity).bytes(self.phase, self.cap)?;
        let actual_live = self.live_bytes.checked_add(actual_bytes).ok_or(
            BatchInfrastructureError::AllocationTooLarge {
                what: self.phase,
                requested: usize::MAX,
                cap: self.cap,
            },
        )?;
        self.ensure_within_cap(actual_live)?;
        self.live_bytes = actual_live;
        Ok(())
    }

    fn ensure_additional(&self, additional: usize) -> Result<(), BatchInfrastructureError> {
        let requested = self.live_bytes.checked_add(additional).ok_or(
            BatchInfrastructureError::AllocationTooLarge {
                what: self.phase,
                requested: usize::MAX,
                cap: self.cap,
            },
        )?;
        self.ensure_within_cap(requested)
    }

    fn ensure_within_cap(&self, requested: usize) -> Result<(), BatchInfrastructureError> {
        if requested > self.cap {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: self.phase,
                requested,
                cap: self.cap,
            });
        }
        Ok(())
    }
}

/// Checked sum for caller-derived batch counts.
#[doc(hidden)]
pub fn checked_batch_count_sum(
    counts: impl IntoIterator<Item = usize>,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    counts.into_iter().try_fold(0usize, |total, count| {
        total
            .checked_add(count)
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
    })
}

/// Checked product for caller-derived batch counts.
#[doc(hidden)]
pub fn checked_batch_count_product(
    left: usize,
    right: usize,
    what: &'static str,
) -> Result<usize, BatchInfrastructureError> {
    left.checked_mul(right)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })
}

/// Grow a batch vector geometrically before a push, with checked bytes and typed failure.
#[doc(hidden)]
pub fn try_batch_reserve_for_push<T>(
    values: &mut Vec<T>,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    if values.len() < values.capacity() {
        return Ok(());
    }
    let required =
        values
            .len()
            .checked_add(1)
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    let target = values
        .capacity()
        .checked_mul(2)
        .ok_or(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?
        .max(required);
    try_batch_reserve_to(values, target, what)
}

/// Reserve a batch vector to at least `target` elements with typed failure.
#[doc(hidden)]
pub fn try_batch_reserve_to<T>(
    values: &mut Vec<T>,
    target: usize,
    what: &'static str,
) -> Result<(), BatchInfrastructureError> {
    if target <= values.capacity() {
        return Ok(());
    }
    let requested_bytes =
        target
            .checked_mul(size_of::<T>())
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    if requested_bytes > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: requested_bytes,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    let additional =
        target
            .checked_sub(values.len())
            .ok_or(BatchInfrastructureError::AllocationTooLarge {
                what,
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    values.try_reserve_exact(additional).map_err(|_| {
        BatchInfrastructureError::HostAllocationFailed {
            what,
            bytes: requested_bytes,
        }
    })?;
    let actual_bytes = values.capacity().checked_mul(size_of::<T>()).ok_or(
        BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        },
    )?;
    if actual_bytes > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(BatchInfrastructureError::AllocationTooLarge {
            what,
            requested: actual_bytes,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BatchAllocationBudget, BatchAllocationRequest};
    use crate::{BatchInfrastructureError, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

    #[test]
    fn aggregate_plan_accepts_exact_cap_and_rejects_one_byte_over() {
        let requests = [
            BatchAllocationRequest::of::<u32>(2),
            BatchAllocationRequest::of::<u8>(4),
        ];
        BatchAllocationBudget::with_cap("test batch metadata", 12)
            .preflight(&requests)
            .expect("exact cap");
        assert_eq!(
            BatchAllocationBudget::with_cap("test batch metadata", 11)
                .preflight(&requests)
                .expect_err("one byte over cap"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "test batch metadata",
                requested: 12,
                cap: 11,
            }
        );
    }

    #[test]
    fn external_owner_baseline_is_part_of_the_same_batch_cap() {
        let external = DEFAULT_MAX_HOST_ALLOCATION_BYTES - 4;
        let request = BatchAllocationRequest::of::<u32>(1);
        BatchAllocationBudget::with_external_live("test collective owners", external)
            .preflight(&[request])
            .expect("external owners plus metadata at exact cap");

        assert_eq!(
            BatchAllocationBudget::with_external_live("test collective owners", external + 1)
                .preflight(&[request])
                .expect_err("external owners plus metadata one byte over"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "test collective owners",
                requested: DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }
        );
    }

    #[test]
    fn element_byte_overflow_is_saturated_and_typed() {
        assert_eq!(
            BatchAllocationBudget::with_cap("test batch metadata", usize::MAX)
                .preflight(&[BatchAllocationRequest::of::<u16>(usize::MAX)])
                .expect_err("element byte overflow"),
            BatchInfrastructureError::AllocationTooLarge {
                what: "test batch metadata",
                requested: usize::MAX,
                cap: usize::MAX,
            }
        );
    }

    #[test]
    fn actual_capacities_are_cumulative_and_allocator_failure_is_exact() {
        let mut exact = BatchAllocationBudget::with_cap("test batch metadata", 8);
        let first = exact.try_vec::<u32>(1, "first").expect("first vector");
        let second = exact.try_vec::<u32>(1, "second").expect("second vector");
        assert_eq!(first.capacity(), 1);
        assert_eq!(second.capacity(), 1);

        let mut impossible = BatchAllocationBudget::with_cap("test batch metadata", usize::MAX);
        assert_eq!(
            impossible
                .try_vec::<u8>(usize::MAX, "impossible vector")
                .expect_err("allocator rejection"),
            BatchInfrastructureError::HostAllocationFailed {
                what: "impossible vector",
                bytes: usize::MAX,
            }
        );
    }
}
