// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

/// Error returned when a host vector cannot reserve its requested capacity.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("host allocation failed for {requested_bytes} bytes")]
pub struct HostAllocationError {
    requested_bytes: usize,
}

impl HostAllocationError {
    /// Build an allocation error for an element count and type.
    #[doc(hidden)]
    #[must_use]
    pub const fn for_elements<T>(element_count: usize) -> Self {
        allocation_error::<T>(element_count)
    }

    /// Requested allocation size in bytes, saturated on element-size overflow.
    #[doc(hidden)]
    #[must_use]
    pub const fn requested_bytes(self) -> usize {
        self.requested_bytes
    }
}

/// Error returned when actual allocator capacity exceeds a host phase budget.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error(
    "actual host allocation capacity requires {requested_bytes} bytes, exceeding the {cap_bytes}-byte phase cap"
)]
pub struct HostAllocationLimitError {
    requested_bytes: usize,
    cap_bytes: usize,
}

impl HostAllocationLimitError {
    /// Aggregate allocator-reported bytes that would be simultaneously live.
    #[doc(hidden)]
    #[must_use]
    pub const fn requested_bytes(self) -> usize {
        self.requested_bytes
    }

    /// Maximum permitted simultaneously live host bytes.
    #[doc(hidden)]
    #[must_use]
    pub const fn cap_bytes(self) -> usize {
        self.cap_bytes
    }
}

/// Codec-neutral accounting for allocator-reported capacities in one host phase.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostAllocationBudget {
    live_bytes: usize,
    cap_bytes: usize,
}

impl HostAllocationBudget {
    /// Start an empty host phase with an explicit byte cap.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(cap_bytes: usize) -> Self {
        Self {
            live_bytes: 0,
            cap_bytes,
        }
    }

    /// Allocator-reported capacity bytes accounted in this phase.
    #[doc(hidden)]
    #[must_use]
    pub const fn live_bytes(self) -> usize {
        self.live_bytes
    }

    /// Maximum permitted simultaneously live host bytes.
    #[doc(hidden)]
    #[must_use]
    pub const fn cap_bytes(self) -> usize {
        self.cap_bytes
    }

    /// Check one owner's minimum requested capacity before asking the allocator.
    ///
    /// This does not mutate the budget. Callers can reject requests whose logical
    /// minimum already exceeds the phase cap, then account the allocator-reported
    /// capacity after a successful reservation.
    #[doc(hidden)]
    pub fn check_capacity<T>(&self, capacity: usize) -> Result<usize, HostAllocationLimitError> {
        let owner_bytes = host_capacity_bytes::<T>(capacity);
        self.checked_live_bytes(owner_bytes)?;
        Ok(owner_bytes)
    }

    /// Account one owner by its allocator-reported element capacity.
    #[doc(hidden)]
    pub fn account_capacity<T>(
        &mut self,
        capacity: usize,
    ) -> Result<usize, HostAllocationLimitError> {
        let owner_bytes = host_capacity_bytes::<T>(capacity);
        self.account_bytes(owner_bytes)?;
        Ok(owner_bytes)
    }

    /// Account one owner whose allocator-reported byte capacity is already known.
    #[doc(hidden)]
    pub fn account_bytes(&mut self, owner_bytes: usize) -> Result<(), HostAllocationLimitError> {
        let requested_bytes = self.checked_live_bytes(owner_bytes)?;
        self.live_bytes = requested_bytes;
        Ok(())
    }

    /// Account one vector owner using `Vec::capacity`, not its logical length.
    #[doc(hidden)]
    pub fn account_vec<T>(&mut self, values: &Vec<T>) -> Result<usize, HostAllocationLimitError> {
        self.account_capacity::<T>(values.capacity())
    }

    fn checked_live_bytes(&self, owner_bytes: usize) -> Result<usize, HostAllocationLimitError> {
        let requested_bytes = self.live_bytes.saturating_add(owner_bytes);
        if requested_bytes > self.cap_bytes {
            return Err(HostAllocationLimitError {
                requested_bytes,
                cap_bytes: self.cap_bytes,
            });
        }
        Ok(requested_bytes)
    }
}

/// Byte footprint implied by an allocator-reported vector capacity.
#[doc(hidden)]
#[must_use]
pub const fn host_capacity_bytes<T>(capacity: usize) -> usize {
    capacity.saturating_mul(core::mem::size_of::<T>())
}

/// Reserve an exact host vector capacity without invoking the infallible allocator path.
#[doc(hidden)]
pub fn try_host_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, HostAllocationError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| HostAllocationError::for_elements::<T>(capacity))?;
    Ok(values)
}

/// Allocate and initialize a host vector without invoking the infallible allocator path.
#[doc(hidden)]
pub fn try_host_vec_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, HostAllocationError> {
    let mut values = try_host_vec_with_capacity(len)?;
    values.resize(len, value);
    Ok(values)
}

/// Copy a slice into a host vector without invoking the infallible allocator path.
#[doc(hidden)]
pub fn try_host_vec_from_slice<T: Copy>(source: &[T]) -> Result<Vec<T>, HostAllocationError> {
    let mut values = try_host_vec_with_capacity(source.len())?;
    values.extend_from_slice(source);
    Ok(values)
}

/// Resize a host vector after fallibly reserving any required additional capacity.
#[doc(hidden)]
pub fn try_host_vec_resize<T: Clone>(
    values: &mut Vec<T>,
    new_len: usize,
    value: T,
) -> Result<(), HostAllocationError> {
    if new_len > values.len() {
        values
            .try_reserve_exact(new_len - values.len())
            .map_err(|_| HostAllocationError::for_elements::<T>(new_len))?;
    }
    values.resize(new_len, value);
    Ok(())
}

const fn allocation_error<T>(element_count: usize) -> HostAllocationError {
    HostAllocationError {
        requested_bytes: match element_count.checked_mul(core::mem::size_of::<T>()) {
            Some(bytes) => bytes,
            None => usize::MAX,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{
        host_capacity_bytes, try_host_vec_filled, try_host_vec_from_slice, try_host_vec_resize,
        try_host_vec_with_capacity, HostAllocationBudget, HostAllocationLimitError,
    };

    #[test]
    fn impossible_capacity_reports_saturated_requested_bytes() {
        let error = try_host_vec_with_capacity::<u32>(usize::MAX).unwrap_err();
        assert_eq!(error.requested_bytes(), usize::MAX);
    }

    #[test]
    fn initialized_copied_and_resized_vectors_preserve_contents() {
        assert_eq!(try_host_vec_filled(3, 7u16).unwrap(), [7, 7, 7]);
        assert_eq!(try_host_vec_from_slice(&[1u8, 2, 3]).unwrap(), [1, 2, 3]);

        let mut values = try_host_vec_from_slice(&[4u8]).unwrap();
        try_host_vec_resize(&mut values, 3, 9).unwrap();
        assert_eq!(values, [4, 9, 9]);
        try_host_vec_resize(&mut values, 1, 0).unwrap();
        assert_eq!(values, [4]);
    }

    #[test]
    fn actual_capacity_budget_accepts_exact_cap_and_rejects_one_over() {
        let mut exact = HostAllocationBudget::new(16);
        assert_eq!(exact.account_capacity::<u32>(4), Ok(16));
        assert_eq!(exact.live_bytes(), 16);

        let mut one_over = HostAllocationBudget::new(15);
        assert_eq!(
            one_over.account_capacity::<u32>(4),
            Err(HostAllocationLimitError {
                requested_bytes: 16,
                cap_bytes: 15,
            })
        );
        assert_eq!(one_over.live_bytes(), 0);
    }

    #[test]
    fn logical_capacity_preflight_does_not_mutate_the_budget() {
        let budget = HostAllocationBudget::new(16);
        assert_eq!(budget.check_capacity::<u32>(4), Ok(16));
        assert_eq!(budget.live_bytes(), 0);
        assert_eq!(
            budget.check_capacity::<u32>(5),
            Err(HostAllocationLimitError {
                requested_bytes: 20,
                cap_bytes: 16,
            })
        );
        assert_eq!(budget.live_bytes(), 0);
    }

    #[test]
    fn allocator_overcapacity_is_accounted_instead_of_logical_length() {
        let mut values = try_host_vec_with_capacity::<u8>(17).unwrap();
        values.extend_from_slice(&[0; 8]);
        let actual_bytes = host_capacity_bytes::<u8>(values.capacity());
        assert!(actual_bytes >= 17);

        let mut budget = HostAllocationBudget::new(16);
        assert_eq!(
            budget.account_vec(&values),
            Err(HostAllocationLimitError {
                requested_bytes: actual_bytes,
                cap_bytes: 16,
            })
        );
    }

    #[test]
    fn existing_vector_growth_is_reconciled_from_current_capacity() {
        let mut values = try_host_vec_with_capacity::<u8>(8).unwrap();
        values.try_reserve_exact(9).unwrap();
        let actual_bytes = host_capacity_bytes::<u8>(values.capacity());

        let mut exact = HostAllocationBudget::new(actual_bytes);
        assert_eq!(exact.account_vec(&values), Ok(actual_bytes));

        let mut one_under = HostAllocationBudget::new(actual_bytes.saturating_sub(1));
        assert!(matches!(
            one_under.account_vec(&values),
            Err(HostAllocationLimitError {
                requested_bytes,
                cap_bytes,
            }) if requested_bytes == actual_bytes && cap_bytes == actual_bytes.saturating_sub(1)
        ));
    }

    #[test]
    fn zero_sized_capacity_uses_zero_budget_bytes() {
        assert_eq!(host_capacity_bytes::<()>(usize::MAX), 0);
        let mut budget = HostAllocationBudget::new(0);
        assert_eq!(budget.account_capacity::<()>(usize::MAX), Ok(0));
    }
}
