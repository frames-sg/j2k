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

/// Error returned by a complete host phase allocation or live-byte budget.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HostPhaseError {
    /// A fallible host allocation could not reserve its requested minimum bytes.
    #[error("host allocation failed for {requested_bytes} bytes while allocating {what}")]
    AllocationFailed {
        /// Minimum requested allocation size, saturated on overflow.
        requested_bytes: usize,
        /// Static operation or owner description.
        what: &'static str,
    },
    /// A logical request or allocator-reported capacity exceeded the phase cap.
    #[error(
        "host phase {what} requires {requested_bytes} bytes, exceeding its {cap_bytes}-byte cap"
    )]
    LimitExceeded {
        /// Aggregate bytes that would be simultaneously live, saturated on overflow.
        requested_bytes: usize,
        /// Maximum permitted simultaneously live host bytes.
        cap_bytes: usize,
        /// Static host-phase description.
        what: &'static str,
    },
}

/// Codec-neutral owner of one simultaneously-live host allocation budget.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPhaseBudget {
    inner: HostAllocationBudget,
    what: &'static str,
}

impl HostPhaseBudget {
    /// Start an empty phase using the workspace host-allocation ceiling.
    #[doc(hidden)]
    #[must_use]
    pub const fn new(what: &'static str) -> Self {
        Self::with_cap(what, crate::DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    /// Start an empty phase with an explicit ceiling.
    #[doc(hidden)]
    #[must_use]
    pub const fn with_cap(what: &'static str, cap_bytes: usize) -> Self {
        Self {
            inner: HostAllocationBudget::new(cap_bytes),
            what,
        }
    }

    /// Start a default-cap phase and charge already-live owners.
    #[doc(hidden)]
    pub fn with_live_bytes(what: &'static str, live_bytes: usize) -> Result<Self, HostPhaseError> {
        let mut budget = Self::new(what);
        budget.account_bytes(live_bytes)?;
        Ok(budget)
    }

    /// Allocator-reported bytes currently charged to this phase.
    #[doc(hidden)]
    #[must_use]
    pub const fn live_bytes(&self) -> usize {
        self.inner.live_bytes()
    }

    /// Check one logical element capacity without mutating the phase.
    #[doc(hidden)]
    pub fn preflight_capacity<T>(&self, capacity: usize) -> Result<usize, HostPhaseError> {
        self.inner
            .check_capacity::<T>(capacity)
            .map_err(|error| self.limit_error(error))
    }

    /// Check additional byte ownership without mutating the phase.
    #[doc(hidden)]
    pub fn preflight_bytes(&self, additional: usize) -> Result<(), HostPhaseError> {
        let mut projected = self.inner;
        projected
            .account_bytes(additional)
            .map_err(|error| self.limit_error(error))
    }

    /// Charge allocator-reported element capacity.
    #[doc(hidden)]
    pub fn account_capacity<T>(&mut self, capacity: usize) -> Result<usize, HostPhaseError> {
        self.inner
            .account_capacity::<T>(capacity)
            .map_err(|error| self.limit_error(error))
    }

    /// Charge an already-known owner byte count.
    #[doc(hidden)]
    pub fn account_bytes(&mut self, bytes: usize) -> Result<(), HostPhaseError> {
        self.inner
            .account_bytes(bytes)
            .map_err(|error| self.limit_error(error))
    }

    /// Charge one vector by its actual allocator-reported capacity.
    #[doc(hidden)]
    pub fn account_vec<T>(&mut self, values: &Vec<T>) -> Result<usize, HostPhaseError> {
        self.account_capacity::<T>(values.capacity())
    }

    /// Fallibly reserve a vector and charge its actual capacity.
    #[doc(hidden)]
    pub fn try_vec_with_capacity<T>(&mut self, capacity: usize) -> Result<Vec<T>, HostPhaseError> {
        self.try_vec_with_capacity_named(capacity, self.what)
    }

    /// Fallibly reserve a named vector owner and charge its actual capacity.
    #[doc(hidden)]
    pub fn try_vec_with_capacity_named<T>(
        &mut self,
        capacity: usize,
        allocation_what: &'static str,
    ) -> Result<Vec<T>, HostPhaseError> {
        self.preflight_capacity::<T>(capacity)?;
        let values = try_host_vec_with_capacity(capacity)
            .map_err(|error| allocation_phase_error(error, allocation_what))?;
        self.account_vec(&values)?;
        Ok(values)
    }

    /// Fallibly allocate a filled vector and charge its actual capacity.
    #[doc(hidden)]
    pub fn try_vec_filled<T: Clone>(
        &mut self,
        len: usize,
        value: T,
    ) -> Result<Vec<T>, HostPhaseError> {
        self.preflight_capacity::<T>(len)?;
        let values = try_host_vec_filled(len, value)
            .map_err(|error| allocation_phase_error(error, self.what))?;
        self.account_vec(&values)?;
        Ok(values)
    }

    /// Fallibly copy a slice and charge the vector's actual capacity.
    #[doc(hidden)]
    pub fn try_vec_from_slice<T: Copy>(&mut self, source: &[T]) -> Result<Vec<T>, HostPhaseError> {
        self.try_vec_from_slice_named(source, self.what)
    }

    /// Fallibly clone a slice and charge the vector's actual capacity.
    #[doc(hidden)]
    pub fn try_clone_slice<T: Clone>(&mut self, source: &[T]) -> Result<Vec<T>, HostPhaseError> {
        let mut values = self.try_vec_with_capacity(source.len())?;
        values.extend_from_slice(source);
        Ok(values)
    }

    /// Fallibly copy a named slice owner and charge its actual capacity.
    #[doc(hidden)]
    pub fn try_vec_from_slice_named<T: Copy>(
        &mut self,
        source: &[T],
        allocation_what: &'static str,
    ) -> Result<Vec<T>, HostPhaseError> {
        let mut values = self.try_vec_with_capacity_named(source.len(), allocation_what)?;
        values.extend_from_slice(source);
        Ok(values)
    }

    /// Fallibly move an array into a named vector owner.
    #[doc(hidden)]
    pub fn try_vec_from_array_named<T, const N: usize>(
        &mut self,
        source: [T; N],
        allocation_what: &'static str,
    ) -> Result<Vec<T>, HostPhaseError> {
        let mut values = self.try_vec_with_capacity_named(N, allocation_what)?;
        values.extend(source);
        Ok(values)
    }

    /// Fallibly reserve a vector sized by a checked product.
    #[doc(hidden)]
    pub fn try_vec_for_product<T>(
        &mut self,
        factors: &[usize],
        allocation_what: &'static str,
    ) -> Result<Vec<T>, HostPhaseError> {
        let count = checked_host_phase_product(factors, allocation_what)?;
        self.try_vec_with_capacity_named(count, allocation_what)
    }

    /// Fallibly collect an exact-size iterator.
    #[doc(hidden)]
    pub fn try_collect_exact<T>(
        &mut self,
        iter: impl ExactSizeIterator<Item = T>,
    ) -> Result<Vec<T>, HostPhaseError> {
        let mut values = self.try_vec_with_capacity(iter.len())?;
        values.extend(iter);
        Ok(values)
    }

    /// Fallibly collect an exact-size iterator whose items use an adapter error.
    #[doc(hidden)]
    pub fn try_collect_results_exact<T, E>(
        &mut self,
        iter: impl ExactSizeIterator<Item = Result<T, E>>,
    ) -> Result<Vec<T>, E>
    where
        E: From<HostPhaseError>,
    {
        let mut values = self.try_vec_with_capacity(iter.len()).map_err(E::from)?;
        for value in iter {
            values.push(value?);
        }
        Ok(values)
    }

    /// Fallibly grow an existing vector and charge only actual capacity growth.
    #[doc(hidden)]
    pub fn try_vec_reserve<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<(), HostPhaseError> {
        let required_capacity = values.len().saturating_add(additional);
        let previous_capacity = values.capacity();
        let minimum_growth = required_capacity.saturating_sub(previous_capacity);
        self.preflight_capacity::<T>(minimum_growth)?;
        values.try_reserve_exact(additional).map_err(|_| {
            allocation_phase_error(
                HostAllocationError::for_elements::<T>(required_capacity),
                self.what,
            )
        })?;
        let actual_growth = values.capacity().saturating_sub(previous_capacity);
        self.account_capacity::<T>(actual_growth)?;
        Ok(())
    }

    /// Fallibly reserve and push one value.
    #[doc(hidden)]
    pub fn try_vec_push<T>(&mut self, values: &mut Vec<T>, value: T) -> Result<(), HostPhaseError> {
        self.try_vec_reserve(values, 1)?;
        values.push(value);
        Ok(())
    }

    /// Fallibly reserve and extend a vector from a slice.
    #[doc(hidden)]
    pub fn try_vec_extend_from_slice<T: Copy>(
        &mut self,
        values: &mut Vec<T>,
        source: &[T],
    ) -> Result<(), HostPhaseError> {
        self.try_vec_reserve(values, source.len())?;
        values.extend_from_slice(source);
        Ok(())
    }

    fn limit_error(&self, error: HostAllocationLimitError) -> HostPhaseError {
        HostPhaseError::LimitExceeded {
            requested_bytes: error.requested_bytes(),
            cap_bytes: error.cap_bytes(),
            what: self.what,
        }
    }
}

/// Checked product for host allocation element counts.
#[doc(hidden)]
pub fn checked_host_phase_product(
    factors: &[usize],
    what: &'static str,
) -> Result<usize, HostPhaseError> {
    factors.iter().try_fold(1usize, |count, factor| {
        count
            .checked_mul(*factor)
            .ok_or(HostPhaseError::LimitExceeded {
                requested_bytes: usize::MAX,
                cap_bytes: crate::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what,
            })
    })
}

/// Checked sum for host allocation element or byte counts.
#[doc(hidden)]
pub fn checked_host_phase_sum(
    values: &[usize],
    what: &'static str,
) -> Result<usize, HostPhaseError> {
    values.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or(HostPhaseError::LimitExceeded {
                requested_bytes: usize::MAX,
                cap_bytes: crate::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what,
            })
    })
}

const fn allocation_phase_error(error: HostAllocationError, what: &'static str) -> HostPhaseError {
    HostPhaseError::AllocationFailed {
        requested_bytes: error.requested_bytes(),
        what,
    }
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
        HostPhaseBudget, HostPhaseError,
    };
    use alloc::vec::Vec;
    use core::mem::size_of;

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

    #[test]
    fn phase_budget_reports_exact_cap_and_one_byte_over_with_context() {
        let mut exact = HostPhaseBudget::with_cap("test phase", 8);
        let values = exact
            .try_vec_with_capacity::<u32>(2)
            .expect("exact phase cap");
        assert_eq!(exact.live_bytes(), values.capacity() * size_of::<u32>());

        let mut one_under =
            HostPhaseBudget::with_cap("test phase", values.capacity() * size_of::<u32>() - 1);
        assert!(matches!(
            one_under.account_vec(&values),
            Err(HostPhaseError::LimitExceeded {
                requested_bytes,
                cap_bytes,
                what: "test phase",
            }) if requested_bytes == values.capacity() * size_of::<u32>()
                && cap_bytes + 1 == requested_bytes
        ));
    }

    #[test]
    fn phase_budget_growth_and_zero_sized_values_use_actual_capacity() {
        let mut budget = HostPhaseBudget::with_cap("growth", usize::MAX);
        let mut values = Vec::<u8>::new();
        budget.try_vec_push(&mut values, 1).expect("first growth");
        let first_capacity = values.capacity();
        assert_eq!(budget.live_bytes(), first_capacity);
        budget
            .try_vec_push(&mut values, 2)
            .expect("reused capacity");
        assert_eq!(budget.live_bytes(), values.capacity());

        let mut zero = HostPhaseBudget::with_cap("zst", 0);
        assert!(zero.try_vec_with_capacity::<()>(usize::MAX).is_ok());
        assert_eq!(zero.live_bytes(), 0);
    }

    #[test]
    fn failed_phase_allocation_does_not_charge_the_budget() {
        let mut budget = HostPhaseBudget::with_cap("failure", usize::MAX);
        assert!(matches!(
            budget.try_vec_with_capacity::<u32>(usize::MAX),
            Err(HostPhaseError::AllocationFailed {
                requested_bytes: usize::MAX,
                what: "failure",
            })
        ));
        assert_eq!(budget.live_bytes(), 0);
    }
}
