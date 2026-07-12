// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible allocation helpers that reconcile allocator-reported capacity.

use alloc::string::String;
use alloc::vec::Vec;
use core::mem::size_of;

use crate::{ProfileError, ProfileResult};

pub(crate) fn checked_add(left: usize, right: usize, what: &'static str) -> ProfileResult<usize> {
    left.checked_add(right)
        .ok_or(ProfileError::SizeOverflow { what })
}

pub(crate) fn element_bytes<T>(capacity: usize, what: &'static str) -> ProfileResult<usize> {
    capacity
        .checked_mul(size_of::<T>())
        .ok_or(ProfileError::SizeOverflow { what })
}

pub(crate) fn ensure_limit(
    requested: usize,
    limit: usize,
    what: &'static str,
) -> ProfileResult<()> {
    if requested > limit {
        return Err(ProfileError::LimitExceeded {
            what,
            requested,
            limit,
        });
    }
    Ok(())
}

pub(crate) struct HeapBudget {
    used: usize,
    limit: usize,
}

impl HeapBudget {
    pub(crate) const fn new(used: usize, limit: usize) -> Self {
        Self { used, limit }
    }

    pub(crate) fn include(&mut self, bytes: usize, what: &'static str) -> ProfileResult<()> {
        let requested = checked_add(self.used, bytes, what)?;
        ensure_limit(requested, self.limit, what)?;
        self.used = requested;
        Ok(())
    }

    pub(crate) const fn used(&self) -> usize {
        self.used
    }

    pub(crate) fn release(&mut self, bytes: usize, what: &'static str) -> ProfileResult<()> {
        self.used = self
            .used
            .checked_sub(bytes)
            .ok_or(ProfileError::SizeOverflow { what })?;
        Ok(())
    }
}

pub(crate) fn try_vec<T>(
    capacity: usize,
    budget: &mut HeapBudget,
    what: &'static str,
) -> ProfileResult<Vec<T>> {
    let requested_bytes = element_bytes::<T>(capacity, what)?;
    budget.include(requested_bytes, what)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| ProfileError::AllocationFailed {
            what,
            requested: capacity,
        })?;
    let actual_bytes = element_bytes::<T>(values.capacity(), what)?;
    let used_without_request =
        budget
            .used
            .checked_sub(requested_bytes)
            .ok_or(ProfileError::SizeOverflow {
                what: "profile allocation budget reconciliation",
            })?;
    let actual_used = checked_add(used_without_request, actual_bytes, what)?;
    ensure_limit(actual_used, budget.limit, what)?;
    budget.used = actual_used;
    Ok(values)
}

pub(crate) fn try_string(
    text: &str,
    budget: &mut HeapBudget,
    what: &'static str,
) -> ProfileResult<String> {
    let requested = text.len();
    budget.include(requested, what)?;
    let mut value = String::new();
    value
        .try_reserve_exact(requested)
        .map_err(|_| ProfileError::AllocationFailed { what, requested })?;
    let used_without_request =
        budget
            .used
            .checked_sub(requested)
            .ok_or(ProfileError::SizeOverflow {
                what: "profile string budget reconciliation",
            })?;
    let actual_used = checked_add(used_without_request, value.capacity(), what)?;
    ensure_limit(actual_used, budget.limit, what)?;
    budget.used = actual_used;
    value.push_str(text);
    Ok(value)
}

pub(crate) fn try_string_capacity(
    capacity: usize,
    budget: &mut HeapBudget,
    what: &'static str,
) -> ProfileResult<String> {
    budget.include(capacity, what)?;
    let mut value = String::new();
    value
        .try_reserve_exact(capacity)
        .map_err(|_| ProfileError::AllocationFailed {
            what,
            requested: capacity,
        })?;
    let used_without_request =
        budget
            .used
            .checked_sub(capacity)
            .ok_or(ProfileError::SizeOverflow {
                what: "profile string capacity reconciliation",
            })?;
    let actual_used = checked_add(used_without_request, value.capacity(), what)?;
    ensure_limit(actual_used, budget.limit, what)?;
    budget.used = actual_used;
    Ok(value)
}

pub(crate) fn try_extend_string(
    value: &mut String,
    text: &str,
    budget: &mut HeapBudget,
    what: &'static str,
) -> ProfileResult<()> {
    let required = checked_add(value.len(), text.len(), what)?;
    if required > value.capacity() {
        let old_capacity = value.capacity();
        let additional = required - value.len();
        value
            .try_reserve_exact(additional)
            .map_err(|_| ProfileError::AllocationFailed {
                what,
                requested: required,
            })?;
        let used_without_old =
            budget
                .used
                .checked_sub(old_capacity)
                .ok_or(ProfileError::SizeOverflow {
                    what: "profile string growth reconciliation",
                })?;
        let actual_used = checked_add(used_without_old, value.capacity(), what)?;
        ensure_limit(actual_used, budget.limit, what)?;
        budget.used = actual_used;
    }
    value.push_str(text);
    Ok(())
}

pub(crate) fn try_output_string(
    requested: usize,
    limit: usize,
    what: &'static str,
) -> ProfileResult<String> {
    ensure_limit(requested, limit, what)?;
    let mut value = String::new();
    value
        .try_reserve_exact(requested)
        .map_err(|_| ProfileError::AllocationFailed { what, requested })?;
    ensure_limit(value.capacity(), limit, what)?;
    Ok(value)
}

#[cfg(test)]
pub(crate) fn reconcile_test_capacity(
    retained: usize,
    requested: usize,
    actual: usize,
    limit: usize,
) -> ProfileResult<usize> {
    let planned = checked_add(retained, requested, "test allocation")?;
    ensure_limit(planned, limit, "test allocation")?;
    let actual = checked_add(retained, actual, "test allocation")?;
    ensure_limit(actual, limit, "test allocation")?;
    Ok(actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocator_reported_overcapacity_is_rejected() {
        assert_eq!(
            ProfileError::LimitExceeded {
                what: "test allocation",
                requested: 9,
                limit: 8,
            },
            reconcile_test_capacity(1, 4, 8, 8)
                .expect_err("allocator overcapacity must be reconciled")
        );
    }

    #[test]
    fn allocator_reservation_failure_is_typed() {
        let mut budget = HeapBudget::new(0, usize::MAX);
        assert_eq!(
            ProfileError::AllocationFailed {
                what: "test vector",
                requested: usize::MAX,
            },
            try_vec::<u8>(usize::MAX, &mut budget, "test vector")
                .expect_err("impossible reservation must fail")
        );
    }
}
