// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::{
    try_host_vec_filled, try_host_vec_with_capacity, HostAllocationError,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

use crate::error::JpegError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AllocationBudgetError {
    MemoryCapExceeded { requested: usize, cap: usize },
    HostAllocationFailed { bytes: usize },
}

/// Tracks the worst live-byte peak while existing vectors grow in a known
/// order. A reallocating vector may briefly own both its retained allocation
/// and the requested replacement, so final projected capacity is not enough
/// to enforce a hard live-allocation cap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReallocationPeak {
    live_bytes: usize,
    peak_bytes: usize,
}

impl ReallocationPeak {
    pub(crate) const fn new(live_bytes: usize) -> Self {
        Self {
            live_bytes,
            peak_bytes: live_bytes,
        }
    }

    pub(crate) fn include_vec<T>(&mut self, values: &Vec<T>, requested_len: usize) {
        if requested_len <= values.capacity() {
            return;
        }
        let retained_bytes = values.capacity().saturating_mul(size_of::<T>());
        let requested_bytes = requested_len.saturating_mul(size_of::<T>());
        self.include_growth(retained_bytes, requested_bytes);
    }

    pub(crate) const fn bytes(self) -> usize {
        self.peak_bytes
    }

    fn include_growth(&mut self, retained_bytes: usize, requested_bytes: usize) {
        self.peak_bytes = self
            .peak_bytes
            .max(self.live_bytes.saturating_add(requested_bytes));
        self.live_bytes = self
            .live_bytes
            .saturating_sub(retained_bytes)
            .saturating_add(requested_bytes);
    }
}

pub(crate) fn checked_allocation_bytes<T>(element_count: usize) -> Result<usize, JpegError> {
    let requested = element_count
        .checked_mul(size_of::<T>())
        .ok_or_else(cap_overflow)?;
    ensure_allocation_bytes(requested)?;
    Ok(requested)
}

pub(crate) fn checked_allocation_len<T>(left: usize, right: usize) -> Result<usize, JpegError> {
    let element_count = left.checked_mul(right).ok_or_else(cap_overflow)?;
    checked_allocation_bytes::<T>(element_count)?;
    Ok(element_count)
}

pub(crate) fn checked_add_allocation_bytes(
    total: usize,
    additional: usize,
) -> Result<usize, JpegError> {
    let requested = total.checked_add(additional).ok_or_else(cap_overflow)?;
    ensure_allocation_bytes(requested)?;
    Ok(requested)
}

pub(crate) fn ensure_allocation_bytes(requested: usize) -> Result<(), JpegError> {
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(JpegError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(())
}

pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, JpegError> {
    checked_allocation_bytes::<T>(capacity)?;
    let values = try_host_vec_with_capacity(capacity).map_err(host_allocation_error)?;
    ensure_vec_capacity_bytes(&values, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    Ok(values)
}

pub(crate) fn try_vec_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, JpegError> {
    checked_allocation_bytes::<T>(len)?;
    let values = try_host_vec_filled(len, value).map_err(host_allocation_error)?;
    ensure_vec_capacity_bytes(&values, DEFAULT_MAX_HOST_ALLOCATION_BYTES)?;
    Ok(values)
}

pub(crate) fn try_resize_filled<T: Clone>(
    values: &mut Vec<T>,
    new_len: usize,
    value: T,
) -> Result<(), JpegError> {
    try_reserve_for_len(values, new_len)?;
    values.resize(new_len, value);
    Ok(())
}

pub(crate) fn try_reserve_for_len<T>(values: &mut Vec<T>, new_len: usize) -> Result<(), JpegError> {
    try_reserve_for_len_with_budget(values, new_len, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
        .map_err(map_allocation_budget_error)
}

/// Reserve one vector while maintaining an actual-capacity live-byte total.
/// The preflight counts retained plus replacement storage for a growth, and
/// the postflight feeds the allocator's reported capacity into the next
/// reservation in the sequence.
pub(crate) fn try_reserve_for_len_with_live_budget<T>(
    values: &mut Vec<T>,
    new_len: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<(), JpegError> {
    try_reserve_for_len_with_live_budget_typed(values, new_len, live_bytes, cap)
        .map_err(map_allocation_budget_error)
}

pub(crate) fn try_new_vec_with_live_budget<T>(
    capacity: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<Vec<T>, AllocationBudgetError> {
    let initial_live_bytes = *live_bytes;
    let mut values = Vec::new();
    if let Err(error) =
        try_reserve_for_len_with_live_budget_typed(&mut values, capacity, live_bytes, cap)
    {
        *live_bytes = initial_live_bytes;
        return Err(error);
    }
    Ok(values)
}

fn try_reserve_for_len_with_live_budget_typed<T>(
    values: &mut Vec<T>,
    new_len: usize,
    live_bytes: &mut usize,
    cap: usize,
) -> Result<(), AllocationBudgetError> {
    let retained_bytes = values.capacity().saturating_mul(size_of::<T>());
    let mut peak = ReallocationPeak::new(*live_bytes);
    peak.include_vec(values, new_len);
    ensure_budget_bytes(peak.bytes(), cap)?;

    try_reserve_for_len_with_budget(values, new_len, cap)?;

    let actual_bytes = values.capacity().saturating_mul(size_of::<T>());
    *live_bytes = (*live_bytes)
        .saturating_sub(retained_bytes)
        .saturating_add(actual_bytes);
    ensure_budget_bytes(*live_bytes, cap)
}

fn try_reserve_for_len_with_budget<T>(
    values: &mut Vec<T>,
    new_len: usize,
    cap: usize,
) -> Result<(), AllocationBudgetError> {
    let requested_bytes = checked_budget_element_bytes::<T>(new_len, cap)?;
    if new_len > values.capacity() {
        values
            .try_reserve_exact(new_len.saturating_sub(values.len()))
            .map_err(|_| AllocationBudgetError::HostAllocationFailed {
                bytes: requested_bytes,
            })?;
    }
    let actual_bytes = values.capacity().checked_mul(size_of::<T>()).ok_or(
        AllocationBudgetError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        },
    )?;
    ensure_budget_bytes(actual_bytes, cap)
}

fn checked_budget_element_bytes<T>(
    element_count: usize,
    cap: usize,
) -> Result<usize, AllocationBudgetError> {
    let requested = element_count.checked_mul(size_of::<T>()).ok_or(
        AllocationBudgetError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        },
    )?;
    ensure_budget_bytes(requested, cap)?;
    Ok(requested)
}

fn ensure_budget_bytes(requested: usize, cap: usize) -> Result<(), AllocationBudgetError> {
    if requested > cap {
        return Err(AllocationBudgetError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

fn map_allocation_budget_error(error: AllocationBudgetError) -> JpegError {
    match error {
        AllocationBudgetError::MemoryCapExceeded { requested, cap } => {
            JpegError::MemoryCapExceeded { requested, cap }
        }
        AllocationBudgetError::HostAllocationFailed { bytes } => {
            JpegError::HostAllocationFailed { bytes }
        }
    }
}

fn ensure_live_bytes(requested: usize, cap: usize) -> Result<(), JpegError> {
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

fn ensure_vec_capacity_bytes<T>(values: &Vec<T>, cap: usize) -> Result<(), JpegError> {
    let requested =
        values
            .capacity()
            .checked_mul(size_of::<T>())
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    ensure_live_bytes(requested, cap)
}

fn host_allocation_error(error: HostAllocationError) -> JpegError {
    JpegError::HostAllocationFailed {
        bytes: error.requested_bytes(),
    }
}

fn cap_overflow() -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_vec_capacity_bytes, host_allocation_error, try_new_vec_with_live_budget,
        try_reserve_for_len_with_live_budget, AllocationBudgetError, HostAllocationError,
        ReallocationPeak,
    };
    use crate::JpegError;

    #[test]
    fn allocator_failure_keeps_its_typed_category() {
        let error = host_allocation_error(HostAllocationError::for_elements::<u8>(4096));
        assert_eq!(error, JpegError::HostAllocationFailed { bytes: 4096 });
    }

    #[test]
    fn final_fit_stale_growth_still_counts_old_and_new_storage() {
        let mut peak = ReallocationPeak::new(300);
        peak.include_growth(300, 400);

        assert_eq!(peak.bytes(), 700);
        assert!(peak.bytes() > 512);
    }

    #[test]
    fn prior_actual_overcapacity_affects_the_next_transient_peak() {
        let prior = alloc::vec::Vec::<u8>::with_capacity(350);
        let mut next = alloc::vec::Vec::<u8>::new();
        let mut live_bytes = prior.capacity();
        let requested = live_bytes + 200;
        let cap = requested - 1;

        assert!(matches!(
            try_reserve_for_len_with_live_budget(&mut next, 200, &mut live_bytes, cap),
            Err(JpegError::MemoryCapExceeded {
                requested: actual,
                cap: limit,
            }) if actual == requested && limit == cap
        ));
        assert_eq!(next.capacity(), 0, "preflight must run before reserve");
    }

    #[test]
    fn actual_vector_capacity_is_checked_against_the_selected_cap() {
        let values = alloc::vec::Vec::<u16>::with_capacity(33);
        let requested = values.capacity() * core::mem::size_of::<u16>();
        let cap = requested - 1;

        assert!(matches!(
            ensure_vec_capacity_bytes(&values, cap),
            Err(JpegError::MemoryCapExceeded {
                requested: actual,
                cap: limit,
            }) if actual == requested && limit == cap
        ));
    }

    #[test]
    fn failed_new_vector_budget_is_transactional() {
        let mut live_bytes = 7;

        assert_eq!(
            try_new_vec_with_live_budget::<u16>(2, &mut live_bytes, 10),
            Err(AllocationBudgetError::MemoryCapExceeded {
                requested: 11,
                cap: 10,
            })
        );
        assert_eq!(live_bytes, 7);
    }
}
