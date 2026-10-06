// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate accounting for allocations created while decoded tile storage is live.

use super::DecompositionStorage;
use crate::error::{Result, ValidationError};
use crate::{try_reserve_decode_elements, try_resize_decode_elements, DEFAULT_MAX_DECODE_BYTES};
use alloc::vec::Vec;
use core::mem::size_of;

#[derive(Clone, Copy)]
pub(crate) struct DecodeAllocationBudget {
    live_bytes: usize,
    #[cfg(test)]
    cap: usize,
}

impl DecodeAllocationBudget {
    pub(crate) fn for_storage(storage: &DecompositionStorage<'_>) -> Result<Self> {
        // The structural plan carries parsed metadata, channels, coefficient
        // and graph capacities, ROI arrays, and IDWT workspace. Segment
        // ownership is intentionally separate so retained capacity and packet
        // growth can be charged exactly once here.
        let segment_bytes = storage
            .segments
            .capacity()
            .checked_mul(size_of::<super::Segment<'_>>())
            .ok_or(ValidationError::ImageTooLarge)?;
        let live_bytes = storage
            .structural_workspace_bytes
            .checked_add(segment_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        Self::from_live_bytes(live_bytes)
    }

    pub(crate) fn from_live_bytes(live_bytes: usize) -> Result<Self> {
        if live_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(Self {
            live_bytes,
            #[cfg(test)]
            cap: DEFAULT_MAX_DECODE_BYTES,
        })
    }

    #[cfg(test)]
    pub(crate) fn from_live_bytes_with_cap(live_bytes: usize, cap: usize) -> Result<Self> {
        if live_bytes > cap {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(Self { live_bytes, cap })
    }

    pub(crate) const fn live_bytes(self) -> usize {
        self.live_bytes
    }

    pub(crate) fn include_elements<T>(&mut self, count: usize) -> Result<()> {
        let additional = count
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        self.include_bytes(additional)
    }

    pub(crate) fn include_bytes(&mut self, additional: usize) -> Result<()> {
        self.live_bytes = self
            .live_bytes
            .checked_add(additional)
            .ok_or(ValidationError::ImageTooLarge)?;
        #[cfg(test)]
        let cap = self.cap;
        #[cfg(not(test))]
        let cap = DEFAULT_MAX_DECODE_BYTES;
        if self.live_bytes > cap {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(())
    }

    pub(crate) fn include_capacity_overage<T>(
        &mut self,
        planned_count: usize,
        actual_capacity: usize,
    ) -> Result<()> {
        if actual_capacity > planned_count {
            self.include_elements::<T>(actual_capacity - planned_count)?;
        }
        Ok(())
    }

    pub(crate) fn reserve_new<T>(&mut self, values: &mut Vec<T>, target_len: usize) -> Result<()> {
        *values = Vec::new();
        self.include_elements::<T>(target_len)?;
        try_reserve_decode_elements(values, target_len)?;
        if let Err(error) = self.include_capacity_overage::<T>(target_len, values.capacity()) {
            *values = Vec::new();
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn reserve_additional<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
    ) -> Result<()> {
        let target_len = values
            .len()
            .checked_add(additional)
            .ok_or(ValidationError::ImageTooLarge)?;
        if target_len <= values.capacity() {
            return Ok(());
        }

        let old_capacity = values.capacity();
        let geometric_capacity = old_capacity.saturating_mul(2).max(target_len);
        // A growing Vec can keep its old allocation live until the replacement
        // succeeds. Prefer geometric growth when that replacement peak fits;
        // near the cap, fall back to the exact required replacement.
        let mut replacement_budget = *self;
        let replacement_capacity = if replacement_budget
            .include_elements::<T>(geometric_capacity)
            .is_ok()
        {
            geometric_capacity
        } else {
            replacement_budget = *self;
            replacement_budget.include_elements::<T>(target_len)?;
            target_len
        };
        try_reserve_decode_elements(values, replacement_capacity)?;
        replacement_budget
            .include_capacity_overage::<T>(replacement_capacity, values.capacity())?;
        replacement_budget.release_elements::<T>(old_capacity)?;
        *self = replacement_budget;
        Ok(())
    }

    pub(crate) fn resize_new<T: Clone>(
        &mut self,
        values: &mut Vec<T>,
        target_len: usize,
        value: T,
    ) -> Result<()> {
        *values = Vec::new();
        self.include_elements::<T>(target_len)?;
        try_resize_decode_elements(values, target_len, value)?;
        if let Err(error) = self.include_capacity_overage::<T>(target_len, values.capacity()) {
            *values = Vec::new();
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn release_elements<T>(&mut self, count: usize) -> Result<()> {
        let released = count
            .checked_mul(size_of::<T>())
            .ok_or(ValidationError::ImageTooLarge)?;
        self.live_bytes = self
            .live_bytes
            .checked_sub(released)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    }

    #[cfg(feature = "parallel")]
    pub(crate) fn release_bytes(&mut self, released: usize) -> Result<()> {
        self.live_bytes = self
            .live_bytes
            .checked_sub(released)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeAllocationBudget;
    use crate::error::{DecodeError, ValidationError};
    use crate::DEFAULT_MAX_DECODE_BYTES;
    use alloc::vec::Vec;

    #[test]
    fn aggregate_budget_rejects_a_second_live_owner() {
        let mut budget =
            DecodeAllocationBudget::from_live_bytes(DEFAULT_MAX_DECODE_BYTES - size_of::<u16>())
                .expect("baseline fits");
        let mut owner: Vec<u16> = Vec::new();
        budget.reserve_new(&mut owner, 1).expect("first owner fits");

        let error = budget
            .include_elements::<u8>(1)
            .expect_err("second live owner exceeds cap");
        assert!(matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ));
    }

    #[test]
    fn incremental_growth_is_geometric_with_exact_fallback_near_the_cap() {
        let mut geometric = DecodeAllocationBudget::from_live_bytes(0).expect("empty budget");
        let mut records = Vec::<u32>::new();
        for value in 0..9 {
            geometric
                .reserve_additional(&mut records, 1)
                .expect("geometric record growth");
            records.push(value);
        }
        assert!(records.capacity() >= 16);
        assert_eq!(
            geometric.live_bytes(),
            records.capacity() * size_of::<u32>()
        );

        let replacement_peak_elements = 4 + 5;
        let mut tight = DecodeAllocationBudget::from_live_bytes_with_cap(
            0,
            replacement_peak_elements * size_of::<u32>(),
        )
        .expect("tight budget");
        let mut tight_records = Vec::<u32>::new();
        tight
            .reserve_new(&mut tight_records, 4)
            .expect("initial records");
        tight_records.extend(0..4);
        tight
            .reserve_additional(&mut tight_records, 1)
            .expect("exact replacement fits when doubled replacement does not");
        assert_eq!(tight_records.capacity(), 5);
        assert_eq!(tight.live_bytes(), 5 * size_of::<u32>());
    }
}
