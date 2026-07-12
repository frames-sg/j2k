// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{HostLiveBudget, JpegToHtj2kError};
use crate::allocation::{checked_capacity_bytes, try_vec_with_capacity};

const OUT_OF_RANGE_RESULT: &str = "batch worker returned an out-of-range tile index";
const DUPLICATE_RESULT: &str = "batch worker returned duplicate results for one tile";
const MISSING_RESULT: &str = "batch worker did not return a result for every tile";

pub(super) struct BatchResultSlots<T> {
    slots: Vec<Option<T>>,
}

impl<T> BatchResultSlots<T> {
    pub(super) fn try_new(len: usize) -> Result<Self, JpegToHtj2kError> {
        let mut slots = try_vec_with_capacity(len)?;
        slots.resize_with(len, || None);
        Ok(Self { slots })
    }

    pub(super) fn insert(&mut self, tile_index: usize, result: T) -> Result<(), JpegToHtj2kError> {
        let slot = self
            .slots
            .get_mut(tile_index)
            .ok_or(JpegToHtj2kError::InternalInvariant {
                what: OUT_OF_RANGE_RESULT,
            })?;
        if slot.is_some() {
            return Err(JpegToHtj2kError::InternalInvariant {
                what: DUPLICATE_RESULT,
            });
        }
        *slot = Some(result);
        Ok(())
    }

    pub(super) fn retained_slot_bytes(&self) -> Result<usize, JpegToHtj2kError> {
        Ok(checked_capacity_bytes::<Option<T>>(self.slots.capacity())?)
    }

    #[cfg(test)]
    pub(super) fn into_results(self) -> Result<Vec<T>, JpegToHtj2kError> {
        self.into_results_with_live_budget(0, |_| Ok(0))
    }

    pub(super) fn into_results_with_live_budget(
        self,
        external_live_bytes: usize,
        retained_bytes: impl Fn(&T) -> Result<usize, JpegToHtj2kError>,
    ) -> Result<Vec<T>, JpegToHtj2kError> {
        let mut results = try_vec_with_capacity(self.slots.len())?;
        let mut budget = HostLiveBudget::process_cap();
        budget.add_bytes(external_live_bytes)?;
        budget.add_capacity::<Option<T>>(self.slots.capacity())?;
        budget.add_capacity::<T>(results.capacity())?;
        for result in self.slots.iter().flatten() {
            budget.add_bytes(retained_bytes(result)?)?;
        }
        for slot in self.slots {
            results.push(slot.ok_or(JpegToHtj2kError::InternalInvariant {
                what: MISSING_RESULT,
            })?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests;
