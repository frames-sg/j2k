// SPDX-License-Identifier: MIT OR Apache-2.0

use super::PinnedUploadStagingPool;
use crate::{allocation::host_allocation_error, context::PinnedUploadStaging, error::CudaError};

impl PinnedUploadStagingPool {
    pub(crate) fn begin_new_active_checkout(&mut self, bytes: usize) -> Result<(), CudaError> {
        let (active_buffers, active_bytes) = self.checked_added_active(bytes)?;
        self.checked_retained_totals(active_buffers, active_bytes)?;
        self.active_buffers = active_buffers;
        self.active_bytes = active_bytes;
        Ok(())
    }

    pub(crate) fn confirm_new_active_checkout(&mut self) -> Result<(), CudaError> {
        let (retained_buffers, retained_bytes) = self.checked_current_retained_totals()?;
        self.observe_high_water(retained_buffers, retained_bytes);
        Ok(())
    }

    pub(super) fn transition_cached_to_active(&mut self, bytes: usize) -> Result<(), CudaError> {
        let cached_bytes =
            self.cached_bytes
                .checked_sub(bytes)
                .ok_or(CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging byte accounting underflow",
                })?;
        let (active_buffers, active_bytes) = self.checked_added_active(bytes)?;
        let retained_buffers = self
            .buffers
            .len()
            .checked_add(self.uncertain.len())
            .and_then(|count| count.checked_add(self.active_buffers))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging retained count accounting overflow",
            })?;
        let retained_bytes = self
            .cached_bytes
            .checked_add(self.uncertain_bytes)
            .and_then(|total| total.checked_add(self.active_bytes))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging retained byte accounting overflow",
            })?;
        self.cached_bytes = cached_bytes;
        self.active_buffers = active_buffers;
        self.active_bytes = active_bytes;
        self.observe_high_water(retained_buffers, retained_bytes);
        Ok(())
    }

    pub(crate) fn finish_active_checkout(&mut self, bytes: usize) -> Result<(), CudaError> {
        let Some(active_buffers) = self.active_buffers.checked_sub(1) else {
            self.accounting_poisoned = true;
            return Err(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging active count accounting underflow",
            });
        };
        let Some(active_bytes) = self.active_bytes.checked_sub(bytes) else {
            self.accounting_poisoned = true;
            return Err(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging active byte accounting underflow",
            });
        };
        self.active_buffers = active_buffers;
        self.active_bytes = active_bytes;
        Ok(())
    }

    pub(crate) fn try_admit_active(
        &mut self,
        staging: PinnedUploadStaging,
    ) -> Result<(), (CudaError, PinnedUploadStaging)> {
        let Some(active_buffers) = self.active_buffers.checked_sub(1) else {
            return self.poisoned_active_transition("active count underflow", staging);
        };
        let Some(active_bytes) = self.active_bytes.checked_sub(staging.len) else {
            return self.poisoned_active_transition("active byte underflow", staging);
        };
        let Some(cached_bytes) = self.cached_bytes.checked_add(staging.len) else {
            return self.poisoned_active_transition("cached byte overflow", staging);
        };
        let retained = match self.checked_current_retained_totals() {
            Ok(retained) => retained,
            Err(error) => return Err((error, staging)),
        };
        if self.buffers.try_reserve(1).is_err() {
            self.metrics.metadata_failures = self.metrics.metadata_failures.saturating_add(1);
            let error =
                host_allocation_error::<PinnedUploadStaging>(self.buffers.len().saturating_add(1));
            return Err((error, staging));
        }
        self.active_buffers = active_buffers;
        self.active_bytes = active_bytes;
        self.cached_bytes = cached_bytes;
        self.buffers.push(staging);
        self.observe_high_water(retained.0, retained.1);
        Ok(())
    }

    pub(crate) fn try_quarantine_active_checkout(
        &mut self,
        staging: PinnedUploadStaging,
    ) -> Result<(), (CudaError, PinnedUploadStaging)> {
        let Some(active_buffers) = self.active_buffers.checked_sub(1) else {
            return self.poisoned_active_transition("active count underflow", staging);
        };
        let Some(active_bytes) = self.active_bytes.checked_sub(staging.len) else {
            return self.poisoned_active_transition("active byte underflow", staging);
        };
        let Some(uncertain_bytes) = self.uncertain_bytes.checked_add(staging.len) else {
            return self.poisoned_active_transition("uncertain byte overflow", staging);
        };
        let retained = match self.checked_current_retained_totals() {
            Ok(retained) => retained,
            Err(error) => return Err((error, staging)),
        };
        if self.uncertain.try_reserve(1).is_err() {
            self.accounting_poisoned = true;
            self.metrics.metadata_failures = self.metrics.metadata_failures.saturating_add(1);
            let error = host_allocation_error::<PinnedUploadStaging>(
                self.uncertain.len().saturating_add(1),
            );
            return Err((error, staging));
        }
        self.active_buffers = active_buffers;
        self.active_bytes = active_bytes;
        self.uncertain_bytes = uncertain_bytes;
        self.uncertain.push(staging);
        self.observe_high_water(retained.0, retained.1);
        Ok(())
    }

    fn checked_added_active(&self, bytes: usize) -> Result<(usize, usize), CudaError> {
        let buffers = self
            .active_buffers
            .checked_add(1)
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging active count accounting overflow",
            })?;
        let bytes = self
            .active_bytes
            .checked_add(bytes)
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging active byte accounting overflow",
            })?;
        Ok((buffers, bytes))
    }

    fn checked_retained_totals(
        &self,
        active_buffers: usize,
        active_bytes: usize,
    ) -> Result<(usize, usize), CudaError> {
        let buffers = self
            .buffers
            .len()
            .checked_add(self.uncertain.len())
            .and_then(|count| count.checked_add(active_buffers))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging retained count accounting overflow",
            })?;
        let bytes = self
            .cached_bytes
            .checked_add(self.uncertain_bytes)
            .and_then(|total| total.checked_add(active_bytes))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging retained byte accounting overflow",
            })?;
        Ok((buffers, bytes))
    }

    fn checked_current_retained_totals(&self) -> Result<(usize, usize), CudaError> {
        self.checked_retained_totals(self.active_buffers, self.active_bytes)
    }

    fn poisoned_active_transition(
        &mut self,
        reason: &'static str,
        staging: PinnedUploadStaging,
    ) -> Result<(), (CudaError, PinnedUploadStaging)> {
        self.accounting_poisoned = true;
        Err((CudaError::InternalInvariant { what: reason }, staging))
    }
}

#[cfg(test)]
mod tests;
