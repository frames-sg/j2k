// SPDX-License-Identifier: MIT OR Apache-2.0

mod active;
mod diagnostics;

use self::diagnostics::PinnedUploadStagingPoolMetrics;
pub use self::diagnostics::{
    CudaPinnedUploadStagingPoolDiagnostics, CudaPinnedUploadStagingPoolLimits,
};
use crate::{allocation::host_allocation_error, context::PinnedUploadStaging, error::CudaError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PinnedUploadStagingAdmission {
    Admit,
    Evict,
    Reject,
}

pub(crate) struct PinnedUploadStagingPool {
    buffers: Vec<PinnedUploadStaging>,
    cached_bytes: usize,
    uncertain: Vec<PinnedUploadStaging>,
    uncertain_bytes: usize,
    active_buffers: usize,
    active_bytes: usize,
    accounting_poisoned: bool,
    limits: CudaPinnedUploadStagingPoolLimits,
    metrics: PinnedUploadStagingPoolMetrics,
}

impl PinnedUploadStagingPool {
    pub(crate) fn new() -> Self {
        Self::with_limits(CudaPinnedUploadStagingPoolLimits::default())
    }

    pub(crate) fn with_limits(limits: CudaPinnedUploadStagingPoolLimits) -> Self {
        Self {
            buffers: Vec::new(),
            cached_bytes: 0,
            uncertain: Vec::new(),
            uncertain_bytes: 0,
            active_buffers: 0,
            active_bytes: 0,
            accounting_poisoned: false,
            limits,
            metrics: PinnedUploadStagingPoolMetrics::default(),
        }
    }

    pub(crate) fn diagnostics(&self) -> Result<CudaPinnedUploadStagingPoolDiagnostics, CudaError> {
        if self.accounting_poisoned {
            return Err(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging has an untracked uncertain allocation",
            });
        }
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
        Ok(CudaPinnedUploadStagingPoolDiagnostics {
            limits: self.limits,
            cached_buffers: self.buffers.len(),
            cached_bytes: self.cached_bytes,
            uncertain_buffers: self.uncertain.len(),
            uncertain_bytes: self.uncertain_bytes,
            active_buffers: self.active_buffers,
            active_bytes: self.active_bytes,
            retained_buffers,
            retained_bytes,
            peak_cached_buffers: self.metrics.peak_cached_buffers,
            peak_cached_bytes: self.metrics.peak_cached_bytes,
            peak_uncertain_buffers: self.metrics.peak_uncertain_buffers,
            peak_uncertain_bytes: self.metrics.peak_uncertain_bytes,
            peak_active_buffers: self.metrics.peak_active_buffers,
            peak_active_bytes: self.metrics.peak_active_bytes,
            peak_retained_buffers: self.metrics.peak_retained_buffers,
            peak_retained_bytes: self.metrics.peak_retained_bytes,
            evicted_buffers: self.metrics.evicted_buffers,
            rejected_buffers: self.metrics.rejected_buffers,
            metadata_failures: self.metrics.metadata_failures,
        })
    }

    pub(crate) fn take_best_fit(
        &mut self,
        minimum_len: usize,
    ) -> Result<Option<PinnedUploadStaging>, CudaError> {
        self.ensure_no_uncertain_release()?;
        let Some(index) = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.len >= minimum_len)
            .min_by_key(|(index, buffer)| (buffer.len, *index))
            .map(|(index, _)| index)
        else {
            return Ok(None);
        };
        self.transition_cached_to_active(self.buffers[index].len)?;
        let staging = self.buffers.remove(index);
        Ok(Some(staging))
    }

    pub(crate) fn admission(
        &self,
        candidate_bytes: usize,
    ) -> Result<PinnedUploadStagingAdmission, CudaError> {
        self.ensure_no_uncertain_release()?;
        if self.limits.max_cached_buffers == 0 || candidate_bytes > self.limits.max_cached_bytes {
            return Ok(PinnedUploadStagingAdmission::Reject);
        }
        let next_buffers =
            self.buffers
                .len()
                .checked_add(1)
                .ok_or(CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging count accounting overflow",
                })?;
        let next_bytes =
            self.cached_bytes
                .checked_add(candidate_bytes)
                .ok_or(CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging byte accounting overflow",
                })?;
        if next_buffers <= self.limits.max_cached_buffers
            && next_bytes <= self.limits.max_cached_bytes
        {
            Ok(PinnedUploadStagingAdmission::Admit)
        } else if self.buffers.is_empty() {
            Ok(PinnedUploadStagingAdmission::Reject)
        } else {
            Ok(PinnedUploadStagingAdmission::Evict)
        }
    }

    pub(crate) fn cached_plus_request_fits_host_cap(
        &self,
        requested_bytes: usize,
        host_cap: usize,
    ) -> Result<bool, CudaError> {
        self.ensure_no_uncertain_release()?;
        let aggregate = self
            .cached_bytes
            .checked_add(self.active_bytes)
            .and_then(|total| total.checked_add(requested_bytes))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging current-request accounting overflow",
            })?;
        Ok(aggregate <= host_cap)
    }

    pub(crate) fn evict_largest_oldest(
        &mut self,
    ) -> Result<Option<PinnedUploadStaging>, CudaError> {
        let Some(index) = self
            .buffers
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.len
                    .cmp(&right.len)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
        else {
            return Ok(None);
        };
        let next_cached_bytes = self
            .cached_bytes
            .checked_sub(self.buffers[index].len)
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA pinned upload staging byte accounting underflow",
            })?;
        let staging = self.buffers.remove(index);
        self.cached_bytes = next_cached_bytes;
        self.metrics.evicted_buffers = self.metrics.evicted_buffers.saturating_add(1);
        Ok(Some(staging))
    }

    pub(crate) fn note_rejection(&mut self) {
        self.metrics.rejected_buffers = self.metrics.rejected_buffers.saturating_add(1);
    }

    pub(crate) fn try_retain_after_uncertain_release(
        &mut self,
        staging: PinnedUploadStaging,
    ) -> Result<(), (CudaError, PinnedUploadStaging)> {
        let Some(next_uncertain_bytes) = self.uncertain_bytes.checked_add(staging.len) else {
            self.accounting_poisoned = true;
            return Err((
                CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging byte accounting overflow",
                },
                staging,
            ));
        };
        let Some(next_uncertain_buffers) = self.uncertain.len().checked_add(1) else {
            self.accounting_poisoned = true;
            return Err((
                CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging count accounting overflow",
                },
                staging,
            ));
        };
        let Some(retained_buffers) = self
            .buffers
            .len()
            .checked_add(next_uncertain_buffers)
            .and_then(|count| count.checked_add(self.active_buffers))
        else {
            self.accounting_poisoned = true;
            return Err((
                CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging retained count accounting overflow",
                },
                staging,
            ));
        };
        let Some(retained_bytes) = self
            .cached_bytes
            .checked_add(next_uncertain_bytes)
            .and_then(|total| total.checked_add(self.active_bytes))
        else {
            self.accounting_poisoned = true;
            return Err((
                CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging retained byte accounting overflow",
                },
                staging,
            ));
        };
        if self.uncertain.try_reserve(1).is_err() {
            self.metrics.metadata_failures = self.metrics.metadata_failures.saturating_add(1);
            self.accounting_poisoned = true;
            let error = host_allocation_error::<PinnedUploadStaging>(
                self.uncertain.len().saturating_add(1),
            );
            return Err((error, staging));
        }
        self.uncertain.push(staging);
        self.uncertain_bytes = next_uncertain_bytes;
        self.observe_high_water(retained_buffers, retained_bytes);
        Ok(())
    }

    pub(crate) fn prepare_unwind_quarantine_slots(&mut self) -> Result<(), CudaError> {
        self.ensure_no_uncertain_release()?;
        let required_slots =
            self.active_buffers
                .checked_add(2)
                .ok_or(CudaError::InternalInvariant {
                    what: "CUDA pinned upload staging quarantine count overflow",
                })?;
        if self.uncertain.capacity() < required_slots
            && self.uncertain.try_reserve_exact(required_slots).is_err()
        {
            self.metrics.metadata_failures = self.metrics.metadata_failures.saturating_add(1);
            return Err(host_allocation_error::<PinnedUploadStaging>(required_slots));
        }
        Ok(())
    }

    pub(crate) fn drain_cached(&mut self) -> std::vec::Drain<'_, PinnedUploadStaging> {
        self.cached_bytes = 0;
        self.buffers.drain(..)
    }

    pub(crate) fn drain_uncertain(&mut self) -> std::vec::Drain<'_, PinnedUploadStaging> {
        self.uncertain_bytes = 0;
        self.uncertain.drain(..)
    }

    fn observe_high_water(&mut self, retained_buffers: usize, retained_bytes: usize) {
        self.metrics.peak_cached_buffers = self.metrics.peak_cached_buffers.max(self.buffers.len());
        self.metrics.peak_cached_bytes = self.metrics.peak_cached_bytes.max(self.cached_bytes);
        self.metrics.peak_uncertain_buffers = self
            .metrics
            .peak_uncertain_buffers
            .max(self.uncertain.len());
        self.metrics.peak_uncertain_bytes =
            self.metrics.peak_uncertain_bytes.max(self.uncertain_bytes);
        self.metrics.peak_active_buffers =
            self.metrics.peak_active_buffers.max(self.active_buffers);
        self.metrics.peak_active_bytes = self.metrics.peak_active_bytes.max(self.active_bytes);
        self.metrics.peak_retained_buffers =
            self.metrics.peak_retained_buffers.max(retained_buffers);
        self.metrics.peak_retained_bytes = self.metrics.peak_retained_bytes.max(retained_bytes);
    }

    fn ensure_no_uncertain_release(&self) -> Result<(), CudaError> {
        if self.accounting_poisoned {
            Err(CudaError::StatePoisoned {
                message: "page-locked upload staging accounting is poisoned".to_string(),
            })
        } else if self.uncertain.is_empty() {
            Ok(())
        } else {
            Err(CudaError::StatePoisoned {
                message:
                    "page-locked upload staging is quarantined after an uncertain CUDA release"
                        .to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests;
