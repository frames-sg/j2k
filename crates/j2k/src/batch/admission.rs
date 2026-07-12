// SPDX-License-Identifier: MIT OR Apache-2.0

//! Weighted runtime admission for direct plans and generic fallbacks.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use j2k_core::BatchInfrastructureError;

use super::allocation::J2K_BATCH_EXECUTION_CAP_BYTES;

pub(super) struct BatchAllocationBudget {
    cap: usize,
    state: Mutex<AdmissionState>,
    changed: Condvar,
    #[cfg(test)]
    waiter_changed: Condvar,
}

struct AdmissionState {
    live: usize,
    waiters: usize,
}

impl BatchAllocationBudget {
    pub(super) fn with_baseline(baseline: usize) -> Result<Arc<Self>, BatchInfrastructureError> {
        if baseline > J2K_BATCH_EXECUTION_CAP_BYTES {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: "shared J2K direct plan",
                requested: baseline,
                cap: J2K_BATCH_EXECUTION_CAP_BYTES,
            });
        }
        Ok(Arc::new(Self {
            cap: J2K_BATCH_EXECUTION_CAP_BYTES,
            state: Mutex::new(AdmissionState {
                live: baseline,
                waiters: 0,
            }),
            changed: Condvar::new(),
            #[cfg(test)]
            waiter_changed: Condvar::new(),
        }))
    }

    pub(super) fn claim(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<BatchAllocationClaim, BatchInfrastructureError> {
        if bytes > self.cap {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: "one J2K batch execution claim",
                requested: bytes,
                cap: self.cap,
            });
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        while self.cap.saturating_sub(state.live) < bytes {
            state.waiters = state.waiters.saturating_add(1);
            #[cfg(test)]
            self.waiter_changed.notify_all();
            state = match self.changed.wait(state) {
                Ok(mut state) => {
                    state.waiters = state.waiters.saturating_sub(1);
                    state
                }
                Err(poisoned) => {
                    let mut state = poisoned.into_inner();
                    state.waiters = state.waiters.saturating_sub(1);
                    return Err(BatchInfrastructureError::SchedulerPoisoned);
                }
            };
        }
        state.live =
            state
                .live
                .checked_add(bytes)
                .ok_or(BatchInfrastructureError::AllocationTooLarge {
                    what: "J2K batch execution claims",
                    requested: usize::MAX,
                    cap: self.cap,
                })?;
        drop(state);
        Ok(BatchAllocationClaim {
            budget: Arc::clone(self),
            bytes,
        })
    }

    fn lock_state(&self) -> MutexGuard<'_, AdmissionState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    fn wait_until_waiting(&self) {
        let mut state = self.lock_state();
        while state.waiters == 0 {
            state = self
                .waiter_changed
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        drop(state);
    }
}

pub(super) struct BatchAllocationClaim {
    budget: Arc<BatchAllocationBudget>,
    bytes: usize,
}

impl BatchAllocationClaim {
    pub(super) fn reconcile(&mut self, bytes: usize) -> Result<(), BatchInfrastructureError> {
        if bytes > self.bytes {
            return Err(BatchInfrastructureError::AllocationTooLarge {
                what: "reconciled J2K batch execution claim",
                requested: bytes,
                cap: self.bytes,
            });
        }
        let released = self.bytes - bytes;
        let mut state = self
            .budget
            .state
            .lock()
            .map_err(|_| BatchInfrastructureError::SchedulerPoisoned)?;
        state.live = state
            .live
            .checked_sub(released)
            .ok_or(BatchInfrastructureError::SchedulerPoisoned)?;
        self.bytes = bytes;
        drop(state);
        if released != 0 {
            self.budget.changed.notify_all();
        }
        Ok(())
    }
}

impl Drop for BatchAllocationClaim {
    fn drop(&mut self) {
        let mut state = self.budget.lock_state();
        state.live = state.live.saturating_sub(self.bytes);
        drop(state);
        self.budget.changed.notify_all();
    }
}

#[cfg(test)]
mod tests;
