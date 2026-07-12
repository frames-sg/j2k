// SPDX-License-Identifier: MIT OR Apache-2.0

//! Context-wide authority for simultaneously live host allocations.

use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use crate::error::CudaError;

#[derive(Default)]
struct HostBudgetState {
    external_bytes: usize,
    pinned_bytes: usize,
    provisional_bytes: usize,
    peak_bytes: usize,
    poisoned: bool,
}

pub(crate) struct SharedCudaHostBudget {
    state: Mutex<HostBudgetState>,
}

/// One mutable external-owner registration in a CUDA context's host budget.
///
/// Dropping the owner releases its exact current charge. Adapter-side owner
/// graphs should keep this value alive until every allocation represented by
/// the charge has been released.
#[doc(hidden)]
#[must_use = "dropping this owner releases its external host-memory charge"]
pub struct CudaExternalHostOwner {
    authority: Arc<SharedCudaHostBudget>,
    bytes: usize,
}

/// Full-headroom reservation for one external-owner replacement transaction.
#[doc(hidden)]
#[must_use = "the reserved context headroom is released when this transaction drops"]
pub struct CudaExternalHostReservation<'a> {
    owner: &'a mut CudaExternalHostOwner,
    authority: Arc<SharedCudaHostBudget>,
    external_live_bytes: usize,
    reserved_bytes: usize,
    finished: bool,
}

pub(crate) struct CudaHostBudgetGuard<'a> {
    state: MutexGuard<'a, HostBudgetState>,
}

impl SharedCudaHostBudget {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(HostBudgetState::default()),
        })
    }

    fn lock(&self) -> Result<CudaHostBudgetGuard<'_>, CudaError> {
        let state = self
            .state
            .lock()
            .map_err(|error| CudaError::StatePoisoned {
                message: error.to_string(),
            })?;
        if state.poisoned {
            return Err(CudaError::StatePoisoned {
                message: "CUDA context host-memory authority is poisoned".to_string(),
            });
        }
        Ok(CudaHostBudgetGuard { state })
    }

    pub(crate) fn register_external_owner(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<CudaExternalHostOwner, CudaError> {
        let mut guard = self.lock()?;
        guard.reconcile_external(0, bytes)?;
        Ok(CudaExternalHostOwner {
            authority: Arc::clone(self),
            bytes,
        })
    }

    pub(crate) fn reserve_pinned(&self, bytes: usize) -> Result<(), CudaError> {
        let mut guard = self.lock()?;
        guard.reserve_pinned(bytes)
    }

    pub(crate) fn release_pinned(&self, bytes: usize) -> Result<(), CudaError> {
        let mut guard = self.lock()?;
        guard.release_pinned(bytes)
    }

    pub(crate) fn pinned_bytes(&self) -> Result<usize, CudaError> {
        Ok(self.lock()?.state.pinned_bytes)
    }

    pub(crate) fn clear_pinned_after_context_drop(&self) {
        match self.state.lock() {
            Ok(mut state) => state.pinned_bytes = 0,
            Err(poisoned) => poisoned.into_inner().poisoned = true,
        }
    }

    pub(crate) fn poison(&self) {
        match self.state.lock() {
            Ok(mut state) => state.poisoned = true,
            Err(poisoned) => poisoned.into_inner().poisoned = true,
        }
    }

    #[cfg(test)]
    pub(crate) fn current_bytes(&self) -> Result<usize, CudaError> {
        let state = self.lock()?;
        state.state.combined()
    }
}

impl CudaHostBudgetGuard<'_> {
    fn reconcile_external(&mut self, old_bytes: usize, new_bytes: usize) -> Result<(), CudaError> {
        if self.state.external_bytes < old_bytes {
            self.poison();
            return Err(CudaError::InternalInvariant {
                what: "CUDA context external host-owner accounting underflow",
            });
        }
        let external = self
            .state
            .external_bytes
            .checked_sub(old_bytes)
            .and_then(|bytes| bytes.checked_add(new_bytes))
            .ok_or(CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external host owners",
            })?;
        let combined = external.checked_add(self.state.pinned_bytes).ok_or(
            CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external owners and pinned upload staging",
            },
        )?;
        let admitted = combined.checked_add(self.state.provisional_bytes).ok_or(
            CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external owners, pinned staging, and provisional headroom",
            },
        )?;
        if admitted > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
            return Err(CudaError::HostAllocationTooLarge {
                requested: admitted,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external owners and pinned upload staging",
            });
        }
        self.state.external_bytes = external;
        self.state.peak_bytes = self.state.peak_bytes.max(combined);
        Ok(())
    }

    fn reserve_pinned(&mut self, bytes: usize) -> Result<(), CudaError> {
        let pinned = self.state.pinned_bytes.checked_add(bytes).ok_or(
            CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA pinned upload staging",
            },
        )?;
        let actual_combined = self.state.external_bytes.checked_add(pinned).ok_or(
            CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external owners and pinned upload staging",
            },
        )?;
        let combined = actual_combined
            .checked_add(self.state.provisional_bytes)
            .ok_or(CudaError::HostAllocationTooLarge {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external owners, pinned staging, and provisional headroom",
            })?;
        if combined > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
            return Err(CudaError::HostAllocationTooLarge {
                requested: combined,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context external owners and pinned upload staging",
            });
        }
        self.state.pinned_bytes = pinned;
        self.state.peak_bytes = self.state.peak_bytes.max(actual_combined);
        Ok(())
    }

    fn release_pinned(&mut self, bytes: usize) -> Result<(), CudaError> {
        let Some(pinned) = self.state.pinned_bytes.checked_sub(bytes) else {
            self.poison();
            return Err(CudaError::InternalInvariant {
                what: "CUDA pinned upload host-authority accounting underflow",
            });
        };
        self.state.pinned_bytes = pinned;
        Ok(())
    }

    fn poison(&mut self) {
        self.state.poisoned = true;
    }
}

impl HostBudgetState {
    #[cfg(test)]
    fn combined(&self) -> Result<usize, CudaError> {
        self.external_bytes
            .checked_add(self.pinned_bytes)
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA context host-authority diagnostics overflow",
            })
    }
}

impl CudaExternalHostOwner {
    /// Whether this owner is registered with `context`'s exact authority.
    #[must_use]
    pub fn is_for_context(&self, context: &super::CudaContext) -> bool {
        Arc::ptr_eq(&self.authority, &context.inner.host_budget)
    }

    /// Current bytes registered by this external owner.
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Reconcile this registration transactionally to an exact new byte count.
    pub fn reconcile(&mut self, bytes: usize) -> Result<(), CudaError> {
        let authority = Arc::clone(&self.authority);
        let mut guard = authority.lock()?;
        guard.reconcile_external(self.bytes, bytes)?;
        self.bytes = bytes;
        Ok(())
    }

    /// Reserve all currently free context headroom for an external-owner
    /// replacement transaction.
    ///
    /// The authority mutex is released before caller allocation work. Competing
    /// session growth and pinned uploads observe the provisional reservation,
    /// so they cannot consume the headroom before commit.
    pub fn reserve_replacement(
        &mut self,
        excluded_bytes: usize,
    ) -> Result<CudaExternalHostReservation<'_>, CudaError> {
        if excluded_bytes > self.bytes {
            return Err(CudaError::InternalInvariant {
                what: "CUDA external owner exclusion exceeds its registered bytes",
            });
        }
        let authority = Arc::clone(&self.authority);
        let mut guard = authority.lock()?;
        let admitted = guard
            .state
            .external_bytes
            .checked_add(guard.state.pinned_bytes)
            .and_then(|bytes| bytes.checked_add(guard.state.provisional_bytes))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA context admitted host-owner accounting overflow",
            })?;
        let external_live_bytes =
            admitted
                .checked_sub(excluded_bytes)
                .ok_or(CudaError::InternalInvariant {
                    what: "CUDA context external-owner transaction exclusion underflow",
                })?;
        let reserved_bytes = DEFAULT_MAX_HOST_ALLOCATION_BYTES
            .checked_sub(admitted)
            .ok_or(CudaError::HostAllocationTooLarge {
                requested: admitted,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                what: "CUDA context admitted host owners",
            })?;
        guard.state.provisional_bytes = guard
            .state
            .provisional_bytes
            .checked_add(reserved_bytes)
            .ok_or(CudaError::InternalInvariant {
            what: "CUDA context provisional host reservation overflow",
        })?;
        drop(guard);
        Ok(CudaExternalHostReservation {
            owner: self,
            authority,
            external_live_bytes,
            reserved_bytes,
            finished: false,
        })
    }

    /// Context total excluding part of this owner's current charge.
    pub fn context_live_bytes_excluding(&self, excluded_bytes: usize) -> Result<usize, CudaError> {
        if excluded_bytes > self.bytes {
            return Err(CudaError::InternalInvariant {
                what: "CUDA external owner exclusion exceeds its registered bytes",
            });
        }
        let state = self.authority.lock()?;
        state
            .state
            .external_bytes
            .checked_add(state.state.pinned_bytes)
            .and_then(|bytes| bytes.checked_sub(excluded_bytes))
            .ok_or(CudaError::InternalInvariant {
                what: "CUDA context external-owner exclusion accounting failed",
            })
    }

    /// Exact total currently registered by this context.
    pub fn context_live_bytes(&self) -> Result<usize, CudaError> {
        self.context_live_bytes_excluding(0)
    }

    /// Exact page-locked staging bytes currently registered by this context.
    pub fn pinned_retained_bytes(&self) -> Result<usize, CudaError> {
        Ok(self.authority.lock()?.state.pinned_bytes)
    }
}

impl CudaExternalHostReservation<'_> {
    /// Context bytes the caller must count before its replacement allocation.
    #[must_use]
    pub const fn external_live_bytes(&self) -> usize {
        self.external_live_bytes
    }

    /// Commit the exact replacement charge and release provisional headroom.
    pub fn commit(mut self, new_bytes: usize) -> Result<(), CudaError> {
        let authority = Arc::clone(&self.authority);
        let mut guard = authority.lock()?;
        self.release_reservation_locked(&mut guard)?;
        guard.reconcile_external(self.owner.bytes, new_bytes)?;
        self.owner.bytes = new_bytes;
        self.finished = true;
        Ok(())
    }

    fn release_reservation_locked(
        &mut self,
        guard: &mut CudaHostBudgetGuard<'_>,
    ) -> Result<(), CudaError> {
        let Some(provisional) = guard
            .state
            .provisional_bytes
            .checked_sub(self.reserved_bytes)
        else {
            guard.poison();
            return Err(CudaError::InternalInvariant {
                what: "CUDA context provisional host reservation underflow",
            });
        };
        guard.state.provisional_bytes = provisional;
        self.finished = true;
        Ok(())
    }
}

impl Drop for CudaExternalHostReservation<'_> {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let authority = Arc::clone(&self.authority);
        match authority.lock() {
            Ok(mut guard) => {
                if self.release_reservation_locked(&mut guard).is_err() {
                    guard.poison();
                }
            }
            Err(_) => authority.poison(),
        };
    }
}

impl super::CudaContext {
    /// Register one exact external host-owner graph with this context.
    #[doc(hidden)]
    pub fn register_external_host_owner(
        &self,
        bytes: usize,
    ) -> Result<CudaExternalHostOwner, CudaError> {
        self.inner.host_budget.register_external_owner(bytes)
    }

    pub(crate) fn reserve_pinned_host_bytes(&self, bytes: usize) -> Result<(), CudaError> {
        self.inner.host_budget.reserve_pinned(bytes)
    }

    pub(crate) fn release_pinned_host_bytes(&self, bytes: usize) -> Result<(), CudaError> {
        self.inner.host_budget.release_pinned(bytes)
    }

    pub(crate) fn authority_pinned_host_bytes(&self) -> Result<usize, CudaError> {
        self.inner.host_budget.pinned_bytes()
    }

    pub(crate) fn poison_host_budget(&self) {
        self.inner.host_budget.poison();
    }
}

impl Drop for CudaExternalHostOwner {
    fn drop(&mut self) {
        let Ok(mut guard) = self.authority.lock() else {
            return;
        };
        if guard.reconcile_external(self.bytes, 0).is_err() {
            guard.poison();
        }
    }
}

#[cfg(test)]
mod tests;
