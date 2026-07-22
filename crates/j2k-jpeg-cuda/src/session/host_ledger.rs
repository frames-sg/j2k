// SPDX-License-Identifier: MIT OR Apache-2.0

//! Clone-shared accounting for host owners that outlive cache resolution.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard,
};

use crate::Error;

pub(super) enum HostCacheTransactionError<E> {
    Operation(E),
    Accounting(Error),
    OperationAndAccounting { operation: E, accounting: Error },
}

pub(super) struct SharedHostLedger {
    active_bytes: AtomicUsize,
    cache_retained_bytes: AtomicUsize,
    peak_active_bytes: AtomicUsize,
    peak_combined_bytes: AtomicUsize,
    poisoned: AtomicBool,
    allocation_gate: Mutex<()>,
    context_owner: Mutex<Option<j2k_cuda_runtime::CudaExternalHostOwner>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the byte-unit suffix distinguishes every ownership metric from entry counts"
)]
pub(super) struct HostLedgerDiagnostics {
    pub(super) active_bytes: usize,
    pub(super) pinned_retained_bytes: usize,
    pub(super) peak_active_bytes: usize,
    pub(super) peak_combined_bytes: usize,
}

#[must_use = "dropping this lease immediately removes the owner from session accounting"]
pub(crate) struct HostOwnerLease {
    ledger: Arc<SharedHostLedger>,
    bytes: usize,
}

impl SharedHostLedger {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            active_bytes: AtomicUsize::new(0),
            cache_retained_bytes: AtomicUsize::new(0),
            peak_active_bytes: AtomicUsize::new(0),
            peak_combined_bytes: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            allocation_gate: Mutex::new(()),
            context_owner: Mutex::new(None),
        })
    }

    pub(super) fn lock_allocations(&self) -> Result<MutexGuard<'_, ()>, Error> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::InFlightHostLedgerPoisoned);
        }
        self.allocation_gate.lock().map_err(|_| {
            self.poison();
            Error::InFlightHostLedgerPoisoned
        })
    }

    pub(super) fn active_bytes(&self) -> Result<usize, Error> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::InFlightHostLedgerPoisoned);
        }
        Ok(self.active_bytes.load(Ordering::Acquire))
    }

    pub(super) fn diagnostics(&self) -> Result<HostLedgerDiagnostics, Error> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::InFlightHostLedgerPoisoned);
        }
        let pinned_retained_bytes = self.pinned_retained_bytes()?;
        Ok(HostLedgerDiagnostics {
            active_bytes: self.active_bytes.load(Ordering::Acquire),
            pinned_retained_bytes,
            peak_active_bytes: self.peak_active_bytes.load(Ordering::Acquire),
            peak_combined_bytes: self.peak_combined_bytes.load(Ordering::Acquire),
        })
    }

    pub(super) fn observe_combined(&self, combined_bytes: usize) -> Result<(), Error> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::InFlightHostLedgerPoisoned);
        }
        self.peak_combined_bytes
            .fetch_max(combined_bytes, Ordering::AcqRel);
        Ok(())
    }

    pub(super) fn combined_bytes(&self, cache_retained_bytes: usize) -> Result<usize, Error> {
        let owner = self.context_owner()?;
        if let Some(owner) = owner.as_ref() {
            return owner
                .context_live_bytes()
                .map_err(crate::runtime::cuda_error);
        }
        cache_retained_bytes
            .checked_add(self.active_bytes()?)
            .ok_or_else(|| {
                capacity_error(
                    usize::MAX,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    "CUDA JPEG cache and in-flight host owners",
                )
            })
    }

    pub(super) fn combined_bytes_without_pinned(
        &self,
        cache_retained_bytes: usize,
    ) -> Result<usize, Error> {
        let owner = self.context_owner()?;
        if let Some(owner) = owner.as_ref() {
            return owner
                .context_live_bytes()
                .and_then(|bytes| {
                    owner.pinned_retained_bytes().and_then(|pinned| {
                        bytes.checked_sub(pinned).ok_or(
                            j2k_cuda_runtime::CudaError::InternalInvariant {
                                what: "CUDA JPEG pinned exclusion underflow",
                            },
                        )
                    })
                })
                .map_err(crate::runtime::cuda_error);
        }
        cache_retained_bytes
            .checked_add(self.active_bytes()?)
            .ok_or_else(|| {
                capacity_error(
                    usize::MAX,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    "CUDA JPEG cache and in-flight host owners",
                )
            })
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        cache_retained_bytes: usize,
        bytes: usize,
        cap: usize,
        what: &'static str,
    ) -> Result<HostOwnerLease, Error> {
        if self.poisoned.load(Ordering::Acquire) {
            return Err(Error::InFlightHostLedgerPoisoned);
        }
        let active = self.active_bytes.load(Ordering::Acquire);
        let next = active
            .checked_add(bytes)
            .ok_or_else(|| capacity_error(usize::MAX, cap, what))?;
        let pinned_retained = self.pinned_retained_bytes()?;
        let aggregate = cache_retained_bytes
            .checked_add(pinned_retained)
            .and_then(|total| total.checked_add(next))
            .ok_or_else(|| capacity_error(usize::MAX, cap, what))?;
        if aggregate > cap {
            return Err(capacity_error(aggregate, cap, what));
        }
        self.reconcile_context_external(cache_retained_bytes, next)?;
        self.cache_retained_bytes
            .store(cache_retained_bytes, Ordering::Release);
        self.active_bytes.store(next, Ordering::Release);
        self.peak_active_bytes.fetch_max(next, Ordering::AcqRel);
        self.peak_combined_bytes
            .fetch_max(aggregate, Ordering::AcqRel);
        Ok(HostOwnerLease {
            ledger: Arc::clone(self),
            bytes,
        })
    }

    pub(super) fn allocate_host_owner<T>(
        self: &Arc<Self>,
        cache_retained_bytes: usize,
        allocate: impl FnOnce(usize) -> Result<(T, usize), Error>,
    ) -> Result<(T, HostOwnerLease), Error> {
        let active = self.active_bytes.load(Ordering::Acquire);
        let registered_before = cache_retained_bytes.checked_add(active).ok_or_else(|| {
            capacity_error(
                usize::MAX,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                "CUDA JPEG cache and in-flight host owners",
            )
        })?;
        let mut context_owner = self.context_owner()?;
        let Some(context_owner) = context_owner.as_mut() else {
            drop(context_owner);
            let external_live = self.combined_bytes(cache_retained_bytes)?;
            let (owner, actual_bytes) = allocate(external_live)?;
            let lease = self.reserve(
                cache_retained_bytes,
                actual_bytes,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                "CUDA JPEG cache, pinned staging, and in-flight host owners",
            )?;
            return Ok((owner, lease));
        };
        let reservation = context_owner
            .reserve_replacement(0)
            .map_err(crate::runtime::cuda_error)?;
        let external_live = reservation.external_live_bytes();
        let (result, registered_after, actual_bytes) = match allocate(external_live) {
            Ok((owner, actual)) => {
                let Some(registered) = registered_before.checked_add(actual) else {
                    return match reservation.commit(registered_before) {
                        Ok(()) => Err(capacity_error(
                            usize::MAX,
                            j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                            "CUDA JPEG cache and in-flight host owners",
                        )),
                        Err(accounting) => Err(Error::OperationAndHostAccountingFailed {
                            primary: Box::new(capacity_error(
                                usize::MAX,
                                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                                "CUDA JPEG cache and in-flight host owners",
                            )),
                            accounting: Box::new(crate::runtime::cuda_error(accounting)),
                        }),
                    };
                };
                (Ok(owner), registered, actual)
            }
            Err(error) => (Err(error), registered_before, 0),
        };
        let accounting = reservation
            .commit(registered_after)
            .map_err(crate::runtime::cuda_error);
        let owner = match (result, accounting) {
            (Ok(owner), Ok(())) => owner,
            (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
            (Err(primary), Err(accounting)) => {
                return Err(Error::OperationAndHostAccountingFailed {
                    primary: Box::new(primary),
                    accounting: Box::new(accounting),
                });
            }
        };
        let next_active = active.checked_add(actual_bytes).ok_or_else(|| {
            self.poison();
            Error::InFlightHostLedgerPoisoned
        })?;
        self.cache_retained_bytes
            .store(cache_retained_bytes, Ordering::Release);
        self.active_bytes.store(next_active, Ordering::Release);
        self.peak_active_bytes
            .fetch_max(next_active, Ordering::AcqRel);
        let combined = cache_retained_bytes
            .checked_add(next_active)
            .and_then(|bytes| bytes.checked_add(context_owner.pinned_retained_bytes().ok()?));
        if let Some(combined) = combined {
            self.peak_combined_bytes
                .fetch_max(combined, Ordering::AcqRel);
        }
        Ok((
            owner,
            HostOwnerLease {
                ledger: Arc::clone(self),
                bytes: actual_bytes,
            },
        ))
    }

    pub(super) fn bind_context(
        &self,
        context: &j2k_cuda_runtime::CudaContext,
    ) -> Result<(), Error> {
        let _allocation = self.lock_allocations()?;
        let mut owner = self.context_owner()?;
        if let Some(owner) = owner.as_ref() {
            if owner.is_for_context(context) {
                return Ok(());
            }
            return Err(Error::UnsupportedCudaRequest {
                reason: "CUDA JPEG host-owner ledger is already bound to another CUDA context",
            });
        }
        let bytes = self
            .cache_retained_bytes
            .load(Ordering::Acquire)
            .checked_add(self.active_bytes.load(Ordering::Acquire))
            .ok_or_else(|| {
                capacity_error(
                    usize::MAX,
                    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    "CUDA JPEG cache and in-flight host owners",
                )
            })?;
        *owner = Some(
            context
                .register_external_host_owner(bytes)
                .map_err(crate::runtime::cuda_error)?,
        );
        Ok(())
    }

    pub(super) fn ensure_context(
        &self,
        context: &j2k_cuda_runtime::CudaContext,
    ) -> Result<(), Error> {
        let owner = self.context_owner()?;
        match owner.as_ref() {
            Some(owner) if owner.is_for_context(context) => Ok(()),
            _ => Err(Error::UnsupportedCudaRequest {
                reason: "CUDA pinned-upload operation does not belong to this session context",
            }),
        }
    }

    pub(super) fn transact_cache<T, E>(
        &self,
        cache_retained_bytes: usize,
        operation: impl FnOnce(usize) -> (Result<T, E>, usize),
    ) -> Result<T, HostCacheTransactionError<E>> {
        let mut owner = self
            .context_owner()
            .map_err(HostCacheTransactionError::Accounting)?;
        if let Some(owner) = owner.as_mut() {
            let reservation = owner
                .reserve_replacement(cache_retained_bytes)
                .map_err(|error| {
                    HostCacheTransactionError::Accounting(crate::runtime::cuda_error(error))
                })?;
            let (result, retained_bytes) = operation(reservation.external_live_bytes());
            let active = self.active_bytes.load(Ordering::Acquire);
            let registered = match retained_bytes.checked_add(active) {
                Some(bytes) => bytes,
                None => usize::MAX,
            };
            let accounting = reservation
                .commit(registered)
                .map_err(crate::runtime::cuda_error);
            self.cache_retained_bytes
                .store(retained_bytes, Ordering::Release);
            match (result, accounting) {
                (Ok(value), Ok(())) => Ok(value),
                (Err(error), Ok(())) => Err(HostCacheTransactionError::Operation(error)),
                (Ok(_), Err(error)) => Err(HostCacheTransactionError::Accounting(error)),
                (Err(operation), Err(accounting)) => {
                    Err(HostCacheTransactionError::OperationAndAccounting {
                        operation,
                        accounting,
                    })
                }
            }
        } else {
            let active = self.active_bytes.load(Ordering::Acquire);
            let (result, retained_bytes) = operation(active);
            match result {
                Ok(value) => {
                    self.cache_retained_bytes
                        .store(retained_bytes, Ordering::Release);
                    Ok(value)
                }
                Err(error) => {
                    self.cache_retained_bytes
                        .store(retained_bytes, Ordering::Release);
                    Err(HostCacheTransactionError::Operation(error))
                }
            }
        }
    }

    fn reconcile_context_external(
        &self,
        cache_bytes: usize,
        active_bytes: usize,
    ) -> Result<(), Error> {
        let bytes = cache_bytes.checked_add(active_bytes).ok_or_else(|| {
            capacity_error(
                usize::MAX,
                j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                "CUDA JPEG cache and in-flight host owners",
            )
        })?;
        let mut owner = self.context_owner()?;
        if let Some(owner) = owner.as_mut() {
            owner.reconcile(bytes).map_err(crate::runtime::cuda_error)?;
        }
        Ok(())
    }

    fn pinned_retained_bytes(&self) -> Result<usize, Error> {
        let owner = self.context_owner()?;
        owner.as_ref().map_or(Ok(0), |owner| {
            owner
                .pinned_retained_bytes()
                .map_err(crate::runtime::cuda_error)
        })
    }

    fn context_owner(
        &self,
    ) -> Result<MutexGuard<'_, Option<j2k_cuda_runtime::CudaExternalHostOwner>>, Error> {
        self.context_owner.lock().map_err(|_| {
            self.poisoned.store(true, Ordering::Release);
            Error::InFlightHostLedgerPoisoned
        })
    }

    pub(super) fn poison(&self) {
        self.poisoned.store(true, Ordering::Release);
        self.active_bytes.store(usize::MAX, Ordering::Release);
    }
}

impl HostOwnerLease {
    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }
}

impl core::fmt::Debug for HostOwnerLease {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("HostOwnerLease")
            .field("bytes", &self.bytes())
            .finish_non_exhaustive()
    }
}

impl Drop for HostOwnerLease {
    fn drop(&mut self) {
        if self.ledger.poisoned.load(Ordering::Acquire) {
            return;
        }
        let Ok(_allocation) = self.ledger.lock_allocations() else {
            return;
        };
        let active = self.ledger.active_bytes.load(Ordering::Acquire);
        let Some(next) = active.checked_sub(self.bytes) else {
            self.ledger.poison();
            return;
        };
        let cache = self.ledger.cache_retained_bytes.load(Ordering::Acquire);
        if self.ledger.reconcile_context_external(cache, next).is_err() {
            self.ledger.poison();
            return;
        }
        self.ledger.active_bytes.store(next, Ordering::Release);
    }
}

fn capacity_error(requested: usize, cap: usize, what: &'static str) -> Error {
    Error::HostAllocationTooLarge {
        requested,
        cap,
        what,
    }
}

#[cfg(test)]
mod tests;
