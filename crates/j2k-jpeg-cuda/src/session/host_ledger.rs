// SPDX-License-Identifier: MIT OR Apache-2.0

//! Clone-shared accounting for host owners that outlive cache resolution.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex, MutexGuard,
};

use crate::Error;

pub(super) struct SharedHostLedger {
    active_bytes: AtomicUsize,
    peak_active_bytes: AtomicUsize,
    peak_combined_bytes: AtomicUsize,
    poisoned: AtomicBool,
    allocation_gate: Mutex<()>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[expect(
    clippy::struct_field_names,
    reason = "the byte-unit suffix distinguishes every ownership metric from entry counts"
)]
pub(super) struct HostLedgerDiagnostics {
    pub(super) active_bytes: usize,
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
            peak_active_bytes: AtomicUsize::new(0),
            peak_combined_bytes: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            allocation_gate: Mutex::new(()),
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
        Ok(HostLedgerDiagnostics {
            active_bytes: self.active_bytes.load(Ordering::Acquire),
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
        let mut active = self.active_bytes.load(Ordering::Acquire);
        loop {
            let next = active
                .checked_add(bytes)
                .ok_or_else(|| capacity_error(usize::MAX, cap, what))?;
            let aggregate = cache_retained_bytes
                .checked_add(next)
                .ok_or_else(|| capacity_error(usize::MAX, cap, what))?;
            if aggregate > cap {
                return Err(capacity_error(aggregate, cap, what));
            }
            match self.active_bytes.compare_exchange_weak(
                active,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak_active_bytes.fetch_max(next, Ordering::AcqRel);
                    self.peak_combined_bytes
                        .fetch_max(aggregate, Ordering::AcqRel);
                    return Ok(HostOwnerLease {
                        ledger: Arc::clone(self),
                        bytes,
                    });
                }
                Err(observed) => active = observed,
            }
        }
    }

    fn poison(&self) {
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
        let mut active = self.ledger.active_bytes.load(Ordering::Acquire);
        loop {
            let Some(next) = active.checked_sub(self.bytes) else {
                self.ledger.poison();
                return;
            };
            match self.ledger.active_bytes.compare_exchange_weak(
                active,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(observed) => active = observed,
            }
        }
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
mod tests {
    use super::SharedHostLedger;
    use crate::Error;
    use std::sync::atomic::Ordering;

    #[test]
    fn exact_reservation_releases_and_one_over_never_mutates() {
        let ledger = SharedHostLedger::new();
        let lease = ledger.reserve(7, 5, 12, "test ledger").unwrap();
        assert_eq!(ledger.active_bytes().unwrap(), 5);
        drop(lease);
        assert_eq!(ledger.active_bytes().unwrap(), 0);
        assert_eq!(ledger.diagnostics().unwrap().peak_active_bytes, 5);
        assert_eq!(ledger.diagnostics().unwrap().peak_combined_bytes, 12);
        assert!(matches!(
            ledger.reserve(7, 5, 11, "test ledger"),
            Err(Error::HostAllocationTooLarge {
                requested: 12,
                cap: 11,
                ..
            })
        ));
        assert_eq!(ledger.active_bytes().unwrap(), 0);
        assert_eq!(ledger.diagnostics().unwrap().peak_active_bytes, 5);
        assert_eq!(ledger.diagnostics().unwrap().peak_combined_bytes, 12);
    }

    #[test]
    fn impossible_release_poison_is_fail_closed_without_panicking() {
        let ledger = SharedHostLedger::new();
        let lease = ledger.reserve(0, 1, 1, "test ledger").unwrap();
        ledger.active_bytes.store(0, Ordering::Release);
        drop(lease);
        assert!(matches!(
            ledger.active_bytes(),
            Err(Error::InFlightHostLedgerPoisoned)
        ));
        assert!(matches!(
            ledger.reserve(0, 0, usize::MAX, "test ledger"),
            Err(Error::InFlightHostLedgerPoisoned)
        ));
    }
}
