// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared checked host-allocation accounting for native encode phases.
//!
//! The ledger is concurrency-safe so later transform and Tier-1 lanes can
//! preclaim or allocate from the same encode domain before spawning workers.
//! Heap owners retain inseparable claims until the ledger is sealed for final
//! handoff; no allocation may begin after sealing.

use alloc::vec::Vec;
use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{EncodeError, EncodeResult, DEFAULT_MAX_CODEC_BYTES};

const SEALED_BIT: usize = 1usize << (usize::BITS - 1);
const LIVE_BYTES_MASK: usize = !SEALED_BIT;

/// Atomic accounting domain for simultaneously live native encode bytes.
#[derive(Debug)]
pub(crate) struct EncodeAllocationLedger {
    cap: usize,
    retained_bytes: usize,
    /// Live bytes and the sealed transition share one atomic state. This
    /// prevents a claim CAS from succeeding after a concurrent seal.
    allocation_state: AtomicUsize,
    peak_bytes: AtomicUsize,
    poisoned: AtomicBool,
}

impl EncodeAllocationLedger {
    pub(crate) fn new(retained_bytes: usize) -> EncodeResult<Self> {
        Self::with_cap(retained_bytes, DEFAULT_MAX_CODEC_BYTES)
    }

    #[cfg(test)]
    pub(crate) fn with_test_cap(retained_bytes: usize, cap: usize) -> EncodeResult<Self> {
        Self::with_cap(retained_bytes, cap)
    }

    pub(super) fn with_cap(retained_bytes: usize, cap: usize) -> EncodeResult<Self> {
        Self::with_labeled_cap(retained_bytes, cap, "retained native encode inputs")
    }

    pub(crate) fn with_phase_cap(
        retained_bytes: usize,
        cap: usize,
        what: &'static str,
    ) -> EncodeResult<Self> {
        Self::with_labeled_cap(retained_bytes, cap, what)
    }

    fn with_labeled_cap(
        retained_bytes: usize,
        cap: usize,
        what: &'static str,
    ) -> EncodeResult<Self> {
        if cap > LIVE_BYTES_MASK {
            return Err(EncodeError::InternalInvariant {
                what: "native encode allocation cap uses the sealed state bit",
            });
        }
        if retained_bytes > cap {
            return Err(EncodeError::AllocationTooLarge {
                what,
                requested: retained_bytes,
                cap,
            });
        }
        Ok(Self {
            cap,
            retained_bytes,
            allocation_state: AtomicUsize::new(retained_bytes),
            peak_bytes: AtomicUsize::new(retained_bytes),
            poisoned: AtomicBool::new(false),
        })
    }

    pub(crate) fn live_bytes(&self) -> usize {
        self.allocation_state.load(Ordering::Acquire) & LIVE_BYTES_MASK
    }

    #[cfg(test)]
    pub(crate) fn peak_bytes(&self) -> usize {
        self.peak_bytes.load(Ordering::Acquire)
    }

    pub(crate) fn claim(
        &self,
        bytes: usize,
        what: &'static str,
    ) -> EncodeResult<EncodeAllocationClaim<'_>> {
        self.add_live(bytes, what)?;
        Ok(EncodeAllocationClaim {
            ledger: self,
            bytes,
        })
    }

    pub(crate) fn try_vec_with_capacity<T>(
        &self,
        count: usize,
        what: &'static str,
    ) -> EncodeResult<BudgetedVec<'_, T>> {
        BudgetedVec::try_with_capacity(self, count, what)
    }

    /// Prevent every subsequent claim before tracked outputs are handed off.
    pub(crate) fn seal(&self) -> EncodeResult<()> {
        self.ensure_healthy("native encode allocation ledger was poisoned")?;
        let mut state = self.allocation_state.load(Ordering::Acquire);
        loop {
            if state & SEALED_BIT != 0 {
                return Err(EncodeError::InternalInvariant {
                    what: "native encode allocation ledger was sealed more than once",
                });
            }
            match self.allocation_state.compare_exchange_weak(
                state,
                state | SEALED_BIT,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => state = observed,
            }
        }
    }

    /// Verify that all scoped claims were released after sealed handoff.
    pub(crate) fn finalize(&self) -> EncodeResult<()> {
        self.ensure_healthy("native encode allocation ledger release underflowed")?;
        if self.allocation_state.load(Ordering::Acquire) & SEALED_BIT == 0 {
            return Err(EncodeError::InternalInvariant {
                what: "native encode allocation ledger finalized before sealing",
            });
        }
        if self.live_bytes() != self.retained_bytes {
            return Err(EncodeError::InternalInvariant {
                what: "native encode allocation claims remained live at final handoff",
            });
        }
        Ok(())
    }

    fn add_live(&self, bytes: usize, what: &'static str) -> EncodeResult<()> {
        self.ensure_healthy("native encode allocation ledger was poisoned")?;
        let mut state = self.allocation_state.load(Ordering::Acquire);
        self.add_live_from_observed_state(bytes, what, &mut state)
    }

    fn add_live_from_observed_state(
        &self,
        bytes: usize,
        what: &'static str,
        state: &mut usize,
    ) -> EncodeResult<()> {
        loop {
            if *state & SEALED_BIT != 0 {
                return Err(EncodeError::InternalInvariant {
                    what: "native encode allocation attempted after final handoff",
                });
            }
            let live = *state & LIVE_BYTES_MASK;
            let requested = live
                .checked_add(bytes)
                .ok_or(EncodeError::ArithmeticOverflow { what })?;
            if requested > self.cap {
                return Err(EncodeError::AllocationTooLarge {
                    what,
                    requested,
                    cap: self.cap,
                });
            }
            match self.allocation_state.compare_exchange_weak(
                *state,
                requested,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak_bytes.fetch_max(requested, Ordering::AcqRel);
                    return Ok(());
                }
                Err(observed) => *state = observed,
            }
        }
    }

    fn release(&self, bytes: usize) -> bool {
        let result =
            self.allocation_state
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                    let live = state & LIVE_BYTES_MASK;
                    let sealed = state & SEALED_BIT;
                    live.checked_sub(bytes)
                        .filter(|remaining| *remaining >= self.retained_bytes)
                        .map(|remaining| sealed | remaining)
                });
        if result.is_err() {
            self.poisoned.store(true, Ordering::Release);
            false
        } else {
            true
        }
    }

    fn ensure_healthy(&self, what: &'static str) -> EncodeResult<()> {
        if self.poisoned.load(Ordering::Acquire) {
            Err(EncodeError::InternalInvariant { what })
        } else {
            Ok(())
        }
    }

    fn ensure_handoff_ready(&self) -> EncodeResult<()> {
        self.ensure_healthy("native encode allocation ledger was poisoned")?;
        if self.allocation_state.load(Ordering::Acquire) & SEALED_BIT == 0 {
            return Err(EncodeError::InternalInvariant {
                what: "tracked allocation handed off before ledger sealing",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
impl EncodeAllocationLedger {
    fn claim_with_interleaving_barriers<'a>(
        &'a self,
        bytes: usize,
        what: &'static str,
        observed_unsealed_state: &std::sync::Barrier,
        resume_after_seal: &std::sync::Barrier,
    ) -> EncodeResult<EncodeAllocationClaim<'a>> {
        self.ensure_healthy("native encode allocation ledger was poisoned")?;
        let mut state = self.allocation_state.load(Ordering::Acquire);
        observed_unsealed_state.wait();
        resume_after_seal.wait();
        self.add_live_from_observed_state(bytes, what, &mut state)?;
        Ok(EncodeAllocationClaim {
            ledger: self,
            bytes,
        })
    }
}

/// Scoped claim released atomically when its owner is dropped.
#[derive(Debug)]
pub(crate) struct EncodeAllocationClaim<'a> {
    ledger: &'a EncodeAllocationLedger,
    bytes: usize,
}

impl EncodeAllocationClaim<'_> {
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn reconcile(
        &mut self,
        actual_bytes: usize,
        what: &'static str,
    ) -> EncodeResult<()> {
        if actual_bytes > self.bytes {
            self.ledger.add_live(actual_bytes - self.bytes, what)?;
        } else if !self.ledger.release(self.bytes - actual_bytes) {
            return Err(EncodeError::InternalInvariant {
                what: "native encode allocation claim reconciliation underflowed",
            });
        }
        self.bytes = actual_bytes;
        Ok(())
    }

    pub(crate) fn absorb(&mut self, mut other: Self) -> EncodeResult<()> {
        if !core::ptr::eq(self.ledger, other.ledger) {
            self.ledger.poisoned.store(true, Ordering::Release);
            other.ledger.poisoned.store(true, Ordering::Release);
            return Err(EncodeError::InternalInvariant {
                what: "allocation claims from different encode ledgers were combined",
            });
        }
        self.bytes =
            self.bytes
                .checked_add(other.bytes)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "combined native encode allocation claim",
                })?;
        other.bytes = 0;
        Ok(())
    }
}

impl Drop for EncodeAllocationClaim<'_> {
    fn drop(&mut self) {
        let _released = self.ledger.release(self.bytes);
    }
}

/// A vector whose allocator-returned capacity and ledger claim share a lifetime.
#[derive(Debug)]
pub(crate) struct BudgetedVec<'a, T> {
    values: Vec<T>,
    claim: EncodeAllocationClaim<'a>,
    what: &'static str,
}

impl<'a, T> BudgetedVec<'a, T> {
    pub(crate) fn try_with_capacity(
        ledger: &'a EncodeAllocationLedger,
        count: usize,
        what: &'static str,
    ) -> EncodeResult<Self> {
        let requested_bytes = checked_element_bytes::<T>(count, what)?;
        let mut claim = ledger.claim(requested_bytes, what)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, requested_bytes))?;
        claim.reconcile(checked_element_bytes::<T>(values.capacity(), what)?, what)?;
        Ok(Self {
            values,
            claim,
            what,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn capacity(&self) -> usize {
        self.values.capacity()
    }

    #[cfg(test)]
    pub(crate) fn allocation_bytes(&self) -> usize {
        self.claim.bytes()
    }

    pub(crate) fn try_push(&mut self, value: T) -> EncodeResult<()> {
        if self.values.len() == self.values.capacity() {
            return Err(EncodeError::InternalInvariant { what: self.what });
        }
        self.values.push(value);
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.values
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Move this vector's capacity claim into a longer-lived aggregate owner.
    pub(crate) fn transfer_to(
        self,
        aggregate: &mut EncodeAllocationClaim<'a>,
    ) -> EncodeResult<Vec<T>> {
        let Self {
            values,
            claim,
            what: _,
        } = self;
        aggregate.absorb(claim)?;
        Ok(values)
    }

    /// Release tracking only after the ledger has forbidden further claims.
    pub(crate) fn into_untracked(self) -> EncodeResult<Vec<T>> {
        self.claim.ledger.ensure_handoff_ready()?;
        let Self {
            values,
            claim,
            what: _,
        } = self;
        drop(claim);
        Ok(values)
    }
}

impl<T> Deref for BudgetedVec<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for BudgetedVec<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Clone> BudgetedVec<'_, T> {
    pub(crate) fn try_extend_from_slice(&mut self, values: &[T]) -> EncodeResult<()> {
        let new_len = self
            .values
            .len()
            .checked_add(values.len())
            .ok_or(EncodeError::ArithmeticOverflow { what: self.what })?;
        if new_len > self.values.capacity() {
            return Err(EncodeError::InternalInvariant { what: self.what });
        }
        self.values.extend_from_slice(values);
        Ok(())
    }
}

pub(crate) fn checked_add_bytes(
    left: usize,
    right: usize,
    what: &'static str,
) -> EncodeResult<usize> {
    left.checked_add(right)
        .ok_or(EncodeError::ArithmeticOverflow { what })
}

pub(crate) fn checked_mul_bytes(
    left: usize,
    right: usize,
    what: &'static str,
) -> EncodeResult<usize> {
    left.checked_mul(right)
        .ok_or(EncodeError::ArithmeticOverflow { what })
}

pub(crate) fn checked_element_bytes<T>(count: usize, what: &'static str) -> EncodeResult<usize> {
    checked_mul_bytes(count, size_of::<T>(), what)
}

pub(crate) fn host_allocation_failed(what: &'static str, bytes: usize) -> EncodeError {
    EncodeError::HostAllocationFailed { what, bytes }
}

/// Allocate an untracked vector fallibly after its caller has charged the
/// complete owner to a phase or concurrent-worker bound.
pub(crate) fn try_untracked_vec<T>(count: usize, what: &'static str) -> EncodeResult<Vec<T>> {
    let requested = checked_element_bytes::<T>(count, what)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| host_allocation_failed(what, requested))?;
    Ok(values)
}

pub(crate) fn try_untracked_vec_filled<T: Clone>(
    count: usize,
    value: T,
    what: &'static str,
) -> EncodeResult<Vec<T>> {
    let mut values = try_untracked_vec(count, what)?;
    values.resize(count, value);
    Ok(values)
}

pub(crate) fn try_reserve_untracked<T>(
    values: &mut Vec<T>,
    additional: usize,
    what: &'static str,
) -> EncodeResult<()> {
    if additional == 0 {
        return Ok(());
    }
    let requested_capacity = values
        .len()
        .checked_add(additional)
        .ok_or(EncodeError::ArithmeticOverflow { what })?;
    if requested_capacity <= values.capacity() {
        return Ok(());
    }
    let requested = checked_element_bytes::<T>(requested_capacity, what)?;
    values
        .try_reserve_exact(additional)
        .map_err(|_| host_allocation_failed(what, requested))
}

/// Grow an already-accounted vector geometrically without exceeding its
/// checked element limit. Byte-emitting hot paths use this instead of an exact
/// one-element reserve on every output byte.
pub(crate) fn try_reserve_untracked_bounded<T>(
    values: &mut Vec<T>,
    additional: usize,
    max_capacity: usize,
    what: &'static str,
) -> EncodeResult<()> {
    let required = values
        .len()
        .checked_add(additional)
        .ok_or(EncodeError::ArithmeticOverflow { what })?;
    if required > max_capacity {
        return Err(EncodeError::InternalInvariant { what });
    }
    if required <= values.capacity() {
        return Ok(());
    }

    let doubled = values.capacity().checked_mul(2).unwrap_or(max_capacity);
    let target = required
        .max(doubled)
        .max(64.min(max_capacity))
        .min(max_capacity);
    let reserve = target
        .checked_sub(values.len())
        .ok_or(EncodeError::InternalInvariant { what })?;
    let requested = checked_element_bytes::<T>(target, what)?;
    values
        .try_reserve_exact(reserve)
        .map_err(|_| host_allocation_failed(what, requested))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use std::sync::{Arc, Barrier};
    use std::thread;

    #[test]
    fn exact_cap_claim_is_accepted_and_released() {
        let ledger = EncodeAllocationLedger::with_test_cap(3, 8).expect("valid baseline");
        {
            let claim = ledger.claim(5, "test claim").expect("exact cap claim");
            assert_eq!(claim.bytes(), 5);
            assert_eq!(ledger.live_bytes(), 8);
            assert_eq!(ledger.peak_bytes(), 8);
        }
        assert_eq!(ledger.live_bytes(), 3);
    }

    #[test]
    fn one_byte_over_cap_is_rejected_without_changing_live_bytes() {
        let ledger = EncodeAllocationLedger::with_test_cap(3, 8).expect("valid baseline");
        let error = ledger
            .claim(6, "test claim")
            .expect_err("claim exceeds cap");
        assert_eq!(
            error,
            EncodeError::AllocationTooLarge {
                what: "test claim",
                requested: 9,
                cap: 8,
            }
        );
        assert_eq!(ledger.live_bytes(), 3);
    }

    #[test]
    fn element_byte_overflow_is_typed() {
        let error = checked_element_bytes::<u64>(usize::MAX, "test vector")
            .expect_err("element bytes overflow");
        assert_eq!(
            error,
            EncodeError::ArithmeticOverflow {
                what: "test vector"
            }
        );
    }

    #[test]
    fn allocator_failure_mapping_is_typed_and_source_specific() {
        assert_eq!(
            host_allocation_failed("packet headers", 42),
            EncodeError::HostAllocationFailed {
                what: "packet headers",
                bytes: 42,
            }
        );
    }

    #[test]
    fn bounded_untracked_growth_is_geometric_and_honors_exact_limit() {
        let mut values = try_untracked_vec::<u8>(2, "bounded test vector").expect("initial vector");
        values.extend_from_slice(&[1, 2]);
        try_reserve_untracked_bounded(&mut values, 1, 16, "bounded test vector")
            .expect("geometric growth");
        assert!(values.capacity() >= 4);
        assert!(values.capacity() <= 16);

        while values.len() < 16 {
            try_reserve_untracked_bounded(&mut values, 1, 16, "bounded test vector")
                .expect("growth through exact limit");
            values.push(0);
        }
        assert_eq!(values.len(), 16);
        assert!(values.capacity() <= 16);
        assert_eq!(
            try_reserve_untracked_bounded(&mut values, 1, 16, "bounded test vector")
                .expect_err("one element beyond limit is rejected"),
            EncodeError::InternalInvariant {
                what: "bounded test vector",
            }
        );
    }

    #[test]
    fn bounded_untracked_growth_reports_length_overflow() {
        let mut values = vec![0_u8];
        assert_eq!(
            try_reserve_untracked_bounded(
                &mut values,
                usize::MAX,
                usize::MAX,
                "overflowing bounded vector",
            )
            .expect_err("length overflow is typed"),
            EncodeError::ArithmeticOverflow {
                what: "overflowing bounded vector",
            }
        );
    }

    #[test]
    fn tracked_vector_requires_sealed_handoff_and_finalizes_cleanly() {
        let ledger = EncodeAllocationLedger::with_test_cap(0, 1024).expect("valid cap");
        let mut values = ledger
            .try_vec_with_capacity::<u32>(7, "test vector capacity exhausted")
            .expect("fallible vector allocation");
        values.try_push(9).expect("planned push");
        assert_eq!(ledger.live_bytes(), values.allocation_bytes());
        ledger.seal().expect("seal ledger");
        let values = values.into_untracked().expect("sealed handoff");
        assert_eq!(values, [9]);
        ledger.finalize().expect("all claims released");
    }

    #[test]
    fn ledger_is_safe_to_share_across_preclaimed_workers() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EncodeAllocationLedger>();
    }

    #[test]
    fn sealed_state_rejects_every_later_claim_without_changing_live_bytes() {
        let ledger = EncodeAllocationLedger::with_test_cap(3, 8).expect("valid baseline");
        ledger.seal().expect("seal ledger");
        let error = ledger
            .claim(1, "late test claim")
            .expect_err("sealed ledger rejects claims");
        assert_eq!(
            error,
            EncodeError::InternalInvariant {
                what: "native encode allocation attempted after final handoff",
            }
        );
        assert_eq!(ledger.live_bytes(), 3);
        ledger.finalize().expect("sealed baseline is finalized");
    }

    #[test]
    fn joined_workers_reject_concurrent_calls_made_after_seal() {
        const WORKERS: usize = 8;
        let ledger =
            Arc::new(EncodeAllocationLedger::with_test_cap(3, 64).expect("valid shared ledger"));
        let ready = Arc::new(Barrier::new(WORKERS + 1));
        let claim_after_seal = Arc::new(Barrier::new(WORKERS + 1));
        let mut workers = Vec::with_capacity(WORKERS);
        for _ in 0..WORKERS {
            let ledger = Arc::clone(&ledger);
            let ready = Arc::clone(&ready);
            let claim_after_seal = Arc::clone(&claim_after_seal);
            workers.push(thread::spawn(move || {
                ready.wait();
                claim_after_seal.wait();
                ledger.claim(1, "joined worker claim").is_ok()
            }));
        }

        ready.wait();
        ledger.seal().expect("seal after every worker is ready");
        claim_after_seal.wait();
        for worker in workers {
            assert!(!worker.join().expect("worker joins"));
        }
        assert_eq!(ledger.live_bytes(), 3);
        ledger.finalize().expect("joined releases restore baseline");
    }

    #[test]
    fn stale_preseal_observation_cannot_commit_after_concurrent_seal() {
        let ledger =
            Arc::new(EncodeAllocationLedger::with_test_cap(3, 64).expect("valid shared ledger"));
        let observed = Arc::new(Barrier::new(2));
        let resume = Arc::new(Barrier::new(2));
        let worker = {
            let ledger = Arc::clone(&ledger);
            let observed = Arc::clone(&observed);
            let resume = Arc::clone(&resume);
            thread::spawn(move || {
                ledger
                    .claim_with_interleaving_barriers(1, "stale observed claim", &observed, &resume)
                    .is_ok()
            })
        };

        // The worker has loaded the unsealed combined state but has not CASed.
        observed.wait();
        ledger.seal().expect("seal while claim is paused");
        resume.wait();
        assert!(!worker.join().expect("interleaved worker joins"));
        assert_eq!(ledger.live_bytes(), 3);
        ledger
            .finalize()
            .expect("stale claim did not alter baseline");
    }
}
