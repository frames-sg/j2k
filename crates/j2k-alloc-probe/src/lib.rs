// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process-wide allocation measurement for serial codec regression probes.
//!
//! The meter is process-global because codec work may allocate on Rayon worker
//! threads. A measurement must join all work it starts before returning.
//! Measurements are serialized and cannot be nested. Epoch checks prevent a
//! delayed allocator call from reporting into a later measurement, and each
//! measurement waits for already-admitted reporters before taking its
//! snapshot.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[cfg(test)]
use std::sync::atomic::AtomicBool;

const IDLE: u8 = 0;
const ARMING: u8 = 1;
const ACTIVE: u8 = 2;

#[cfg(test)]
const TEST_HOOK_IDLE: u8 = 0;
#[cfg(test)]
const TEST_HOOK_ARMED: u8 = 1;
#[cfg(test)]
const TEST_HOOK_PAUSED: u8 = 2;

static STATE: AtomicU8 = AtomicU8::new(IDLE);
static EPOCH: AtomicU64 = AtomicU64::new(0);
static ACTIVE_REPORTERS: AtomicU64 = AtomicU64::new(0);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DEALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REQUESTED_BYTES: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
static TEST_HOOK: AtomicU8 = AtomicU8::new(TEST_HOOK_IDLE);
#[cfg(test)]
static TEST_HOOK_RELEASE: AtomicBool = AtomicBool::new(false);
#[cfg(test)]
static TEST_OBSERVED_EPOCH: AtomicU64 = AtomicU64::new(u64::MAX);
#[cfg(test)]
static TEST_RECHECKED_EPOCH: AtomicU64 = AtomicU64::new(u64::MAX);
#[cfg(test)]
static TEST_HOOK_RECORDED: AtomicBool = AtomicBool::new(false);

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation delegates to `System` with the exact pointer, layout,
// and replacement size supplied by the caller. Atomic bookkeeping neither
// dereferences nor changes allocation pointers.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let reporter = Reporter::begin();
        // SAFETY: forwards the caller's valid allocation layout unchanged.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            reporter.record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let reporter = Reporter::begin();
        // SAFETY: forwards the caller's valid allocation layout unchanged.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            reporter.record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let reporter = Reporter::begin();
        reporter.record_deallocation();
        // SAFETY: forwards the pointer and its original layout unchanged.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let reporter = Reporter::begin();
        // SAFETY: forwards the pointer, original layout, and requested
        // replacement size unchanged.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() {
            reporter.record_reallocation(new_size);
        }
        replacement
    }
}

/// Successful allocator activity observed during one measurement window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationStats {
    /// Successful allocation calls, including zeroed allocations.
    pub allocations: u64,
    /// Successful reallocation calls.
    pub reallocations: u64,
    /// Deallocation calls observed during the window.
    ///
    /// This count is diagnostic only: it may include allocations created
    /// before the window and never reduces a budget.
    pub deallocations: u64,
    /// Sum of requested bytes for successful allocations and reallocations.
    ///
    /// Reallocations contribute their full new requested size. Deallocations
    /// never reduce this value, so freeing pre-existing storage cannot conceal
    /// later allocation work.
    pub requested_bytes: u64,
}

impl AllocationStats {
    /// Sum of successful allocation and reallocation calls.
    #[must_use]
    pub const fn allocation_calls(self) -> u64 {
        self.allocations.saturating_add(self.reallocations)
    }
}

/// Upper bounds for successful allocation work in one measured operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    allocation_calls: u64,
    requested_bytes: u64,
}

impl Budget {
    /// Require no successful allocation or reallocation calls.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            allocation_calls: 0,
            requested_bytes: 0,
        }
    }

    /// Bound total requested bytes while allowing any number of calls.
    #[must_use]
    pub const fn total_bytes(max_requested_bytes: u64) -> Self {
        Self {
            allocation_calls: u64::MAX,
            requested_bytes: max_requested_bytes,
        }
    }

    /// Also bound the sum of successful allocation and reallocation calls.
    #[must_use]
    pub const fn with_max_calls(mut self, max_allocation_calls: u64) -> Self {
        self.allocation_calls = max_allocation_calls;
        self
    }

    const fn accepts(self, stats: AllocationStats) -> bool {
        stats.allocation_calls() <= self.allocation_calls
            && stats.requested_bytes <= self.requested_bytes
    }
}

/// Measure one operation with process-global counters.
///
/// The operation must join every worker task it starts before returning.
/// Activity from unrelated threads in the same process is intentionally part
/// of the measurement.
///
/// # Panics
///
/// Panics if another measurement is active or being armed.
pub fn measure<R>(operation: impl FnOnce() -> R) -> (R, AllocationStats) {
    let guard = MeasurementGuard::begin();
    let result = operation();
    let stats = guard.finish();
    (result, stats)
}

/// Run an operation and panic after metering is disabled if it exceeds a
/// budget.
///
/// # Panics
///
/// Panics if the budget is exceeded or another measurement is active.
pub fn assert_allocations<R>(label: &str, budget: Budget, operation: impl FnOnce() -> R) -> R {
    let (result, stats) = measure(operation);
    assert!(
        budget.accepts(stats),
        "{label} exceeded allocation budget: stats={stats:?}, budget={budget:?}"
    );
    result
}

struct MeasurementGuard {
    finished: bool,
}

impl MeasurementGuard {
    fn begin() -> Self {
        if STATE
            .compare_exchange(IDLE, ARMING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            panic!("allocation measurements must be serial and non-nested");
        }
        reset_counters();
        EPOCH.fetch_add(1, Ordering::AcqRel);
        STATE.store(ACTIVE, Ordering::Release);
        Self { finished: false }
    }

    fn finish(mut self) -> AllocationStats {
        STATE.store(ARMING, Ordering::Release);
        wait_for_active_reporters();
        let stats = snapshot();
        STATE.store(IDLE, Ordering::Release);
        self.finished = true;
        stats
    }
}

impl Drop for MeasurementGuard {
    fn drop(&mut self) {
        if !self.finished {
            STATE.store(ARMING, Ordering::Release);
            wait_for_active_reporters();
            STATE.store(IDLE, Ordering::Release);
        }
    }
}

struct Reporter {
    active: bool,
    #[cfg(test)]
    test_hooked: bool,
}

impl Reporter {
    fn begin() -> Self {
        let observed_epoch = EPOCH.load(Ordering::Acquire);
        let observed_active = STATE.load(Ordering::Acquire) == ACTIVE;

        let test_hooked = pause_test_reporter_after_observation(observed_active);
        #[cfg(not(test))]
        let _ = test_hooked;

        if !observed_active {
            return Self {
                active: false,
                #[cfg(test)]
                test_hooked: false,
            };
        }

        ACTIVE_REPORTERS.fetch_add(1, Ordering::AcqRel);
        let rechecked_epoch = EPOCH.load(Ordering::Acquire);
        let active = STATE.load(Ordering::Acquire) == ACTIVE && rechecked_epoch == observed_epoch;
        #[cfg(test)]
        if test_hooked {
            TEST_OBSERVED_EPOCH.store(observed_epoch, Ordering::Release);
            TEST_RECHECKED_EPOCH.store(rechecked_epoch, Ordering::Release);
        }
        if !active {
            ACTIVE_REPORTERS.fetch_sub(1, Ordering::Release);
        }
        Self {
            active,
            #[cfg(test)]
            test_hooked,
        }
    }

    fn record_allocation(&self, size: usize) {
        if self.active {
            #[cfg(test)]
            self.mark_test_hook_recorded();
            saturating_increment(&ALLOCATIONS);
            saturating_add(&REQUESTED_BYTES, size_as_u64(size));
        }
    }

    fn record_reallocation(&self, new_size: usize) {
        if self.active {
            #[cfg(test)]
            self.mark_test_hook_recorded();
            saturating_increment(&REALLOCATIONS);
            saturating_add(&REQUESTED_BYTES, size_as_u64(new_size));
        }
    }

    fn record_deallocation(&self) {
        if self.active {
            #[cfg(test)]
            self.mark_test_hook_recorded();
            saturating_increment(&DEALLOCATIONS);
        }
    }

    #[cfg(test)]
    fn mark_test_hook_recorded(&self) {
        if self.test_hooked {
            TEST_HOOK_RECORDED.store(true, Ordering::Release);
        }
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        if self.active {
            ACTIVE_REPORTERS.fetch_sub(1, Ordering::Release);
        }
    }
}

fn pause_test_reporter_after_observation(observed_active: bool) -> bool {
    #[cfg(test)]
    if observed_active
        && TEST_HOOK
            .compare_exchange(
                TEST_HOOK_ARMED,
                TEST_HOOK_PAUSED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    {
        while !TEST_HOOK_RELEASE.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        return true;
    }
    #[cfg(not(test))]
    let _ = observed_active;
    false
}

fn wait_for_active_reporters() {
    while ACTIVE_REPORTERS.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
    }
}

fn reset_counters() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    DEALLOCATIONS.store(0, Ordering::Relaxed);
    REQUESTED_BYTES.store(0, Ordering::Relaxed);
}

fn snapshot() -> AllocationStats {
    AllocationStats {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        deallocations: DEALLOCATIONS.load(Ordering::Relaxed),
        requested_bytes: REQUESTED_BYTES.load(Ordering::Relaxed),
    }
}

fn saturating_increment(counter: &AtomicU64) {
    saturating_add(counter, 1);
}

fn saturating_add(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn size_as_u64(size: usize) -> u64 {
    u64::try_from(size).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::alloc::{alloc, dealloc, Layout};
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::{
        measure, TEST_HOOK, TEST_HOOK_ARMED, TEST_HOOK_IDLE, TEST_HOOK_PAUSED, TEST_HOOK_RECORDED,
        TEST_HOOK_RELEASE, TEST_OBSERVED_EPOCH, TEST_RECHECKED_EPOCH,
    };

    static WORKER_START: AtomicBool = AtomicBool::new(false);
    static WORKER_ALLOCATED: AtomicBool = AtomicBool::new(false);
    static WORKER_DROP: AtomicBool = AtomicBool::new(false);

    #[test]
    fn delayed_allocator_observation_cannot_leak_into_the_next_measurement() {
        TEST_HOOK.store(TEST_HOOK_IDLE, Ordering::Release);
        TEST_HOOK_RELEASE.store(false, Ordering::Release);
        TEST_OBSERVED_EPOCH.store(u64::MAX, Ordering::Release);
        TEST_RECHECKED_EPOCH.store(u64::MAX, Ordering::Release);
        TEST_HOOK_RECORDED.store(false, Ordering::Release);
        WORKER_START.store(false, Ordering::Release);
        WORKER_ALLOCATED.store(false, Ordering::Release);
        WORKER_DROP.store(false, Ordering::Release);

        let worker = std::thread::spawn(|| {
            while !WORKER_START.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            let layout = Layout::from_size_align(4096, 1).expect("valid test allocation layout");
            // SAFETY: `layout` is nonzero and valid. The returned pointer is
            // checked and later freed exactly once with the same layout.
            let allocation = unsafe { alloc(layout) };
            assert!(!allocation.is_null(), "test allocation must succeed");
            WORKER_ALLOCATED.store(true, Ordering::Release);
            while !WORKER_DROP.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
            // SAFETY: `allocation` came from `alloc(layout)`, is non-null, and
            // has not been freed or reallocated.
            unsafe { dealloc(allocation, layout) };
        });

        let _ = measure(|| {
            TEST_HOOK.store(TEST_HOOK_ARMED, Ordering::Release);
            WORKER_START.store(true, Ordering::Release);
            while TEST_HOOK.load(Ordering::Acquire) != TEST_HOOK_PAUSED {
                std::hint::spin_loop();
            }
        });

        let _ = measure(|| {
            TEST_HOOK_RELEASE.store(true, Ordering::Release);
            while !WORKER_ALLOCATED.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        });

        WORKER_DROP.store(true, Ordering::Release);
        worker.join().expect("allocation worker");

        assert_ne!(
            TEST_OBSERVED_EPOCH.load(Ordering::Acquire),
            TEST_RECHECKED_EPOCH.load(Ordering::Acquire),
            "the planted allocator call must span two measurement epochs"
        );
        assert!(
            !TEST_HOOK_RECORDED.load(Ordering::Acquire),
            "an allocation observed in the prior epoch leaked into the next measurement"
        );
    }
}
