// SPDX-License-Identifier: MIT OR Apache-2.0

//! Process-wide allocation measurement for serial codec regression probes.
//!
//! The meter is process-global because codec work may allocate on Rayon worker
//! threads. Callers must run measurements serially and must not nest them.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static REALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicI64 = AtomicI64::new(0);
static PEAK_LIVE_BYTES: AtomicI64 = AtomicI64::new(0);

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

// SAFETY: every operation delegates to `System` with the exact caller-provided
// layout. The surrounding atomic bookkeeping neither dereferences nor changes
// allocation pointers.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller's valid allocation layout to `System`.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ACTIVE.load(Ordering::Acquire) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarding the caller's valid allocation layout to `System`.
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && ACTIVE.load(Ordering::Acquire) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ACTIVE.load(Ordering::Acquire) {
            record_deallocation(layout.size());
        }
        // SAFETY: forwarding the pointer and its original layout to `System`.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarding the pointer, its original layout, and requested
        // replacement size to `System`.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if !replacement.is_null() && ACTIVE.load(Ordering::Acquire) {
            record_reallocation(layout.size(), new_size);
        }
        replacement
    }
}

/// Allocation activity observed during one measurement window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocationStats {
    /// Successful allocation calls, including zeroed allocations.
    pub allocations: u64,
    /// Successful reallocation calls.
    pub reallocations: u64,
    /// Sum of requested bytes for successful allocations and reallocations.
    pub allocated_bytes: u64,
    /// Greatest positive change in live requested bytes during the window.
    pub peak_live_bytes: u64,
    /// Signed change in live requested bytes at the end of the window.
    pub retained_bytes: i64,
}

/// Upper bounds for one measured operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    allocation_calls: u64,
    peak_live_bytes: u64,
    retained_bytes: u64,
}

impl Budget {
    /// Require no successful allocation or reallocation calls and no positive
    /// live-byte growth.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            allocation_calls: 0,
            peak_live_bytes: 0,
            retained_bytes: 0,
        }
    }

    /// Bound peak live bytes while requiring the operation to retain no bytes.
    #[must_use]
    pub const fn peak(max_peak_live_bytes: u64) -> Self {
        Self {
            allocation_calls: u64::MAX,
            peak_live_bytes: max_peak_live_bytes,
            retained_bytes: 0,
        }
    }

    /// Bound both peak live bytes and positive bytes retained by the result.
    #[must_use]
    pub const fn peak_retaining(max_peak_live_bytes: u64, max_retained_bytes: u64) -> Self {
        Self {
            allocation_calls: u64::MAX,
            peak_live_bytes: max_peak_live_bytes,
            retained_bytes: max_retained_bytes,
        }
    }

    /// Bound the sum of successful allocation and reallocation calls.
    #[must_use]
    pub const fn with_max_allocations(mut self, max_allocation_calls: u64) -> Self {
        self.allocation_calls = max_allocation_calls;
        self
    }

    fn accepts(self, stats: AllocationStats) -> bool {
        let allocation_calls = stats.allocations.saturating_add(stats.reallocations);
        let positive_retained = u64::try_from(stats.retained_bytes).unwrap_or(0);
        allocation_calls <= self.allocation_calls
            && stats.peak_live_bytes <= self.peak_live_bytes
            && positive_retained <= self.retained_bytes
    }
}

/// Measure one operation with process-global counters.
///
/// The returned value remains live when the final statistics are sampled, so
/// allocations owned by that value contribute to `retained_bytes`.
///
/// # Panics
///
/// Panics if another measurement is already active.
pub fn measure<R>(operation: impl FnOnce() -> R) -> (R, AllocationStats) {
    if ACTIVE
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        panic!("allocation measurements must be serial and non-nested");
    }
    reset_counters();
    let guard = MeasurementGuard;
    let result = operation();
    ACTIVE.store(false, Ordering::Release);
    std::mem::forget(guard);
    (result, snapshot())
}

/// Run an operation and panic after metering is disabled if it exceeds a
/// budget.
///
/// # Panics
///
/// Panics when the budget is exceeded or a measurement is already active.
pub fn assert_allocations<R>(label: &str, budget: Budget, operation: impl FnOnce() -> R) -> R {
    let (result, stats) = measure(operation);
    assert!(
        budget.accepts(stats),
        "{label} exceeded allocation budget: stats={stats:?}, budget={budget:?}"
    );
    result
}

struct MeasurementGuard;

impl Drop for MeasurementGuard {
    fn drop(&mut self) {
        ACTIVE.store(false, Ordering::Release);
    }
}

fn reset_counters() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(0, Ordering::Relaxed);
}

fn snapshot() -> AllocationStats {
    let peak_live = PEAK_LIVE_BYTES.load(Ordering::Relaxed).max(0);
    AllocationStats {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: u64::try_from(peak_live).unwrap_or(u64::MAX),
        retained_bytes: LIVE_BYTES.load(Ordering::Relaxed),
    }
}

fn record_allocation(size: usize) {
    saturating_increment(&ALLOCATIONS);
    saturating_add_u64(&ALLOCATED_BYTES, size_as_u64(size));
    add_live(size_as_i64(size));
}

fn record_deallocation(size: usize) {
    add_live(-size_as_i64(size));
}

fn record_reallocation(old_size: usize, new_size: usize) {
    saturating_increment(&REALLOCATIONS);
    saturating_add_u64(&ALLOCATED_BYTES, size_as_u64(new_size));
    add_live(size_as_i64(new_size).saturating_sub(size_as_i64(old_size)));
}

fn add_live(delta: i64) {
    let previous = LIVE_BYTES
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
            Some(live.saturating_add(delta))
        })
        .unwrap_or_else(|live| live);
    let current = previous.saturating_add(delta);
    PEAK_LIVE_BYTES.fetch_max(current, Ordering::Relaxed);
}

fn saturating_increment(counter: &AtomicU64) {
    saturating_add_u64(counter, 1);
}

fn saturating_add_u64(counter: &AtomicU64, value: u64) {
    let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(value))
    });
}

fn size_as_u64(size: usize) -> u64 {
    u64::try_from(size).unwrap_or(u64::MAX)
}

fn size_as_i64(size: usize) -> i64 {
    i64::try_from(size).unwrap_or(i64::MAX)
}
