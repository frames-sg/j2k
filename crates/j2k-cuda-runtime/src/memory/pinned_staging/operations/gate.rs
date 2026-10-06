// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::context::CudaContext;
use std::cell::RefCell;
use std::sync::{Mutex, MutexGuard};
use std::{cell::Cell, marker::PhantomData};

#[doc(hidden)]
/// Clone-shared transaction guard for one context's page-locked upload staging.
///
/// Holding this guard keeps staging-pool diagnostics, host admission, upload,
/// and recycle in one serialized operation across every clone of the context.
#[must_use = "the pinned-upload transaction ends when this guard is dropped"]
pub struct CudaPinnedUploadOperationGuard<'a> {
    pub(super) context: &'a CudaContext,
    pub(super) _gate: TrackedPinnedUploadGate<'a>,
    pub(super) _not_sync: PhantomData<Cell<()>>,
}

impl<'a> CudaPinnedUploadOperationGuard<'a> {
    pub(super) fn held(&self) -> HeldPinnedUploadGate<'a> {
        HeldPinnedUploadGate {
            context: self.context,
        }
    }
}

thread_local! {
    /// Pinned-upload gates held by this thread, keyed by gate address.
    static HELD_GATES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

fn gate_key(gate: &Mutex<()>) -> usize {
    std::ptr::from_ref(gate).addr()
}

pub(super) fn held_by_current_thread(gate: &Mutex<()>) -> bool {
    let key = gate_key(gate);
    HELD_GATES.with_borrow(|held| held.contains(&key))
}

/// A locked pinned-upload gate that stays registered as held by this thread
/// until it is dropped. The gate is not re-entrant, so work that runs while
/// this thread already holds it must reuse that hold instead of locking again.
pub(super) struct TrackedPinnedUploadGate<'a> {
    key: usize,
    _guard: MutexGuard<'a, ()>,
}

impl<'a> TrackedPinnedUploadGate<'a> {
    pub(super) fn new(gate: &'a Mutex<()>, guard: MutexGuard<'a, ()>) -> Self {
        let key = gate_key(gate);
        HELD_GATES.with_borrow_mut(|held| held.push(key));
        Self { key, _guard: guard }
    }
}

impl Drop for TrackedPinnedUploadGate<'_> {
    fn drop(&mut self) {
        HELD_GATES.with_borrow_mut(|held| {
            if let Some(index) = held.iter().rposition(|&key| key == self.key) {
                held.swap_remove(index);
            }
        });
    }
}

/// Proof that the current thread holds one context's pinned-upload gate.
/// Staging recycle and release run through it.
#[derive(Clone, Copy)]
pub(super) struct HeldPinnedUploadGate<'a> {
    pub(super) context: &'a CudaContext,
}

impl<'a> HeldPinnedUploadGate<'a> {
    /// The current thread's existing hold on `context`'s gate, if it has one.
    pub(super) fn of_current_thread(context: &'a CudaContext) -> Option<Self> {
        held_by_current_thread(&context.inner.pinned_upload_operation).then_some(Self { context })
    }
}
