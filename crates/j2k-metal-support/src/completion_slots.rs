// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capacity-accounted completion owners shared by deferred codec sessions.

use core::ops::{Index, IndexMut};

use j2k_core::{try_batch_reserve_for_push, BatchInfrastructureError};

/// One completion slot, including its allocation-free reuse link.
#[doc(hidden)]
pub struct CompletionSlot<T> {
    result: Option<T>,
    next_free: Option<usize>,
    active: bool,
}

/// Reusable pending/completed slots. Releasing a slot never allocates.
///
/// Sessions reserve storage before committing a request. They keep their own
/// execution policy and account any resources retained by completed values.
#[doc(hidden)]
pub struct CompletionSlots<T> {
    slots: Vec<CompletionSlot<T>>,
    free: Option<usize>,
    active: usize,
}

impl<T> Default for CompletionSlots<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: None,
            active: 0,
        }
    }
}

impl<T> CompletionSlots<T> {
    /// Initialized slot count, including reusable holes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether no pending or completed request remains.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Allocator capacity to charge as [`CompletionSlot<T>`] records.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// Whether a released slot can serve the next request without growth.
    #[must_use]
    pub const fn can_reuse(&self) -> bool {
        self.free.is_some()
    }

    /// Reserve the next slot under the shared host-allocation cap.
    ///
    /// # Errors
    /// Returns an allocation error when the slot storage cannot be reserved.
    pub fn try_reserve_slot(&mut self, what: &'static str) -> Result<(), BatchInfrastructureError> {
        if !self.can_reuse() {
            try_batch_reserve_for_push(&mut self.slots, what)?;
        }
        Ok(())
    }

    /// Install transactionally admitted replacement storage without allocating.
    /// Returns false without mutation if its capacity cannot hold the live slots.
    pub fn replace_storage(&mut self, mut replacement: Vec<CompletionSlot<T>>) -> bool {
        if !replacement.is_empty() || replacement.capacity() < self.slots.len() {
            return false;
        }
        replacement.append(&mut self.slots);
        self.slots = replacement;
        true
    }

    /// Commit a pending slot after reservation. Returns none if not reserved.
    pub fn allocate(&mut self) -> Option<usize> {
        let index = if let Some(index) = self.free {
            let slot = self.slots.get_mut(index)?;
            self.free = slot.next_free.take();
            slot.active = true;
            index
        } else {
            if self.slots.len() == self.slots.capacity() {
                return None;
            }
            let index = self.slots.len();
            self.slots.push(CompletionSlot {
                result: None,
                next_free: None,
                active: true,
            });
            index
        };
        self.active += 1;
        Some(index)
    }

    /// The result of an active slot, empty while its request is pending.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Option<T>> {
        self.slots
            .get(index)
            .filter(|slot| slot.active)
            .map(|slot| &slot.result)
    }

    /// Mutable result storage for an active request.
    #[must_use]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Option<T>> {
        self.slots
            .get_mut(index)
            .filter(|slot| slot.active)
            .map(|slot| &mut slot.result)
    }

    /// Retained results, including empty pending and reusable slots.
    pub fn iter(&self) -> impl Iterator<Item = &Option<T>> {
        self.slots.iter().map(|slot| &slot.result)
    }

    /// Consume or abandon an active slot, returning its optional completion.
    /// The outer none identifies an invalid or already-released slot.
    pub fn release(&mut self, index: usize) -> Option<Option<T>> {
        let slot = self.slots.get_mut(index).filter(|slot| slot.active)?;
        let result = slot.result.take();
        slot.active = false;
        slot.next_free = self.free;
        self.free = Some(index);
        self.active -= 1;
        if self.active == 0 {
            self.slots.clear();
            self.free = None;
        }
        Some(result)
    }

    /// Cancel a queued request or discard its already-completed result.
    /// Returned owners let the session update codec-specific retention ledgers.
    pub fn cancel<Q>(
        &mut self,
        queued: &mut Vec<Q>,
        index: usize,
        output_slot: impl Fn(&Q) -> usize,
    ) -> Option<(Option<Q>, Option<T>)> {
        self.get(index)?;
        let request = queued
            .iter()
            .position(|request| output_slot(request) == index)
            .map(|position| queued.remove(position));
        let result = self.release(index)?;
        Some((request, result))
    }
}

impl<T> Index<usize> for CompletionSlots<T> {
    type Output = Option<T>;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("active completion slot")
    }
}

impl<T> IndexMut<usize> for CompletionSlots<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).expect("active completion slot")
    }
}
