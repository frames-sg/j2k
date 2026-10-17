// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible, capped byte storage for encoded JPEG entropy and frames.

use alloc::vec::Vec;

use j2k_core::{try_host_vec_with_capacity, DEFAULT_MAX_HOST_ALLOCATION_BYTES};

use crate::encoder::JpegEncodeError;

/// Capacity reserved above entropy bytes for every baseline frame assembled here.
pub(crate) const JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY: usize = 768;

/// Maximum entropy payload that can still fit in the shared encoded-frame budget.
pub(crate) const JPEG_BASELINE_MAX_ENTROPY_BYTES: usize =
    DEFAULT_MAX_HOST_ALLOCATION_BYTES - JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;

pub(crate) fn checked_jpeg_baseline_frame_capacity(
    entropy_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    let requested = entropy_capacity
        .checked_add(JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY)
        .ok_or_else(|| memory_cap_error(usize::MAX, DEFAULT_MAX_HOST_ALLOCATION_BYTES))?;
    if entropy_capacity > JPEG_BASELINE_MAX_ENTROPY_BYTES {
        return Err(memory_cap_error(
            requested,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        ));
    }
    Ok(requested)
}

#[derive(Debug)]
pub(crate) struct CappedBytes {
    bytes: Vec<u8>,
    max_len: usize,
}

/// Exclusive cursor into a fixed amount of already owned vector capacity.
///
/// The vector length remains unchanged until `commit`, so speculative bytes
/// beyond the cursor are never exposed.
#[cfg(target_arch = "aarch64")]
pub(crate) struct ExistingSpareCursor<'a, const CAPACITY: usize> {
    bytes: &'a mut Vec<u8>,
}

#[expect(
    clippy::inline_always,
    reason = "measured entropy output paths inline the capacity gate and fixed-spare access into their callers"
)]
impl CappedBytes {
    #[cfg(test)]
    pub(crate) fn new(max_len: usize) -> Self {
        Self {
            bytes: Vec::new(),
            max_len,
        }
    }

    pub(crate) fn try_with_capacity(
        capacity: usize,
        max_len: usize,
    ) -> Result<Self, JpegEncodeError> {
        if capacity > max_len {
            return Err(memory_cap_error(capacity, max_len));
        }
        let bytes = try_host_vec_with_capacity(capacity).map_err(|error| {
            JpegEncodeError::HostAllocationFailed {
                bytes: error.requested_bytes(),
            }
        })?;
        ensure_capacity_within_limit(bytes.capacity(), max_len)?;
        Ok(Self { bytes, max_len })
    }

    pub(crate) fn push(&mut self, byte: u8) -> Result<(), JpegEncodeError> {
        self.reserve_additional(1)?;
        self.bytes.push(byte);
        Ok(())
    }

    #[inline(always)]
    pub(crate) fn extend_from_slice(&mut self, bytes: &[u8]) -> Result<(), JpegEncodeError> {
        self.reserve_additional(bytes.len())?;
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn has_existing_spare<const CAPACITY: usize>(&self) -> bool {
        self.bytes
            .len()
            .checked_add(CAPACITY)
            .is_some_and(|end| end <= self.bytes.capacity() && end <= self.max_len)
    }

    /// Borrow a previously validated fixed spare region without reserving.
    ///
    /// # Safety
    ///
    /// `has_existing_spare::<CAPACITY>()` must have returned true immediately
    /// before this call, with no intervening mutation of this object.
    #[inline(always)]
    #[cfg(target_arch = "aarch64")]
    pub(crate) unsafe fn existing_spare_cursor_unchecked<const CAPACITY: usize>(
        &mut self,
    ) -> ExistingSpareCursor<'_, CAPACITY> {
        debug_assert!(self.has_existing_spare::<CAPACITY>());
        ExistingSpareCursor {
            bytes: &mut self.bytes,
        }
    }

    pub(crate) fn into_vec(self) -> Vec<u8> {
        self.bytes
    }

    pub(crate) fn capacity(&self) -> usize {
        self.bytes.capacity()
    }

    fn reserve_additional(&mut self, additional: usize) -> Result<(), JpegEncodeError> {
        let required = self
            .bytes
            .len()
            .checked_add(additional)
            .ok_or_else(|| memory_cap_error(usize::MAX, self.max_len))?;
        if required > self.max_len {
            return Err(memory_cap_error(required, self.max_len));
        }
        if required <= self.bytes.capacity() {
            return Ok(());
        }

        let doubled = self.bytes.capacity().saturating_mul(2).max(8);
        let target_capacity = required.max(doubled).min(self.max_len);
        let retained_capacity = self.bytes.capacity();
        let transient_peak = retained_capacity
            .checked_add(target_capacity)
            .ok_or_else(|| memory_cap_error(usize::MAX, self.max_len))?;
        if transient_peak > self.max_len {
            return Err(memory_cap_error(transient_peak, self.max_len));
        }
        self.bytes
            .try_reserve_exact(target_capacity - self.bytes.len())
            .map_err(|_| JpegEncodeError::HostAllocationFailed {
                bytes: target_capacity,
            })?;
        let actual_capacity = self.bytes.capacity();
        ensure_capacity_within_limit(actual_capacity, self.max_len)?;
        let actual_peak = retained_capacity
            .checked_add(actual_capacity)
            .ok_or_else(|| memory_cap_error(usize::MAX, self.max_len))?;
        if actual_peak > self.max_len {
            return Err(memory_cap_error(actual_peak, self.max_len));
        }
        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
#[expect(
    clippy::inline_always,
    reason = "the speculative writer keeps cursor creation and commit bookkeeping out of its measured block body"
)]
impl<const CAPACITY: usize> ExistingSpareCursor<'_, CAPACITY> {
    #[inline(always)]
    pub(crate) fn start(&mut self) -> *mut u8 {
        // SAFETY: construction proves the complete fixed region lies within
        // the allocation, including its first byte when CAPACITY is nonzero.
        unsafe { self.bytes.as_mut_ptr().add(self.bytes.len()) }
    }

    /// Expose the contiguously initialized prefix ending at `cursor`.
    ///
    /// # Safety
    ///
    /// `cursor` must be within this fixed region, and every byte in the
    /// half-open `[start, cursor)` prefix must be initialized before this call.
    #[inline(always)]
    pub(crate) unsafe fn commit(self, cursor: *mut u8) {
        let start_len = self.bytes.len();
        // SAFETY: both pointers belong to the unchanged vector allocation.
        let start = unsafe { self.bytes.as_mut_ptr().add(start_len) };
        // SAFETY: both pointers share the allocation, and the caller's
        // fixed-region contract places `cursor` at or after `start`.
        let written = unsafe { cursor.offset_from_unsigned(start) };
        debug_assert!(written <= CAPACITY);
        // SAFETY: the caller initialized this complete contiguous prefix, and
        // construction proved it lies within both capacity and the logical cap.
        unsafe { self.bytes.set_len(start_len + written) };
    }
}

fn ensure_capacity_within_limit(
    actual_capacity: usize,
    max_len: usize,
) -> Result<(), JpegEncodeError> {
    if actual_capacity > max_len {
        return Err(memory_cap_error(actual_capacity, max_len));
    }
    Ok(())
}

fn memory_cap_error(requested: usize, cap: usize) -> JpegEncodeError {
    JpegEncodeError::MemoryCapExceeded { requested, cap }
}

#[cfg(test)]
mod tests;
