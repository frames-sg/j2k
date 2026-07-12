// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free HT scratch slice preparation after fallible reservation.

use alloc::vec::Vec;

#[expect(
    clippy::inline_always,
    reason = "fuse scratch clearing into the phase loop"
)]
#[inline(always)]
pub(crate) fn zeroed_u16_scratch(buffer: &mut Vec<u16>, len: usize) -> Option<&mut [u16]> {
    ensure_reserved(buffer.capacity(), len)?;
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer[..len].fill(0);
    buffer.get_mut(..len)
}

#[cfg(test)]
pub(crate) fn zeroed_u32_scratch(buffer: &mut Vec<u32>, len: usize) -> Option<&mut [u32]> {
    ensure_reserved(buffer.capacity(), len)?;
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer[..len].fill(0);
    buffer.get_mut(..len)
}

#[expect(
    clippy::inline_always,
    reason = "fuse scratch resizing into the phase loop"
)]
#[inline(always)]
pub(crate) fn resized_u16_scratch(buffer: &mut Vec<u16>, len: usize) -> Option<&mut [u16]> {
    ensure_reserved(buffer.capacity(), len)?;
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer.get_mut(..len)
}

#[expect(
    clippy::inline_always,
    reason = "fuse scratch resizing into the phase loop"
)]
#[inline(always)]
pub(crate) fn resized_u32_scratch(buffer: &mut Vec<u32>, len: usize) -> Option<&mut [u32]> {
    ensure_reserved(buffer.capacity(), len)?;
    if buffer.len() < len {
        buffer.resize(len, 0);
    }
    buffer.get_mut(..len)
}

#[inline]
fn ensure_reserved(capacity: usize, required: usize) -> Option<()> {
    (capacity >= required).then_some(())
}
