// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
use std::cell::Cell;
use std::mem::size_of_val;

use j2k_metal_support::{private_buffer, shared_buffer, shared_buffer_with_bytes};
use metal::{Buffer, Device};

#[cfg(test)]
std::thread_local! {
    static JPEG_PRIVATE_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static JPEG_SHARED_BUFFER_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_jpeg_private_buffer_allocations_for_test() {
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(0));
}

#[cfg(test)]
pub(crate) fn reset_jpeg_shared_buffer_allocations_for_test() {
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(0));
}

#[cfg(test)]
pub(crate) fn jpeg_private_buffer_allocations_for_test() -> usize {
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(Cell::get)
}

#[cfg(test)]
pub(crate) fn jpeg_shared_buffer_allocations_for_test() -> usize {
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(Cell::get)
}

pub(crate) fn new_shared_buffer(device: &Device, bytes: usize) -> Buffer {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    shared_buffer(device, bytes)
}

pub(crate) fn new_shared_buffer_with_data(device: &Device, bytes: &[u8]) -> Buffer {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    shared_buffer_with_bytes(device, bytes)
}

pub(crate) fn new_private_buffer(device: &Device, bytes: usize) -> Buffer {
    #[cfg(test)]
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    private_buffer(device, bytes)
}

pub(crate) fn new_decode_plane_buffer(
    device: &Device,
    bytes: usize,
    returned_publicly: bool,
) -> Buffer {
    if returned_publicly {
        new_shared_buffer(device, bytes)
    } else {
        new_private_buffer(device, bytes)
    }
}

struct ReusablePrivateBuffer {
    key: &'static str,
    capacity: usize,
    buffer: Buffer,
}

struct ReusableSharedBuffer {
    key: &'static str,
    capacity: usize,
    buffer: Buffer,
}

#[derive(Default)]
pub(crate) struct MetalBatchScratch {
    private_buffers: Vec<ReusablePrivateBuffer>,
    shared_buffers: Vec<ReusableSharedBuffer>,
}

impl MetalBatchScratch {
    pub(crate) fn private_buffer(
        &mut self,
        device: &Device,
        key: &'static str,
        bytes: usize,
    ) -> Buffer {
        let bytes = bytes.max(1);
        if let Some(entry) = self
            .private_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= bytes)
        {
            return entry.buffer.clone();
        }

        let buffer = new_private_buffer(device, bytes);
        if let Some(entry) = self
            .private_buffers
            .iter_mut()
            .find(|entry| entry.key == key)
        {
            entry.capacity = bytes;
            entry.buffer = buffer.clone();
        } else {
            self.private_buffers.push(ReusablePrivateBuffer {
                key,
                capacity: bytes,
                buffer: buffer.clone(),
            });
        }
        buffer
    }

    pub(crate) fn shared_buffer_with_bytes(
        &mut self,
        device: &Device,
        key: &'static str,
        bytes: &[u8],
    ) -> Buffer {
        let capacity = bytes.len().max(1);
        let buffer = if let Some(entry) = self
            .shared_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= capacity)
        {
            entry.buffer.clone()
        } else {
            let buffer = new_shared_buffer(device, capacity);
            if let Some(entry) = self
                .shared_buffers
                .iter_mut()
                .find(|entry| entry.key == key)
            {
                entry.capacity = capacity;
                entry.buffer = buffer.clone();
            } else {
                self.shared_buffers.push(ReusableSharedBuffer {
                    key,
                    capacity,
                    buffer: buffer.clone(),
                });
            }
            buffer
        };

        if !bytes.is_empty() {
            // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
            unsafe {
                core::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.contents().cast::<u8>(),
                    bytes.len(),
                );
            }
        }
        buffer
    }

    pub(crate) fn shared_buffer_with_slice<T>(
        &mut self,
        device: &Device,
        key: &'static str,
        values: &[T],
    ) -> Buffer {
        // SAFETY: The immutable slice is reinterpreted as its initialized byte range.
        let bytes = unsafe {
            core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
        };
        self.shared_buffer_with_bytes(device, key, bytes)
    }
}
