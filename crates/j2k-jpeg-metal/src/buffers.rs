// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::accelerator::GpuAbi;
use j2k_metal_support::{
    checked_buffer_fill_bytes, checked_buffer_read as support_checked_buffer_read,
    checked_buffer_read_vec, checked_buffer_write, checked_private_buffer, checked_shared_buffer,
    checked_shared_buffer_with_bytes, checked_shared_buffer_with_slice, MetalSupportError,
};
use metal::{Buffer, DeviceRef};
#[cfg(test)]
use std::cell::Cell;

use crate::{error::metal_kernel_support_error, Error};

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

pub(crate) fn new_shared_buffer(device: &DeviceRef, bytes: usize) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_shared_buffer(device, bytes).map_err(buffer_allocation_error)
}

pub(crate) fn new_shared_buffer_with_data(
    device: &DeviceRef,
    bytes: &[u8],
) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_shared_buffer_with_bytes(device, bytes).map_err(buffer_allocation_error)
}

pub(crate) fn new_shared_buffer_with_slice<T: GpuAbi>(
    device: &DeviceRef,
    values: &[T],
) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_SHARED_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_shared_buffer_with_slice(device, values).map_err(buffer_allocation_error)
}

pub(crate) fn new_private_buffer(device: &DeviceRef, bytes: usize) -> Result<Buffer, Error> {
    #[cfg(test)]
    JPEG_PRIVATE_BUFFER_ALLOCATIONS.with(|allocations| allocations.set(allocations.get() + 1));
    checked_private_buffer(device, bytes).map_err(buffer_allocation_error)
}

fn buffer_allocation_error(error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("JPEG Metal buffer allocation failed: {error}"),
        error,
    )
}

fn buffer_access_error(context: &str, error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("JPEG Metal {context} buffer access invalid: {error}"),
        error,
    )
}

fn buffer_readback_error(context: &str, error: MetalSupportError) -> Error {
    buffer_access_error(context, error)
}

pub(crate) fn checked_buffer_read<T: GpuAbi>(buffer: &Buffer, context: &str) -> Result<T, Error> {
    // SAFETY: JPEG readback helpers are called only for CPU-initialized buffers
    // or after `commit_and_wait_jpeg` has completed the producing commands.
    unsafe { support_checked_buffer_read::<T>(buffer, 0) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn checked_buffer_slice<T: GpuAbi>(
    buffer: &Buffer,
    len: usize,
    context: &str,
) -> Result<Vec<T>, Error> {
    checked_buffer_slice_at(buffer, 0, len, context)
}

pub(crate) fn checked_buffer_slice_at<T: GpuAbi>(
    buffer: &Buffer,
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<Vec<T>, Error> {
    // SAFETY: JPEG readback helpers are called only for CPU-initialized buffers
    // or after `commit_and_wait_jpeg` has completed the producing commands.
    unsafe { checked_buffer_read_vec::<T>(buffer, byte_offset, len) }
        .map_err(|error| buffer_readback_error(context, error))
}

pub(crate) fn checked_copy_bytes_to_buffer_at(
    buffer: &Buffer,
    byte_offset: usize,
    bytes: &[u8],
    context: &str,
) -> Result<(), Error> {
    // SAFETY: Viewport-cache writes occur during CPU staging while the cached
    // buffer is not submitted to a Metal command buffer.
    unsafe { checked_buffer_write::<u8>(buffer, byte_offset, bytes) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn checked_fill_buffer_u8(
    buffer: &Buffer,
    len: usize,
    value: u8,
    context: &str,
) -> Result<(), Error> {
    // SAFETY: Viewport-cache fills occur during CPU staging while the cached
    // buffer is not submitted to a Metal command buffer.
    unsafe { checked_buffer_fill_bytes(buffer, 0, len, value) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn new_decode_plane_buffer(
    device: &DeviceRef,
    bytes: usize,
    returned_publicly: bool,
) -> Result<Buffer, Error> {
    if returned_publicly {
        new_shared_buffer(device, bytes)
    } else {
        new_private_buffer(device, bytes)
    }
}

#[cfg(test)]
mod tests {
    use j2k_metal_support::MetalSupportError;

    use super::{buffer_access_error, buffer_readback_error};
    use crate::Error;

    #[test]
    fn buffer_access_errors_keep_jpeg_context() {
        let error = buffer_access_error(
            "status readback",
            MetalSupportError::BufferAlignment {
                offset_bytes: 1,
                align: 4,
            },
        );
        assert!(matches!(
            error,
            Error::MetalSupport { message, source: MetalSupportError::BufferAlignment { .. } }
                if message.contains("JPEG Metal status readback")
                    && message.contains("not aligned")
        ));
    }

    #[test]
    fn readback_allocation_errors_keep_the_typed_element_count_without_fake_bytes() {
        let source = MetalSupportError::BufferReadbackAllocation {
            abi_name: "test status",
            element_count: usize::MAX,
        };
        let error = buffer_readback_error("status readback", source.clone());
        assert!(matches!(
            &error,
            Error::MetalSupport { source: stored, .. } if stored == &source
        ));
        assert!(error.to_string().contains("test status"));
        assert!(error.to_string().contains(&usize::MAX.to_string()));
        assert!(std::error::Error::source(&error).is_some());
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
        device: &DeviceRef,
        key: &'static str,
        bytes: usize,
    ) -> Result<Buffer, Error> {
        let bytes = bytes.max(1);
        if let Some(entry) = self
            .private_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= bytes)
        {
            return Ok(entry.buffer.clone());
        }

        let buffer = new_private_buffer(device, bytes)?;
        if let Some(entry) = self
            .private_buffers
            .iter_mut()
            .find(|entry| entry.key == key)
        {
            entry.capacity = bytes;
            entry.buffer = buffer.clone();
        } else {
            crate::batch_allocation::try_reserve_for_push(
                &mut self.private_buffers,
                "JPEG Metal private scratch metadata",
            )?;
            self.private_buffers.push(ReusablePrivateBuffer {
                key,
                capacity: bytes,
                buffer: buffer.clone(),
            });
        }
        Ok(buffer)
    }

    pub(crate) fn shared_buffer_with_bytes(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        bytes: &[u8],
    ) -> Result<Buffer, Error> {
        let capacity = bytes.len().max(1);
        let buffer = if let Some(entry) = self
            .shared_buffers
            .iter()
            .find(|entry| entry.key == key && entry.capacity >= capacity)
        {
            entry.buffer.clone()
        } else {
            let buffer = new_shared_buffer(device, capacity)?;
            if let Some(entry) = self
                .shared_buffers
                .iter_mut()
                .find(|entry| entry.key == key)
            {
                entry.capacity = capacity;
                entry.buffer = buffer.clone();
            } else {
                crate::batch_allocation::try_reserve_for_push(
                    &mut self.shared_buffers,
                    "JPEG Metal shared scratch metadata",
                )?;
                self.shared_buffers.push(ReusableSharedBuffer {
                    key,
                    capacity,
                    buffer: buffer.clone(),
                });
            }
            buffer
        };

        if !bytes.is_empty() {
            // SAFETY: This scratch buffer is exclusively leased during CPU
            // initialization and has not yet been submitted to Metal.
            unsafe { checked_buffer_write::<u8>(&buffer, 0, bytes) }
                .map_err(|error| buffer_access_error("shared scratch upload", error))?;
        }
        Ok(buffer)
    }

    pub(crate) fn shared_buffer_with_slice<T: GpuAbi>(
        &mut self,
        device: &DeviceRef,
        key: &'static str,
        values: &[T],
    ) -> Result<Buffer, Error> {
        let bytes = T::slice_as_bytes(values);
        self.shared_buffer_with_bytes(device, key, bytes)
    }
}
