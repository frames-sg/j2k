// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::cell::Cell;
use std::mem::{align_of, size_of, size_of_val};

use j2k_metal_support::{private_buffer, shared_buffer, shared_buffer_with_bytes};
use metal::{Buffer, Device};

use crate::Error;

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

fn checked_buffer_required_bytes<T>(len: usize, context: &str) -> Result<usize, Error> {
    let element_size = size_of::<T>();
    if element_size == 0 {
        return Err(Error::MetalKernel {
            message: format!("JPEG Metal {context} readback uses zero-sized element"),
        });
    }
    element_size
        .checked_mul(len)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("JPEG Metal {context} readback size overflow"),
        })
}

fn checked_buffer_required_range<T>(
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<usize, Error> {
    let required_bytes = checked_buffer_required_bytes::<T>(len, context)?;
    let required_end =
        byte_offset
            .checked_add(required_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: format!("JPEG Metal {context} readback range overflow"),
            })?;
    if !byte_offset.is_multiple_of(align_of::<T>()) {
        return Err(Error::MetalKernel {
            message: format!("JPEG Metal {context} readback offset is not element-aligned"),
        });
    }
    Ok(required_end)
}

fn checked_buffer_contents<T>(
    buffer: &Buffer,
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<*const T, Error> {
    let required_end = checked_buffer_required_range::<T>(byte_offset, len, context)?;
    let available_bytes = usize::try_from(buffer.length()).map_err(|_| Error::MetalKernel {
        message: format!("JPEG Metal {context} readback buffer length exceeds usize"),
    })?;
    if required_end > available_bytes {
        return Err(Error::MetalKernel {
            message: format!(
                "JPEG Metal {context} readback exceeds buffer length: need {required_end}, buffer has {available_bytes}"
            ),
        });
    }
    let contents = buffer.contents();
    if contents.is_null() {
        return Err(Error::MetalKernel {
            message: format!("JPEG Metal {context} readback buffer is not CPU-visible"),
        });
    }
    // SAFETY: The range and alignment checks above keep the pointer arithmetic
    // in bounds for a CPU-visible Metal buffer.
    Ok(unsafe { contents.cast::<u8>().add(byte_offset).cast::<T>() })
}

pub(crate) fn checked_buffer_read<T: Copy>(buffer: &Buffer, context: &str) -> Result<T, Error> {
    let contents = checked_buffer_contents::<T>(buffer, 0, 1, context)?;
    // SAFETY: `checked_buffer_contents` verified CPU visibility and that the
    // Metal buffer has enough bytes for one `T`; callers invoke this only after
    // the producing command buffer has completed.
    Ok(unsafe { contents.read() })
}

pub(crate) fn checked_buffer_slice<'a, T>(
    buffer: &'a Buffer,
    len: usize,
    context: &str,
) -> Result<&'a [T], Error> {
    checked_buffer_slice_at(buffer, 0, len, context)
}

pub(crate) fn checked_buffer_slice_at<'a, T>(
    buffer: &'a Buffer,
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<&'a [T], Error> {
    let contents = checked_buffer_contents::<T>(buffer, byte_offset, len, context)?;
    // SAFETY: `checked_buffer_contents` verified CPU visibility and byte
    // length for `len` elements; callers invoke this only after the producing
    // command buffer has completed.
    Ok(unsafe { core::slice::from_raw_parts(contents, len) })
}

pub(crate) fn checked_copy_bytes_to_buffer_at(
    buffer: &Buffer,
    byte_offset: usize,
    bytes: &[u8],
    context: &str,
) -> Result<(), Error> {
    if bytes.is_empty() {
        return Ok(());
    }
    let contents = checked_buffer_contents::<u8>(buffer, byte_offset, bytes.len(), context)?;
    // SAFETY: `checked_buffer_contents` verified CPU visibility and byte
    // length for the destination range.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.cast_mut(), bytes.len());
    }
    Ok(())
}

pub(crate) fn checked_fill_buffer_u8(
    buffer: &Buffer,
    len: usize,
    value: u8,
    context: &str,
) -> Result<(), Error> {
    if len == 0 {
        return Ok(());
    }
    let contents = checked_buffer_contents::<u8>(buffer, 0, len, context)?;
    // SAFETY: `checked_buffer_contents` verified CPU visibility and byte
    // length for the destination range.
    unsafe {
        core::ptr::write_bytes(contents.cast_mut(), value, len);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_buffer_required_range_rejects_overflow_and_alignment_errors() {
        assert_eq!(
            checked_buffer_required_bytes::<u32>(2, "status").expect("u32 byte count"),
            8
        );
        assert_eq!(
            checked_buffer_required_range::<u16>(4, 3, "status").expect("u16 byte range"),
            10
        );

        let size_overflow =
            checked_buffer_required_bytes::<u16>(usize::MAX, "status").expect_err("overflow");
        assert!(
            matches!(size_overflow, Error::MetalKernel { message } if message.contains("size overflow"))
        );

        let range_overflow =
            checked_buffer_required_range::<u8>(usize::MAX, 1, "status").expect_err("range");
        assert!(
            matches!(range_overflow, Error::MetalKernel { message } if message.contains("range overflow"))
        );

        let misaligned =
            checked_buffer_required_range::<u16>(1, 1, "status").expect_err("alignment");
        assert!(
            matches!(misaligned, Error::MetalKernel { message } if message.contains("element-aligned"))
        );
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
