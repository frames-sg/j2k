// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::{align_of, size_of, size_of_val};

use metal::{Buffer, Device, MTLResourceOptions};

use crate::Error;

use super::{
    direct_scratch::{take_recyclable_shared_buffer, DirectScratchBuffer},
    MetalRuntime, J2K_CLASSIC_MAX_COEFF_COUNT,
};

fn checked_buffer_required_bytes<T>(len: usize, context: &str) -> Result<usize, Error> {
    let element_size = size_of::<T>();
    if element_size == 0 {
        return Err(Error::MetalKernel {
            message: format!("J2K Metal {context} readback uses zero-sized element"),
        });
    }
    element_size
        .checked_mul(len)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("J2K Metal {context} readback size overflow"),
        })
}

fn checked_buffer_required_range<T>(
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<(usize, usize), Error> {
    let required_bytes = checked_buffer_required_bytes::<T>(len, context)?;
    let end = byte_offset
        .checked_add(required_bytes)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("J2K Metal {context} readback range overflow"),
        })?;
    if !byte_offset.is_multiple_of(align_of::<T>()) {
        return Err(Error::MetalKernel {
            message: format!("J2K Metal {context} readback offset is not element-aligned"),
        });
    }
    Ok((required_bytes, end))
}

fn checked_buffer_contents<T>(
    buffer: &Buffer,
    byte_offset: usize,
    len: usize,
    context: &str,
) -> Result<*const T, Error> {
    let (_, required_end) = checked_buffer_required_range::<T>(byte_offset, len, context)?;
    let available_bytes = usize::try_from(buffer.length()).map_err(|_| Error::MetalKernel {
        message: format!("J2K Metal {context} readback buffer length exceeds usize"),
    })?;
    if required_end > available_bytes {
        return Err(Error::MetalKernel {
            message: format!(
                "J2K Metal {context} readback exceeds buffer length: need {required_end}, buffer has {available_bytes}"
            ),
        });
    }
    let contents = buffer.contents();
    if contents.is_null() {
        return Err(Error::MetalKernel {
            message: format!("J2K Metal {context} readback buffer is not CPU-visible"),
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

pub(crate) fn buffer_is_cpu_visible(buffer: &Buffer) -> bool {
    !buffer.contents().is_null()
}

pub(super) fn owned_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    let size = size_of_val(data).max(1);
    let buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
    if !data.is_empty() {
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                size_of_val(data),
            );
        }
    }
    buffer
}

pub(super) fn wrap_f32_output_buffer(device: &Device, output: &mut [f32]) -> Buffer {
    if output.is_empty() {
        device.new_buffer(
            size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    } else {
        device.new_buffer_with_bytes_no_copy(
            output.as_mut_ptr().cast(),
            size_of_val(output) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

pub(super) fn borrow_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

pub(super) fn borrow_mut_slice_buffer<T>(device: &Device, data: &mut [T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_bytes_no_copy(
            data.as_mut_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }
}

pub(super) fn copied_slice_buffer<T>(device: &Device, data: &[T]) -> Buffer {
    if data.is_empty() {
        device.new_buffer(1, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_data(
            data.as_ptr().cast(),
            size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

pub(super) fn copied_recyclable_shared_slice_buffer<T>(
    runtime: &MetalRuntime,
    data: &[T],
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<Buffer, Error> {
    let size = size_of_val(data).max(1);
    let buffer = take_recyclable_shared_buffer(runtime, size, recyclable_shared_buffers)?;
    if !data.is_empty() {
        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        unsafe {
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                buffer.contents().cast::<u8>(),
                size_of_val(data),
            );
        }
    }
    Ok(buffer)
}

pub(super) fn zeroed_shared_buffer(device: &Device, bytes: usize) -> Buffer {
    // Keep zero-byte callers on the shared helper path instead of
    // early-returning with a bespoke placeholder.
    let bytes = bytes.max(1);
    let buffer = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    // SAFETY: `contents` points to a shared Metal buffer of at least `bytes`
    // bytes, and `bytes` is clamped to a non-zero allocation size above.
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
    }
    buffer
}

pub(super) fn zeroed_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let buffer = take_recyclable_shared_buffer(runtime, bytes, recyclable_shared_buffers)?;
    // SAFETY: `contents` points to a shared Metal buffer of at least `bytes`
    // bytes, and `bytes` is clamped to a non-zero allocation size above.
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes);
    }
    Ok(buffer)
}

fn classic_coefficients_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .and_then(|count| count.checked_mul(size_of::<u32>()))
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K coefficient scratch size overflow".to_string(),
        })
}

pub(super) fn take_classic_coefficients_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_coefficients_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

fn classic_states_scratch_bytes(job_count: usize) -> Result<usize, Error> {
    job_count
        .max(1)
        .checked_mul(J2K_CLASSIC_MAX_COEFF_COUNT)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect states scratch overflow".to_string(),
        })
}

pub(super) fn take_classic_states_scratch_buffer(
    runtime: &MetalRuntime,
    job_count: usize,
) -> Result<DirectScratchBuffer, Error> {
    let bytes = classic_states_scratch_bytes(job_count)?;
    Ok(DirectScratchBuffer {
        bytes,
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_buffer_required_bytes_rejects_overflow_and_zero_sized_elements() {
        assert_eq!(
            checked_buffer_required_bytes::<u32>(3, "status").expect("u32 byte count"),
            12
        );
        assert_eq!(
            checked_buffer_required_range::<u16>(4, 3, "status").expect("u16 byte range"),
            (6, 10)
        );

        let overflow =
            checked_buffer_required_bytes::<u16>(usize::MAX, "status").expect_err("overflow");
        assert!(
            matches!(overflow, Error::MetalKernel { message } if message.contains("size overflow"))
        );

        let zero_sized = checked_buffer_required_bytes::<()>(1, "status").expect_err("zero-sized");
        assert!(
            matches!(zero_sized, Error::MetalKernel { message } if message.contains("zero-sized"))
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
