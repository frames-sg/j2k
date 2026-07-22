// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::{size_of, size_of_val};

use j2k_core::accelerator::GpuAbi;
use j2k_metal_support::{
    checked_buffer_fill_bytes, checked_buffer_read as support_checked_buffer_read,
    checked_buffer_read_vec, checked_buffer_write, checked_private_buffer, checked_shared_buffer,
    checked_shared_buffer_with_slice, MetalSupportError,
};
use metal::{Buffer, Device};

use crate::{error::metal_kernel_support_error, Error};

use super::abi::J2K_CLASSIC_MAX_COEFF_COUNT;
use super::{
    direct_scratch::{take_recyclable_shared_buffer, DirectScratchBuffer},
    MetalRuntime,
};

fn buffer_access_error(context: &str, error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("J2K Metal {context} buffer access invalid: {error}"),
        error,
    )
}

fn buffer_allocation_error(error: MetalSupportError) -> Error {
    metal_kernel_support_error(
        format!("J2K Metal buffer allocation failed: {error}"),
        error,
    )
}

pub(super) fn new_shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    checked_shared_buffer(device, bytes).map_err(buffer_allocation_error)
}

pub(super) fn new_private_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    checked_private_buffer(device, bytes).map_err(buffer_allocation_error)
}

pub(super) fn new_shared_buffer_with_slice<T: GpuAbi>(
    device: &Device,
    data: &[T],
) -> Result<Buffer, Error> {
    checked_shared_buffer_with_slice(device, data).map_err(buffer_allocation_error)
}

pub(crate) fn checked_buffer_read<T: GpuAbi>(buffer: &Buffer, context: &str) -> Result<T, Error> {
    // SAFETY: J2K readback helpers are called only for CPU-initialized buffers
    // or after the producing Metal command buffer has completed.
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
    // SAFETY: J2K readback helpers are called only for CPU-initialized buffers
    // or after the producing Metal command buffer has completed.
    unsafe { checked_buffer_read_vec::<T>(buffer, byte_offset, len) }
        .map_err(|error| buffer_access_error(context, error))
}

pub(crate) fn buffer_is_cpu_visible(buffer: &Buffer) -> bool {
    !buffer.contents().is_null()
}

/// Copy caller-owned GPU ABI input into Metal-owned shared storage.
pub(super) fn copied_slice_buffer<T: GpuAbi>(device: &Device, data: &[T]) -> Result<Buffer, Error> {
    new_shared_buffer_with_slice(device, data)
}

pub(super) fn copied_recyclable_shared_slice_buffer<T: GpuAbi>(
    runtime: &MetalRuntime,
    data: &[T],
    recyclable_shared_buffers: &mut Vec<crate::buffer_pool::PooledBuffer>,
) -> Result<Buffer, Error> {
    let size = size_of_val(data).max(1);
    let buffer = take_recyclable_shared_buffer(runtime, size, recyclable_shared_buffers)?;
    // SAFETY: The recycled buffer is exclusively held by this preparation
    // path and has not yet been submitted to Metal.
    unsafe { checked_buffer_write(&buffer, 0, data) }
        .map_err(|error| buffer_access_error("recyclable upload", error))?;
    Ok(buffer)
}

pub(super) fn zeroed_shared_buffer(device: &Device, bytes: usize) -> Result<Buffer, Error> {
    // Keep zero-byte callers on the shared helper path instead of
    // early-returning with a bespoke placeholder.
    let bytes = bytes.max(1);
    let buffer = new_shared_buffer(device, bytes)?;
    // SAFETY: The new buffer has not been submitted to Metal and has no other
    // CPU access while it is initialized.
    unsafe { checked_buffer_fill_bytes(&buffer, 0, bytes, 0) }
        .map_err(|error| buffer_access_error("new shared buffer clear", error))?;
    Ok(buffer)
}

pub(super) fn zeroed_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<crate::buffer_pool::PooledBuffer>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let buffer = take_recyclable_shared_buffer(runtime, bytes, recyclable_shared_buffers)?;
    // SAFETY: The recycled buffer is exclusively held by this preparation
    // path and has not yet been resubmitted to Metal.
    unsafe { checked_buffer_fill_bytes(&buffer, 0, bytes, 0) }
        .map_err(|error| buffer_access_error("recyclable buffer clear", error))?;
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
        buffer: runtime.take_private_buffer(bytes)?,
    })
}

#[cfg(test)]
mod tests {
    use j2k_metal_support::MetalSupportError;

    use super::buffer_access_error;
    use crate::Error;

    #[test]
    fn buffer_access_errors_keep_j2k_context() {
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
                if message.contains("J2K Metal status readback")
                    && message.contains("not aligned")
        ));
    }
}
