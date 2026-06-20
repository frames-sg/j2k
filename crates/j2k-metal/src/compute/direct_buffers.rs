// SPDX-License-Identifier: Apache-2.0

use std::mem::{size_of, size_of_val};

use metal::{Buffer, Device, MTLResourceOptions};

use crate::Error;

use super::{
    direct_scratch::{take_recyclable_shared_buffer, DirectScratchBuffer},
    MetalRuntime, J2K_CLASSIC_MAX_COEFF_COUNT,
};

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

pub(super) fn zeroed_recyclable_shared_buffer(
    runtime: &MetalRuntime,
    bytes: usize,
    recyclable_shared_buffers: &mut Vec<(usize, Buffer)>,
) -> Result<Buffer, Error> {
    let bytes = bytes.max(1);
    let buffer = take_recyclable_shared_buffer(runtime, bytes, recyclable_shared_buffers)?;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
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
