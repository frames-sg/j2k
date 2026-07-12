// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{accelerator::GpuAbi, DEFAULT_MAX_HOST_ALLOCATION_BYTES};
use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{
        runtime::{Class, Sel},
        Message,
    },
    Buffer, DeviceRef, MTLBuffer, MTLResourceOptions, MTLTexture, MTLTextureDescriptor, Texture,
    TextureDescriptor, TextureDescriptorRef,
};

use crate::MetalSupportError;

pub(crate) fn checked_buffer_allocation_length(
    requested: usize,
    max_buffer_length: u64,
) -> Result<u64, MetalSupportError> {
    let cap = usize::try_from(max_buffer_length)
        .unwrap_or(usize::MAX)
        .min(DEFAULT_MAX_HOST_ALLOCATION_BYTES);
    if requested > cap {
        return Err(MetalSupportError::BufferAllocationTooLarge { requested, cap });
    }
    u64::try_from(requested)
        .map_err(|_| MetalSupportError::BufferAllocationTooLarge { requested, cap })
}

fn checked_typed_buffer_bytes<T: GpuAbi>(
    device: &DeviceRef,
    len: usize,
) -> Result<usize, MetalSupportError> {
    let element_size = core::mem::size_of::<T>();
    if element_size == 0 {
        return Err(MetalSupportError::BufferZeroSizedType { abi_name: T::NAME });
    }
    len.checked_mul(element_size)
        .ok_or(MetalSupportError::BufferAllocationTooLarge {
            requested: usize::MAX,
            cap: usize::try_from(device.max_buffer_length())
                .unwrap_or(usize::MAX)
                .min(DEFAULT_MAX_HOST_ALLOCATION_BYTES),
        })
}

pub(crate) unsafe fn checked_buffer_from_retained_ptr(
    raw: *mut MTLBuffer,
    requested: usize,
) -> Result<Buffer, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::BufferAllocationFailed { requested })
    } else {
        // SAFETY: The caller guarantees that a non-null pointer is a retained
        // MTLBuffer result whose ownership transfers to this wrapper.
        Ok(unsafe { Buffer::from_ptr(raw) })
    }
}

fn allocate_buffer(
    device: &DeviceRef,
    bytes: usize,
    options: MTLResourceOptions,
) -> Result<Buffer, MetalSupportError> {
    let requested = bytes.max(1);
    let length = checked_buffer_allocation_length(requested, device.max_buffer_length())?;
    // SAFETY: The selector and argument ABI match MTLDevice's
    // newBufferWithLength:options:. The retained result is checked for nil.
    let raw: *mut MTLBuffer = unsafe {
        device
            .send_message(
                Sel::register("newBufferWithLength:options:"),
                (length, options.bits()),
            )
            .map_err(|error| MetalSupportError::BufferAllocation {
                message: error.to_string(),
            })?
    };
    // SAFETY: The selector returns either nil or a retained MTLBuffer.
    unsafe { checked_buffer_from_retained_ptr(raw, requested) }
}

fn allocate_shared_buffer_with_bytes(
    device: &DeviceRef,
    bytes: &[u8],
) -> Result<Buffer, MetalSupportError> {
    if bytes.is_empty() {
        return allocate_buffer(device, 1, MTLResourceOptions::StorageModeShared);
    }
    let requested = bytes.len();
    let length = checked_buffer_allocation_length(requested, device.max_buffer_length())?;
    // SAFETY: The non-empty slice remains live for the synchronous copy. The
    // selector ABI is exact, and the retained result is checked before wrap.
    let raw: *mut MTLBuffer = unsafe {
        device
            .send_message(
                Sel::register("newBufferWithBytes:length:options:"),
                (
                    bytes.as_ptr().cast::<core::ffi::c_void>(),
                    length,
                    MTLResourceOptions::StorageModeShared.bits(),
                ),
            )
            .map_err(|error| MetalSupportError::BufferAllocation {
                message: error.to_string(),
            })?
    };
    // SAFETY: The selector returns either nil or a retained MTLBuffer.
    unsafe { checked_buffer_from_retained_ptr(raw, requested) }
}

/// Allocate a shared Metal buffer while checking limits and a nil result.
///
/// Zero-length requests allocate one byte.
///
/// # Errors
///
/// Returns a typed allocation error for limit, dispatch, or nil failures.
pub fn checked_shared_buffer(
    device: &DeviceRef,
    bytes: usize,
) -> Result<Buffer, MetalSupportError> {
    allocate_buffer(device, bytes, MTLResourceOptions::StorageModeShared)
}

/// Allocate a private Metal buffer while checking limits and a nil result.
///
/// Zero-length requests allocate one byte.
///
/// # Errors
///
/// Returns a typed allocation error for limit, dispatch, or nil failures.
pub fn checked_private_buffer(
    device: &DeviceRef,
    bytes: usize,
) -> Result<Buffer, MetalSupportError> {
    allocate_buffer(device, bytes, MTLResourceOptions::StorageModePrivate)
}

/// Allocate a checked shared Metal buffer initialized from bytes.
///
/// # Errors
///
/// Returns a typed allocation error for limit, dispatch, or nil failures.
pub fn checked_shared_buffer_with_bytes(
    device: &DeviceRef,
    bytes: &[u8],
) -> Result<Buffer, MetalSupportError> {
    allocate_shared_buffer_with_bytes(device, bytes)
}

/// Allocate a checked shared Metal buffer initialized from GPU ABI values.
///
/// # Errors
///
/// Returns a typed ABI or allocation error.
pub fn checked_shared_buffer_with_slice<T: GpuAbi>(
    device: &DeviceRef,
    values: &[T],
) -> Result<Buffer, MetalSupportError> {
    checked_typed_buffer_bytes::<T>(device, values.len())?;
    allocate_shared_buffer_with_bytes(device, T::slice_as_bytes(values))
}

/// Allocate a checked shared Metal buffer for `len` GPU ABI values.
///
/// # Errors
///
/// Returns a typed ABI or allocation error.
pub fn checked_shared_buffer_for_len<T: GpuAbi>(
    device: &DeviceRef,
    len: usize,
) -> Result<Buffer, MetalSupportError> {
    let bytes = checked_typed_buffer_bytes::<T>(device, len)?;
    checked_shared_buffer(device, bytes)
}

/// Allocate a checked private Metal buffer for `len` GPU ABI values.
///
/// # Errors
///
/// Returns a typed ABI or allocation error.
pub fn checked_private_buffer_for_len<T: GpuAbi>(
    device: &DeviceRef,
    len: usize,
) -> Result<Buffer, MetalSupportError> {
    let bytes = checked_typed_buffer_bytes::<T>(device, len)?;
    checked_private_buffer(device, bytes)
}

pub(crate) unsafe fn checked_texture_descriptor_from_retained_ptr(
    raw: *mut MTLTextureDescriptor,
) -> Result<TextureDescriptor, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::TextureDescriptorUnavailable)
    } else {
        // SAFETY: The caller guarantees a retained descriptor pointer.
        Ok(unsafe { TextureDescriptor::from_ptr(raw) })
    }
}

/// Create a texture descriptor and reject a null Objective-C allocation.
///
/// # Errors
///
/// Returns [`MetalSupportError::TextureDescriptorUnavailable`] when the
/// descriptor factory returns nil.
pub fn checked_texture_descriptor() -> Result<TextureDescriptor, MetalSupportError> {
    let class = Class::get("MTLTextureDescriptor")
        .ok_or(MetalSupportError::TextureDescriptorUnavailable)?;
    // SAFETY: `new` returns a retained descriptor pointer checked before wrap.
    let raw: *mut MTLTextureDescriptor = unsafe {
        class
            .send_message(Sel::register("new"), ())
            .map_err(|error| MetalSupportError::TextureAllocation {
                message: format!("Metal texture descriptor creation failed: {error}"),
            })?
    };
    // SAFETY: `new` returns either nil or a retained descriptor pointer.
    unsafe { checked_texture_descriptor_from_retained_ptr(raw) }
}

pub(crate) unsafe fn checked_texture_from_retained_ptr(
    raw: *mut MTLTexture,
    dimensions: (u64, u64, u64, u64),
) -> Result<Texture, MetalSupportError> {
    if raw.is_null() {
        Err(MetalSupportError::TextureAllocationFailed {
            width: dimensions.0,
            height: dimensions.1,
            depth: dimensions.2,
            array_length: dimensions.3,
        })
    } else {
        // SAFETY: The caller guarantees a retained MTLTexture pointer.
        Ok(unsafe { Texture::from_ptr(raw) })
    }
}

fn checked_texture_descriptor_geometry(
    descriptor: &TextureDescriptorRef,
) -> Result<(u64, u64, u64, u64), MetalSupportError> {
    let dimensions = (
        descriptor.width(),
        descriptor.height(),
        descriptor.depth(),
        descriptor.array_length(),
    );
    if dimensions.0 == 0 || dimensions.1 == 0 || dimensions.2 == 0 || dimensions.3 == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "width, height, depth, and array length must be nonzero",
        });
    }
    if descriptor.mipmap_level_count() == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "mipmap level count must be nonzero",
        });
    }
    if descriptor.sample_count() == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "sample count must be nonzero",
        });
    }
    Ok(dimensions)
}

pub(crate) fn checked_texture_planned_bytes(planned: u64) -> Result<usize, MetalSupportError> {
    let cap = DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    let requested =
        usize::try_from(planned).map_err(|_| MetalSupportError::TextureAllocationTooLarge {
            requested: usize::MAX,
            cap,
        })?;
    if requested == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "device reported a zero-byte texture allocation plan",
        });
    }
    if requested > cap {
        return Err(MetalSupportError::TextureAllocationTooLarge { requested, cap });
    }
    Ok(requested)
}

fn checked_texture_allocation_plan(
    device: &DeviceRef,
    descriptor: &TextureDescriptorRef,
) -> Result<(u64, u64, u64, u64), MetalSupportError> {
    let dimensions = checked_texture_descriptor_geometry(descriptor)?;
    checked_texture_planned_bytes(device.heap_texture_size_and_align(descriptor).size)?;
    Ok(dimensions)
}

/// Allocate a Metal texture after validating nonzero geometry, planned heap
/// bytes against the repository resource cap, and the Objective-C result.
///
/// # Errors
///
/// Returns a typed allocation error when dispatch fails or Metal returns nil.
pub fn checked_texture(
    device: &DeviceRef,
    descriptor: &TextureDescriptorRef,
) -> Result<Texture, MetalSupportError> {
    let dimensions = checked_texture_allocation_plan(device, descriptor)?;
    // SAFETY: The selector ABI is exact; the descriptor remains live and the
    // retained result is checked before ownership is transferred.
    let raw: *mut MTLTexture = unsafe {
        device
            .send_message(
                Sel::register("newTextureWithDescriptor:"),
                (descriptor.as_ptr(),),
            )
            .map_err(|error| MetalSupportError::TextureAllocation {
                message: error.to_string(),
            })?
    };
    // SAFETY: The selector returns either nil or a retained texture pointer.
    unsafe { checked_texture_from_retained_ptr(raw, dimensions) }
}
