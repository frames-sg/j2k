// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{accelerator::GpuAbi, DEFAULT_MAX_HOST_ALLOCATION_BYTES};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLTexture, MTLTextureDescriptor};

use crate::MetalSupportError;

pub(crate) fn checked_buffer_allocation_length(
    requested: usize,
    max_buffer_length: usize,
) -> Result<usize, MetalSupportError> {
    let cap = max_buffer_length.min(DEFAULT_MAX_HOST_ALLOCATION_BYTES);
    if requested > cap {
        return Err(MetalSupportError::BufferAllocationTooLarge { requested, cap });
    }
    Ok(requested)
}

fn checked_typed_buffer_bytes<T: GpuAbi>(
    device: &ProtocolObject<dyn MTLDevice>,
    len: usize,
) -> Result<usize, MetalSupportError> {
    let element_size = core::mem::size_of::<T>();
    if element_size == 0 {
        return Err(MetalSupportError::BufferZeroSizedType { abi_name: T::NAME });
    }
    len.checked_mul(element_size)
        .ok_or(MetalSupportError::BufferAllocationTooLarge {
            requested: usize::MAX,
            cap: device
                .maxBufferLength()
                .min(DEFAULT_MAX_HOST_ALLOCATION_BYTES),
        })
}

fn allocate_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
    options: MTLResourceOptions,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    let requested = bytes.max(1);
    let length = checked_buffer_allocation_length(requested, device.maxBufferLength())?;
    device
        .newBufferWithLength_options(length, options)
        .ok_or(MetalSupportError::BufferAllocationFailed { requested })
}

fn allocate_shared_buffer_with_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    if bytes.is_empty() {
        return allocate_buffer(device, 1, MTLResourceOptions::StorageModeShared);
    }
    let requested = bytes.len();
    let length = checked_buffer_allocation_length(requested, device.maxBufferLength())?;
    let pointer = core::ptr::NonNull::new(bytes.as_ptr().cast_mut().cast::<core::ffi::c_void>())
        .expect("a non-empty slice has a non-null data pointer");
    // SAFETY: `pointer` addresses exactly `length == bytes.len()` initialized
    // bytes, and Metal copies them synchronously before this call returns.
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            length,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or(MetalSupportError::BufferAllocationFailed { requested })
}

/// Allocate a shared Metal buffer while checking limits and a nil result.
///
/// Zero-length requests allocate one byte.
///
/// # Errors
///
/// Returns a typed allocation error for limit, dispatch, or nil failures.
pub fn checked_shared_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
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
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    allocate_buffer(device, bytes, MTLResourceOptions::StorageModePrivate)
}

/// Allocate a checked shared Metal buffer initialized from bytes.
///
/// # Errors
///
/// Returns a typed allocation error for limit, dispatch, or nil failures.
pub fn checked_shared_buffer_with_bytes(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    allocate_shared_buffer_with_bytes(device, bytes)
}

/// Allocate a checked shared Metal buffer initialized from GPU ABI values.
///
/// # Errors
///
/// Returns a typed ABI or allocation error.
pub fn checked_shared_buffer_with_slice<T: GpuAbi>(
    device: &ProtocolObject<dyn MTLDevice>,
    values: &[T],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    checked_typed_buffer_bytes::<T>(device, values.len())?;
    allocate_shared_buffer_with_bytes(device, T::slice_as_bytes(values))
}

/// Allocate a checked shared Metal buffer for `len` GPU ABI values.
///
/// # Errors
///
/// Returns a typed ABI or allocation error.
pub fn checked_shared_buffer_for_len<T: GpuAbi>(
    device: &ProtocolObject<dyn MTLDevice>,
    len: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    let bytes = checked_typed_buffer_bytes::<T>(device, len)?;
    checked_shared_buffer(device, bytes)
}

/// Allocate a checked private Metal buffer for `len` GPU ABI values.
///
/// # Errors
///
/// Returns a typed ABI or allocation error.
pub fn checked_private_buffer_for_len<T: GpuAbi>(
    device: &ProtocolObject<dyn MTLDevice>,
    len: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalSupportError> {
    let bytes = checked_typed_buffer_bytes::<T>(device, len)?;
    checked_private_buffer(device, bytes)
}

fn checked_texture_descriptor_geometry(
    descriptor: &MTLTextureDescriptor,
) -> Result<(u64, u64, u64, u64), MetalSupportError> {
    let dimensions = (
        u64::try_from(descriptor.width()).unwrap_or(u64::MAX),
        u64::try_from(descriptor.height()).unwrap_or(u64::MAX),
        u64::try_from(descriptor.depth()).unwrap_or(u64::MAX),
        u64::try_from(descriptor.arrayLength()).unwrap_or(u64::MAX),
    );
    if dimensions.0 == 0 || dimensions.1 == 0 || dimensions.2 == 0 || dimensions.3 == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "width, height, depth, and array length must be nonzero",
        });
    }
    if descriptor.mipmapLevelCount() == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "mipmap level count must be nonzero",
        });
    }
    if descriptor.sampleCount() == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "sample count must be nonzero",
        });
    }
    Ok(dimensions)
}

pub(crate) fn checked_texture_planned_bytes(planned: usize) -> Result<usize, MetalSupportError> {
    let cap = DEFAULT_MAX_HOST_ALLOCATION_BYTES;
    if planned == 0 {
        return Err(MetalSupportError::TextureDescriptorInvalid {
            reason: "device reported a zero-byte texture allocation plan",
        });
    }
    if planned > cap {
        return Err(MetalSupportError::TextureAllocationTooLarge {
            requested: planned,
            cap,
        });
    }
    Ok(planned)
}

fn checked_texture_allocation_plan(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &MTLTextureDescriptor,
) -> Result<(u64, u64, u64, u64), MetalSupportError> {
    let dimensions = checked_texture_descriptor_geometry(descriptor)?;
    checked_texture_planned_bytes(
        device
            .heapTextureSizeAndAlignWithDescriptor(descriptor)
            .size,
    )?;
    Ok(dimensions)
}

/// Allocate a Metal texture after validating nonzero geometry, planned heap
/// bytes against the repository resource cap, and the Objective-C result.
///
/// # Errors
///
/// Returns a typed allocation error when dispatch fails or Metal returns nil.
pub fn checked_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    descriptor: &MTLTextureDescriptor,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, MetalSupportError> {
    let dimensions = checked_texture_allocation_plan(device, descriptor)?;
    device
        .newTextureWithDescriptor(descriptor)
        .ok_or(MetalSupportError::TextureAllocationFailed {
            width: dimensions.0,
            height: dimensions.1,
            depth: dimensions.2,
            array_length: dimensions.3,
        })
}
