// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::accelerator::GpuAbi;
use metal::Buffer;

use crate::MetalSupportError;

pub(crate) fn checked_buffer_typed_range<T: GpuAbi>(
    buffer_len: usize,
    offset_bytes: usize,
    len: usize,
) -> Result<usize, MetalSupportError> {
    let element_size = core::mem::size_of::<T>();
    if element_size == 0 {
        return Err(MetalSupportError::BufferZeroSizedType { abi_name: T::NAME });
    }
    let byte_len = len
        .checked_mul(element_size)
        .ok_or(MetalSupportError::BufferBounds {
            offset_bytes,
            byte_len: usize::MAX,
            buffer_len,
        })?;
    let end = offset_bytes
        .checked_add(byte_len)
        .ok_or(MetalSupportError::BufferBounds {
            offset_bytes,
            byte_len,
            buffer_len,
        })?;
    if end > buffer_len {
        return Err(MetalSupportError::BufferBounds {
            offset_bytes,
            byte_len,
            buffer_len,
        });
    }

    let align = core::mem::align_of::<T>();
    if align > 1 && !offset_bytes.is_multiple_of(align) {
        return Err(MetalSupportError::BufferAlignment {
            offset_bytes,
            align,
        });
    }

    Ok(byte_len)
}

fn checked_buffer_contents_ptr<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<*mut T, MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    let byte_len = checked_buffer_typed_range::<T>(buffer_len, offset_bytes, len)?;

    let base = buffer.contents().cast::<u8>();
    if base.is_null() {
        return Err(MetalSupportError::BufferContentsUnavailable);
    }
    let address =
        (base as usize)
            .checked_add(offset_bytes)
            .ok_or(MetalSupportError::BufferBounds {
                offset_bytes,
                byte_len,
                buffer_len,
            })?;
    let align = core::mem::align_of::<T>();
    if align > 1 && !address.is_multiple_of(align) {
        return Err(MetalSupportError::BufferAlignment {
            offset_bytes,
            align,
        });
    }

    // SAFETY: Bounds and alignment were validated above.
    Ok(unsafe { base.add(offset_bytes).cast::<T>() })
}

/// Copy one GPU ABI value out of a CPU-visible Metal buffer.
///
/// # Errors
///
/// Returns a typed range, ABI, alignment, or visibility error.
///
/// # Safety
///
/// All Metal writers must have completed, and the range must remain immutable
/// during the copy.
pub unsafe fn checked_buffer_read<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
) -> Result<T, MetalSupportError> {
    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, 1)?;
    // SAFETY: The pointer is aligned/in bounds and the caller synchronizes it.
    Ok(unsafe { ptr.cast_const().read() })
}

/// Copy GPU ABI values out of a CPU-visible Metal buffer.
///
/// # Errors
///
/// Returns a typed range, ABI, alignment, visibility, or allocation error.
///
/// # Safety
///
/// All Metal writers must have completed, and the range must remain immutable
/// during the copy.
pub unsafe fn checked_buffer_read_vec<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
) -> Result<Vec<T>, MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    checked_buffer_typed_range::<T>(buffer_len, offset_bytes, len)?;
    if len == 0 {
        return Ok(Vec::new());
    }

    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, len)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| MetalSupportError::BufferReadbackAllocation {
            abi_name: T::NAME,
            element_count: len,
        })?;
    // SAFETY: Capacity, range, ABI validity, and synchronization are proven by
    // the checks above plus the caller contract.
    unsafe {
        core::ptr::copy_nonoverlapping(ptr.cast_const(), values.as_mut_ptr(), len);
        values.set_len(len);
    }
    Ok(values)
}

/// Copy GPU ABI values into a CPU-visible Metal buffer.
///
/// # Errors
///
/// Returns a typed range, ABI, alignment, or visibility error.
///
/// # Safety
///
/// No Metal command or other CPU access may overlap this write.
pub unsafe fn checked_buffer_write<T: GpuAbi>(
    buffer: &Buffer,
    offset_bytes: usize,
    values: &[T],
) -> Result<(), MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    checked_buffer_typed_range::<T>(buffer_len, offset_bytes, values.len())?;
    if values.is_empty() {
        return Ok(());
    }

    let ptr = checked_buffer_contents_ptr::<T>(buffer, offset_bytes, values.len())?;
    // SAFETY: The checked destination is valid and the caller has exclusive
    // CPU/GPU access for the copy.
    unsafe { core::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len()) };
    Ok(())
}

/// Fill a checked byte range in a CPU-visible Metal buffer.
///
/// # Errors
///
/// Returns a typed range or visibility error.
///
/// # Safety
///
/// No Metal command or other CPU access may overlap this fill.
pub unsafe fn checked_buffer_fill_bytes(
    buffer: &Buffer,
    offset_bytes: usize,
    len: usize,
    value: u8,
) -> Result<(), MetalSupportError> {
    let buffer_len = usize::try_from(buffer.length()).unwrap_or(usize::MAX);
    checked_buffer_typed_range::<u8>(buffer_len, offset_bytes, len)?;
    if len == 0 {
        return Ok(());
    }

    let ptr = checked_buffer_contents_ptr::<u8>(buffer, offset_bytes, len)?;
    // SAFETY: The checked destination is valid and the caller has exclusive
    // CPU/GPU access for the fill.
    unsafe { core::ptr::write_bytes(ptr, value, len) };
    Ok(())
}
