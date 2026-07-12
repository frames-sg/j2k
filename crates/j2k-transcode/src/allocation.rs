// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use j2k_core::{
    try_host_vec_filled, try_host_vec_from_slice, try_host_vec_with_capacity,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

use crate::{DctTransformError, JpegToHtj2kError, TranscodeStageError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TranscodeAllocationError {
    MemoryCapExceeded { requested: usize, cap: usize },
    HostAllocationFailed { bytes: usize },
}

impl From<TranscodeAllocationError> for JpegToHtj2kError {
    fn from(value: TranscodeAllocationError) -> Self {
        match value {
            TranscodeAllocationError::MemoryCapExceeded { requested, cap } => {
                Self::MemoryCapExceeded { requested, cap }
            }
            TranscodeAllocationError::HostAllocationFailed { bytes } => {
                Self::HostAllocationFailed { bytes }
            }
        }
    }
}

impl From<TranscodeAllocationError> for TranscodeStageError {
    fn from(value: TranscodeAllocationError) -> Self {
        match value {
            TranscodeAllocationError::MemoryCapExceeded { requested, cap } => {
                Self::MemoryCapExceeded { requested, cap }
            }
            TranscodeAllocationError::HostAllocationFailed { bytes } => {
                Self::HostAllocationFailed { bytes }
            }
        }
    }
}

impl From<TranscodeAllocationError> for DctTransformError {
    fn from(value: TranscodeAllocationError) -> Self {
        match value {
            TranscodeAllocationError::MemoryCapExceeded { requested, cap } => {
                Self::MemoryCapExceeded { requested, cap }
            }
            TranscodeAllocationError::HostAllocationFailed { bytes } => {
                Self::HostAllocationFailed { bytes }
            }
        }
    }
}

pub(crate) fn checked_allocation_bytes<T>(
    element_count: usize,
) -> Result<usize, TranscodeAllocationError> {
    let requested = element_count.checked_mul(size_of::<T>()).ok_or(
        TranscodeAllocationError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        },
    )?;
    ensure_allocation_bytes(requested)?;
    Ok(requested)
}

pub(crate) fn checked_capacity_bytes<T>(
    capacity: usize,
) -> Result<usize, TranscodeAllocationError> {
    checked_allocation_bytes::<T>(capacity)
}

pub(crate) fn checked_allocation_len<T>(
    left: usize,
    right: usize,
) -> Result<usize, TranscodeAllocationError> {
    let element_count =
        left.checked_mul(right)
            .ok_or(TranscodeAllocationError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    checked_allocation_bytes::<T>(element_count)?;
    Ok(element_count)
}

pub(crate) fn ensure_allocation_bytes(requested: usize) -> Result<(), TranscodeAllocationError> {
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(TranscodeAllocationError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(())
}

pub(crate) fn checked_add_allocation_bytes(
    total: usize,
    additional: usize,
) -> Result<usize, TranscodeAllocationError> {
    let requested =
        total
            .checked_add(additional)
            .ok_or(TranscodeAllocationError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })?;
    ensure_allocation_bytes(requested)?;
    Ok(requested)
}

pub(crate) fn try_vec_with_capacity<T>(
    capacity: usize,
) -> Result<Vec<T>, TranscodeAllocationError> {
    checked_allocation_bytes::<T>(capacity)?;
    let values = try_host_vec_with_capacity(capacity).map_err(|error| {
        TranscodeAllocationError::HostAllocationFailed {
            bytes: error.requested_bytes(),
        }
    })?;
    checked_capacity_bytes::<T>(values.capacity())?;
    Ok(values)
}

pub(crate) fn try_vec_filled<T: Clone>(
    len: usize,
    value: T,
) -> Result<Vec<T>, TranscodeAllocationError> {
    checked_allocation_bytes::<T>(len)?;
    let values = try_host_vec_filled(len, value).map_err(|error| {
        TranscodeAllocationError::HostAllocationFailed {
            bytes: error.requested_bytes(),
        }
    })?;
    checked_capacity_bytes::<T>(values.capacity())?;
    Ok(values)
}

pub(crate) fn try_vec_from_slice<T: Copy>(
    source: &[T],
) -> Result<Vec<T>, TranscodeAllocationError> {
    checked_allocation_bytes::<T>(source.len())?;
    let values = try_host_vec_from_slice(source).map_err(|error| {
        TranscodeAllocationError::HostAllocationFailed {
            bytes: error.requested_bytes(),
        }
    })?;
    checked_capacity_bytes::<T>(values.capacity())?;
    Ok(values)
}

pub(crate) fn try_vec_resize_with<T>(
    values: &mut Vec<T>,
    new_len: usize,
    mut value: impl FnMut() -> T,
) -> Result<(), TranscodeAllocationError> {
    checked_allocation_bytes::<T>(new_len)?;
    if new_len > values.len() {
        values
            .try_reserve_exact(new_len - values.len())
            .map_err(|_| TranscodeAllocationError::HostAllocationFailed {
                bytes: new_len.saturating_mul(size_of::<T>()),
            })?;
        checked_capacity_bytes::<T>(values.capacity())?;
    }
    values.resize_with(new_len, &mut value);
    Ok(())
}

pub(crate) fn try_extend_from_slice<T: Copy>(
    values: &mut Vec<T>,
    source: &[T],
) -> Result<(), TranscodeAllocationError> {
    let required_len = values.len().checked_add(source.len()).ok_or(
        TranscodeAllocationError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        },
    )?;
    try_vec_reserve_len(values, required_len)?;
    values.extend_from_slice(source);
    Ok(())
}

pub(crate) fn try_vec_reserve_len<T>(
    values: &mut Vec<T>,
    required_len: usize,
) -> Result<(), TranscodeAllocationError> {
    checked_allocation_bytes::<T>(required_len)?;
    if required_len > values.capacity() {
        values
            .try_reserve_exact(required_len - values.len())
            .map_err(|_| TranscodeAllocationError::HostAllocationFailed {
                bytes: required_len.saturating_mul(size_of::<T>()),
            })?;
        checked_capacity_bytes::<T>(values.capacity())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{checked_allocation_bytes, try_vec_with_capacity, TranscodeAllocationError};
    use crate::DctTransformError;
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    fn element_byte_overflow_is_a_typed_cap_error() {
        assert!(matches!(
            checked_allocation_bytes::<u64>(usize::MAX),
            Err(TranscodeAllocationError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        ));
    }

    #[test]
    fn over_cap_capacity_is_rejected_before_reserve() {
        let count = DEFAULT_MAX_HOST_ALLOCATION_BYTES / size_of::<u64>() + 1;
        assert!(matches!(
            try_vec_with_capacity::<u64>(count),
            Err(TranscodeAllocationError::MemoryCapExceeded { requested, cap })
                if requested > cap && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn transform_error_preserves_allocator_failure_category() {
        let error =
            DctTransformError::from(TranscodeAllocationError::HostAllocationFailed { bytes: 4096 });
        assert_eq!(
            error,
            DctTransformError::HostAllocationFailed { bytes: 4096 }
        );
    }
}
