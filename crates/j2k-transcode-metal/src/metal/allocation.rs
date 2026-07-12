// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{Buffer, Device, MetalTranscodeError};
use core::mem::size_of;
use j2k_core::{
    accelerator::GpuAbi, try_host_vec_from_slice, try_host_vec_with_capacity,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
use j2k_metal_support::{
    checked_private_buffer_for_len as support_private_buffer_for_len,
    checked_shared_buffer_for_len as support_shared_buffer_for_len,
    checked_shared_buffer_with_slice as support_shared_buffer_with_slice, MetalSupportError,
};

pub(super) fn checked_element_product(
    factors: &[usize],
    what: &'static str,
) -> Result<usize, MetalTranscodeError> {
    factors
        .iter()
        .try_fold(1usize, |count, factor| count.checked_mul(*factor))
        .ok_or(MetalTranscodeError::HostAllocationTooLarge {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        })
}

pub(super) fn checked_host_element_count<T>(
    factors: &[usize],
    what: &'static str,
) -> Result<usize, MetalTranscodeError> {
    let element_count = checked_element_product(factors, what)?;
    checked_host_workspace_bytes(&[element_count.saturating_mul(size_of::<T>())], what)?;
    Ok(element_count)
}

pub(super) fn checked_device_element_count<T>(
    factors: &[usize],
    what: &'static str,
) -> Result<usize, MetalTranscodeError> {
    let Some(element_count) = factors
        .iter()
        .try_fold(1usize, |count, factor| count.checked_mul(*factor))
    else {
        return Err(MetalTranscodeError::DeviceAllocationTooLarge {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        });
    };
    checked_device_workspace_bytes(&[element_count.saturating_mul(size_of::<T>())], what)?;
    Ok(element_count)
}

pub(super) fn checked_host_workspace_bytes(
    parts: &[usize],
    what: &'static str,
) -> Result<usize, MetalTranscodeError> {
    let requested = parts
        .iter()
        .fold(0usize, |total, part| total.saturating_add(*part));
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(MetalTranscodeError::HostAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        });
    }
    Ok(requested)
}

pub(super) fn checked_device_workspace_bytes(
    parts: &[usize],
    what: &'static str,
) -> Result<usize, MetalTranscodeError> {
    let requested = parts
        .iter()
        .fold(0usize, |total, part| total.saturating_add(*part));
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(MetalTranscodeError::DeviceAllocationTooLarge {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            what,
        });
    }
    Ok(requested)
}

pub(super) fn try_transcode_vec_with_capacity<T>(
    element_count: usize,
    what: &'static str,
) -> Result<Vec<T>, MetalTranscodeError> {
    let element_count = checked_host_element_count::<T>(&[element_count], what)?;
    try_host_vec_with_capacity(element_count).map_err(|error| {
        MetalTranscodeError::HostAllocationFailed {
            requested: error.requested_bytes(),
            what,
        }
    })
}

pub(super) fn try_transcode_vec_from_slice<T: Copy>(
    source: &[T],
    what: &'static str,
) -> Result<Vec<T>, MetalTranscodeError> {
    checked_host_element_count::<T>(&[source.len()], what)?;
    try_host_vec_from_slice(source).map_err(|error| MetalTranscodeError::HostAllocationFailed {
        requested: error.requested_bytes(),
        what,
    })
}

pub(super) fn shared_buffer_with_slice<T: GpuAbi>(
    device: &Device,
    values: &[T],
    what: &'static str,
) -> Result<Buffer, MetalTranscodeError> {
    checked_device_element_count::<T>(&[values.len()], what)?;
    support_shared_buffer_with_slice(device, values)
        .map_err(|error| map_device_allocation_error(error, what))
}

pub(super) fn shared_buffer_for_len<T: GpuAbi>(
    device: &Device,
    len: usize,
    what: &'static str,
) -> Result<Buffer, MetalTranscodeError> {
    checked_device_element_count::<T>(&[len], what)?;
    support_shared_buffer_for_len::<T>(device, len)
        .map_err(|error| map_device_allocation_error(error, what))
}

pub(super) fn private_buffer_for_len<T: GpuAbi>(
    device: &Device,
    len: usize,
    what: &'static str,
) -> Result<Buffer, MetalTranscodeError> {
    checked_device_element_count::<T>(&[len], what)?;
    support_private_buffer_for_len::<T>(device, len)
        .map_err(|error| map_device_allocation_error(error, what))
}

fn map_device_allocation_error(
    error: MetalSupportError,
    what: &'static str,
) -> MetalTranscodeError {
    match error {
        MetalSupportError::BufferAllocationTooLarge { requested, cap } => {
            MetalTranscodeError::DeviceAllocationTooLarge {
                requested,
                cap,
                what,
            }
        }
        MetalSupportError::BufferAllocationFailed { requested } => {
            MetalTranscodeError::DeviceAllocationFailed { requested, what }
        }
        source => MetalTranscodeError::support(what, source),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checked_device_element_count, checked_device_workspace_bytes, checked_host_element_count,
        checked_host_workspace_bytes,
    };
    use crate::MetalTranscodeError;
    use core::mem::size_of;
    use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

    #[test]
    fn host_and_device_products_reject_overflow_and_cap_excess() {
        assert!(matches!(
            checked_host_element_count::<u64>(&[usize::MAX, 2], "host test"),
            Err(MetalTranscodeError::HostAllocationTooLarge {
                requested: usize::MAX,
                what: "host test",
                ..
            })
        ));
        let over_cap = DEFAULT_MAX_HOST_ALLOCATION_BYTES / size_of::<u64>() + 1;
        assert!(matches!(
            checked_device_element_count::<u64>(&[over_cap], "device test"),
            Err(MetalTranscodeError::DeviceAllocationTooLarge {
                requested,
                what: "device test",
                ..
            }) if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));

        let half = DEFAULT_MAX_HOST_ALLOCATION_BYTES / 2 + 1;
        assert!(matches!(
            checked_host_workspace_bytes(&[half, half], "host workspace"),
            Err(MetalTranscodeError::HostAllocationTooLarge {
                what: "host workspace",
                ..
            })
        ));
        assert!(matches!(
            checked_device_workspace_bytes(&[half, half], "device workspace"),
            Err(MetalTranscodeError::DeviceAllocationTooLarge {
                what: "device workspace",
                ..
            })
        ));
    }
}
