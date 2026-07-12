// SPDX-License-Identifier: MIT OR Apache-2.0

//! Private host-allocation accounting for resident Metal JPEG encode.

use core::mem::size_of;

use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;
#[cfg(target_os = "macos")]
use j2k_core::{try_host_vec_filled, try_host_vec_with_capacity, HostAllocationError};
use j2k_jpeg::JpegEncodeError;

/// Check one returned entropy owner using its requested or actual capacity.
pub(crate) fn checked_single_output_bytes(
    output_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    checked_peak(output_capacity)
}

/// Check the phase that temporarily retains both parameter ABI vectors.
pub(super) fn checked_batch_conversion_bytes<NeutralParams, MetalParams>(
    neutral_param_capacity: usize,
    metal_param_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    let neutral_params = element_bytes::<NeutralParams>(neutral_param_capacity);
    let metal_params = element_bytes::<MetalParams>(metal_param_capacity);
    checked_peak(saturated_sum([neutral_params, metal_params]))
}

/// Check actual Rust host owners during Metal command/readback phases.
///
/// Device buffers are intentionally excluded. The status readback remains live
/// while returned chunks are copied, so parameters, statuses, the outer result,
/// and actual chunk capacities are counted together.
pub(crate) fn checked_batch_runtime_bytes<MetalParams, Status>(
    metal_param_capacity: usize,
    status_capacity: usize,
    output_outer_capacity: usize,
    output_payload_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    let metal_params = element_bytes::<MetalParams>(metal_param_capacity);
    let statuses = element_bytes::<Status>(status_capacity);
    let output_outer = element_bytes::<Vec<u8>>(output_outer_capacity);
    checked_peak(saturated_sum([
        metal_params,
        statuses,
        output_outer,
        output_payload_capacity,
    ]))
}

#[cfg(target_os = "macos")]
pub(crate) fn try_vec_with_capacity<T>(capacity: usize) -> Result<Vec<T>, crate::Error> {
    try_host_vec_with_capacity(capacity).map_err(host_allocation_error)
}

#[cfg(target_os = "macos")]
pub(crate) fn try_vec_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>, crate::Error> {
    try_host_vec_filled(len, value).map_err(host_allocation_error)
}

#[cfg(target_os = "macos")]
pub(super) fn try_collect_exact<T, I>(iter: I) -> Result<Vec<T>, crate::Error>
where
    I: ExactSizeIterator<Item = T>,
{
    let mut values = try_vec_with_capacity(iter.len())?;
    values.extend(iter);
    Ok(values)
}

fn element_bytes<T>(count: usize) -> usize {
    count.saturating_mul(size_of::<T>())
}

fn saturated_sum(values: impl IntoIterator<Item = usize>) -> usize {
    values.into_iter().fold(0usize, usize::saturating_add)
}

fn checked_peak(requested: usize) -> Result<usize, JpegEncodeError> {
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(requested)
}

#[cfg(target_os = "macos")]
fn host_allocation_error(error: HostAllocationError) -> crate::Error {
    JpegEncodeError::HostAllocationFailed {
        bytes: error.requested_bytes(),
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    type NeutralParams = [u8; 84];
    type MetalParams = [u8; 84];
    type Status = [u32; 4];

    #[test]
    fn metal_single_output_checks_requested_and_actual_capacity() {
        assert_eq!(
            checked_single_output_bytes(DEFAULT_MAX_HOST_ALLOCATION_BYTES).unwrap(),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_single_output_bytes(DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1),
            Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
                if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
                    && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn metal_runtime_peak_accepts_the_cap_and_rejects_one_more_entropy_byte() {
        let tile_count = 2;
        let fixed =
            tile_count * (size_of::<MetalParams>() + size_of::<Status>() + size_of::<Vec<u8>>());
        let entropy_capacity = DEFAULT_MAX_HOST_ALLOCATION_BYTES - fixed;

        assert_eq!(
            checked_batch_runtime_bytes::<MetalParams, Status>(
                tile_count,
                tile_count,
                tile_count,
                entropy_capacity,
            )
            .expect("exact cap is valid"),
            DEFAULT_MAX_HOST_ALLOCATION_BYTES
        );
        assert!(matches!(
            checked_batch_runtime_bytes::<MetalParams, Status>(
                tile_count,
                tile_count,
                tile_count,
                entropy_capacity + 1,
            ),
            Err(JpegEncodeError::MemoryCapExceeded {
                requested,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            }) if requested == DEFAULT_MAX_HOST_ALLOCATION_BYTES + 1
        ));
    }

    #[test]
    fn metal_conversion_peak_counts_old_and_new_parameter_vectors() {
        let neutral_capacity = DEFAULT_MAX_HOST_ALLOCATION_BYTES / size_of::<NeutralParams>();
        assert!(matches!(
            checked_batch_conversion_bytes::<NeutralParams, MetalParams>(neutral_capacity, 1),
            Err(JpegEncodeError::MemoryCapExceeded { requested, cap })
                if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES
                    && cap == DEFAULT_MAX_HOST_ALLOCATION_BYTES
        ));
    }

    #[test]
    fn metal_private_budget_saturates_size_arithmetic_overflow() {
        assert!(matches!(
            checked_batch_runtime_bytes::<MetalParams, Status>(
                usize::MAX,
                usize::MAX,
                usize::MAX,
                usize::MAX,
            ),
            Err(JpegEncodeError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
            })
        ));
    }
}
