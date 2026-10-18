// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

use crate::adapter::{
    checked_cpu_encode_live_bytes, checked_encode_host_live_bytes, cpu_owned_plane_capacity_limit,
    jpeg_baseline_entropy_capacity_bytes, JpegBaselineSampling,
};
use crate::allocation::AllocationBudgetError;
use crate::encoded_output::JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY;

use super::entropy::entropy_host_workspace_bytes;
use super::{JpegEncodeError, JpegSamples};

pub(super) struct CpuEncodeCapacityPlan {
    pub(super) entropy_capacity: usize,
    pub(super) entropy_workspace_bytes: usize,
    pub(super) plane_capacity_limit: usize,
}

pub(super) fn checked_cpu_encode_capacity_plan(
    samples: JpegSamples<'_>,
    sampling: JpegBaselineSampling,
    expected_sample_len: usize,
    restart_interval: Option<u16>,
) -> Result<CpuEncodeCapacityPlan, JpegEncodeError> {
    let (width, height, owned_plane_bytes) = match samples {
        JpegSamples::Gray8 { width, height, .. } => (width, height, 0),
        JpegSamples::Rgb8 { width, height, .. } => (width, height, expected_sample_len),
    };
    let theoretical_entropy_capacity =
        jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    let entropy_capacity = if restart_interval.is_none() {
        theoretical_entropy_capacity.min(cpu_serial_entropy_capacity_limit(
            owned_plane_bytes,
            usize::from(sampling.components),
        )?)
    } else {
        theoretical_entropy_capacity
    };
    let entropy_workspace_bytes =
        entropy_host_workspace_bytes(width, height, sampling, restart_interval, entropy_capacity)?;
    checked_cpu_encode_live_bytes(
        owned_plane_bytes,
        usize::from(sampling.components),
        entropy_capacity,
        entropy_workspace_bytes,
    )?;
    let plane_capacity_limit =
        cpu_owned_plane_capacity_limit(entropy_capacity, entropy_workspace_bytes)?;
    Ok(CpuEncodeCapacityPlan {
        entropy_capacity,
        entropy_workspace_bytes,
        plane_capacity_limit,
    })
}

// The serial CPU path retains component planes and entropy while allocating
// the final frame. Bound entropy so that its full reservation plus the equally
// sized frame payload and fixed frame overhead fit the shared host cap. A
// stream that exceeds this allowance fails through the existing capped writer.
fn cpu_serial_entropy_capacity_limit(
    owned_plane_bytes: usize,
    component_count: usize,
) -> Result<usize, JpegEncodeError> {
    let plane_metadata_bytes = core::mem::size_of::<Cow<'static, [u8]>>()
        .checked_mul(component_count)
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let fixed_bytes = checked_encode_host_live_bytes([
        owned_plane_bytes,
        plane_metadata_bytes,
        JPEG_BASELINE_FRAME_OVERHEAD_CAPACITY,
    ])?;
    Ok((DEFAULT_MAX_HOST_ALLOCATION_BYTES - fixed_bytes) / 2)
}

pub(super) fn checked_sample_byte_len(
    width: u32,
    height: u32,
    components: usize,
) -> Result<usize, JpegEncodeError> {
    let requested = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(components))
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    if requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(requested)
}

pub(super) fn component_plane_capacity_bytes(
    outer_capacity: usize,
    planes: &[Cow<'_, [u8]>],
) -> Result<usize, JpegEncodeError> {
    let outer = outer_capacity
        .checked_mul(core::mem::size_of::<Cow<'_, [u8]>>())
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    let owned = planes
        .iter()
        .filter_map(|plane| match plane {
            Cow::Borrowed(_) => None,
            Cow::Owned(samples) => Some(samples.capacity()),
        })
        .try_fold(0usize, usize::checked_add)
        .ok_or(JpegEncodeError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    checked_encode_host_live_bytes([outer, owned])
}

pub(super) fn map_allocation_budget_error(error: AllocationBudgetError) -> JpegEncodeError {
    match error {
        AllocationBudgetError::MemoryCapExceeded { requested, cap } => {
            JpegEncodeError::MemoryCapExceeded { requested, cap }
        }
        AllocationBudgetError::HostAllocationFailed { bytes } => {
            JpegEncodeError::HostAllocationFailed { bytes }
        }
    }
}
