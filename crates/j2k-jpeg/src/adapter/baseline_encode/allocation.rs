// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase-wide host-allocation budgets for baseline JPEG encode paths.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::mem::size_of;

use j2k_core::try_host_vec_with_capacity;

use super::planning::{jpeg_baseline_gpu_encode_tile_plan, same_source_buffer_batch_end};
use super::types::{
    JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile,
    JpegBaselineSampling,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions};

pub(super) fn try_encode_metadata_vec<T>(
    capacity: usize,
    live_bytes: &mut usize,
) -> Result<Vec<T>, JpegEncodeError> {
    let planned_bytes = checked_element_capacity_bytes::<T>(capacity)?;
    checked_encode_host_live_bytes([*live_bytes, planned_bytes])?;
    let values = try_host_vec_with_capacity(capacity).map_err(|error| {
        JpegEncodeError::HostAllocationFailed {
            bytes: error.requested_bytes(),
        }
    })?;
    let actual_bytes = checked_element_capacity_bytes::<T>(values.capacity())?;
    *live_bytes = checked_encode_host_live_bytes([*live_bytes, actual_bytes])?;
    Ok(values)
}

/// Check the aggregate capacity of simultaneously live encode allocations.
pub(crate) fn checked_encode_host_live_bytes(
    capacities: impl IntoIterator<Item = usize>,
) -> Result<usize, JpegEncodeError> {
    let requested = capacities
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(cap_overflow)?;
    if requested > j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES {
        return Err(JpegEncodeError::MemoryCapExceeded {
            requested,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        });
    }
    Ok(requested)
}

pub(super) fn encoded_frame_capacity_bytes(
    frames: &[EncodedJpeg],
) -> Result<usize, JpegEncodeError> {
    frames
        .iter()
        .map(|frame| frame.data.capacity())
        .try_fold(0usize, |total, capacity| {
            checked_encode_host_live_bytes([total, capacity])
        })
}

pub(super) fn byte_chunk_capacity_bytes(
    outer_capacity: usize,
    chunks: &[Vec<u8>],
) -> Result<usize, JpegEncodeError> {
    let outer = checked_element_capacity_bytes::<Vec<u8>>(outer_capacity)?;
    let payload = chunks
        .iter()
        .map(Vec::capacity)
        .try_fold(0usize, |total, capacity| {
            checked_encode_host_live_bytes([total, capacity])
        })?;
    checked_encode_host_live_bytes([outer, payload])
}

pub(super) fn checked_gpu_group_runtime_live_bytes(
    external_live_bytes: usize,
    entropy_bytes: usize,
    entropy_outer_bytes: usize,
    params_bytes: usize,
    frame_bytes: usize,
) -> Result<usize, JpegEncodeError> {
    checked_encode_host_live_bytes([
        external_live_bytes,
        entropy_bytes,
        entropy_outer_bytes,
        params_bytes.max(frame_bytes),
    ])
}

/// Check the live host peak for one resident GPU entropy-and-frame operation.
pub(super) fn checked_gpu_tile_live_bytes(
    entropy_capacity: usize,
) -> Result<usize, JpegEncodeError> {
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
    checked_encode_host_live_bytes([entropy_capacity, frame_capacity])
}

/// Aggregate capacities for one contiguous resident-source batch group.
#[derive(Debug, Default)]
pub(super) struct GpuEncodeGroupAllocation {
    tile_count: usize,
    entropy_bytes: usize,
    frame_bytes: usize,
}

impl GpuEncodeGroupAllocation {
    pub(super) fn add_tile(&mut self, entropy_capacity: usize) -> Result<(), JpegEncodeError> {
        let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
        self.tile_count = self.tile_count.checked_add(1).ok_or_else(cap_overflow)?;
        self.entropy_bytes =
            checked_encode_host_live_bytes([self.entropy_bytes, entropy_capacity])?;
        self.frame_bytes = checked_encode_host_live_bytes([self.frame_bytes, frame_capacity])?;
        Ok(())
    }
}

/// Whole-call live budget for ordered resident GPU batch orchestration.
#[derive(Debug)]
pub(super) struct GpuBatchAllocationBudget {
    fixed_metadata: usize,
    retained_frames: usize,
    peak: usize,
}

impl GpuBatchAllocationBudget {
    /// Account for the two caller-length vectors retained throughout the call.
    pub(super) fn new(tile_count: usize) -> Result<Self, JpegEncodeError> {
        let fixed_metadata = checked_encode_host_live_bytes([
            checked_element_capacity_bytes::<EncodedJpeg>(tile_count)?,
            checked_element_capacity_bytes::<JpegBaselineGpuEncodeTile>(tile_count)?,
        ])?;
        Ok(Self {
            fixed_metadata,
            retained_frames: 0,
            peak: fixed_metadata,
        })
    }

    pub(super) fn with_fixed_metadata_bytes(
        fixed_metadata: usize,
    ) -> Result<Self, JpegEncodeError> {
        checked_encode_host_live_bytes([fixed_metadata])?;
        Ok(Self {
            fixed_metadata,
            retained_frames: 0,
            peak: fixed_metadata,
        })
    }

    /// Add one source group, including prior outputs and current entropy owners.
    pub(super) fn add_group(
        &mut self,
        group: &GpuEncodeGroupAllocation,
    ) -> Result<(), JpegEncodeError> {
        if group.tile_count == 0 {
            return Ok(());
        }

        let (plan_metadata_bytes, entropy_outer_bytes) = if group.tile_count == 1 {
            (0, 0)
        } else {
            (
                checked_element_capacity_bytes::<JpegBaselineGpuEncodeParams>(group.tile_count)?,
                checked_element_capacity_bytes::<Vec<u8>>(group.tile_count)?,
            )
        };
        let group_peak = checked_encode_host_live_bytes([
            self.fixed_metadata,
            self.retained_frames,
            group.entropy_bytes,
            entropy_outer_bytes,
            group.frame_bytes.max(plan_metadata_bytes),
        ])?;
        self.peak = self.peak.max(group_peak);
        self.retained_frames =
            checked_encode_host_live_bytes([self.retained_frames, group.frame_bytes])?;
        let retained_peak =
            checked_encode_host_live_bytes([self.fixed_metadata, self.retained_frames])?;
        self.peak = self.peak.max(retained_peak);
        Ok(())
    }

    pub(super) fn peak_bytes(&self) -> usize {
        self.peak
    }
}

/// Validate the whole batch lifecycle before any backend entropy submission.
pub(super) fn checked_gpu_batch_live_bytes<T, K>(
    gpu_tiles: &[JpegBaselineGpuEncodeTile],
    source_tiles: &[T],
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
    fixed_metadata_bytes: usize,
    mut source_key: impl FnMut(&T) -> K,
) -> Result<usize, JpegBaselineGpuEncodeError>
where
    K: PartialEq,
{
    if gpu_tiles.len() != source_tiles.len() {
        return Err(JpegEncodeError::InternalInvariant {
            reason: "GPU JPEG baseline planning metadata length mismatch",
        }
        .into());
    }

    let mut budget = GpuBatchAllocationBudget::with_fixed_metadata_bytes(fixed_metadata_bytes)?;
    let mut start = 0usize;
    while start < gpu_tiles.len() {
        let end = same_source_buffer_batch_end(source_tiles, start, &mut source_key);
        let mut group = GpuEncodeGroupAllocation::default();
        for tile in &gpu_tiles[start..end] {
            let input_offset = if end - start == 1 {
                0
            } else {
                tile.byte_offset
            };
            let tile_plan = jpeg_baseline_gpu_encode_tile_plan(
                *tile,
                options,
                expected_backend,
                sampling,
                input_offset,
                0,
            )?;
            group.add_tile(tile_plan.entropy_capacity)?;
        }
        budget.add_group(&group)?;
        start = end;
    }
    Ok(budget.peak_bytes())
}

/// Check CPU plane, entropy-workspace, and final-frame copy phases.
pub(crate) fn checked_cpu_encode_live_bytes(
    owned_plane_bytes: usize,
    component_count: usize,
    entropy_capacity: usize,
    entropy_workspace_bytes: usize,
) -> Result<usize, JpegEncodeError> {
    let plane_bytes = checked_encode_host_live_bytes([
        owned_plane_bytes,
        checked_element_capacity_bytes::<Cow<'static, [u8]>>(component_count)?,
    ])?;
    let entropy_peak = checked_encode_host_live_bytes([plane_bytes, entropy_workspace_bytes])?;
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
    let assembly_peak =
        checked_encode_host_live_bytes([plane_bytes, entropy_capacity, frame_capacity])?;
    Ok(entropy_peak.max(assembly_peak))
}

/// Maximum actual component-plane capacity that leaves room for every later
/// entropy and frame-assembly owner.
pub(crate) fn cpu_owned_plane_capacity_limit(
    entropy_capacity: usize,
    entropy_workspace_bytes: usize,
) -> Result<usize, JpegEncodeError> {
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy_capacity)?;
    let assembly_bytes = checked_encode_host_live_bytes([entropy_capacity, frame_capacity])?;
    let later_phase_bytes = entropy_workspace_bytes.max(assembly_bytes);
    j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES
        .checked_sub(later_phase_bytes)
        .ok_or_else(cap_overflow)
}

pub(super) fn checked_element_capacity_bytes<T>(capacity: usize) -> Result<usize, JpegEncodeError> {
    let requested = capacity
        .checked_mul(size_of::<T>())
        .ok_or_else(cap_overflow)?;
    checked_encode_host_live_bytes([requested])
}

fn cap_overflow() -> JpegEncodeError {
    JpegEncodeError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[cfg(test)]
mod tests;
