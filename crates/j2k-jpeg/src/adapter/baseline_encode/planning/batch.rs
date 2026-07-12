// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch metadata planning and contiguous source grouping.

use super::jpeg_baseline_gpu_encode_tile_plan;
use crate::adapter::baseline_encode::allocation::try_encode_metadata_vec;
use crate::adapter::baseline_encode::types::{
    JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeTile,
    JpegBaselineSampling,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{JpegBackend, JpegEncodeOptions};

/// Build validated backend-neutral GPU baseline JPEG encode parameters for a batch span.
///
/// The caller is responsible for passing only tiles that share the same backend
/// input allocation. This helper validates each tile, computes per-tile entropy
/// offsets, and returns the combined entropy capacity for the backend batch job.
#[cfg(test)]
pub(in crate::adapter::baseline_encode) fn jpeg_baseline_gpu_encode_batch_plan(
    tiles: &[JpegBaselineGpuEncodeTile],
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
) -> Result<JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError> {
    jpeg_baseline_gpu_encode_batch_plan_with_live_bytes(
        tiles,
        options,
        expected_backend,
        sampling,
        0,
    )
}

pub(in crate::adapter::baseline_encode) fn jpeg_baseline_gpu_encode_batch_plan_with_live_bytes(
    tiles: &[JpegBaselineGpuEncodeTile],
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
    initial_live_bytes: usize,
) -> Result<JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError> {
    let mut live_bytes = initial_live_bytes;
    let mut params = try_encode_metadata_vec(tiles.len(), &mut live_bytes)?;
    let mut total_entropy_capacity = 0usize;
    for tile in tiles {
        let tile_plan = jpeg_baseline_gpu_encode_tile_plan(
            *tile,
            options,
            expected_backend,
            sampling,
            tile.byte_offset,
            total_entropy_capacity,
        )?;
        total_entropy_capacity = total_entropy_capacity
            .checked_add(tile_plan.entropy_capacity)
            .ok_or(JpegBaselineGpuEncodeError::BatchEntropyCapacityOverflow)?;
        checked_jpeg_baseline_frame_capacity(total_entropy_capacity)?;
        params.push(tile_plan.params);
    }

    Ok(JpegBaselineGpuEncodeBatchPlan {
        params,
        total_entropy_capacity,
    })
}

/// Return the end index of a contiguous same-source-buffer batch span.
pub(in crate::adapter::baseline_encode) fn same_source_buffer_batch_end<T, K>(
    tiles: &[T],
    start: usize,
    mut source_key: impl FnMut(&T) -> K,
) -> usize
where
    K: PartialEq,
{
    let key = source_key(&tiles[start]);
    let mut end = start + 1;
    while end < tiles.len() && source_key(&tiles[end]) == key {
        end += 1;
    }
    end
}
