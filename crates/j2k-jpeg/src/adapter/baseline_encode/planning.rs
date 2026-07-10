// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy capacity and backend-neutral GPU tile/batch planning.

use alloc::vec::Vec;

use super::types::{
    JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeParams,
    JpegBaselineGpuEncodeTile, JpegBaselineGpuEncodeTilePlan, JpegBaselineSampling,
};
use super::validation::{
    jpeg_baseline_gpu_encode_format_abi, validate_jpeg_baseline_gpu_encode_tile,
};
use crate::encoder::{JpegBackend, JpegEncodeError, JpegEncodeOptions};

/// Conservative upper bound for entropy bytes produced by the CPU encoder.
fn jpeg_baseline_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = u64::from(width.div_ceil(mcu_width));
    let mcu_rows = u64::from(height.div_ceil(mcu_height));
    let total_mcus = mcus_per_row
        .checked_mul(mcu_rows)
        .ok_or_else(|| JpegEncodeError::Internal("JPEG MCU count overflow".into()))?;
    let blocks_per_mcu = u64::from(
        sampling.h[0] * sampling.v[0]
            + sampling.h[1] * sampling.v[1]
            + sampling.h[2] * sampling.v[2],
    );
    let restart_markers = restart_interval.map_or(0, |interval| {
        total_mcus.saturating_sub(1) / u64::from(interval)
    });
    let capacity = total_mcus
        .checked_mul(blocks_per_mcu)
        .and_then(|blocks| blocks.checked_mul(512))
        .and_then(|bytes| bytes.checked_add(restart_markers.saturating_mul(2)))
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| JpegEncodeError::Internal("JPEG entropy capacity overflow".into()))?;
    usize::try_from(capacity)
        .map_err(|_| JpegEncodeError::Internal("JPEG entropy capacity exceeds usize".into()))
}

/// Return a GPU ABI-safe entropy capacity for resident baseline encode.
fn jpeg_baseline_gpu_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegBaselineGpuEncodeError> {
    let capacity = jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    if capacity > u32::MAX as usize {
        return Err(JpegBaselineGpuEncodeError::EntropyCapacityTooLarge);
    }
    Ok(capacity)
}

/// Build backend-neutral GPU baseline JPEG encode parameters.
pub(super) fn jpeg_baseline_gpu_encode_params(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    sampling: JpegBaselineSampling,
    entropy_capacity: usize,
    input_offset_bytes: usize,
    entropy_offset_bytes: usize,
) -> Result<JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = tile.output_width.div_ceil(mcu_width);
    let mcu_rows = tile.output_height.div_ceil(mcu_height);
    let pitch_bytes =
        u32::try_from(tile.pitch_bytes).map_err(|_| JpegBaselineGpuEncodeError::PitchTooLarge)?;
    let input_offset_bytes = u32::try_from(input_offset_bytes)
        .map_err(|_| JpegBaselineGpuEncodeError::InputOffsetTooLarge)?;
    let entropy_offset_bytes = u32::try_from(entropy_offset_bytes)
        .map_err(|_| JpegBaselineGpuEncodeError::EntropyOffsetTooLarge)?;
    let format = jpeg_baseline_gpu_encode_format_abi(tile.format)?;

    Ok(JpegBaselineGpuEncodeParams {
        input_offset_bytes,
        input_width: tile.width,
        input_height: tile.height,
        output_width: tile.output_width,
        output_height: tile.output_height,
        pitch_bytes,
        mcus_per_row,
        mcu_rows,
        restart_interval_mcus: u32::from(options.restart_interval.unwrap_or(0)),
        format,
        components: u32::from(sampling.components),
        max_h: u32::from(sampling.max_h),
        max_v: u32::from(sampling.max_v),
        h0: u32::from(sampling.h[0]),
        v0: u32::from(sampling.v[0]),
        h1: u32::from(sampling.h[1]),
        v1: u32::from(sampling.v[1]),
        h2: u32::from(sampling.h[2]),
        v2: u32::from(sampling.v[2]),
        entropy_offset_bytes,
        entropy_capacity: u32::try_from(entropy_capacity)
            .map_err(|_| JpegBaselineGpuEncodeError::EntropyCapacityTooLarge)?,
    })
}

/// Build a validated backend-neutral GPU baseline JPEG encode plan for one tile.
pub(super) fn jpeg_baseline_gpu_encode_tile_plan(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
    input_offset_bytes: usize,
    entropy_offset_bytes: usize,
) -> Result<JpegBaselineGpuEncodeTilePlan, JpegBaselineGpuEncodeError> {
    validate_jpeg_baseline_gpu_encode_tile(tile, options, expected_backend)?;
    let entropy_capacity = jpeg_baseline_gpu_entropy_capacity_bytes(
        tile.output_width,
        tile.output_height,
        sampling,
        options.restart_interval,
    )?;
    let params = jpeg_baseline_gpu_encode_params(
        tile,
        options,
        sampling,
        entropy_capacity,
        input_offset_bytes,
        entropy_offset_bytes,
    )?;
    Ok(JpegBaselineGpuEncodeTilePlan {
        params,
        entropy_capacity,
    })
}

/// Build validated backend-neutral GPU baseline JPEG encode parameters for a batch span.
///
/// The caller is responsible for passing only tiles that share the same backend
/// input allocation. This helper validates each tile, computes per-tile entropy
/// offsets, and returns the combined entropy capacity for the backend batch job.
pub(super) fn jpeg_baseline_gpu_encode_batch_plan(
    tiles: &[JpegBaselineGpuEncodeTile],
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
    sampling: JpegBaselineSampling,
) -> Result<JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError> {
    let mut params = Vec::with_capacity(tiles.len());
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
        params.push(tile_plan.params);
    }

    Ok(JpegBaselineGpuEncodeBatchPlan {
        params,
        total_entropy_capacity,
    })
}

/// Return the end index of a contiguous same-source-buffer batch span.
pub(super) fn same_source_buffer_batch_end<T, K>(
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
