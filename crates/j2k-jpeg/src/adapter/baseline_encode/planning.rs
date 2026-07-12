// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy capacity and backend-neutral GPU tile/batch planning.

mod batch;

#[cfg(test)]
pub(super) use batch::jpeg_baseline_gpu_encode_batch_plan;
pub(super) use batch::{
    jpeg_baseline_gpu_encode_batch_plan_with_live_bytes, same_source_buffer_batch_end,
};

use super::types::{
    JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile,
    JpegBaselineGpuEncodeTilePlan, JpegBaselineSampling,
};
use super::validation::{
    jpeg_baseline_gpu_encode_format_abi, validate_jpeg_baseline_gpu_encode_tile,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{JpegBackend, JpegEncodeError, JpegEncodeOptions};

/// Conservative upper bound for entropy bytes produced by the CPU encoder.
pub(crate) fn jpeg_baseline_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegEncodeError> {
    let mcu_width = u32::from(sampling.max_h) * 8;
    let mcu_height = u32::from(sampling.max_v) * 8;
    let mcus_per_row = u64::from(width.div_ceil(mcu_width));
    let mcu_rows = u64::from(height.div_ceil(mcu_height));
    let total_mcus =
        mcus_per_row
            .checked_mul(mcu_rows)
            .ok_or(JpegEncodeError::InternalInvariant {
                reason: "JPEG MCU count overflow",
            })?;
    let restart_markers = restart_interval.map_or(0, |interval| {
        total_mcus.saturating_sub(1) / u64::from(interval)
    });
    jpeg_baseline_entropy_capacity_for_mcus(total_mcus, sampling, restart_markers)?
        .checked_add(16)
        .ok_or_else(entropy_capacity_overflow)
}

/// Conservative bound for a known MCU span and its emitted restart markers.
pub(crate) fn jpeg_baseline_entropy_capacity_for_mcus(
    total_mcus: u64,
    sampling: JpegBaselineSampling,
    restart_markers: u64,
) -> Result<usize, JpegEncodeError> {
    let blocks_per_mcu = sampling
        .h
        .iter()
        .copied()
        .zip(sampling.v.iter().copied())
        .take(usize::from(sampling.components))
        .try_fold(0u64, |total, (h, v)| {
            u64::from(h)
                .checked_mul(u64::from(v))
                .and_then(|blocks| total.checked_add(blocks))
        })
        .ok_or_else(entropy_capacity_overflow)?;
    let capacity = total_mcus
        .checked_mul(blocks_per_mcu)
        .and_then(|blocks| blocks.checked_mul(512))
        .and_then(|bytes| {
            restart_markers
                .checked_mul(2)
                .and_then(|markers| bytes.checked_add(markers))
        })
        .ok_or_else(entropy_capacity_overflow)?;
    usize::try_from(capacity).map_err(|_| JpegEncodeError::InternalInvariant {
        reason: "JPEG entropy capacity exceeds usize",
    })
}

fn entropy_capacity_overflow() -> JpegEncodeError {
    JpegEncodeError::InternalInvariant {
        reason: "JPEG entropy capacity overflow",
    }
}

/// Return a GPU ABI-safe entropy capacity for resident baseline encode.
fn jpeg_baseline_gpu_entropy_capacity_bytes(
    width: u32,
    height: u32,
    sampling: JpegBaselineSampling,
    restart_interval: Option<u16>,
) -> Result<usize, JpegBaselineGpuEncodeError> {
    let capacity = jpeg_baseline_entropy_capacity_bytes(width, height, sampling, restart_interval)?;
    checked_jpeg_baseline_frame_capacity(capacity)?;
    if capacity > u32::MAX as usize {
        return Err(JpegBaselineGpuEncodeError::EntropyCapacityTooLarge);
    }
    Ok(capacity)
}

/// Validate one resident GPU tile and all backend-neutral planning bounds.
///
/// # Errors
///
/// Returns a typed option, geometry, buffer-range, or GPU ABI planning error.
#[doc(hidden)]
pub fn preflight_jpeg_baseline_gpu_encode_tile(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
) -> Result<(), JpegBaselineGpuEncodeError> {
    let sampling = super::tables::baseline_encode_tables(options)?.sampling;
    jpeg_baseline_gpu_encode_tile_plan(
        tile,
        options,
        expected_backend,
        sampling,
        tile.byte_offset,
        0,
    )?;
    Ok(())
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
