// SPDX-License-Identifier: MIT OR Apache-2.0

//! Execution and accounting for one contiguous resident-source group.

use alloc::vec::Vec;

use super::ensure_entropy_output_within_plan;
use crate::adapter::baseline_encode::allocation::{
    byte_chunk_capacity_bytes, checked_element_capacity_bytes, checked_encode_host_live_bytes,
    checked_gpu_group_runtime_live_bytes, encoded_frame_capacity_bytes,
};
use crate::adapter::baseline_encode::frame::assemble_jpeg_baseline_frame;
use crate::adapter::baseline_encode::planning::{
    jpeg_baseline_gpu_encode_batch_plan_with_live_bytes, jpeg_baseline_gpu_encode_tile_plan,
};
use crate::adapter::baseline_encode::types::{
    JpegBaselineEncodeTables, JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeParams,
    JpegBaselineGpuEncodeTile,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{EncodedJpeg, JpegEncodeError, JpegEncodeOptions};

pub(super) fn encode_same_source_group<T, A>(
    tiles: &[T],
    gpu_tiles: &[JpegBaselineGpuEncodeTile],
    options: JpegEncodeOptions,
    tables: &JpegBaselineEncodeTables,
    adapter: &mut A,
    encoded: &mut Vec<EncodedJpeg>,
    external_live_bytes: usize,
) -> Result<(), A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let backend = adapter.backend();
    let plan = jpeg_baseline_gpu_encode_batch_plan_with_live_bytes(
        gpu_tiles,
        options,
        backend,
        tables.sampling,
        external_live_bytes,
    )
    .map_err(|error| adapter.map_plan_error(error))?;
    let params_bytes =
        checked_element_capacity_bytes::<JpegBaselineGpuEncodeParams>(plan.params.capacity())?;
    let entropy_outer_bytes = checked_element_capacity_bytes::<Vec<u8>>(tiles.len())?;
    let mut group_frame_capacity = 0usize;
    for gpu_tile in gpu_tiles {
        let tile_plan = jpeg_baseline_gpu_encode_tile_plan(
            *gpu_tile,
            options,
            backend,
            tables.sampling,
            gpu_tile.byte_offset,
            0,
        )
        .map_err(|error| adapter.map_plan_error(error))?;
        group_frame_capacity = checked_encode_host_live_bytes([
            group_frame_capacity,
            checked_jpeg_baseline_frame_capacity(tile_plan.entropy_capacity)?,
        ])?;
    }
    checked_gpu_group_runtime_live_bytes(
        external_live_bytes,
        plan.total_entropy_capacity,
        entropy_outer_bytes,
        params_bytes,
        group_frame_capacity,
    )?;
    let entropy_chunks = adapter.encode_batch_entropy(tiles, tables, plan)?;
    if entropy_chunks.len() != tiles.len() {
        return Err(JpegEncodeError::InternalInvariant {
            reason: "GPU JPEG baseline batch returned the wrong number of entropy chunks",
        }
        .into());
    }
    ensure_entropy_output_within_plan(entropy_chunks.capacity(), tiles.len())?;
    for (gpu_tile, entropy) in gpu_tiles.iter().zip(entropy_chunks.iter()) {
        let tile_plan = jpeg_baseline_gpu_encode_tile_plan(
            *gpu_tile,
            options,
            backend,
            tables.sampling,
            gpu_tile.byte_offset,
            0,
        )
        .map_err(|error| adapter.map_plan_error(error))?;
        ensure_entropy_output_within_plan(entropy.capacity(), tile_plan.entropy_capacity)?;
    }
    let entropy_live_bytes = byte_chunk_capacity_bytes(entropy_chunks.capacity(), &entropy_chunks)?;
    checked_encode_host_live_bytes([
        external_live_bytes,
        entropy_live_bytes,
        group_frame_capacity,
    ])?;
    let group_start = encoded.len();
    for (gpu_tile, entropy) in gpu_tiles.iter().zip(entropy_chunks.iter()) {
        let frame = assemble_jpeg_baseline_frame(
            entropy,
            gpu_tile.output_width,
            gpu_tile.output_height,
            tables,
            options,
            backend,
        )?;
        let current_group_frames = encoded_frame_capacity_bytes(&encoded[group_start..])?;
        checked_encode_host_live_bytes([
            external_live_bytes,
            entropy_live_bytes,
            current_group_frames,
            frame.data.capacity(),
        ])?;
        encoded.push(frame);
    }
    Ok(())
}
