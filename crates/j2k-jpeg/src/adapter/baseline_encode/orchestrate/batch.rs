// SPDX-License-Identifier: MIT OR Apache-2.0

//! Ordered resident GPU batch planning, submission, and frame retention.

mod group;

use alloc::vec::Vec;

use super::{encode_jpeg_baseline_gpu_tile_with_tables, ensure_entropy_output_within_plan};
use crate::adapter::baseline_encode::allocation::{
    checked_encode_host_live_bytes, checked_gpu_batch_live_bytes, encoded_frame_capacity_bytes,
    try_encode_metadata_vec, GpuBatchAllocationBudget,
};
use crate::adapter::baseline_encode::planning::same_source_buffer_batch_end;
use crate::adapter::baseline_encode::tables::baseline_encode_tables;
use crate::adapter::baseline_encode::types::JpegBaselineGpuEncodeHostAdapter;
use crate::encoder::{EncodedJpeg, JpegEncodeOptions};
use group::encode_same_source_group;

/// Encode resident GPU tiles through a backend adapter.
///
/// The driver groups only contiguous tiles that share the same resident source
/// allocation, preserving input order in the returned frames.
pub fn encode_jpeg_baseline_gpu_batch<T, A>(
    tiles: &[T],
    options: JpegEncodeOptions,
    adapter: &mut A,
) -> Result<Vec<EncodedJpeg>, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    encode_jpeg_baseline_gpu_batch_with_external_live(tiles, options, adapter, 0)
}

/// Encode a resident batch while charging adapter owners already live.
#[doc(hidden)]
pub fn encode_jpeg_baseline_gpu_batch_with_external_live<T, A>(
    tiles: &[T],
    options: JpegEncodeOptions,
    adapter: &mut A,
    external_live_bytes: usize,
) -> Result<Vec<EncodedJpeg>, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    if tiles.is_empty() {
        return Ok(Vec::new());
    }
    let tables = baseline_encode_tables(options)?;
    GpuBatchAllocationBudget::new(tiles.len()).map_err(A::Error::from)?;
    let mut metadata_live_bytes = external_live_bytes;
    let mut gpu_tiles =
        try_encode_metadata_vec(tiles.len(), &mut metadata_live_bytes).map_err(A::Error::from)?;
    for &tile in tiles {
        gpu_tiles.push(adapter.gpu_tile(tile)?);
    }
    let mut encoded =
        try_encode_metadata_vec(tiles.len(), &mut metadata_live_bytes).map_err(A::Error::from)?;

    let backend = adapter.backend();
    checked_gpu_batch_live_bytes(
        &gpu_tiles,
        tiles,
        options,
        backend,
        tables.sampling,
        metadata_live_bytes,
        |tile| adapter.source_key(tile),
    )
    .map_err(|error| adapter.map_plan_error(error))?;
    let mut start = 0usize;
    while start < tiles.len() {
        let end = same_source_buffer_batch_end(tiles, start, |tile| adapter.source_key(tile));
        let prior_frame_bytes = encoded_frame_capacity_bytes(&encoded)?;
        let external_live_bytes =
            checked_encode_host_live_bytes([metadata_live_bytes, prior_frame_bytes])?;
        if end - start == 1 {
            encoded.push(encode_jpeg_baseline_gpu_tile_with_tables(
                tiles[start],
                gpu_tiles[start],
                options,
                &tables,
                adapter,
                external_live_bytes,
            )?);
            start = end;
            continue;
        }
        encode_same_source_group(
            &tiles[start..end],
            &gpu_tiles[start..end],
            options,
            &tables,
            adapter,
            &mut encoded,
            external_live_bytes,
        )?;
        start = end;
    }
    Ok(encoded)
}
