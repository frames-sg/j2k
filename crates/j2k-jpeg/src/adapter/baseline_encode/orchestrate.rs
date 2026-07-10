// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public resident GPU baseline JPEG encode orchestration.

use alloc::vec::Vec;

use super::frame::assemble_jpeg_baseline_frame;
use super::planning::{
    jpeg_baseline_gpu_encode_batch_plan, jpeg_baseline_gpu_encode_tile_plan,
    same_source_buffer_batch_end,
};
use super::tables::baseline_encode_tables;
use super::types::{JpegBaselineEncodeTables, JpegBaselineGpuEncodeHostAdapter};
use crate::encoder::{EncodedJpeg, JpegEncodeError, JpegEncodeOptions};

/// Encode one resident GPU tile through a backend adapter.
pub fn encode_jpeg_baseline_gpu_tile<T, A>(
    tile: T,
    options: JpegEncodeOptions,
    adapter: &mut A,
) -> Result<EncodedJpeg, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let tables = baseline_encode_tables(options)?;
    encode_jpeg_baseline_gpu_tile_with_tables(tile, options, &tables, adapter)
}

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
    if tiles.is_empty() {
        return Ok(Vec::new());
    }

    let tables = baseline_encode_tables(options)?;
    let mut encoded = Vec::with_capacity(tiles.len());
    let mut start = 0usize;
    while start < tiles.len() {
        let end = same_source_buffer_batch_end(tiles, start, |tile| adapter.source_key(tile));
        if end - start == 1 {
            encoded.push(encode_jpeg_baseline_gpu_tile_with_tables(
                tiles[start],
                options,
                &tables,
                adapter,
            )?);
            start = end;
            continue;
        }

        let gpu_tiles = tiles[start..end]
            .iter()
            .copied()
            .map(|tile| adapter.gpu_tile(tile))
            .collect::<Result<Vec<_>, _>>()?;
        let plan = jpeg_baseline_gpu_encode_batch_plan(
            &gpu_tiles,
            options,
            adapter.backend(),
            tables.sampling,
        )
        .map_err(|error| adapter.map_plan_error(error))?;
        let entropy_chunks = adapter.encode_batch_entropy(&tiles[start..end], &tables, plan)?;
        if entropy_chunks.len() != gpu_tiles.len() {
            return Err(JpegEncodeError::Internal(
                "GPU JPEG baseline batch returned the wrong number of entropy chunks".into(),
            )
            .into());
        }
        for (gpu_tile, entropy) in gpu_tiles.iter().zip(entropy_chunks.iter()) {
            encoded.push(assemble_jpeg_baseline_frame(
                entropy,
                gpu_tile.output_width,
                gpu_tile.output_height,
                &tables,
                options,
                adapter.backend(),
            )?);
        }
        start = end;
    }
    Ok(encoded)
}

fn encode_jpeg_baseline_gpu_tile_with_tables<T, A>(
    tile: T,
    options: JpegEncodeOptions,
    tables: &JpegBaselineEncodeTables,
    adapter: &mut A,
) -> Result<EncodedJpeg, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let gpu_tile = adapter.gpu_tile(tile)?;
    let plan = jpeg_baseline_gpu_encode_tile_plan(
        gpu_tile,
        options,
        adapter.backend(),
        tables.sampling,
        0,
        0,
    )
    .map_err(|error| adapter.map_plan_error(error))?;
    let entropy = adapter.encode_tile_entropy(tile, tables, plan)?;
    assemble_jpeg_baseline_frame(
        &entropy,
        gpu_tile.output_width,
        gpu_tile.output_height,
        tables,
        options,
        adapter.backend(),
    )
    .map_err(Into::into)
}
