// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public resident GPU single-tile baseline JPEG orchestration.

use super::allocation::{checked_encode_host_live_bytes, checked_gpu_tile_live_bytes};
use super::frame::assemble_jpeg_baseline_frame;
use super::planning::jpeg_baseline_gpu_encode_tile_plan;
use super::tables::baseline_encode_tables;
use super::types::{
    JpegBaselineEncodeTables, JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeTile,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::encoder::{EncodedJpeg, JpegEncodeError, JpegEncodeOptions};

mod batch;

pub use self::batch::{
    encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_batch_with_external_live,
};

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
    encode_jpeg_baseline_gpu_tile_with_external_live(tile, options, adapter, 0)
}

/// Encode one resident tile while charging adapter owners already live.
#[doc(hidden)]
pub fn encode_jpeg_baseline_gpu_tile_with_external_live<T, A>(
    tile: T,
    options: JpegEncodeOptions,
    adapter: &mut A,
    external_live_bytes: usize,
) -> Result<EncodedJpeg, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let tables = baseline_encode_tables(options)?;
    let gpu_tile = adapter.gpu_tile(tile)?;
    encode_jpeg_baseline_gpu_tile_with_tables(
        tile,
        gpu_tile,
        options,
        &tables,
        adapter,
        external_live_bytes,
    )
}

fn encode_jpeg_baseline_gpu_tile_with_tables<T, A>(
    tile: T,
    gpu_tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    tables: &JpegBaselineEncodeTables,
    adapter: &mut A,
    external_live_bytes: usize,
) -> Result<EncodedJpeg, A::Error>
where
    T: Copy,
    A: JpegBaselineGpuEncodeHostAdapter<T>,
{
    let plan = jpeg_baseline_gpu_encode_tile_plan(
        gpu_tile,
        options,
        adapter.backend(),
        tables.sampling,
        0,
        0,
    )
    .map_err(|error| adapter.map_plan_error(error))?;
    let tile_live_bytes = checked_gpu_tile_live_bytes(plan.entropy_capacity)?;
    checked_encode_host_live_bytes([external_live_bytes, tile_live_bytes])?;
    let entropy_capacity = plan.entropy_capacity;
    let entropy = adapter.encode_tile_entropy(tile, tables, plan)?;
    ensure_entropy_output_within_plan(entropy.capacity(), entropy_capacity)?;
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy.len())?;
    checked_encode_host_live_bytes([external_live_bytes, entropy.capacity(), frame_capacity])?;
    let frame = assemble_jpeg_baseline_frame(
        &entropy,
        gpu_tile.output_width,
        gpu_tile.output_height,
        tables,
        options,
        adapter.backend(),
    )
    .map_err(A::Error::from)?;
    checked_encode_host_live_bytes([
        external_live_bytes,
        entropy.capacity(),
        frame.data.capacity(),
    ])?;
    Ok(frame)
}

fn ensure_entropy_output_within_plan(
    returned_capacity: usize,
    planned_capacity: usize,
) -> Result<(), JpegEncodeError> {
    if returned_capacity > planned_capacity {
        return Err(JpegEncodeError::InternalInvariant {
            reason: "GPU JPEG baseline entropy output exceeded its planned capacity",
        });
    }
    Ok(())
}
