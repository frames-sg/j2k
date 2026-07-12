// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline JPEG adapter planning, orchestration, and frame assembly.

mod allocation;
mod frame;
mod orchestrate;
mod planning;
mod tables;
mod types;
mod validation;

pub(crate) use self::allocation::{
    checked_cpu_encode_live_bytes, checked_encode_host_live_bytes, cpu_owned_plane_capacity_limit,
};
pub use self::frame::assemble_jpeg_baseline_frame;
pub(crate) use self::frame::assemble_jpeg_baseline_frame_with_quant_tables;
pub use self::orchestrate::{
    encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_batch_with_external_live,
    encode_jpeg_baseline_gpu_tile, encode_jpeg_baseline_gpu_tile_with_external_live,
};
pub use self::planning::preflight_jpeg_baseline_gpu_encode_tile;
pub(crate) use self::planning::{
    jpeg_baseline_entropy_capacity_bytes, jpeg_baseline_entropy_capacity_for_mcus,
};
pub use self::tables::{baseline_encode_tables, JPEG_BASELINE_ZIGZAG};
pub use self::types::{
    JpegBaselineEncodeTables, JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError,
    JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile,
    JpegBaselineGpuEncodeTilePlan, JpegBaselineHuffmanTable, JpegBaselineSampling,
};
pub(crate) use self::validation::{
    validate_jpeg_baseline_dimensions, validate_jpeg_baseline_restart_interval,
};

#[cfg(test)]
mod tests;
