// SPDX-License-Identifier: MIT OR Apache-2.0

//! Baseline JPEG adapter planning, orchestration, and frame assembly.

mod frame;
mod orchestrate;
mod planning;
mod tables;
mod types;
mod validation;

pub use self::frame::assemble_jpeg_baseline_frame;
pub(crate) use self::frame::assemble_jpeg_baseline_frame_with_quant_tables;
pub use self::orchestrate::{encode_jpeg_baseline_gpu_batch, encode_jpeg_baseline_gpu_tile};
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
