// SPDX-License-Identifier: MIT OR Apache-2.0

//! Public adapter-facing JPEG planning APIs.
//!
//! GPU and device-output adapter crates use this module to build validated
//! decode plans without depending on private codec internals.

mod baseline_encode;
mod device_plan;
mod fast_packet;

use crate::Decoder;

pub use crate::internal::checkpoint::DeviceCheckpoint;
pub use baseline_encode::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, encode_jpeg_baseline_gpu_batch,
    encode_jpeg_baseline_gpu_tile, JpegBaselineEncodeTables, JpegBaselineGpuEncodeBatchPlan,
    JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeParams,
    JpegBaselineGpuEncodeTile, JpegBaselineGpuEncodeTilePlan, JpegBaselineHuffmanTable,
    JpegBaselineSampling, JPEG_BASELINE_ZIGZAG,
};
pub(crate) use baseline_encode::{
    assemble_jpeg_baseline_frame_with_quant_tables, validate_jpeg_baseline_dimensions,
    validate_jpeg_baseline_restart_interval,
};
pub use device_plan::{
    build_device_plan, summarize_device_batch, DeviceBatchSummary, DeviceComponentPlan,
    DeviceDecodePlan,
};
pub use fast_packet::{
    build_fast420_packet, build_fast422_packet, build_fast444_packet, build_gray_packet,
    FastPacketError, JpegCanonicalHuffmanTable, JpegEntropyCheckpointV1, JpegFast420PacketV1,
    JpegFast422PacketV1, JpegFast444PacketV1, JpegGrayPacketV1, JpegHuffmanTable, TableKind,
};

/// Return the original JPEG byte stream owned by a decoder.
pub fn decoder_bytes<'a>(decoder: &'a Decoder<'a>) -> &'a [u8] {
    decoder.bytes
}
