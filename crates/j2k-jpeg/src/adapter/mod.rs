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
    encode_jpeg_baseline_gpu_batch_with_external_live, encode_jpeg_baseline_gpu_tile,
    encode_jpeg_baseline_gpu_tile_with_external_live, preflight_jpeg_baseline_gpu_encode_tile,
    JpegBaselineEncodeTables, JpegBaselineGpuEncodeBatchPlan, JpegBaselineGpuEncodeError,
    JpegBaselineGpuEncodeHostAdapter, JpegBaselineGpuEncodeParams, JpegBaselineGpuEncodeTile,
    JpegBaselineGpuEncodeTilePlan, JpegBaselineHuffmanTable, JpegBaselineSampling,
    JPEG_BASELINE_ZIGZAG,
};
pub(crate) use baseline_encode::{
    assemble_jpeg_baseline_frame_with_quant_tables, checked_cpu_encode_live_bytes,
    checked_encode_host_live_bytes, cpu_owned_plane_capacity_limit,
    jpeg_baseline_entropy_capacity_bytes, jpeg_baseline_entropy_capacity_for_mcus,
    validate_jpeg_baseline_dimensions, validate_jpeg_baseline_restart_interval,
};
pub use device_plan::{
    build_device_plan, summarize_device_batch, DeviceBatchSummary, DeviceComponentPlan,
    DeviceDecodePlan,
};
pub use fast_packet::{
    build_fast420_packet, build_fast422_packet, build_fast444_packet, build_gray_packet,
    classify_color_fast_packet_family, FastPacketError, JpegCachedPlan, JpegCachedPlanBuildError,
    JpegCanonicalHuffmanTable, JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1,
    JpegFast444PacketV1, JpegFastPacket, JpegFastPacketFamily, JpegFastPacketState,
    JpegGrayPacketV1, JpegHuffmanTable, JpegPlanCache, JpegPlanCacheDiagnostics,
    JpegPlanCacheError, JpegPlanCacheInsert, SharedJpegFastPacket, SharedJpegInput, TableKind,
    DEFAULT_JPEG_PLAN_CACHE_ENTRIES, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES,
};

/// Return the original JPEG byte stream owned by a decoder.
pub fn decoder_bytes<'a>(decoder: &'a Decoder<'a>) -> &'a [u8] {
    decoder.bytes
}

/// Return allocator-reported retained bytes for one prepared decoder graph.
///
/// # Errors
///
/// Returns a typed JPEG invariant or allocation-size error.
#[doc(hidden)]
pub fn decoder_retained_allocation_bytes(decoder: &Decoder<'_>) -> Result<usize, crate::JpegError> {
    device_plan::retained_decoder_allocation_bytes(decoder)
}
