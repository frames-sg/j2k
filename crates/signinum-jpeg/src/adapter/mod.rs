// SPDX-License-Identifier: Apache-2.0

//! Public adapter-facing JPEG planning APIs.
//!
//! GPU and device-output adapter crates use this module to build validated
//! decode plans without depending on private codec internals.

mod baseline_encode;
mod device_plan;
pub mod metal_fast420;

use crate::Decoder;

pub use crate::internal::checkpoint::DeviceCheckpoint;
pub use baseline_encode::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, jpeg_baseline_entropy_capacity_bytes,
    jpeg_baseline_sampling_for, validate_jpeg_baseline_dimensions,
    validate_jpeg_baseline_restart_interval, JpegBaselineEncodeTables, JpegBaselineHuffmanTable,
    JpegBaselineSampling, JPEG_BASELINE_ZIGZAG,
};
pub use device_plan::{
    build_device_plan, summarize_device_batch, DeviceBatchSummary, DeviceComponentPlan,
    DeviceDecodePlan,
};
pub use metal_fast420::{
    build_metal_fast420_packet, build_metal_fast420_packet_for_decoder, build_metal_fast422_packet,
    build_metal_fast422_packet_for_decoder, build_metal_fast444_packet,
    build_metal_fast444_packet_for_decoder, build_metal_gray_packet,
    build_metal_gray_packet_for_decoder, JpegMetalEntropyCheckpointV1, JpegMetalFast420PacketV1,
    JpegMetalFast422PacketV1, JpegMetalFast444PacketV1, JpegMetalGrayPacketV1,
    MetalFast420PacketError, MetalHuffmanTable,
};

pub fn decoder_bytes<'a>(decoder: &'a Decoder<'a>) -> &'a [u8] {
    decoder.bytes
}
