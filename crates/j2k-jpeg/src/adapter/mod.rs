// SPDX-License-Identifier: Apache-2.0

//! Public adapter-facing JPEG planning APIs.
//!
//! GPU and device-output adapter crates use this module to build validated
//! decode plans without depending on private codec internals.

mod baseline_encode;
mod device_plan;
/// Backend-neutral fast-path packet builders and packet types.
pub mod fast_packet;

use crate::Decoder;

pub use crate::internal::checkpoint::DeviceCheckpoint;
pub(crate) use baseline_encode::assemble_jpeg_baseline_frame_with_quant_tables;
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
pub use fast_packet::{
    build_fast420_packet, build_fast420_packet_for_decoder, build_fast422_packet,
    build_fast422_packet_for_decoder, build_fast444_packet, build_fast444_packet_for_decoder,
    build_gray_packet, build_gray_packet_for_decoder, FastPacketError, JpegEntropyCheckpointV1,
    JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1, JpegGrayPacketV1,
    JpegHuffmanTable,
};

/// Return the original JPEG byte stream owned by a decoder.
pub fn decoder_bytes<'a>(decoder: &'a Decoder<'a>) -> &'a [u8] {
    decoder.bytes
}
