// SPDX-License-Identifier: Apache-2.0

#[doc(hidden)]
mod device_plan;
#[doc(hidden)]
pub mod metal_fast420;

use crate::Decoder;

pub use crate::internal::checkpoint::DeviceCheckpoint;
pub use device_plan::{
    build_device_plan, summarize_device_batch, DeviceBatchSummary, DeviceComponentPlan,
    DeviceDecodePlan,
};
pub use metal_fast420::{
    build_metal_fast420_packet, build_metal_fast420_packet_for_decoder, JpegMetalFast420PacketV1,
    MetalFast420PacketError, MetalHuffmanTable,
};

pub fn decoder_bytes<'a>(decoder: &'a Decoder<'a>) -> &'a [u8] {
    decoder.bytes
}
