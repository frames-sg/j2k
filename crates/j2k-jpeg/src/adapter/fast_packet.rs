// SPDX-License-Identifier: MIT OR Apache-2.0

mod allocation;
mod build;
mod cache;
mod checkpoints;
mod entropy;
mod error;
mod family;
mod header;
mod types;

pub use build::{
    build_fast420_packet, build_fast422_packet, build_fast444_packet, build_gray_packet,
};
pub use cache::{
    JpegCachedPlan, JpegCachedPlanBuildError, JpegFastPacket, JpegFastPacketState, JpegPlanCache,
    JpegPlanCacheDiagnostics, JpegPlanCacheError, JpegPlanCacheInsert, SharedJpegFastPacket,
    SharedJpegInput, DEFAULT_JPEG_PLAN_CACHE_ENTRIES, DEFAULT_JPEG_PLAN_CACHE_HOST_BYTES,
};
pub use error::{FastPacketError, TableKind};
pub use family::{classify_color_fast_packet_family, JpegFastPacketFamily};
pub use types::{
    JpegCanonicalHuffmanTable, JpegEntropyCheckpointV1, JpegFast420PacketV1, JpegFast422PacketV1,
    JpegFast444PacketV1, JpegGrayPacketV1, JpegHuffmanTable,
};

#[cfg(test)]
mod tests;
