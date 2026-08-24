// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal-backed JPEG decode and encode adapters.
//!
//! The crate exposes the same CPU-visible JPEG decode surface as
//! `j2k-jpeg`, with optional Metal-resident surfaces and batch submission
//! helpers on macOS. Non-macOS builds return `Error::MetalUnavailable`.
//!
//! The 0.9 expert surface uses `objc2-metal` directly. Owned Metal objects are
//! retained protocol objects and borrowed buffers or textures are
//! protocol-object references; the former `metal-rs` API is not preserved.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]
#[cfg(target_os = "macos")]
mod abi;
mod batch;
mod batch_allocation;
#[cfg(target_os = "macos")]
mod buffers;
mod codec;
mod codec_batch;
#[cfg(target_os = "macos")]
mod compute;
mod decode_request;
mod decode_surface;
mod decoder;
mod encode;
mod error;
mod fast_packets;
#[cfg(target_os = "macos")]
mod metal_types;
mod plan_owner_ledger;
#[cfg(target_os = "macos")]
mod resident_batch;
mod routing;
mod session;
mod surface;
mod tile_batch;
mod viewport;
#[cfg(test)]
mod viewport_tests;
#[cfg(target_os = "macos")]
pub use codec_batch::{
    MetalBufferBatchTarget, MetalTextureBatchTarget, Rgb8MetalBatchOp, Rgb8MetalBatchRequest,
    Rgb8MetalBatchSource,
};
pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};
pub use decoder::Decoder;
pub use encode::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile,
};
pub use error::Error;
pub(crate) use fast_packets::JpegFastPackets;
use j2k_core::ImageCodec;
pub use j2k_core::SurfaceResidency;
#[cfg(test)]
use j2k_core::{BackendKind, BackendRequest, Downscale, PixelFormat, Rect};
pub(crate) use j2k_jpeg::adapter::{SharedJpegFastPacket, SharedJpegInput};
use j2k_jpeg::Warning as CpuWarning;
#[cfg(test)]
use j2k_jpeg::{Decoder as CpuDecoder, ScratchPool as CpuScratchPool};
#[cfg(target_os = "macos")]
pub(crate) use resident_batch::report_required_output_dimensions;
#[cfg(target_os = "macos")]
pub use resident_batch::JpegMetalResidentBatchReport;
pub use session::{MetalBackendSession, MetalSession};
pub(crate) use surface::Storage;
pub use surface::Surface;
#[cfg(target_os = "macos")]
pub use surface::{
    MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalTextureTile, ResidentPrivateJpegTile,
};
pub use tile_batch::JpegTileBatch;
pub use viewport::{
    choose_viewport_surface_strategy, decode_viewport_to_surface, is_contiguous_viewport_workload,
    suggest_viewport_workload, viewport_source_bounds, ViewportSurfaceStrategy, ViewportTile,
    ViewportWorkload,
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// JPEG codec marker used by J2K's generic decode traits.
pub struct Codec;

#[doc(hidden)]
impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = ScratchPool;
}

#[cfg(target_os = "macos")]
pub use decode_surface::decode_rgb8_batch_to_device_with_session;
pub(crate) use decode_surface::{
    choose_route, decode_compatible_batch_with_session, decode_surface_from_decoder,
    decode_surface_from_shared_input, upload_surface,
};
#[cfg(target_os = "macos")]
pub(crate) use decode_surface::{reject_cpu_staged_metal_upload, scaled_dims};

pub use j2k_jpeg::{
    DecoderContext, Downscale as JpegDownscale, Info, PixelFormat as JpegPixelFormat,
    Rect as JpegRectPublic, ScratchPool,
};

#[cfg(test)]
mod tests;
