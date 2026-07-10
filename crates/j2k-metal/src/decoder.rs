// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 decode requests, routing, execution, and codec adapters.

mod adapters;
mod core;
mod direct_paths;
mod request;
mod routes;
mod surface;

pub use adapters::Codec;
pub use core::J2kDecoder;
pub use request::{
    DecodeOperation, DecodeRouteReport, DecodeSurfaceWithReport, MetalDecodeOp, MetalDecodeRequest,
};

#[cfg(target_os = "macos")]
pub(crate) use direct_paths::{
    decode_full_color_batch_direct_to_device, decode_full_grayscale_batch_direct_to_device,
    is_direct_runtime_fallback_error,
};

#[cfg(test)]
mod tests;
