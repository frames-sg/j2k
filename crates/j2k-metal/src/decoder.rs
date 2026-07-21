// SPDX-License-Identifier: MIT OR Apache-2.0

//! JPEG 2000 decode requests, routing, execution, and codec adapters.

mod adapters;
mod core;
mod direct_paths;
mod request;
mod routes;
mod surface;

#[cfg(target_os = "macos")]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;

#[cfg(target_os = "macos")]
use crate::{Error, MetalBackendSession, MetalDirectFallbackReason};

pub use adapters::Codec;
pub use core::J2kDecoder;
pub use request::{
    DecodeOperation, DecodeRouteReport, DecodeSurfaceWithReport, MetalDecodeOp, MetalDecodeRequest,
};

#[cfg(target_os = "macos")]
pub(crate) use direct_paths::{
    decode_full_color_batch_direct_to_device_routed,
    decode_full_grayscale_batch_direct_to_device_routed, is_direct_runtime_fallback_error,
};
#[cfg(target_os = "macos")]
pub(crate) fn prepare_full_grayscale_direct_plan_with_session(
    input: &[u8],
    fmt: PixelFormat,
    session: &MetalBackendSession,
) -> Result<Arc<crate::compute::PreparedDirectGrayscalePlan>, Error> {
    let mut decoder = J2kDecoder::new(input)?;
    decoder
        .ensure_prepared_direct_gray_plan_with_session(fmt, session)?
        .ok_or_else(|| Error::MetalDirectFallback {
            message: format!(
                "explicit J2K MetalDirect destination could not build a full grayscale plan; fmt={fmt:?}"
            ),
            reason: MetalDirectFallbackReason::UnsupportedPlan,
        })
}

#[cfg(test)]
mod tests;
