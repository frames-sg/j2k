// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    J2kLosslessEncodeOptions, MetalLosslessEncodeBatchRequest, MetalLosslessEncodeOutcome,
    SubmittedJ2kLosslessMetalBufferEncodeBatch, SubmittedJ2kLosslessMetalEncodeBatch,
};

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted host-byte batch encode on non-macOS.
pub fn submit_lossless_batch(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalEncodeBatch, crate::Error> {
    let _ = (request, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for submitted Metal-buffer batch encode on non-macOS.
pub fn submit_lossless_batch_to_metal(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<SubmittedJ2kLosslessMetalBufferEncodeBatch, crate::Error> {
    let _ = (request, options, session);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for reported batch encode on non-macOS.
#[doc(hidden)]
pub fn encode_lossless_batch_with_report(
    request: MetalLosslessEncodeBatchRequest<'_, '_>,
    options: &J2kLosslessEncodeOptions,
    session: &crate::MetalBackendSession,
) -> Result<Vec<MetalLosslessEncodeOutcome>, crate::Error> {
    let _ = (request, options, session);
    Err(crate::Error::MetalUnavailable)
}
