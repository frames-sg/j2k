// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::J2kLosslessSamples;

#[cfg(target_os = "macos")]
use super::validation::validation_pixel_format;
#[cfg(target_os = "macos")]
use crate::compute;
#[cfg(target_os = "macos")]
use j2k_core::DeviceSurface;

#[cfg(target_os = "macos")]
/// Validate a lossless codestream by decoding it through the default Metal session.
pub fn validate_lossless_roundtrip_on_metal(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), crate::Error> {
    let session = crate::MetalBackendSession::system_default()?;
    validate_lossless_roundtrip_on_metal_with_session(samples, codestream, &session)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for Metal roundtrip validation on non-macOS.
pub fn validate_lossless_roundtrip_on_metal(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
) -> Result<(), crate::Error> {
    let _ = (samples, codestream);
    Err(crate::Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
/// Validate a lossless codestream by decoding it through a provided Metal session.
pub fn validate_lossless_roundtrip_on_metal_with_session(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let fmt = validation_pixel_format(samples)?;
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_request_to_device_with_session(
        crate::MetalDecodeRequest::full(fmt, j2k_core::BackendRequest::Metal),
        session,
    )?;

    if surface.dimensions() != (samples.width, samples.height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation geometry mismatch: expected {}x{}, got {}x{}",
                samples.width,
                samples.height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != fmt {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation format mismatch: expected {:?}, got {:?}",
                fmt,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = samples.width as usize * fmt.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation pitch mismatch: expected {expected_pitch}, got {}",
                surface.pitch_bytes()
            ),
        });
    }
    if surface.byte_len() != samples.data.len() {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal validation length mismatch: expected {} bytes, got {} bytes",
                samples.data.len(),
                surface.byte_len()
            ),
        });
    }

    let (buffer, byte_offset) =
        surface
            .metal_buffer()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffer_matches_bytes(samples.data, buffer, byte_offset, session)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for session validation on non-macOS.
pub fn validate_lossless_roundtrip_on_metal_with_session(
    samples: J2kLosslessSamples<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let _ = (samples, codestream, session);
    Err(crate::Error::MetalUnavailable)
}
