// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{compute, DeviceSurface, MetalLosslessEncodeTile};

#[cfg(target_os = "macos")]
pub(super) fn validate_lossless_roundtrip_on_metal_tile_with_session(
    tile: MetalLosslessEncodeTile<'_>,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let mut decoder = crate::J2kDecoder::new(codestream)?;
    let surface = decoder.decode_request_to_device_with_session(
        crate::MetalDecodeRequest::full(tile.format, j2k_core::BackendRequest::Metal),
        session,
    )?;
    if surface.dimensions() != (tile.output_width, tile.output_height) {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation geometry mismatch: expected {}x{}, got {}x{}",
                tile.output_width,
                tile.output_height,
                surface.dimensions().0,
                surface.dimensions().1
            ),
        });
    }
    if surface.pixel_format() != tile.format {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal resident validation format mismatch: expected {:?}, got {:?}",
                tile.format,
                surface.pixel_format()
            ),
        });
    }
    let expected_pitch = tile.output_width as usize * tile.format.bytes_per_pixel();
    if surface.pitch_bytes() != expected_pitch || tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal resident validation requires contiguous source and decoded rows"
                .to_string(),
        });
    }
    let byte_len = expected_pitch
        .checked_mul(tile.output_height as usize)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal resident validation byte length overflow".to_string(),
        })?;
    let (decoded_buffer, decoded_offset) =
        surface
            .metal_buffer_trusted()
            .ok_or(crate::Error::UnsupportedMetalRequest {
                reason: "J2K Metal resident validation decode did not return a Metal buffer",
            })?;
    compute::validate_metal_buffers_match(
        tile.buffer,
        tile.byte_offset,
        decoded_buffer,
        decoded_offset,
        byte_len,
        session,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn validate_lossless_roundtrip_on_metal_region_with_session(
    source: MetalLosslessEncodeTile<'_>,
    output_width: u32,
    output_height: u32,
    bytes_per_pixel: usize,
    codestream: &[u8],
    session: &crate::MetalBackendSession,
) -> Result<(), crate::Error> {
    let staged_buffer =
        compute::copy_interleaved_padded_to_shared_buffer(compute::PaddedInterleavedCopy {
            src_buffer: source.buffer,
            src_byte_offset: source.byte_offset,
            src_width: source.width,
            src_height: source.height,
            src_pitch_bytes: source.pitch_bytes,
            dst_width: output_width,
            dst_height: output_height,
            bytes_per_pixel,
            session,
        })?;
    let staged_tile = MetalLosslessEncodeTile {
        buffer: &staged_buffer,
        byte_offset: 0,
        width: output_width,
        height: output_height,
        pitch_bytes: output_width as usize * bytes_per_pixel,
        output_width,
        output_height,
        format: source.format,
    };
    validate_lossless_roundtrip_on_metal_tile_with_session(staged_tile, codestream, session)
}
