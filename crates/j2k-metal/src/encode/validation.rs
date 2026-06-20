// SPDX-License-Identifier: Apache-2.0

use j2k::J2kLosslessSamples;
use j2k_core::PixelFormat;

use super::MetalLosslessEncodeTile;

pub(super) fn validation_pixel_format(
    samples: J2kLosslessSamples<'_>,
) -> Result<PixelFormat, crate::Error> {
    match (samples.components, samples.bit_depth) {
        (1, 1..=8) => Ok(PixelFormat::Gray8),
        (3, 1..=8) => Ok(PixelFormat::Rgb8),
        (1, 9..=16) => Ok(PixelFormat::Gray16),
        (3, 9..=16) => Ok(PixelFormat::Rgb16),
        _ => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal validation supports only grayscale or RGB samples up to 16 bits",
        }),
    }
}

pub(super) fn lossless_sample_shape(format: PixelFormat) -> Result<(u8, u8), crate::Error> {
    match format {
        PixelFormat::Gray8 => Ok((1, 8)),
        PixelFormat::Rgb8 => Ok((3, 8)),
        PixelFormat::Gray16 => Ok((1, 16)),
        PixelFormat::Rgb16 => Ok((3, 16)),
        PixelFormat::Rgba8 | PixelFormat::Rgba16 => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode from RGBA tiles requires explicit alpha handling",
        }),
        _ => Err(crate::Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode received an unknown pixel format",
        }),
    }
}

pub(super) fn validate_metal_encode_tile(
    tile: MetalLosslessEncodeTile<'_>,
) -> Result<(), crate::Error> {
    if tile.width == 0 || tile.height == 0 || tile.output_width == 0 || tile.output_height == 0 {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode tile dimensions must be nonzero".to_string(),
        });
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode input tile exceeds output tile dimensions".to_string(),
        });
    }
    let row_bytes = tile
        .width
        .checked_mul(tile.format.bytes_per_pixel() as u32)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode row byte count overflow".to_string(),
        })? as usize;
    if tile.pitch_bytes < row_bytes {
        return Err(crate::Error::MetalKernel {
            message: "J2K Metal encode tile pitch is shorter than one row".to_string(),
        });
    }
    let required_end = tile
        .byte_offset
        .checked_add(
            tile.pitch_bytes
                .checked_mul(tile.height.saturating_sub(1) as usize)
                .and_then(|prefix| prefix.checked_add(row_bytes))
                .ok_or_else(|| crate::Error::MetalKernel {
                    message: "J2K Metal encode input byte range overflow".to_string(),
                })?,
        )
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal encode input byte range overflow".to_string(),
        })?;
    let buffer_len =
        usize::try_from(tile.buffer.length()).map_err(|_| crate::Error::MetalKernel {
            message: "J2K Metal encode buffer length exceeds usize".to_string(),
        })?;
    if required_end > buffer_len {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal encode input byte range exceeds buffer length: need {required_end}, buffer has {buffer_len}"
            ),
        });
    }
    Ok(())
}

pub(super) fn validate_padded_contiguous_metal_encode_tile(
    tile: MetalLosslessEncodeTile<'_>,
    bytes_per_pixel: usize,
) -> Result<(), crate::Error> {
    if tile.width != tile.output_width || tile.height != tile.output_height {
        return Err(crate::Error::MetalKernel {
            message:
                "J2K Metal no-copy encode requires input dimensions to match output dimensions"
                    .to_string(),
        });
    }
    let expected_pitch = (tile.output_width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| crate::Error::MetalKernel {
            message: "J2K Metal no-copy encode pitch overflow".to_string(),
        })?;
    if tile.pitch_bytes != expected_pitch {
        return Err(crate::Error::MetalKernel {
            message: format!(
                "J2K Metal no-copy encode requires contiguous rows: expected pitch {expected_pitch}, got {}",
                tile.pitch_bytes
            ),
        });
    }
    Ok(())
}
