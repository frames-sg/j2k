// SPDX-License-Identifier: MIT OR Apache-2.0

//! Input, option, pixel-format, and resident-buffer validation.

use super::types::{JpegBaselineGpuEncodeError, JpegBaselineGpuEncodeTile};
use crate::encoder::{JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSubsampling};
use crate::PixelFormat;

/// Validate that dimensions can be represented in baseline JPEG markers.
pub(crate) fn validate_jpeg_baseline_dimensions(
    width: u32,
    height: u32,
) -> Result<(), JpegEncodeError> {
    if width == 0 || height == 0 {
        return Err(JpegEncodeError::EmptyDimensions);
    }
    if width > u32::from(u16::MAX) || height > u32::from(u16::MAX) {
        return Err(JpegEncodeError::DimensionsTooLarge { width, height });
    }
    Ok(())
}

/// Validate a user-provided restart interval.
pub(crate) fn validate_jpeg_baseline_restart_interval(
    restart_interval: Option<u16>,
) -> Result<(), JpegEncodeError> {
    if restart_interval == Some(0) {
        return Err(JpegEncodeError::InvalidRestartInterval);
    }
    Ok(())
}

/// Validate resident GPU baseline JPEG encode tile metadata.
pub(super) fn validate_jpeg_baseline_gpu_encode_tile(
    tile: JpegBaselineGpuEncodeTile,
    options: JpegEncodeOptions,
    expected_backend: JpegBackend,
) -> Result<(), JpegBaselineGpuEncodeError> {
    match options.backend {
        JpegBackend::Auto => {}
        requested if requested == expected_backend => {}
        requested => {
            return Err(JpegBaselineGpuEncodeError::UnsupportedBackend {
                requested,
                expected: expected_backend,
            });
        }
    }

    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    validate_jpeg_baseline_dimensions(tile.output_width, tile.output_height)?;
    if tile.width == 0 || tile.height == 0 {
        return Err(JpegEncodeError::EmptyDimensions.into());
    }
    if tile.width > tile.output_width || tile.height > tile.output_height {
        return Err(JpegBaselineGpuEncodeError::InputExceedsOutputDimensions);
    }

    let bytes_per_pixel = jpeg_baseline_gpu_encode_bytes_per_pixel(tile.format, options)?;
    let width = usize::try_from(tile.width)
        .map_err(|_| JpegBaselineGpuEncodeError::RowByteCountOverflow)?;
    let row_bytes = width
        .checked_mul(bytes_per_pixel)
        .ok_or(JpegBaselineGpuEncodeError::RowByteCountOverflow)?;
    if tile.pitch_bytes < row_bytes {
        return Err(JpegBaselineGpuEncodeError::PitchTooShort {
            row_bytes,
            pitch_bytes: tile.pitch_bytes,
        });
    }
    let height =
        usize::try_from(tile.height).map_err(|_| JpegBaselineGpuEncodeError::InputRangeOverflow)?;
    let last_row = height
        .checked_sub(1)
        .and_then(|row| row.checked_mul(tile.pitch_bytes))
        .ok_or(JpegBaselineGpuEncodeError::InputRangeOverflow)?;
    let required_end = tile
        .byte_offset
        .checked_add(last_row)
        .and_then(|offset| offset.checked_add(row_bytes))
        .ok_or(JpegBaselineGpuEncodeError::InputRangeOverflow)?;
    if required_end > tile.buffer_len {
        return Err(JpegBaselineGpuEncodeError::InputRangeExceedsBuffer {
            required_end,
            buffer_len: tile.buffer_len,
        });
    }

    Ok(())
}

pub(super) fn jpeg_baseline_gpu_encode_bytes_per_pixel(
    format: PixelFormat,
    options: JpegEncodeOptions,
) -> Result<usize, JpegBaselineGpuEncodeError> {
    match (format, options.subsampling) {
        (PixelFormat::Gray8, JpegSubsampling::Gray) => Ok(1),
        (
            PixelFormat::Rgb8,
            JpegSubsampling::Ybr444 | JpegSubsampling::Ybr422 | JpegSubsampling::Ybr420,
        ) => Ok(3),
        (PixelFormat::Gray8 | PixelFormat::Rgb8, _) => {
            Err(JpegBaselineGpuEncodeError::IncompatibleSubsampling {
                subsampling: options.subsampling,
                samples: if format == PixelFormat::Gray8 {
                    "Gray8"
                } else {
                    "Rgb8"
                },
            })
        }
        _ => Err(JpegBaselineGpuEncodeError::UnsupportedPixelFormat { format }),
    }
}

pub(super) fn jpeg_baseline_gpu_encode_format_abi(
    format: PixelFormat,
) -> Result<u32, JpegBaselineGpuEncodeError> {
    match format {
        PixelFormat::Gray8 => Ok(0),
        PixelFormat::Rgb8 => Ok(1),
        _ => Err(JpegBaselineGpuEncodeError::UnsupportedPixelFormat { format }),
    }
}
