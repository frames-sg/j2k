// SPDX-License-Identifier: Apache-2.0

//! Stride-aware output writers. One implementor per internal JPEG output mode;
//! the decode loop is generic over `<W: OutputWriter>` and monomorphized at
//! each call site so there is no dynamic dispatch on the per-pixel hot path.

use crate::error::JpegError;
use signinum_core::{validate_strided_output_buffer, BufferError, PixelFormat};

pub(crate) mod gray8;
pub(crate) mod rgb8;
pub(crate) mod rgba8;

pub(crate) use gray8::Gray8Writer;
pub(crate) use rgb8::Rgb8Writer;
pub(crate) use rgba8::Rgba8Writer;

/// A writer that can expose one or two mutable interleaved RGB rows so the
/// decoder can fill final output bytes directly.
pub(crate) trait InterleavedRgbWriter {
    fn with_rgb_rows<R, F>(&mut self, y: u32, row_count: usize, fill: F) -> Result<R, JpegError>
    where
        F: FnOnce(&mut [u8], Option<&mut [u8]>) -> Result<R, JpegError>;
}

/// A destination for decoded pixel rows. Each writer carries a mutable slice
/// of the caller's output buffer and the stride in bytes between rows.
pub(crate) trait OutputWriter {
    /// Write one full-width row of RGB data at output row `y`.
    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), JpegError>;

    /// Write one full-width row of YCbCr data at output row `y`.
    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
    ) -> Result<(), JpegError>;

    /// Write one full-width row of grayscale data.
    fn write_gray_row(&mut self, y: u32, gray_row: &[u8]) -> Result<(), JpegError>;
}

/// Validate that the caller's `out`/`stride` pair is large enough to hold an
/// `image_width × image_height` image at `bytes_per_pixel`.
pub(crate) fn validate_buffer(
    out: &[u8],
    stride: usize,
    image_width: u32,
    image_height: u32,
    bytes_per_pixel: usize,
) -> Result<(), JpegError> {
    let fmt = match bytes_per_pixel {
        1 => PixelFormat::Gray8,
        3 => PixelFormat::Rgb8,
        4 => PixelFormat::Rgba8,
        _ => {
            return Err(JpegError::OutputBufferTooSmall {
                required: usize::MAX,
                provided: out.len(),
            })
        }
    };
    validate_strided_output_buffer((image_width, image_height), out.len(), stride, fmt).map_err(
        |err| match err {
            BufferError::StrideTooSmall { row_bytes, stride } => JpegError::InvalidStride {
                stride,
                row: row_bytes,
            },
            BufferError::OutputTooSmall { required, have } => JpegError::OutputBufferTooSmall {
                required,
                provided: have,
            },
            BufferError::OutputExceedsCapacity { lower_bound, have } => {
                JpegError::OutputBufferTooSmall {
                    required: lower_bound,
                    provided: have,
                }
            }
            BufferError::SizeOverflow { .. } => JpegError::OutputBufferTooSmall {
                required: usize::MAX,
                provided: out.len(),
            },
            BufferError::InputTooSmall { .. }
            | BufferError::StrideNotAligned { .. }
            | BufferError::SampleTypeMismatch { .. }
            | _ => JpegError::OutputBufferTooSmall {
                required: usize::MAX,
                provided: out.len(),
            },
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_buffer_accepts_tight_fit() {
        let out = alloc::vec![0u8; 16 * 16 * 3];
        validate_buffer(&out, 16 * 3, 16, 16, 3).unwrap();
    }

    #[test]
    fn validates_buffer_accepts_padded_stride() {
        let out = alloc::vec![0u8; 16 * 64];
        validate_buffer(&out, 64, 16, 16, 3).unwrap();
    }

    #[test]
    fn validates_buffer_rejects_stride_less_than_row_width() {
        let out = alloc::vec![0u8; 16 * 16 * 3];
        let err = validate_buffer(&out, 16, 16, 16, 3).unwrap_err();
        assert!(matches!(err, JpegError::InvalidStride { .. }));
    }

    #[test]
    fn validates_buffer_rejects_undersized_output() {
        let out = alloc::vec![0u8; 10];
        let err = validate_buffer(&out, 16 * 3, 16, 16, 3).unwrap_err();
        assert!(matches!(err, JpegError::OutputBufferTooSmall { .. }));
    }
}
