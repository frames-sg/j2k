// SPDX-License-Identifier: Apache-2.0

//! Reusable host output buffers for JPEG tile decode.

use alloc::vec::Vec;
use signinum_core::{strided_output_len, BufferError, PixelFormat};

/// Caller-owned reusable host pixel buffer.
///
/// The buffer uses a tight stride by default and can be resized across viewport
/// reads. Resizing to a same-or-smaller byte requirement keeps existing vector
/// capacity, so callers can reuse allocations while still passing ordinary
/// `&mut [u8]` slices into decode APIs.
#[derive(Debug, Clone)]
pub struct JpegOutputBuffer {
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
}

impl JpegOutputBuffer {
    /// Create a tightly packed output buffer for `dimensions` and `fmt`.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the requested shape overflows byte counts.
    pub fn new(dimensions: (u32, u32), fmt: PixelFormat) -> Result<Self, BufferError> {
        let stride = tight_stride(dimensions.0, fmt)?;
        Self::with_stride(dimensions, stride, fmt)
    }

    /// Create an output buffer with an explicit row stride.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the stride is too small or sizes overflow.
    pub fn with_stride(
        dimensions: (u32, u32),
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<Self, BufferError> {
        let len = strided_output_len(dimensions, stride, fmt)?;
        if dimensions.0 != 0 {
            let row = tight_stride(dimensions.0, fmt)?;
            if stride < row {
                return Err(BufferError::StrideTooSmall {
                    row_bytes: row,
                    stride,
                });
            }
        }
        Ok(Self {
            bytes: alloc::vec![0; len],
            dimensions,
            stride,
            fmt,
        })
    }

    /// Resize to a tightly packed output shape.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the requested shape overflows byte counts.
    pub fn resize(&mut self, dimensions: (u32, u32), fmt: PixelFormat) -> Result<(), BufferError> {
        let stride = tight_stride(dimensions.0, fmt)?;
        self.resize_with_stride(dimensions, stride, fmt)
    }

    /// Resize with an explicit row stride.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the stride is too small or sizes overflow.
    pub fn resize_with_stride(
        &mut self,
        dimensions: (u32, u32),
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<(), BufferError> {
        let len = strided_output_len(dimensions, stride, fmt)?;
        if dimensions.0 != 0 {
            let row = tight_stride(dimensions.0, fmt)?;
            if stride < row {
                return Err(BufferError::StrideTooSmall {
                    row_bytes: row,
                    stride,
                });
            }
        }
        self.bytes.resize(len, 0);
        self.dimensions = dimensions;
        self.stride = stride;
        self.fmt = fmt;
        Ok(())
    }

    /// Borrow the decoded pixel bytes.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    /// Mutably borrow the decoded pixel bytes.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

    /// Current dimensions in pixels.
    #[must_use]
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Current row stride in bytes.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Current pixel format.
    #[must_use]
    pub fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    /// Current logical byte length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether the logical byte length is zero.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Retained vector capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.bytes.capacity()
    }
}

fn tight_stride(width: u32, fmt: PixelFormat) -> Result<usize, BufferError> {
    (width as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(BufferError::SizeOverflow {
            what: "tight JPEG output stride",
        })
}
