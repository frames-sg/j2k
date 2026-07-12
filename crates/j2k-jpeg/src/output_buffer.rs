// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable host output buffers for JPEG tile decode.

use alloc::vec::Vec;
use j2k_core::{
    strided_output_len_capped, try_host_vec_filled, validate_strided_output_buffer, BufferError,
    HostAllocationError, PixelFormat, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

const OUTPUT_BUFFER_ALLOCATION: &str = "JPEG output buffer";

/// Caller-owned reusable host pixel buffer.
///
/// The buffer uses a tight stride by default and can be resized across viewport
/// reads. Resizing to a same-or-smaller byte requirement keeps existing vector
/// capacity, so callers can reuse allocations while still passing ordinary
/// `&mut [u8]` slices into decode APIs.
#[derive(Debug)]
pub struct JpegOutputBuffer {
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    stride: usize,
    fmt: PixelFormat,
}

impl JpegOutputBuffer {
    /// Create a tightly packed output buffer for `dimensions` and `fmt`.
    ///
    /// Uses the shared default host allocation cap.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the requested shape overflows byte counts,
    /// exceeds the default host allocation cap, or cannot be reserved.
    pub fn new(dimensions: (u32, u32), fmt: PixelFormat) -> Result<Self, BufferError> {
        Self::new_with_max_bytes(dimensions, fmt, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    /// Create a tightly packed output buffer with an explicit allocation cap.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the requested shape overflows byte counts,
    /// exceeds `max_bytes`, or cannot be reserved.
    fn new_with_max_bytes(
        dimensions: (u32, u32),
        fmt: PixelFormat,
        max_bytes: usize,
    ) -> Result<Self, BufferError> {
        let stride = tight_stride(dimensions.0, fmt)?;
        Self::with_stride_with_max_bytes(dimensions, stride, fmt, max_bytes)
    }

    /// Create an output buffer with an explicit row stride.
    ///
    /// Uses the shared default host allocation cap.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the stride is too small, sizes overflow, the
    /// allocation exceeds the default host allocation cap, or reservation fails.
    pub fn with_stride(
        dimensions: (u32, u32),
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<Self, BufferError> {
        Self::with_stride_with_max_bytes(dimensions, stride, fmt, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    /// Create an output buffer with explicit row stride and allocation cap.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the stride is too small, sizes overflow, the
    /// allocation exceeds `max_bytes`, or reservation fails.
    fn with_stride_with_max_bytes(
        dimensions: (u32, u32),
        stride: usize,
        fmt: PixelFormat,
        max_bytes: usize,
    ) -> Result<Self, BufferError> {
        let len = strided_output_len_capped(
            dimensions,
            stride,
            fmt,
            max_bytes,
            OUTPUT_BUFFER_ALLOCATION,
        )?;
        validate_strided_output_buffer(dimensions, len, stride, fmt)?;
        Ok(Self {
            bytes: try_output_bytes(len, max_bytes)?,
            dimensions,
            stride,
            fmt,
        })
    }

    /// Resize to a tightly packed output shape.
    ///
    /// Uses the shared default host allocation cap.
    ///
    /// Existing decoded bytes may be discarded when the allocation must grow
    /// or stale capacity exceeds the active cap.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the requested shape overflows byte counts,
    /// exceeds the default host allocation cap, or cannot be reserved.
    pub fn resize(&mut self, dimensions: (u32, u32), fmt: PixelFormat) -> Result<(), BufferError> {
        self.resize_with_max_bytes(dimensions, fmt, DEFAULT_MAX_HOST_ALLOCATION_BYTES)
    }

    /// Resize to a tightly packed output shape with an explicit allocation cap.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the requested shape overflows byte counts,
    /// exceeds `max_bytes`, or cannot be reserved.
    fn resize_with_max_bytes(
        &mut self,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        max_bytes: usize,
    ) -> Result<(), BufferError> {
        let stride = tight_stride(dimensions.0, fmt)?;
        self.resize_with_stride_with_max_bytes(dimensions, stride, fmt, max_bytes)
    }

    /// Resize with an explicit row stride.
    ///
    /// Uses the shared default host allocation cap.
    /// Existing decoded bytes may be discarded when storage must be replaced.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the stride is too small, sizes overflow, the
    /// allocation exceeds the default host allocation cap, or reservation fails.
    pub fn resize_with_stride(
        &mut self,
        dimensions: (u32, u32),
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<(), BufferError> {
        self.resize_with_stride_with_max_bytes(
            dimensions,
            stride,
            fmt,
            DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        )
    }

    /// Resize with explicit row stride and allocation cap.
    /// Existing decoded bytes may be discarded when storage must be replaced.
    ///
    /// # Errors
    /// Returns [`BufferError`] if the stride is too small, sizes overflow, the
    /// allocation exceeds `max_bytes`, or reservation fails.
    fn resize_with_stride_with_max_bytes(
        &mut self,
        dimensions: (u32, u32),
        stride: usize,
        fmt: PixelFormat,
        max_bytes: usize,
    ) -> Result<(), BufferError> {
        let len = strided_output_len_capped(
            dimensions,
            stride,
            fmt,
            max_bytes,
            OUTPUT_BUFFER_ALLOCATION,
        )?;
        validate_strided_output_buffer(dimensions, len, stride, fmt)?;
        if self.bytes.capacity() > max_bytes || len > self.bytes.capacity() {
            self.clear_storage();
            self.bytes = try_output_bytes(len, max_bytes)?;
        } else {
            self.bytes.resize(len, 0);
        }
        self.dimensions = dimensions;
        self.stride = stride;
        self.fmt = fmt;
        Ok(())
    }

    fn clear_storage(&mut self) {
        self.bytes = Vec::new();
        self.dimensions = (0, 0);
        self.stride = 0;
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

fn host_allocation_error(error: HostAllocationError, what: &'static str) -> BufferError {
    BufferError::HostAllocationFailed {
        bytes: error.requested_bytes(),
        what,
    }
}

fn try_output_bytes(len: usize, max_bytes: usize) -> Result<Vec<u8>, BufferError> {
    let bytes = try_host_vec_filled(len, 0)
        .map_err(|error| host_allocation_error(error, OUTPUT_BUFFER_ALLOCATION))?;
    ensure_output_capacity(bytes.capacity(), max_bytes)?;
    Ok(bytes)
}

fn ensure_output_capacity(capacity: usize, max_bytes: usize) -> Result<(), BufferError> {
    if capacity > max_bytes {
        return Err(BufferError::AllocationTooLarge {
            requested: capacity,
            cap: max_bytes,
            what: OUTPUT_BUFFER_ALLOCATION,
        });
    }
    Ok(())
}

fn tight_stride(width: u32, fmt: PixelFormat) -> Result<usize, BufferError> {
    (width as usize)
        .checked_mul(fmt.bytes_per_pixel())
        .ok_or(BufferError::SizeOverflow {
            what: "tight JPEG output stride",
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUGE_DIMENSIONS: (u32, u32) = (65_500, 65_500);

    fn assert_allocation_too_large(error: &BufferError) {
        assert!(
            matches!(
                error,
                BufferError::AllocationTooLarge {
                    requested,
                    cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    what: "JPEG output buffer",
                } if *requested > DEFAULT_MAX_HOST_ALLOCATION_BYTES
            ),
            "expected AllocationTooLarge, got {error:?}"
        );
    }

    #[test]
    fn new_rejects_huge_output_before_allocation() {
        let err = JpegOutputBuffer::new(HUGE_DIMENSIONS, PixelFormat::Rgba16)
            .expect_err("huge output must be capped");
        assert_allocation_too_large(&err);
    }

    #[test]
    fn with_stride_rejects_huge_output_before_allocation() {
        let stride = HUGE_DIMENSIONS.0 as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let err = JpegOutputBuffer::with_stride(HUGE_DIMENSIONS, stride, PixelFormat::Rgba16)
            .expect_err("huge output must be capped");
        assert_allocation_too_large(&err);
    }

    #[test]
    fn resize_rejects_huge_output_before_allocation() {
        let mut buffer =
            JpegOutputBuffer::new((1, 1), PixelFormat::Rgba8).expect("small output buffer");
        let err = buffer
            .resize(HUGE_DIMENSIONS, PixelFormat::Rgba16)
            .expect_err("huge output must be capped");
        assert_allocation_too_large(&err);
        assert_eq!(buffer.dimensions(), (1, 1));
    }

    #[test]
    fn resize_with_stride_rejects_huge_output_before_allocation() {
        let mut buffer =
            JpegOutputBuffer::new((1, 1), PixelFormat::Rgba8).expect("small output buffer");
        let stride = HUGE_DIMENSIONS.0 as usize * PixelFormat::Rgba16.bytes_per_pixel();
        let err = buffer
            .resize_with_stride(HUGE_DIMENSIONS, stride, PixelFormat::Rgba16)
            .expect_err("huge output must be capped");
        assert_allocation_too_large(&err);
        assert_eq!(buffer.dimensions(), (1, 1));
    }

    #[test]
    fn explicit_max_bytes_helpers_enforce_smaller_caps() {
        let err = JpegOutputBuffer::new_with_max_bytes((2, 2), PixelFormat::Rgba8, 15)
            .expect_err("caller cap should be enforced");
        assert!(matches!(
            err,
            BufferError::AllocationTooLarge {
                requested: 16,
                cap: 15,
                what: "JPEG output buffer",
            }
        ));
    }

    #[test]
    fn allocator_failure_keeps_its_public_typed_category() {
        let error = host_allocation_error(
            HostAllocationError::for_elements::<u16>(2048),
            OUTPUT_BUFFER_ALLOCATION,
        );
        assert_eq!(
            error,
            BufferError::HostAllocationFailed {
                bytes: 4096,
                what: OUTPUT_BUFFER_ALLOCATION,
            }
        );
    }

    #[test]
    fn resize_drops_stale_capacity_before_using_a_smaller_cap() {
        let mut buffer =
            JpegOutputBuffer::new_with_max_bytes((4, 4), PixelFormat::Rgba8, 64).unwrap();
        assert!(buffer.capacity() >= 64);
        buffer
            .resize_with_max_bytes((1, 1), PixelFormat::Rgba8, 4)
            .expect("stale capacity must be replaced");
        assert_eq!(buffer.dimensions(), (1, 1));
        assert_eq!(buffer.len(), 4);
        assert!(buffer.capacity() <= 4);
    }

    #[test]
    fn growth_replaces_disposable_decoded_contents() {
        let mut buffer = JpegOutputBuffer::new((1, 1), PixelFormat::Gray8).unwrap();
        buffer.as_mut_slice().fill(0xa5);
        buffer.resize((2, 1), PixelFormat::Gray8).unwrap();
        assert_eq!(buffer.as_slice(), [0, 0]);
    }

    #[test]
    fn actual_allocator_capacity_is_checked_against_the_cap() {
        assert!(matches!(
            ensure_output_capacity(17, 16),
            Err(BufferError::AllocationTooLarge {
                requested: 17,
                cap: 16,
                what: OUTPUT_BUFFER_ALLOCATION,
            })
        ));
    }
}
