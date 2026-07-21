// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

use j2k_core::PixelFormat;
use metal::{Buffer, DeviceRef};

use super::{buffer_len, validate_bounds, validate_registry_id};
use crate::MetalSupportError;

/// Validated byte layout for one image stored in a Metal buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MetalImageLayout {
    byte_offset: usize,
    dimensions: (u32, u32),
    pitch_bytes: usize,
    pixel_format: PixelFormat,
    image_count: usize,
    image_stride_bytes: usize,
    byte_len: usize,
}

const METAL_BUFFER_OFFSET_ALIGNMENT: usize = 4;

/// Exclusively writable image subrange in a caller-owned Metal buffer.
///
/// This type is the checked handoff boundary for decoders that write directly
/// into storage allocated by another runtime. It retains the Metal allocation,
/// records its device identity, and exposes only the validated image layout.
/// Constructing it is unsafe because the Rust type system cannot prove that a
/// framework holding another `MTLBuffer` reference will honor exclusive GPU
/// write access to the selected subrange.
pub struct MetalImageDestination {
    buffer: Buffer,
    device_registry_id: u64,
    buffer_len: usize,
    layout: MetalImageLayout,
}

impl fmt::Debug for MetalImageDestination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalImageDestination")
            .field("device_registry_id", &self.device_registry_id)
            .field("buffer_len", &self.buffer_len)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl MetalImageDestination {
    /// Validate and retain an exclusively writable Metal image subrange.
    ///
    /// # Safety
    ///
    /// Until codec completion or an explicit GPU dependency has been
    /// established, the caller must prevent every CPU access and every GPU
    /// read or write that overlaps `layout`. Once the codec has inserted a
    /// same-device consumer-queue wait, later reads on that queue are allowed;
    /// unsynchronized access and overlapping writes remain forbidden. A
    /// decoder may bind the raw buffer only for writes inside the validated
    /// range and must retain this value until completion or dependency
    /// registration makes consumer access safe.
    ///
    /// # Errors
    ///
    /// Returns a typed bounds or alignment error when the image subrange cannot
    /// be safely bound as a Metal kernel destination.
    pub unsafe fn from_exclusive_buffer(
        buffer: Buffer,
        layout: MetalImageLayout,
    ) -> Result<Self, MetalSupportError> {
        let buffer_len = buffer_len(&buffer)?;
        validate_bounds(layout, buffer_len)?;
        if !layout
            .byte_offset()
            .is_multiple_of(METAL_BUFFER_OFFSET_ALIGNMENT)
        {
            return Err(MetalSupportError::BufferAlignment {
                offset_bytes: layout.byte_offset(),
                align: METAL_BUFFER_OFFSET_ALIGNMENT,
            });
        }
        Ok(Self {
            device_registry_id: buffer.device().registry_id(),
            buffer,
            buffer_len,
            layout,
        })
    }

    /// Validated image layout within the destination allocation.
    #[must_use]
    pub const fn layout(&self) -> MetalImageLayout {
        self.layout
    }

    /// Total allocation length retained by this destination.
    #[must_use]
    pub const fn buffer_len(&self) -> usize {
        self.buffer_len
    }

    /// Registry identifier of the Metal device that owns the allocation.
    #[must_use]
    pub const fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    /// Validate that a Metal device can write this destination.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalImageDeviceMismatch`] when the
    /// destination allocation belongs to another Metal device.
    pub fn validate_device(&self, device: &DeviceRef) -> Result<(), MetalSupportError> {
        validate_registry_id(self.device_registry_id, device.registry_id())
    }

    /// Validate the dimensions and pixel format expected by a decoder store.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalImageLayout`] when the decoder output
    /// geometry or pixel format differs from the validated destination layout.
    pub fn validate_image(
        &self,
        dimensions: (u32, u32),
        pixel_format: PixelFormat,
    ) -> Result<(), MetalSupportError> {
        if self.layout.dimensions() != dimensions {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "destination dimensions do not match decoded image",
            });
        }
        if self.layout.pixel_format() != pixel_format {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "destination pixel format does not match decoded image",
            });
        }
        Ok(())
    }

    /// Validate a complete dense image group stored in this destination.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalImageLayout`] when the decoded image
    /// geometry, pixel format, or group count differs from this range.
    pub fn validate_batch(
        &self,
        dimensions: (u32, u32),
        pixel_format: PixelFormat,
        image_count: usize,
    ) -> Result<(), MetalSupportError> {
        self.validate_image(dimensions, pixel_format)?;
        if self.layout.image_count() != image_count {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "destination image count does not match decoded group",
            });
        }
        Ok(())
    }

    /// Borrow the allocation for an audited backend write operation.
    ///
    /// # Safety
    ///
    /// The returned handle may only be bound for writes within [`Self::layout`]
    /// under the exclusive-access and completion guarantees established by the
    /// constructor. It must not escape the audited backend operation.
    #[must_use]
    pub unsafe fn raw_buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl MetalImageLayout {
    /// Validate image geometry and construct a Metal buffer layout.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalImageLayout`] when row geometry or
    /// the represented byte range overflows, or when the pitch is shorter than
    /// one pixel row.
    pub fn new(
        byte_offset: usize,
        dimensions: (u32, u32),
        pitch_bytes: usize,
        pixel_format: PixelFormat,
    ) -> Result<Self, MetalSupportError> {
        let image_byte_len = checked_image_byte_len(dimensions, pitch_bytes, pixel_format)?;
        Self::new_batch(
            byte_offset,
            dimensions,
            pitch_bytes,
            pixel_format,
            1,
            image_byte_len,
        )
    }

    /// Validate a homogeneous image group inside one Metal allocation.
    ///
    /// Only `byte_offset`, the group base, must satisfy Metal buffer-binding
    /// alignment. `image_stride_bytes` may place later Gray8 images at byte
    /// offsets that are not independently bindable; final-store kernels bind
    /// the allocation once and apply those item offsets in-kernel.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalImageLayout`] for zero image count,
    /// short or sample-misaligned strides, or overflowing byte geometry.
    pub fn new_batch(
        byte_offset: usize,
        dimensions: (u32, u32),
        pitch_bytes: usize,
        pixel_format: PixelFormat,
        image_count: usize,
        image_stride_bytes: usize,
    ) -> Result<Self, MetalSupportError> {
        let image_byte_len = checked_image_byte_len(dimensions, pitch_bytes, pixel_format)?;
        if image_count == 0 {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "image count must be nonzero",
            });
        }
        if image_stride_bytes < image_byte_len {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "image stride is shorter than one pitched image",
            });
        }
        if !pitch_bytes.is_multiple_of(pixel_format.bytes_per_sample())
            || !image_stride_bytes.is_multiple_of(pixel_format.bytes_per_sample())
        {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "row and image strides must be aligned to the sample width",
            });
        }
        let prior_images = image_count - 1;
        let byte_len = image_stride_bytes
            .checked_mul(prior_images)
            .and_then(|offset| offset.checked_add(image_byte_len))
            .ok_or(MetalSupportError::MetalImageLayout {
                reason: "pitched image group byte length overflows usize",
            })?;
        byte_offset
            .checked_add(byte_len)
            .ok_or(MetalSupportError::MetalImageLayout {
                reason: "image group byte range overflows usize",
            })?;
        Ok(Self {
            byte_offset,
            dimensions,
            pitch_bytes,
            pixel_format,
            image_count,
            image_stride_bytes,
            byte_len,
        })
    }

    /// Number of homogeneous images represented by this allocation range.
    #[must_use]
    pub const fn image_count(self) -> usize {
        self.image_count
    }

    /// Byte distance between the first pixel of consecutive images.
    #[must_use]
    pub const fn image_stride_bytes(self) -> usize {
        self.image_stride_bytes
    }

    /// Byte offset of one image relative to the validated group base.
    ///
    /// Returns `None` when `image_index` is outside the group or multiplication
    /// overflows.
    #[must_use]
    pub const fn image_offset_bytes(self, image_index: usize) -> Option<usize> {
        if image_index >= self.image_count {
            return None;
        }
        self.image_stride_bytes.checked_mul(image_index)
    }

    /// Byte offset of the first pixel row.
    #[must_use]
    pub const fn byte_offset(self) -> usize {
        self.byte_offset
    }

    /// Image dimensions in pixels.
    #[must_use]
    pub const fn dimensions(self) -> (u32, u32) {
        self.dimensions
    }

    /// Number of bytes between consecutive image rows.
    #[must_use]
    pub const fn pitch_bytes(self) -> usize {
        self.pitch_bytes
    }

    /// Pixel format stored in the image range.
    #[must_use]
    pub const fn pixel_format(self) -> PixelFormat {
        self.pixel_format
    }

    /// Number of bytes represented by the pitched image or image group.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        self.byte_len
    }
}

fn checked_image_byte_len(
    dimensions: (u32, u32),
    pitch_bytes: usize,
    pixel_format: PixelFormat,
) -> Result<usize, MetalSupportError> {
    if dimensions.0 == 0 || dimensions.1 == 0 {
        return Err(MetalSupportError::MetalImageLayout {
            reason: "image dimensions must be nonzero",
        });
    }
    let row_bytes = usize::try_from(dimensions.0)
        .ok()
        .and_then(|width| width.checked_mul(pixel_format.bytes_per_pixel()))
        .ok_or(MetalSupportError::MetalImageLayout {
            reason: "pixel row byte count overflows usize",
        })?;
    if pitch_bytes < row_bytes {
        return Err(MetalSupportError::MetalImageLayout {
            reason: "pitch is shorter than one pixel row",
        });
    }
    pitch_bytes
        .checked_mul(dimensions.1 as usize)
        .ok_or(MetalSupportError::MetalImageLayout {
            reason: "pitched image byte length overflows usize",
        })
}
