// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::PixelFormat;
use j2k_metal_support::{MetalImageLayout, ResidentMetalImage};
use metal::{Buffer, BufferRef, CommandBuffer};

use crate::error::metal_kernel_support_error;
use crate::Error;

#[doc(hidden)]
pub struct ResidentPrivateJpegTile {
    image: ResidentMetalImage,
    // Keep the producer resources alive for the lifetime of every tile clone.
    status_buffer: Buffer,
    command_buffer: CommandBuffer,
}

impl ResidentPrivateJpegTile {
    pub(crate) fn new(
        buffer: Buffer,
        byte_offset: usize,
        dimensions: (u32, u32),
        pixel_format: PixelFormat,
        pitch_bytes: usize,
        status_buffer: Buffer,
        command_buffer: CommandBuffer,
    ) -> Result<Self, Error> {
        let layout = MetalImageLayout::new(byte_offset, dimensions, pitch_bytes, pixel_format)
            .map_err(|source| {
                metal_kernel_support_error("JPEG private resident tile layout", source)
            })?;
        // SAFETY: both private-tile producers wait for the command buffer and
        // validate its status before constructing this wrapper.
        let image = unsafe { ResidentMetalImage::from_completed_buffer(buffer, layout) }.map_err(
            |source| metal_kernel_support_error("JPEG private resident tile adoption", source),
        )?;
        Ok(Self {
            image,
            status_buffer,
            command_buffer,
        })
    }

    /// Byte offset of the first decoded pixel in the backing buffer.
    pub fn byte_offset(&self) -> usize {
        self.image.byte_offset()
    }

    /// Dimensions of the decoded tile.
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }

    /// Pixel format of the decoded tile.
    pub fn pixel_format(&self) -> PixelFormat {
        self.image.pixel_format()
    }

    /// Number of bytes between consecutive decoded rows.
    pub fn pitch_bytes(&self) -> usize {
        self.image.pitch_bytes()
    }

    /// Borrow the common immutable resident image.
    pub fn resident_image(&self) -> &ResidentMetalImage {
        &self.image
    }

    /// Consume the private tile and return the common resident image.
    pub fn into_resident_image(self) -> ResidentMetalImage {
        self.image
    }

    /// Return the raw private Metal output buffer.
    ///
    /// # Safety
    ///
    /// The producer command has completed before this tile is returned, but
    /// the caller must synchronize every later access made through the returned
    /// buffer or a handle cloned from it. That obligation covers raw handles
    /// obtained from every clone of this tile; no two accesses may overlap when
    /// either can write the decoded range.
    pub unsafe fn buffer(&self) -> &BufferRef {
        self.buffer_trusted()
    }

    pub(crate) fn buffer_trusted(&self) -> &BufferRef {
        // SAFETY: this crate-private accessor is used only for read-only Metal
        // binding after the producer has completed.
        unsafe { self.image.raw_buffer() }.as_ref()
    }

    /// Consume this wrapper and transfer ownership of its decoded buffer.
    ///
    /// # Safety
    ///
    /// The producer command has already completed. Other tile clones, and
    /// buffers obtained by consuming them, can still refer to the same Metal
    /// allocation. No surviving tile offers safe host readback, and borrowed
    /// raw access remains unsafe; normal Metal synchronization remains each
    /// buffer recipient's responsibility after a handoff.
    #[deprecated(note = "use into_resident_image; raw Metal handles require unsafe interop")]
    pub unsafe fn into_buffer(self) -> Buffer {
        // SAFETY: the caller accepts the raw-handle synchronization contract.
        unsafe { self.image.raw_buffer() }.to_owned()
    }

    #[cfg(test)]
    pub(crate) fn status_buffer_trusted(&self) -> &BufferRef {
        self.status_buffer.as_ref()
    }
}

impl Clone for ResidentPrivateJpegTile {
    fn clone(&self) -> Self {
        Self {
            image: self.image.clone(),
            status_buffer: self.status_buffer.clone(),
            command_buffer: self.command_buffer.clone(),
        }
    }
}
