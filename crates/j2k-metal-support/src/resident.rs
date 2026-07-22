// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fmt, mem};

use j2k_core::{DeviceSubmission, PixelFormat};
use metal::{Buffer, BufferRef, CommandBuffer, DeviceRef, MTLCommandBufferStatus};

use crate::{wait_for_completion, MetalSupportError};

mod destination;

pub use self::destination::{MetalImageDestination, MetalImageLayout};

/// Owned, logically immutable Metal-resident image.
///
/// Clones retain the private Metal allocation for read-only GPU use. Safe APIs
/// never expose the raw Metal handle, so mutation requires crossing the
/// documented unsafe interop boundary.
#[derive(Clone)]
pub struct ResidentMetalImage {
    buffer: Buffer,
    device_registry_id: u64,
    buffer_len: usize,
    layout: MetalImageLayout,
}

impl fmt::Debug for ResidentMetalImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResidentMetalImage")
            .field("device_registry_id", &self.device_registry_id())
            .field("buffer_len", &self.buffer_len)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

impl ResidentMetalImage {
    /// Adopt a completed, uniquely controlled Metal buffer as an immutable image.
    ///
    /// # Safety
    ///
    /// All CPU and GPU writes to the described range must have completed. The
    /// caller must ensure no surviving raw handle can mutate the allocation for
    /// the lifetime of this image or any clone derived from it.
    ///
    /// # Errors
    ///
    /// Returns a typed layout or bounds error when the image range does not fit
    /// in the allocation.
    pub unsafe fn from_completed_buffer(
        buffer: Buffer,
        layout: MetalImageLayout,
    ) -> Result<Self, MetalSupportError> {
        let buffer_len = buffer_len(&buffer)?;
        validate_bounds(layout, buffer_len)?;
        Ok(Self::from_validated(buffer, buffer_len, layout))
    }

    /// Adopt an exclusively controlled Metal buffer before its producer completes.
    ///
    /// This is a low-level building block for backends that already own a
    /// command submission. Prefer [`SubmittedMetalImages`] for new submission
    /// APIs because it makes the completion lifetime explicit.
    ///
    /// # Safety
    ///
    /// The only incomplete access may be writes encoded by the producer
    /// submission retained by the caller. The image and every raw allocation
    /// alias must remain inaccessible to readers and other writers until that
    /// submission completes successfully. On producer failure, the image must
    /// not be exposed. After completion, no writer may access the allocation
    /// for the lifetime of this image or its clones.
    ///
    /// # Errors
    ///
    /// Returns a typed layout or bounds error when the image range does not fit
    /// in the allocation.
    #[doc(hidden)]
    pub unsafe fn from_exclusive_pending_buffer(
        buffer: Buffer,
        layout: MetalImageLayout,
    ) -> Result<Self, MetalSupportError> {
        let buffer_len = buffer_len(&buffer)?;
        validate_bounds(layout, buffer_len)?;
        Ok(Self::from_validated(buffer, buffer_len, layout))
    }

    fn from_validated(buffer: Buffer, buffer_len: usize, layout: MetalImageLayout) -> Self {
        let device_registry_id = buffer.device().registry_id();
        Self {
            buffer,
            device_registry_id,
            buffer_len,
            layout,
        }
    }

    /// Construct another immutable view into the same allocation.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::BufferBounds`] when the requested view does
    /// not fit in the allocation.
    pub fn view(&self, layout: MetalImageLayout) -> Result<Self, MetalSupportError> {
        validate_bounds(layout, self.buffer_len)?;
        let parent_end = self
            .layout
            .byte_offset()
            .checked_add(self.layout.byte_len())
            .ok_or(MetalSupportError::MetalImageLayout {
                reason: "resident image byte range overflows usize",
            })?;
        let child_end = layout.byte_offset().checked_add(layout.byte_len()).ok_or(
            MetalSupportError::MetalImageLayout {
                reason: "image view byte range overflows usize",
            },
        )?;
        if layout.byte_offset() < self.layout.byte_offset() || child_end > parent_end {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "image view falls outside the resident image range",
            });
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            device_registry_id: self.device_registry_id,
            buffer_len: self.buffer_len,
            layout,
        })
    }

    /// Validated layout of this image view.
    #[must_use]
    pub const fn layout(&self) -> MetalImageLayout {
        self.layout
    }

    /// Byte offset of the first image row.
    #[must_use]
    pub const fn byte_offset(&self) -> usize {
        self.layout.byte_offset()
    }

    /// Image dimensions in pixels.
    #[must_use]
    pub const fn dimensions(&self) -> (u32, u32) {
        self.layout.dimensions()
    }

    /// Number of bytes between consecutive image rows.
    #[must_use]
    pub const fn pitch_bytes(&self) -> usize {
        self.layout.pitch_bytes()
    }

    /// Pixel format stored in this image.
    #[must_use]
    pub const fn pixel_format(&self) -> PixelFormat {
        self.layout.pixel_format()
    }

    /// Number of bytes represented by this image view.
    #[must_use]
    pub const fn byte_len(&self) -> usize {
        self.layout.byte_len()
    }

    /// Metal device registry identifier that owns the allocation.
    #[must_use]
    pub fn device_registry_id(&self) -> u64 {
        self.device_registry_id
    }

    /// Validate that a Metal device can use this image.
    ///
    /// # Errors
    ///
    /// Returns [`MetalSupportError::MetalImageDeviceMismatch`] for a different
    /// Metal device.
    pub fn validate_device(&self, device: &DeviceRef) -> Result<(), MetalSupportError> {
        validate_registry_id(self.device_registry_id(), device.registry_id())
    }

    /// Borrow the raw Metal allocation for an audited backend operation.
    ///
    /// # Safety
    ///
    /// The caller may bind the returned handle only for GPU reads covered by a
    /// submission that retains either this image or a private allocation handle
    /// until completion. Any derived raw handles must remain inside the audited
    /// backend boundary. No CPU or GPU writer may access the allocation.
    #[must_use]
    pub unsafe fn raw_buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// Submitted Metal work that resolves to immutable resident images.
pub struct SubmittedMetalImages {
    command_buffer: Option<CommandBuffer>,
    outputs: Vec<ResidentMetalImage>,
    inputs: Vec<ResidentMetalImage>,
}

impl fmt::Debug for SubmittedMetalImages {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmittedMetalImages")
            .field("pending", &self.command_buffer.is_some())
            .field("output_count", &self.outputs.len())
            .field("input_count", &self.inputs.len())
            .finish()
    }
}

impl SubmittedMetalImages {
    /// Wrap an uncommitted command buffer and its exclusively owned outputs.
    ///
    /// # Safety
    ///
    /// `command_buffer` must belong to `device`. Each output buffer must be a
    /// fresh allocation whose only pending writer is encoded in that command
    /// buffer, and no raw alias capable of later mutation may survive this
    /// call. Every resident input whose raw buffer was bound by the command must
    /// be included in `inputs`.
    ///
    /// # Errors
    ///
    /// Returns a typed layout, bounds, or device-identity error before the
    /// command buffer is committed.
    pub unsafe fn from_uncommitted(
        device: &DeviceRef,
        command_buffer: CommandBuffer,
        outputs: Vec<(Buffer, MetalImageLayout)>,
        inputs: Vec<ResidentMetalImage>,
    ) -> Result<Self, MetalSupportError> {
        if outputs.is_empty() {
            return Err(MetalSupportError::MetalImageLayout {
                reason: "a Metal image submission must contain an output",
            });
        }
        let registry_id = device.registry_id();
        for input in &inputs {
            validate_registry_id(input.device_registry_id(), registry_id)?;
        }
        let outputs = outputs
            .into_iter()
            .map(|(buffer, layout)| {
                validate_registry_id(buffer.device().registry_id(), registry_id)?;
                let len = buffer_len(&buffer)?;
                validate_bounds(layout, len)?;
                Ok(ResidentMetalImage::from_validated(buffer, len, layout))
            })
            .collect::<Result<Vec<_>, MetalSupportError>>()?;
        Ok(Self {
            command_buffer: Some(command_buffer),
            outputs,
            inputs,
        })
    }

    fn complete(&mut self) -> Result<(), MetalSupportError> {
        let Some(command_buffer) = self.command_buffer.take() else {
            return Ok(());
        };
        if matches!(
            command_buffer.status(),
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            command_buffer.commit();
        }
        wait_for_completion(&command_buffer)
    }
}

impl DeviceSubmission for SubmittedMetalImages {
    type Output = Vec<ResidentMetalImage>;
    type Error = MetalSupportError;

    fn wait(mut self) -> Result<Self::Output, Self::Error> {
        self.complete()?;
        Ok(mem::take(&mut self.outputs))
    }
}

impl Drop for SubmittedMetalImages {
    fn drop(&mut self) {
        if let Err(error) = self.complete() {
            log::error!("Metal image submission failed while being dropped: {error}");
        }
    }
}

fn buffer_len(buffer: &BufferRef) -> Result<usize, MetalSupportError> {
    usize::try_from(buffer.length()).map_err(|_| MetalSupportError::MetalImageLayout {
        reason: "Metal buffer length exceeds usize",
    })
}

fn validate_bounds(layout: MetalImageLayout, buffer_len: usize) -> Result<(), MetalSupportError> {
    let end = layout.byte_offset().checked_add(layout.byte_len()).ok_or(
        MetalSupportError::MetalImageLayout {
            reason: "image byte range overflows usize",
        },
    )?;
    if end > buffer_len {
        return Err(MetalSupportError::BufferBounds {
            offset_bytes: layout.byte_offset(),
            byte_len: layout.byte_len(),
            buffer_len,
        });
    }
    Ok(())
}

pub(crate) fn validate_registry_id(
    image_registry_id: u64,
    requested_registry_id: u64,
) -> Result<(), MetalSupportError> {
    if image_registry_id == requested_registry_id {
        Ok(())
    } else {
        Err(MetalSupportError::MetalImageDeviceMismatch {
            image_registry_id,
            requested_registry_id,
        })
    }
}
