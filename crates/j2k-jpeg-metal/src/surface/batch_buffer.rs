// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::{BufferError, Downscale, PixelFormat, Rect};
use metal::{Buffer, BufferRef};

#[cfg(test)]
use super::{Storage, Surface};
use crate::buffers::new_shared_buffer;
use crate::{
    report_required_output_dimensions, scaled_dims, Error, JpegMetalResidentBatchReport,
    MetalBackendSession,
};

#[derive(Clone)]
/// Reusable caller-owned Metal buffer for full-tile JPEG batch output.
pub struct MetalBatchOutputBuffer {
    pub(super) buffer: Buffer,
    pub(super) access_gate: Arc<Mutex<()>>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    tile_stride_bytes: usize,
    tile_capacity: usize,
}

impl MetalBatchOutputBuffer {
    /// Allocate a reusable RGB8 output buffer for `tile_capacity` full-size tiles.
    pub fn new_rgb8_tiles(
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        tile_capacity: usize,
    ) -> Result<Self, Error> {
        Self::new_tiles(session, dimensions, PixelFormat::Rgb8, tile_capacity)
    }

    /// Ensure this output buffer can hold `tile_capacity` RGB8 tiles with `dimensions`.
    ///
    /// The existing allocation is retained when it already has the requested
    /// layout and at least the requested capacity. Otherwise the buffer is
    /// replaced with a new allocation.
    pub fn ensure_rgb8_tiles(
        &mut self,
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        tile_capacity: usize,
    ) -> Result<(), Error> {
        if self.dimensions == dimensions
            && self.fmt == PixelFormat::Rgb8
            && self.tile_capacity >= tile_capacity
        {
            return Ok(());
        }

        *self = Self::new_rgb8_tiles(session, dimensions, tile_capacity)?;
        Ok(())
    }

    /// Ensure this output buffer fits a full-image scaled RGB8 batch.
    pub fn ensure_rgb8_scaled_tiles(
        &mut self,
        session: &MetalBackendSession,
        full_dimensions: (u32, u32),
        scale: Downscale,
        tile_capacity: usize,
    ) -> Result<(), Error> {
        self.ensure_rgb8_tiles(session, scaled_dims(full_dimensions, scale), tile_capacity)
    }

    /// Ensure this output buffer fits a region-scaled RGB8 batch.
    pub fn ensure_rgb8_region_scaled_tiles(
        &mut self,
        session: &MetalBackendSession,
        roi: Rect,
        scale: Downscale,
        tile_capacity: usize,
    ) -> Result<(), Error> {
        let scaled = roi.scaled_covering(scale);
        self.ensure_rgb8_tiles(session, (scaled.w, scaled.h), tile_capacity)
    }

    /// Ensure this output buffer fits a preflighted RGB8 Metal resident batch.
    ///
    /// Ineligible reports return an error without replacing the existing
    /// allocation. Eligible empty reports are a no-op.
    #[doc(hidden)]
    pub fn ensure_rgb8_batch_report(
        &mut self,
        session: &MetalBackendSession,
        report: &JpegMetalResidentBatchReport,
    ) -> Result<(), Error> {
        let Some(dimensions) = report_required_output_dimensions(report)? else {
            return Ok(());
        };
        self.ensure_rgb8_tiles(session, dimensions, report.required_tile_capacity())
    }

    fn new_tiles(
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        tile_capacity: usize,
    ) -> Result<Self, Error> {
        if dimensions.0 == 0 || dimensions.1 == 0 || tile_capacity == 0 {
            return Err(Error::UnsupportedMetalRequest {
                reason: "JPEG Metal batch output requires nonzero dimensions and tile capacity",
            });
        }
        let row_bytes = dimensions
            .0
            .checked_mul(u32::try_from(fmt.bytes_per_pixel()).map_err(|_| {
                BufferError::SizeOverflow {
                    what: "JPEG Metal output row bytes",
                }
            })?)
            .ok_or(BufferError::SizeOverflow {
                what: "JPEG Metal output row bytes",
            })? as usize;
        let tile_stride_bytes =
            row_bytes
                .checked_mul(dimensions.1 as usize)
                .ok_or(BufferError::SizeOverflow {
                    what: "JPEG Metal output tile bytes",
                })?;
        let byte_len =
            tile_stride_bytes
                .checked_mul(tile_capacity)
                .ok_or(BufferError::SizeOverflow {
                    what: "JPEG Metal batch output bytes",
                })?;
        let buffer = new_shared_buffer(session.device(), byte_len)?;
        Ok(Self {
            buffer,
            access_gate: Arc::new(Mutex::new(())),
            dimensions,
            fmt,
            pitch_bytes: row_bytes,
            tile_stride_bytes,
            tile_capacity,
        })
    }

    /// Return the raw backing Metal buffer.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned buffer or any handle cloned from it. The internal safe-access
    /// gate cannot observe work submitted through raw handles. No such access
    /// may overlap a safe decode into this output or readback from a [`crate::Surface`]
    /// that aliases this allocation.
    pub unsafe fn buffer(&self) -> &BufferRef {
        self.buffer_trusted()
    }

    pub(crate) fn buffer_trusted(&self) -> &BufferRef {
        self.buffer.as_ref()
    }

    pub(crate) fn lock_for_safe_access(&self) -> Result<MutexGuard<'_, ()>, Error> {
        self.access_gate.lock().map_err(|_| Error::MetalKernel {
            message: "JPEG Metal batch output access gate was poisoned".to_string(),
        })
    }

    #[cfg(test)]
    pub(crate) fn shares_access_gate_with(&self, surface: &Surface) -> bool {
        matches!(
            &surface.storage,
            Storage::Metal {
                access_gate: Some(access_gate),
                ..
            } if Arc::ptr_eq(&self.access_gate, access_gate)
        )
    }

    /// Tile dimensions for this output allocation.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Pixel format for this output allocation.
    pub fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    /// Number of reusable tile slots in the buffer.
    pub fn tile_capacity(&self) -> usize {
        self.tile_capacity
    }

    /// Number of bytes between rows in one tile.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Number of bytes reserved for each tile slot.
    pub fn tile_stride_bytes(&self) -> usize {
        self.tile_stride_bytes
    }

    /// Total byte length of the backing allocation.
    pub fn byte_len(&self) -> usize {
        self.tile_stride_bytes * self.tile_capacity
    }

    pub(crate) fn clone_buffer(&self) -> Buffer {
        self.buffer.clone()
    }
}
