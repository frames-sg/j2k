// SPDX-License-Identifier: MIT OR Apache-2.0

use std::borrow::Cow;
#[cfg(target_os = "macos")]
use std::sync::{Arc, Mutex, MutexGuard};

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, BufferError, DeviceMemoryRange,
    DeviceSurface, Downscale, PixelFormat, Rect, SurfaceMetadata, SurfaceResidency,
};

#[cfg(target_os = "macos")]
use crate::buffers::checked_buffer_slice_at;
use crate::{
    report_required_output_dimensions, scaled_dims, Error, JpegMetalResidentBatchReport,
    MetalBackendSession,
};

#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, BufferRef, CommandBuffer, MTLPixelFormat, MTLResourceOptions, MTLStorageMode,
    MTLTextureType, MTLTextureUsage, Texture, TextureDescriptor, TextureRef,
};

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(target_os = "macos")]
    Metal {
        buffer: Buffer,
        offset: usize,
        access_gate: Option<Arc<Mutex<()>>>,
    },
}

#[derive(Clone)]
/// Decoded image surface returned by the JPEG Metal backend.
pub struct Surface {
    pub(crate) backend: BackendKind,
    pub(crate) residency: SurfaceResidency,
    pub(crate) dimensions: (u32, u32),
    pub(crate) fmt: PixelFormat,
    pub(crate) pitch_bytes: usize,
    pub(crate) storage: Storage,
}

impl Surface {
    fn metadata(&self) -> SurfaceMetadata {
        SurfaceMetadata::new(
            self.backend,
            self.residency,
            self.dimensions,
            self.fmt,
            self.pitch_bytes,
        )
    }

    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Current residency for the surface bytes.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Return the tightly packed surface bytes.
    ///
    /// Host storage is borrowed. Metal storage is copied into an owned snapshot
    /// so safe Rust never exposes a slice that aliases later GPU access.
    ///
    /// # Panics
    ///
    /// Panics if Metal storage cannot be synchronized or read. Use
    /// [`Surface::download_into`] when readback errors must be handled.
    pub fn as_bytes(&self) -> Cow<'_, [u8]> {
        self.storage_bytes()
            .expect("Metal surface storage must be synchronized, CPU-visible, and bounded")
    }

    fn storage_bytes(&self) -> Result<Cow<'_, [u8]>, Error> {
        match &self.storage {
            Storage::Host(bytes) => Ok(Cow::Borrowed(bytes)),
            #[cfg(target_os = "macos")]
            Storage::Metal {
                buffer,
                offset,
                access_gate,
            } => {
                let _access = match access_gate {
                    Some(gate) => Some(gate.lock().map_err(|_| Error::MetalKernel {
                        message: "JPEG Metal surface access gate was poisoned".to_string(),
                    })?),
                    None => None,
                };
                let len = self.byte_len();
                checked_buffer_slice_at::<u8>(buffer, *offset, len, "surface bytes").map(Cow::Owned)
            }
        }
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        let bytes = self.storage_bytes()?;
        copy_tight_pixels_to_strided_output(bytes.as_ref(), self.dimensions, self.fmt, out, stride)
            .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    /// Return the raw Metal buffer and byte offset when the surface is Metal-backed.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned buffer or any handle cloned from it. The internal safe-access
    /// gate cannot observe work submitted through raw handles. In particular,
    /// no command may write the surface range while [`Surface::as_bytes`] or
    /// [`Surface::download_into`] reads it, and no raw access may overlap a safe
    /// decoder write through an aliasing [`MetalBatchOutputBuffer`].
    pub unsafe fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
        self.metal_buffer_trusted()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal_buffer_trusted(&self) -> Option<(&Buffer, usize)> {
        match &self.storage {
            Storage::Metal { buffer, offset, .. } => Some((buffer, *offset)),
            Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self::from_metal_buffer_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal {
                buffer,
                offset,
                access_gate: None,
            },
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_cpu_staged_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self::from_cpu_staged_metal_buffer_offset(buffer, dimensions, fmt, 0)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_cpu_staged_metal_buffer_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::CpuStagedMetalUpload,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal {
                buffer,
                offset,
                access_gate: None,
            },
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_batch_output_buffer_offset(
        output: &MetalBatchOutputBuffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            storage: Storage::Metal {
                buffer: output.buffer.clone(),
                offset,
                access_gate: Some(Arc::clone(&output.access_gate)),
            },
        }
    }
}

#[doc(hidden)]
impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.metadata().backend
    }

    fn residency(&self) -> SurfaceResidency {
        self.metadata().residency
    }

    fn dimensions(&self) -> (u32, u32) {
        self.metadata().dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.metadata().pixel_format
    }

    fn byte_len(&self) -> usize {
        self.metadata().byte_len()
    }

    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        match &self.storage {
            Storage::Host(_) => None,
            #[cfg(target_os = "macos")]
            Storage::Metal { buffer, offset, .. } => Some(DeviceMemoryRange::new(
                BackendKind::Metal,
                u64::try_from(buffer.as_ptr() as usize).ok()?,
                *offset,
                self.byte_len(),
            )),
        }
    }
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub struct ResidentPrivateJpegTile {
    buffer: Buffer,
    byte_offset: usize,
    dimensions: (u32, u32),
    pixel_format: PixelFormat,
    pitch_bytes: usize,
    // Keep the producer resources alive for the lifetime of every tile clone.
    status_buffer: Buffer,
    command_buffer: CommandBuffer,
}

#[cfg(target_os = "macos")]
impl ResidentPrivateJpegTile {
    pub(crate) fn new(
        buffer: Buffer,
        byte_offset: usize,
        dimensions: (u32, u32),
        pixel_format: PixelFormat,
        pitch_bytes: usize,
        status_buffer: Buffer,
        command_buffer: CommandBuffer,
    ) -> Self {
        Self {
            buffer,
            byte_offset,
            dimensions,
            pixel_format,
            pitch_bytes,
            status_buffer,
            command_buffer,
        }
    }

    /// Byte offset of the first decoded pixel in the backing buffer.
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    /// Dimensions of the decoded tile.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Pixel format of the decoded tile.
    pub fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    /// Number of bytes between consecutive decoded rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
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
        self.buffer.as_ref()
    }

    /// Consume this wrapper and transfer ownership of its decoded buffer.
    ///
    /// The producer command has already completed. Other tile clones, and
    /// buffers obtained by consuming them, can still refer to the same Metal
    /// allocation. No surviving tile offers safe host readback, and borrowed
    /// raw access remains unsafe; normal Metal synchronization remains each
    /// buffer recipient's responsibility after a handoff.
    pub fn into_buffer(self) -> Buffer {
        self.buffer
    }

    #[cfg(test)]
    pub(crate) fn status_buffer_trusted(&self) -> &BufferRef {
        self.status_buffer.as_ref()
    }
}

#[cfg(target_os = "macos")]
impl Clone for ResidentPrivateJpegTile {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            byte_offset: self.byte_offset,
            dimensions: self.dimensions,
            pixel_format: self.pixel_format,
            pitch_bytes: self.pitch_bytes,
            status_buffer: self.status_buffer.clone(),
            command_buffer: self.command_buffer.clone(),
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable caller-owned Metal buffer for full-tile JPEG batch output.
pub struct MetalBatchOutputBuffer {
    buffer: Buffer,
    access_gate: Arc<Mutex<()>>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    tile_stride_bytes: usize,
    tile_capacity: usize,
}

#[cfg(target_os = "macos")]
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
        let byte_len_u64 = u64::try_from(byte_len).map_err(|_| BufferError::SizeOverflow {
            what: "JPEG Metal batch output bytes",
        })?;
        let buffer = session
            .device()
            .new_buffer(byte_len_u64, MTLResourceOptions::StorageModeShared);
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
    /// may overlap a safe decode into this output or readback from a [`Surface`]
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

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable caller-owned Metal textures for full-tile JPEG batch output.
pub struct MetalBatchTextureOutput {
    textures: Vec<Texture>,
    access_gate: Arc<Mutex<()>>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    metal_fmt: MTLPixelFormat,
}

#[cfg(target_os = "macos")]
impl MetalBatchTextureOutput {
    /// Allocate reusable private RGBA8 textures for `tile_capacity` full-size tiles.
    pub fn new_rgba8_tiles(
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        tile_capacity: usize,
    ) -> Result<Self, Error> {
        if dimensions.0 == 0 || dimensions.1 == 0 || tile_capacity == 0 {
            return Err(Error::UnsupportedMetalRequest {
                reason:
                    "JPEG Metal batch texture output requires nonzero dimensions and tile capacity",
            });
        }

        let descriptor = TextureDescriptor::new();
        descriptor.set_texture_type(MTLTextureType::D2);
        descriptor.set_pixel_format(MTLPixelFormat::RGBA8Unorm);
        descriptor.set_width(u64::from(dimensions.0));
        descriptor.set_height(u64::from(dimensions.1));
        descriptor.set_depth(1);
        descriptor.set_mipmap_level_count(1);
        descriptor.set_sample_count(1);
        descriptor.set_storage_mode(MTLStorageMode::Private);
        descriptor.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);

        let mut textures = Vec::with_capacity(tile_capacity);
        for _ in 0..tile_capacity {
            textures.push(session.device().new_texture(&descriptor));
        }

        Ok(Self {
            textures,
            access_gate: Arc::new(Mutex::new(())),
            dimensions,
            fmt: PixelFormat::Rgba8,
            metal_fmt: MTLPixelFormat::RGBA8Unorm,
        })
    }

    /// Ensure this output set can hold `tile_capacity` RGBA8 textures with `dimensions`.
    ///
    /// Existing textures are retained when they already have the requested
    /// layout and at least the requested capacity. Otherwise the texture set is
    /// replaced with new private RGBA8 textures.
    pub fn ensure_rgba8_tiles(
        &mut self,
        session: &MetalBackendSession,
        dimensions: (u32, u32),
        tile_capacity: usize,
    ) -> Result<(), Error> {
        if self.dimensions == dimensions
            && self.fmt == PixelFormat::Rgba8
            && self.metal_fmt == MTLPixelFormat::RGBA8Unorm
            && self.tile_capacity() >= tile_capacity
        {
            return Ok(());
        }

        *self = Self::new_rgba8_tiles(session, dimensions, tile_capacity)?;
        Ok(())
    }

    /// Ensure this output set fits a full-image scaled RGBA8 texture batch.
    pub fn ensure_rgba8_scaled_tiles(
        &mut self,
        session: &MetalBackendSession,
        full_dimensions: (u32, u32),
        scale: Downscale,
        tile_capacity: usize,
    ) -> Result<(), Error> {
        self.ensure_rgba8_tiles(session, scaled_dims(full_dimensions, scale), tile_capacity)
    }

    /// Ensure this output set fits a region-scaled RGBA8 texture batch.
    pub fn ensure_rgba8_region_scaled_tiles(
        &mut self,
        session: &MetalBackendSession,
        roi: Rect,
        scale: Downscale,
        tile_capacity: usize,
    ) -> Result<(), Error> {
        let scaled = roi.scaled_covering(scale);
        self.ensure_rgba8_tiles(session, (scaled.w, scaled.h), tile_capacity)
    }

    /// Ensure this texture set fits a preflighted RGB8 Metal resident batch.
    ///
    /// Ineligible reports return an error without replacing the existing
    /// textures. Eligible empty reports are a no-op.
    #[doc(hidden)]
    pub fn ensure_rgba8_batch_report(
        &mut self,
        session: &MetalBackendSession,
        report: &JpegMetalResidentBatchReport,
    ) -> Result<(), Error> {
        let Some(dimensions) = report_required_output_dimensions(report)? else {
            return Ok(());
        };
        self.ensure_rgba8_tiles(session, dimensions, report.required_tile_capacity())
    }

    /// Tile dimensions for this output allocation.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Pixel format for this output allocation.
    pub fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    /// Metal pixel format for each backing texture.
    pub fn metal_pixel_format(&self) -> MTLPixelFormat {
        self.metal_fmt
    }

    /// Number of reusable tile texture slots.
    pub fn tile_capacity(&self) -> usize {
        self.textures.len()
    }

    /// Return a raw reusable output texture by tile slot.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned texture or any handle cloned from it. The internal safe-access
    /// gate cannot observe work submitted through raw handles. No such access
    /// may overlap a safe decode into this output, any clone or subset that
    /// shares its allocation gate, or access through a derived
    /// [`MetalTextureTile`].
    pub unsafe fn texture(&self, index: usize) -> Option<&TextureRef> {
        self.texture_trusted(index)
    }

    pub(crate) fn texture_trusted(&self, index: usize) -> Option<&TextureRef> {
        self.textures.get(index).map(std::convert::AsRef::as_ref)
    }

    pub(crate) fn clone_texture_trusted(&self, index: usize) -> Option<Texture> {
        self.textures.get(index).cloned()
    }

    pub(crate) fn clone_access_gate(&self) -> Arc<Mutex<()>> {
        Arc::clone(&self.access_gate)
    }

    pub(crate) fn lock_for_safe_access(&self) -> Result<MutexGuard<'_, ()>, Error> {
        self.access_gate.lock().map_err(|_| Error::MetalKernel {
            message: "JPEG Metal batch texture output access gate was poisoned".to_string(),
        })
    }

    pub(crate) fn clone_slots(&self, indices: &[usize]) -> Result<Self, Error> {
        let mut textures = Vec::with_capacity(indices.len());
        for &index in indices {
            textures.push(
                self.clone_texture_trusted(index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?,
            );
        }
        Ok(Self {
            textures,
            access_gate: Arc::clone(&self.access_gate),
            dimensions: self.dimensions,
            fmt: self.fmt,
            metal_fmt: self.metal_fmt,
        })
    }

    #[cfg(test)]
    pub(crate) fn shares_access_gate_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.access_gate, &other.access_gate)
    }

    #[cfg(test)]
    pub(crate) fn shares_access_gate_with_tile(&self, tile: &MetalTextureTile) -> bool {
        Arc::ptr_eq(&self.access_gate, &tile.access_gate)
    }
}

#[cfg(target_os = "macos")]
/// One decoded JPEG tile resident in a caller-owned Metal texture.
pub struct MetalTextureTile {
    texture: Texture,
    access_gate: Arc<Mutex<()>>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
}

#[cfg(target_os = "macos")]
impl MetalTextureTile {
    pub(crate) fn new(
        texture: Texture,
        access_gate: Arc<Mutex<()>>,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self {
            texture,
            access_gate,
            dimensions,
            fmt,
        }
    }

    /// Return the raw Metal texture containing the decoded tile.
    ///
    /// # Safety
    ///
    /// The caller must synchronize every CPU and GPU access made through the
    /// returned texture or any handle cloned from it. The safe decode gate
    /// shared with the originating [`MetalBatchTextureOutput`] cannot observe
    /// work submitted through raw handles. No raw access may overlap a safe
    /// decode through that output, one of its clones or subsets, or another
    /// tile derived from the same allocation.
    pub unsafe fn texture(&self) -> &TextureRef {
        self.texture_trusted()
    }

    pub(crate) fn texture_trusted(&self) -> &TextureRef {
        self.texture.as_ref()
    }

    /// Decoded tile dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    /// Decoded tile pixel format.
    pub fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }
}

#[cfg(target_os = "macos")]
impl Clone for MetalTextureTile {
    fn clone(&self) -> Self {
        Self {
            texture: self.texture.clone(),
            access_gate: Arc::clone(&self.access_gate),
            dimensions: self.dimensions,
            fmt: self.fmt,
        }
    }
}
