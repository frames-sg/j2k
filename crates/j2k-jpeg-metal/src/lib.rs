// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal-backed JPEG decode and encode adapters.
//!
//! The crate exposes the same CPU-visible JPEG decode surface as
//! `j2k-jpeg`, with optional Metal-resident surfaces and batch submission
//! helpers on macOS. Non-macOS builds keep the public API available but return
//! `Error::MetalUnavailable` for explicit Metal-only work.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

#[cfg(target_os = "macos")]
mod abi;
mod batch;
#[cfg(target_os = "macos")]
mod buffers;
#[cfg(target_os = "macos")]
mod compute;
mod encode;
mod routing;
mod session;
/// Viewport planning and composition helpers for JPEG decode surfaces.
pub mod viewport;

use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;

use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, BackendRequest, BufferError, CodecError,
    DecodeOutcome, DeviceMemoryRange, DeviceSubmission, DeviceSurface, Downscale, ImageCodec,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
};
use j2k_jpeg::{
    adapter::{
        build_fast420_packet, build_fast420_packet_for_decoder, build_fast422_packet,
        build_fast422_packet_for_decoder, build_fast444_packet, build_fast444_packet_for_decoder,
        decoder_bytes, JpegFast420PacketV1, JpegFast422PacketV1, JpegFast444PacketV1,
    },
    Decoder as CpuDecoder, DecoderContext as CpuDecoderContext, JpegError, JpegView,
    ScratchPool as CpuScratchPool, Warning as CpuWarning,
};
#[cfg(target_os = "macos")]
use j2k_metal_support::{system_default_device, MetalSupportError};

pub use encode::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile,
};

#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{
    Buffer, BufferRef, CommandBuffer, Device, MTLPixelFormat, MTLResourceOptions, MTLStorageMode,
    MTLTextureType, MTLTextureUsage, Texture, TextureDescriptor, TextureRef,
};

#[derive(Debug, thiserror::Error)]
/// Errors returned by the Metal JPEG backend.
pub enum Error {
    /// Error returned by the CPU JPEG parser or fallback decoder.
    #[error(transparent)]
    Decode(#[from] JpegError),
    /// Error returned while assembling a baseline JPEG encode result.
    #[error(transparent)]
    Encode(#[from] j2k_jpeg::JpegEncodeError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is not supported by this crate.
    #[error("backend request {request:?} is not supported by j2k-jpeg-metal")]
    UnsupportedBackend {
        /// Backend requested by the caller.
        request: BackendRequest,
    },
    /// A Metal-specific request is structurally unsupported.
    #[error("unsupported JPEG Metal request: {reason}")]
    UnsupportedMetalRequest {
        /// Static reason describing the rejected request.
        reason: &'static str,
    },
    /// Metal is not available on the current host.
    #[error("Metal is unavailable on this host")]
    MetalUnavailable,
    /// Metal runtime creation or device setup failed.
    #[error("Metal runtime error: {message}")]
    MetalRuntime {
        /// Runtime error message.
        message: String,
    },
    /// Metal kernel launch, validation, or completion failed.
    #[error("Metal kernel error: {message}")]
    MetalKernel {
        /// Kernel error message.
        message: String,
    },
    /// Shared Metal backend state was poisoned by a prior panic.
    #[error("Metal state `{state}` is poisoned")]
    MetalStatePoisoned {
        /// Name of the poisoned state.
        state: &'static str,
    },
}

impl CodecError for Error {
    fn is_truncated(&self) -> bool {
        matches!(self, Self::Decode(inner) if inner.is_truncated())
    }

    fn is_not_implemented(&self) -> bool {
        matches!(self, Self::Decode(inner) if inner.is_not_implemented())
    }

    fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::UnsupportedBackend { .. }
                | Self::MetalUnavailable
                | Self::UnsupportedMetalRequest { .. }
        ) || matches!(self, Self::Decode(inner) if inner.is_unsupported())
    }

    fn is_buffer_error(&self) -> bool {
        matches!(self, Self::Buffer(_))
            || matches!(self, Self::Decode(inner) if inner.is_buffer_error())
    }
}

#[derive(Clone)]
pub(crate) enum Storage {
    Host(Vec<u8>),
    #[cfg(target_os = "macos")]
    Metal {
        buffer: Buffer,
        offset: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Where a decoded surface is currently resident.
pub enum SurfaceResidency {
    /// Pixel bytes are resident in host memory.
    Host,
    /// Pixel bytes were produced directly by a Metal decode kernel.
    MetalResidentDecode,
    /// Pixel bytes were decoded on CPU and uploaded into a Metal buffer.
    CpuStagedMetalUpload,
}

#[derive(Clone)]
/// Decoded image surface returned by the JPEG Metal backend.
pub struct Surface {
    backend: BackendKind,
    residency: SurfaceResidency,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    storage: Storage,
}

impl Surface {
    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    /// Current residency for the surface bytes.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Return the tightly packed surface bytes.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.storage {
            Storage::Host(bytes) => bytes,
            #[cfg(target_os = "macos")]
            Storage::Metal { buffer, offset } => {
                let len = self.byte_len();
                // SAFETY: Metal surface byte views are bounded by validated dimensions and formats.
                unsafe {
                    core::slice::from_raw_parts(buffer.contents().cast::<u8>().add(*offset), len)
                }
            }
        }
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        copy_tight_pixels_to_strided_output(self.as_bytes(), self.dimensions, self.fmt, out, stride)
            .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    /// Return the Metal buffer and byte offset when the surface is Metal-backed.
    pub fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
        match &self.storage {
            Storage::Metal { buffer, offset } => Some((buffer, *offset)),
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
            storage: Storage::Metal { buffer, offset },
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
            storage: Storage::Metal { buffer, offset },
        }
    }
}

impl DeviceSurface for Surface {
    fn backend_kind(&self) -> BackendKind {
        self.backend
    }

    fn residency(&self) -> j2k_core::SurfaceResidency {
        match self.residency {
            SurfaceResidency::Host => j2k_core::SurfaceResidency::Host,
            SurfaceResidency::MetalResidentDecode => {
                j2k_core::SurfaceResidency::MetalResidentDecode
            }
            SurfaceResidency::CpuStagedMetalUpload => {
                j2k_core::SurfaceResidency::CpuStagedMetalUpload
            }
        }
    }

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.pitch_bytes * self.dimensions.1 as usize
    }

    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        match &self.storage {
            Storage::Host(_) => None,
            #[cfg(target_os = "macos")]
            Storage::Metal { buffer, offset } => Some(DeviceMemoryRange::new(
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
#[derive(Clone)]
pub struct ResidentPrivateJpegTile {
    pub buffer: Buffer,
    pub byte_offset: usize,
    pub dimensions: (u32, u32),
    pub pixel_format: PixelFormat,
    pub pitch_bytes: usize,
    pub status_buffer: Buffer,
    pub command_buffer: CommandBuffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable caller-owned Metal buffer for full-tile JPEG batch output.
pub struct MetalBatchOutputBuffer {
    buffer: Buffer,
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
            dimensions,
            fmt,
            pitch_bytes: row_bytes,
            tile_stride_bytes,
            tile_capacity,
        })
    }

    /// Backing Metal buffer.
    pub fn buffer(&self) -> &BufferRef {
        self.buffer.as_ref()
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

    /// Return a reusable output texture by tile slot.
    pub fn texture(&self, index: usize) -> Option<&TextureRef> {
        self.textures.get(index).map(std::convert::AsRef::as_ref)
    }

    pub(crate) fn clone_texture(&self, index: usize) -> Option<Texture> {
        self.textures.get(index).cloned()
    }

    pub(crate) fn clone_slots(&self, indices: &[usize]) -> Result<Self, Error> {
        let mut textures = Vec::with_capacity(indices.len());
        for &index in indices {
            textures.push(
                self.clone_texture(index)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal batch texture output slot was missing".to_string(),
                    })?,
            );
        }
        Ok(Self {
            textures,
            dimensions: self.dimensions,
            fmt: self.fmt,
            metal_fmt: self.metal_fmt,
        })
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// One decoded JPEG tile resident in a caller-owned Metal texture.
pub struct MetalTextureTile {
    texture: Texture,
    dimensions: (u32, u32),
    fmt: PixelFormat,
}

#[cfg(target_os = "macos")]
impl MetalTextureTile {
    pub(crate) fn new(texture: Texture, dimensions: (u32, u32), fmt: PixelFormat) -> Self {
        Self {
            texture,
            dimensions,
            fmt,
        }
    }

    /// Backing Metal texture containing the decoded tile.
    pub fn texture(&self) -> &TextureRef {
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
#[derive(Clone)]
/// Reusable Metal device session for decode and encode submissions.
pub struct MetalBackendSession {
    device: Device,
    runtime: Arc<OnceLock<Result<compute::MetalRuntime, MetalSupportError>>>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    /// Create a session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            runtime: Arc::new(OnceLock::new()),
        }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        system_default_device()
            .map(Self::new)
            .map_err(|error| compute::runtime_initialization_error(&error))
    }

    /// Metal device used by this session.
    pub fn device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }
}

#[cfg(target_os = "macos")]
impl j2k_core::AcceleratorSession for MetalBackendSession {
    fn backend_kind(&self) -> BackendKind {
        BackendKind::Metal
    }
}

#[cfg(target_os = "macos")]
impl core::fmt::Debug for MetalBackendSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalBackendSession")
            .field("device", &self.device.name())
            .field("runtime_initialized", &self.runtime.get().is_some())
            .finish()
    }
}

#[cfg(not(target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default)]
/// Placeholder Metal session for non-macOS builds.
pub struct MetalBackendSession {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
impl MetalBackendSession {
    /// Return `Error::MetalUnavailable` on hosts without Metal support.
    pub fn system_default() -> Result<Self, Error> {
        Err(Error::MetalUnavailable)
    }
}

#[derive(Default)]
/// Shared batching session used by `JpegTileBatch` and submit APIs.
pub struct MetalSession {
    shared: session::SharedSession,
}

impl MetalSession {
    /// Create a tile batching session that reuses an existing Metal backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend_session: MetalBackendSession) -> Self {
        Self {
            shared: session::SharedSession(Arc::new(Mutex::new(
                session::SessionState::with_backend_session(backend_session),
            ))),
        }
    }

    /// Number of Metal or emulated submissions flushed through this session.
    pub fn submissions(&self) -> Result<u64, Error> {
        Ok(self.shared.lock()?.submissions)
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .finish()
    }
}

/// Convenience wrapper for submitting a group of JPEG tiles to one decoder
/// session.
///
/// The batch preserves submission order and lets compatible requests share a
/// Metal submission. Callers still own slide metadata, level selection, cache
/// policy, and viewport planning.
#[derive(Default)]
pub struct JpegTileBatch {
    session: MetalSession,
    submissions: Vec<batch::MetalSubmission>,
}

impl JpegTileBatch {
    /// Create an empty tile batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty tile batch with capacity for `capacity` submissions.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            submissions: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Number of queued tile requests.
    pub fn len(&self) -> usize {
        self.submissions.len()
    }

    /// Whether the batch has no queued tile requests.
    pub fn is_empty(&self) -> bool {
        self.submissions.is_empty()
    }

    /// Number of Metal session submissions already flushed.
    ///
    /// Queued requests normally do not increment this until `decode_all` waits
    /// on the first result.
    pub fn submissions(&self) -> Result<u64, Error> {
        self.session.submissions()
    }

    /// Queue a full-tile decode request, copying the compressed tile bytes into
    /// the batch.
    pub fn push_tile(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile(Arc::<[u8]>::from(input), fmt, backend)
    }

    /// Queue a full-tile decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_request(input, fmt, backend, batch::BatchOp::Full)
    }

    /// Queue a region decode request, copying the compressed tile bytes into
    /// the batch.
    pub fn push_tile_region(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile_region(Arc::<[u8]>::from(input), fmt, roi, backend)
    }

    /// Queue a region decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile_region(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_request(input, fmt, backend, batch::BatchOp::Region(roi))
    }

    /// Queue a scaled decode request, copying the compressed tile bytes into
    /// the batch.
    pub fn push_tile_scaled(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile_scaled(Arc::<[u8]>::from(input), fmt, scale, backend)
    }

    /// Queue a scaled decode request backed by shared compressed tile bytes.
    pub fn push_shared_tile_scaled(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_request(input, fmt, backend, batch::BatchOp::Scaled(scale))
    }

    /// Queue a region decode at reduced resolution, copying the compressed tile
    /// bytes into the batch.
    pub fn push_tile_region_scaled(
        &mut self,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_tile_region_scaled(Arc::<[u8]>::from(input), fmt, roi, scale, backend)
    }

    /// Queue a region decode at reduced resolution backed by shared compressed
    /// tile bytes.
    pub fn push_shared_tile_region_scaled(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        self.push_shared_request(
            input,
            fmt,
            backend,
            batch::BatchOp::RegionScaled { roi, scale },
        )
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(self.submissions.len());
        for submission in self.submissions {
            surfaces.push(submission.wait()?);
        }
        Ok(surfaces)
    }

    fn push_shared_request(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        backend: BackendRequest,
        op: batch::BatchOp,
    ) -> Result<usize, Error> {
        let slot = self.submissions.len();
        let submission = {
            let mut state = self.session.shared.lock()?;
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            let slot = state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                op,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
            batch::MetalSubmission {
                session: self.session.shared.clone(),
                slot,
            }
        };
        self.submissions.push(submission);
        Ok(slot)
    }
}

/// JPEG decoder that can return host or Metal-resident surfaces.
pub struct Decoder<'a> {
    inner: CpuDecoder<'a>,
    source: Arc<[u8]>,
    fast444_packet: Option<Arc<JpegFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegFast420PacketV1>>,
}

impl<'a> Decoder<'a> {
    /// Parse a JPEG byte slice into a decoder with any available Metal packets.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        let inner = CpuDecoder::new(input)?;
        Ok(Self {
            fast444_packet: build_fast444_packet(input).ok().map(Arc::new),
            fast422_packet: build_fast422_packet(input).ok().map(Arc::new),
            fast420_packet: build_fast420_packet(input).ok().map(Arc::new),
            inner,
            source: Arc::<[u8]>::from(input),
        })
    }

    /// Create a decoder from an already parsed JPEG view.
    pub fn from_view(view: JpegView<'a>) -> Result<Self, Error> {
        let inner = CpuDecoder::from_view(view)?;
        let source = Arc::<[u8]>::from(decoder_bytes(&inner));
        let fast444_packet = build_fast444_packet_for_decoder(&inner).ok().map(Arc::new);
        let fast422_packet = build_fast422_packet_for_decoder(&inner).ok().map(Arc::new);
        let fast420_packet = build_fast420_packet_for_decoder(&inner).ok().map(Arc::new);
        Ok(Self {
            inner,
            source,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        })
    }

    /// Borrow the underlying CPU JPEG decoder.
    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast444_packet(&self) -> Option<&JpegFast444PacketV1> {
        self.fast444_packet.as_deref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast422_packet(&self) -> Option<&JpegFast422PacketV1> {
        self.fast422_packet.as_deref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn fast420_packet(&self) -> Option<&JpegFast420PacketV1> {
        self.fast420_packet.as_deref()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn rgb8_region_scaled_metal_request(
        &self,
        roi: Rect,
        scale: Downscale,
    ) -> batch::QueuedRequest {
        self.rgb8_metal_request(batch::BatchOp::RegionScaled { roi, scale })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn rgb8_metal_request(&self, op: batch::BatchOp) -> batch::QueuedRequest {
        batch::QueuedRequest::new_shared(
            Arc::clone(&self.source),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            op,
            self.fast444_packet.clone(),
            self.fast422_packet.clone(),
            self.fast420_packet.clone(),
        )
    }

    /// Consume this wrapper and return the underlying CPU JPEG decoder.
    pub fn into_inner(self) -> CpuDecoder<'a> {
        self.inner
    }

    /// Decode a region at the requested scale into a device surface when possible.
    pub fn decode_region_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let mut pool = CpuScratchPool::new();
        decode_region_scaled_surface_from_decoder(
            &self.inner,
            &mut pool,
            fmt,
            roi,
            scale,
            backend,
            self.fast444_packet.as_deref(),
            self.fast422_packet.as_deref(),
            self.fast420_packet.as_deref(),
        )
    }

    /// Decode a full image into a device surface using a reusable Metal session.
    pub fn decode_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let mut pool = CpuScratchPool::new();
            let decision = choose_route(
                &self.inner,
                BackendRequest::Metal,
                fmt,
                batch::BatchOp::Full,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            match decision {
                routing::RouteDecision::MetalKernel => {
                    reject_cpu_staged_metal_upload(compute::decode_to_surface_with_session(
                        &self.inner,
                        &mut pool,
                        fmt,
                        self.fast444_packet.as_deref(),
                        self.fast422_packet.as_deref(),
                        self.fast420_packet.as_deref(),
                        session,
                    )?)
                }
                routing::RouteDecision::CpuHost
                | routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. }
                | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = session;
            let decision = choose_route(
                &self.inner,
                BackendRequest::Metal,
                fmt,
                batch::BatchOp::Full,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_private_rgb8_tile_with_session(
        &mut self,
        session: &MetalBackendSession,
    ) -> Result<ResidentPrivateJpegTile, Error> {
        let decision = choose_route(
            &self.inner,
            BackendRequest::Metal,
            PixelFormat::Rgb8,
            batch::BatchOp::Full,
            self.fast444_packet.as_deref(),
            self.fast422_packet.as_deref(),
            self.fast420_packet.as_deref(),
        );
        if let Some(err) = routing::decision_error(decision) {
            return Err(err);
        }
        match decision {
            routing::RouteDecision::MetalKernel => compute::decode_private_rgb8_tile_with_session(
                &self.inner,
                self.fast444_packet.as_deref(),
                self.fast422_packet.as_deref(),
                self.fast420_packet.as_deref(),
                session,
            ),
            routing::RouteDecision::CpuHost
            | routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. }
            | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
        }
    }
}

impl ImageCodec for Decoder<'_> {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl<'a> ImageDecode<'a> for Decoder<'a> {
    type View = JpegView<'a>;

    fn inspect(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?.to_core_info())
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(JpegView::parse(input)?)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Self::from_view(view)
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self.inner.decode_into(out, stride, fmt)?.into())
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_into_with_scratch(pool, out, stride, fmt)?
            .into())
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_region_into_with_scratch(pool, out, stride, fmt, roi.into())?
            .into())
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_scaled_into_with_scratch(pool, out, stride, fmt, scale)?
            .into())
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self
            .inner
            .decode_region_scaled_into_with_scratch(pool, out, stride, fmt, roi.into(), scale)?
            .into())
    }
}

impl<'a> ImageDecodeDevice<'a> for Decoder<'a> {
    type DeviceSurface = Surface;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// JPEG codec marker used by J2K's generic decode traits.
pub struct Codec;

#[cfg(target_os = "macos")]
struct Rgb8MetalBatchPlan {
    requests: Vec<batch::QueuedRequest>,
    output_dimensions: Option<(u32, u32)>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Preflight report for RGB8 JPEG Metal resident decoder batches.
pub struct JpegMetalResidentBatchReport {
    /// Requested decode operation.
    pub op: j2k_jpeg::JpegDecodeOp,
    /// Number of decoder tiles in the batch.
    pub tile_count: usize,
    /// Required output dimensions when the batch is eligible and shape-compatible.
    pub output_dimensions: Option<(u32, u32)>,
    /// Whether the batch can use reusable RGB8 Metal resident output.
    pub eligibility: j2k_jpeg::JpegBackendEligibility,
}

#[cfg(target_os = "macos")]
impl JpegMetalResidentBatchReport {
    /// Required number of tile slots in caller-owned Metal output.
    #[must_use]
    pub fn required_tile_capacity(&self) -> usize {
        self.tile_count
    }
}

#[cfg(target_os = "macos")]
fn report_required_output_dimensions(
    report: &JpegMetalResidentBatchReport,
) -> Result<Option<(u32, u32)>, Error> {
    if !report.eligibility.eligible {
        return Err(Error::UnsupportedMetalRequest {
            reason: report
                .eligibility
                .reason
                .unwrap_or("JPEG Metal resident batch report is not eligible"),
        });
    }
    if report.tile_count == 0 {
        return Ok(None);
    }
    report
        .output_dimensions
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal resident batch report is missing output dimensions",
        })
        .map(Some)
}

#[cfg(target_os = "macos")]
fn rgb8_metal_output_dimensions_for_op(
    full_dimensions: (u32, u32),
    op: j2k_jpeg::JpegDecodeOp,
) -> Option<(u32, u32)> {
    match op {
        j2k_jpeg::JpegDecodeOp::Full => Some(full_dimensions),
        j2k_jpeg::JpegDecodeOp::Scaled(scale) => Some(scaled_dims(full_dimensions, scale)),
        j2k_jpeg::JpegDecodeOp::RegionScaled { roi, scale } => {
            let scaled = Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            }
            .scaled_covering(scale);
            Some((scaled.w, scaled.h))
        }
        j2k_jpeg::JpegDecodeOp::Region(_) => None,
    }
}

#[cfg(target_os = "macos")]
fn decoder_resident_sampling_family(decoder: &Decoder<'_>) -> batch::SamplingFamily {
    if decoder.fast420_packet().is_some() {
        batch::SamplingFamily::Fast420
    } else if decoder.fast422_packet().is_some() {
        batch::SamplingFamily::Fast422
    } else if decoder.fast444_packet().is_some() {
        batch::SamplingFamily::Fast444
    } else {
        batch::SamplingFamily::Other
    }
}

#[cfg(target_os = "macos")]
fn decoder_resident_restart_interval_mcus(decoder: &Decoder<'_>) -> u32 {
    if let Some(packet) = decoder.fast420_packet() {
        packet.restart_interval_mcus
    } else if let Some(packet) = decoder.fast422_packet() {
        packet.restart_interval_mcus
    } else if let Some(packet) = decoder.fast444_packet() {
        packet.restart_interval_mcus
    } else {
        0
    }
}

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

/// Inputs for a batched RGB8 Metal decode: raw JPEG bytes or pre-parsed
/// decoders that carry cached Metal fast-packet state.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub enum Rgb8MetalBatchSource<'a, 'b> {
    /// Raw JPEG byte streams, parsed per call.
    Bytes(&'a [&'a [u8]]),
    /// Already parsed `Decoder` wrappers; reuses their cached Metal
    /// fast-packet state when building the resident batch request.
    Decoders(&'a [&'a Decoder<'b>]),
}

#[cfg(target_os = "macos")]
impl Rgb8MetalBatchSource<'_, '_> {
    fn is_empty(&self) -> bool {
        match self {
            Rgb8MetalBatchSource::Bytes(inputs) => inputs.is_empty(),
            Rgb8MetalBatchSource::Decoders(decoders) => decoders.is_empty(),
        }
    }
}

/// Geometry op applied to every tile of a batched RGB8 Metal decode.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rgb8MetalBatchOp {
    /// Full-tile decode at native dimensions.
    Full,
    /// Whole-tile downscale (half, quarter, or eighth).
    Scaled(Downscale),
    /// Scaled decode of one region, shared by every tile in the batch.
    RegionScaled {
        /// Region of interest to decode from every source tile.
        roi: Rect,
        /// Downscale factor applied to the selected region.
        scale: Downscale,
    },
}

/// A batched RGB8 Metal decode request: what to decode and how.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub struct Rgb8MetalBatchRequest<'a, 'b> {
    /// Source JPEG bytes or prepared decoders for the batch.
    pub source: Rgb8MetalBatchSource<'a, 'b>,
    /// Geometry operation applied to each source tile.
    pub op: Rgb8MetalBatchOp,
}

/// Caller-owned Metal buffer target for a batched RGB8 decode.
#[cfg(target_os = "macos")]
pub enum MetalBufferBatchTarget<'a> {
    /// Reuse the buffer as-is; its shape must already fit the batch.
    Reusable(&'a MetalBatchOutputBuffer),
    /// Grow the buffer to fit the batch before decoding.
    Resizable(&'a mut MetalBatchOutputBuffer),
}

/// Caller-owned Metal RGBA8 texture target for a batched RGB8 decode.
#[cfg(target_os = "macos")]
pub enum MetalTextureBatchTarget<'a> {
    /// Reuse the texture set as-is; its shape must already fit the batch.
    Reusable(&'a MetalBatchTextureOutput),
    /// Grow the texture set to fit the batch before decoding.
    Resizable(&'a mut MetalBatchTextureOutput),
}

impl Codec {
    #[cfg(target_os = "macos")]
    /// Inspect a cached RGB8 decoder batch for reusable Metal resident output.
    ///
    /// The report exposes whether the batch is resident-output eligible and,
    /// when eligible, the exact output dimensions and tile capacity callers
    /// should allocate before dispatch.
    pub fn inspect_rgb8_decoder_batch_metal_output(
        decoders: &[&Decoder<'_>],
        op: j2k_jpeg::JpegDecodeOp,
    ) -> JpegMetalResidentBatchReport {
        if decoders.is_empty() {
            return JpegMetalResidentBatchReport {
                op,
                tile_count: 0,
                output_dimensions: None,
                eligibility: j2k_jpeg::JpegBackendEligibility {
                    eligible: true,
                    reason: None,
                },
            };
        }

        let mut output_dimensions = None;
        let mut sampling_family = None;
        for decoder in decoders {
            let request = j2k_jpeg::JpegCapabilityRequest {
                op,
                fmt: PixelFormat::Rgb8,
            };
            let report = j2k_jpeg::JpegCapabilityReport::for_decoder(decoder.inner(), request);
            let eligibility = report.metal_resident_rgb8_batch_output();
            if !eligibility.eligible {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility,
                };
            }

            if decoder.fast444_packet().is_none()
                && decoder.fast422_packet().is_none()
                && decoder.fast420_packet().is_none()
            {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility: j2k_jpeg::JpegBackendEligibility {
                        eligible: false,
                        reason: Some(
                            "JPEG Metal reusable resident batch output requires cached fast-packet state",
                        ),
                    },
                };
            }

            let Some(dimensions) =
                rgb8_metal_output_dimensions_for_op(decoder.inner().info().dimensions, op)
            else {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility,
                };
            };
            if let Some(first) = output_dimensions {
                if first != dimensions {
                    return JpegMetalResidentBatchReport {
                        op,
                        tile_count: decoders.len(),
                        output_dimensions: None,
                        eligibility: j2k_jpeg::JpegBackendEligibility {
                            eligible: false,
                            reason: Some(
                                "JPEG Metal reusable RGB8 batch output requires matching output dimensions",
                            ),
                        },
                    };
                }
            } else {
                output_dimensions = Some(dimensions);
            }

            let decoder_sampling_family = decoder_resident_sampling_family(decoder);
            if let Some(first) = sampling_family {
                if first != decoder_sampling_family {
                    return JpegMetalResidentBatchReport {
                        op,
                        tile_count: decoders.len(),
                        output_dimensions: None,
                        eligibility: j2k_jpeg::JpegBackendEligibility {
                            eligible: false,
                            reason: Some(
                                "JPEG Metal reusable resident batch output requires one batch to use the same fast-packet sampling family",
                            ),
                        },
                    };
                }
            } else {
                sampling_family = Some(decoder_sampling_family);
            }

            if op == j2k_jpeg::JpegDecodeOp::Full
                && matches!(
                    decoder_sampling_family,
                    batch::SamplingFamily::Fast422 | batch::SamplingFamily::Fast444
                )
                && decoder_resident_restart_interval_mcus(decoder) != 0
            {
                return JpegMetalResidentBatchReport {
                    op,
                    tile_count: decoders.len(),
                    output_dimensions: None,
                    eligibility: j2k_jpeg::JpegBackendEligibility {
                        eligible: false,
                        reason: Some(
                            "JPEG Metal reusable resident batch output does not support restart-coded full-tile 4:2:2 or 4:4:4 batches",
                        ),
                    },
                };
            }
        }

        JpegMetalResidentBatchReport {
            op,
            tile_count: decoders.len(),
            output_dimensions,
            eligibility: j2k_jpeg::JpegBackendEligibility {
                eligible: true,
                reason: None,
            },
        }
    }

    #[cfg(target_os = "macos")]
    fn observe_rgb8_batch_output_dimensions(
        first_output_dimensions: &mut Option<(u32, u32)>,
        output_dimensions: (u32, u32),
    ) -> Result<(), Error> {
        if let Some(first) = *first_output_dimensions {
            if first != output_dimensions {
                return Err(Error::UnsupportedMetalRequest {
                    reason:
                        "JPEG Metal reusable RGB8 batch output requires matching output dimensions",
                });
            }
        } else {
            *first_output_dimensions = Some(output_dimensions);
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn rgb8_metal_batch_requests(
        inputs: &[&[u8]],
        mut op_for_decoder: impl FnMut(&CpuDecoder<'_>) -> batch::BatchOp,
    ) -> Result<Vec<batch::QueuedRequest>, Error> {
        let plan = Self::rgb8_metal_batch_requests_with_output_dimensions(inputs, |decoder| {
            (op_for_decoder(decoder), decoder.info().dimensions)
        })?;
        Ok(plan.requests)
    }

    #[cfg(target_os = "macos")]
    fn rgb8_metal_batch_requests_with_output_dimensions(
        inputs: &[&[u8]],
        mut op_and_dimensions_for_decoder: impl FnMut(&CpuDecoder<'_>) -> (batch::BatchOp, (u32, u32)),
    ) -> Result<Rgb8MetalBatchPlan, Error> {
        let mut state = session::SessionState::default();
        let mut requests = Vec::with_capacity(inputs.len());
        let mut first_output_dimensions = None;
        for input in inputs {
            let decoder = CpuDecoder::new(input)?;
            let (op, output_dimensions) = op_and_dimensions_for_decoder(&decoder);
            Self::observe_rgb8_batch_output_dimensions(
                &mut first_output_dimensions,
                output_dimensions,
            )?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, BackendRequest::Metal);
            requests.push(batch::QueuedRequest::new_shared(
                input,
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                op,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
        }
        Ok(Rgb8MetalBatchPlan {
            requests,
            output_dimensions: first_output_dimensions,
        })
    }

    #[cfg(target_os = "macos")]
    fn rgb8_metal_decoder_batch_requests_with_output_dimensions(
        decoders: &[&Decoder<'_>],
        mut op_and_dimensions_for_decoder: impl FnMut(&Decoder<'_>) -> (batch::BatchOp, (u32, u32)),
    ) -> Result<Rgb8MetalBatchPlan, Error> {
        let mut requests = Vec::with_capacity(decoders.len());
        let mut first_output_dimensions = None;
        for decoder in decoders {
            let (op, output_dimensions) = op_and_dimensions_for_decoder(decoder);
            Self::observe_rgb8_batch_output_dimensions(
                &mut first_output_dimensions,
                output_dimensions,
            )?;
            requests.push(decoder.rgb8_metal_request(op));
        }
        Ok(Rgb8MetalBatchPlan {
            requests,
            output_dimensions: first_output_dimensions,
        })
    }
    #[cfg(target_os = "macos")]
    fn rgb8_batch_op_and_dimensions(
        op: Rgb8MetalBatchOp,
        dimensions: (u32, u32),
    ) -> (batch::BatchOp, (u32, u32)) {
        match op {
            Rgb8MetalBatchOp::Full => (batch::BatchOp::Full, dimensions),
            Rgb8MetalBatchOp::Scaled(scale) => {
                let (w, h) = dimensions;
                (
                    batch::BatchOp::RegionScaled {
                        roi: Rect { x: 0, y: 0, w, h },
                        scale,
                    },
                    scaled_dims((w, h), scale),
                )
            }
            Rgb8MetalBatchOp::RegionScaled { roi, scale } => {
                let scaled = roi.scaled_covering(scale);
                (
                    batch::BatchOp::RegionScaled { roi, scale },
                    (scaled.w, scaled.h),
                )
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn rgb8_batch_jpeg_decode_op(op: Rgb8MetalBatchOp) -> j2k_jpeg::JpegDecodeOp {
        match op {
            Rgb8MetalBatchOp::Full => j2k_jpeg::JpegDecodeOp::Full,
            Rgb8MetalBatchOp::Scaled(scale) => j2k_jpeg::JpegDecodeOp::Scaled(scale),
            Rgb8MetalBatchOp::RegionScaled { roi, scale } => j2k_jpeg::JpegDecodeOp::RegionScaled {
                roi: roi.into(),
                scale,
            },
        }
    }

    #[cfg(target_os = "macos")]
    fn plan_rgb8_metal_batch(
        source: Rgb8MetalBatchSource<'_, '_>,
        op: Rgb8MetalBatchOp,
        track_output_dimensions: bool,
    ) -> Result<(Rgb8MetalBatchPlan, usize), Error> {
        match source {
            Rgb8MetalBatchSource::Bytes(inputs) => {
                if track_output_dimensions {
                    Self::rgb8_metal_batch_requests_with_output_dimensions(inputs, |decoder| {
                        Self::rgb8_batch_op_and_dimensions(op, decoder.info().dimensions)
                    })
                    .map(|plan| (plan, inputs.len()))
                } else {
                    Self::rgb8_metal_batch_requests(inputs, |decoder| {
                        Self::rgb8_batch_op_and_dimensions(op, decoder.info().dimensions).0
                    })
                    .map(|requests| {
                        (
                            Rgb8MetalBatchPlan {
                                requests,
                                output_dimensions: None,
                            },
                            inputs.len(),
                        )
                    })
                }
            }
            Rgb8MetalBatchSource::Decoders(decoders) => {
                Self::rgb8_metal_decoder_batch_requests_with_output_dimensions(
                    decoders,
                    |decoder| {
                        Self::rgb8_batch_op_and_dimensions(op, decoder.inner().info().dimensions)
                    },
                )
                .map(|plan| (plan, decoders.len()))
            }
        }
    }

    #[cfg(target_os = "macos")]
    const fn rgb8_buffer_batch_unsupported_reason(op: Rgb8MetalBatchOp) -> &'static str {
        match op {
            Rgb8MetalBatchOp::Full => {
                "JPEG Metal reusable batch output currently supports batchable full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs"
            }
            Rgb8MetalBatchOp::Scaled(_) => {
                "JPEG Metal reusable scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with half, quarter, or eighth scaling"
            }
            Rgb8MetalBatchOp::RegionScaled { .. } => {
                "JPEG Metal reusable region-scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with matching output shapes"
            }
        }
    }

    #[cfg(target_os = "macos")]
    const fn rgb8_texture_batch_unsupported_reason(op: Rgb8MetalBatchOp) -> &'static str {
        match op {
            Rgb8MetalBatchOp::Full => {
                "JPEG Metal texture batch output currently supports batchable full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs"
            }
            Rgb8MetalBatchOp::Scaled(_) => {
                "JPEG Metal texture scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with half, quarter, or eighth scaling"
            }
            Rgb8MetalBatchOp::RegionScaled { .. } => {
                "JPEG Metal texture region-scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with matching output shapes"
            }
        }
    }

    #[cfg(target_os = "macos")]
    /// Decode a batched RGB8 JPEG request into a caller-owned Metal buffer.
    ///
    /// This is the single buffer-output entry point for full, scaled, and
    /// region-scaled batches sourced from raw bytes or pre-parsed decoders;
    /// `MetalBufferBatchTarget::Resizable` grows the buffer to fit before
    /// decoding.
    pub fn decode_rgb8_batch_into_buffer_with_session(
        request: Rgb8MetalBatchRequest<'_, '_>,
        target: MetalBufferBatchTarget<'_>,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        if request.source.is_empty() {
            return Ok(Vec::new());
        }

        let resizable = matches!(target, MetalBufferBatchTarget::Resizable(_));
        let (plan, tile_count) =
            Self::plan_rgb8_metal_batch(request.source, request.op, resizable)?;
        let output: &MetalBatchOutputBuffer = match target {
            MetalBufferBatchTarget::Reusable(output) => output,
            MetalBufferBatchTarget::Resizable(output) => {
                if let Rgb8MetalBatchSource::Decoders(decoders) = request.source {
                    let report = Self::inspect_rgb8_decoder_batch_metal_output(
                        decoders,
                        Self::rgb8_batch_jpeg_decode_op(request.op),
                    );
                    output.ensure_rgb8_batch_report(session, &report)?;
                }
                let Some(output_dimensions) = plan.output_dimensions else {
                    return Ok(Vec::new());
                };
                output.ensure_rgb8_tiles(session, output_dimensions, tile_count)?;
                output
            }
        };

        let results = match request.op {
            Rgb8MetalBatchOp::Full => compute::decode_full_rgb8_batch_into_output_with_session(
                &plan.requests,
                output,
                session,
            )?,
            Rgb8MetalBatchOp::Scaled(_) | Rgb8MetalBatchOp::RegionScaled { .. } => {
                compute::decode_region_scaled_rgb8_batch_into_output_with_session(
                    &plan.requests,
                    output,
                    session,
                )?
            }
        };
        results.ok_or(Error::UnsupportedMetalRequest {
            reason: Self::rgb8_buffer_batch_unsupported_reason(request.op),
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a batched RGB8 JPEG request into caller-owned Metal RGBA8 textures.
    ///
    /// This is the single texture-output entry point for full, scaled, and
    /// region-scaled batches sourced from raw bytes or pre-parsed decoders;
    /// `MetalTextureBatchTarget::Resizable` grows the texture set to fit
    /// before decoding.
    pub fn decode_rgb8_batch_into_textures_with_session(
        request: Rgb8MetalBatchRequest<'_, '_>,
        target: MetalTextureBatchTarget<'_>,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        if request.source.is_empty() {
            return Ok(Vec::new());
        }

        let resizable = matches!(target, MetalTextureBatchTarget::Resizable(_));
        let (plan, tile_count) =
            Self::plan_rgb8_metal_batch(request.source, request.op, resizable)?;
        let output: &MetalBatchTextureOutput = match target {
            MetalTextureBatchTarget::Reusable(output) => output,
            MetalTextureBatchTarget::Resizable(output) => {
                if let Rgb8MetalBatchSource::Decoders(decoders) = request.source {
                    let report = Self::inspect_rgb8_decoder_batch_metal_output(
                        decoders,
                        Self::rgb8_batch_jpeg_decode_op(request.op),
                    );
                    output.ensure_rgba8_batch_report(session, &report)?;
                }
                let Some(output_dimensions) = plan.output_dimensions else {
                    return Ok(Vec::new());
                };
                output.ensure_rgba8_tiles(session, output_dimensions, tile_count)?;
                output
            }
        };

        let results = match request.op {
            Rgb8MetalBatchOp::Full => compute::decode_full_rgb8_batch_into_textures_with_session(
                &plan.requests,
                output,
                session,
            )?,
            Rgb8MetalBatchOp::Scaled(_) | Rgb8MetalBatchOp::RegionScaled { .. } => {
                compute::decode_region_scaled_rgb8_batch_into_textures_with_session(
                    &plan.requests,
                    output,
                    session,
                )?
            }
        };
        results.ok_or(Error::UnsupportedMetalRequest {
            reason: Self::rgb8_texture_batch_unsupported_reason(request.op),
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a full-tile RGB8 JPEG decoder batch into resizable caller-owned
    /// Metal RGBA8 textures.
    ///
    /// Convenience wrapper over [`Codec::decode_rgb8_batch_into_textures_with_session`]
    /// for the resident whole-slide tile path.
    pub fn decode_rgb8_decoder_batch_into_resizable_metal_textures_with_session(
        decoders: &[&Decoder<'_>],
        output: &mut MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        Self::decode_rgb8_batch_into_textures_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Decoders(decoders),
                op: Rgb8MetalBatchOp::Full,
            },
            MetalTextureBatchTarget::Resizable(output),
            session,
        )
    }

    #[cfg(target_os = "macos")]
    /// Decode a region-scaled RGB8 JPEG batch into a resizable caller-owned
    /// Metal buffer.
    ///
    /// Convenience wrapper over [`Codec::decode_rgb8_batch_into_buffer_with_session`]
    /// for the viewport composition path.
    pub fn decode_rgb8_region_scaled_batch_into_resizable_metal_buffer_with_session(
        inputs: &[&[u8]],
        roi: Rect,
        scale: Downscale,
        output: &mut MetalBatchOutputBuffer,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        Self::decode_rgb8_batch_into_buffer_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Bytes(inputs),
                op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
            },
            MetalBufferBatchTarget::Resizable(output),
            session,
        )
    }

    #[cfg(target_os = "macos")]
    /// Decode a region-scaled RGB8 JPEG batch into resizable caller-owned
    /// Metal RGBA8 textures.
    ///
    /// Convenience wrapper over [`Codec::decode_rgb8_batch_into_textures_with_session`]
    /// for the viewport composition path.
    pub fn decode_rgb8_region_scaled_batch_into_resizable_metal_textures_with_session(
        inputs: &[&[u8]],
        roi: Rect,
        scale: Downscale,
        output: &mut MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        Self::decode_rgb8_batch_into_textures_with_session(
            Rgb8MetalBatchRequest {
                source: Rgb8MetalBatchSource::Bytes(inputs),
                op: Rgb8MetalBatchOp::RegionScaled { roi, scale },
            },
            MetalTextureBatchTarget::Resizable(output),
            session,
        )
    }

    #[allow(clippy::too_many_arguments)]
    /// Submit a scaled region tile decode into a reusable Metal session.
    pub fn submit_tile_region_scaled_to_device(
        ctx: &mut j2k_core::DecoderContext<CpuDecoderContext>,
        session: &mut MetalSession,
        pool: &mut CpuScratchPool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<<Self as TileBatchDecodeSubmit>::SubmittedSurface, Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.lock()?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::RegionScaled { roi, scale },
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}

impl<'a> ImageDecodeSubmit<'a> for Decoder<'a> {
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = batch::MetalSubmission;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Full,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Region(roi),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let fast444_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast444_packet.clone()
        } else {
            None
        };
        let fast422_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast422_packet.clone()
        } else {
            None
        };
        let fast420_packet = if matches!(backend, BackendRequest::Auto | BackendRequest::Metal) {
            self.fast420_packet.clone()
        } else {
            None
        };
        let slot = session
            .shared
            .lock()?
            .queue_request(batch::QueuedRequest::new_shared(
                Arc::clone(&self.source),
                fmt,
                backend,
                batch::BatchOp::RegionScaled { roi, scale },
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ));
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuDecoderContext;
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = batch::MetalSubmission;

    fn submit_tile_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.lock()?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::Full,
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_tile_region_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.lock()?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::Region(roi),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_tile_scaled_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.lock()?;
            let input = state.intern_input_slice(input);
            let (fast444_packet, fast422_packet, fast420_packet) =
                state.resolve_fast_packets(&input, backend);
            state.queue_request(batch::QueuedRequest::new_shared(
                input,
                fmt,
                backend,
                batch::BatchOp::Scaled(scale),
                fast444_packet,
                fast422_packet,
                fast420_packet,
            ))
        };
        Ok(batch::MetalSubmission {
            session: session.shared.clone(),
            slot,
        })
    }

    fn submit_tile_region_scaled_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        Codec::submit_tile_region_scaled_to_device(
            ctx, session, pool, input, fmt, roi, scale, backend,
        )
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;
}

impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuDecoderContext;
    type DeviceSurface = Surface;

    fn decode_tiles_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut session = MetalSession::default();
        let submissions = inputs
            .iter()
            .map(|input| {
                <Self as TileBatchDecodeSubmit>::submit_tile_to_device(
                    ctx,
                    &mut session,
                    pool,
                    input,
                    fmt,
                    backend,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        submissions
            .into_iter()
            .map(DeviceSubmission::wait)
            .collect()
    }
}

pub(crate) fn decode_surface_from_bytes(
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    fast444_packet: Option<Arc<JpegFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegFast420PacketV1>>,
) -> Result<Surface, Error> {
    let decoder = CpuDecoder::new(input)?;
    let mut pool = CpuScratchPool::new();
    let build_auto_packets =
        matches!(backend, BackendRequest::Auto) && decoder.info().restart_interval.is_some();
    let build_metal_packets = matches!(backend, BackendRequest::Metal);
    let fast444_packet = if build_auto_packets || build_metal_packets {
        fast444_packet.or_else(|| {
            build_fast444_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast422_packet = if build_auto_packets || build_metal_packets {
        fast422_packet.or_else(|| {
            build_fast422_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast420_packet = if build_auto_packets || build_metal_packets {
        fast420_packet.or_else(|| {
            build_fast420_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    decode_surface_from_decoder(
        &decoder,
        &mut pool,
        fmt,
        backend,
        op,
        fast444_packet.as_deref(),
        fast422_packet.as_deref(),
        fast420_packet.as_deref(),
    )
}

#[cfg(not(target_os = "macos"))]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn decode_compatible_batch(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let _ = requests;
    Ok(None)
}

#[allow(clippy::unnecessary_wraps)]
pub(crate) fn decode_compatible_batch_with_session(
    requests: &[batch::QueuedRequest],
    session: &mut session::SessionState,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    #[cfg(target_os = "macos")]
    {
        compute::decode_full_batch_to_surfaces_with_session_state(requests, session)
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = session;
        decode_compatible_batch(requests)
    }
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn decode_rgb8_batch_to_device_with_session(
    inputs: &[&[u8]],
    session: &MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if inputs.len() < 2 {
        return Ok(None);
    }

    let mut state = session::SessionState::default();
    let mut requests = Vec::with_capacity(inputs.len());
    for input in inputs {
        let input = state.intern_input_slice(input);
        let (fast444_packet, fast422_packet, fast420_packet) =
            state.resolve_fast_packets(&input, BackendRequest::Metal);
        requests.push(batch::QueuedRequest::new_shared(
            input,
            PixelFormat::Rgb8,
            BackendRequest::Metal,
            batch::BatchOp::Full,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        ));
    }

    compute::decode_full_batch_to_surfaces_with_session(&requests, session)
}

#[allow(clippy::too_many_arguments)]
fn decode_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    match op {
        batch::BatchOp::Full => match backend {
            BackendRequest::Cpu => decode_full_cpu_upload(decoder, pool, fmt),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(
                    decoder,
                    backend,
                    fmt,
                    op,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                );
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => decode_full_cpu_upload(decoder, pool, fmt),
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_to_surface(
                                decoder,
                                pool,
                                fmt,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (
                                decoder,
                                pool,
                                fmt,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            );
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Region(roi) => match backend {
            BackendRequest::Cpu => decode_region_cpu_upload(decoder, pool, fmt, roi),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(
                    decoder,
                    backend,
                    fmt,
                    op,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                );
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_region_cpu_upload(decoder, pool, fmt, roi)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_region_to_surface(
                                decoder,
                                pool,
                                fmt,
                                roi.into(),
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (
                                decoder,
                                pool,
                                fmt,
                                roi,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            );
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::Scaled(scale) => match backend {
            BackendRequest::Cpu => decode_scaled_cpu_upload(decoder, pool, fmt, scale),
            BackendRequest::Auto | BackendRequest::Metal => {
                let decision = choose_route(
                    decoder,
                    backend,
                    fmt,
                    op,
                    fast444_packet,
                    fast422_packet,
                    fast420_packet,
                );
                if let Some(err) = routing::decision_error(decision) {
                    return Err(err);
                }
                match decision {
                    routing::RouteDecision::CpuHost => {
                        decode_scaled_cpu_upload(decoder, pool, fmt, scale)
                    }
                    routing::RouteDecision::MetalKernel => {
                        #[cfg(target_os = "macos")]
                        {
                            reject_cpu_staged_metal_upload(compute::decode_scaled_to_surface(
                                decoder,
                                pool,
                                fmt,
                                scale,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            )?)
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            let _ = (
                                decoder,
                                pool,
                                fmt,
                                scale,
                                fast444_packet,
                                fast422_packet,
                                fast420_packet,
                            );
                            Err(Error::MetalUnavailable)
                        }
                    }
                    routing::RouteDecision::RejectExplicitMetal { .. }
                    | routing::RouteDecision::RejectUnsupportedBackend { .. }
                    | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
                }
            }
            BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
        },
        batch::BatchOp::RegionScaled { roi, scale } => decode_region_scaled_surface_from_decoder(
            decoder,
            pool,
            fmt,
            roi,
            scale,
            backend,
            fast444_packet,
            fast422_packet,
            fast420_packet,
        ),
    }
}

fn decode_full_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let dims = decoder.info().dimensions;
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_into_with_scratch(pool, &mut out, stride, fmt)?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_region_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let dims = (roi.w, roi.h);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_region_into_with_scratch(pool, &mut out, stride, fmt, roi.into())?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

fn decode_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Surface, Error> {
    let dims = scaled_dims(decoder.info().dimensions, scale);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_scaled_into_with_scratch(pool, &mut out, stride, fmt, scale)?;
    upload_surface(out, dims, fmt, BackendRequest::Cpu)
}

#[allow(clippy::too_many_arguments)]
fn decode_region_scaled_surface_from_decoder(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<Surface, Error> {
    match backend {
        BackendRequest::Cpu => {
            decode_region_scaled_cpu_upload(decoder, pool, fmt, roi, scale, BackendRequest::Cpu)
        }
        BackendRequest::Auto | BackendRequest::Metal => {
            let decision = choose_route(
                decoder,
                backend,
                fmt,
                batch::BatchOp::RegionScaled { roi, scale },
                fast444_packet,
                fast422_packet,
                fast420_packet,
            );
            if let Some(err) = routing::decision_error(decision) {
                return Err(err);
            }
            match decision {
                routing::RouteDecision::CpuHost => decode_region_scaled_cpu_upload(
                    decoder,
                    pool,
                    fmt,
                    roi,
                    scale,
                    BackendRequest::Cpu,
                ),
                routing::RouteDecision::MetalKernel => {
                    #[cfg(target_os = "macos")]
                    {
                        reject_cpu_staged_metal_upload(compute::decode_region_scaled_to_surface(
                            decoder,
                            pool,
                            fmt,
                            roi.into(),
                            scale,
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        )?)
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        let _ = (
                            decoder,
                            pool,
                            fmt,
                            roi,
                            scale,
                            fast444_packet,
                            fast422_packet,
                            fast420_packet,
                        );
                        Err(Error::MetalUnavailable)
                    }
                }
                routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. }
                | routing::RouteDecision::MetalUnavailable => unreachable!("handled above"),
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

#[cfg(target_os = "macos")]
fn reject_cpu_staged_metal_upload(surface: Surface) -> Result<Surface, Error> {
    if surface.residency() == SurfaceResidency::CpuStagedMetalUpload {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal explicit device decode requires a direct resident Metal decode; use the CPU path for CPU-staged output",
        });
    }
    Ok(surface)
}

#[allow(clippy::too_many_arguments)]
fn choose_route(
    decoder: &CpuDecoder<'_>,
    backend: BackendRequest,
    fmt: PixelFormat,
    op: batch::BatchOp,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> routing::RouteDecision {
    let capabilities = routing::JpegMetalCapabilities::for_request(
        decoder,
        fmt,
        op,
        fast444_packet,
        fast422_packet,
        fast420_packet,
    );
    let decision = routing::decide_route(backend, capabilities);
    if j2k_profile::gpu_route_profile_enabled() {
        let labels = decision.profile_labels();
        j2k_profile::emit_gpu_route_fields(
            "jpeg",
            "metal",
            &[
                j2k_profile::ProfileField::label("request", format_args!("{backend:?}")),
                j2k_profile::ProfileField::label("fmt", format_args!("{fmt:?}")),
                j2k_profile::ProfileField::label("op", jpeg_batch_op_profile(op)),
                j2k_profile::ProfileField::label("has_fast_packet", capabilities.has_fast_packet()),
                j2k_profile::ProfileField::label(
                    "supports_output_format",
                    capabilities.supports_output_format(),
                ),
                j2k_profile::ProfileField::label("decision", labels.decision),
                j2k_profile::ProfileField::label("reason", labels.reason),
            ],
        );
    }
    decision
}

fn jpeg_batch_op_profile(op: batch::BatchOp) -> &'static str {
    match op {
        batch::BatchOp::Full => "full",
        batch::BatchOp::Region(_) => "region",
        batch::BatchOp::Scaled(_) => "scaled",
        batch::BatchOp::RegionScaled { .. } => "region_scaled",
    }
}

fn decode_region_scaled_cpu_upload(
    decoder: &CpuDecoder<'_>,
    pool: &mut CpuScratchPool,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let scaled = roi.scaled_covering(scale);
    let dims = (scaled.w, scaled.h);
    let stride = dims.0 as usize * fmt.bytes_per_pixel();
    let mut out = vec![0u8; stride * dims.1 as usize];
    decoder.decode_region_scaled_into_with_scratch(
        pool,
        &mut out,
        stride,
        fmt,
        roi.into(),
        scale,
    )?;
    upload_surface(out, dims, fmt, backend)
}

fn scaled_dims(full: (u32, u32), scale: Downscale) -> (u32, u32) {
    (
        full.0.div_ceil(scale.denominator()),
        full.1.div_ceil(scale.denominator()),
    )
}

pub(crate) fn upload_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu => Ok(Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions,
            fmt,
            pitch_bytes,
            storage: Storage::Host(bytes),
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                let device = Device::system_default().ok_or(Error::MetalUnavailable)?;
                let buffer = device.new_buffer_with_data(
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                Ok(Surface {
                    backend: BackendKind::Metal,
                    residency: SurfaceResidency::CpuStagedMetalUpload,
                    dimensions,
                    fmt,
                    pitch_bytes,
                    storage: Storage::Metal { buffer, offset: 0 },
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                if matches!(backend, BackendRequest::Auto) {
                    Ok(Surface {
                        backend: BackendKind::Cpu,
                        residency: SurfaceResidency::Host,
                        dimensions,
                        fmt,
                        pitch_bytes,
                        storage: Storage::Host(bytes),
                    })
                } else {
                    Err(Error::MetalUnavailable)
                }
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

pub use j2k_jpeg::{
    DecoderContext, Downscale as JpegDownscale, PixelFormat as JpegPixelFormat, ScratchPool,
};
pub use j2k_jpeg::{Info, Rect as JpegRectPublic};

#[cfg(test)]
mod tests;
