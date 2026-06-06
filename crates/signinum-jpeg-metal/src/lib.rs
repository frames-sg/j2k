// SPDX-License-Identifier: Apache-2.0

//! Metal-backed JPEG decode and encode adapters.
//!
//! The crate exposes the same CPU-visible JPEG decode surface as
//! `signinum-jpeg`, with optional Metal-resident surfaces and batch submission
//! helpers on macOS. Non-macOS builds keep the public API available but return
//! `Error::MetalUnavailable` for explicit Metal-only work.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
#[cfg(target_os = "macos")]
mod compute;
mod encode;
mod profile;
mod routing;
mod session;
/// Viewport planning and composition helpers for JPEG decode surfaces.
pub mod viewport;

use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::sync::Mutex;
#[cfg(target_os = "macos")]
use std::sync::OnceLock;

use signinum_core::{
    copy_tight_pixels_to_strided_output, BackendKind, BackendRequest, BufferError, CodecError,
    DecodeOutcome, DeviceSubmission, DeviceSurface, Downscale, ImageCodec, ImageDecode,
    ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, Rect, TileBatchDecodeDevice,
    TileBatchDecodeSubmit,
};
use signinum_jpeg::{
    adapter::{
        build_metal_fast420_packet, build_metal_fast420_packet_for_decoder,
        build_metal_fast422_packet, build_metal_fast422_packet_for_decoder,
        build_metal_fast444_packet, build_metal_fast444_packet_for_decoder, decoder_bytes,
        JpegMetalFast420PacketV1, JpegMetalFast422PacketV1, JpegMetalFast444PacketV1,
    },
    Decoder as CpuDecoder, DecoderContext as CpuDecoderContext, JpegError, JpegView,
    ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

pub use encode::{
    encode_jpeg_baseline_batch_from_metal_buffers, encode_jpeg_baseline_from_metal_buffer,
    JpegBaselineMetalEncodeTile,
};

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
    Encode(#[from] signinum_jpeg::JpegEncodeError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is not supported by this crate.
    #[error("backend request {request:?} is not supported by signinum-jpeg-metal")]
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
    /// Metal kernel launch, validation, or completion failed.
    #[error("Metal kernel error: {message}")]
    MetalKernel {
        /// Kernel error message.
        message: String,
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
                | Self::MetalKernel { .. }
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

    fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }

    fn pixel_format(&self) -> PixelFormat {
        self.fmt
    }

    fn byte_len(&self) -> usize {
        self.pitch_bytes * self.dimensions.1 as usize
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
    runtime: Arc<OnceLock<Result<compute::MetalRuntime, String>>>,
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
        Device::system_default()
            .map(Self::new)
            .ok_or(Error::MetalUnavailable)
    }

    /// Metal device used by this session.
    pub fn device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
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
    pub fn submissions(&self) -> u64 {
        self.shared.0.lock().expect("metal session").submissions
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
    pub fn submissions(&self) -> u64 {
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
        Ok(self.push_shared_request(input, fmt, backend, batch::BatchOp::Full))
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
        Ok(self.push_shared_request(input, fmt, backend, batch::BatchOp::Region(roi)))
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
        Ok(self.push_shared_request(input, fmt, backend, batch::BatchOp::Scaled(scale)))
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
        Ok(self.push_shared_request(
            input,
            fmt,
            backend,
            batch::BatchOp::RegionScaled { roi, scale },
        ))
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
    ) -> usize {
        let slot = self.submissions.len();
        let submission = {
            let mut state = self.session.shared.0.lock().expect("metal session");
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
        slot
    }
}

/// JPEG decoder that can return host or Metal-resident surfaces.
pub struct Decoder<'a> {
    inner: CpuDecoder<'a>,
    source: Arc<[u8]>,
    fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
}

impl<'a> Decoder<'a> {
    /// Parse a JPEG byte slice into a decoder with any available Metal packets.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        let inner = CpuDecoder::new(input)?;
        Ok(Self {
            fast444_packet: build_metal_fast444_packet(input).ok().map(Arc::new),
            fast422_packet: build_metal_fast422_packet(input).ok().map(Arc::new),
            fast420_packet: build_metal_fast420_packet(input).ok().map(Arc::new),
            inner,
            source: Arc::<[u8]>::from(input),
        })
    }

    /// Create a decoder from an already parsed JPEG view.
    pub fn from_view(view: JpegView<'a>) -> Result<Self, Error> {
        let inner = CpuDecoder::from_view(view)?;
        let source = Arc::<[u8]>::from(decoder_bytes(&inner));
        let fast444_packet = build_metal_fast444_packet_for_decoder(&inner)
            .ok()
            .map(Arc::new);
        let fast422_packet = build_metal_fast422_packet_for_decoder(&inner)
            .ok()
            .map(Arc::new);
        let fast420_packet = build_metal_fast420_packet_for_decoder(&inner)
            .ok()
            .map(Arc::new);
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

    fn inspect(input: &'a [u8]) -> Result<signinum_core::Info, Self::Error> {
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
/// JPEG codec marker used by Signinum's generic decode traits.
pub struct Codec;

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = CpuScratchPool;
}

impl Codec {
    #[cfg(target_os = "macos")]
    fn rgb8_metal_batch_requests(
        inputs: &[&[u8]],
        mut op_for_decoder: impl FnMut(&CpuDecoder<'_>) -> batch::BatchOp,
    ) -> Result<Vec<batch::QueuedRequest>, Error> {
        let mut state = session::SessionState::default();
        let mut requests = Vec::with_capacity(inputs.len());
        for input in inputs {
            let decoder = CpuDecoder::new(input)?;
            let op = op_for_decoder(&decoder);
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
        Ok(requests)
    }

    #[cfg(target_os = "macos")]
    /// Decode a full-tile RGB8 JPEG batch into a reusable caller-owned Metal buffer.
    pub fn decode_rgb8_batch_into_metal_buffer_with_session(
        inputs: &[&[u8]],
        output: &MetalBatchOutputBuffer,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let requests = Self::rgb8_metal_batch_requests(inputs, |_| batch::BatchOp::Full)?;

        compute::decode_full_rgb8_batch_into_output_with_session(&requests, output, session)?
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "JPEG Metal reusable batch output currently supports batchable full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs",
            })
    }

    #[cfg(target_os = "macos")]
    /// Decode a full-tile RGB8 JPEG batch into reusable caller-owned Metal RGBA8 textures.
    pub fn decode_rgb8_batch_into_metal_textures_with_session(
        inputs: &[&[u8]],
        output: &MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let requests = Self::rgb8_metal_batch_requests(inputs, |_| batch::BatchOp::Full)?;

        compute::decode_full_rgb8_batch_into_textures_with_session(&requests, output, session)?
            .ok_or(Error::UnsupportedMetalRequest {
                reason: "JPEG Metal texture batch output currently supports batchable full-tile RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs",
            })
    }

    #[cfg(target_os = "macos")]
    /// Decode a scaled RGB8 JPEG batch into a reusable caller-owned Metal buffer.
    pub fn decode_rgb8_scaled_batch_into_metal_buffer_with_session(
        inputs: &[&[u8]],
        scale: Downscale,
        output: &MetalBatchOutputBuffer,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let requests = Self::rgb8_metal_batch_requests(inputs, |decoder| {
            let (w, h) = decoder.info().dimensions;
            batch::BatchOp::RegionScaled {
                roi: Rect { x: 0, y: 0, w, h },
                scale,
            }
        })?;

        compute::decode_region_scaled_rgb8_batch_into_output_with_session(
            &requests, output, session,
        )?
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with half, quarter, or eighth scaling",
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a scaled RGB8 JPEG batch into reusable caller-owned Metal RGBA8 textures.
    pub fn decode_rgb8_scaled_batch_into_metal_textures_with_session(
        inputs: &[&[u8]],
        scale: Downscale,
        output: &MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let requests = Self::rgb8_metal_batch_requests(inputs, |decoder| {
            let (w, h) = decoder.info().dimensions;
            batch::BatchOp::RegionScaled {
                roi: Rect { x: 0, y: 0, w, h },
                scale,
            }
        })?;

        compute::decode_region_scaled_rgb8_batch_into_textures_with_session(
            &requests, output, session,
        )?
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal texture scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with half, quarter, or eighth scaling",
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a region-scaled RGB8 JPEG batch into a reusable caller-owned Metal buffer.
    pub fn decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
        inputs: &[&[u8]],
        roi: Rect,
        scale: Downscale,
        output: &MetalBatchOutputBuffer,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<Surface, Error>>, Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let requests = Self::rgb8_metal_batch_requests(inputs, |_| batch::BatchOp::RegionScaled {
            roi,
            scale,
        })?;

        compute::decode_region_scaled_rgb8_batch_into_output_with_session(
            &requests, output, session,
        )?
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable region-scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with matching output shapes",
        })
    }

    #[cfg(target_os = "macos")]
    /// Decode a region-scaled RGB8 JPEG batch into reusable caller-owned Metal RGBA8 textures.
    pub fn decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
        inputs: &[&[u8]],
        roi: Rect,
        scale: Downscale,
        output: &MetalBatchTextureOutput,
        session: &MetalBackendSession,
    ) -> Result<Vec<Result<MetalTextureTile, Error>>, Error> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let requests = Self::rgb8_metal_batch_requests(inputs, |_| batch::BatchOp::RegionScaled {
            roi,
            scale,
        })?;

        compute::decode_region_scaled_rgb8_batch_into_textures_with_session(
            &requests, output, session,
        )?
        .ok_or(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal texture region-scaled batch output currently supports batchable RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs with matching output shapes",
        })
    }

    #[allow(clippy::too_many_arguments)]
    /// Submit a scaled region tile decode into a reusable Metal session.
    pub fn submit_tile_region_scaled_to_device(
        ctx: &mut signinum_core::DecoderContext<CpuDecoderContext>,
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
            let mut state = session.shared.0.lock().expect("metal session");
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
            .0
            .lock()
            .expect("metal session")
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
            .0
            .lock()
            .expect("metal session")
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
            .0
            .lock()
            .expect("metal session")
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
            .0
            .lock()
            .expect("metal session")
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let _ = (ctx, pool);
        let slot = {
            let mut state = session.shared.0.lock().expect("metal session");
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
        ctx: &mut signinum_core::DecoderContext<Self::Context>,
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

pub(crate) fn decode_surface_from_bytes(
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: batch::BatchOp,
    fast444_packet: Option<Arc<JpegMetalFast444PacketV1>>,
    fast422_packet: Option<Arc<JpegMetalFast422PacketV1>>,
    fast420_packet: Option<Arc<JpegMetalFast420PacketV1>>,
) -> Result<Surface, Error> {
    let decoder = CpuDecoder::new(input)?;
    let mut pool = CpuScratchPool::new();
    let build_auto_packets =
        matches!(backend, BackendRequest::Auto) && decoder.info().restart_interval.is_some();
    let build_metal_packets = matches!(backend, BackendRequest::Metal);
    let fast444_packet = if build_auto_packets || build_metal_packets {
        fast444_packet.or_else(|| {
            build_metal_fast444_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast422_packet = if build_auto_packets || build_metal_packets {
        fast422_packet.or_else(|| {
            build_metal_fast422_packet_for_decoder(&decoder)
                .ok()
                .map(Arc::new)
        })
    } else {
        None
    };
    let fast420_packet = if build_auto_packets || build_metal_packets {
        fast420_packet.or_else(|| {
            build_metal_fast420_packet_for_decoder(&decoder)
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
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
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
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
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
    fast444_packet: Option<&JpegMetalFast444PacketV1>,
    fast422_packet: Option<&JpegMetalFast422PacketV1>,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
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
    if profile::gpu_route_profile_enabled() {
        let request_s = format!("{backend:?}");
        let fmt_s = format!("{fmt:?}");
        let has_fast_packet_s = capabilities.has_fast_packet().to_string();
        let supports_format_s = capabilities.supports_output_format().to_string();
        let (decision_s, reason_s) = jpeg_route_decision_profile(decision);
        profile::emit_gpu_route_profile(
            "jpeg",
            "gpu_route",
            "metal",
            &[
                ("request", request_s.as_str()),
                ("fmt", fmt_s.as_str()),
                ("op", jpeg_batch_op_profile(op)),
                ("has_fast_packet", has_fast_packet_s.as_str()),
                ("supports_output_format", supports_format_s.as_str()),
                ("decision", decision_s),
                ("reason", reason_s),
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

fn jpeg_route_decision_profile(decision: routing::RouteDecision) -> (&'static str, &'static str) {
    match decision {
        routing::RouteDecision::CpuHost => ("cpu_host", "none"),
        routing::RouteDecision::MetalKernel => ("metal_kernel", "none"),
        routing::RouteDecision::RejectExplicitMetal { reason } => {
            let reason_code = if reason.contains("fast") {
                "no_fast_packet"
            } else {
                "unsupported_format"
            };
            ("reject_explicit_metal", reason_code)
        }
        routing::RouteDecision::RejectUnsupportedBackend { .. } => {
            ("reject_unsupported_backend", "unsupported_backend")
        }
        routing::RouteDecision::MetalUnavailable => ("metal_unavailable", "metal_unavailable"),
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

pub use signinum_jpeg::{
    DecoderContext, Downscale as JpegDownscale, PixelFormat as JpegPixelFormat, ScratchPool,
};
pub use signinum_jpeg::{Info, Rect as JpegRectPublic};

#[cfg(test)]
mod tests {
    use super::*;
    use signinum_jpeg::adapter::{build_metal_fast420_packet, build_metal_fast444_packet};
    use signinum_jpeg::{
        encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
    };

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_444: &[u8] = include_bytes!("../fixtures/jpeg/baseline_444_8x8.jpg");
    #[cfg(not(target_os = "macos"))]
    const GRAYSCALE: &[u8] = include_bytes!("../fixtures/jpeg/grayscale_8x8.jpg");

    #[test]
    fn auto_route_prefers_cpu_host_for_nonrestart_packets() {
        let decoder_420 = CpuDecoder::new(BASELINE_420).expect("420 decoder");
        let packet_420 = build_metal_fast420_packet(BASELINE_420).expect("420 packet");
        assert_eq!(
            choose_route(
                &decoder_420,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Full,
                None,
                None,
                Some(&packet_420),
            ),
            routing::RouteDecision::CpuHost
        );

        let decoder_444 = CpuDecoder::new(BASELINE_444).expect("444 decoder");
        let packet_444 = build_metal_fast444_packet(BASELINE_444).expect("444 packet");
        assert_eq!(
            choose_route(
                &decoder_444,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Scaled(Downscale::Quarter),
                Some(&packet_444),
                None,
                None,
            ),
            routing::RouteDecision::CpuHost
        );
    }

    #[test]
    fn auto_route_keeps_small_single_restart_packets_on_cpu_host() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
        let packet = build_metal_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

        assert_eq!(
            choose_route(
                &decoder,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Full,
                None,
                None,
                Some(&packet)
            ),
            routing::RouteDecision::CpuHost
        );
        assert_eq!(
            choose_route(
                &decoder,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::Region(Rect {
                    x: 0,
                    y: 0,
                    w: 16,
                    h: 16,
                }),
                None,
                None,
                Some(&packet),
            ),
            routing::RouteDecision::CpuHost
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_backend_session_reuses_compiled_runtime() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        assert!(session.runtime.get().is_none());

        let mut first = Decoder::new(BASELINE_420).expect("first decoder");
        let first_surface = first
            .decode_to_device_with_session(PixelFormat::Rgb8, &session)
            .expect("first session decode");
        assert_eq!(
            first_surface.residency(),
            SurfaceResidency::MetalResidentDecode
        );
        let first_runtime = session
            .runtime
            .get()
            .and_then(|runtime| runtime.as_ref().ok())
            .map(std::ptr::from_ref::<compute::MetalRuntime>)
            .expect("session runtime after first decode");

        let mut second = Decoder::new(BASELINE_420).expect("second decoder");
        second
            .decode_to_device_with_session(PixelFormat::Rgb8, &session)
            .expect("second session decode");
        let second_runtime = session
            .runtime
            .get()
            .and_then(|runtime| runtime.as_ref().ok())
            .map(std::ptr::from_ref::<compute::MetalRuntime>)
            .expect("session runtime after second decode");

        assert_eq!(first_runtime, second_runtime);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_rgb8_batch_decode_uses_backend_session_runtime() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        assert!(session.runtime.get().is_none());

        let inputs = [BASELINE_420, BASELINE_420];
        let results = decode_rgb8_batch_to_device_with_session(&inputs, &session)
            .expect("session batch decode")
            .expect("baseline JPEG batch should use Metal batch path");

        assert_eq!(results.len(), 2);
        assert!(session.runtime.get().is_some());
        for result in results {
            let surface = result.expect("surface");
            assert_eq!(surface.backend_kind(), BackendKind::Metal);
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (16, 16));
            assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn queued_jpeg_batch_decode_uses_metal_session_runtime() {
        use signinum_core::DeviceSubmission as _;

        let backend_session = MetalBackendSession::system_default().expect("Metal backend session");
        assert!(backend_session.runtime.get().is_none());
        let mut session = MetalSession::with_backend_session(backend_session.clone());
        let mut ctx = signinum_core::DecoderContext::<signinum_jpeg::DecoderContext>::new();
        let mut pool = ScratchPool::new();

        let submissions = (0..2)
            .map(|_| {
                <Codec as signinum_core::TileBatchDecodeSubmit>::submit_tile_to_device(
                    &mut ctx,
                    &mut session,
                    &mut pool,
                    BASELINE_420,
                    PixelFormat::Rgb8,
                    BackendRequest::Metal,
                )
                .expect("queued Metal tile submit")
            })
            .collect::<Vec<_>>();

        for submission in submissions {
            let surface = submission.wait().expect("queued Metal surface");
            assert_eq!(surface.backend_kind(), BackendKind::Metal);
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (16, 16));
        }

        assert_eq!(session.submissions(), 1);
        assert!(
            backend_session.runtime.get().is_some(),
            "queued MetalSession batch decode should reuse its backend runtime"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn default_queued_jpeg_batch_decode_lazily_initializes_backend_session() {
        use signinum_core::DeviceSubmission as _;

        let mut session = MetalSession::default();
        assert!(session
            .shared
            .0
            .lock()
            .expect("metal session")
            .backend_session
            .is_none());
        let mut ctx = signinum_core::DecoderContext::<signinum_jpeg::DecoderContext>::new();
        let mut pool = ScratchPool::new();

        let submissions = (0..2)
            .map(|_| {
                <Codec as signinum_core::TileBatchDecodeSubmit>::submit_tile_to_device(
                    &mut ctx,
                    &mut session,
                    &mut pool,
                    BASELINE_420,
                    PixelFormat::Rgb8,
                    BackendRequest::Metal,
                )
                .expect("queued Metal tile submit")
            })
            .collect::<Vec<_>>();

        for submission in submissions {
            let surface = submission.wait().expect("queued Metal surface");
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        }

        let runtime_initialized = session
            .shared
            .0
            .lock()
            .expect("metal session")
            .backend_session
            .as_ref()
            .and_then(|backend| backend.runtime.get())
            .is_some();
        assert!(runtime_initialized);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_batch_decode_can_write_into_reusable_metal_output_buffer() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
        let inputs = [BASELINE_420, BASELINE_420];
        let (expected, _) = CpuDecoder::new(BASELINE_420)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");

        let surfaces =
            Codec::decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
                .expect("decode into reusable output");

        assert_eq!(surfaces.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(
            output.tile_stride_bytes(),
            16 * 16 * PixelFormat::Rgb8.bytes_per_pixel()
        );
        for (index, result) in surfaces.into_iter().enumerate() {
            let surface = result.expect("surface");
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (16, 16));
            assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
            let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
            assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
            assert_eq!(offset, index * output.tile_stride_bytes());
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast444_batch_decode_can_write_into_reusable_metal_output_buffer() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchOutputBuffer::new_rgb8_tiles(&session, (8, 8), 2).expect("output buffer");
        let inputs = [BASELINE_444, BASELINE_444];
        let (expected, _) = CpuDecoder::new(BASELINE_444)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");

        let surfaces =
            Codec::decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
                .expect("decode into reusable output");

        assert_eq!(surfaces.len(), 2);
        for (index, result) in surfaces.into_iter().enumerate() {
            let surface = result.expect("surface");
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (8, 8));
            assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
            let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
            assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
            assert_eq!(offset, index * output.tile_stride_bytes());
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_scaled_batch_decode_can_write_into_reusable_metal_output_buffer() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let scale = Downscale::Quarter;
        let output =
            MetalBatchOutputBuffer::new_rgb8_tiles(&session, (4, 4), 2).expect("output buffer");
        let inputs = [BASELINE_420, BASELINE_420];
        let (expected, _) = CpuDecoder::new(BASELINE_420)
            .expect("cpu decoder")
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled decode");

        let surfaces = Codec::decode_rgb8_scaled_batch_into_metal_buffer_with_session(
            &inputs, scale, &output, &session,
        )
        .expect("decode scaled into reusable output");

        assert_eq!(surfaces.len(), 2);
        for (index, result) in surfaces.into_iter().enumerate() {
            let surface = result.expect("surface");
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (4, 4));
            assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
            let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
            assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
            assert_eq!(offset, index * output.tile_stride_bytes());
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_region_scaled_batch_decode_can_write_into_reusable_metal_output_buffer() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = Downscale::Quarter;
        let scaled = roi.scaled_covering(scale);
        let output = MetalBatchOutputBuffer::new_rgb8_tiles(&session, (scaled.w, scaled.h), 2)
            .expect("output buffer");
        let inputs = [BASELINE_444, BASELINE_444];
        let (expected, _) = CpuDecoder::new(BASELINE_444)
            .expect("cpu decoder")
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled decode");

        let surfaces = Codec::decode_rgb8_region_scaled_batch_into_metal_buffer_with_session(
            &inputs, roi, scale, &output, &session,
        )
        .expect("decode region scaled into reusable output");

        assert_eq!(surfaces.len(), 2);
        for (index, result) in surfaces.into_iter().enumerate() {
            let surface = result.expect("surface");
            assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
            assert_eq!(surface.dimensions(), (scaled.w, scaled.h));
            assert_eq!(surface.pixel_format(), PixelFormat::Rgb8);
            let (buffer, offset) = surface.metal_buffer().expect("metal buffer");
            assert!(std::ptr::eq(buffer.as_ref(), output.buffer()));
            assert_eq!(offset, index * output.tile_stride_bytes());
            assert_eq!(surface.as_bytes(), expected.as_slice());
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast444_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 5,
            h: 4,
        };
        let scale = Downscale::Quarter;
        let scaled = roi.scaled_covering(scale);
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
            .expect("texture output");
        let inputs = [BASELINE_444, BASELINE_444];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
            .expect("cpu decoder")
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles = Codec::decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
            &inputs, roi, scale, &output, &session,
        )
        .expect("decode region scaled into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (scaled.w, scaled.h));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn warm_session_reuses_private_intermediate_buffers_for_reusable_output_batches() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
        let inputs = [BASELINE_420, BASELINE_420];

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let first =
            Codec::decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
                .expect("first decode");
        for surface in first {
            assert_eq!(
                surface.expect("surface").residency(),
                SurfaceResidency::MetalResidentDecode
            );
        }
        let allocations_after_first = compute::jpeg_private_buffer_allocations_for_test();

        let second =
            Codec::decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
                .expect("second decode");
        for surface in second {
            assert_eq!(
                surface.expect("surface").residency(),
                SurfaceResidency::MetalResidentDecode
            );
        }

        assert!(
            allocations_after_first > 0,
            "first batch should allocate private intermediate buffers"
        );
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            allocations_after_first,
            "warm session batch should reuse private intermediate buffers"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn warm_session_reuses_shared_upload_buffers_for_reusable_output_batches() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchOutputBuffer::new_rgb8_tiles(&session, (16, 16), 2).expect("output buffer");
        let inputs = [BASELINE_420, BASELINE_420];

        compute::reset_jpeg_shared_buffer_allocations_for_test();
        Codec::decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
            .expect("first decode");
        let allocations_after_first = compute::jpeg_shared_buffer_allocations_for_test();

        Codec::decode_rgb8_batch_into_metal_buffer_with_session(&inputs, &output, &session)
            .expect("second decode");

        assert!(
            allocations_after_first > 0,
            "first batch should allocate shared upload/status buffers"
        );
        assert_eq!(
            compute::jpeg_shared_buffer_allocations_for_test(),
            allocations_after_first,
            "warm session batch should reuse shared upload/status buffers"
        );
    }

    #[cfg(target_os = "macos")]
    fn rgb_to_rgba_opaque(rgb: &[u8]) -> Vec<u8> {
        let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
        for pixel in rgb.chunks_exact(3) {
            rgba.extend_from_slice(pixel);
            rgba.push(u8::MAX);
        }
        rgba
    }

    #[cfg(target_os = "macos")]
    fn download_rgba8_texture(
        session: &MetalBackendSession,
        texture: &metal::TextureRef,
        dimensions: (u32, u32),
    ) -> Vec<u8> {
        let row_bytes = dimensions.0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let byte_len = row_bytes * dimensions.1 as usize;
        let buffer = session.device().new_buffer(
            byte_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let queue = session.device().new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        blit.copy_from_texture_to_buffer(
            texture,
            0,
            0,
            metal::MTLOrigin { x: 0, y: 0, z: 0 },
            metal::MTLSize::new(u64::from(dimensions.0), u64::from(dimensions.1), 1),
            &buffer,
            0,
            row_bytes as u64,
            byte_len as u64,
            metal::MTLBlitOption::None,
        );
        blit.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        unsafe { core::slice::from_raw_parts(buffer.contents().cast::<u8>(), byte_len).to_vec() }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast444_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, (8, 8), 2).expect("texture output");
        let inputs = [BASELINE_444, BASELINE_444];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (8, 8));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (8, 8));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_scaled_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let scale = Downscale::Quarter;
        let output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, (4, 4), 2).expect("texture output");
        let inputs = [BASELINE_420, BASELINE_420];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
            .expect("cpu decoder")
            .decode_scaled(PixelFormat::Rgb8, scale)
            .expect("cpu scaled decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles = Codec::decode_rgb8_scaled_batch_into_metal_textures_with_session(
            &inputs, scale, &output, &session,
        )
        .expect("decode scaled into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (4, 4));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (4, 4));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast422_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let roi = Rect {
            x: 1,
            y: 1,
            w: 9,
            h: 6,
        };
        let scale = Downscale::Half;
        let scaled = roi.scaled_covering(scale);
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
            .expect("texture output");
        let inputs = [BASELINE_422, BASELINE_422];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
            .expect("cpu decoder")
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles = Codec::decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
            &inputs, roi, scale, &output, &session,
        )
        .expect("decode region scaled into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (scaled.w, scaled.h));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast420_region_scaled_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let roi = Rect {
            x: 1,
            y: 2,
            w: 10,
            h: 9,
        };
        let scale = Downscale::Quarter;
        let scaled = roi.scaled_covering(scale);
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (scaled.w, scaled.h), 2)
            .expect("texture output");
        let inputs = [BASELINE_420, BASELINE_420];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
            .expect("cpu decoder")
            .decode_region_scaled(
                PixelFormat::Rgb8,
                signinum_jpeg::Rect {
                    x: roi.x,
                    y: roi.y,
                    w: roi.w,
                    h: roi.h,
                },
                scale,
            )
            .expect("cpu region scaled decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles = Codec::decode_rgb8_region_scaled_batch_into_metal_textures_with_session(
            &inputs, roi, scale, &output, &session,
        )
        .expect("decode region scaled into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (scaled.w, scaled.h));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (scaled.w, scaled.h));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast420_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2)
            .expect("texture output");
        let inputs = [BASELINE_420, BASELINE_420];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (16, 16));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (16, 16));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast422_batch_decode_can_write_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 8), 2).expect("texture output");
        let inputs = [BASELINE_422, BASELINE_422];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        assert_eq!(output.tile_capacity(), 2);
        assert_eq!(output.dimensions(), (16, 8));
        assert_eq!(output.pixel_format(), PixelFormat::Rgba8);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (16, 8));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_texture_batch_decode_avoids_private_rgba_staging_buffers() {
        let cases = [
            (BASELINE_420, (16, 16), 0),
            (BASELINE_422, (16, 8), 0),
            (BASELINE_444, (8, 8), 0),
        ];

        for (input, dimensions, expected_private_allocations) in cases {
            let session = MetalBackendSession::system_default().expect("Metal backend session");
            let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
                .expect("texture output");
            let inputs = [input, input];

            compute::reset_jpeg_private_buffer_allocations_for_test();
            let tiles = Codec::decode_rgb8_batch_into_metal_textures_with_session(
                &inputs, &output, &session,
            )
            .expect("decode into reusable textures");
            assert_eq!(tiles.len(), 2);
            for tile in tiles {
                assert_eq!(
                    tile.expect("texture tile").pixel_format(),
                    PixelFormat::Rgba8
                );
            }

            assert_eq!(
                compute::jpeg_private_buffer_allocations_for_test(),
                expected_private_allocations,
                "texture batch decode should not allocate a private RGBA staging buffer for {dimensions:?}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast444_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, (8, 8), 2).expect("texture output");
        let inputs = [BASELINE_444, BASELINE_444];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_444)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (8, 8));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "fused 4:4:4 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast422_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output =
            MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 8), 2).expect("texture output");
        let inputs = [BASELINE_422, BASELINE_422];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_422)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (16, 8));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "fused 4:2:2 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_wide_fast422_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let dimensions = (48, 16);
        let rgb = signinum_test_support::patterned_rgb8(dimensions.0, dimensions.1);
        let jpeg = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: dimensions.0,
                height: dimensions.1,
            },
            JpegEncodeOptions {
                quality: 92,
                subsampling: JpegSubsampling::Ybr422,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode 4:2:2 source jpeg");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
            .expect("texture output");
        let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
        let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), dimensions);
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "wide fused 4:2:2 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, (16, 16), 2)
            .expect("texture output");
        let inputs = [BASELINE_420, BASELINE_420];
        let (expected_rgb, _) = CpuDecoder::new(BASELINE_420)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), (16, 16));
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rgb8_wide_row_fast420_texture_batch_decode_fuses_directly_into_reusable_metal_textures() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let dimensions = (32, 16);
        let rgb = signinum_test_support::patterned_rgb8(dimensions.0, dimensions.1);
        let jpeg = encode_jpeg_baseline(
            JpegSamples::Rgb8 {
                data: &rgb,
                width: dimensions.0,
                height: dimensions.1,
            },
            JpegEncodeOptions {
                quality: 92,
                subsampling: JpegSubsampling::Ybr420,
                restart_interval: None,
                backend: JpegBackend::Cpu,
            },
        )
        .expect("encode 4:2:0 source jpeg");
        let output = MetalBatchTextureOutput::new_rgba8_tiles(&session, dimensions, 2)
            .expect("texture output");
        let inputs = [jpeg.data.as_slice(), jpeg.data.as_slice()];
        let (expected_rgb, _) = CpuDecoder::new(&jpeg.data)
            .expect("cpu decoder")
            .decode(PixelFormat::Rgb8)
            .expect("cpu decode");
        let expected_rgba = rgb_to_rgba_opaque(&expected_rgb);

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let tiles =
            Codec::decode_rgb8_batch_into_metal_textures_with_session(&inputs, &output, &session)
                .expect("decode into reusable textures");

        assert_eq!(tiles.len(), 2);
        for (index, tile) in tiles.into_iter().enumerate() {
            let tile = tile.expect("texture tile");
            assert_eq!(tile.dimensions(), dimensions);
            assert_eq!(tile.pixel_format(), PixelFormat::Rgba8);
            assert!(std::ptr::eq(
                tile.texture(),
                output.texture(index).expect("output texture")
            ));
            assert_eq!(
                download_rgba8_texture(&session, tile.texture(), tile.dimensions()),
                expected_rgba
            );
        }
        assert_eq!(
            compute::jpeg_private_buffer_allocations_for_test(),
            0,
            "wide-row fused 4:2:0 texture batch decode should not allocate private Y/Cb/Cr staging planes"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_device_decode_uses_private_internal_planes() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

        compute::reset_jpeg_private_buffer_allocations_for_test();
        let surface = decoder
            .decode_to_device_with_session(PixelFormat::Rgb8, &session)
            .expect("resident JPEG Metal decode");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert!(
            compute::jpeg_private_buffer_allocations_for_test() > 0,
            "resident JPEG Metal decode should use Private internal planes"
        );
        let _ = surface.as_bytes();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_private_rgb8_tile_uses_private_output_buffer() {
        let session = MetalBackendSession::system_default().expect("Metal backend session");
        let mut decoder = Decoder::new(BASELINE_420).expect("decoder");

        let tile = decoder
            .decode_private_rgb8_tile_with_session(&session)
            .expect("resident private JPEG Metal decode");

        assert_eq!(tile.dimensions, (16, 16));
        assert_eq!(tile.pixel_format, PixelFormat::Rgb8);
        assert_eq!(tile.pitch_bytes, 16 * PixelFormat::Rgb8.bytes_per_pixel());
        assert_eq!(tile.byte_offset, 0);
        assert_eq!(tile.buffer.storage_mode(), metal::MTLStorageMode::Private);
        assert!(tile.status_buffer.length() > 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn jpeg_gray_region_decode_uses_private_internal_planes() {
        let roi = Rect {
            x: 4,
            y: 4,
            w: 8,
            h: 8,
        };
        let mut expected_decoder = Decoder::new(BASELINE_420).expect("expected decoder");
        let mut expected = vec![0; roi.w as usize * roi.h as usize];
        expected_decoder
            .decode_region_into(
                &mut CpuScratchPool::new(),
                &mut expected,
                roi.w as usize,
                PixelFormat::Gray8,
                roi,
            )
            .expect("expected CPU region decode");

        let mut decoder = Decoder::new(BASELINE_420).expect("decoder");
        compute::reset_jpeg_private_buffer_allocations_for_test();
        let surface = decoder
            .decode_region_to_device(PixelFormat::Gray8, roi, BackendRequest::Metal)
            .expect("resident JPEG Metal region decode");
        assert_eq!(surface.residency(), SurfaceResidency::MetalResidentDecode);
        assert!(
            compute::jpeg_private_buffer_allocations_for_test() >= 3,
            "resident Gray8 region decode should keep decoded Y/Cb/Cr planes Private"
        );
        assert_eq!(surface.as_bytes(), expected.as_slice());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn uploaded_metal_surface_is_marked_cpu_staged() {
        let surface = upload_surface(
            vec![1, 2, 3],
            (1, 1),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
        )
        .expect("CPU staged Metal upload");

        assert_eq!(surface.residency(), SurfaceResidency::CpuStagedMetalUpload);
    }

    #[test]
    fn auto_route_prefers_cpu_host_for_region_scaled_even_with_restart_packets() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("restart decoder");
        let packet = build_metal_fast420_packet(BASELINE_420_RESTART).expect("restart packet");

        assert_eq!(
            choose_route(
                &decoder,
                BackendRequest::Auto,
                PixelFormat::Rgb8,
                batch::BatchOp::RegionScaled {
                    roi: Rect {
                        x: 0,
                        y: 0,
                        w: 16,
                        h: 16,
                    },
                    scale: Downscale::Quarter,
                },
                None,
                None,
                Some(&packet),
            ),
            routing::RouteDecision::CpuHost
        );
    }

    #[cfg(not(target_os = "macos"))]
    #[test]
    fn session_decode_rejects_unsupported_shape_before_host_unavailability() {
        let mut decoder = Decoder::new(GRAYSCALE).expect("decoder");
        let session = MetalBackendSession::default();

        assert!(matches!(
            decoder.decode_to_device_with_session(PixelFormat::Gray8, &session),
            Err(Error::UnsupportedMetalRequest { .. })
        ));
    }
}
