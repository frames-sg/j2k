// SPDX-License-Identifier: Apache-2.0

//! Metal-backed JPEG 2000 and HTJ2K decode and encode adapters.
//!
//! This crate wraps the CPU/native J2K implementation with optional
//! Metal-resident decode surfaces, batch decode sessions, and lossless encode
//! helpers on macOS. Non-macOS builds keep the same API surface and return
//! `Error::MetalUnavailable` for explicit Metal-only requests.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(unreachable_pub)]

mod batch;
#[cfg(target_os = "macos")]
mod buffer_pool;
#[cfg(any(test, target_os = "macos"))]
mod classic;
#[cfg(target_os = "macos")]
mod compute;
#[cfg(target_os = "macos")]
mod direct;
mod encode;
#[cfg(any(test, target_os = "macos"))]
mod ht;
#[cfg(target_os = "macos")]
mod hybrid;
#[cfg(any(test, target_os = "macos"))]
mod idwt;
#[cfg(any(test, target_os = "macos"))]
mod mct;
mod profile;
#[cfg(target_os = "macos")]
mod profile_env;
mod routing;
#[cfg(any(test, target_os = "macos"))]
mod store;

use core::convert::Infallible;
#[cfg(target_os = "macos")]
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::{Mutex, OnceLock},
};
use std::{ops::Range, sync::Arc};

use j2k::{
    adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
    J2kContext as CpuJ2kContext, J2kDecoder as CpuDecoder, J2kError,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
use j2k_core::{
    copy_tight_pixels_to_strided_output, BackendKind, BackendRequest, BufferError, CodecError,
    DecodeOutcome, DeviceMemoryRange, DeviceSubmission, DeviceSurface, Downscale, ImageCodec,
    ImageDecode, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect,
    TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
};
#[cfg(target_os = "macos")]
use j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage, J2kDirectColorPlan, J2kDirectGrayscalePlan,
};

#[cfg(target_os = "macos")]
use j2k_metal_support::{system_default_device, MetalSupportError};
#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
#[cfg(target_os = "macos")]
use metal::{Buffer, Device, MTLResourceOptions};

#[doc(hidden)]
pub use batch::{benchmark_group_region_scaled_requests, BenchmarkGroupedRequests};
pub use encode::{
    encode_lossless_batch_with_report, submit_lossless_batch, submit_lossless_batch_to_metal,
    validate_lossless_roundtrip_on_metal, validate_lossless_roundtrip_on_metal_with_session,
    MetalEncodeInputStaging, MetalEncodeStageAccelerator, MetalEncodedJ2k,
    MetalLosslessBufferEncodeBatchOutcome, MetalLosslessBufferEncodeOutcome,
    MetalLosslessEncodeBatchRequest, MetalLosslessEncodeBatchStats, MetalLosslessEncodeConfig,
    MetalLosslessEncodeOutcome, MetalLosslessEncodeResidency, MetalLosslessEncodeStageStats,
    MetalLosslessEncodeTile, SubmittedJ2kLosslessMetalBufferEncodeBatch,
    SubmittedJ2kLosslessMetalEncodeBatch,
};

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_region_scaled_direct_plan_prepare(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<(), Error> {
    hybrid::benchmark_region_scaled_direct_plan_prepare(input, fmt, roi, scale)
}

#[cfg(not(target_os = "macos"))]
#[doc(hidden)]
pub fn benchmark_region_scaled_direct_plan_prepare(
    _input: &[u8],
    _fmt: PixelFormat,
    _roi: Rect,
    _scale: Downscale,
) -> Result<(), Error> {
    Err(Error::MetalUnavailable)
}

#[derive(Debug, thiserror::Error)]
/// Errors returned by the Metal J2K backend.
pub enum Error {
    /// Error returned by the CPU or native J2K decoder.
    #[error(transparent)]
    Decode(#[from] J2kError),
    /// Output buffer validation failed.
    #[error(transparent)]
    Buffer(#[from] BufferError),
    /// The requested backend is unsupported by this crate.
    #[error("backend request {request:?} is not supported by j2k-metal")]
    UnsupportedBackend {
        /// Backend requested by the caller.
        request: BackendRequest,
    },
    /// A Metal-specific request is structurally unsupported.
    #[error("unsupported J2K Metal request: {reason}")]
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
                | Self::UnsupportedMetalRequest { .. }
                | Self::MetalUnavailable
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
    Metal(Buffer),
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectGrayPlanCacheEntry {
    plan: J2kDirectGrayscalePlan,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct DirectColorPlanCacheEntry {
    plan: J2kDirectColorPlan,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
}

#[cfg(target_os = "macos")]
const DIRECT_PLAN_CACHE_CAP: usize = 128;
#[cfg(target_os = "macos")]
const AUTO_REPEATED_GRAYSCALE_MIN_DIM: u32 = 512;
#[cfg(target_os = "macos")]
const AUTO_REPEATED_GRAYSCALE_MIN_COUNT: usize = 16;

#[derive(Clone)]
/// Decoded J2K image surface returned by the Metal backend.
pub struct Surface {
    backend: BackendKind,
    residency: SurfaceResidency,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    pitch_bytes: usize,
    byte_offset: usize,
    storage: Storage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Where a decoded J2K surface is currently resident.
pub enum SurfaceResidency {
    /// Pixel bytes are resident in host memory.
    Host,
    /// Pixel bytes were produced directly by a Metal decode kernel.
    MetalResidentDecode,
    /// Pixel bytes were decoded on CPU and uploaded into a Metal buffer.
    CpuStagedMetalUpload,
}

#[cfg(target_os = "macos")]
const CPU_STAGED_METAL_REQUIRES_EXPLICIT_API: &str =
    "CPU-staged Metal upload requires the explicit CPU-staged API; BackendRequest::Metal only accepts resident Metal decode";
impl Surface {
    /// Current residency for the surface bytes.
    pub fn residency(&self) -> SurfaceResidency {
        self.residency
    }

    /// Number of bytes between consecutive rows.
    pub fn pitch_bytes(&self) -> usize {
        self.pitch_bytes
    }

    fn checked_storage_range(&self, storage_len: usize) -> Result<Range<usize>, Error> {
        let len = self.byte_len();
        let end = self
            .byte_offset
            .checked_add(len)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal surface byte range overflows usize".to_string(),
            })?;
        if end > storage_len {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal surface byte range {start}..{end} exceeds storage length {storage_len}",
                    start = self.byte_offset
                ),
            });
        }
        Ok(self.byte_offset..end)
    }

    fn storage_bytes(&self) -> Result<&[u8], Error> {
        match &self.storage {
            Storage::Host(bytes) => {
                let range = self.checked_storage_range(bytes.len())?;
                Ok(&bytes[range])
            }
            #[cfg(target_os = "macos")]
            Storage::Metal(buffer) => {
                let storage_len =
                    usize::try_from(buffer.length()).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal buffer length does not fit usize".to_string(),
                    })?;
                let range = self.checked_storage_range(storage_len)?;
                let contents = buffer.contents();
                if contents.is_null() {
                    return Err(Error::MetalKernel {
                        message: "J2K Metal surface buffer is not host-addressable".to_string(),
                    });
                }
                // SAFETY: Metal surface byte views are bounded by validated dimensions and formats.
                unsafe {
                    Ok(core::slice::from_raw_parts(
                        contents.cast::<u8>().add(range.start),
                        range.len(),
                    ))
                }
            }
        }
    }

    /// Return the tightly packed surface bytes.
    ///
    /// Metal-backed surfaces are expected to use host-addressable buffers. This
    /// method panics only if the surface metadata is internally inconsistent;
    /// fallible operations such as [`Self::download_into`] return those errors.
    pub fn as_bytes(&self) -> &[u8] {
        self.storage_bytes()
            .expect("validated J2K Metal surface byte range")
    }

    /// Copy the tightly packed surface into a caller-provided strided buffer.
    pub fn download_into(&self, out: &mut [u8], stride: usize) -> Result<(), Error> {
        copy_tight_pixels_to_strided_output(
            self.storage_bytes()?,
            self.dimensions,
            self.fmt,
            out,
            stride,
        )
        .map_err(Error::from)
    }

    #[cfg(target_os = "macos")]
    /// Return the Metal buffer and byte offset when the surface is Metal-backed.
    pub fn metal_buffer(&self) -> Option<(&Buffer, usize)> {
        match &self.storage {
            Storage::Metal(buffer) => Some((buffer, self.byte_offset)),
            Storage::Host(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            byte_offset: 0,
            storage: Storage::Metal(buffer),
        }
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn from_metal_buffer_with_offset(
        buffer: Buffer,
        dimensions: (u32, u32),
        fmt: PixelFormat,
        byte_offset: usize,
    ) -> Self {
        Self {
            backend: BackendKind::Metal,
            residency: SurfaceResidency::MetalResidentDecode,
            dimensions,
            fmt,
            pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
            byte_offset,
            storage: Storage::Metal(buffer),
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
            Storage::Metal(buffer) => Some(DeviceMemoryRange::new(
                BackendKind::Metal,
                u64::try_from(buffer.as_ptr() as usize).ok()?,
                self.byte_offset,
                self.byte_len(),
            )),
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
/// Reusable Metal device session for J2K decode and encode submissions.
pub struct MetalBackendSession {
    device: Device,
    runtime: Arc<OnceLock<Result<Arc<crate::compute::MetalRuntime>, MetalSupportError>>>,
    direct_gray_plan_cache: Arc<Mutex<HashMap<u64, DirectGrayPlanCacheEntry>>>,
    direct_color_plan_cache: Arc<Mutex<HashMap<u64, DirectColorPlanCacheEntry>>>,
    region_scaled_color_plan_cache:
        Arc<Mutex<HashMap<u64, Arc<crate::compute::PreparedDirectColorPlan>>>>,
}

#[cfg(target_os = "macos")]
impl MetalBackendSession {
    /// Create a session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            runtime: Arc::new(OnceLock::new()),
            direct_gray_plan_cache: Arc::new(Mutex::new(HashMap::new())),
            direct_color_plan_cache: Arc::new(Mutex::new(HashMap::new())),
            region_scaled_color_plan_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a session from the system default Metal device.
    pub fn system_default() -> Result<Self, Error> {
        system_default_device()
            .map(Self::new)
            .map_err(|error| crate::compute::runtime_initialization_error(&error))
    }

    /// Metal device used by this session.
    pub fn device(&self) -> &metal::DeviceRef {
        self.device.as_ref()
    }

    pub(crate) fn runtime(&self) -> Result<Arc<crate::compute::MetalRuntime>, Error> {
        match self.runtime.get_or_init(|| {
            crate::compute::MetalRuntime::new_with_device(&self.device).map(Arc::new)
        }) {
            Ok(runtime) => Ok(runtime.clone()),
            Err(error) => Err(crate::compute::runtime_initialization_error(error)),
        }
    }

    #[cfg(test)]
    pub(crate) fn direct_cache_ids_for_test(&self) -> (usize, usize, usize) {
        (
            Arc::as_ptr(&self.direct_gray_plan_cache) as usize,
            Arc::as_ptr(&self.direct_color_plan_cache) as usize,
            Arc::as_ptr(&self.region_scaled_color_plan_cache) as usize,
        )
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
            .finish_non_exhaustive()
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

#[derive(Clone, Default)]
/// Shared batching session used by J2K Metal submit APIs.
pub struct MetalSession {
    shared: batch::SharedSession,
    #[cfg(target_os = "macos")]
    backend: Option<MetalBackendSession>,
}

impl MetalSession {
    /// Create a batching session backed by an explicit Metal backend session.
    #[cfg(target_os = "macos")]
    pub fn with_backend_session(backend: MetalBackendSession) -> Self {
        Self {
            shared: batch::SharedSession::default(),
            backend: Some(backend),
        }
    }

    /// Metal backend session owned by this batching session, if any.
    #[cfg(target_os = "macos")]
    pub fn backend_session(&self) -> Option<&MetalBackendSession> {
        self.backend.as_ref()
    }

    /// Number of Metal or emulated submissions flushed through this session.
    pub fn submissions(&self) -> Result<u64, Error> {
        Ok(self.shared.lock()?.submissions)
    }

    fn record_submit(&mut self) -> Result<(), Error> {
        let mut session = self.shared.lock()?;
        session.submissions = session.submissions.saturating_add(1);
        Ok(())
    }
}

impl core::fmt::Debug for MetalSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalSession")
            .field("submissions", &self.submissions())
            .finish()
    }
}

/// Convenience wrapper for submitting a group of J2K/HTJ2K tiles to one
/// decoder session.
///
/// This is intentionally codec-scoped: callers own slide metadata, tile
/// coordinates, cache policy, and viewport decisions. The batch only preserves
/// submission order and lets compatible tile requests share the Metal session.
#[derive(Default)]
pub struct MetalTileBatch {
    session: MetalSession,
    submissions: Vec<batch::MetalSubmission>,
}

impl MetalTileBatch {
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
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::Full,
        )?;
        self.submissions.push(submission);
        Ok(slot)
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
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::Region(roi),
        )?;
        self.submissions.push(submission);
        Ok(slot)
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
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::Scaled(scale),
        )?;
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Queue a region decode at reduced resolution, copying the compressed tile bytes.
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

    /// Queue a region decode at reduced resolution backed by shared compressed tile bytes.
    pub fn push_shared_tile_region_scaled(
        &mut self,
        input: Arc<[u8]>,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<usize, Error> {
        let slot = self.submissions.len();
        let submission = batch::queue_tile_request_shared(
            &mut self.session,
            input,
            fmt,
            backend,
            batch::BatchOp::RegionScaled { roi, scale },
        )?;
        self.submissions.push(submission);
        Ok(slot)
    }

    /// Decode all queued tile requests and return surfaces in submission order.
    pub fn decode_all(self) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(self.submissions.len());
        for submission in self.submissions {
            surfaces.push(submission.wait()?);
        }
        Ok(surfaces)
    }
}

/// JPEG 2000 decoder that can return host or Metal-resident surfaces.
pub struct J2kDecoder<'a> {
    inner: CpuDecoder<'a>,
    pool: CpuJ2kScratchPool,
    #[cfg(target_os = "macos")]
    native_image: Option<NativeImage<'a>>,
    #[cfg(target_os = "macos")]
    native_context: NativeDecoderContext<'a>,
    #[cfg(target_os = "macos")]
    native_direct_gray_plan: Option<J2kDirectGrayscalePlan>,
    #[cfg(target_os = "macos")]
    native_prepared_direct_gray_plan: Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>,
    #[cfg(target_os = "macos")]
    native_direct_color_plan: Option<J2kDirectColorPlan>,
    #[cfg(target_os = "macos")]
    native_prepared_direct_color_plan: Option<Arc<crate::compute::PreparedDirectColorPlan>>,
}

impl<'a> J2kDecoder<'a> {
    /// Parse a J2K or HTJ2K codestream into a decoder.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
            pool: CpuJ2kScratchPool::new(),
            #[cfg(target_os = "macos")]
            native_image: None,
            #[cfg(target_os = "macos")]
            native_context: NativeDecoderContext::default(),
            #[cfg(target_os = "macos")]
            native_direct_gray_plan: None,
            #[cfg(target_os = "macos")]
            native_prepared_direct_gray_plan: None,
            #[cfg(target_os = "macos")]
            native_direct_color_plan: None,
            #[cfg(target_os = "macos")]
            native_prepared_direct_color_plan: None,
        })
    }

    /// Create a decoder from an already parsed J2K view.
    pub fn from_view(view: J2kView<'a>) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
            pool: CpuJ2kScratchPool::new(),
            #[cfg(target_os = "macos")]
            native_image: None,
            #[cfg(target_os = "macos")]
            native_context: NativeDecoderContext::default(),
            #[cfg(target_os = "macos")]
            native_direct_gray_plan: None,
            #[cfg(target_os = "macos")]
            native_prepared_direct_gray_plan: None,
            #[cfg(target_os = "macos")]
            native_direct_color_plan: None,
            #[cfg(target_os = "macos")]
            native_prepared_direct_color_plan: None,
        })
    }

    /// Borrow the underlying CPU J2K decoder.
    pub fn inner(&self) -> &CpuDecoder<'a> {
        &self.inner
    }

    /// Decode a full image into a Metal-resident device surface.
    pub fn decode_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if let Some(error) =
            routing::decision_error(routing::decide_route(BackendRequest::Metal, fmt))
        {
            return Err(error);
        }

        #[cfg(target_os = "macos")]
        {
            crate::compute::with_runtime_for_session(session, |_| {
                if let Some(surface) = self.decode_direct_to_surface_with_session(fmt, session)? {
                    Ok(surface)
                } else {
                    self.decode_full_to_metal_surface_with_device(fmt, &session.device)
                }
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = session;
            Err(Error::MetalUnavailable)
        }
    }

    /// Decode a full image into a host-backed surface.
    pub fn decode_to_host_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        self.decode_to_cpu_surface(fmt)
    }

    /// Decode a source region into a host-backed surface.
    pub fn decode_region_to_host_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        self.decode_region_to_cpu_surface(fmt, plan)
    }

    /// Decode a full image at reduced resolution into a host-backed surface.
    pub fn decode_scaled_to_host_surface(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?;
        self.decode_scaled_to_cpu_surface(fmt, scale, plan)
    }

    /// Decode a source region at reduced resolution into a host-backed surface.
    pub fn decode_region_scaled_to_host_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?;
        self.decode_region_scaled_to_cpu_surface(fmt, roi, scale, plan)
    }

    /// Decode a full image on CPU and upload the result into a Metal surface.
    pub fn decode_to_cpu_staged_metal_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let dims = self.inner.info().dimensions;
            let stride = dims.0 as usize * fmt.bytes_per_pixel();
            let mut out = vec![0u8; stride * dims.1 as usize];
            self.inner
                .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
            Ok(upload_surface_to_metal_with_device(
                &out,
                dims,
                fmt,
                session.device(),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (fmt, session);
            Err(Error::MetalUnavailable)
        }
    }

    /// Decode a source region on CPU and upload the result into a Metal surface.
    pub fn decode_region_to_cpu_staged_metal_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let plan = DeviceDecodePlan::for_image(
                self.inner.info().dimensions,
                DeviceDecodeRequest::Region { roi },
            )?;
            let dims = plan.output_dims();
            let stride = dims.0 as usize * fmt.bytes_per_pixel();
            let mut out = vec![0u8; stride * dims.1 as usize];
            self.inner.decode_region_into(
                &mut self.pool,
                &mut out,
                stride,
                fmt,
                plan.source_rect(),
            )?;
            Ok(upload_surface_to_metal_with_device(
                &out,
                dims,
                fmt,
                session.device(),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (fmt, roi, session);
            Err(Error::MetalUnavailable)
        }
    }

    /// Decode a full image at reduced resolution on CPU and upload it into Metal.
    pub fn decode_scaled_to_cpu_staged_metal_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let plan = DeviceDecodePlan::for_image(
                self.inner.info().dimensions,
                DeviceDecodeRequest::Scaled { scale },
            )?;
            let dims = plan.output_dims();
            let stride = dims.0 as usize * fmt.bytes_per_pixel();
            let mut out = vec![0u8; stride * dims.1 as usize];
            self.inner
                .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
            Ok(upload_surface_to_metal_with_device(
                &out,
                dims,
                fmt,
                session.device(),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (fmt, scale, session);
            Err(Error::MetalUnavailable)
        }
    }

    /// Decode a scaled source region on CPU and upload it into a Metal surface.
    pub fn decode_region_scaled_to_cpu_staged_metal_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            let plan = DeviceDecodePlan::for_image(
                self.inner.info().dimensions,
                DeviceDecodeRequest::RegionScaled { roi, scale },
            )?;
            let dims = plan.output_dims();
            let stride = dims.0 as usize * fmt.bytes_per_pixel();
            let mut out = vec![0u8; stride * dims.1 as usize];
            self.inner.decode_region_scaled_into(
                &mut self.pool,
                &mut out,
                stride,
                fmt,
                roi,
                scale,
            )?;
            Ok(upload_surface_to_metal_with_device(
                &out,
                dims,
                fmt,
                session.device(),
            ))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (fmt, roi, scale, session);
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    fn ensure_native_image(&mut self) -> Result<(), Error> {
        if self.native_image.is_none() {
            self.native_image = Some(
                NativeImage::new(self.inner.bytes(), &NativeDecodeSettings::default())
                    .map_err(|error| J2kError::Backend(error.to_string()))?,
            );
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn ensure_prepared_direct_gray_plan_with_session(
        &mut self,
        session: &MetalBackendSession,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>, Error> {
        let cache_key = direct_gray_plan_cache_key(self.inner.bytes());
        if self.native_prepared_direct_gray_plan.is_none() {
            if let Some((plan, prepared)) = cached_session_direct_gray_plan(session, cache_key) {
                self.native_direct_gray_plan = Some(plan);
                self.native_prepared_direct_gray_plan = Some(prepared);
            }
        }
        if self.native_prepared_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };

            let plan = match image.build_direct_grayscale_plan_with_context(native_context) {
                Ok(plan) => plan,
                Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
                    return Ok(None);
                }
                Err(error) => {
                    return Err(Error::Decode(J2kError::Backend(format!(
                        "failed to build J2K MetalDirect grayscale plan: {error}"
                    ))));
                }
            };
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            store_session_direct_gray_plan(session, cache_key, &plan, prepared.clone());
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }

        Ok(self.native_prepared_direct_gray_plan.clone())
    }

    #[cfg(target_os = "macos")]
    fn ensure_prepared_direct_color_plan_with_session(
        &mut self,
        session: &MetalBackendSession,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
        let cache_key = direct_plan_cache_key(self.inner.bytes());
        if self.native_prepared_direct_color_plan.is_none() {
            if let Some((plan, prepared)) = cached_session_direct_color_plan(session, cache_key) {
                self.native_direct_color_plan = Some(plan);
                self.native_prepared_direct_color_plan = Some(prepared);
            }
        }
        if self.native_prepared_direct_color_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };

            let plan = match image.build_direct_color_plan_with_context(native_context) {
                Ok(plan) => plan,
                Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
                    return Ok(None);
                }
                Err(error) => {
                    return Err(Error::Decode(J2kError::Backend(format!(
                        "failed to build J2K MetalDirect color plan: {error}"
                    ))));
                }
            };
            let prepared = Arc::new(crate::compute::prepare_direct_color_plan(&plan)?);
            store_session_direct_color_plan(session, cache_key, &plan, prepared.clone());
            self.native_direct_color_plan = Some(plan);
            self.native_prepared_direct_color_plan = Some(prepared);
        }

        Ok(self.native_prepared_direct_color_plan.clone())
    }

    #[cfg(target_os = "macos")]
    fn ensure_prepared_direct_gray_plan(
        &mut self,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>, Error> {
        if self.native_prepared_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };
            let plan = match image.build_direct_grayscale_plan_with_context(native_context) {
                Ok(plan) => plan,
                Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
                    return Ok(None);
                }
                Err(error) => {
                    return Err(Error::Decode(J2kError::Backend(format!(
                        "failed to build J2K MetalDirect grayscale plan: {error}"
                    ))));
                }
            };
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        Ok(self.native_prepared_direct_gray_plan.clone())
    }

    #[cfg(target_os = "macos")]
    fn ensure_prepared_direct_color_plan(
        &mut self,
    ) -> Result<Option<Arc<crate::compute::PreparedDirectColorPlan>>, Error> {
        if self.native_prepared_direct_color_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };
            let plan = match image.build_direct_color_plan_with_context(native_context) {
                Ok(plan) => plan,
                Err(error) if direct::is_unsupported_direct_plan_error(&error.to_string()) => {
                    return Ok(None);
                }
                Err(error) => {
                    return Err(Error::Decode(J2kError::Backend(format!(
                        "failed to build J2K MetalDirect color plan: {error}"
                    ))));
                }
            };
            let prepared = Arc::new(crate::compute::prepare_direct_color_plan(&plan)?);
            self.native_direct_color_plan = Some(plan);
            self.native_prepared_direct_color_plan = Some(prepared);
        }
        Ok(self.native_prepared_direct_color_plan.clone())
    }

    #[cfg(target_os = "macos")]
    fn decode_direct_to_surface(&mut self, fmt: PixelFormat) -> Result<Option<Surface>, Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let Some(plan) = self.ensure_prepared_direct_gray_plan()? else {
                return Ok(None);
            };
            return Ok(Some(
                crate::compute::execute_prepared_direct_grayscale_plan(&plan, fmt)?,
            ));
        }

        if matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        ) {
            let Some(plan) = self.ensure_prepared_direct_color_plan()? else {
                return Ok(None);
            };
            return match crate::compute::execute_prepared_direct_color_plan(&plan, fmt) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_color_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            };
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    fn decode_direct_to_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &MetalBackendSession,
    ) -> Result<Option<Surface>, Error> {
        if matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            let Some(plan) = self.ensure_prepared_direct_gray_plan_with_session(session)? else {
                return Ok(None);
            };
            return Ok(Some(
                crate::compute::execute_prepared_direct_grayscale_plan_with_device(
                    &plan,
                    fmt,
                    &session.device,
                )?,
            ));
        }

        if matches!(
            fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        ) {
            let Some(plan) = self.ensure_prepared_direct_color_plan_with_session(session)? else {
                return Ok(None);
            };
            return match crate::compute::execute_prepared_direct_color_plan_with_device(
                &plan,
                fmt,
                &session.device,
            ) {
                Ok(surface) => Ok(Some(surface)),
                Err(error) if is_direct_color_runtime_fallback_error(&error) => Ok(None),
                Err(error) => Err(error),
            };
        }

        Ok(None)
    }

    #[cfg(target_os = "macos")]
    fn decode_region_scaled_direct_to_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<Option<Surface>, Error> {
        crate::hybrid::decode_region_scaled_direct_to_surface(self.inner.bytes(), fmt, roi, scale)
    }

    #[cfg(target_os = "macos")]
    fn decode_region_scaled_direct_to_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Option<Surface>, Error> {
        crate::hybrid::decode_region_scaled_direct_to_surface_with_session(
            self.inner.bytes(),
            fmt,
            roi,
            scale,
            session,
        )
    }
    #[cfg(target_os = "macos")]
    fn decode_full_to_metal_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_to_surface(image, native_context, fmt)
    }

    #[cfg(target_os = "macos")]
    fn decode_full_to_metal_surface_with_device(
        &mut self,
        fmt: PixelFormat,
        device: &Device,
    ) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_to_surface_with_device(image, native_context, fmt, device)
    }

    #[cfg(target_os = "macos")]
    fn decode_repeated_grayscale_cpu_to_surfaces(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        let mut surfaces = Vec::with_capacity(count);
        for _ in 0..count {
            surfaces.push(self.decode_to_cpu_surface(fmt)?);
        }
        Ok(surfaces)
    }

    #[cfg(target_os = "macos")]
    fn should_auto_use_direct_for_repeated(
        plan: &J2kDirectGrayscalePlan,
        fmt: PixelFormat,
        count: usize,
    ) -> bool {
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) || count == 0 {
            return false;
        }

        let max_dim = plan.dimensions.0.max(plan.dimensions.1);
        max_dim >= AUTO_REPEATED_GRAYSCALE_MIN_DIM && count >= AUTO_REPEATED_GRAYSCALE_MIN_COUNT
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if self.native_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };
            let plan = image
                .build_direct_grayscale_plan_with_context(native_context)
                .map_err(|error| J2kError::Backend(error.to_string()))?;
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        let Some(plan) = self.native_prepared_direct_gray_plan.as_ref() else {
            return Ok(Vec::new());
        };
        crate::compute::execute_repeated_prepared_direct_grayscale_plan(plan, fmt, count)
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_color_direct_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let surface = self.decode_to_surface_impl(fmt, BackendRequest::Metal)?;
        Ok(vec![surface; count])
    }

    #[cfg(target_os = "macos")]
    #[doc(hidden)]
    pub fn decode_repeated_grayscale_auto_to_device(
        &mut self,
        fmt: PixelFormat,
        count: usize,
    ) -> Result<Vec<Surface>, Error> {
        if count == 0 {
            return Ok(Vec::new());
        }
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        }
        let dims = self.inner.info().dimensions;
        if dims.0.max(dims.1) < AUTO_REPEATED_GRAYSCALE_MIN_DIM
            || count < AUTO_REPEATED_GRAYSCALE_MIN_COUNT
        {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        }
        if self.native_direct_gray_plan.is_none() {
            self.ensure_native_image()?;
            let (Some(image), native_context) =
                (self.native_image.as_ref(), &mut self.native_context)
            else {
                return Err(Error::Decode(J2kError::Backend(
                    "native image cache missing".to_string(),
                )));
            };
            let Ok(plan) = image.build_direct_grayscale_plan_with_context(native_context) else {
                return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
            };
            let prepared = Arc::new(crate::compute::prepare_direct_grayscale_plan(&plan)?);
            self.native_direct_gray_plan = Some(plan);
            self.native_prepared_direct_gray_plan = Some(prepared);
        }
        let Some(plan) = self.native_direct_gray_plan.as_ref() else {
            return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
        };
        if Self::should_auto_use_direct_for_repeated(plan, fmt, count) {
            let Some(prepared) = self.native_prepared_direct_gray_plan.as_ref() else {
                return self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count);
            };
            crate::compute::execute_repeated_prepared_direct_grayscale_plan(prepared, fmt, count)
        } else {
            self.decode_repeated_grayscale_cpu_to_surfaces(fmt, count)
        }
    }

    fn decode_to_cpu_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    fn decode_region_to_cpu_surface(
        &mut self,
        fmt: PixelFormat,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_region_into(&mut self.pool, &mut out, stride, fmt, plan.source_rect())?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    fn decode_scaled_to_cpu_surface(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    fn decode_region_scaled_to_cpu_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        let dims = plan.output_dims();
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        self.inner
            .decode_region_scaled_into(&mut self.pool, &mut out, stride, fmt, roi, scale)?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    #[cfg(target_os = "macos")]
    fn decode_region_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_region_to_surface(
            image,
            native_context,
            fmt,
            plan.source_rect(),
        )
    }

    #[cfg(target_os = "macos")]
    fn decode_scaled_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        crate::compute::decode_scaled_to_surface(self.inner.bytes(), plan.source_dims(), fmt, scale)
    }

    #[cfg(target_os = "macos")]
    fn decode_region_scaled_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        plan: DeviceDecodePlan,
    ) -> Result<Surface, Error> {
        if let Some(surface) = self.decode_region_scaled_direct_to_surface(fmt, roi, scale)? {
            return Ok(surface);
        }
        crate::compute::decode_region_scaled_to_surface(
            self.inner.bytes(),
            plan.source_dims(),
            fmt,
            roi,
            scale,
        )
    }

    #[cfg(target_os = "macos")]
    fn decode_region_to_metal_surface_with_device(
        &mut self,
        fmt: PixelFormat,
        plan: DeviceDecodePlan,
        device: &Device,
    ) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(J2kError::Backend(
                "native image cache missing".to_string(),
            )));
        };
        crate::compute::decode_image_region_to_surface_with_device(
            image,
            native_context,
            fmt,
            plan.source_rect(),
            device,
        )
    }

    #[cfg(target_os = "macos")]
    fn decode_scaled_to_metal_surface_with_device(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        plan: DeviceDecodePlan,
        device: &Device,
    ) -> Result<Surface, Error> {
        crate::compute::decode_scaled_to_surface_with_device(
            self.inner.bytes(),
            plan.source_dims(),
            fmt,
            scale,
            device,
        )
    }
    #[cfg(target_os = "macos")]
    fn decode_region_scaled_to_metal_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        plan: DeviceDecodePlan,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if let Some(surface) =
            self.decode_region_scaled_direct_to_surface_with_session(fmt, roi, scale, session)?
        {
            return Ok(surface);
        }
        crate::compute::with_runtime_for_session(session, |_| {
            crate::compute::decode_region_scaled_to_surface_with_device(
                self.inner.bytes(),
                plan.source_dims(),
                fmt,
                roi,
                scale,
                &session.device,
            )
        })
    }

    pub(crate) fn decode_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let route = routing::decide_route(backend, fmt);
        if let Some(error) = routing::decision_error(route) {
            return Err(error);
        }

        match route {
            routing::RouteDecision::CpuHost => self.decode_to_cpu_surface(fmt),
            #[cfg(target_os = "macos")]
            routing::RouteDecision::MetalKernel => {
                if let Some(surface) = self.decode_direct_to_surface(fmt)? {
                    Ok(surface)
                } else {
                    self.decode_full_to_metal_surface(fmt)
                }
            }
            routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. } => {
                unreachable!("handled by decision_error")
            }
            #[cfg(not(target_os = "macos"))]
            routing::RouteDecision::MetalUnavailable => unreachable!("handled by decision_error"),
        }
    }

    pub(crate) fn decode_region_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let route = routing::decide_route(backend, fmt);
        if let Some(error) = routing::decision_error(route) {
            return Err(error);
        }

        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        match route {
            routing::RouteDecision::CpuHost => self.decode_region_to_cpu_surface(fmt, plan),
            #[cfg(target_os = "macos")]
            routing::RouteDecision::MetalKernel => self.decode_region_to_metal_surface(fmt, plan),
            routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. } => {
                unreachable!("handled by decision_error")
            }
            #[cfg(not(target_os = "macos"))]
            routing::RouteDecision::MetalUnavailable => unreachable!("handled by decision_error"),
        }
    }

    /// Decode a source region into a Metal-resident device surface.
    pub fn decode_region_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if let Some(error) =
            routing::decision_error(routing::decide_route(BackendRequest::Metal, fmt))
        {
            return Err(error);
        }

        #[cfg(target_os = "macos")]
        {
            let plan = DeviceDecodePlan::for_image(
                self.inner.info().dimensions,
                DeviceDecodeRequest::Region { roi },
            )?;
            crate::compute::with_runtime_for_session(session, |_| {
                self.decode_region_to_metal_surface_with_device(fmt, plan, &session.device)
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (roi, session);
            Err(Error::MetalUnavailable)
        }
    }

    pub(crate) fn decode_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let route = routing::decide_route(backend, fmt);
        if let Some(error) = routing::decision_error(route) {
            return Err(error);
        }

        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?;
        match route {
            routing::RouteDecision::CpuHost => self.decode_scaled_to_cpu_surface(fmt, scale, plan),
            #[cfg(target_os = "macos")]
            routing::RouteDecision::MetalKernel => {
                self.decode_scaled_to_metal_surface(fmt, scale, plan)
            }
            routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. } => {
                unreachable!("handled by decision_error")
            }
            #[cfg(not(target_os = "macos"))]
            routing::RouteDecision::MetalUnavailable => unreachable!("handled by decision_error"),
        }
    }

    pub(crate) fn decode_region_scaled_to_surface_impl(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        let route = routing::decide_route(backend, fmt);
        if let Some(error) = routing::decision_error(route) {
            return Err(error);
        }
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?;
        match route {
            routing::RouteDecision::CpuHost => {
                self.decode_region_scaled_to_cpu_surface(fmt, roi, scale, plan)
            }
            #[cfg(target_os = "macos")]
            routing::RouteDecision::MetalKernel => {
                self.decode_region_scaled_to_metal_surface(fmt, roi, scale, plan)
            }
            routing::RouteDecision::RejectExplicitMetal { .. }
            | routing::RouteDecision::RejectUnsupportedBackend { .. } => {
                unreachable!("handled by decision_error")
            }
            #[cfg(not(target_os = "macos"))]
            routing::RouteDecision::MetalUnavailable => unreachable!("handled by decision_error"),
        }
    }

    /// Decode a full image at reduced resolution into a Metal-resident surface.
    pub fn decode_scaled_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if let Some(error) =
            routing::decision_error(routing::decide_route(BackendRequest::Metal, fmt))
        {
            return Err(error);
        }
        if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
            return Err(Error::UnsupportedMetalRequest {
                reason: "J2K Metal session scaled decode currently supports Gray8/Gray16 only",
            });
        }

        #[cfg(target_os = "macos")]
        {
            let plan = DeviceDecodePlan::for_image(
                self.inner.info().dimensions,
                DeviceDecodeRequest::Scaled { scale },
            )?;
            crate::compute::with_runtime_for_session(session, |_| {
                self.decode_scaled_to_metal_surface_with_device(fmt, scale, plan, &session.device)
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (scale, session);
            Err(Error::MetalUnavailable)
        }
    }

    /// Decode a scaled source region into a Metal-resident device surface.
    pub fn decode_region_scaled_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if let Some(error) =
            routing::decision_error(routing::decide_route(BackendRequest::Metal, fmt))
        {
            return Err(error);
        }

        #[cfg(target_os = "macos")]
        {
            let plan = DeviceDecodePlan::for_image(
                self.inner.info().dimensions,
                DeviceDecodeRequest::RegionScaled { roi, scale },
            )?;
            self.decode_region_scaled_to_metal_surface_with_session(fmt, roi, scale, plan, session)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (roi, scale, session);
            Err(Error::MetalUnavailable)
        }
    }
}

#[cfg(target_os = "macos")]
fn direct_plan_cache_key(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

#[cfg(target_os = "macos")]
fn direct_gray_plan_cache_key(bytes: &[u8]) -> u64 {
    direct_plan_cache_key(bytes)
}

#[cfg(target_os = "macos")]
fn cached_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: u64,
) -> Option<(
    J2kDirectGrayscalePlan,
    Arc<crate::compute::PreparedDirectGrayscalePlan>,
)> {
    let guard = session.direct_gray_plan_cache.lock().ok()?;
    guard
        .get(&key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone()))
}

#[cfg(target_os = "macos")]
fn store_session_direct_gray_plan(
    session: &MetalBackendSession,
    key: u64,
    plan: &J2kDirectGrayscalePlan,
    prepared: Arc<crate::compute::PreparedDirectGrayscalePlan>,
) {
    if let Ok(mut guard) = session.direct_gray_plan_cache.lock() {
        evict_one_direct_plan_if_needed(&mut guard);
        guard.insert(
            key,
            DirectGrayPlanCacheEntry {
                plan: plan.clone(),
                prepared,
            },
        );
    }
}

#[cfg(target_os = "macos")]
fn cached_session_direct_color_plan(
    session: &MetalBackendSession,
    key: u64,
) -> Option<(
    J2kDirectColorPlan,
    Arc<crate::compute::PreparedDirectColorPlan>,
)> {
    let guard = session.direct_color_plan_cache.lock().ok()?;
    guard
        .get(&key)
        .map(|entry| (entry.plan.clone(), entry.prepared.clone()))
}

#[cfg(target_os = "macos")]
fn store_session_direct_color_plan(
    session: &MetalBackendSession,
    key: u64,
    plan: &J2kDirectColorPlan,
    prepared: Arc<crate::compute::PreparedDirectColorPlan>,
) {
    if let Ok(mut guard) = session.direct_color_plan_cache.lock() {
        evict_one_direct_plan_if_needed(&mut guard);
        guard.insert(
            key,
            DirectColorPlanCacheEntry {
                plan: plan.clone(),
                prepared,
            },
        );
    }
}

#[cfg(target_os = "macos")]
fn evict_one_direct_plan_if_needed<T>(cache: &mut HashMap<u64, T>) {
    if cache.len() < DIRECT_PLAN_CACHE_CAP {
        return;
    }
    if let Some(key) = cache.keys().next().copied() {
        cache.remove(&key);
    }
}

#[cfg(target_os = "macos")]
fn is_direct_color_runtime_fallback_error(error: &Error) -> bool {
    is_direct_runtime_fallback_error(error)
}

#[cfg(target_os = "macos")]
fn is_direct_runtime_fallback_error(error: &Error) -> bool {
    matches!(
        error,
        Error::MetalKernel { message }
            if message.contains("unsupported classic kernel input")
                || message.contains("unsupported HT kernel input")
                || message.contains("direct component plan")
                || message.contains("currently supports grayscale direct plans only")
                || message.contains("currently supports color direct plans only")
    )
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_grayscale_batch_direct_to_device(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(fmt, PixelFormat::Gray8 | PixelFormat::Gray16) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full grayscale batch does not support {fmt:?}"),
        });
    }

    let mut plans = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let Some(plan) = decoder.ensure_prepared_direct_gray_plan()? else {
            return Err(Error::MetalKernel {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full grayscale Gray8/Gray16 only; fmt={fmt:?}"
                ),
            });
        };
        plans.push(plan);
    }
    crate::compute::execute_prepared_direct_grayscale_plan_batch(&plans, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_color_batch_direct_to_device(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }
    if !matches!(
        fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        return Err(Error::MetalKernel {
            message: format!("J2K MetalDirect full color batch does not support {fmt:?}"),
        });
    }

    let mut plans = Vec::with_capacity(inputs.len());
    for input in inputs {
        let mut decoder = J2kDecoder::new(input.as_ref())?;
        let Some(plan) = decoder.ensure_prepared_direct_color_plan()? else {
            return Err(Error::MetalKernel {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full RGB color only; fmt={fmt:?}"
                ),
            });
        };
        plans.push(plan);
    }
    match crate::compute::execute_prepared_direct_color_plan_batch(&plans, fmt) {
        Ok(surfaces) => Ok(surfaces),
        Err(error) if is_direct_color_runtime_fallback_error(&error) => {
            Err(Error::UnsupportedMetalRequest {
                reason: CPU_STAGED_METAL_REQUIRES_EXPLICIT_API,
            })
        }
        Err(error) => Err(error),
    }
}

impl ImageCodec for J2kDecoder<'_> {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(J2kView::parse(input)?)
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
        Ok(self.inner.decode_into(out, stride, fmt)?)
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
            .decode_into_with_scratch(pool, out, stride, fmt)?)
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        Ok(self.inner.decode_region_into(pool, out, stride, fmt, roi)?)
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
            .decode_scaled_into(pool, out, stride, fmt, scale)?)
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
            .decode_region_scaled_into(pool, out, stride, fmt, roi, scale)?)
    }
}

impl<'a> ImageDecodeDevice<'a> for J2kDecoder<'a> {
    type DeviceSurface = Surface;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// J2K codec marker used by J2K's generic decode traits.
pub struct Codec;

impl ImageCodec for Codec {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl<'a> ImageDecodeSubmit<'a> for J2kDecoder<'a> {
    type Session = MetalSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(
            self.decode_to_surface_impl(fmt, backend),
        ))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(
            self.decode_region_to_surface_impl(fmt, roi, backend),
        ))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(
            self.decode_scaled_to_surface_impl(fmt, scale, backend),
        ))
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(
            self.decode_region_scaled_to_surface_impl(fmt, roi, scale, backend),
        ))
    }
}

impl TileBatchDecodeSubmit for Codec {
    type Context = CpuJ2kContext;
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
        batch::queue_tile_request(session, input, fmt, backend, batch::BatchOp::Full)
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
        batch::queue_tile_request(session, input, fmt, backend, batch::BatchOp::Region(roi))
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
        batch::queue_tile_request(session, input, fmt, backend, batch::BatchOp::Scaled(scale))
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
        let _ = (ctx, pool);
        batch::queue_tile_request(
            session,
            input,
            fmt,
            backend,
            batch::BatchOp::RegionScaled { roi, scale },
        )
    }
}

impl TileBatchDecodeManyDevice for Codec {
    type Context = CpuJ2kContext;
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
            .map(j2k_core::DeviceSubmission::wait)
            .collect()
    }
}

impl TileBatchDecodeDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;
}

fn upload_surface(
    bytes: Vec<u8>,
    dimensions: (u32, u32),
    fmt: PixelFormat,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    match backend {
        BackendRequest::Cpu | BackendRequest::Auto => Ok(Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions,
            fmt,
            pitch_bytes,
            byte_offset: 0,
            storage: Storage::Host(bytes),
        }),
        BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                let _ = bytes;
                Err(Error::UnsupportedMetalRequest {
                    reason: CPU_STAGED_METAL_REQUIRES_EXPLICIT_API,
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = bytes;
                Err(Error::MetalUnavailable)
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

#[cfg(target_os = "macos")]
fn upload_surface_to_metal_with_device(
    bytes: &[u8],
    dimensions: (u32, u32),
    fmt: PixelFormat,
    device: &metal::DeviceRef,
) -> Surface {
    let pitch_bytes = dimensions.0 as usize * fmt.bytes_per_pixel();
    let buffer = device.new_buffer_with_data(
        bytes.as_ptr().cast(),
        bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    Surface {
        backend: BackendKind::Metal,
        residency: SurfaceResidency::CpuStagedMetalUpload,
        dimensions,
        fmt,
        pitch_bytes,
        byte_offset: 0,
        storage: Storage::Metal(buffer),
    }
}

pub use j2k::{J2kContext, J2kScratchPool};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_runtime_failures_are_not_unsupported_errors() {
        for err in [
            Error::MetalRuntime {
                message: "runtime".to_string(),
            },
            Error::MetalKernel {
                message: "kernel".to_string(),
            },
            Error::MetalStatePoisoned {
                state: "J2K Metal session",
            },
        ] {
            assert!(!err.is_unsupported(), "{err:?}");
        }
    }

    #[test]
    fn cpu_uploaded_surface_reports_host_residency() {
        let surface = upload_surface(
            vec![1, 2, 3],
            (1, 1),
            PixelFormat::Rgb8,
            BackendRequest::Cpu,
        )
        .expect("create CPU surface");

        assert_eq!(surface.backend_kind(), BackendKind::Cpu);
        assert_eq!(surface.residency(), SurfaceResidency::Host);
        #[cfg(target_os = "macos")]
        assert!(surface.metal_buffer().is_none());
    }

    #[test]
    fn download_into_reports_inconsistent_surface_storage_range() {
        let surface = Surface {
            backend: BackendKind::Cpu,
            residency: SurfaceResidency::Host,
            dimensions: (2, 1),
            fmt: PixelFormat::Gray8,
            pitch_bytes: 2,
            byte_offset: 0,
            storage: Storage::Host(vec![7]),
        };
        let mut out = [0_u8; 2];

        let err = surface
            .download_into(&mut out, 2)
            .expect_err("inconsistent surface storage should be reported");

        assert!(matches!(
            err,
            Error::MetalKernel { message }
                if message == "J2K Metal surface byte range 0..2 exceeds storage length 1"
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_backend_sessions_own_distinct_direct_plan_caches() {
        let Some(device) = Device::system_default() else {
            eprintln!("skipping session cache ownership test: no Metal device");
            return;
        };

        let first = MetalBackendSession::new(device.clone());
        let second = MetalBackendSession::new(device);

        assert_ne!(
            first.direct_cache_ids_for_test(),
            second.direct_cache_ids_for_test()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn explicit_metal_request_does_not_stage_cpu_pixels() {
        if Device::system_default().is_none() {
            eprintln!("skipping surface residency test: no Metal device");
            return;
        }

        let result = upload_surface(
            vec![1, 2, 3],
            (1, 1),
            PixelFormat::Rgb8,
            BackendRequest::Metal,
        );

        assert!(matches!(
            result,
            Err(Error::UnsupportedMetalRequest { reason })
                if reason.contains("CPU-staged")
                    && reason.contains("explicit")
                    && reason.contains("Metal")
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn repeated_region_scaled_color_batch_reuses_prepared_plan() {
        if Device::system_default().is_none() {
            eprintln!("skipping repeated color plan reuse test: no Metal device");
            return;
        }

        let pixels = j2k_test_support::gradient_u8(64, 64, 3);
        let options = j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..j2k_native::EncodeOptions::default()
        };
        let input = Arc::<[u8]>::from(
            j2k_native::encode(&pixels, 64, 64, 3, 8, false, &options).expect("encode rgb8"),
        );
        let roi = Rect {
            x: 8,
            y: 8,
            w: 32,
            h: 32,
        };
        let scale = Downscale::Quarter;
        let requests = vec![(input.clone(), roi, scale); 4];
        let _guard = hybrid::region_scaled_color_plan_test_lock_for_test();
        hybrid::reset_region_scaled_color_plan_builds_for_test();

        let surfaces =
            hybrid::decode_region_scaled_color_batch_direct_to_device(&requests, PixelFormat::Rgb8)
                .expect("repeated RGB region-scaled batch");

        assert_eq!(surfaces.len(), requests.len());
        assert_eq!(
            hybrid::region_scaled_color_plan_builds_for_test(),
            1,
            "repeated RGB ROI+scaled batches should build and crop one prepared direct color plan"
        );
    }
}
