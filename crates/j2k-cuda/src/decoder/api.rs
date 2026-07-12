// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use super::resident::{
    decode_batch_to_cuda_resident_surface_with_profile_control,
    decode_region_scaled_to_cuda_resident_surface_impl,
    decode_region_to_cuda_resident_surface_impl, decode_scaled_to_cuda_resident_surface_impl,
    decode_to_cuda_resident_surface_impl, decode_to_cuda_resident_surface_with_profile_impl,
};
use super::{
    checked_surface_len, submit_ready_device, validate_surface_request,
    wrap_cpu_staged_cuda_surface, wrap_surface, BackendRequest, CpuBackedImageDecode, CpuDecoder,
    CpuJ2kScratchPool, CudaHtj2kProfileReport, CudaSession, DecodeOutcome, DeviceDecodePlan,
    DeviceDecodeRequest, Downscale, Error, ImageCodec, ImageDecodeDevice, ImageDecodeSubmit,
    J2kDecodeWarning, J2kDecoder, J2kView, PixelFormat, ReadySubmission, Rect, Surface,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
use crate::allocation::try_vec_filled;

impl<'a> J2kDecoder<'a> {
    /// Create a CUDA-facing decoder from compressed bytes.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            bytes: input,
            inner: CpuDecoder::new(input)?,
            pool: CpuJ2kScratchPool::new(),
        })
    }

    fn decode_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            return self.decode_to_cuda_resident_surface_impl(session, fmt);
        }
        let dims = self.inner.info().dimensions;
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        j2k_profile::emit_gpu_route_surface_profile(
            ("j2k", "cuda"),
            (
                "full",
                format_args!("{backend:?}"),
                format_args!("{fmt:?}"),
                "cpu_decode_then_wrap",
            ),
            dims,
            [],
        );
        self.inner
            .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_to_cuda_resident_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        decode_to_cuda_resident_surface_impl(self, session, fmt)
    }

    fn decode_region_to_cuda_resident_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<Surface, Error> {
        decode_region_to_cuda_resident_surface_impl(self, session, fmt, roi)
    }

    fn decode_scaled_to_cuda_resident_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<Surface, Error> {
        decode_scaled_to_cuda_resident_surface_impl(self, session, fmt, scale)
    }

    fn decode_region_scaled_to_cuda_resident_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<Surface, Error> {
        decode_region_scaled_to_cuda_resident_surface_impl(self, session, fmt, roi, scale)
    }

    fn decode_region_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            return self.decode_region_to_cuda_resident_surface_impl(session, fmt, roi);
        }
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        let dims = plan.output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner
            .decode_region_into(&mut self.pool, &mut out, stride, fmt, plan.source_rect())?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            return self.decode_scaled_to_cuda_resident_surface_impl(session, fmt, scale);
        }
        let dims = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner
            .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    fn decode_region_scaled_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if matches!(backend, BackendRequest::Cuda) {
            return self
                .decode_region_scaled_to_cuda_resident_surface_impl(session, fmt, roi, scale);
        }
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?;
        let dims = plan.output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner.decode_region_scaled_into(
            &mut self.pool,
            &mut out,
            stride,
            fmt,
            plan.source_rect(),
            scale,
        )?;
        wrap_surface(out, dims, fmt, backend, session)
    }

    /// Strictly decode a full HTJ2K image into a CUDA-backed surface using an
    /// existing backend session.
    pub fn decode_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        self.decode_to_surface_impl(session, fmt, BackendRequest::Cuda)
    }

    /// Strictly decode a full HTJ2K image into a CUDA-backed surface and return
    /// a structured profile report for CPU planning and CUDA stages.
    #[doc(hidden)]
    pub fn decode_to_device_with_session_and_profile(
        &mut self,
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
        decode_to_cuda_resident_surface_with_profile_impl(self, session, fmt)
    }

    /// Strictly decode a batch of full HTJ2K images into CUDA-backed surfaces
    /// using an existing backend session.
    pub fn decode_batch_to_device_with_session(
        inputs: &[&[u8]],
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<Vec<Surface>, Error> {
        decode_batch_to_cuda_resident_surface_with_profile_control(inputs, session, fmt, false)
            .map(|(surfaces, _report)| surfaces)
    }

    /// Strictly decode a batch of full HTJ2K images into CUDA-backed surfaces
    /// and return one aggregate profile report for the shared batch.
    #[doc(hidden)]
    pub fn decode_batch_to_device_with_session_and_profile(
        inputs: &[&[u8]],
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
        decode_batch_to_cuda_resident_surface_with_profile_control(inputs, session, fmt, true)
    }

    /// Strictly decode a full-resolution HTJ2K region into a CUDA-backed
    /// surface using an existing backend session.
    pub(crate) fn decode_region_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        self.decode_region_to_surface_impl(session, fmt, roi, BackendRequest::Cuda)
    }

    /// Strictly decode a reduced-resolution HTJ2K image into a CUDA-backed
    /// surface using an existing backend session.
    pub(crate) fn decode_scaled_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        self.decode_scaled_to_surface_impl(session, fmt, scale, BackendRequest::Cuda)
    }

    /// Strictly decode a reduced-resolution HTJ2K region into a CUDA-backed
    /// surface using an existing backend session.
    pub(crate) fn decode_region_scaled_to_device_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        self.decode_region_scaled_to_surface_impl(session, fmt, roi, scale, BackendRequest::Cuda)
    }

    /// Decode a full image through the CPU path and wrap it as a host surface.
    pub fn decode_to_host_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        let mut session = CudaSession::default();
        self.decode_to_surface_impl(&mut session, fmt, BackendRequest::Cpu)
    }

    /// Decode a full image on CPU and upload it into a CUDA buffer using an
    /// existing backend session.
    pub fn decode_to_cpu_staged_cuda_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner
            .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
        wrap_cpu_staged_cuda_surface(&out, dims, fmt, session)
    }

    /// Decode a region on CPU and upload it into a CUDA buffer using an
    /// existing backend session.
    pub fn decode_region_to_cpu_staged_cuda_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Region { roi },
        )?;
        let dims = plan.output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner
            .decode_region_into(&mut self.pool, &mut out, stride, fmt, plan.source_rect())?;
        wrap_cpu_staged_cuda_surface(&out, dims, fmt, session)
    }

    /// Decode a scaled image on CPU and upload it into a CUDA buffer using an
    /// existing backend session.
    pub fn decode_scaled_to_cpu_staged_cuda_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        let dims = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::Scaled { scale },
        )?
        .output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner
            .decode_scaled_into(&mut self.pool, &mut out, stride, fmt, scale)?;
        wrap_cpu_staged_cuda_surface(&out, dims, fmt, session)
    }

    /// Decode a scaled region on CPU and upload it into a CUDA buffer using an
    /// existing backend session.
    pub fn decode_region_scaled_to_cpu_staged_cuda_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        let plan = DeviceDecodePlan::for_image(
            self.inner.info().dimensions,
            DeviceDecodeRequest::RegionScaled { roi, scale },
        )?;
        let dims = plan.output_dims();
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner.decode_region_scaled_into(
            &mut self.pool,
            &mut out,
            stride,
            fmt,
            plan.source_rect(),
            scale,
        )?;
        wrap_cpu_staged_cuda_surface(&out, dims, fmt, session)
    }
}

fn allocate_cpu_surface(dims: (u32, u32), fmt: PixelFormat) -> Result<(Vec<u8>, usize), Error> {
    let (stride, len) = checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "j2k CUDA CPU-staged surface",
    )?;
    Ok((
        try_vec_filled(len, 0u8, "j2k CUDA CPU-staged surface")?,
        stride,
    ))
}

#[cfg(not(feature = "cuda-runtime"))]
fn decode_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
fn decode_to_cuda_resident_surface_with_profile_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
fn decode_region_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _roi: Rect,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
fn decode_scaled_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _scale: Downscale,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
fn decode_region_scaled_to_cuda_resident_surface_impl(
    _decoder: &mut J2kDecoder<'_>,
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _roi: Rect,
    _scale: Downscale,
) -> Result<Surface, Error> {
    Err(Error::CudaUnavailable)
}

#[cfg(not(feature = "cuda-runtime"))]
fn decode_batch_to_cuda_resident_surface_with_profile_control(
    _inputs: &[&[u8]],
    _session: &mut CudaSession,
    _fmt: PixelFormat,
    _collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    Err(Error::CudaUnavailable)
}

#[doc(hidden)]
impl ImageCodec for J2kDecoder<'_> {
    type Error = Error;
    type Warning = J2kDecodeWarning;
    type Pool = crate::J2kScratchPool;
}

impl<'a> CpuBackedImageDecode<'a> for J2kDecoder<'a> {
    type Cpu = CpuDecoder<'a>;
    type View = J2kView<'a>;

    fn inspect_cpu(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?)
    }

    fn parse_cpu(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(J2kView::parse(input)?)
    }

    fn from_cpu_view(view: Self::View) -> Result<Self, Self::Error> {
        let bytes = view.bytes();
        Ok(Self {
            bytes,
            inner: CpuDecoder::from_view(view)?,
            pool: CpuJ2kScratchPool::new(),
        })
    }

    fn cpu_decoder_mut(&mut self) -> &mut Self::Cpu {
        &mut self.inner
    }

    fn map_cpu_outcome(
        outcome: DecodeOutcome<<Self::Cpu as ImageCodec>::Warning>,
    ) -> DecodeOutcome<Self::Warning> {
        outcome
    }
}

#[doc(hidden)]
impl<'a> ImageDecodeDevice<'a> for J2kDecoder<'a> {
    type DeviceSurface = Surface;
}

#[doc(hidden)]
impl<'a> ImageDecodeSubmit<'a> for J2kDecoder<'a> {
    type Session = CudaSession;
    type DeviceSurface = Surface;
    type SubmittedSurface = ReadySubmission<Surface, Error>;

    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            self.decode_to_surface_impl(session, fmt, backend)
        }))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            self.decode_region_to_surface_impl(session, fmt, roi, backend)
        }))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            self.decode_scaled_to_surface_impl(session, fmt, scale, backend)
        }))
    }

    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        Ok(submit_ready_device(session, |session| {
            self.decode_region_scaled_to_surface_impl(session, fmt, roi, scale, backend)
        }))
    }
}
