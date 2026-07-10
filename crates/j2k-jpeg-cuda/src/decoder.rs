// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{
    submit_ready_device, BackendRequest, CpuBackedImageDecode, DecodeOutcome, Downscale,
    ImageCodec, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect,
};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::adapter::decoder_bytes;
use j2k_jpeg::{DecodeRequest, Decoder as CpuDecoder, JpegView, Warning as CpuWarning};

#[cfg(feature = "cuda-runtime")]
use crate::owned_decode::decode_owned_cuda_rgb8;
use crate::owned_decode::unsupported_owned_cuda_output_format;
use crate::runtime::{validate_surface_request, wrap_surface};
use crate::{CudaSession, Error, Surface};

/// JPEG decoder that can return host or CUDA-resident surfaces.
pub struct Decoder<'a> {
    inner: CpuDecoder<'a>,
}

impl<'a> Decoder<'a> {
    /// Parse a JPEG byte slice into a CUDA-capable decoder wrapper.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
            inner: CpuDecoder::new(input)?,
        })
    }

    fn decode_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        j2k_profile::emit_gpu_route_surface_profile(
            ("jpeg", "cuda"),
            (
                "full",
                format_args!("{backend:?}"),
                format_args!("{fmt:?}"),
                "begin",
            ),
            self.inner.info().dimensions,
            [],
        );
        if backend == BackendRequest::Cuda {
            if fmt == PixelFormat::Rgb8 {
                return self.decode_cuda_rgb8(session);
            }
            return Err(unsupported_owned_cuda_output_format());
        }
        let (bytes, _outcome) = self.inner.decode_request(DecodeRequest::full(fmt))?;
        j2k_profile::emit_gpu_route_decision_profile(
            ("jpeg", "cuda"),
            (
                "full",
                format_args!("{backend:?}"),
                format_args!("{fmt:?}"),
                "cpu_decode_then_wrap",
            ),
            [],
        );
        wrap_surface(bytes, self.inner.info().dimensions, fmt, backend, session)
    }

    #[cfg(feature = "cuda-runtime")]
    fn decode_cuda_rgb8(&mut self, session: &mut CudaSession) -> Result<Surface, Error> {
        let dimensions = self.inner.info().dimensions;
        let surface = decode_owned_cuda_rgb8(decoder_bytes(&self.inner), dimensions, session)?;
        j2k_profile::emit_gpu_route_surface_profile(
            ("jpeg", "cuda"),
            ("full", "Cuda", "Rgb8", "owned_cuda"),
            dimensions,
            [],
        );
        Ok(surface)
    }

    #[cfg(not(feature = "cuda-runtime"))]
    #[expect(
        clippy::unused_self,
        reason = "feature-disabled shim preserves the runtime-enabled method signature"
    )]
    fn decode_cuda_rgb8(&mut self, _session: &mut CudaSession) -> Result<Surface, Error> {
        j2k_profile::emit_gpu_route_decision_profile(
            ("jpeg", "cuda"),
            ("full", "Cuda", "Rgb8", "owned_cuda_unavailable"),
            [j2k_profile::ProfileField::label(
                "reason",
                "cuda_runtime_feature_disabled",
            )],
        );
        Err(Error::CudaUnavailable)
    }

    fn decode_region_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if backend == BackendRequest::Cuda {
            return Err(Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG owned decode does not support region output",
            });
        }
        let (bytes, outcome) = self
            .inner
            .decode_request(DecodeRequest::region(fmt, roi.into()))?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
    }

    fn decode_scaled_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
        if backend == BackendRequest::Cuda {
            return Err(Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG owned decode does not support scaled output",
            });
        }
        let (bytes, outcome) = self
            .inner
            .decode_request(DecodeRequest::scaled(fmt, scale))?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
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
        if backend == BackendRequest::Cuda {
            return Err(Error::UnsupportedCudaRequest {
                reason: "J2K CUDA JPEG owned decode does not support scaled region output",
            });
        }
        let (bytes, outcome) =
            self.inner
                .decode_request(DecodeRequest::region_scaled(fmt, roi.into(), scale))?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
    }
}

#[doc(hidden)]
impl ImageCodec for Decoder<'_> {
    type Error = Error;
    type Warning = CpuWarning;
    type Pool = crate::ScratchPool;
}

impl<'a> CpuBackedImageDecode<'a> for Decoder<'a> {
    type Cpu = CpuDecoder<'a>;
    type View = JpegView<'a>;

    fn inspect_cpu(input: &'a [u8]) -> Result<j2k_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?.to_core_info())
    }

    fn parse_cpu(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(JpegView::parse(input)?)
    }

    fn from_cpu_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
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
impl<'a> ImageDecodeDevice<'a> for Decoder<'a> {
    type DeviceSurface = Surface;
}

#[doc(hidden)]
impl<'a> ImageDecodeSubmit<'a> for Decoder<'a> {
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
