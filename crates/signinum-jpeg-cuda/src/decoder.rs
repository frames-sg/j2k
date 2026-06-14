// SPDX-License-Identifier: Apache-2.0

use signinum_core::{
    submit_ready_device, BackendRequest, DecodeOutcome, Downscale, ImageCodec, ImageDecode,
    ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect,
};
#[cfg(feature = "cuda-runtime")]
use signinum_jpeg::adapter::decoder_bytes;
use signinum_jpeg::{
    Decoder as CpuDecoder, JpegView, ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

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
        if signinum_profile::gpu_route_profile_enabled() {
            let request_s = format!("{backend:?}");
            let fmt_s = format!("{fmt:?}");
            let width_s = self.inner.info().dimensions.0.to_string();
            let height_s = self.inner.info().dimensions.1.to_string();
            signinum_profile::emit_gpu_route_profile(
                "jpeg",
                "cuda",
                &[
                    ("op", "full"),
                    ("request", request_s.as_str()),
                    ("fmt", fmt_s.as_str()),
                    ("width", width_s.as_str()),
                    ("height", height_s.as_str()),
                    ("decision", "begin"),
                ],
            );
        }
        if backend == BackendRequest::Cuda {
            if fmt == PixelFormat::Rgb8 {
                return self.decode_cuda_rgb8(session);
            }
            return Err(unsupported_owned_cuda_output_format());
        }
        let (bytes, _outcome) = self.inner.decode(fmt)?;
        if signinum_profile::gpu_route_profile_enabled() {
            let request_s = format!("{backend:?}");
            let fmt_s = format!("{fmt:?}");
            signinum_profile::emit_gpu_route_profile(
                "jpeg",
                "cuda",
                &[
                    ("op", "full"),
                    ("request", request_s.as_str()),
                    ("fmt", fmt_s.as_str()),
                    ("decision", "cpu_decode_then_wrap"),
                ],
            );
        }
        wrap_surface(bytes, self.inner.info().dimensions, fmt, backend, session)
    }

    #[cfg(feature = "cuda-runtime")]
    fn decode_cuda_rgb8(&mut self, session: &mut CudaSession) -> Result<Surface, Error> {
        let dimensions = self.inner.info().dimensions;
        let surface = decode_owned_cuda_rgb8(decoder_bytes(&self.inner), dimensions, session)?;
        if signinum_profile::gpu_route_profile_enabled() {
            let width_s = dimensions.0.to_string();
            let height_s = dimensions.1.to_string();
            signinum_profile::emit_gpu_route_profile(
                "jpeg",
                "cuda",
                &[
                    ("op", "full"),
                    ("request", "Cuda"),
                    ("fmt", "Rgb8"),
                    ("width", width_s.as_str()),
                    ("height", height_s.as_str()),
                    ("decision", "owned_cuda"),
                ],
            );
        }
        Ok(surface)
    }

    #[cfg(not(feature = "cuda-runtime"))]
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn decode_cuda_rgb8(&mut self, _session: &mut CudaSession) -> Result<Surface, Error> {
        if signinum_profile::gpu_route_profile_enabled() {
            signinum_profile::emit_gpu_route_profile(
                "jpeg",
                "cuda",
                &[
                    ("op", "full"),
                    ("request", "Cuda"),
                    ("fmt", "Rgb8"),
                    ("decision", "owned_cuda_unavailable"),
                    ("reason", "cuda_runtime_feature_disabled"),
                ],
            );
        }
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
                reason: "Signinum CUDA JPEG owned decode does not support region output",
            });
        }
        let (bytes, outcome) = self.inner.decode_region(fmt, roi.into())?;
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
                reason: "Signinum CUDA JPEG owned decode does not support scaled output",
            });
        }
        let (bytes, outcome) = self.inner.decode_scaled(fmt, scale)?;
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
                reason: "Signinum CUDA JPEG owned decode does not support scaled region output",
            });
        }
        let (bytes, outcome) = self.inner.decode_region_scaled(fmt, roi.into(), scale)?;
        wrap_surface(
            bytes,
            (outcome.decoded.w, outcome.decoded.h),
            fmt,
            backend,
            session,
        )
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
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
        })
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
