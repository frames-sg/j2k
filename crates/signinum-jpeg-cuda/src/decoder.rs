// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "cuda-runtime")]
use signinum_core::BackendKind;
use signinum_core::{
    BackendRequest, DecodeOutcome, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect,
};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::CudaError;
#[cfg(feature = "cuda-runtime")]
use signinum_jpeg::adapter::decoder_bytes;
use signinum_jpeg::{
    Decoder as CpuDecoder, JpegView, ScratchPool as CpuScratchPool, Warning as CpuWarning,
};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::runtime::{validate_surface_request, wrap_surface};
#[cfg(feature = "cuda-runtime")]
use crate::surface::{CudaSurfaceStats, Storage};
use crate::{profile, CudaSession, Error, Surface};

pub struct Decoder<'a> {
    inner: CpuDecoder<'a>,
}

impl<'a> Decoder<'a> {
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
        if profile::gpu_route_profile_enabled() {
            let request_s = format!("{backend:?}");
            let fmt_s = format!("{fmt:?}");
            let width_s = self.inner.info().dimensions.0.to_string();
            let height_s = self.inner.info().dimensions.1.to_string();
            profile::emit_gpu_route_profile(
                "jpeg",
                "gpu_route",
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
        if backend == BackendRequest::Cuda && fmt == PixelFormat::Rgb8 {
            if let Some(surface) = self.try_decode_cuda_rgb8(session)? {
                return Ok(surface);
            }
        }
        let (bytes, _outcome) = self.inner.decode(fmt)?;
        if profile::gpu_route_profile_enabled() {
            let request_s = format!("{backend:?}");
            let fmt_s = format!("{fmt:?}");
            profile::emit_gpu_route_profile(
                "jpeg",
                "gpu_route",
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
    fn try_decode_cuda_rgb8(
        &mut self,
        session: &mut CudaSession,
    ) -> Result<Option<Surface>, Error> {
        let dimensions = self.inner.info().dimensions;
        let bytes = decoder_bytes(&self.inner);
        let context = session.cuda_context()?;
        match context.decode_jpeg_rgb8_with_nvjpeg(bytes, dimensions) {
            Ok(output) => {
                let pitch_bytes = dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
                let (buffer, stats) = output.into_parts();
                if profile::gpu_route_profile_enabled() {
                    let width_s = dimensions.0.to_string();
                    let height_s = dimensions.1.to_string();
                    let kernel_dispatches_s = stats.kernel_dispatches().to_string();
                    let decode_dispatches_s = stats.decode_kernel_dispatches().to_string();
                    let hardware_decode_s = stats.used_hardware_decode().to_string();
                    profile::emit_gpu_route_profile(
                        "jpeg",
                        "gpu_route",
                        "cuda",
                        &[
                            ("op", "full"),
                            ("request", "Cuda"),
                            ("fmt", "Rgb8"),
                            ("width", width_s.as_str()),
                            ("height", height_s.as_str()),
                            ("decision", "nvjpeg"),
                            ("kernel_dispatches", kernel_dispatches_s.as_str()),
                            ("decode_kernel_dispatches", decode_dispatches_s.as_str()),
                            ("hardware_decode", hardware_decode_s.as_str()),
                        ],
                    );
                }
                Ok(Some(Surface {
                    backend: BackendKind::Cuda,
                    dimensions,
                    fmt: PixelFormat::Rgb8,
                    pitch_bytes,
                    stats: CudaSurfaceStats {
                        kernel_dispatches: stats.kernel_dispatches(),
                        copy_kernel_dispatches: stats.copy_kernel_dispatches(),
                        decode_kernel_dispatches: stats.decode_kernel_dispatches(),
                        hardware_decode: stats.used_hardware_decode(),
                    },
                    storage: Storage::Cuda(buffer),
                }))
            }
            Err(
                CudaError::NvjpegUnavailable { .. }
                | CudaError::Nvjpeg { .. }
                | CudaError::NvjpegDimensions { .. },
            ) => {
                if profile::gpu_route_profile_enabled() {
                    profile::emit_gpu_route_profile(
                        "jpeg",
                        "gpu_route",
                        "cuda",
                        &[
                            ("op", "full"),
                            ("request", "Cuda"),
                            ("fmt", "Rgb8"),
                            ("decision", "nvjpeg_fallback"),
                            ("reason", "nvjpeg_unavailable_or_rejected"),
                        ],
                    );
                }
                Ok(None)
            }
            Err(error) => Err(cuda_error(error)),
        }
    }

    #[cfg(not(feature = "cuda-runtime"))]
    #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
    fn try_decode_cuda_rgb8(
        &mut self,
        _session: &mut CudaSession,
    ) -> Result<Option<Surface>, Error> {
        if profile::gpu_route_profile_enabled() {
            profile::emit_gpu_route_profile(
                "jpeg",
                "gpu_route",
                "cuda",
                &[
                    ("op", "full"),
                    ("request", "Cuda"),
                    ("fmt", "Rgb8"),
                    ("decision", "nvjpeg_fallback"),
                    ("reason", "cuda_runtime_feature_disabled"),
                ],
            );
        }
        Ok(None)
    }

    fn decode_region_to_surface_impl(
        &mut self,
        session: &mut CudaSession,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Surface, Error> {
        validate_surface_request(backend)?;
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
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_to_surface_impl(session, fmt, backend),
        ))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_region_to_surface_impl(session, fmt, roi, backend),
        ))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_scaled_to_surface_impl(session, fmt, scale, backend),
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
        validate_surface_request(backend)?;
        session.record_submit();
        Ok(ReadySubmission::from_result(
            self.decode_region_scaled_to_surface_impl(session, fmt, roi, scale, backend),
        ))
    }
}
