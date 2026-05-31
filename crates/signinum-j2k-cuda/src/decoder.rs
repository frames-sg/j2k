// SPDX-License-Identifier: Apache-2.0

use core::convert::Infallible;

#[cfg(feature = "cuda-runtime")]
use signinum_core::BackendKind;
use signinum_core::{
    BackendRequest, DecodeOutcome, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect,
};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{
    CudaDeviceBuffer, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables, CudaJ2kIdwtJob, CudaJ2kInverseMctJob,
    CudaJ2kRect, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb8Job,
};
use signinum_j2k::{
    adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
    J2kDecoder as CpuDecoder, J2kError, J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_native::{
    ht_uvlc_table0, ht_uvlc_table1, ht_vlc_table0, ht_vlc_table1, J2kDirectBandId,
};
use signinum_j2k_native::{
    DecodeSettings, DecoderContext as NativeDecoderContext, Image as NativeImage,
};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::runtime::{validate_surface_request, wrap_cpu_staged_cuda_surface, wrap_surface};
#[cfg(feature = "cuda-runtime")]
use crate::surface::Storage;
use crate::{
    profile, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport,
    CudaSession, Error, Surface,
};
#[cfg(feature = "cuda-runtime")]
use crate::{CudaHtj2kStoreStep, CudaHtj2kTransform, CudaSurfaceStats, SurfaceResidency};

#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_KERNELS_NOT_READY: &str =
    "strict CUDA HTJ2K resident codestream decode kernels are not available in this build";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED: &str =
    "strict CUDA HTJ2K resident decode currently accepts Gray8, Gray16, Rgb8, Rgba8, Rgb16, and Rgba16 output";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_STORE_UNSUPPORTED: &str =
    "strict CUDA HTJ2K resident decode requires a single grayscale store step";

/// CUDA-facing JPEG 2000 decoder wrapper.
pub struct J2kDecoder<'a> {
    inner: CpuDecoder<'a>,
    pool: CpuJ2kScratchPool,
}

impl<'a> J2kDecoder<'a> {
    /// Create a CUDA-facing decoder from compressed bytes.
    pub fn new(input: &'a [u8]) -> Result<Self, Error> {
        Ok(Self {
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
        if profile::gpu_route_profile_enabled() {
            let request_s = format!("{backend:?}");
            let fmt_s = format!("{fmt:?}");
            let width_s = dims.0.to_string();
            let height_s = dims.1.to_string();
            profile::emit_gpu_route_profile(
                "j2k",
                "gpu_route",
                "cuda",
                &[
                    ("op", "full"),
                    ("request", request_s.as_str()),
                    ("fmt", fmt_s.as_str()),
                    ("width", width_s.as_str()),
                    ("height", height_s.as_str()),
                    ("decision", "cpu_decode_then_wrap"),
                ],
            );
        }
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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
    pub fn decode_to_device_with_session_and_profile(
        &mut self,
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
        decode_to_cuda_resident_surface_with_profile_impl(self, session, fmt)
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

    /// Build a flat CUDA HTJ2K grayscale decode plan and return stage timings.
    pub fn build_cuda_htj2k_grayscale_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.inner.bytes(), &DecodeSettings::default())
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_with_context(&mut native_context)
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, fmt, (0, 0))?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }

    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_grayscale_region_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.inner.bytes(), &DecodeSettings::default())
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_region_with_context(
                &mut native_context,
                (roi.x, roi.y, roi.w, roi.h),
            )
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
            &native_plan,
            fmt,
            (roi.x, roi.y),
            (roi.w, roi.h),
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }

    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_grayscale_scaled_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
        output_dimensions: (u32, u32),
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(
            self.inner.bytes(),
            &DecodeSettings {
                target_resolution: Some(output_dimensions),
                ..DecodeSettings::default()
            },
        )
        .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_with_context(&mut native_context)
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan(&native_plan, fmt, (0, 0))?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }

    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
        &mut self,
        fmt: PixelFormat,
        scaled_roi: Rect,
        output_dimensions: (u32, u32),
    ) -> Result<(CudaHtj2kDecodePlan, CudaHtj2kProfileReport), Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(
            self.inner.bytes(),
            &DecodeSettings {
                target_resolution: Some(output_dimensions),
                ..DecodeSettings::default()
            },
        )
        .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_grayscale_plan_region_with_context(
                &mut native_context,
                (scaled_roi.x, scaled_roi.y, scaled_roi.w, scaled_roi.h),
            )
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let cuda_plan = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
            &native_plan,
            fmt,
            (scaled_roi.x, scaled_roi.y),
            (scaled_roi.w, scaled_roi.h),
        )?;
        let flatten_us = profile::elapsed_us(flatten_start);

        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count: cuda_plan.code_blocks().len(),
            payload_bytes: cuda_plan.payload().len(),
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");
        Ok((cuda_plan, report))
    }

    /// Build flat CUDA HTJ2K RGB component plans and return stage timings.
    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_color_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.inner.bytes(), &DecodeSettings::default())
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let mut payload = Vec::new();
        let mut components = Vec::with_capacity(native_plan.component_plans.len());
        for component_plan in &native_plan.component_plans {
            let mut component =
                CudaHtj2kDecodePlan::from_grayscale_direct_plan(component_plan, fmt, (0, 0))?;
            component.append_payload_to_shared(&mut payload)?;
            components.push(component);
        }
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            dimensions: native_plan.dimensions,
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_color_scaled_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
        output_dimensions: (u32, u32),
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(
            self.inner.bytes(),
            &DecodeSettings {
                target_resolution: Some(output_dimensions),
                ..DecodeSettings::default()
            },
        )
        .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let mut payload = Vec::new();
        let mut components = Vec::with_capacity(native_plan.component_plans.len());
        for component_plan in &native_plan.component_plans {
            let mut component =
                CudaHtj2kDecodePlan::from_grayscale_direct_plan(component_plan, fmt, (0, 0))?;
            component.append_payload_to_shared(&mut payload)?;
            components.push(component);
        }
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            dimensions: native_plan.dimensions,
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_color_region_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(self.inner.bytes(), &DecodeSettings::default())
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let mut payload = Vec::new();
        let mut components = Vec::with_capacity(native_plan.component_plans.len());
        for component_plan in &native_plan.component_plans {
            let mut component = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
                component_plan,
                fmt,
                (roi.x, roi.y),
                (roi.w, roi.h),
            )?;
            component.append_payload_to_shared(&mut payload)?;
            components.push(component);
        }
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            dimensions: (roi.w, roi.h),
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    #[cfg(feature = "cuda-runtime")]
    fn build_cuda_htj2k_color_region_scaled_plans_with_profile(
        &mut self,
        fmt: PixelFormat,
        scaled_roi: Rect,
        output_dimensions: (u32, u32),
    ) -> Result<CudaHtj2kColorDecodePlans, Error> {
        let total_start = profile::profile_now(true);

        let parse_start = profile::profile_now(true);
        let image = NativeImage::new(
            self.inner.bytes(),
            &DecodeSettings {
                target_resolution: Some(output_dimensions),
                ..DecodeSettings::default()
            },
        )
        .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let parse_us = profile::elapsed_us(parse_start);

        let plan_start = profile::profile_now(true);
        let mut native_context = NativeDecoderContext::default();
        let native_plan = image
            .build_direct_color_plan_with_context(&mut native_context)
            .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
        let plan_us = profile::elapsed_us(plan_start);

        let flatten_start = profile::profile_now(true);
        let mut payload = Vec::new();
        let mut components = Vec::with_capacity(native_plan.component_plans.len());
        for component_plan in &native_plan.component_plans {
            let mut component = CudaHtj2kDecodePlan::from_grayscale_direct_plan_region(
                component_plan,
                fmt,
                (scaled_roi.x, scaled_roi.y),
                (scaled_roi.w, scaled_roi.h),
            )?;
            component.append_payload_to_shared(&mut payload)?;
            components.push(component);
        }
        let flatten_us = profile::elapsed_us(flatten_start);
        let block_count = components
            .iter()
            .map(|plan| plan.code_blocks().len())
            .sum::<usize>();
        let payload_bytes = payload.len();
        let report = CudaHtj2kProfileReport {
            parse_us,
            plan_us,
            flatten_us,
            total_us: profile::elapsed_us(total_start),
            block_count,
            payload_bytes,
            dispatch_count: 0,
            residency: crate::SurfaceResidency::CudaResidentDecode,
            detail: CudaHtj2kDecodeProfileDetail::default(),
            ..CudaHtj2kProfileReport::default()
        };
        report.emit("plan");

        Ok(CudaHtj2kColorDecodePlans {
            dimensions: (scaled_roi.w, scaled_roi.h),
            mct_dimensions: native_plan.dimensions,
            bit_depths: native_plan.bit_depths,
            mct: native_plan.mct,
            transform: CudaHtj2kTransform::from(native_plan.transform),
            payload,
            components,
            report,
        })
    }

    /// Decode a full image on CPU and upload it into a CUDA buffer using an
    /// existing backend session.
    pub fn decode_to_cpu_staged_cuda_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        session: &mut CudaSession,
    ) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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
        let stride = dims.0 as usize * fmt.bytes_per_pixel();
        let mut out = vec![0u8; stride * dims.1 as usize];
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

#[cfg(feature = "cuda-runtime")]
struct CudaCoefficientBand {
    band_id: J2kDirectBandId,
    buffer: CudaDeviceBuffer,
}

#[cfg(feature = "cuda-runtime")]
struct CudaDecodedComponent {
    buffer: CudaDeviceBuffer,
    store: CudaHtj2kStoreStep,
    dispatches: usize,
    decode_dispatches: usize,
    timings: CudaDecodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default)]
struct CudaDecodeStageTimings {
    h2d: u128,
    job_upload: u128,
    ht_cleanup: u128,
    ht_refine: u128,
    dequant: u128,
    ht_dispatch_count: usize,
    idwt: u128,
    dequant_dispatch_count: usize,
    idwt_dispatch_count: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaDecodeStageTimings {
    fn add_to_report(self, report: &mut CudaHtj2kProfileReport) {
        report.h2d_us = report.h2d_us.saturating_add(self.h2d);
        report.detail.job_upload_us = report.detail.job_upload_us.saturating_add(self.job_upload);
        report.ht_cleanup_us = report.ht_cleanup_us.saturating_add(self.ht_cleanup);
        report.ht_refine_us = report.ht_refine_us.saturating_add(self.ht_refine);
        report.dequant_us = report.dequant_us.saturating_add(self.dequant);
        report.idwt_us = report.idwt_us.saturating_add(self.idwt);
        report.detail.ht_dispatch_count = report
            .detail
            .ht_dispatch_count
            .saturating_add(self.ht_dispatch_count);
        report.detail.dequant_dispatch_count = report
            .detail
            .dequant_dispatch_count
            .saturating_add(self.dequant_dispatch_count);
        report.detail.idwt_dispatch_count = report
            .detail
            .idwt_dispatch_count
            .saturating_add(self.idwt_dispatch_count);
    }
}

#[cfg(feature = "cuda-runtime")]
struct CudaHtj2kColorDecodePlans {
    dimensions: (u32, u32),
    mct_dimensions: (u32, u32),
    bit_depths: [u8; 3],
    mct: bool,
    transform: CudaHtj2kTransform,
    payload: Vec<u8>,
    components: Vec<CudaHtj2kDecodePlan>,
    report: CudaHtj2kProfileReport,
}

#[cfg(feature = "cuda-runtime")]
fn decode_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    decode_to_cuda_resident_surface_with_profile_impl(decoder, session, fmt)
        .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_to_cuda_resident_surface_with_profile_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let wall_started = profile::profile_now(true);
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_surface_with_profile(decoder, session, fmt, wall_started)
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_surface_with_profile(decoder, session, fmt, wall_started)
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_region_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let plan = DeviceDecodePlan::for_image(
        decoder.inner.info().dimensions,
        DeviceDecodeRequest::Region { roi },
    )?;
    if plan.is_full_frame() {
        return decode_to_cuda_resident_surface_impl(decoder, session, fmt);
    }

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_region_surface(decoder, session, fmt, plan.source_rect())
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_region_surface(decoder, session, fmt, plan.source_rect())
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_scaled_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scale: Downscale,
) -> Result<Surface, Error> {
    if scale == Downscale::None {
        return decode_to_cuda_resident_surface_impl(decoder, session, fmt);
    }
    let output_dimensions = DeviceDecodePlan::for_image(
        decoder.inner.info().dimensions,
        DeviceDecodeRequest::Scaled { scale },
    )?
    .output_dims();

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_scaled_surface(decoder, session, fmt, output_dimensions)
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_scaled_surface(decoder, session, fmt, output_dimensions)
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_region_scaled_to_cuda_resident_surface_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<Surface, Error> {
    if scale == Downscale::None {
        return decode_region_to_cuda_resident_surface_impl(decoder, session, fmt, roi);
    }
    let source_dimensions = decoder.inner.info().dimensions;
    let scaled_dimensions =
        DeviceDecodePlan::for_image(source_dimensions, DeviceDecodeRequest::Scaled { scale })?
            .output_dims();
    let plan = DeviceDecodePlan::for_image(
        source_dimensions,
        DeviceDecodeRequest::RegionScaled { roi, scale },
    )?;
    let scaled_roi = plan.output_rect();

    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_region_scaled_surface(
                decoder,
                session,
                fmt,
                scaled_roi,
                scaled_dimensions,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_region_scaled_surface(
                decoder,
                session,
                fmt,
                scaled_roi,
                scaled_dimensions,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_profile(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    wall_started: Option<profile::ProfileInstant>,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_plan_with_profile(fmt)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_region_plan_with_profile(fmt, roi)?;
    decode_grayscale_cuda_resident_surface_with_plan(session, fmt, &plan, &mut report)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_scaled_plan_with_profile(fmt, output_dimensions)?;
    decode_grayscale_cuda_resident_surface_with_plan(session, fmt, &plan, &mut report)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_grayscale_cuda_resident_surface_with_plan(session, fmt, &plan, &mut report)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_plan(
    session: &mut CudaSession,
    fmt: PixelFormat,
    plan: &CudaHtj2kDecodePlan,
    report: &mut CudaHtj2kProfileReport,
) -> Result<Surface, Error> {
    decode_grayscale_cuda_resident_surface_with_plan_profile(session, fmt, plan, report, None)
        .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_plan_profile(
    session: &mut CudaSession,
    fmt: PixelFormat,
    plan: &CudaHtj2kDecodePlan,
    report: &mut CudaHtj2kProfileReport,
    wall_started: Option<profile::ProfileInstant>,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let context = session.cuda_context()?;
    let tables = CudaHtj2kDecodeTables {
        vlc_table0: ht_vlc_table0(),
        vlc_table1: ht_vlc_table1(),
        uvlc_table0: ht_uvlc_table0(),
        uvlc_table1: ht_uvlc_table1(),
    };
    let table_upload_start = profile::profile_now(true);
    let table_resources = context
        .upload_htj2k_decode_table_resources(tables)
        .map_err(cuda_error)?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    report.h2d_us = report.h2d_us.saturating_add(table_upload_us);
    report.detail.table_upload_us = report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let component = decode_cuda_component_plan(&context, plan, &table_resources)?;
    let input_width = component
        .store
        .input_rect
        .x1
        .saturating_sub(component.store.input_rect.x0);
    let (store_output, store_us) = context
        .time_default_stream_named_us("signinum.htj2k.decode.store.gray", || match fmt {
            PixelFormat::Gray8 => context.j2k_store_gray8_device(
                &component.buffer,
                CudaJ2kStoreGray8Job {
                    input_width,
                    source_x: component.store.source_x,
                    source_y: component.store.source_y,
                    copy_width: component.store.copy_width,
                    copy_height: component.store.copy_height,
                    output_width: component.store.output_width,
                    output_height: component.store.output_height,
                    output_x: component.store.output_x,
                    output_y: component.store.output_y,
                    addend: component.store.addend,
                    bit_depth: u32::from(plan.bit_depth()),
                },
            ),
            PixelFormat::Gray16 => context.j2k_store_gray16_device(
                &component.buffer,
                CudaJ2kStoreGray16Job {
                    input_width,
                    source_x: component.store.source_x,
                    source_y: component.store.source_y,
                    copy_width: component.store.copy_width,
                    copy_height: component.store.copy_height,
                    output_width: component.store.output_width,
                    output_height: component.store.output_height,
                    output_x: component.store.output_x,
                    output_y: component.store.output_y,
                    addend: component.store.addend,
                    bit_depth: u32::from(plan.bit_depth()),
                },
            ),
            _ => {
                unreachable!("validated grayscale CUDA output format");
            }
        })
        .map_err(cuda_error)?;
    let (surface_buffer, store_stats) = store_output.into_parts();
    let dispatches = component
        .dispatches
        .saturating_add(store_stats.kernel_dispatches());
    let decode_dispatches = component
        .decode_dispatches
        .saturating_add(store_stats.decode_kernel_dispatches());
    report.dispatch_count = dispatches;
    component.timings.add_to_report(report);
    report.store_us = report.store_us.saturating_add(store_us);
    report.detail.store_dispatch_count = report
        .detail
        .store_dispatch_count
        .saturating_add(store_stats.kernel_dispatches());
    report.detail.wall_total_us = profile::elapsed_us(wall_started);
    profile::finalize_decode_total_us(report);
    report.emit("decode");

    let dimensions = (component.store.output_width, component.store.output_height);
    let surface = Surface {
        backend: BackendKind::Cuda,
        residency: SurfaceResidency::CudaResidentDecode,
        dimensions,
        fmt,
        pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            total: dispatches,
            copy: 0,
            decode: decode_dispatches,
        },
        storage: Storage::Cuda(surface_buffer),
    };
    Ok((surface, report.clone()))
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_surface_with_profile(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    wall_started: Option<profile::ProfileInstant>,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let color = decoder.build_cuda_htj2k_color_plans_with_profile(fmt)?;
    decode_color_cuda_resident_surface_with_plans_profile(session, fmt, color, wall_started)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let color = decoder.build_cuda_htj2k_color_scaled_plans_with_profile(fmt, output_dimensions)?;
    decode_color_cuda_resident_surface_with_plans(session, fmt, color)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let color = decoder.build_cuda_htj2k_color_region_plans_with_profile(fmt, roi)?;
    decode_color_cuda_resident_surface_with_plans(session, fmt, color)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let color = decoder.build_cuda_htj2k_color_region_scaled_plans_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_color_cuda_resident_surface_with_plans(session, fmt, color)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_surface_with_plans(
    session: &mut CudaSession,
    fmt: PixelFormat,
    color: CudaHtj2kColorDecodePlans,
) -> Result<Surface, Error> {
    decode_color_cuda_resident_surface_with_plans_profile(session, fmt, color, None)
        .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_surface_with_plans_profile(
    session: &mut CudaSession,
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    wall_started: Option<profile::ProfileInstant>,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    if color.components.len() != 3 {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    let context = session.cuda_context()?;
    let tables = CudaHtj2kDecodeTables {
        vlc_table0: ht_vlc_table0(),
        vlc_table1: ht_vlc_table1(),
        uvlc_table0: ht_uvlc_table0(),
        uvlc_table1: ht_uvlc_table1(),
    };
    let table_upload_start = profile::profile_now(true);
    let table_resources = context
        .upload_htj2k_decode_table_resources(tables)
        .map_err(cuda_error)?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    color.report.h2d_us = color.report.h2d_us.saturating_add(table_upload_us);
    color.report.detail.table_upload_us = color
        .report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let payload_upload_start = profile::profile_now(true);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(&color.payload, &table_resources)
        .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    color.report.h2d_us = color.report.h2d_us.saturating_add(payload_upload_us);
    color.report.detail.payload_upload_us = color
        .report
        .detail
        .payload_upload_us
        .saturating_add(payload_upload_us);
    let mut decoded_components = Vec::with_capacity(3);
    for plan in &color.components {
        decoded_components.push(decode_cuda_component_plan_with_resources(
            &context,
            plan,
            &decode_resources,
        )?);
    }
    let [component0, component1, component2] = decoded_components.as_slice() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    };
    validate_color_stores(
        [&component0.store, &component1.store, &component2.store],
        color.dimensions,
    )?;

    let mut dispatches = decoded_components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>();
    let mut decode_dispatches = decoded_components
        .iter()
        .map(|component| component.decode_dispatches)
        .sum::<usize>();
    for component in &decoded_components {
        component.timings.add_to_report(&mut color.report);
    }
    let addends = if color.mct {
        let mct_len = u32::try_from(checked_area(
            color.mct_dimensions.0,
            color.mct_dimensions.1,
        )?)
        .map_err(|_| Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
        let stats = context
            .time_default_stream_named_us("signinum.htj2k.decode.mct", || {
                context.j2k_inverse_mct_device(
                    &component0.buffer,
                    &component1.buffer,
                    &component2.buffer,
                    CudaJ2kInverseMctJob {
                        len: mct_len,
                        irreversible97: u32::from(
                            color.transform == CudaHtj2kTransform::Irreversible97,
                        ),
                        addend0: bit_depth_addend(color.bit_depths[0]),
                        addend1: bit_depth_addend(color.bit_depths[1]),
                        addend2: bit_depth_addend(color.bit_depths[2]),
                    },
                )
            })
            .map_err(cuda_error)?;
        let (stats, mct_us) = stats;
        dispatches = dispatches.saturating_add(stats.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(stats.decode_kernel_dispatches());
        color.report.mct_us = color.report.mct_us.saturating_add(mct_us);
        color.report.detail.mct_dispatch_count = color
            .report
            .detail
            .mct_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        [0.0, 0.0, 0.0]
    } else {
        [
            component0.store.addend,
            component1.store.addend,
            component2.store.addend,
        ]
    };

    let input_width0 = component0
        .store
        .input_rect
        .x1
        .saturating_sub(component0.store.input_rect.x0);
    let input_width1 = component1
        .store
        .input_rect
        .x1
        .saturating_sub(component1.store.input_rect.x0);
    let input_width2 = component2
        .store
        .input_rect
        .x1
        .saturating_sub(component2.store.input_rect.x0);
    let (store_output, store_us) = context
        .time_default_stream_named_us("signinum.htj2k.decode.store.color", || match fmt {
            PixelFormat::Rgb8 | PixelFormat::Rgba8 => context.j2k_store_rgb8_device(
                &component0.buffer,
                &component1.buffer,
                &component2.buffer,
                CudaJ2kStoreRgb8Job {
                    input_width0,
                    input_width1,
                    input_width2,
                    source_x0: component0.store.source_x,
                    source_y0: component0.store.source_y,
                    source_x1: component1.store.source_x,
                    source_y1: component1.store.source_y,
                    source_x2: component2.store.source_x,
                    source_y2: component2.store.source_y,
                    copy_width: component0.store.copy_width,
                    copy_height: component0.store.copy_height,
                    output_width: component0.store.output_width,
                    output_height: component0.store.output_height,
                    output_x: component0.store.output_x,
                    output_y: component0.store.output_y,
                    addend0: addends[0],
                    addend1: addends[1],
                    addend2: addends[2],
                    bit_depth0: u32::from(color.bit_depths[0]),
                    bit_depth1: u32::from(color.bit_depths[1]),
                    bit_depth2: u32::from(color.bit_depths[2]),
                    rgba: u32::from(fmt == PixelFormat::Rgba8),
                },
            ),
            PixelFormat::Rgb16 | PixelFormat::Rgba16 => context.j2k_store_rgb16_device(
                &component0.buffer,
                &component1.buffer,
                &component2.buffer,
                CudaJ2kStoreRgb16Job {
                    input_width0,
                    input_width1,
                    input_width2,
                    source_x0: component0.store.source_x,
                    source_y0: component0.store.source_y,
                    source_x1: component1.store.source_x,
                    source_y1: component1.store.source_y,
                    source_x2: component2.store.source_x,
                    source_y2: component2.store.source_y,
                    copy_width: component0.store.copy_width,
                    copy_height: component0.store.copy_height,
                    output_width: component0.store.output_width,
                    output_height: component0.store.output_height,
                    output_x: component0.store.output_x,
                    output_y: component0.store.output_y,
                    addend0: addends[0],
                    addend1: addends[1],
                    addend2: addends[2],
                    bit_depth0: u32::from(color.bit_depths[0]),
                    bit_depth1: u32::from(color.bit_depths[1]),
                    bit_depth2: u32::from(color.bit_depths[2]),
                    rgba: u32::from(fmt == PixelFormat::Rgba16),
                },
            ),
            _ => {
                unreachable!("validated color CUDA output format");
            }
        })
        .map_err(cuda_error)?;
    let (surface_buffer, store_stats) = store_output.into_parts();
    dispatches = dispatches.saturating_add(store_stats.kernel_dispatches());
    decode_dispatches = decode_dispatches.saturating_add(store_stats.decode_kernel_dispatches());
    color.report.dispatch_count = dispatches;
    color.report.store_us = color.report.store_us.saturating_add(store_us);
    color.report.detail.store_dispatch_count = color
        .report
        .detail
        .store_dispatch_count
        .saturating_add(store_stats.kernel_dispatches());
    color.report.detail.wall_total_us = profile::elapsed_us(wall_started);
    profile::finalize_decode_total_us(&mut color.report);
    color.report.emit("decode");

    let surface = Surface {
        backend: BackendKind::Cuda,
        residency: SurfaceResidency::CudaResidentDecode,
        dimensions: color.dimensions,
        fmt,
        pitch_bytes: color.dimensions.0 as usize * fmt.bytes_per_pixel(),
        stats: CudaSurfaceStats {
            total: dispatches,
            copy: 0,
            decode: decode_dispatches,
        },
        storage: Storage::Cuda(surface_buffer),
    };
    Ok((surface, color.report))
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

#[cfg(feature = "cuda-runtime")]
fn decode_cuda_component_plan(
    context: &signinum_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    tables: &CudaHtj2kDecodeTableResources,
) -> Result<CudaDecodedComponent, Error> {
    let resource_upload_start = profile::profile_now(true);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(plan.payload(), tables)
        .map_err(cuda_error)?;
    let resource_upload_us = profile::elapsed_us(resource_upload_start);
    let mut component =
        decode_cuda_component_plan_with_resources(context, plan, &decode_resources)?;
    component.timings.h2d = component.timings.h2d.saturating_add(resource_upload_us);
    component.timings.job_upload = component
        .timings
        .job_upload
        .saturating_add(resource_upload_us);
    Ok(component)
}

#[cfg(any(feature = "cuda-runtime", test))]
fn split_htj2k_subband_decode_dispatches(kernel_dispatches: usize) -> (usize, usize) {
    if kernel_dispatches == 0 {
        return (0, 0);
    }

    let dequant_dispatches = usize::from(kernel_dispatches > 1);
    (
        kernel_dispatches.saturating_sub(dequant_dispatches),
        dequant_dispatches,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_cuda_component_plan_with_resources(
    context: &signinum_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    decode_resources: &CudaHtj2kDecodeResources,
) -> Result<CudaDecodedComponent, Error> {
    let mut bands = Vec::with_capacity(plan.subbands().len() + plan.idwt_steps().len());
    let mut dispatches = 0usize;
    let mut decode_dispatches = 0usize;
    let mut timings = CudaDecodeStageTimings::default();

    for subband in plan.subbands() {
        let start = subband.code_block_start as usize;
        let end = start + subband.code_block_count as usize;
        let jobs = plan.code_blocks()[start..end]
            .iter()
            .map(|block| cuda_code_block_job_from_plan_block(block, subband.width))
            .collect::<Result<Vec<_>, Error>>()?;
        let output_words = checked_area(subband.width, subband.height)?;
        let stage_start = profile::profile_now(true);
        let output = context
            .decode_htj2k_codeblocks_with_resources(decode_resources, &jobs, output_words)
            .map_err(cuda_error)?;
        let stage_timings = output.stage_timings();
        let stage_wall_us = profile::elapsed_us(stage_start);
        let gpu_stage_us = stage_timings
            .ht_cleanup_us
            .saturating_add(stage_timings.dequant_us);
        timings.h2d = timings
            .h2d
            .saturating_add(stage_wall_us.saturating_sub(gpu_stage_us));
        timings.ht_cleanup = timings
            .ht_cleanup
            .saturating_add(stage_timings.ht_cleanup_us);
        timings.ht_refine = timings.ht_refine.saturating_add(stage_timings.ht_refine_us);
        timings.dequant = timings.dequant.saturating_add(stage_timings.dequant_us);
        let execution = output.execution();
        let (ht_dispatches, dequant_dispatches) =
            split_htj2k_subband_decode_dispatches(execution.kernel_dispatches());
        timings.ht_dispatch_count = timings.ht_dispatch_count.saturating_add(ht_dispatches);
        timings.dequant_dispatch_count = timings
            .dequant_dispatch_count
            .saturating_add(dequant_dispatches);
        dispatches = dispatches.saturating_add(execution.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(execution.decode_kernel_dispatches());
        let (buffer, _, _) = output.into_parts();
        bands.push(CudaCoefficientBand {
            band_id: subband.band_id,
            buffer,
        });
    }

    for step in plan.idwt_steps() {
        let ll = find_cuda_band(&bands, step.ll_band_id)?;
        let hl = find_cuda_band(&bands, step.hl_band_id)?;
        let lh = find_cuda_band(&bands, step.lh_band_id)?;
        let hh = find_cuda_band(&bands, step.hh_band_id)?;
        let (output, idwt_us) = context
            .time_default_stream_named_us("signinum.htj2k.decode.idwt", || {
                context.j2k_inverse_dwt_single_device(
                    &ll.buffer,
                    &hl.buffer,
                    &lh.buffer,
                    &hh.buffer,
                    CudaJ2kIdwtJob {
                        rect: cuda_runtime_rect(step.rect),
                        ll_rect: cuda_runtime_rect(step.ll_rect),
                        hl_rect: cuda_runtime_rect(step.hl_rect),
                        lh_rect: cuda_runtime_rect(step.lh_rect),
                        hh_rect: cuda_runtime_rect(step.hh_rect),
                        irreversible97: u32::from(
                            step.transform == CudaHtj2kTransform::Irreversible97,
                        ),
                    },
                )
            })
            .map_err(cuda_error)?;
        timings.idwt = timings.idwt.saturating_add(idwt_us);
        let (buffer, stats) = output.into_parts();
        dispatches = dispatches.saturating_add(stats.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(stats.decode_kernel_dispatches());
        timings.idwt_dispatch_count = timings
            .idwt_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        bands.push(CudaCoefficientBand {
            band_id: step.output_band_id,
            buffer,
        });
    }

    let [store] = plan.store_steps() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_STORE_UNSUPPORTED,
        });
    };
    let input_index = bands
        .iter()
        .position(|band| band.band_id == store.input_band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    let input = bands.swap_remove(input_index);
    Ok(CudaDecodedComponent {
        buffer: input.buffer,
        store: *store,
        dispatches,
        decode_dispatches,
        timings,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_code_block_job_from_plan_block(
    block: &crate::CudaHtj2kCodeBlock,
    subband_width: u32,
) -> Result<CudaHtj2kCodeBlockJob, Error> {
    let output_offset = block
        .output_y
        .checked_mul(subband_width)
        .and_then(|base| base.checked_add(block.output_x))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    Ok(CudaHtj2kCodeBlockJob {
        payload_offset: block.payload_offset,
        width: block.width,
        height: block.height,
        payload_len: block.payload_len,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        missing_bit_planes: block.missing_bit_planes,
        num_bitplanes: block.num_bitplanes,
        number_of_coding_passes: block.number_of_coding_passes,
        output_stride: block.output_stride,
        output_offset,
        dequantization_step: block.dequantization_step,
        stripe_causal: block.stripe_causal != 0,
    })
}

#[cfg(feature = "cuda-runtime")]
fn validate_color_stores(
    stores: [&CudaHtj2kStoreStep; 3],
    dimensions: (u32, u32),
) -> Result<(), Error> {
    let first = stores[0];
    for store in stores {
        let input_width = store.input_rect.x1.saturating_sub(store.input_rect.x0);
        let input_height = store.input_rect.y1.saturating_sub(store.input_rect.y0);
        let source_end_x =
            store
                .source_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        let source_end_y =
            store
                .source_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        if store.output_x != 0
            || store.output_y != 0
            || store.copy_width != dimensions.0
            || store.copy_height != dimensions.1
            || store.output_width != dimensions.0
            || store.output_height != dimensions.1
            || source_end_x > input_width
            || source_end_y > input_height
            || store.source_x != first.source_x
            || store.source_y != first.source_y
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn bit_depth_addend(bit_depth: u8) -> f32 {
    let shift = bit_depth.saturating_sub(1).min(15);
    f32::from(1_u16 << shift)
}

#[cfg(feature = "cuda-runtime")]
fn checked_area(width: u32, height: u32) -> Result<usize, Error> {
    width
        .try_into()
        .ok()
        .and_then(|width: usize| width.checked_mul(height as usize))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
fn find_cuda_band(
    bands: &[CudaCoefficientBand],
    band_id: J2kDirectBandId,
) -> Result<&CudaCoefficientBand, Error> {
    bands
        .iter()
        .find(|band| band.band_id == band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_runtime_rect(rect: crate::CudaHtj2kRect) -> CudaJ2kRect {
    CudaJ2kRect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use super::cuda_code_block_job_from_plan_block;
    use signinum_core::PixelFormat;
    use signinum_j2k_native::{encode_htj2k, EncodeOptions};

    use crate::CudaHtj2kCodeBlock;

    #[test]
    fn cuda_runtime_code_block_job_preserves_plan_output_stride() {
        let block = CudaHtj2kCodeBlock {
            subband_index: 0,
            payload_offset: 13,
            payload_len: 5,
            cleanup_length: 5,
            refinement_length: 0,
            output_x: 3,
            output_y: 2,
            width: 4,
            height: 5,
            output_stride: 99,
            missing_bit_planes: 1,
            number_of_coding_passes: 1,
            num_bitplanes: 8,
            stripe_causal: 0,
            dequantization_step: 1.0,
        };

        let job = cuda_code_block_job_from_plan_block(&block, 64)
            .expect("valid CUDA code-block runtime job");

        assert_eq!(job.output_offset, 131);
        assert_eq!(job.output_stride, 99);
    }

    #[test]
    fn color_plan_flattens_one_shared_payload_for_component_decode() {
        let pixels: Vec<u8> = (0u16..4 * 4 * 3)
            .map(|idx| u8::try_from((idx * 13 + idx / 3) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream =
            encode_htj2k(&pixels, 4, 4, 3, 8, false, &options).expect("encode HTJ2K RGB fixture");
        let mut decoder = crate::J2kDecoder::new(&codestream).expect("decoder");

        let color = decoder
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("CUDA color plans");

        assert_eq!(color.components.len(), 3);
        assert!(!color.payload.is_empty());
        assert_eq!(color.report.payload_bytes, color.payload.len());
        for component in &color.components {
            assert!(component.payload().is_empty());
            for block in component.code_blocks() {
                let start = usize::try_from(block.payload_offset).expect("payload offset");
                let end = start + block.payload_len as usize;
                assert!(end <= color.payload.len());
            }
        }
    }
}

impl ImageCodec for J2kDecoder<'_> {
    type Error = Error;
    type Warning = Infallible;
    type Pool = CpuJ2kScratchPool;
}

impl<'a> ImageDecode<'a> for J2kDecoder<'a> {
    type View = J2kView<'a>;

    fn inspect(input: &'a [u8]) -> Result<signinum_core::Info, Self::Error> {
        Ok(CpuDecoder::inspect(input)?)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        Ok(J2kView::parse(input)?)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        Ok(Self {
            inner: CpuDecoder::from_view(view)?,
            pool: CpuJ2kScratchPool::new(),
        })
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

#[cfg(test)]
mod dispatch_tests {
    use super::split_htj2k_subband_decode_dispatches;

    #[test]
    fn htj2k_decode_dispatch_split_separates_ht_and_dequant_counts() {
        assert_eq!(split_htj2k_subband_decode_dispatches(0), (0, 0));
        assert_eq!(split_htj2k_subband_decode_dispatches(1), (1, 0));
        assert_eq!(split_htj2k_subband_decode_dispatches(2), (1, 1));
        assert_eq!(split_htj2k_subband_decode_dispatches(3), (2, 1));
    }
}
