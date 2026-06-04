// SPDX-License-Identifier: Apache-2.0

#[cfg(all(test, feature = "cuda-runtime"))]
use core::cell::Cell;
use core::convert::Infallible;
#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use signinum_core::BackendKind;
use signinum_core::{
    BackendRequest, DecodeOutcome, Downscale, ImageCodec, ImageDecode, ImageDecodeDevice,
    ImageDecodeSubmit, PixelFormat, ReadySubmission, Rect,
};
#[cfg(feature = "cuda-runtime")]
use signinum_cuda_runtime::{
    CudaBufferPool, CudaBufferPoolTakeTrace, CudaDeviceBuffer, CudaError, CudaHtj2kCleanupTarget,
    CudaHtj2kCodeBlockJob, CudaHtj2kDecodeResources, CudaHtj2kDecodeTableResources,
    CudaHtj2kDequantizeTarget, CudaJ2kIdwtJob, CudaJ2kIdwtTarget, CudaJ2kInverseMctJob,
    CudaJ2kRect, CudaJ2kStoreGray16Job, CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job,
    CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job, CudaJ2kStoreRgb8MctJob,
    CudaJ2kStoreRgb8MctTarget, CudaPooledDeviceBuffer, CudaQueuedExecution, CudaQueuedHtj2kCleanup,
};
use signinum_j2k::{
    adapter::device_plan::{DeviceDecodePlan, DeviceDecodeRequest},
    J2kDecoder as CpuDecoder, J2kError, J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
#[cfg(feature = "cuda-runtime")]
use signinum_j2k_native::J2kDirectBandId;
use signinum_j2k_native::{
    DecodeSettings, DecoderContext as NativeDecoderContext, Image as NativeImage,
};

#[cfg(feature = "cuda-runtime")]
use crate::runtime::cuda_error;
use crate::runtime::{validate_surface_request, wrap_cpu_staged_cuda_surface, wrap_surface};
#[cfg(feature = "cuda-runtime")]
use crate::surface::{cuda_range_storage, Storage};
use crate::{
    profile, CudaHtj2kDecodePlan, CudaHtj2kDecodeProfileDetail, CudaHtj2kProfileReport,
    CudaSession, Error, Surface,
};
#[cfg(feature = "cuda-runtime")]
use crate::{
    CudaHtj2kIdwtStep, CudaHtj2kStoreStep, CudaHtj2kTransform, CudaSurfaceStats, SurfaceResidency,
};

#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_KERNELS_NOT_READY: &str =
    "strict CUDA HTJ2K resident codestream decode kernels are not available in this build";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED: &str =
    "strict CUDA HTJ2K resident decode currently accepts Gray8, Gray16, Rgb8, Rgba8, Rgb16, and Rgba16 output";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_STORE_UNSUPPORTED: &str =
    "strict CUDA HTJ2K resident decode requires a single grayscale store step";
#[cfg(feature = "cuda-runtime")]
const CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE: &str =
    "strict CUDA HTJ2K resident batch decode payload is too large";
#[cfg(feature = "cuda-runtime")]
const CUDA_IDWT_TRACE_ENV_VAR: &str = "SIGNINUM_CUDA_IDWT_TRACE";

#[cfg(all(test, feature = "cuda-runtime"))]
std::thread_local! {
    static CUDA_HTJ2K_BATCH_DECODE_CALLS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn testing_reset_cuda_htj2k_batch_decode_calls() {
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(|calls| calls.set(0));
}

#[cfg(all(test, feature = "cuda-runtime"))]
pub(crate) fn testing_cuda_htj2k_batch_decode_calls() -> usize {
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(Cell::get)
}

#[cfg(any(test, feature = "cuda-runtime"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CudaIdwtBatchHostTraceRow {
    component_count: usize,
    step_count: usize,
    output_alloc_us: u128,
    target_build_us: u128,
    enqueue_us: u128,
    output_take_count: usize,
    output_pool_reuse_count: usize,
    output_pool_alloc_count: usize,
    output_pool_scanned_count: usize,
    output_pool_max_free_count: usize,
    output_requested_bytes: usize,
}

#[cfg(any(test, feature = "cuda-runtime"))]
fn format_cuda_idwt_batch_host_trace_row(row: CudaIdwtBatchHostTraceRow) -> String {
    format!(
        "signinum_profile codec=j2k op=cuda_idwt_batch_host path=decode \
         component_count={} step_count={} output_alloc_us={} target_build_us={} enqueue_us={} \
         output_take_count={} output_pool_reuse_count={} output_pool_alloc_count={} \
         output_pool_scanned_count={} output_pool_max_free_count={} output_requested_bytes={}",
        row.component_count,
        row.step_count,
        row.output_alloc_us,
        row.target_build_us,
        row.enqueue_us,
        row.output_take_count,
        row.output_pool_reuse_count,
        row.output_pool_alloc_count,
        row.output_pool_scanned_count,
        row.output_pool_max_free_count,
        row.output_requested_bytes
    )
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct CudaIdwtOutputPoolTraceTotals {
    take_count: usize,
    reuse_count: usize,
    alloc_count: usize,
    scanned_count: usize,
    max_free_count: usize,
    requested_bytes: usize,
}

#[cfg(feature = "cuda-runtime")]
impl CudaIdwtOutputPoolTraceTotals {
    fn add_take(&mut self, trace: CudaBufferPoolTakeTrace) {
        self.take_count = self.take_count.saturating_add(1);
        if trace.reused {
            self.reuse_count = self.reuse_count.saturating_add(1);
        } else {
            self.alloc_count = self.alloc_count.saturating_add(1);
        }
        self.scanned_count = self.scanned_count.saturating_add(trace.scanned_count);
        self.max_free_count = self.max_free_count.max(trace.free_count_before);
        self.requested_bytes = self.requested_bytes.saturating_add(trace.requested_len);
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_idwt_trace_enabled() -> bool {
    std::env::var_os(CUDA_IDWT_TRACE_ENV_VAR).is_some()
}

#[cfg(feature = "cuda-runtime")]
fn elapsed_host_us(start: Option<std::time::Instant>) -> u128 {
    start.map_or(0, |start| start.elapsed().as_micros())
}

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
        let mut native_context = NativeDecoderContext::default();
        build_cuda_htj2k_color_plans_from_bytes_with_profile(
            self.inner.bytes(),
            fmt,
            &mut native_context,
        )
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
    buffer: CudaPooledDeviceBuffer,
}

#[cfg(feature = "cuda-runtime")]
struct CudaPendingDequantBand {
    band_index: usize,
    jobs: Vec<CudaHtj2kCodeBlockJob>,
    output_words: usize,
}

#[cfg(feature = "cuda-runtime")]
struct CudaComponentDecodeWork {
    bands: Vec<CudaCoefficientBand>,
    pending_dequant_bands: Vec<CudaPendingDequantBand>,
    store: CudaHtj2kStoreStep,
    dispatches: usize,
    decode_dispatches: usize,
    timings: CudaDecodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
struct CudaQueuedIdwtBatch {
    queued: Vec<CudaQueuedExecution>,
    kernel_dispatches: usize,
    decode_dispatches: usize,
}

#[cfg(feature = "cuda-runtime")]
struct CudaDecodedComponent {
    buffer: CudaPooledDeviceBuffer,
    store: CudaHtj2kStoreStep,
    dispatches: usize,
    decode_dispatches: usize,
    timings: CudaDecodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
struct CudaPreparedRgb8MctBatchStore {
    color: CudaHtj2kColorDecodePlans,
    decoded_components: Vec<CudaDecodedComponent>,
    dispatches: usize,
    decode_dispatches: usize,
    job: CudaJ2kStoreRgb8MctJob,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy, Debug, Default)]
struct CudaDecodeStageTimings {
    h2d: u128,
    payload_upload: u128,
    status_d2h: u128,
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
        report.detail.payload_upload_us = report
            .detail
            .payload_upload_us
            .saturating_add(self.payload_upload);
        report.detail.status_d2h_us = report.detail.status_d2h_us.saturating_add(self.status_d2h);
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
    decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, false)
        .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_to_cuda_resident_surface_with_profile_impl(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    decode_to_cuda_resident_surface_with_profile_control(decoder, session, fmt, true)
}

#[cfg(feature = "cuda-runtime")]
fn decode_to_cuda_resident_surface_with_profile_control(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let collect_stage_timings = collect_stage_timings || profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    match fmt {
        PixelFormat::Gray8 | PixelFormat::Gray16 => {
            decode_grayscale_cuda_resident_surface_with_profile(
                decoder,
                session,
                fmt,
                wall_started,
                collect_stage_timings,
            )
        }
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_surface_with_profile(
                decoder,
                session,
                fmt,
                wall_started,
                collect_stage_timings,
            )
        }
        _ => Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_OUTPUT_FORMAT_UNSUPPORTED,
        }),
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_batch_to_cuda_resident_surface_with_profile_control(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    #[cfg(all(test, feature = "cuda-runtime"))]
    CUDA_HTJ2K_BATCH_DECODE_CALLS.with(|calls| calls.set(calls.get().saturating_add(1)));

    let collect_stage_timings = collect_stage_timings || profile::profile_stages_enabled();
    if inputs.is_empty() {
        return Ok((
            Vec::new(),
            CudaHtj2kProfileReport {
                residency: SurfaceResidency::CudaResidentDecode,
                ..CudaHtj2kProfileReport::default()
            },
        ));
    }
    match fmt {
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            decode_color_cuda_resident_batch_surfaces_with_profile(
                inputs,
                session,
                fmt,
                collect_stage_timings,
            )
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
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_plan_with_profile(fmt)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_region_plan_with_profile(fmt, roi)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) =
        decoder.build_cuda_htj2k_grayscale_scaled_plan_with_profile(fmt, output_dimensions)?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let (plan, mut report) = decoder.build_cuda_htj2k_grayscale_region_scaled_plan_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_grayscale_cuda_resident_surface_with_plan_profile(
        session,
        fmt,
        &plan,
        &mut report,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_grayscale_cuda_resident_surface_with_plan_profile(
    session: &mut CudaSession,
    fmt: PixelFormat,
    plan: &CudaHtj2kDecodePlan,
    report: &mut CudaHtj2kProfileReport,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let context = session.cuda_context()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = session.htj2k_decode_table_resources()?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    report.h2d_us = report.h2d_us.saturating_add(table_upload_us);
    report.detail.table_upload_us = report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let pool = session.decode_buffer_pool()?;
    let component = decode_cuda_component_plan(
        &context,
        plan,
        &table_resources,
        &pool,
        collect_stage_timings,
    )?;
    let input_width = component
        .store
        .input_rect
        .x1
        .saturating_sub(component.store.input_rect.x0);
    let component_buffer = pooled_cuda_buffer(&component.buffer)?;
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "signinum.htj2k.decode.store.gray",
            || match fmt {
                PixelFormat::Gray8 => context.j2k_store_gray8_device(
                    component_buffer,
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
                    component_buffer,
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
            },
        )
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
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let color = decoder.build_cuda_htj2k_color_plans_with_profile(fmt)?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    output_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let color = decoder.build_cuda_htj2k_color_scaled_plans_with_profile(fmt, output_dimensions)?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_region_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let color = decoder.build_cuda_htj2k_color_region_plans_with_profile(fmt, roi)?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_region_scaled_surface(
    decoder: &mut J2kDecoder<'_>,
    session: &mut CudaSession,
    fmt: PixelFormat,
    scaled_roi: Rect,
    scaled_dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let collect_stage_timings = profile::profile_stages_enabled();
    let wall_started = profile::profile_now(collect_stage_timings);
    let color = decoder.build_cuda_htj2k_color_region_scaled_plans_with_profile(
        fmt,
        scaled_roi,
        scaled_dimensions,
    )?;
    decode_color_cuda_resident_surface_with_plans_profile(
        session,
        fmt,
        color,
        wall_started,
        collect_stage_timings,
    )
    .map(|(surface, _report)| surface)
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_surface_with_plans_profile(
    session: &mut CudaSession,
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    if color.components.len() != 3 {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    let context = session.cuda_context()?;
    let pool = session.decode_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = session.htj2k_decode_table_resources()?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    color.report.h2d_us = color.report.h2d_us.saturating_add(table_upload_us);
    color.report.detail.table_upload_us = color
        .report
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(&color.payload, &table_resources)
        .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);
    profile::add_payload_resource_upload_us(&mut color.report, payload_upload_us);
    let mut component_work = Vec::with_capacity(3);
    for plan in &color.components {
        component_work.push(decode_cuda_component_subbands_with_resources(
            &context,
            plan,
            &pool,
            collect_stage_timings,
        )?);
    }
    run_component_cleanup_dequant_batches(
        &context,
        &decode_resources,
        &mut component_work,
        &pool,
        collect_stage_timings,
    )?;
    finish_color_cuda_resident_surface_with_component_work(
        &context,
        &pool,
        fmt,
        color,
        component_work,
        wall_started,
        collect_stage_timings,
        true,
        true,
    )
}

#[cfg(feature = "cuda-runtime")]
fn decode_color_cuda_resident_batch_surfaces_with_profile(
    inputs: &[&[u8]],
    session: &mut CudaSession,
    fmt: PixelFormat,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, CudaHtj2kProfileReport), Error> {
    let batch_wall_started = profile::profile_now(collect_stage_timings);
    let mut colors = Vec::with_capacity(inputs.len());
    let mut shared_payload = Vec::new();
    let mut native_context = NativeDecoderContext::default();
    for input in inputs {
        let mut color =
            build_cuda_htj2k_color_plans_from_bytes_with_profile(input, fmt, &mut native_context)?;
        if color.components.len() != 3 {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        append_color_payload_to_shared(&mut color, &mut shared_payload)?;
        colors.push(color);
    }

    let context = session.cuda_context()?;
    let pool = session.decode_batch_buffer_pool()?;
    let table_upload_start = profile::profile_now(collect_stage_timings);
    let table_resources = session.htj2k_decode_table_resources()?;
    let table_upload_us = profile::elapsed_us(table_upload_start);
    let payload_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables_and_pool(
            &shared_payload,
            &table_resources,
            &pool,
        )
        .map_err(cuda_error)?;
    let payload_upload_us = profile::elapsed_us(payload_upload_start);

    let component_count = colors
        .iter()
        .map(|color| color.components.len())
        .sum::<usize>();
    let mut all_component_work = Vec::with_capacity(component_count);
    for color in &colors {
        for plan in &color.components {
            all_component_work.push(decode_cuda_component_subbands_with_resources(
                &context,
                plan,
                &pool,
                collect_stage_timings,
            )?);
        }
    }
    run_component_cleanup_dequant_batches(
        &context,
        &decode_resources,
        &mut all_component_work,
        &pool,
        collect_stage_timings,
    )?;
    let batch_components = colors
        .iter()
        .flat_map(|color| color.components.iter())
        .collect::<Vec<_>>();
    let idwt_batched = can_batch_color_idwt(&batch_components);
    let pending_idwt_batch = if idwt_batched {
        run_color_component_idwt_batches(
            &context,
            &batch_components,
            &mut all_component_work,
            &pool,
            collect_stage_timings,
        )?
    } else {
        None
    };
    drop(batch_components);

    let can_use_batch_store =
        idwt_batched && can_batch_rgb8_mct_color_store(fmt, &colors, &all_component_work)?;
    let (surfaces, reports) = if can_use_batch_store {
        finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
            &context,
            fmt,
            colors,
            all_component_work,
            collect_stage_timings,
        )?
    } else {
        let mut surfaces = Vec::with_capacity(colors.len());
        let mut reports = Vec::with_capacity(colors.len());
        let mut work_iter = all_component_work.into_iter();
        for color in colors {
            let component_count = color.components.len();
            let component_work = work_iter.by_ref().take(component_count).collect::<Vec<_>>();
            if component_work.len() != component_count {
                return Err(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                });
            }
            let (surface, report) = finish_color_cuda_resident_surface_with_component_work(
                &context,
                &pool,
                fmt,
                color,
                component_work,
                None,
                collect_stage_timings,
                !idwt_batched,
                false,
            )?;
            surfaces.push(surface);
            reports.push(report);
        }
        (surfaces, reports)
    };
    drop(pending_idwt_batch);

    let aggregate = finalize_color_batch_decode_report(
        &reports,
        table_upload_us,
        payload_upload_us,
        batch_wall_started,
    );
    aggregate.emit("decode_batch");

    Ok((surfaces, aggregate))
}

#[cfg(feature = "cuda-runtime")]
fn build_cuda_htj2k_color_plans_from_bytes_with_profile<'a>(
    input: &'a [u8],
    fmt: PixelFormat,
    native_context: &mut NativeDecoderContext<'a>,
) -> Result<CudaHtj2kColorDecodePlans, Error> {
    let total_start = profile::profile_now(true);

    let parse_start = profile::profile_now(true);
    let image = NativeImage::new(input, &DecodeSettings::default())
        .map_err(|error| Error::Decode(J2kError::Backend(error.to_string())))?;
    let parse_us = profile::elapsed_us(parse_start);

    let plan_start = profile::profile_now(true);
    let native_plan = image
        .build_direct_color_plan_with_context(native_context)
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
fn finalize_color_batch_decode_report(
    reports: &[CudaHtj2kProfileReport],
    table_upload_us: u128,
    payload_upload_us: u128,
    batch_wall_started: Option<profile::ProfileInstant>,
) -> CudaHtj2kProfileReport {
    let mut aggregate = aggregate_decode_reports(reports);
    aggregate.h2d_us = aggregate
        .h2d_us
        .saturating_add(table_upload_us)
        .saturating_add(payload_upload_us);
    aggregate.detail.table_upload_us = aggregate
        .detail
        .table_upload_us
        .saturating_add(table_upload_us);
    aggregate.detail.payload_upload_us = aggregate
        .detail
        .payload_upload_us
        .saturating_add(payload_upload_us);
    aggregate.detail.wall_total_us = profile::elapsed_us(batch_wall_started);
    profile::finalize_decode_total_us(&mut aggregate);
    aggregate
}

#[cfg(feature = "cuda-runtime")]
fn can_batch_rgb8_mct_color_store(
    fmt: PixelFormat,
    colors: &[CudaHtj2kColorDecodePlans],
    all_component_work: &[CudaComponentDecodeWork],
) -> Result<bool, Error> {
    if !matches!(fmt, PixelFormat::Rgb8 | PixelFormat::Rgba8) {
        return Ok(false);
    }

    let mut offset = 0usize;
    for color in colors {
        let component_count = color.components.len();
        if component_count != 3 || offset.saturating_add(component_count) > all_component_work.len()
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        if !color.mct {
            return Ok(false);
        }
        let component_work = &all_component_work[offset..offset + component_count];
        let stores = [
            &component_work[0].store,
            &component_work[1].store,
            &component_work[2].store,
        ];
        validate_color_stores(stores, color.dimensions)?;
        if !can_fuse_mct_store_for_stores(stores) {
            return Ok(false);
        }
        offset = offset.saturating_add(component_count);
    }

    if offset != all_component_work.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    Ok(!colors.is_empty())
}

#[cfg(feature = "cuda-runtime")]
fn finish_color_cuda_resident_batch_surfaces_with_rgb8_mct_store(
    context: &signinum_cuda_runtime::CudaContext,
    fmt: PixelFormat,
    colors: Vec<CudaHtj2kColorDecodePlans>,
    all_component_work: Vec<CudaComponentDecodeWork>,
    collect_stage_timings: bool,
) -> Result<(Vec<Surface>, Vec<CudaHtj2kProfileReport>), Error> {
    let mut prepared = Vec::with_capacity(colors.len());
    let mut work_iter = all_component_work.into_iter();
    for color in colors {
        let component_count = color.components.len();
        let component_work = work_iter.by_ref().take(component_count).collect::<Vec<_>>();
        if component_work.len() != component_count {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
        prepared.push(prepare_rgb8_mct_batch_store(fmt, color, component_work)?);
    }
    if work_iter.next().is_some() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }

    let targets = prepared
        .iter()
        .map(rgb8_mct_batch_store_target)
        .collect::<Result<Vec<_>, Error>>()?;
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "signinum.htj2k.decode.store.color.batch",
            || context.j2k_store_rgb8_mct_batch_contiguous_device(&targets),
        )
        .map_err(cuda_error)?;
    drop(targets);
    let (surface_buffer, surface_ranges, store_stats) = store_output.into_parts();
    if surface_ranges.len() != prepared.len() {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }
    let shared_surface_buffer = Arc::new(surface_buffer);

    let mut surfaces = Vec::with_capacity(prepared.len());
    let mut reports = Vec::with_capacity(prepared.len());
    let store_dispatches = store_stats.kernel_dispatches();
    let store_decode_dispatches = store_stats.decode_kernel_dispatches();
    for (index, (mut prepared, surface_range)) in prepared
        .into_iter()
        .zip(surface_ranges.into_iter())
        .enumerate()
    {
        let report_store_dispatches = if index == 0 { store_dispatches } else { 0 };
        let report_store_decode_dispatches = if index == 0 {
            store_decode_dispatches
        } else {
            0
        };
        let report_store_us = if index == 0 { store_us } else { 0 };
        let dispatches = prepared.dispatches.saturating_add(report_store_dispatches);
        let decode_dispatches = prepared
            .decode_dispatches
            .saturating_add(report_store_decode_dispatches);
        prepared.color.report.dispatch_count = dispatches;
        prepared.color.report.store_us = prepared
            .color
            .report
            .store_us
            .saturating_add(report_store_us);
        prepared.color.report.detail.store_dispatch_count = prepared
            .color
            .report
            .detail
            .store_dispatch_count
            .saturating_add(report_store_dispatches);
        profile::finalize_decode_total_us(&mut prepared.color.report);

        let dimensions = prepared.color.dimensions;
        surfaces.push(Surface {
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
            storage: cuda_range_storage(
                shared_surface_buffer.clone(),
                surface_range.offset,
                surface_range.len,
            ),
        });
        reports.push(prepared.color.report);
    }

    Ok((surfaces, reports))
}

#[cfg(feature = "cuda-runtime")]
fn prepare_rgb8_mct_batch_store(
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    component_work: Vec<CudaComponentDecodeWork>,
) -> Result<CudaPreparedRgb8MctBatchStore, Error> {
    let decoded_components = component_work
        .into_iter()
        .map(finish_cuda_component_decode)
        .collect::<Result<Vec<_>, Error>>()?;
    let [component0, component1, component2] = decoded_components.as_slice() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    };
    let stores = [&component0.store, &component1.store, &component2.store];
    validate_color_stores(stores, color.dimensions)?;
    if !color.mct || !can_fuse_mct_store_for_stores(stores) {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    }

    let dispatches = decoded_components
        .iter()
        .map(|component| component.dispatches)
        .sum::<usize>();
    let decode_dispatches = decoded_components
        .iter()
        .map(|component| component.decode_dispatches)
        .sum::<usize>();
    for component in &decoded_components {
        component.timings.add_to_report(&mut color.report);
    }

    let addends = [
        bit_depth_addend(color.bit_depths[0]),
        bit_depth_addend(color.bit_depths[1]),
        bit_depth_addend(color.bit_depths[2]),
    ];
    let job = CudaJ2kStoreRgb8MctJob {
        store: CudaJ2kStoreRgb8Job {
            input_width0: color_store_input_width(&component0.store),
            input_width1: color_store_input_width(&component1.store),
            input_width2: color_store_input_width(&component2.store),
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
        irreversible97: u32::from(color.transform == CudaHtj2kTransform::Irreversible97),
    };

    Ok(CudaPreparedRgb8MctBatchStore {
        color,
        decoded_components,
        dispatches,
        decode_dispatches,
        job,
    })
}

#[cfg(feature = "cuda-runtime")]
fn rgb8_mct_batch_store_target(
    prepared: &CudaPreparedRgb8MctBatchStore,
) -> Result<CudaJ2kStoreRgb8MctTarget<'_>, Error> {
    let [component0, component1, component2] = prepared.decoded_components.as_slice() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        });
    };
    Ok(CudaJ2kStoreRgb8MctTarget {
        plane0: pooled_cuda_buffer(&component0.buffer)?,
        plane1: pooled_cuda_buffer(&component1.buffer)?,
        plane2: pooled_cuda_buffer(&component2.buffer)?,
        job: prepared.job,
    })
}

#[cfg(feature = "cuda-runtime")]
fn can_fuse_mct_store_for_stores(stores: [&CudaHtj2kStoreStep; 3]) -> bool {
    let input_width0 = color_store_input_width(stores[0]);
    let input_width1 = color_store_input_width(stores[1]);
    let input_width2 = color_store_input_width(stores[2]);
    input_width0 == input_width1
        && input_width0 == input_width2
        && stores[0].source_x == stores[1].source_x
        && stores[0].source_x == stores[2].source_x
        && stores[0].source_y == stores[1].source_y
        && stores[0].source_y == stores[2].source_y
}

#[cfg(feature = "cuda-runtime")]
fn color_store_input_width(store: &CudaHtj2kStoreStep) -> u32 {
    store.input_rect.x1.saturating_sub(store.input_rect.x0)
}

#[cfg(feature = "cuda-runtime")]
#[allow(clippy::too_many_arguments)]
fn finish_color_cuda_resident_surface_with_component_work(
    context: &signinum_cuda_runtime::CudaContext,
    pool: &CudaBufferPool,
    fmt: PixelFormat,
    mut color: CudaHtj2kColorDecodePlans,
    mut component_work: Vec<CudaComponentDecodeWork>,
    wall_started: Option<profile::ProfileInstant>,
    collect_stage_timings: bool,
    run_idwt: bool,
    emit_report: bool,
) -> Result<(Surface, CudaHtj2kProfileReport), Error> {
    let pending_idwt_batch = if run_idwt {
        let batch_components = color.components.iter().collect::<Vec<_>>();
        if can_batch_color_idwt(&batch_components) {
            run_color_component_idwt_batches(
                context,
                &batch_components,
                &mut component_work,
                pool,
                collect_stage_timings,
            )?
        } else {
            for (plan, work) in color.components.iter().zip(component_work.iter_mut()) {
                run_cuda_component_idwt_steps(
                    context,
                    plan.idwt_steps(),
                    work,
                    pool,
                    collect_stage_timings,
                )?;
            }
            None
        }
    } else {
        None
    };
    let decoded_components = component_work
        .into_iter()
        .map(finish_cuda_component_decode)
        .collect::<Result<Vec<_>, Error>>()?;
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
    let component0_buffer = pooled_cuda_buffer(&component0.buffer)?;
    let component1_buffer = pooled_cuda_buffer(&component1.buffer)?;
    let component2_buffer = pooled_cuda_buffer(&component2.buffer)?;
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
    let irreversible97 = u32::from(color.transform == CudaHtj2kTransform::Irreversible97);
    let mct_store_addends = [
        bit_depth_addend(color.bit_depths[0]),
        bit_depth_addend(color.bit_depths[1]),
        bit_depth_addend(color.bit_depths[2]),
    ];
    let can_fuse_mct_store = color.mct
        && input_width0 == input_width1
        && input_width0 == input_width2
        && component0.store.source_x == component1.store.source_x
        && component0.store.source_x == component2.store.source_x
        && component0.store.source_y == component1.store.source_y
        && component0.store.source_y == component2.store.source_y;
    let addends = if color.mct && can_fuse_mct_store {
        mct_store_addends
    } else if color.mct {
        let mct_len = u32::try_from(checked_area(
            color.mct_dimensions.0,
            color.mct_dimensions.1,
        )?)
        .map_err(|_| Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
        let stats = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "signinum.htj2k.decode.mct",
                || {
                    context.j2k_inverse_mct_device(
                        component0_buffer,
                        component1_buffer,
                        component2_buffer,
                        CudaJ2kInverseMctJob {
                            len: mct_len,
                            irreversible97,
                            addend0: mct_store_addends[0],
                            addend1: mct_store_addends[1],
                            addend2: mct_store_addends[2],
                        },
                    )
                },
            )
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
    let (store_output, store_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "signinum.htj2k.decode.store.color",
            || match fmt {
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    let store_job = CudaJ2kStoreRgb8Job {
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
                    };
                    if can_fuse_mct_store {
                        context.j2k_store_rgb8_mct_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            CudaJ2kStoreRgb8MctJob {
                                store: store_job,
                                irreversible97,
                            },
                        )
                    } else {
                        context.j2k_store_rgb8_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            store_job,
                        )
                    }
                }
                PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
                    let store_job = CudaJ2kStoreRgb16Job {
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
                    };
                    if can_fuse_mct_store {
                        context.j2k_store_rgb16_mct_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            CudaJ2kStoreRgb16MctJob {
                                store: store_job,
                                irreversible97,
                            },
                        )
                    } else {
                        context.j2k_store_rgb16_device(
                            component0_buffer,
                            component1_buffer,
                            component2_buffer,
                            store_job,
                        )
                    }
                }
                _ => {
                    unreachable!("validated color CUDA output format");
                }
            },
        )
        .map_err(cuda_error)?;
    drop(pending_idwt_batch);
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
    if emit_report {
        color.report.emit("decode");
    }

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

#[cfg(feature = "cuda-runtime")]
fn append_color_payload_to_shared(
    color: &mut CudaHtj2kColorDecodePlans,
    shared_payload: &mut Vec<u8>,
) -> Result<(), Error> {
    let base = u64::try_from(shared_payload.len()).map_err(|_| Error::UnsupportedCudaRequest {
        reason: CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
    })?;
    shared_payload
        .try_reserve(color.payload.len())
        .map_err(|_| Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_BATCH_PAYLOAD_TOO_LARGE,
        })?;
    for component in &mut color.components {
        component.rebase_payload_offsets(base)?;
    }
    shared_payload.append(&mut color.payload);
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn aggregate_decode_reports(reports: &[CudaHtj2kProfileReport]) -> CudaHtj2kProfileReport {
    let mut aggregate = CudaHtj2kProfileReport {
        residency: SurfaceResidency::CudaResidentDecode,
        ..CudaHtj2kProfileReport::default()
    };
    for report in reports {
        add_decode_report(&mut aggregate, report);
    }
    aggregate
}

#[cfg(feature = "cuda-runtime")]
fn add_decode_report(aggregate: &mut CudaHtj2kProfileReport, report: &CudaHtj2kProfileReport) {
    aggregate.parse_us = aggregate.parse_us.saturating_add(report.parse_us);
    aggregate.plan_us = aggregate.plan_us.saturating_add(report.plan_us);
    aggregate.flatten_us = aggregate.flatten_us.saturating_add(report.flatten_us);
    aggregate.h2d_us = aggregate.h2d_us.saturating_add(report.h2d_us);
    aggregate.ht_cleanup_us = aggregate.ht_cleanup_us.saturating_add(report.ht_cleanup_us);
    aggregate.ht_refine_us = aggregate.ht_refine_us.saturating_add(report.ht_refine_us);
    aggregate.dequant_us = aggregate.dequant_us.saturating_add(report.dequant_us);
    aggregate.idwt_us = aggregate.idwt_us.saturating_add(report.idwt_us);
    aggregate.mct_us = aggregate.mct_us.saturating_add(report.mct_us);
    aggregate.store_us = aggregate.store_us.saturating_add(report.store_us);
    aggregate.block_count = aggregate.block_count.saturating_add(report.block_count);
    aggregate.payload_bytes = aggregate.payload_bytes.saturating_add(report.payload_bytes);
    aggregate.dispatch_count = aggregate
        .dispatch_count
        .saturating_add(report.dispatch_count);
    aggregate.detail.table_upload_us = aggregate
        .detail
        .table_upload_us
        .saturating_add(report.detail.table_upload_us);
    aggregate.detail.payload_upload_us = aggregate
        .detail
        .payload_upload_us
        .saturating_add(report.detail.payload_upload_us);
    aggregate.detail.job_upload_us = aggregate
        .detail
        .job_upload_us
        .saturating_add(report.detail.job_upload_us);
    aggregate.detail.status_d2h_us = aggregate
        .detail
        .status_d2h_us
        .saturating_add(report.detail.status_d2h_us);
    aggregate.detail.output_d2h_us = aggregate
        .detail
        .output_d2h_us
        .saturating_add(report.detail.output_d2h_us);
    aggregate.detail.ht_dispatch_count = aggregate
        .detail
        .ht_dispatch_count
        .saturating_add(report.detail.ht_dispatch_count);
    aggregate.detail.dequant_dispatch_count = aggregate
        .detail
        .dequant_dispatch_count
        .saturating_add(report.detail.dequant_dispatch_count);
    aggregate.detail.idwt_dispatch_count = aggregate
        .detail
        .idwt_dispatch_count
        .saturating_add(report.detail.idwt_dispatch_count);
    aggregate.detail.mct_dispatch_count = aggregate
        .detail
        .mct_dispatch_count
        .saturating_add(report.detail.mct_dispatch_count);
    aggregate.detail.store_dispatch_count = aggregate
        .detail
        .store_dispatch_count
        .saturating_add(report.detail.store_dispatch_count);
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

#[cfg(feature = "cuda-runtime")]
fn decode_cuda_component_plan(
    context: &signinum_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    tables: &CudaHtj2kDecodeTableResources,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let resource_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(plan.payload(), tables)
        .map_err(cuda_error)?;
    let resource_upload_us = profile::elapsed_us(resource_upload_start);
    let mut component = decode_cuda_component_plan_with_resources(
        context,
        plan,
        &decode_resources,
        pool,
        collect_stage_timings,
    )?;
    component.timings.h2d = component.timings.h2d.saturating_add(resource_upload_us);
    component.timings.payload_upload = component
        .timings
        .payload_upload
        .saturating_add(resource_upload_us);
    Ok(component)
}

#[cfg(test)]
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
fn htj2k_batched_cleanup_dispatches(target_count: usize) -> usize {
    usize::from(target_count > 0)
}

#[cfg(any(feature = "cuda-runtime", test))]
fn htj2k_batched_dequant_dispatches(target_count: usize) -> usize {
    usize::from(target_count > 0)
}

#[cfg(feature = "cuda-runtime")]
fn htj2k_batched_cleanup_dequant_dispatches(
    target_count: usize,
    fused_cleanup_dequant: bool,
) -> (usize, usize) {
    if target_count == 0 {
        return (0, 0);
    }
    if fused_cleanup_dequant {
        (1, 0)
    } else {
        (1, 1)
    }
}

#[cfg(feature = "cuda-runtime")]
fn decode_cuda_component_plan_with_resources(
    context: &signinum_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    decode_resources: &CudaHtj2kDecodeResources,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let mut work =
        decode_cuda_component_subbands_with_resources(context, plan, pool, collect_stage_timings)?;
    run_component_cleanup_dequant_batches(
        context,
        decode_resources,
        std::slice::from_mut(&mut work),
        pool,
        collect_stage_timings,
    )?;
    run_cuda_component_idwt_steps(
        context,
        plan.idwt_steps(),
        &mut work,
        pool,
        collect_stage_timings,
    )?;
    finish_cuda_component_decode(work)
}

#[cfg(feature = "cuda-runtime")]
fn decode_cuda_component_subbands_with_resources(
    context: &signinum_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaComponentDecodeWork, Error> {
    let mut bands = Vec::with_capacity(plan.subbands().len() + plan.idwt_steps().len());
    let mut pending_dequant_bands = Vec::with_capacity(plan.subbands().len());
    let dispatches = 0usize;
    let decode_dispatches = 0usize;
    let mut timings = CudaDecodeStageTimings::default();

    for subband in plan.subbands() {
        let start = subband.code_block_start as usize;
        let end = start + subband.code_block_count as usize;
        let jobs = plan.code_blocks()[start..end]
            .iter()
            .map(|block| cuda_code_block_job_from_plan_block(block, subband.width))
            .collect::<Result<Vec<_>, Error>>()?;
        let output_words = checked_area(subband.width, subband.height)?;
        let allocate_start = profile::profile_now(collect_stage_timings);
        let output = context
            .allocate_htj2k_codeblock_coefficients_with_pool(&jobs, output_words, pool)
            .map_err(cuda_error)?;
        let allocate_wall_us = profile::elapsed_us(allocate_start);
        timings.h2d = timings.h2d.saturating_add(allocate_wall_us);
        let (buffer, _, _) = output.into_parts();
        let band_index = bands.len();
        bands.push(CudaCoefficientBand {
            band_id: subband.band_id,
            buffer,
        });
        if !jobs.is_empty() {
            pending_dequant_bands.push(CudaPendingDequantBand {
                band_index,
                jobs,
                output_words,
            });
        }
    }

    let [store] = plan.store_steps() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_STORE_UNSUPPORTED,
        });
    };

    Ok(CudaComponentDecodeWork {
        bands,
        pending_dequant_bands,
        store: *store,
        dispatches,
        decode_dispatches,
        timings,
    })
}

#[cfg(feature = "cuda-runtime")]
fn run_component_cleanup_dequant_batches(
    context: &signinum_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<(), Error> {
    let pending_count = component_work
        .iter()
        .map(|work| work.pending_dequant_bands.len())
        .sum::<usize>();
    if pending_count == 0 {
        return Ok(());
    }
    let accounting_index = component_work
        .iter()
        .position(|work| !work.pending_dequant_bands.is_empty())
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;

    let has_refinement = component_work.iter().any(|work| {
        work.pending_dequant_bands.iter().any(|pending| {
            pending
                .jobs
                .iter()
                .any(|job| job.refinement_length > 0 || u32::from(job.number_of_coding_passes) > 1)
        })
    });
    let cleanup_targets = component_work
        .iter()
        .flat_map(|work| {
            work.pending_dequant_bands
                .iter()
                .map(move |pending| (work, pending))
        })
        .map(|(work, pending)| {
            let coefficients = pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?;
            Ok(CudaHtj2kCleanupTarget {
                coefficients,
                jobs: &pending.jobs,
                output_words: pending.output_words,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    if !has_refinement {
        let stage_start = profile::profile_now(collect_stage_timings);
        let ((stats, runtime_timings), fused_us) = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "signinum.htj2k.decode.cleanup_dequantize.batch",
                || {
                    context
                        .decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed(
                            decode_resources,
                            &cleanup_targets,
                            pool,
                            collect_stage_timings,
                        )
                },
            )
            .map_err(cuda_error)?;
        let stage_wall_us = profile::elapsed_us(stage_start);
        let (cleanup_dispatches, dequant_dispatches) =
            htj2k_batched_cleanup_dequant_dispatches(pending_count, true);
        {
            let accounting = &mut component_work[accounting_index];
            accounting.timings.h2d = accounting
                .timings
                .h2d
                .saturating_add(stage_wall_us.saturating_sub(fused_us));
            accounting.timings.ht_cleanup = accounting.timings.ht_cleanup.saturating_add(fused_us);
            accounting.timings.status_d2h = accounting
                .timings
                .status_d2h
                .saturating_add(runtime_timings.status_d2h_us);
            accounting.timings.ht_dispatch_count = accounting
                .timings
                .ht_dispatch_count
                .saturating_add(cleanup_dispatches);
            accounting.timings.dequant_dispatch_count = accounting
                .timings
                .dequant_dispatch_count
                .saturating_add(dequant_dispatches);
            accounting.dispatches = accounting
                .dispatches
                .saturating_add(stats.kernel_dispatches());
            accounting.decode_dispatches = accounting
                .decode_dispatches
                .saturating_add(stats.decode_kernel_dispatches());
        }

        for work in component_work {
            work.pending_dequant_bands.clear();
        }
        return Ok(());
    }
    let mut queued_cleanup: Option<CudaQueuedHtj2kCleanup> = None;
    let stage_start = profile::profile_now(collect_stage_timings);
    let (stats, cleanup_us, status_d2h_us) = if collect_stage_timings {
        let ((stats, runtime_timings), cleanup_us) = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "signinum.htj2k.decode.cleanup.batch",
                || {
                    context.decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
                        decode_resources,
                        &cleanup_targets,
                        pool,
                        collect_stage_timings,
                    )
                },
            )
            .map_err(cuda_error)?;
        (stats, cleanup_us, runtime_timings.status_d2h_us)
    } else {
        let (queued, cleanup_us) = context
            .time_default_stream_named_us_if(false, "signinum.htj2k.decode.cleanup.batch", || {
                context.decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
                    decode_resources,
                    &cleanup_targets,
                    pool,
                )
            })
            .map_err(cuda_error)?;
        let stats = queued.execution();
        queued_cleanup = Some(queued);
        (stats, cleanup_us, 0)
    };
    drop(cleanup_targets);
    let stage_wall_us = profile::elapsed_us(stage_start);
    {
        let accounting = &mut component_work[accounting_index];
        accounting.timings.h2d = accounting
            .timings
            .h2d
            .saturating_add(stage_wall_us.saturating_sub(cleanup_us));
        accounting.timings.ht_cleanup = accounting.timings.ht_cleanup.saturating_add(cleanup_us);
        accounting.timings.status_d2h = accounting.timings.status_d2h.saturating_add(status_d2h_us);
        if has_refinement {
            accounting.timings.ht_refine = accounting.timings.ht_refine.saturating_add(cleanup_us);
        }
        accounting.timings.ht_dispatch_count = accounting
            .timings
            .ht_dispatch_count
            .saturating_add(htj2k_batched_cleanup_dispatches(pending_count));
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(stats.kernel_dispatches());
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
    }

    let stage_start = profile::profile_now(collect_stage_timings);
    let (stats, dequant_us, dequant_target_count) = {
        let dequant_target_count = pending_count;
        let dequant_result = if let Some(queued) = queued_cleanup.as_ref() {
            context.time_default_stream_named_us_if(
                collect_stage_timings,
                "signinum.htj2k.decode.dequantize.batch",
                || context.j2k_dequantize_queued_htj2k_cleanup_with_pool(queued),
            )
        } else {
            let dequant_targets = component_work
                .iter()
                .flat_map(|work| {
                    work.pending_dequant_bands
                        .iter()
                        .map(move |pending| (work, pending))
                })
                .map(|(work, pending)| {
                    let coefficients = pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?;
                    Ok(CudaHtj2kDequantizeTarget {
                        coefficients,
                        jobs: &pending.jobs,
                        output_words: pending.output_words,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            context.time_default_stream_named_us_if(
                collect_stage_timings,
                "signinum.htj2k.decode.dequantize.batch",
                || {
                    context.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
                        &dequant_targets,
                        pool,
                    )
                },
            )
        };
        let (stats, dequant_us) = match dequant_result {
            Ok(result) => result,
            Err(error) => {
                if let Some(queued) = queued_cleanup.take() {
                    queued.finish().map_err(cuda_error)?;
                }
                return Err(cuda_error(error));
            }
        };
        (stats, dequant_us, dequant_target_count)
    };
    let stage_wall_us = profile::elapsed_us(stage_start);
    {
        let accounting = &mut component_work[accounting_index];
        accounting.timings.h2d = accounting
            .timings
            .h2d
            .saturating_add(stage_wall_us.saturating_sub(dequant_us));
        accounting.timings.dequant = accounting.timings.dequant.saturating_add(dequant_us);
        accounting.timings.dequant_dispatch_count = accounting
            .timings
            .dequant_dispatch_count
            .saturating_add(htj2k_batched_dequant_dispatches(dequant_target_count));
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(stats.kernel_dispatches());
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
    }
    if let Some(queued) = queued_cleanup.take() {
        queued.finish().map_err(cuda_error)?;
    }

    for work in component_work {
        work.pending_dequant_bands.clear();
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn run_cuda_component_idwt_steps(
    context: &signinum_cuda_runtime::CudaContext,
    steps: &[CudaHtj2kIdwtStep],
    work: &mut CudaComponentDecodeWork,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<(), Error> {
    for step in steps {
        let ll = find_cuda_band(&work.bands, step.ll_band_id)?;
        let hl = find_cuda_band(&work.bands, step.hl_band_id)?;
        let lh = find_cuda_band(&work.bands, step.lh_band_id)?;
        let hh = find_cuda_band(&work.bands, step.hh_band_id)?;
        let low_low_device = pooled_cuda_buffer(&ll.buffer)?;
        let high_low_device = pooled_cuda_buffer(&hl.buffer)?;
        let low_high_device = pooled_cuda_buffer(&lh.buffer)?;
        let high_high_device = pooled_cuda_buffer(&hh.buffer)?;
        let job = cuda_idwt_job_from_step(step);
        let (output, idwt_us) = context
            .time_default_stream_named_us_if(
                collect_stage_timings,
                "signinum.htj2k.decode.idwt",
                || {
                    if collect_stage_timings {
                        return context.j2k_inverse_dwt_single_device_with_pool(
                            low_low_device,
                            high_low_device,
                            low_high_device,
                            high_high_device,
                            job,
                            pool,
                        );
                    }
                    context.j2k_inverse_dwt_single_device_untimed_with_pool(
                        low_low_device,
                        high_low_device,
                        low_high_device,
                        high_high_device,
                        job,
                        pool,
                    )
                },
            )
            .map_err(cuda_error)?;
        work.timings.idwt = work.timings.idwt.saturating_add(idwt_us);
        let (buffer, stats) = output.into_parts();
        work.dispatches = work.dispatches.saturating_add(stats.kernel_dispatches());
        work.decode_dispatches = work
            .decode_dispatches
            .saturating_add(stats.decode_kernel_dispatches());
        work.timings.idwt_dispatch_count = work
            .timings
            .idwt_dispatch_count
            .saturating_add(stats.kernel_dispatches());
        work.bands.push(CudaCoefficientBand {
            band_id: step.output_band_id,
            buffer,
        });
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
fn finish_cuda_component_decode(
    mut work: CudaComponentDecodeWork,
) -> Result<CudaDecodedComponent, Error> {
    let input_index = work
        .bands
        .iter()
        .position(|band| band.band_id == work.store.input_band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    let input = work.bands.swap_remove(input_index);
    Ok(CudaDecodedComponent {
        buffer: input.buffer,
        store: work.store,
        dispatches: work.dispatches,
        decode_dispatches: work.decode_dispatches,
        timings: work.timings,
    })
}

#[cfg(feature = "cuda-runtime")]
fn can_batch_color_idwt(components: &[&CudaHtj2kDecodePlan]) -> bool {
    let Some(first) = components.first() else {
        return false;
    };
    components
        .iter()
        .all(|component| component.idwt_steps().len() == first.idwt_steps().len())
}

#[cfg(feature = "cuda-runtime")]
fn run_color_component_idwt_batches(
    context: &signinum_cuda_runtime::CudaContext,
    components: &[&CudaHtj2kDecodePlan],
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<Option<CudaQueuedIdwtBatch>, Error> {
    let (queued_batch, idwt_us) = context
        .time_default_stream_named_us_if(
            collect_stage_timings,
            "signinum.htj2k.decode.idwt.batch",
            || enqueue_color_component_idwt_batches(context, components, component_work, pool),
        )
        .map_err(cuda_error)?;

    if let Some(accounting) = component_work.first_mut() {
        accounting.timings.idwt = accounting.timings.idwt.saturating_add(idwt_us);
        accounting.dispatches = accounting
            .dispatches
            .saturating_add(queued_batch.kernel_dispatches);
        accounting.decode_dispatches = accounting
            .decode_dispatches
            .saturating_add(queued_batch.decode_dispatches);
        accounting.timings.idwt_dispatch_count = accounting
            .timings
            .idwt_dispatch_count
            .saturating_add(queued_batch.kernel_dispatches);
    }
    let _queued_resource_count = queued_batch
        .queued
        .iter()
        .map(CudaQueuedExecution::resource_count)
        .sum::<usize>();
    if collect_stage_timings {
        drop(queued_batch);
        Ok(None)
    } else {
        Ok(Some(queued_batch))
    }
}

#[cfg(feature = "cuda-runtime")]
fn enqueue_color_component_idwt_batches(
    context: &signinum_cuda_runtime::CudaContext,
    components: &[&CudaHtj2kDecodePlan],
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
) -> Result<CudaQueuedIdwtBatch, CudaError> {
    if components.len() != component_work.len() {
        return Err(CudaError::InvalidArgument {
            message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
        });
    }
    let Some(first) = components.first() else {
        return Ok(CudaQueuedIdwtBatch {
            queued: Vec::new(),
            kernel_dispatches: 0,
            decode_dispatches: 0,
        });
    };

    let mut queued = Vec::with_capacity(first.idwt_steps().len());
    let mut kernel_dispatches = 0usize;
    let mut decode_dispatches = 0usize;
    let step_count = first.idwt_steps().len();
    let trace_enabled = cuda_idwt_trace_enabled();
    let enqueue_result = (|| -> Result<(), CudaError> {
        let mut output_pool_trace = CudaIdwtOutputPoolTraceTotals::default();
        let output_alloc_start = trace_enabled.then(std::time::Instant::now);
        for step_index in 0..step_count {
            for (component_index, component) in components.iter().enumerate() {
                let step = component.idwt_steps().get(step_index).ok_or_else(|| {
                    CudaError::InvalidArgument {
                        message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                    }
                })?;
                let width = step.rect.x1.saturating_sub(step.rect.x0);
                let height = step.rect.y1.saturating_sub(step.rect.y0);
                let output_words = checked_area(width, height).map_err(cuda_invalid_decode_plan)?;
                let output_bytes = output_words
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| CudaError::InvalidArgument {
                        message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                    })?;
                let buffer = if trace_enabled {
                    let (buffer, trace) = pool.take_with_trace(output_bytes)?;
                    output_pool_trace.add_take(trace);
                    buffer
                } else {
                    pool.take(output_bytes)?
                };
                component_work[component_index]
                    .bands
                    .push(CudaCoefficientBand {
                        band_id: step.output_band_id,
                        buffer,
                    });
            }
        }
        let output_alloc_us = elapsed_host_us(output_alloc_start);

        let target_build_start = trace_enabled.then(std::time::Instant::now);
        let mut target_batches = Vec::with_capacity(step_count);
        for step_index in 0..step_count {
            let targets = components
                .iter()
                .enumerate()
                .map(|(component_index, component)| {
                    let step = component.idwt_steps().get(step_index).ok_or_else(|| {
                        CudaError::InvalidArgument {
                            message: CUDA_HTJ2K_KERNELS_NOT_READY.to_string(),
                        }
                    })?;
                    let work = &component_work[component_index];
                    let ll = find_cuda_band(&work.bands, step.ll_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let hl = find_cuda_band(&work.bands, step.hl_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let lh = find_cuda_band(&work.bands, step.lh_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let hh = find_cuda_band(&work.bands, step.hh_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    let output = find_cuda_band(&work.bands, step.output_band_id)
                        .map_err(cuda_invalid_decode_plan)?;
                    Ok(CudaJ2kIdwtTarget {
                        ll: pooled_cuda_buffer(&ll.buffer).map_err(cuda_invalid_decode_plan)?,
                        hl: pooled_cuda_buffer(&hl.buffer).map_err(cuda_invalid_decode_plan)?,
                        lh: pooled_cuda_buffer(&lh.buffer).map_err(cuda_invalid_decode_plan)?,
                        hh: pooled_cuda_buffer(&hh.buffer).map_err(cuda_invalid_decode_plan)?,
                        output: pooled_cuda_buffer(&output.buffer)
                            .map_err(cuda_invalid_decode_plan)?,
                        job: cuda_idwt_job_from_step(step),
                    })
                })
                .collect::<Result<Vec<_>, CudaError>>()?;
            target_batches.push(targets);
        }
        let target_build_us = elapsed_host_us(target_build_start);
        let target_slices = target_batches.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let enqueue_start = trace_enabled.then(std::time::Instant::now);
        let queued_execution =
            context.j2k_inverse_dwt_batch_sequence_enqueue_with_pool(&target_slices, pool)?;
        let enqueue_us = elapsed_host_us(enqueue_start);
        let execution = queued_execution.execution();
        kernel_dispatches = kernel_dispatches.saturating_add(execution.kernel_dispatches());
        decode_dispatches = decode_dispatches.saturating_add(execution.decode_kernel_dispatches());
        queued.push(queued_execution);
        if trace_enabled {
            let row = CudaIdwtBatchHostTraceRow {
                component_count: components.len(),
                step_count,
                output_alloc_us,
                target_build_us,
                enqueue_us,
                output_take_count: output_pool_trace.take_count,
                output_pool_reuse_count: output_pool_trace.reuse_count,
                output_pool_alloc_count: output_pool_trace.alloc_count,
                output_pool_scanned_count: output_pool_trace.scanned_count,
                output_pool_max_free_count: output_pool_trace.max_free_count,
                output_requested_bytes: output_pool_trace.requested_bytes,
            };
            eprintln!("{}", format_cuda_idwt_batch_host_trace_row(row));
        }
        Ok(())
    })();
    if let Err(error) = enqueue_result {
        if !queued.is_empty() {
            let _ = context.synchronize();
        }
        return Err(error);
    }

    Ok(CudaQueuedIdwtBatch {
        queued,
        kernel_dispatches,
        decode_dispatches,
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
fn pooled_cuda_buffer(buffer: &CudaPooledDeviceBuffer) -> Result<&CudaDeviceBuffer, Error> {
    buffer
        .as_device_buffer()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
#[allow(clippy::needless_pass_by_value)]
fn cuda_invalid_decode_plan(error: Error) -> CudaError {
    CudaError::InvalidArgument {
        message: error.to_string(),
    }
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

#[cfg(feature = "cuda-runtime")]
fn cuda_idwt_job_from_step(step: &CudaHtj2kIdwtStep) -> CudaJ2kIdwtJob {
    CudaJ2kIdwtJob {
        rect: cuda_runtime_rect(step.rect),
        ll_rect: cuda_runtime_rect(step.ll_rect),
        hl_rect: cuda_runtime_rect(step.hl_rect),
        lh_rect: cuda_runtime_rect(step.lh_rect),
        hh_rect: cuda_runtime_rect(step.hh_rect),
        irreversible97: u32::from(step.transform == CudaHtj2kTransform::Irreversible97),
    }
}

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use super::{
        build_cuda_htj2k_color_plans_from_bytes_with_profile, can_batch_color_idwt,
        cuda_code_block_job_from_plan_block, htj2k_batched_cleanup_dequant_dispatches,
        htj2k_batched_cleanup_dispatches, htj2k_batched_dequant_dispatches, CudaDecodeStageTimings,
    };
    use signinum_core::PixelFormat;
    use signinum_j2k_native::{
        encode_htj2k, DecoderContext as NativeDecoderContext, EncodeOptions,
    };

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
    fn batched_cleanup_and_dequant_dispatch_helpers_count_one_shared_dispatch() {
        assert_eq!(htj2k_batched_cleanup_dispatches(0), 0);
        assert_eq!(htj2k_batched_cleanup_dispatches(1), 1);
        assert_eq!(htj2k_batched_cleanup_dispatches(3), 1);
        assert_eq!(htj2k_batched_dequant_dispatches(0), 0);
        assert_eq!(htj2k_batched_dequant_dispatches(1), 1);
        assert_eq!(htj2k_batched_dequant_dispatches(3), 1);
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(0, true), (0, 0));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(1, true), (1, 0));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(3, true), (1, 0));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(1, false), (1, 1));
        assert_eq!(htj2k_batched_cleanup_dequant_dispatches(3, false), (1, 1));
    }

    #[test]
    fn profiled_cuda_batch_decode_api_accepts_empty_batch() {
        let mut session = crate::CudaSession::default();
        let inputs: [&[u8]; 0] = [];

        let (surfaces, report) =
            crate::J2kDecoder::decode_batch_to_device_with_session_and_profile(
                &inputs,
                PixelFormat::Rgb8,
                &mut session,
            )
            .expect("empty CUDA batch decode");

        assert!(surfaces.is_empty());
        assert_eq!(report.block_count, 0);
        assert_eq!(report.payload_bytes, 0);
    }

    #[test]
    fn cuda_batch_decode_two_color_images_matches_single_when_runtime_required() {
        let pixels_a: Vec<u8> = (0u16..16 * 16 * 3)
            .map(|idx| u8::try_from((idx * 7 + idx / 5) & 0xff).expect("masked byte"))
            .collect();
        let pixels_b: Vec<u8> = (0u16..16 * 16 * 3)
            .map(|idx| u8::try_from((idx * 11 + 23) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream_a =
            encode_htj2k(&pixels_a, 16, 16, 3, 8, false, &options).expect("encode fixture A");
        let codestream_b =
            encode_htj2k(&pixels_b, 16, 16, 3, 8, false, &options).expect("encode fixture B");
        let inputs = [codestream_a.as_slice(), codestream_b.as_slice()];
        let mut batch_session = crate::CudaSession::default();

        let batch = crate::J2kDecoder::decode_batch_to_device_with_session_and_profile(
            &inputs,
            PixelFormat::Rgb8,
            &mut batch_session,
        );
        let (surfaces, report) = match batch {
            Ok(result) => result,
            Err(crate::Error::CudaUnavailable | crate::Error::CudaRuntime { .. })
                if !cuda_runtime_required() =>
            {
                return;
            }
            Err(error) => panic!("batch CUDA decode failed: {error}"),
        };

        assert_eq!(surfaces.len(), 2);
        assert_eq!(report.detail.ht_dispatch_count, 1);
        assert_eq!(report.detail.dequant_dispatch_count, 0);
        assert_eq!(report.detail.store_dispatch_count, 1);
        let batch_pixels_tight =
            crate::Surface::download_batch_tight(&surfaces).expect("download tight CUDA batch");
        assert_eq!(batch_pixels_tight.len(), surfaces.len() * 16 * 16 * 3);
        for (index, codestream) in inputs.iter().enumerate() {
            let mut single_session = crate::CudaSession::default();
            let mut decoder = crate::J2kDecoder::new(codestream).expect("single decoder");
            let single = decoder
                .decode_to_device_with_session(PixelFormat::Rgb8, &mut single_session)
                .expect("single CUDA decode");
            let mut single_pixels = vec![0u8; 16 * 16 * 3];
            let mut batch_pixels = vec![0u8; 16 * 16 * 3];
            single
                .download_into(&mut single_pixels, 16 * 3)
                .expect("download single decode");
            surfaces[index]
                .download_into(&mut batch_pixels, 16 * 3)
                .expect("download batch decode");
            assert_eq!(batch_pixels, single_pixels);
            assert_eq!(
                &batch_pixels_tight[index * 16 * 16 * 3..(index + 1) * 16 * 16 * 3],
                single_pixels.as_slice()
            );
        }
    }

    #[test]
    fn cuda_batch_decode_mixed_idwt_shapes_avoids_fused_batch_store_without_idwt_batch() {
        let codestream_a = rgb8_htj2k_fixture(32, 32, 1, 7);
        let codestream_b = rgb8_htj2k_fixture(32, 32, 2, 19);
        let inputs = [codestream_a.as_slice(), codestream_b.as_slice()];
        let mut batch_session = crate::CudaSession::default();

        let result = crate::J2kDecoder::decode_batch_to_device_with_session(
            &inputs,
            PixelFormat::Rgb8,
            &mut batch_session,
        );
        let surfaces = match result {
            Ok(surfaces) => surfaces,
            Err(crate::Error::CudaUnavailable | crate::Error::CudaRuntime { .. })
                if !cuda_runtime_required() =>
            {
                return;
            }
            Err(crate::Error::UnsupportedCudaRequest { .. }) => return,
            Err(error) => panic!("mixed-shape batch CUDA decode failed: {error}"),
        };

        assert_eq!(surfaces.len(), inputs.len());
        for (index, codestream) in inputs.iter().enumerate() {
            let mut single_session = crate::CudaSession::default();
            let mut decoder = crate::J2kDecoder::new(codestream).expect("single decoder");
            let single = decoder
                .decode_to_device_with_session(PixelFormat::Rgb8, &mut single_session)
                .expect("single CUDA decode");
            let mut single_pixels = vec![0u8; 32 * 32 * 3];
            let mut batch_pixels = vec![0u8; 32 * 32 * 3];
            single
                .download_into(&mut single_pixels, 32 * 3)
                .expect("download single decode");
            surfaces[index]
                .download_into(&mut batch_pixels, 32 * 3)
                .expect("download mixed-shape batch decode");
            assert_eq!(batch_pixels, single_pixels);
        }
    }

    #[test]
    fn decode_stage_timings_report_status_download_detail() {
        let mut report = crate::CudaHtj2kProfileReport::default();
        let timings = CudaDecodeStageTimings {
            h2d: 17,
            status_d2h: 5,
            ..CudaDecodeStageTimings::default()
        };

        timings.add_to_report(&mut report);

        assert_eq!(report.h2d_us, 17);
        assert_eq!(report.detail.status_d2h_us, 5);
    }

    fn cuda_runtime_required() -> bool {
        std::env::var_os("SIGNINUM_REQUIRE_CUDA_RUNTIME").is_some()
    }

    fn rgb8_htj2k_fixture(width: u32, height: u32, levels: u8, seed: u16) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for idx in 0..width * height {
            let seed = u32::from(seed);
            pixels.push(u8::try_from((idx * seed + idx / 3) & 0xff).expect("red"));
            pixels.push(u8::try_from((idx * (seed + 11) + 7) & 0xff).expect("green"));
            pixels.push(u8::try_from((idx * (seed + 23) + 19) & 0xff).expect("blue"));
        }
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: levels,
            ..EncodeOptions::default()
        };
        encode_htj2k(&pixels, width, height, 3, 8, false, &options)
            .expect("encode RGB HTJ2K fixture")
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

    #[test]
    fn byte_color_plan_builder_matches_decoder_color_plan() {
        let pixels: Vec<u8> = (0u16..8 * 8 * 3)
            .map(|idx| u8::try_from((idx * 19 + idx / 5) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream =
            encode_htj2k(&pixels, 8, 8, 3, 8, false, &options).expect("encode HTJ2K RGB fixture");
        let mut decoder = crate::J2kDecoder::new(&codestream).expect("decoder");
        let decoder_plan = decoder
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("decoder CUDA color plans");
        let mut native_context = NativeDecoderContext::default();
        let byte_plan = build_cuda_htj2k_color_plans_from_bytes_with_profile(
            &codestream,
            PixelFormat::Rgb8,
            &mut native_context,
        )
        .expect("byte CUDA color plans");

        assert_eq!(byte_plan.dimensions, decoder_plan.dimensions);
        assert_eq!(byte_plan.mct_dimensions, decoder_plan.mct_dimensions);
        assert_eq!(byte_plan.bit_depths, decoder_plan.bit_depths);
        assert_eq!(byte_plan.mct, decoder_plan.mct);
        assert_eq!(byte_plan.components.len(), decoder_plan.components.len());
        assert_eq!(byte_plan.payload.len(), decoder_plan.payload.len());
        assert_eq!(
            byte_plan
                .components
                .iter()
                .map(|component| component.code_blocks().len())
                .collect::<Vec<_>>(),
            decoder_plan
                .components
                .iter()
                .map(|component| component.code_blocks().len())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multi_image_color_components_can_share_one_idwt_batch() {
        let pixels: Vec<u8> = (0u16..16 * 16 * 3)
            .map(|idx| u8::try_from((idx * 17 + idx / 7) & 0xff).expect("masked byte"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        let codestream =
            encode_htj2k(&pixels, 16, 16, 3, 8, false, &options).expect("encode HTJ2K RGB fixture");
        let mut first = crate::J2kDecoder::new(&codestream).expect("first decoder");
        let mut second = crate::J2kDecoder::new(&codestream).expect("second decoder");
        let first = first
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("first CUDA color plans");
        let second = second
            .build_cuda_htj2k_color_plans_with_profile(PixelFormat::Rgb8)
            .expect("second CUDA color plans");
        let components = first
            .components
            .iter()
            .chain(second.components.iter())
            .collect::<Vec<_>>();

        assert_eq!(components.len(), 6);
        assert!(can_batch_color_idwt(&components));
    }

    #[test]
    fn batched_color_idwt_defers_completion_to_store_sync() {
        let source = include_str!("decoder.rs");

        assert!(
            !source.contains(
                "if !collect_stage_timings {\n        context.synchronize().map_err(cuda_error)?;\n    }"
            ),
            "batched color IDWT should keep queued resources live and let the following store synchronize"
        );
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
    use super::{
        format_cuda_idwt_batch_host_trace_row, htj2k_batched_dequant_dispatches,
        split_htj2k_subband_decode_dispatches, CudaIdwtBatchHostTraceRow,
    };

    #[test]
    fn htj2k_decode_dispatch_split_separates_ht_and_dequant_counts() {
        assert_eq!(split_htj2k_subband_decode_dispatches(0), (0, 0));
        assert_eq!(split_htj2k_subband_decode_dispatches(1), (1, 0));
        assert_eq!(split_htj2k_subband_decode_dispatches(2), (1, 1));
        assert_eq!(split_htj2k_subband_decode_dispatches(3), (2, 1));
    }

    #[test]
    fn htj2k_batched_dequant_dispatch_count_is_one_for_any_non_empty_batch() {
        assert_eq!(htj2k_batched_dequant_dispatches(0), 0);
        assert_eq!(htj2k_batched_dequant_dispatches(1), 1);
        assert_eq!(htj2k_batched_dequant_dispatches(48), 1);
    }

    #[test]
    fn cuda_idwt_batch_host_trace_row_reports_host_split() {
        let row = CudaIdwtBatchHostTraceRow {
            component_count: 327,
            step_count: 5,
            output_alloc_us: 11,
            target_build_us: 22,
            enqueue_us: 33,
            output_take_count: 1635,
            output_pool_reuse_count: 1600,
            output_pool_alloc_count: 35,
            output_pool_scanned_count: 2400,
            output_pool_max_free_count: 1700,
            output_requested_bytes: 28,
        };

        assert_eq!(
            format_cuda_idwt_batch_host_trace_row(row),
            "signinum_profile codec=j2k op=cuda_idwt_batch_host path=decode component_count=327 step_count=5 output_alloc_us=11 target_build_us=22 enqueue_us=33 output_take_count=1635 output_pool_reuse_count=1600 output_pool_alloc_count=35 output_pool_scanned_count=2400 output_pool_max_free_count=1700 output_requested_bytes=28"
        );
    }
}
