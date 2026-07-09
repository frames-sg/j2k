// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    DeviceDecodePlan, DeviceDecodeRequest, J2kContext as CpuJ2kContext, J2kDecodeWarning,
    J2kDecoder as CpuDecoder, J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
use j2k_core::{
    checked_surface_len, BackendKind, BackendRequest, CpuBackedImageDecode, DecodeOutcome,
    Downscale, ImageCodec, ImageDecodeDevice, ImageDecodeSubmit, PixelFormat, ReadySubmission,
    Rect, TileBatchDecodeDevice, TileBatchDecodeManyDevice, TileBatchDecodeSubmit,
    TileRegionScaledDeviceDecodeRequest, DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};
#[cfg(target_os = "macos")]
use j2k_native::{
    DecodeSettings as NativeDecodeSettings, DecoderContext as NativeDecoderContext,
    Image as NativeImage, J2kDirectColorPlan, J2kDirectGrayscalePlan,
};
#[cfg(target_os = "macos")]
use metal::{Device, MTLResourceOptions};

#[cfg(target_os = "macos")]
use crate::direct;
#[cfg(target_os = "macos")]
use crate::error::{adapter_backend_error, native_decode_j2k_error};
#[cfg(all(test, target_os = "macos"))]
use crate::hybrid;
#[cfg(target_os = "macos")]
use crate::session::{
    cached_session_direct_color_plan, cached_session_direct_gray_plan, direct_gray_plan_cache_key,
    direct_plan_cache_key, store_session_direct_color_plan, store_session_direct_gray_plan,
};
use crate::{
    batch, routing, Error, MetalBackendSession, MetalDirectFallbackReason, MetalSession, Storage,
    Surface, SurfaceResidency,
};

macro_rules! define_ensure_prepared_direct_plan {
    (
        with_session: $with_session:ident,
        plain: $plain:ident,
        prepare_fresh: $prepare_fresh:ident,
        plan_field: $plan_field:ident,
        prepared_field: $prepared_field:ident,
        prepared_ty: $prepared_ty:path,
        cache_key: $cache_key:ident,
        cached: $cached:ident,
        store: $store:ident,
        build: $build:ident,
        prepare: $prepare:path,
        label: $label:literal
    ) => {
        #[cfg(target_os = "macos")]
        fn $with_session(
            &mut self,
            session: &MetalBackendSession,
        ) -> Result<Option<Arc<$prepared_ty>>, Error> {
            let cache_key = $cache_key(self.bytes);
            if self.$prepared_field.is_none() {
                if let Some((plan, prepared)) = $cached(session, cache_key) {
                    self.$plan_field = Some(plan);
                    self.$prepared_field = Some(prepared);
                }
            }
            self.$prepare_fresh(Some((session, cache_key)))
        }

        #[cfg(target_os = "macos")]
        fn $plain(&mut self) -> Result<Option<Arc<$prepared_ty>>, Error> {
            self.$prepare_fresh(None)
        }

        #[cfg(target_os = "macos")]
        fn $prepare_fresh(
            &mut self,
            session_cache: Option<(&MetalBackendSession, u64)>,
        ) -> Result<Option<Arc<$prepared_ty>>, Error> {
            if self.$prepared_field.is_none() {
                self.ensure_native_image()?;
                let (Some(image), native_context) =
                    (self.native_image.as_ref(), &mut self.native_context)
                else {
                    return Err(Error::Decode(adapter_backend_error(
                        "native image cache missing".to_string(),
                    )));
                };
                let plan = match image.$build(native_context) {
                    Ok(plan) => plan,
                    Err(error) if direct::is_unsupported_direct_plan_error(&error) => {
                        return Ok(None);
                    }
                    Err(error) => {
                        return Err(Error::Decode(adapter_backend_error(format!(
                            "failed to build J2K MetalDirect {} plan: {error}",
                            $label
                        ))));
                    }
                };
                let prepared = Arc::new($prepare(&plan)?);
                if let Some((session, cache_key)) = session_cache {
                    $store(session, cache_key, &plan, prepared.clone());
                }
                self.$plan_field = Some(plan);
                self.$prepared_field = Some(prepared);
            }
            Ok(self.$prepared_field.clone())
        }
    };
}

#[cfg(target_os = "macos")]
const AUTO_REPEATED_GRAYSCALE_MIN_DIM: u32 = 512;
#[cfg(target_os = "macos")]
const AUTO_REPEATED_GRAYSCALE_MIN_COUNT: usize = 16;

/// Decode operation represented in a route report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub enum DecodeOperation {
    /// Full-image decode.
    Full,
    /// Source-region decode.
    Region,
    /// Full-image scaled decode.
    Scaled,
    /// Source-region scaled decode.
    RegionScaled,
}

/// Geometry operation for a single J2K Metal decode request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalDecodeOp {
    /// Full-image decode at native dimensions.
    Full,
    /// Source-region decode at native scale.
    Region(Rect),
    /// Full-image downscale.
    Scaled(Downscale),
    /// Source-region decode with downscale.
    RegionScaled {
        /// Source region of interest.
        roi: Rect,
        /// Downscale factor applied to the selected region.
        scale: Downscale,
    },
}

impl MetalDecodeOp {
    pub(crate) const fn report_operation(self) -> DecodeOperation {
        match self {
            Self::Full => DecodeOperation::Full,
            Self::Region(_) => DecodeOperation::Region,
            Self::Scaled(_) => DecodeOperation::Scaled,
            Self::RegionScaled { .. } => DecodeOperation::RegionScaled,
        }
    }

    pub(crate) const fn batch_op(self) -> batch::BatchOp {
        match self {
            Self::Full => batch::BatchOp::Full,
            Self::Region(roi) => batch::BatchOp::Region(roi),
            Self::Scaled(scale) => batch::BatchOp::Scaled(scale),
            Self::RegionScaled { roi, scale } => batch::BatchOp::RegionScaled { roi, scale },
        }
    }
}

/// Single-image J2K Metal decode request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalDecodeRequest {
    /// Requested output pixel format.
    pub fmt: PixelFormat,
    /// Decode geometry operation.
    pub op: MetalDecodeOp,
    /// Backend routing preference for device decode APIs.
    pub backend: BackendRequest,
}

impl MetalDecodeRequest {
    /// Full-image decode request.
    pub const fn full(fmt: PixelFormat, backend: BackendRequest) -> Self {
        Self {
            fmt,
            op: MetalDecodeOp::Full,
            backend,
        }
    }

    /// Source-region decode request.
    pub const fn region(fmt: PixelFormat, roi: Rect, backend: BackendRequest) -> Self {
        Self {
            fmt,
            op: MetalDecodeOp::Region(roi),
            backend,
        }
    }

    /// Full-image downscale decode request.
    pub const fn scaled(fmt: PixelFormat, scale: Downscale, backend: BackendRequest) -> Self {
        Self {
            fmt,
            op: MetalDecodeOp::Scaled(scale),
            backend,
        }
    }

    /// Source-region downscale decode request.
    pub const fn region_scaled(
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Self {
        Self {
            fmt,
            op: MetalDecodeOp::RegionScaled { roi, scale },
            backend,
        }
    }
}

/// Route details for a completed decode request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[doc(hidden)]
pub struct DecodeRouteReport {
    /// Decode operation requested by the caller.
    pub operation: DecodeOperation,
    /// Caller backend preference.
    pub requested_backend: BackendRequest,
    /// Backend that produced the returned surface.
    pub selected_backend: BackendKind,
    /// Requested output pixel format.
    pub pixel_format: PixelFormat,
    /// Residency of the returned surface.
    pub surface_residency: SurfaceResidency,
    /// Reason `Auto` selected CPU, when applicable.
    pub fallback_reason: Option<&'static str>,
}

impl DecodeRouteReport {
    fn from_surface(
        operation: DecodeOperation,
        requested_backend: BackendRequest,
        pixel_format: PixelFormat,
        surface: &Surface,
    ) -> Self {
        Self {
            operation,
            requested_backend,
            selected_backend: surface.backend,
            pixel_format,
            surface_residency: surface.residency,
            fallback_reason: decode_fallback_reason(requested_backend, surface.backend),
        }
    }
}

/// Decoded surface paired with the route details that produced it.
#[derive(Clone)]
#[doc(hidden)]
pub struct DecodeSurfaceWithReport {
    /// Returned decoded surface.
    pub surface: Surface,
    /// Route report for the completed decode.
    pub report: DecodeRouteReport,
}

fn decode_fallback_reason(
    requested_backend: BackendRequest,
    selected_backend: BackendKind,
) -> Option<&'static str> {
    if requested_backend == BackendRequest::Auto && selected_backend == BackendKind::Cpu {
        Some(routing::AUTO_DECODE_CPU_FALLBACK_REASON)
    } else {
        None
    }
}

#[cfg(target_os = "macos")]
const CPU_STAGED_METAL_REQUIRES_EXPLICIT_API: &str =
    "CPU-staged Metal upload requires the explicit CPU-staged API; BackendRequest::Metal only accepts resident Metal decode";

/// JPEG 2000 decoder that can return host or Metal-resident surfaces.
pub struct J2kDecoder<'a> {
    bytes: &'a [u8],
    pub(crate) inner: CpuDecoder<'a>,
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
            bytes: input,
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
        let bytes = view.bytes();
        Ok(Self {
            bytes,
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

    /// Decode into a device surface using a request object instead of a
    /// geometry-specific method.
    pub fn decode_request_to_device(
        &mut self,
        request: MetalDecodeRequest,
    ) -> Result<Surface, Error> {
        match request.op {
            MetalDecodeOp::Full => self.decode_to_surface_impl(request.fmt, request.backend),
            MetalDecodeOp::Region(roi) => {
                self.decode_region_to_surface_impl(request.fmt, roi, request.backend)
            }
            MetalDecodeOp::Scaled(scale) => {
                self.decode_scaled_to_surface_impl(request.fmt, scale, request.backend)
            }
            MetalDecodeOp::RegionScaled { roi, scale } => {
                self.decode_region_scaled_to_surface_impl(request.fmt, roi, scale, request.backend)
            }
        }
    }

    /// Decode into a device surface and return route details.
    #[doc(hidden)]
    pub fn decode_request_to_device_with_report(
        &mut self,
        request: MetalDecodeRequest,
    ) -> Result<DecodeSurfaceWithReport, Error> {
        let surface = self.decode_request_to_device(request)?;
        Ok(surface_with_report(
            surface,
            request.op.report_operation(),
            request.backend,
            request.fmt,
        ))
    }

    /// Decode into a Metal-resident device surface using a reusable session.
    pub fn decode_request_to_device_with_session(
        &mut self,
        request: MetalDecodeRequest,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if request.backend != BackendRequest::Metal {
            return self.decode_request_to_device(request);
        }
        if let Some(error) =
            routing::decision_error(routing::decide_route(BackendRequest::Metal, request.fmt))
        {
            return Err(error);
        }

        #[cfg(target_os = "macos")]
        {
            match request.op {
                MetalDecodeOp::Full => crate::compute::with_runtime_for_session(session, |_| {
                    if let Some(surface) =
                        self.decode_direct_to_surface_with_session(request.fmt, session)?
                    {
                        Ok(surface)
                    } else {
                        self.decode_full_to_metal_surface_with_device(
                            request.fmt,
                            session.device_handle(),
                        )
                    }
                }),
                MetalDecodeOp::Region(roi) => {
                    let plan = DeviceDecodePlan::for_image(
                        self.inner.info().dimensions,
                        DeviceDecodeRequest::Region { roi },
                    )?;
                    crate::compute::with_runtime_for_session(session, |_| {
                        self.decode_region_to_metal_surface_with_device(
                            request.fmt,
                            plan,
                            session.device_handle(),
                        )
                    })
                }
                MetalDecodeOp::Scaled(scale) => {
                    let plan = DeviceDecodePlan::for_image(
                        self.inner.info().dimensions,
                        DeviceDecodeRequest::Scaled { scale },
                    )?;
                    crate::compute::with_runtime_for_session(session, |_| {
                        self.decode_scaled_to_metal_surface_with_device(
                            request.fmt,
                            scale,
                            plan,
                            session.device_handle(),
                        )
                    })
                }
                MetalDecodeOp::RegionScaled { roi, scale } => {
                    let plan = DeviceDecodePlan::for_image(
                        self.inner.info().dimensions,
                        DeviceDecodeRequest::RegionScaled { roi, scale },
                    )?;
                    self.decode_region_scaled_to_metal_surface_with_session(
                        request.fmt,
                        roi,
                        scale,
                        plan,
                        session,
                    )
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = session;
            Err(Error::MetalUnavailable)
        }
    }

    /// Decode into a host-backed surface using a request object.
    pub fn decode_request_to_host_surface(
        &mut self,
        request: MetalDecodeRequest,
    ) -> Result<Surface, Error> {
        match request.op {
            MetalDecodeOp::Full => self.decode_to_cpu_surface(request.fmt),
            MetalDecodeOp::Region(roi) => {
                let plan = DeviceDecodePlan::for_image(
                    self.inner.info().dimensions,
                    DeviceDecodeRequest::Region { roi },
                )?;
                self.decode_region_to_cpu_surface(request.fmt, plan)
            }
            MetalDecodeOp::Scaled(scale) => {
                let plan = DeviceDecodePlan::for_image(
                    self.inner.info().dimensions,
                    DeviceDecodeRequest::Scaled { scale },
                )?;
                self.decode_scaled_to_cpu_surface(request.fmt, scale, plan)
            }
            MetalDecodeOp::RegionScaled { roi, scale } => {
                let plan = DeviceDecodePlan::for_image(
                    self.inner.info().dimensions,
                    DeviceDecodeRequest::RegionScaled { roi, scale },
                )?;
                self.decode_region_scaled_to_cpu_surface(request.fmt, roi, scale, plan)
            }
        }
    }

    /// Decode on CPU and upload the result into a Metal surface using a request object.
    pub fn decode_request_to_cpu_staged_metal_surface_with_session(
        &mut self,
        request: MetalDecodeRequest,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        #[cfg(target_os = "macos")]
        {
            match request.op {
                MetalDecodeOp::Full => {
                    let dims = self.inner.info().dimensions;
                    let (mut out, stride) = allocate_cpu_surface(dims, request.fmt)?;
                    self.inner.decode_into_with_scratch(
                        &mut self.pool,
                        &mut out,
                        stride,
                        request.fmt,
                    )?;
                    Ok(upload_surface_to_metal_with_device(
                        &out,
                        dims,
                        request.fmt,
                        session.device(),
                    ))
                }
                MetalDecodeOp::Region(roi) => {
                    let plan = DeviceDecodePlan::for_image(
                        self.inner.info().dimensions,
                        DeviceDecodeRequest::Region { roi },
                    )?;
                    let dims = plan.output_dims();
                    let (mut out, stride) = allocate_cpu_surface(dims, request.fmt)?;
                    self.inner.decode_region_into(
                        &mut self.pool,
                        &mut out,
                        stride,
                        request.fmt,
                        plan.source_rect(),
                    )?;
                    Ok(upload_surface_to_metal_with_device(
                        &out,
                        dims,
                        request.fmt,
                        session.device(),
                    ))
                }
                MetalDecodeOp::Scaled(scale) => {
                    let plan = DeviceDecodePlan::for_image(
                        self.inner.info().dimensions,
                        DeviceDecodeRequest::Scaled { scale },
                    )?;
                    let dims = plan.output_dims();
                    let (mut out, stride) = allocate_cpu_surface(dims, request.fmt)?;
                    self.inner.decode_scaled_into(
                        &mut self.pool,
                        &mut out,
                        stride,
                        request.fmt,
                        scale,
                    )?;
                    Ok(upload_surface_to_metal_with_device(
                        &out,
                        dims,
                        request.fmt,
                        session.device(),
                    ))
                }
                MetalDecodeOp::RegionScaled { roi, scale } => {
                    let plan = DeviceDecodePlan::for_image(
                        self.inner.info().dimensions,
                        DeviceDecodeRequest::RegionScaled { roi, scale },
                    )?;
                    let dims = plan.output_dims();
                    let (mut out, stride) = allocate_cpu_surface(dims, request.fmt)?;
                    self.inner.decode_region_scaled_into(
                        &mut self.pool,
                        &mut out,
                        stride,
                        request.fmt,
                        roi,
                        scale,
                    )?;
                    Ok(upload_surface_to_metal_with_device(
                        &out,
                        dims,
                        request.fmt,
                        session.device(),
                    ))
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (request, session);
            Err(Error::MetalUnavailable)
        }
    }

    #[cfg(target_os = "macos")]
    fn ensure_native_image(&mut self) -> Result<(), Error> {
        if self.native_image.is_none() {
            self.native_image = Some(
                NativeImage::new(self.bytes, &NativeDecodeSettings::default())
                    .map_err(native_decode_j2k_error)?,
            );
        }
        Ok(())
    }

    define_ensure_prepared_direct_plan! {
        with_session: ensure_prepared_direct_gray_plan_with_session,
        plain: ensure_prepared_direct_gray_plan,
        prepare_fresh: prepare_fresh_direct_gray_plan,
        plan_field: native_direct_gray_plan,
        prepared_field: native_prepared_direct_gray_plan,
        prepared_ty: crate::compute::PreparedDirectGrayscalePlan,
        cache_key: direct_gray_plan_cache_key,
        cached: cached_session_direct_gray_plan,
        store: store_session_direct_gray_plan,
        build: build_direct_grayscale_plan_with_context,
        prepare: crate::compute::prepare_direct_grayscale_plan,
        label: "grayscale"
    }

    define_ensure_prepared_direct_plan! {
        with_session: ensure_prepared_direct_color_plan_with_session,
        plain: ensure_prepared_direct_color_plan,
        prepare_fresh: prepare_fresh_direct_color_plan,
        plan_field: native_direct_color_plan,
        prepared_field: native_prepared_direct_color_plan,
        prepared_ty: crate::compute::PreparedDirectColorPlan,
        cache_key: direct_plan_cache_key,
        cached: cached_session_direct_color_plan,
        store: store_session_direct_color_plan,
        build: build_direct_color_plan_with_context,
        prepare: crate::compute::prepare_direct_color_plan,
        label: "color"
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
                    session.device_handle(),
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
                session.device_handle(),
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
        crate::hybrid::decode_region_scaled_direct_to_surface(self.bytes, fmt, roi, scale)
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
            self.bytes, fmt, roi, scale, session,
        )
    }
    #[cfg(target_os = "macos")]
    fn decode_full_to_metal_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        self.ensure_native_image()?;
        let (Some(image), native_context) = (self.native_image.as_ref(), &mut self.native_context)
        else {
            return Err(Error::Decode(adapter_backend_error(
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
            return Err(Error::Decode(adapter_backend_error(
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
                return Err(Error::Decode(adapter_backend_error(
                    "native image cache missing".to_string(),
                )));
            };
            let plan = image
                .build_direct_grayscale_plan_with_context(native_context)
                .map_err(native_decode_j2k_error)?;
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
                return Err(Error::Decode(adapter_backend_error(
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
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
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
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
            return Err(Error::Decode(adapter_backend_error(
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
        crate::compute::decode_scaled_to_surface(self.bytes, plan.source_dims(), fmt, scale)
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
            self.bytes,
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
            return Err(Error::Decode(adapter_backend_error(
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
            self.bytes,
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
                self.bytes,
                plan.source_dims(),
                fmt,
                roi,
                scale,
                session.device_handle(),
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
}

#[cfg(target_os = "macos")]
fn is_direct_color_runtime_fallback_error(error: &Error) -> bool {
    is_direct_runtime_fallback_error(error)
}

#[cfg(target_os = "macos")]
pub(crate) fn is_direct_runtime_fallback_error(error: &Error) -> bool {
    error.is_direct_fallback()
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
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full grayscale Gray8/Gray16 only; fmt={fmt:?}"
                ),
                reason: MetalDirectFallbackReason::UnsupportedPlan,
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
            return Err(Error::MetalDirectFallback {
                message: format!(
                    "explicit J2K MetalDirect batch currently supports full RGB color only; fmt={fmt:?}"
                ),
                reason: MetalDirectFallbackReason::UnsupportedPlan,
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

fn allocate_cpu_surface(dims: (u32, u32), fmt: PixelFormat) -> Result<(Vec<u8>, usize), Error> {
    let (stride, len) = checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "j2k Metal CPU fallback surface",
    )?;
    Ok((vec![0u8; len], stride))
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
        Self::from_view(view)
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

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
/// J2K codec marker used by J2K's generic decode traits.
pub struct Codec;

#[doc(hidden)]
impl ImageCodec for Codec {
    type Error = Error;
    type Warning = J2kDecodeWarning;
    type Pool = crate::J2kScratchPool;
}

#[doc(hidden)]
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
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::full(fmt, backend),
        )))
    }

    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::region(fmt, roi, backend),
        )))
    }

    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        session.record_submit()?;
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::scaled(fmt, scale, backend),
        )))
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
        Ok(ReadySubmission::from_result(self.decode_request_to_device(
            MetalDecodeRequest::region_scaled(fmt, roi, scale, backend),
        )))
    }
}

#[doc(hidden)]
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
        let request = MetalDecodeRequest::full(fmt, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
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
        let request = MetalDecodeRequest::region(fmt, roi, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
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
        let request = MetalDecodeRequest::scaled(fmt, scale, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
    }

    fn submit_tile_region_scaled_to_device(
        ctx: &mut j2k_core::DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        request: TileRegionScaledDeviceDecodeRequest<'_>,
    ) -> Result<Self::SubmittedSurface, Self::Error> {
        let TileRegionScaledDeviceDecodeRequest {
            input,
            fmt,
            roi,
            scale,
            backend,
        } = request;
        let _ = (ctx, pool);
        let request = MetalDecodeRequest::region_scaled(fmt, roi, scale, backend);
        batch::queue_tile_request(
            session,
            input,
            request.fmt,
            request.backend,
            request.op.batch_op(),
        )
    }
}

#[doc(hidden)]
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

#[doc(hidden)]
impl TileBatchDecodeDevice for Codec {
    type Context = CpuJ2kContext;
    type DeviceSurface = Surface;
}

fn surface_with_report(
    surface: Surface,
    operation: DecodeOperation,
    requested_backend: BackendRequest,
    pixel_format: PixelFormat,
) -> DecodeSurfaceWithReport {
    let report =
        DecodeRouteReport::from_surface(operation, requested_backend, pixel_format, &surface);
    DecodeSurfaceWithReport { surface, report }
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

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_core::{CodecError, DeviceSurface};

    #[cfg(target_os = "macos")]
    fn should_run_metal_runtime() -> bool {
        j2k_test_support::metal_runtime_gate(module_path!())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn direct_runtime_fallback_classification_uses_structured_variant() {
        let fallback = Error::MetalDirectFallback {
            message: "arbitrary fallback text".to_string(),
            reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
        };
        assert!(is_direct_runtime_fallback_error(&fallback));

        let message_only = Error::MetalKernel {
            message: "unsupported classic kernel input in direct component plan".to_string(),
        };
        assert!(!is_direct_runtime_fallback_error(&message_only));
    }

    #[test]
    fn metal_decode_request_maps_geometry_to_report_and_batch_ops() {
        let roi = Rect {
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        };
        let requests = [
            (
                MetalDecodeRequest::full(PixelFormat::Gray8, BackendRequest::Auto),
                DecodeOperation::Full,
                batch::BatchOp::Full,
            ),
            (
                MetalDecodeRequest::region(PixelFormat::Gray8, roi, BackendRequest::Auto),
                DecodeOperation::Region,
                batch::BatchOp::Region(roi),
            ),
            (
                MetalDecodeRequest::scaled(
                    PixelFormat::Gray8,
                    Downscale::Half,
                    BackendRequest::Auto,
                ),
                DecodeOperation::Scaled,
                batch::BatchOp::Scaled(Downscale::Half),
            ),
            (
                MetalDecodeRequest::region_scaled(
                    PixelFormat::Gray8,
                    roi,
                    Downscale::Quarter,
                    BackendRequest::Auto,
                ),
                DecodeOperation::RegionScaled,
                batch::BatchOp::RegionScaled {
                    roi,
                    scale: Downscale::Quarter,
                },
            ),
        ];

        for (request, report_operation, batch_op) in requests {
            assert_eq!(request.op.report_operation(), report_operation);
            assert_eq!(request.op.batch_op(), batch_op);
        }
    }

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
        assert!(surface.metal_buffer_trusted().is_none());
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
        if !should_run_metal_runtime() {
            return;
        }

        let Some(device) = metal::Device::system_default() else {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
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
        if !should_run_metal_runtime() {
            return;
        }

        if Device::system_default().is_none() {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
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
        if !should_run_metal_runtime() {
            return;
        }

        if Device::system_default().is_none() {
            j2k_test_support::metal_device_unavailable_is_skip(module_path!());
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
