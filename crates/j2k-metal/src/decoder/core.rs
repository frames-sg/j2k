// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use std::sync::Arc;

use j2k::{
    DeviceDecodePlan, DeviceDecodeRequest, J2kDecoder as CpuDecoder,
    J2kScratchPool as CpuJ2kScratchPool, J2kView,
};
use j2k_core::BackendRequest;
#[cfg(target_os = "macos")]
use j2k_native::{
    DecoderContext as NativeDecoderContext, Image as NativeImage, J2kDirectColorPlan,
    J2kDirectGrayscalePlan,
};

use super::request::{
    surface_with_report, DecodeSurfaceWithReport, MetalDecodeOp, MetalDecodeRequest,
};
use super::surface::{allocate_cpu_surface, upload_surface_to_metal_with_device};
use crate::{routing, Error, MetalBackendSession, Surface};

/// JPEG 2000 decoder that can return host or Metal-resident surfaces.
pub struct J2kDecoder<'a> {
    pub(super) bytes: &'a [u8],
    pub(crate) inner: CpuDecoder<'a>,
    pub(super) pool: CpuJ2kScratchPool,
    #[cfg(target_os = "macos")]
    pub(super) native_image: Option<NativeImage<'a>>,
    #[cfg(target_os = "macos")]
    pub(super) native_context: NativeDecoderContext<'a>,
    #[cfg(target_os = "macos")]
    pub(super) native_direct_gray_plan: Option<Arc<J2kDirectGrayscalePlan>>,
    #[cfg(target_os = "macos")]
    pub(super) native_prepared_direct_gray_plan:
        Option<Arc<crate::compute::PreparedDirectGrayscalePlan>>,
    #[cfg(target_os = "macos")]
    pub(super) native_direct_color_plan: Option<Arc<J2kDirectColorPlan>>,
    #[cfg(target_os = "macos")]
    pub(super) native_prepared_direct_color_plan:
        Option<Arc<crate::compute::PreparedDirectColorPlan>>,
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
                    upload_surface_to_metal_with_device(&out, dims, request.fmt, session.device())
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
                    upload_surface_to_metal_with_device(&out, dims, request.fmt, session.device())
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
                    upload_surface_to_metal_with_device(&out, dims, request.fmt, session.device())
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
                    upload_surface_to_metal_with_device(&out, dims, request.fmt, session.device())
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (request, session);
            Err(Error::MetalUnavailable)
        }
    }
}
