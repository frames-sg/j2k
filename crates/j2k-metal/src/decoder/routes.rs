// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::DeviceDecodePlan;
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

use super::surface::{allocate_cpu_surface, upload_surface};
use super::J2kDecoder;
use super::{MetalDecodeOp, MetalDecodeRequest};
use crate::{routing, Error, MetalBackendSession, Surface};

impl J2kDecoder<'_> {
    pub(super) fn decode_to_cpu_surface(&mut self, fmt: PixelFormat) -> Result<Surface, Error> {
        let dims = self.inner.info().dimensions;
        let (mut out, stride) = allocate_cpu_surface(dims, fmt)?;
        self.inner
            .decode_into_with_scratch(&mut self.pool, &mut out, stride, fmt)?;
        upload_surface(out, dims, fmt, BackendRequest::Cpu)
    }

    pub(super) fn decode_region_to_cpu_surface(
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

    pub(super) fn decode_scaled_to_cpu_surface(
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

    pub(super) fn decode_region_scaled_to_cpu_surface(
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
    fn decode_region_scaled_to_metal_surface(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<Surface, Error> {
        if let Some(surface) =
            crate::hybrid::decode_region_scaled_direct_to_surface(self.bytes, fmt, roi, scale)?
        {
            return Ok(surface);
        }
        crate::engine::decode_region_scaled_to_surface(self.bytes, fmt, roi, scale)
    }

    #[cfg(target_os = "macos")]
    pub(super) fn decode_region_scaled_to_metal_surface_with_session(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        session: &MetalBackendSession,
    ) -> Result<Surface, Error> {
        if let Some(surface) = crate::hybrid::decode_region_scaled_direct_to_surface_with_session(
            self.bytes, fmt, roi, scale, session,
        )? {
            return Ok(surface);
        }
        crate::engine::with_runtime_for_session(session, |_| {
            crate::engine::decode_region_scaled_to_surface_with_device(
                self.bytes,
                fmt,
                roi,
                scale,
                session.device_handle(),
            )
        })
    }

    pub(crate) fn decode_op_to_surface_impl(
        &mut self,
        request: MetalDecodeRequest,
    ) -> Result<Surface, Error> {
        self.decode_op_to_surface_impl_with_session(request, None)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "backend selection, supplied-session execution and permitted fallback stay in one route decision"
    )]
    pub(super) fn decode_op_to_surface_impl_with_session(
        &mut self,
        request: MetalDecodeRequest,
        session: Option<&MetalBackendSession>,
    ) -> Result<Surface, Error> {
        #[cfg(not(target_os = "macos"))]
        let _ = session;
        let plan = DeviceDecodePlan::for_reference_grid(
            self.inner.reference_grid(),
            request.op.device_request(),
        )?;

        #[cfg(target_os = "macos")]
        let selected = match request.op {
            MetalDecodeOp::Full
                if request.backend == BackendRequest::Auto
                    && (request.fmt != PixelFormat::Rgb8
                        || self
                            .inner
                            .info()
                            .tile_layout
                            .is_some_and(|grid| grid.tiles_x == 1 && grid.tiles_y == 1))
                    && j2k::J2kDecoder::inspect_support(self.bytes).is_ok_and(|support| {
                        routing::auto_full_decode_uses_metal(
                            plan.output_dims(),
                            support.component_count(),
                            request.fmt,
                            support.transfer_syntax,
                            support.payload_kind,
                        )
                    }) =>
            {
                BackendRequest::Metal
            }
            MetalDecodeOp::Scaled(scale)
                if request.backend == BackendRequest::Auto
                    && j2k::J2kDecoder::inspect_support(self.bytes).is_ok_and(|support| {
                        routing::auto_scaled_decode_uses_metal(
                            plan.output_dims(),
                            support.component_count(),
                            request.fmt,
                            support.transfer_syntax,
                            support.payload_kind,
                            scale,
                        )
                    }) =>
            {
                BackendRequest::Metal
            }
            _ => request.backend,
        };
        #[cfg(not(target_os = "macos"))]
        let selected = request.backend;
        let route = routing::decide_route(selected, request.fmt);
        if let Some(error) = routing::decision_error(route) {
            return Err(error);
        }

        match (route, session) {
            (routing::RouteDecision::CpuHost, _) => match request.op {
                MetalDecodeOp::Full => self.decode_to_cpu_surface(request.fmt),
                MetalDecodeOp::Region(_) => self.decode_region_to_cpu_surface(request.fmt, plan),
                MetalDecodeOp::Scaled(scale) => {
                    self.decode_scaled_to_cpu_surface(request.fmt, scale, plan)
                }
                MetalDecodeOp::RegionScaled { scale, .. } => self
                    .decode_region_scaled_to_cpu_surface(
                        request.fmt,
                        plan.source_rect(),
                        scale,
                        plan,
                    ),
            },
            #[cfg(target_os = "macos")]
            (routing::RouteDecision::MetalKernel, Some(session)) => match request.op {
                MetalDecodeOp::Full => crate::engine::with_runtime_for_session(session, |_| {
                    self.decode_direct_to_surface_with_session(request.fmt, session)?
                        .map_or_else(
                            || {
                                self.decode_full_to_metal_surface_with_device(
                                    request.fmt,
                                    session.device_handle(),
                                )
                            },
                            Ok,
                        )
                }),
                MetalDecodeOp::Region(_) => self
                    .decode_region_scaled_to_metal_surface_with_session(
                        request.fmt,
                        plan.source_rect(),
                        Downscale::None,
                        session,
                    ),
                MetalDecodeOp::Scaled(scale) | MetalDecodeOp::RegionScaled { scale, .. } => self
                    .decode_region_scaled_to_metal_surface_with_session(
                        request.fmt,
                        plan.source_rect(),
                        scale,
                        session,
                    ),
            },
            #[cfg(target_os = "macos")]
            (routing::RouteDecision::MetalKernel, None) => match request.op {
                MetalDecodeOp::Full => self
                    .decode_direct_to_surface(request.fmt)?
                    .map_or_else(|| self.decode_full_to_metal_surface(request.fmt), Ok),
                MetalDecodeOp::Region(_) => self.decode_region_scaled_to_metal_surface(
                    request.fmt,
                    plan.source_rect(),
                    Downscale::None,
                ),
                MetalDecodeOp::Scaled(scale) | MetalDecodeOp::RegionScaled { scale, .. } => self
                    .decode_region_scaled_to_metal_surface(request.fmt, plan.source_rect(), scale),
            },
            (
                routing::RouteDecision::RejectExplicitMetal { .. }
                | routing::RouteDecision::RejectUnsupportedBackend { .. },
                _,
            ) => {
                unreachable!("handled by decision_error")
            }
            #[cfg(not(target_os = "macos"))]
            (routing::RouteDecision::MetalUnavailable, _) => {
                unreachable!("handled by decision_error")
            }
        }
    }
}
