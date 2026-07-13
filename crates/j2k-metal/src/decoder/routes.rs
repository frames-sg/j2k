// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{DeviceDecodePlan, DeviceDecodeRequest};
use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
#[cfg(target_os = "macos")]
use metal::Device;

use super::surface::{allocate_cpu_surface, upload_surface};
use super::J2kDecoder;
#[cfg(target_os = "macos")]
use crate::error::adapter_backend_error;
#[cfg(target_os = "macos")]
use crate::MetalBackendSession;
use crate::{routing, Error, Surface};

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
    pub(super) fn decode_region_to_metal_surface_with_device(
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
    pub(super) fn decode_scaled_to_metal_surface_with_device(
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
    pub(super) fn decode_region_scaled_to_metal_surface_with_session(
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
