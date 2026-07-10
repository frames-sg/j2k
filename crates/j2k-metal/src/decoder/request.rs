// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, BackendRequest, Downscale, PixelFormat, Rect};

use crate::{batch, routing, Surface, SurfaceResidency};

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

pub(super) fn surface_with_report(
    surface: Surface,
    operation: DecodeOperation,
    requested_backend: BackendRequest,
    pixel_format: PixelFormat,
) -> DecodeSurfaceWithReport {
    let report =
        DecodeRouteReport::from_surface(operation, requested_backend, pixel_format, &surface);
    DecodeSurfaceWithReport { surface, report }
}
