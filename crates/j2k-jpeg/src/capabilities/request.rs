// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decode-operation and backend request contracts.

use crate::Rect;
use j2k_core::{BackendRequest, Downscale, PixelFormat};

/// JPEG decode operation shape for capability routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JpegDecodeOp {
    /// Full-image/tile decode.
    Full,
    /// Source-coordinate region decode.
    Region(Rect),
    /// Full-image/tile decode at reduced resolution.
    Scaled(Downscale),
    /// Source-coordinate region decode at reduced resolution.
    RegionScaled {
        /// Source-coordinate region.
        roi: Rect,
        /// Reduced-resolution factor.
        scale: Downscale,
    },
}

impl JpegDecodeOp {
    pub(super) fn scale(self) -> Downscale {
        match self {
            Self::Full | Self::Region(_) => Downscale::None,
            Self::Scaled(scale) | Self::RegionScaled { scale, .. } => scale,
        }
    }
}

/// Capability request for a JPEG decode route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegCapabilityRequest {
    /// Decode operation shape.
    pub op: JpegDecodeOp,
    /// Requested output pixel format.
    pub fmt: PixelFormat,
}

/// Complete JPEG decode request used by backend path resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JpegDecodeRequest {
    /// Requested backend policy.
    pub backend: BackendRequest,
    /// Requested output pixel format.
    pub fmt: PixelFormat,
    /// Decode operation shape.
    pub op: JpegDecodeOp,
}

impl JpegDecodeRequest {
    /// Return the capability-only portion of the request.
    #[must_use]
    pub const fn capability(self) -> JpegCapabilityRequest {
        JpegCapabilityRequest {
            op: self.op,
            fmt: self.fmt,
        }
    }
}
