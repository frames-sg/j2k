// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};

use crate::batch;

/// Geometry operation for a single JPEG Metal device decode request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    pub(crate) const fn batch_op(self) -> batch::BatchOp {
        match self {
            Self::Full => batch::BatchOp::Full,
            Self::Region(roi) => batch::BatchOp::Region(roi),
            Self::Scaled(scale) => batch::BatchOp::Scaled(scale),
            Self::RegionScaled { roi, scale } => batch::BatchOp::RegionScaled { roi, scale },
        }
    }
}

/// Single-image JPEG Metal device decode request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalDecodeRequest {
    /// Requested output pixel format.
    pub fmt: PixelFormat,
    /// Decode geometry operation.
    pub op: MetalDecodeOp,
    /// Backend routing preference.
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
