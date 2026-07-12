// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use crate::error::J2kError;
use j2k_core::{Downscale, Rect};

/// Device decode shape requested by a GPU adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceDecodeRequest {
    /// Decode the full image at full resolution.
    Full,
    /// Decode a full-resolution source region.
    Region {
        /// Source region of interest.
        roi: Rect,
    },
    /// Decode the full image at reduced resolution.
    Scaled {
        /// Requested downscale factor.
        scale: Downscale,
    },
    /// Decode a source region at reduced resolution.
    RegionScaled {
        /// Source region of interest.
        roi: Rect,
        /// Requested downscale factor.
        scale: Downscale,
    },
}

/// Normalized device decode plan derived from source dimensions and request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceDecodePlan {
    source_dims: (u32, u32),
    source_rect: Rect,
    scale: Downscale,
    output_rect: Rect,
}

impl DeviceDecodePlan {
    /// Build a normalized plan for an image.
    pub fn for_image(
        source_dims: (u32, u32),
        request: DeviceDecodeRequest,
    ) -> Result<Self, J2kError> {
        let (source_rect, scale) = match request {
            DeviceDecodeRequest::Full => (Rect::full(source_dims), Downscale::None),
            DeviceDecodeRequest::Region { roi } => (roi, Downscale::None),
            DeviceDecodeRequest::Scaled { scale } => (Rect::full(source_dims), scale),
            DeviceDecodeRequest::RegionScaled { roi, scale } => (roi, scale),
        };

        if !source_rect.is_within(source_dims) {
            return Err(J2kError::InvalidRegion {
                x: source_rect.x,
                y: source_rect.y,
                w: source_rect.w,
                h: source_rect.h,
                image_w: source_dims.0,
                image_h: source_dims.1,
            });
        }

        Ok(Self {
            source_dims,
            source_rect,
            scale,
            output_rect: source_rect.scaled_covering(scale),
        })
    }

    /// Original image dimensions.
    pub fn source_dims(self) -> (u32, u32) {
        self.source_dims
    }

    /// Full-resolution source rectangle to read.
    pub fn source_rect(self) -> Rect {
        self.source_rect
    }

    /// Requested downscale factor.
    pub fn scale(self) -> Downscale {
        self.scale
    }

    /// Output rectangle in reduced-resolution coordinates.
    pub fn output_rect(self) -> Rect {
        self.output_rect
    }

    /// Output dimensions in pixels.
    pub fn output_dims(self) -> (u32, u32) {
        (self.output_rect.w, self.output_rect.h)
    }

    /// Target resolution hint for native decoders that accept one.
    pub fn target_resolution(self) -> Option<(u32, u32)> {
        (self.scale != Downscale::None).then_some(self.output_dims())
    }

    /// Return true when the request is an unscaled full-frame decode.
    pub fn is_full_frame(self) -> bool {
        self.source_rect == Rect::full(self.source_dims) && self.scale == Downscale::None
    }
}
