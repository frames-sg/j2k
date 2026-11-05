// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use crate::{error::J2kError, J2kReferenceGrid};
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
    image_origin: (u32, u32),
    source_rect: Rect,
    scale: Downscale,
    output_rect: Rect,
}

impl DeviceDecodePlan {
    /// Build a normalized plan for an image whose reference-grid origin is a
    /// multiple of the request's downscale denominator, such as `(0, 0)`.
    ///
    /// For any image, use [`Self::for_reference_grid`]: another origin moves
    /// the reduced grid against the image's own, which this plan does not see.
    pub fn for_image(
        source_dims: (u32, u32),
        request: DeviceDecodeRequest,
    ) -> Result<Self, J2kError> {
        Self::for_reference_grid(J2kReferenceGrid::at_origin(source_dims), request)
    }

    /// Build a normalized plan for an image placed on the reference grid as
    /// `grid` places it.
    ///
    /// A reduced output covers the reduced samples whose reference-grid
    /// footprints overlap the requested region: a reduction by `s` spans
    /// `[ceil(origin / s), ceil((origin + size) / s))` on each axis, so an
    /// origin that is not a multiple of `s` can make the output a sample
    /// narrower than `ceil(size / s)` and shifts where a region lands.
    pub fn for_reference_grid(
        grid: J2kReferenceGrid,
        request: DeviceDecodeRequest,
    ) -> Result<Self, J2kError> {
        let source_dims = grid.image_size;
        let (source_rect, scale) = match request {
            DeviceDecodeRequest::Full => (Rect::full(source_dims), Downscale::None),
            DeviceDecodeRequest::Region { roi } => (roi, Downscale::None),
            DeviceDecodeRequest::Scaled { scale } => (Rect::full(source_dims), scale),
            DeviceDecodeRequest::RegionScaled { roi, scale } => (roi, scale),
        };

        if source_rect.w == 0 || source_rect.h == 0 || !source_rect.is_within(source_dims) {
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
            image_origin: grid.image_origin,
            source_rect,
            scale,
            output_rect: reference_grid_covering(
                grid.image_origin,
                source_dims,
                source_rect,
                scale.denominator(),
            ),
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

    /// Dimensions of the whole image at this plan's scale: the reduced
    /// image a region of [`Self::output_rect`] lies in.
    pub fn reduced_image_dims(self) -> (u32, u32) {
        let whole = reference_grid_covering(
            self.image_origin,
            self.source_dims,
            Rect::full(self.source_dims),
            self.scale.denominator(),
        );
        (whole.w, whole.h)
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

/// The samples of an image's reduction by `denominator` whose reference-grid
/// footprints overlap region `rect` of the full-resolution image of `size` at
/// `origin`, in reduced image coordinates.
///
/// A reduction spans `[ceil(origin / s), ceil((origin + size) / s))` on each
/// axis of the reference grid (ITU-T T.800 B.5). When `origin` is a multiple
/// of `denominator` this is [`Rect::scaled_covering`].
pub(crate) fn reference_grid_covering(
    origin: (u32, u32),
    size: (u32, u32),
    rect: Rect,
    denominator: u32,
) -> Rect {
    let (x, w) = covering_axis(rect.x, rect.w, origin.0, size.0, denominator);
    let (y, h) = covering_axis(rect.y, rect.h, origin.1, size.1, denominator);
    Rect { x, y, w, h }
}

/// [`reference_grid_covering`] on one axis: the first reduced sample and the
/// count, within the reduced image's
/// `[ceil(origin / s), ceil((origin + size) / s))`.
fn covering_axis(start: u32, length: u32, origin: u32, size: u32, denominator: u32) -> (u32, u32) {
    let denominator = u64::from(denominator.max(1));
    let origin = u64::from(origin);
    let first = origin.div_ceil(denominator);
    let end = (origin + u64::from(size)).div_ceil(denominator);
    let start = origin + u64::from(start);
    let low = (start / denominator).clamp(first, end);
    let high = (start + u64::from(length))
        .div_ceil(denominator)
        .clamp(low, end);
    // Both lie within the reduced size, which is at most `size`.
    let offset = u32::try_from(low - first).unwrap_or(u32::MAX);
    let count = u32::try_from(high - low).unwrap_or(u32::MAX);
    (offset, count)
}
