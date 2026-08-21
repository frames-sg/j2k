// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{
    J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectRgbaPlan, J2kRect, J2kWaveletTransform,
};
use crate::{HtCodeBlockPayloadRanges, J2kClassicCodeBlockPayload, J2kCodestreamRange};

/// Contiguous range of compressed-payload records belonging to one tile plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kReferencedPayloadRecordSpan {
    /// Index of the first payload record.
    pub first_record: usize,
    /// Number of payload records.
    pub record_count: usize,
}

impl J2kReferencedPayloadRecordSpan {
    /// Exclusive payload-record index, or `None` on overflow.
    #[must_use]
    pub const fn end_record(self) -> Option<usize> {
        self.first_record.checked_add(self.record_count)
    }
}

/// Direct execution geometry for one codestream tile.
#[derive(Debug)]
pub enum J2kReferencedTileGeometry {
    /// One-component grayscale tile.
    Grayscale(J2kDirectGrayscalePlan),
    /// Three-component RGB tile.
    Color(J2kDirectColorPlan),
    /// Four-component RGBA tile.
    Rgba(J2kDirectRgbaPlan),
}

/// One independently executable tile in a referenced direct plan.
#[derive(Debug)]
pub struct J2kReferencedTilePlan {
    tile_index: usize,
    decoded_rect: J2kRect,
    destination_rect: J2kRect,
    payload_records: J2kReferencedPayloadRecordSpan,
    pub(super) classic_payloads: Vec<J2kClassicCodeBlockPayload>,
    pub(super) classic_ranges: Vec<J2kCodestreamRange>,
    wavelet_transform: J2kWaveletTransform,
    geometry: J2kReferencedTileGeometry,
}

impl J2kReferencedTilePlan {
    /// Construct one producer-validated tile plan.
    #[expect(
        clippy::too_many_arguments,
        reason = "tile geometry, payload ownership, and transform facts remain explicit"
    )]
    #[must_use]
    pub fn new(
        tile_index: usize,
        decoded_rect: J2kRect,
        destination_rect: J2kRect,
        payload_records: J2kReferencedPayloadRecordSpan,
        classic_payloads: Vec<J2kClassicCodeBlockPayload>,
        classic_ranges: Vec<J2kCodestreamRange>,
        wavelet_transform: J2kWaveletTransform,
        geometry: J2kReferencedTileGeometry,
    ) -> Self {
        Self {
            tile_index,
            decoded_rect,
            destination_rect,
            payload_records,
            classic_payloads,
            classic_ranges,
            wavelet_transform,
            geometry,
        }
    }

    /// Zero-based codestream tile index in raster order.
    #[must_use]
    pub const fn tile_index(&self) -> usize {
        self.tile_index
    }

    /// Tile/output-region intersection in reduced full-image coordinates.
    #[must_use]
    pub const fn decoded_rect(&self) -> J2kRect {
        self.decoded_rect
    }

    /// Tile/output-region intersection in dense destination coordinates.
    #[must_use]
    pub const fn destination_rect(&self) -> J2kRect {
        self.destination_rect
    }

    /// Payload-record span in the parent plan.
    #[must_use]
    pub const fn payload_records(&self) -> J2kReferencedPayloadRecordSpan {
        self.payload_records
    }

    /// Classic payload descriptors retained by a mixed tile.
    #[must_use]
    pub fn classic_payloads(&self) -> &[J2kClassicCodeBlockPayload] {
        &self.classic_payloads
    }

    /// Encoded-input ranges retained by a mixed tile.
    #[must_use]
    pub fn classic_ranges(&self) -> &[J2kCodestreamRange] {
        &self.classic_ranges
    }

    /// Grayscale geometry when present.
    #[must_use]
    pub const fn grayscale_geometry(&self) -> Option<&J2kDirectGrayscalePlan> {
        match &self.geometry {
            J2kReferencedTileGeometry::Grayscale(value) => Some(value),
            J2kReferencedTileGeometry::Color(_) | J2kReferencedTileGeometry::Rgba(_) => None,
        }
    }

    /// RGB geometry when present.
    #[must_use]
    pub const fn color_geometry(&self) -> Option<&J2kDirectColorPlan> {
        match &self.geometry {
            J2kReferencedTileGeometry::Color(value) => Some(value),
            J2kReferencedTileGeometry::Grayscale(_) | J2kReferencedTileGeometry::Rgba(_) => None,
        }
    }

    /// RGBA geometry when present.
    #[must_use]
    pub const fn rgba_geometry(&self) -> Option<&J2kDirectRgbaPlan> {
        match &self.geometry {
            J2kReferencedTileGeometry::Rgba(value) => Some(value),
            J2kReferencedTileGeometry::Grayscale(_) | J2kReferencedTileGeometry::Color(_) => None,
        }
    }

    /// Effective wavelet transform after coding-style overrides.
    #[must_use]
    pub const fn wavelet_transform(&self) -> J2kWaveletTransform {
        self.wavelet_transform
    }
}

/// Borrowed image-level geometry shared by classic and HTJ2K plans.
#[derive(Debug, Clone, Copy)]
pub struct J2kReferencedImageGeometry<'a> {
    tiles: &'a [J2kReferencedTilePlan],
    full_dimensions: (u32, u32),
    output_rect: J2kRect,
}

impl<'a> J2kReferencedImageGeometry<'a> {
    const fn new(
        tiles: &'a [J2kReferencedTilePlan],
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
    ) -> Self {
        Self {
            tiles,
            full_dimensions,
            output_rect,
        }
    }

    /// Per-tile direct execution plans in raster order.
    #[must_use]
    pub const fn tiles(self) -> &'a [J2kReferencedTilePlan] {
        self.tiles
    }

    /// Whether no tile geometry is present.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.tiles.is_empty()
    }

    /// Reduced full-image dimensions before an optional region.
    #[must_use]
    pub const fn full_dimensions(self) -> (u32, u32) {
        self.full_dimensions
    }

    /// Requested output rectangle in reduced full-image coordinates.
    #[must_use]
    pub const fn output_rect(self) -> J2kRect {
        self.output_rect
    }

    /// Whether every tile is grayscale.
    #[must_use]
    pub fn is_grayscale(self) -> bool {
        !self.is_empty()
            && self
                .tiles
                .iter()
                .all(|tile| tile.grayscale_geometry().is_some())
    }

    /// Whether every tile is RGB.
    #[must_use]
    pub fn is_color(self) -> bool {
        !self.is_empty()
            && self
                .tiles
                .iter()
                .all(|tile| tile.color_geometry().is_some())
    }

    /// Whether every tile is RGBA.
    #[must_use]
    pub fn is_rgba(self) -> bool {
        !self.is_empty() && self.tiles.iter().all(|tile| tile.rgba_geometry().is_some())
    }

    /// Grayscale geometry for a legacy single-tile plan.
    #[must_use]
    pub fn grayscale_geometry(self) -> Option<&'a J2kDirectGrayscalePlan> {
        (self.tiles.len() == 1)
            .then(|| self.tiles[0].grayscale_geometry())
            .flatten()
    }

    /// RGB geometry for a legacy single-tile plan.
    #[must_use]
    pub fn color_geometry(self) -> Option<&'a J2kDirectColorPlan> {
        (self.tiles.len() == 1)
            .then(|| self.tiles[0].color_geometry())
            .flatten()
    }

    /// RGBA geometry for a legacy single-tile plan.
    #[must_use]
    pub fn rgba_geometry(self) -> Option<&'a J2kDirectRgbaPlan> {
        (self.tiles.len() == 1)
            .then(|| self.tiles[0].rgba_geometry())
            .flatten()
    }

    /// Common wavelet transform when all tiles agree.
    #[must_use]
    pub fn uniform_wavelet_transform(self) -> Option<J2kWaveletTransform> {
        let first = self.tiles.first()?.wavelet_transform();
        self.tiles
            .iter()
            .all(|tile| tile.wavelet_transform() == first)
            .then_some(first)
    }
}

/// Owned classic JPEG 2000 execution geometry referencing caller-owned bytes.
#[derive(Debug)]
pub enum J2kReferencedClassicPlan {
    /// One-component grayscale plan.
    Grayscale {
        /// Per-tile geometry.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions.
        full_dimensions: (u32, u32),
        /// Requested output rectangle.
        output_rect: J2kRect,
        /// Payload descriptors.
        payloads: Vec<J2kClassicCodeBlockPayload>,
        /// Ordered encoded-input ranges referenced by the payload descriptors.
        ranges: Vec<J2kCodestreamRange>,
    },
    /// Three-component RGB plan.
    Color {
        /// Per-tile geometry.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions.
        full_dimensions: (u32, u32),
        /// Requested output rectangle.
        output_rect: J2kRect,
        /// Payload descriptors.
        payloads: Vec<J2kClassicCodeBlockPayload>,
        /// Ordered encoded-input ranges referenced by the payload descriptors.
        ranges: Vec<J2kCodestreamRange>,
    },
    /// Four-component RGBA plan.
    Rgba {
        /// Per-tile geometry.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions.
        full_dimensions: (u32, u32),
        /// Requested output rectangle.
        output_rect: J2kRect,
        /// Payload descriptors.
        payloads: Vec<J2kClassicCodeBlockPayload>,
        /// Ordered encoded-input ranges referenced by the payload descriptors.
        ranges: Vec<J2kCodestreamRange>,
    },
}

/// Owned HTJ2K execution geometry referencing caller-owned bytes.
#[derive(Debug)]
pub enum J2kReferencedHtj2kPlan {
    /// One-component grayscale plan.
    Grayscale {
        /// Per-tile geometry.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions.
        full_dimensions: (u32, u32),
        /// Requested output rectangle.
        output_rect: J2kRect,
        /// Payload records.
        payloads: Vec<HtCodeBlockPayloadRanges>,
    },
    /// Three-component RGB plan.
    Color {
        /// Per-tile geometry.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions.
        full_dimensions: (u32, u32),
        /// Requested output rectangle.
        output_rect: J2kRect,
        /// Payload records.
        payloads: Vec<HtCodeBlockPayloadRanges>,
    },
    /// Four-component RGBA plan.
    Rgba {
        /// Per-tile geometry.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions.
        full_dimensions: (u32, u32),
        /// Requested output rectangle.
        output_rect: J2kRect,
        /// Payload records.
        payloads: Vec<HtCodeBlockPayloadRanges>,
    },
}

macro_rules! shared_plan_methods {
    ($name:ident) => {
        impl $name {
            /// Grayscale geometry for a legacy single-tile plan.
            #[must_use]
            pub fn grayscale_geometry(&self) -> Option<&J2kDirectGrayscalePlan> {
                self.image_geometry().grayscale_geometry()
            }

            /// RGB geometry for a legacy single-tile plan.
            #[must_use]
            pub fn color_geometry(&self) -> Option<&J2kDirectColorPlan> {
                self.image_geometry().color_geometry()
            }

            /// RGBA geometry for a legacy single-tile plan.
            #[must_use]
            pub fn rgba_geometry(&self) -> Option<&J2kDirectRgbaPlan> {
                self.image_geometry().rgba_geometry()
            }

            /// Shared image-level geometry.
            #[must_use]
            pub const fn image_geometry(&self) -> J2kReferencedImageGeometry<'_> {
                match self {
                    Self::Grayscale {
                        tiles,
                        full_dimensions,
                        output_rect,
                        ..
                    }
                    | Self::Color {
                        tiles,
                        full_dimensions,
                        output_rect,
                        ..
                    }
                    | Self::Rgba {
                        tiles,
                        full_dimensions,
                        output_rect,
                        ..
                    } => J2kReferencedImageGeometry::new(
                        tiles.as_slice(),
                        *full_dimensions,
                        *output_rect,
                    ),
                }
            }

            /// Per-tile execution plans.
            #[must_use]
            pub fn tiles(&self) -> &[J2kReferencedTilePlan] {
                self.image_geometry().tiles()
            }

            /// Reduced full-image dimensions.
            #[must_use]
            pub const fn full_dimensions(&self) -> (u32, u32) {
                self.image_geometry().full_dimensions()
            }

            /// Requested output rectangle.
            #[must_use]
            pub const fn output_rect(&self) -> J2kRect {
                self.image_geometry().output_rect()
            }
        }
    };
}

shared_plan_methods!(J2kReferencedClassicPlan);
shared_plan_methods!(J2kReferencedHtj2kPlan);

impl J2kReferencedClassicPlan {
    /// Payload descriptors in geometry traversal order.
    #[must_use]
    pub fn payloads(&self) -> &[J2kClassicCodeBlockPayload] {
        match self {
            Self::Grayscale { payloads, .. }
            | Self::Color { payloads, .. }
            | Self::Rgba { payloads, .. } => payloads,
        }
    }

    /// Encoded-input ranges referenced by [`Self::payloads`].
    #[must_use]
    pub fn ranges(&self) -> &[J2kCodestreamRange] {
        match self {
            Self::Grayscale { ranges, .. }
            | Self::Color { ranges, .. }
            | Self::Rgba { ranges, .. } => ranges,
        }
    }

    /// Construct a producer-validated grayscale plan.
    #[must_use]
    pub fn grayscale(
        tiles: Vec<J2kReferencedTilePlan>,
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
        payloads: Vec<J2kClassicCodeBlockPayload>,
        ranges: Vec<J2kCodestreamRange>,
    ) -> Self {
        Self::Grayscale {
            tiles,
            full_dimensions,
            output_rect,
            payloads,
            ranges,
        }
    }

    /// Construct a producer-validated RGB plan.
    #[must_use]
    pub fn color(
        tiles: Vec<J2kReferencedTilePlan>,
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
        payloads: Vec<J2kClassicCodeBlockPayload>,
        ranges: Vec<J2kCodestreamRange>,
    ) -> Self {
        Self::Color {
            tiles,
            full_dimensions,
            output_rect,
            payloads,
            ranges,
        }
    }

    /// Construct a producer-validated RGBA plan.
    #[must_use]
    pub fn rgba(
        tiles: Vec<J2kReferencedTilePlan>,
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
        payloads: Vec<J2kClassicCodeBlockPayload>,
        ranges: Vec<J2kCodestreamRange>,
    ) -> Self {
        Self::Rgba {
            tiles,
            full_dimensions,
            output_rect,
            payloads,
            ranges,
        }
    }
}

impl J2kReferencedHtj2kPlan {
    /// Payload records in geometry traversal order.
    #[must_use]
    pub fn payloads(&self) -> &[HtCodeBlockPayloadRanges] {
        match self {
            Self::Grayscale { payloads, .. }
            | Self::Color { payloads, .. }
            | Self::Rgba { payloads, .. } => payloads,
        }
    }

    /// Construct a producer-validated grayscale plan.
    #[must_use]
    pub fn grayscale(
        tiles: Vec<J2kReferencedTilePlan>,
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
        payloads: Vec<HtCodeBlockPayloadRanges>,
    ) -> Self {
        Self::Grayscale {
            tiles,
            full_dimensions,
            output_rect,
            payloads,
        }
    }

    /// Construct a producer-validated RGB plan.
    #[must_use]
    pub fn color(
        tiles: Vec<J2kReferencedTilePlan>,
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
        payloads: Vec<HtCodeBlockPayloadRanges>,
    ) -> Self {
        Self::Color {
            tiles,
            full_dimensions,
            output_rect,
            payloads,
        }
    }

    /// Construct a producer-validated RGBA plan.
    #[must_use]
    pub fn rgba(
        tiles: Vec<J2kReferencedTilePlan>,
        full_dimensions: (u32, u32),
        output_rect: J2kRect,
        payloads: Vec<HtCodeBlockPayloadRanges>,
    ) -> Self {
        Self::Rgba {
            tiles,
            full_dimensions,
            output_rect,
            payloads,
        }
    }
}
