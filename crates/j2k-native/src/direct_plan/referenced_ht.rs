// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{J2kDirectColorPlan, J2kDirectGrayscalePlan, J2kDirectRgbaPlan};
use crate::{HtCodeBlockPayloadRanges, J2kRect, J2kWaveletTransform};

/// Contiguous range of compressed-payload records belonging to one tile plan.
///
/// The indices address entries in the parent referenced plan's `payloads()`
/// slice. They are record indices, not byte offsets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct J2kReferencedPayloadRecordSpan {
    /// Index of the first payload record for the tile.
    pub first_record: usize,
    /// Number of payload records for the tile.
    pub record_count: usize,
}

impl J2kReferencedPayloadRecordSpan {
    /// Exclusive payload-record index, or `None` when the span overflows.
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
///
/// Tile plans retain a separate coefficient/IDWT/store sequence so an
/// executor can release tile-local coefficient buffers at the Store boundary.
#[derive(Debug)]
pub struct J2kReferencedTilePlan {
    tile_index: usize,
    decoded_rect: J2kRect,
    destination_rect: J2kRect,
    payload_records: J2kReferencedPayloadRecordSpan,
    wavelet_transform: J2kWaveletTransform,
    geometry: J2kReferencedTileGeometry,
}

impl J2kReferencedTilePlan {
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

    /// Payload-record span in the parent plan's `payloads()` slice.
    #[must_use]
    pub const fn payload_records(&self) -> J2kReferencedPayloadRecordSpan {
        self.payload_records
    }

    /// Grayscale geometry, when this tile is grayscale.
    #[must_use]
    pub const fn grayscale_geometry(&self) -> Option<&J2kDirectGrayscalePlan> {
        match &self.geometry {
            J2kReferencedTileGeometry::Grayscale(geometry) => Some(geometry),
            J2kReferencedTileGeometry::Color(_) | J2kReferencedTileGeometry::Rgba(_) => None,
        }
    }

    /// RGB geometry, when this tile is color.
    #[must_use]
    pub const fn color_geometry(&self) -> Option<&J2kDirectColorPlan> {
        match &self.geometry {
            J2kReferencedTileGeometry::Color(geometry) => Some(geometry),
            J2kReferencedTileGeometry::Grayscale(_) | J2kReferencedTileGeometry::Rgba(_) => None,
        }
    }

    /// RGBA geometry, when this tile has four components.
    #[must_use]
    pub const fn rgba_geometry(&self) -> Option<&J2kDirectRgbaPlan> {
        match &self.geometry {
            J2kReferencedTileGeometry::Rgba(geometry) => Some(geometry),
            J2kReferencedTileGeometry::Grayscale(_) | J2kReferencedTileGeometry::Color(_) => None,
        }
    }

    /// Effective wavelet transform after component and tile coding-style overrides.
    #[must_use]
    pub const fn wavelet_transform(&self) -> J2kWaveletTransform {
        self.wavelet_transform
    }

    pub(crate) const fn new(
        tile_index: usize,
        decoded_rect: J2kRect,
        destination_rect: J2kRect,
        payload_records: J2kReferencedPayloadRecordSpan,
        wavelet_transform: J2kWaveletTransform,
        geometry: J2kReferencedTileGeometry,
    ) -> Self {
        Self {
            tile_index,
            decoded_rect,
            destination_rect,
            payload_records,
            wavelet_transform,
            geometry,
        }
    }
}

/// Owned HTJ2K execution geometry whose compressed payloads remain referenced
/// by offset in the caller-retained encoded input.
#[derive(Debug)]
pub enum J2kReferencedHtj2kPlan {
    /// One-component grayscale plan.
    Grayscale {
        /// Per-tile direct execution geometry in raster order.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions before an optional region is applied.
        full_dimensions: (u32, u32),
        /// Requested output rectangle in reduced full-image coordinates.
        output_rect: J2kRect,
        /// Payload ranges in HT-step/job traversal order.
        payloads: Vec<HtCodeBlockPayloadRanges>,
    },
    /// Three-component RGB plan.
    Color {
        /// Per-tile direct execution geometry in raster order.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions before an optional region is applied.
        full_dimensions: (u32, u32),
        /// Requested output rectangle in reduced full-image coordinates.
        output_rect: J2kRect,
        /// Payload ranges in component/HT-step/job traversal order.
        payloads: Vec<HtCodeBlockPayloadRanges>,
    },
    /// Four-component RGBA plan with explicit alpha semantics supplied by the caller.
    Rgba {
        /// Per-tile direct execution geometry in raster order.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions before an optional region is applied.
        full_dimensions: (u32, u32),
        /// Requested output rectangle in reduced full-image coordinates.
        output_rect: J2kRect,
        /// Payload ranges in R/G/B/A component, HT-step, and job traversal order.
        payloads: Vec<HtCodeBlockPayloadRanges>,
    },
}

impl J2kReferencedHtj2kPlan {
    /// Grayscale execution geometry for a legacy single-tile plan.
    ///
    /// Multi-tile callers must use [`Self::tiles`]; this accessor returns
    /// `None` instead of exposing only the first tile.
    #[must_use]
    pub fn grayscale_geometry(&self) -> Option<&J2kDirectGrayscalePlan> {
        match self {
            Self::Grayscale { tiles, .. } if tiles.len() == 1 => tiles[0].grayscale_geometry(),
            _ => None,
        }
    }

    /// RGB execution geometry for a legacy single-tile plan.
    #[must_use]
    pub fn color_geometry(&self) -> Option<&J2kDirectColorPlan> {
        match self {
            Self::Color { tiles, .. } if tiles.len() == 1 => tiles[0].color_geometry(),
            _ => None,
        }
    }

    /// RGBA execution geometry for a legacy single-tile plan.
    #[must_use]
    pub fn rgba_geometry(&self) -> Option<&J2kDirectRgbaPlan> {
        match self {
            Self::Rgba { tiles, .. } if tiles.len() == 1 => tiles[0].rgba_geometry(),
            _ => None,
        }
    }

    /// Per-tile direct execution plans in codestream raster order.
    #[must_use]
    pub fn tiles(&self) -> &[J2kReferencedTilePlan] {
        match self {
            Self::Grayscale { tiles, .. }
            | Self::Color { tiles, .. }
            | Self::Rgba { tiles, .. } => tiles,
        }
    }

    /// Reduced full-image dimensions before an optional output region.
    #[must_use]
    pub const fn full_dimensions(&self) -> (u32, u32) {
        match self {
            Self::Grayscale {
                full_dimensions, ..
            }
            | Self::Color {
                full_dimensions, ..
            }
            | Self::Rgba {
                full_dimensions, ..
            } => *full_dimensions,
        }
    }

    /// Requested output rectangle in reduced full-image coordinates.
    #[must_use]
    pub const fn output_rect(&self) -> J2kRect {
        match self {
            Self::Grayscale { output_rect, .. }
            | Self::Color { output_rect, .. }
            | Self::Rgba { output_rect, .. } => *output_rect,
        }
    }

    /// Referenced payload ranges in geometry traversal order.
    #[must_use]
    pub fn payloads(&self) -> &[HtCodeBlockPayloadRanges] {
        match self {
            Self::Grayscale { payloads, .. }
            | Self::Color { payloads, .. }
            | Self::Rgba { payloads, .. } => payloads,
        }
    }

    pub(crate) fn grayscale(
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

    pub(crate) fn color(
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

    pub(crate) fn rgba(
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
