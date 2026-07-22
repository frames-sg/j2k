// SPDX-License-Identifier: MIT OR Apache-2.0

//! Referenced classic JPEG 2000 plans without duplicated compressed payloads.

use alloc::vec::Vec;

use super::{
    J2kClassicCodeBlockPayload, J2kCodestreamRange, J2kDirectColorPlan, J2kDirectGrayscalePlan,
    J2kDirectRgbaPlan, J2kReferencedTilePlan,
};
use crate::J2kRect;

/// Owned classic execution geometry whose compressed fragments remain ranges
/// into the caller-retained encoded input.
#[derive(Debug)]
pub enum J2kReferencedClassicPlan {
    /// One-component grayscale plan.
    Grayscale {
        /// Per-tile direct execution geometry in raster order.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions before an optional region is applied.
        full_dimensions: (u32, u32),
        /// Requested output rectangle in reduced full-image coordinates.
        output_rect: J2kRect,
        /// Code-block payload descriptors in classic step/job traversal order.
        payloads: Vec<J2kClassicCodeBlockPayload>,
        /// Ordered encoded-input byte ranges referenced by `payloads`.
        ranges: Vec<J2kCodestreamRange>,
    },
    /// Three-component RGB plan.
    Color {
        /// Per-tile direct execution geometry in raster order.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions before an optional region is applied.
        full_dimensions: (u32, u32),
        /// Requested output rectangle in reduced full-image coordinates.
        output_rect: J2kRect,
        /// Code-block payload descriptors in component/step/job traversal order.
        payloads: Vec<J2kClassicCodeBlockPayload>,
        /// Ordered encoded-input byte ranges referenced by `payloads`.
        ranges: Vec<J2kCodestreamRange>,
    },
    /// Four-component RGBA plan in semantic R, G, B, A order.
    Rgba {
        /// Per-tile direct execution geometry in raster order.
        tiles: Vec<J2kReferencedTilePlan>,
        /// Reduced full-image dimensions before an optional region is applied.
        full_dimensions: (u32, u32),
        /// Requested output rectangle in reduced full-image coordinates.
        output_rect: J2kRect,
        /// Code-block payload descriptors in R/G/B/A component order.
        payloads: Vec<J2kClassicCodeBlockPayload>,
        /// Ordered encoded-input byte ranges referenced by `payloads`.
        ranges: Vec<J2kCodestreamRange>,
    },
}

impl J2kReferencedClassicPlan {
    /// Grayscale execution geometry for a legacy single-tile plan.
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

    /// Code-block payload descriptors in geometry traversal order.
    #[must_use]
    pub fn payloads(&self) -> &[J2kClassicCodeBlockPayload] {
        match self {
            Self::Grayscale { payloads, .. }
            | Self::Color { payloads, .. }
            | Self::Rgba { payloads, .. } => payloads,
        }
    }

    /// Encoded-input fragment ranges referenced by [`Self::payloads`].
    #[must_use]
    pub fn ranges(&self) -> &[J2kCodestreamRange] {
        match self {
            Self::Grayscale { ranges, .. }
            | Self::Color { ranges, .. }
            | Self::Rgba { ranges, .. } => ranges,
        }
    }

    pub(crate) fn grayscale(
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

    pub(crate) fn color(
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

    pub(crate) fn rgba(
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
