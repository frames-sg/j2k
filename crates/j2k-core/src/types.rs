// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::scale::Downscale;

/// Color interpretation of decoded samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Colorspace {
    /// Single-channel grayscale.
    Grayscale,
    /// JPEG-style luma/chroma color.
    YCbCr,
    /// Red, green, blue color.
    Rgb,
    /// Cyan, magenta, yellow, black color.
    Cmyk,
    /// Luma/chroma plus black color.
    Ycck,
    /// Standard RGB color.
    SRgb,
    /// Standard grayscale color.
    SGray,
    /// Color described by an embedded ICC profile.
    IccTagged,
    /// JPEG 2000 reversible color transform.
    Rct,
    /// JPEG 2000 irreversible color transform.
    Ict,
}

/// Regular tile grid layout for a compressed image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileLayout {
    /// Width of one tile in pixels.
    pub tile_width: u32,
    /// Height of one tile in pixels.
    pub tile_height: u32,
    /// Number of tiles across the image.
    pub tiles_x: u32,
    /// Number of tiles down the image.
    pub tiles_y: u32,
}

/// Regular coded-unit grid layout for formats with independently coded units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodedUnitLayout {
    /// Width of one coded unit in pixels.
    pub unit_width: u32,
    /// Height of one coded unit in pixels.
    pub unit_height: u32,
    /// Number of coded units across the image.
    pub units_x: u32,
    /// Number of coded units down the image.
    pub units_y: u32,
}

impl CodedUnitLayout {
    /// Return the saturating total coded-unit count.
    #[must_use]
    pub const fn unit_count(&self) -> u32 {
        self.units_x.saturating_mul(self.units_y)
    }
}

/// Rectangle in source pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left coordinate.
    pub x: u32,
    /// Top coordinate.
    pub y: u32,
    /// Rectangle width.
    pub w: u32,
    /// Rectangle height.
    pub h: u32,
}

impl Rect {
    /// Return a rectangle covering the full dimensions.
    #[must_use]
    pub const fn full(dims: (u32, u32)) -> Self {
        Self {
            x: 0,
            y: 0,
            w: dims.0,
            h: dims.1,
        }
    }

    /// Return whether this rectangle is fully inside `dims`.
    #[must_use]
    pub fn is_within(&self, dims: (u32, u32)) -> bool {
        let (w, h) = dims;
        self.x.checked_add(self.w).is_some_and(|r| r <= w)
            && self.y.checked_add(self.h).is_some_and(|b| b <= h)
    }

    /// Return the smallest scaled rectangle that covers this source rectangle.
    #[must_use]
    pub fn scaled_covering(&self, scale: Downscale) -> Self {
        let denom = scale.denominator();
        let x_end = self.x.saturating_add(self.w);
        let y_end = self.y.saturating_add(self.h);
        let x0 = self.x / denom;
        let y0 = self.y / denom;
        let x1 = x_end.div_ceil(denom);
        let y1 = y_end.div_ceil(denom);
        Self {
            x: x0,
            y: y0,
            w: x1.saturating_sub(x0),
            h: y1.saturating_sub(y0),
        }
    }
}

/// Basic image metadata returned by inspect/parse operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// Image dimensions in pixels.
    pub dimensions: (u32, u32),
    /// Number of image components.
    pub components: u16,
    /// Color interpretation of the components.
    pub colorspace: Colorspace,
    /// Bits per component sample.
    pub bit_depth: u8,
    /// Optional compressed tile grid.
    pub tile_layout: Option<TileLayout>,
    /// Optional coded-unit grid.
    pub coded_unit_layout: Option<CodedUnitLayout>,
    /// Optional restart interval for formats that expose one.
    pub restart_interval: Option<u32>,
    /// Number of resolution levels available in the codestream.
    pub resolution_levels: u8,
}

/// Successful decode metadata plus non-fatal warnings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeOutcome<W> {
    /// Source/output rectangle actually decoded.
    pub decoded: Rect,
    /// Non-fatal warnings observed during decode.
    pub warnings: Vec<W>,
}

impl<W> DecodeOutcome<W> {
    /// Construct a decode outcome from the decoded rectangle and warnings.
    #[must_use]
    pub fn new(decoded: Rect, warnings: Vec<W>) -> Self {
        Self { decoded, warnings }
    }
}
