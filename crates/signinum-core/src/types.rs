// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;

use crate::scale::Downscale;

/// Image colorspace reported by codec inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Colorspace {
    /// One-channel grayscale.
    Grayscale,
    /// Luma/chroma JPEG YCbCr.
    YCbCr,
    /// Red/green/blue.
    Rgb,
    /// Cyan/magenta/yellow/key.
    Cmyk,
    /// YCbCr plus key channel.
    Ycck,
    /// Standard RGB.
    SRgb,
    /// Standard grayscale.
    SGray,
    /// ICC-tagged color data.
    IccTagged,
    /// JPEG 2000 reversible color transform.
    Rct,
    /// JPEG 2000 irreversible color transform.
    Ict,
}

/// Regular tile grid layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileLayout {
    /// Nominal tile width in pixels.
    pub tile_width: u32,
    /// Nominal tile height in pixels.
    pub tile_height: u32,
    /// Number of tiles horizontally.
    pub tiles_x: u32,
    /// Number of tiles vertically.
    pub tiles_y: u32,
}

/// Regular coded-unit grid layout such as JPEG MCU layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CodedUnitLayout {
    /// Coded-unit width in pixels.
    pub unit_width: u32,
    /// Coded-unit height in pixels.
    pub unit_height: u32,
    /// Number of coded units horizontally.
    pub units_x: u32,
    /// Number of coded units vertically.
    pub units_y: u32,
}

impl CodedUnitLayout {
    /// Return the saturated product of horizontal and vertical unit counts.
    pub const fn unit_count(&self) -> u32 {
        self.units_x.saturating_mul(self.units_y)
    }
}

/// Pixel rectangle in source-image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rect {
    /// Left coordinate.
    pub x: u32,
    /// Top coordinate.
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl Rect {
    /// Return a rectangle covering the full image dimensions.
    pub const fn full(dims: (u32, u32)) -> Self {
        Self {
            x: 0,
            y: 0,
            w: dims.0,
            h: dims.1,
        }
    }

    /// Return true when the rectangle lies fully inside `dims`.
    pub fn is_within(&self, dims: (u32, u32)) -> bool {
        let (w, h) = dims;
        self.x.checked_add(self.w).is_some_and(|r| r <= w)
            && self.y.checked_add(self.h).is_some_and(|b| b <= h)
    }

    /// Return the reduced-resolution rectangle that fully covers this source
    /// rectangle at the requested scale.
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

/// Header metadata common to image codecs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// Image dimensions in pixels.
    pub dimensions: (u32, u32),
    /// Number of color components.
    pub components: u8,
    /// Reported colorspace.
    pub colorspace: Colorspace,
    /// Bits per component sample.
    pub bit_depth: u8,
    /// Optional tile grid layout.
    pub tile_layout: Option<TileLayout>,
    /// Optional coded-unit grid layout.
    pub coded_unit_layout: Option<CodedUnitLayout>,
    /// Optional restart interval in coded units.
    pub restart_interval: Option<u32>,
    /// Number of decoded resolution levels available.
    pub resolution_levels: u8,
}

/// Coarse category for non-fatal decode warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum WarningKind {
    /// Input violated a minor compliance rule but decoding continued.
    MinorCompliance,
    /// Input ended early but enough data was recovered for output.
    NonFatalTruncation,
    /// Input used an unusual but accepted feature.
    UnusualFeature,
}

/// Successful decode metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOutcome<W> {
    /// Source rectangle represented by the output.
    pub decoded: Rect,
    /// Non-fatal warnings emitted while decoding.
    pub warnings: Vec<W>,
}

impl<W> DecodeOutcome<W> {
    /// Create decode outcome metadata.
    pub fn new(decoded: Rect, warnings: Vec<W>) -> Self {
        Self { decoded, warnings }
    }
}
