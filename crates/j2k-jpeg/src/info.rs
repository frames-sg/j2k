// SPDX-License-Identifier: MIT OR Apache-2.0

//! Image metadata and primitive value types. See spec Sections 2 and 4.
//!
//! `info.rs` intentionally has **no dependency on `error.rs`** — `error`
//! depends on us (for `Rect` and `SofKind`), and the reverse would create a
//! cycle. `DecodeOutcome`, which does need `Warning`, lives in `decoder.rs`
//! and is added in M1b when the decode methods are introduced.

use alloc::vec::Vec;

/// Start-of-frame variant. Determines the decode pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SofKind {
    /// SOF0: baseline sequential, 8-bit, Huffman.
    Baseline8,
    /// SOF1: extended sequential, 8-bit, Huffman.
    Extended8,
    /// SOF1: extended sequential, 12-bit, Huffman.
    Extended12,
    /// SOF2: progressive, 8-bit, Huffman.
    Progressive8,
    /// SOF2: progressive, 12-bit, Huffman.
    Progressive12,
    /// SOF3: lossless (Annex H predictor), Huffman.
    Lossless,
}

/// Declared input color space after APP14 detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// Single-component grayscale.
    Grayscale,
    /// Three-component luma/chroma JPEG data.
    YCbCr,
    /// Three-component RGB JPEG data.
    Rgb,
    /// Four-component CMYK JPEG data.
    Cmyk,
    /// Four-component YCCK JPEG data.
    Ycck,
}

/// Per-component (H, V) sampling factors, stored in declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingFactors {
    components: [(u8, u8); 4],
    component_count: u8,
    /// `max(H_i)` across components — MCU width in data units.
    pub max_h: u8,
    /// `max(V_i)` across components — MCU height in data units.
    pub max_v: u8,
}

/// Error returned when constructing [`SamplingFactors`] from caller input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SamplingFactorsError {
    /// At least one component is required.
    #[error("sampling metadata must contain at least one component")]
    Empty,
    /// This crate stores at most four component sampling entries.
    #[error("sampling metadata supports at most four components, got {count}")]
    TooManyComponents {
        /// Supplied component count.
        count: usize,
    },
    /// Component sampling factors are outside the JPEG legal range.
    #[error("invalid sampling ({h}x{v}) for component {component}")]
    InvalidSampling {
        /// Component index in declaration order.
        component: usize,
        /// Horizontal sampling factor.
        h: u8,
        /// Vertical sampling factor.
        v: u8,
    },
}

impl SamplingFactors {
    /// Build sampling metadata from component `(H, V)` factors.
    ///
    /// # Errors
    /// Returns [`SamplingFactorsError`] when no components are supplied, more
    /// than four components are supplied, or any sampling factor is outside
    /// the JPEG legal range `1..=4`.
    pub fn from_components(components: &[(u8, u8)]) -> Result<Self, SamplingFactorsError> {
        if components.is_empty() {
            return Err(SamplingFactorsError::Empty);
        }
        if components.len() > 4 {
            return Err(SamplingFactorsError::TooManyComponents {
                count: components.len(),
            });
        }
        for (idx, &(h, v)) in components.iter().enumerate() {
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                return Err(SamplingFactorsError::InvalidSampling {
                    component: idx,
                    h,
                    v,
                });
            }
        }
        Ok(Self::from_validated_components(components))
    }

    pub(crate) fn from_validated_components(components: &[(u8, u8)]) -> Self {
        debug_assert!(!components.is_empty());
        debug_assert!(components.len() <= 4);
        debug_assert!(components
            .iter()
            .all(|&(h, v)| (1..=4).contains(&h) && (1..=4).contains(&v)));
        let mut packed = [(0u8, 0u8); 4];
        let mut max_h = 0u8;
        let mut max_v = 0u8;
        for (idx, &(h, v)) in components.iter().enumerate() {
            packed[idx] = (h, v);
            max_h = max_h.max(h);
            max_v = max_v.max(v);
        }
        Self {
            components: packed,
            component_count: components.len() as u8,
            max_h,
            max_v,
        }
    }

    /// Number of declared components.
    pub fn len(&self) -> usize {
        self.component_count as usize
    }

    /// True when no components were declared.
    pub fn is_empty(&self) -> bool {
        self.component_count == 0
    }

    /// Sampling factors for a component by declaration index.
    pub fn component(&self, index: usize) -> Option<(u8, u8)> {
        self.components().get(index).copied()
    }

    /// Sampling factors in component declaration order.
    pub fn components(&self) -> &[(u8, u8)] {
        &self.components[..self.component_count as usize]
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = (u8, u8)> + '_ {
        self.components().iter().copied()
    }
}

/// Minimum coded unit geometry derived from SOF sampling factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McuGeometry {
    /// MCU width in output pixels.
    pub width: u32,
    /// MCU height in output pixels.
    pub height: u32,
    /// Number of MCU columns covering the image.
    pub columns: u32,
    /// Number of MCU rows covering the image.
    pub rows: u32,
    /// Total MCU count in scan order.
    pub count: u32,
}

/// Restart-marker index for a single-scan JPEG stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartIndex {
    /// Absolute byte offset of the first entropy byte after the SOS header.
    pub scan_data_offset: usize,
    /// Restart interval from DRI, in MCUs.
    pub interval_mcus: u32,
    /// Restart-addressable scan segments in MCU order.
    pub segments: Vec<RestartSegment>,
}

/// One restart-addressable entropy segment in the original JPEG byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestartSegment {
    /// First MCU index decoded from this segment.
    pub start_mcu: u32,
    /// Absolute byte offset of the first entropy byte for this segment.
    pub entropy_offset: usize,
    /// Absolute byte offset of the preceding RST marker's leading `0xff`.
    pub marker_offset: Option<usize>,
    /// Preceding marker byte (`0xd0..=0xd7`) for this segment.
    pub marker: Option<u8>,
}

impl McuGeometry {
    pub(crate) fn from_sampling(dimensions: (u32, u32), sampling: SamplingFactors) -> Self {
        let width = u32::from(sampling.max_h) * 8;
        let height = u32::from(sampling.max_v) * 8;
        let columns = dimensions.0.div_ceil(width);
        let rows = dimensions.1.div_ceil(height);
        Self {
            width,
            height,
            columns,
            rows,
            count: columns.saturating_mul(rows),
        }
    }
}

/// Inclusive axis-aligned rectangle in image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left coordinate in pixels.
    pub x: u32,
    /// Top coordinate in pixels.
    pub y: u32,
    /// Width in pixels.
    pub w: u32,
    /// Height in pixels.
    pub h: u32,
}

impl Rect {
    /// The full image rect for the given dimensions.
    pub fn full(dims: (u32, u32)) -> Self {
        Self {
            x: 0,
            y: 0,
            w: dims.0,
            h: dims.1,
        }
    }

    /// True if the rect is fully inside the bounding box of size `dims`.
    pub fn is_within(&self, dims: (u32, u32)) -> bool {
        let (w, h) = dims;
        self.x.checked_add(self.w).is_some_and(|r| r <= w)
            && self.y.checked_add(self.h).is_some_and(|b| b <= h)
    }
}

impl From<j2k_core::Rect> for Rect {
    fn from(rect: j2k_core::Rect) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            w: rect.w,
            h: rect.h,
        }
    }
}

impl From<Rect> for j2k_core::Rect {
    fn from(rect: Rect) -> Self {
        Self {
            x: rect.x,
            y: rect.y,
            w: rect.w,
            h: rect.h,
        }
    }
}

/// Internal JPEG-specific output format used behind the public core
/// `PixelFormat` + `Downscale` API adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    Rgb8,
    Rgb8Scaled { factor: DownscaleFactor },
    Rgba8 { alpha: u8 },
    Rgba8Scaled { alpha: u8, factor: DownscaleFactor },
    Gray8,
    Gray8Scaled { factor: DownscaleFactor },
    Gray16,
    Gray16Scaled { factor: DownscaleFactor },
    Rgb16,
    Rgb16Scaled { factor: DownscaleFactor },
    Rgba16 { alpha: u16 },
    Rgba16Scaled { alpha: u16, factor: DownscaleFactor },
}

impl OutputFormat {
    pub(crate) fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Rgb8 | Self::Rgb8Scaled { .. } => 3,
            Self::Rgba8 { .. } | Self::Rgba8Scaled { .. } => 4,
            Self::Gray8 | Self::Gray8Scaled { .. } => 1,
            Self::Gray16 | Self::Gray16Scaled { .. } => 2,
            Self::Rgb16 | Self::Rgb16Scaled { .. } => 6,
            Self::Rgba16 { .. } | Self::Rgba16Scaled { .. } => 8,
        }
    }

    pub(crate) fn downscale(self) -> DownscaleFactor {
        match self {
            Self::Rgb8
            | Self::Rgba8 { .. }
            | Self::Gray8
            | Self::Gray16
            | Self::Rgb16
            | Self::Rgba16 { .. } => DownscaleFactor::Full,
            Self::Rgb8Scaled { factor }
            | Self::Rgba8Scaled { factor, .. }
            | Self::Gray8Scaled { factor }
            | Self::Gray16Scaled { factor }
            | Self::Rgb16Scaled { factor }
            | Self::Rgba16Scaled { factor, .. } => factor,
        }
    }
}

/// IDCT-level downscale factor; applies only to DCT-based SOFs (see spec
/// Section 4 matrix).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DownscaleFactor {
    Full,
    Half,
    Quarter,
    Eighth,
}

impl DownscaleFactor {
    pub(crate) const fn denominator(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }

    pub(crate) const fn output_block_size(self) -> u32 {
        match self {
            Self::Full => 8,
            Self::Half => 4,
            Self::Quarter => 2,
            Self::Eighth => 1,
        }
    }
}

/// Override for APP14 color-transform detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorTransform {
    /// Detect the transform from APP14 metadata and component layout.
    Auto,
    /// Treat three-component data as RGB regardless of APP14 metadata.
    ForceRgb,
    /// Treat three-component data as YCbCr regardless of APP14 metadata.
    ForceYCbCr,
}

/// Public decode options for JPEG reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeOptions {
    color_transform: ColorTransform,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            color_transform: ColorTransform::Auto,
        }
    }
}

impl DecodeOptions {
    /// Override APP14 color-transform detection.
    pub fn set_color_transform(&mut self, color_transform: ColorTransform) {
        self.color_transform = color_transform;
    }

    /// Current color-transform override.
    pub fn color_transform(&self) -> ColorTransform {
        self.color_transform
    }

    /// Builder-style color-transform override.
    #[must_use]
    pub fn with_color_transform(mut self, color_transform: ColorTransform) -> Self {
        self.set_color_transform(color_transform);
        self
    }

    pub(crate) fn apply_to_info(self, info: &mut Info) {
        match (self.color_transform, info.sampling.len()) {
            (ColorTransform::Auto, _) => {}
            (ColorTransform::ForceRgb, 3) => info.color_space = ColorSpace::Rgb,
            (ColorTransform::ForceYCbCr, 3) => info.color_space = ColorSpace::YCbCr,
            (ColorTransform::ForceRgb | ColorTransform::ForceYCbCr, _) => {}
        }
    }
}

/// Header-derived image metadata. Populated by `Decoder::inspect` and by
/// `Decoder::new`. `scan_count` is the number of SOS markers observed in
/// the input — for sequential this is always 1; for progressive it is the
/// count of refinement passes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Info {
    /// Image dimensions as `(width, height)` in pixels.
    pub dimensions: (u32, u32),
    /// Header-derived color space after APP14 transform handling.
    pub color_space: ColorSpace,
    /// Per-component sampling factors from the SOF marker.
    pub sampling: SamplingFactors,
    /// Start-of-frame variant that selects the decode pipeline.
    pub sof_kind: SofKind,
    /// Sample precision in bits.
    pub bit_depth: u8,
    /// Restart interval in MCUs, if a DRI marker was present.
    pub restart_interval: Option<u16>,
    /// Derived MCU geometry for the image.
    pub mcu_geometry: McuGeometry,
    /// Number of SOS markers observed in the stream.
    pub scan_count: u16,
}

impl Info {
    /// Convert JPEG metadata into the codec-neutral core metadata type.
    pub fn to_core_info(&self) -> j2k_core::Info {
        j2k_core::Info {
            dimensions: self.dimensions,
            components: self.sampling.len() as u16,
            colorspace: core_colorspace(self.color_space),
            bit_depth: self.bit_depth,
            tile_layout: None,
            coded_unit_layout: Some(j2k_core::CodedUnitLayout {
                unit_width: self.mcu_geometry.width,
                unit_height: self.mcu_geometry.height,
                units_x: self.mcu_geometry.columns,
                units_y: self.mcu_geometry.rows,
            }),
            restart_interval: self.restart_interval.map(u32::from),
            resolution_levels: 1,
        }
    }
}

fn core_colorspace(color_space: ColorSpace) -> j2k_core::Colorspace {
    match color_space {
        ColorSpace::Grayscale => j2k_core::Colorspace::Grayscale,
        ColorSpace::YCbCr => j2k_core::Colorspace::YCbCr,
        ColorSpace::Rgb => j2k_core::Colorspace::Rgb,
        ColorSpace::Cmyk => j2k_core::Colorspace::Cmyk,
        ColorSpace::Ycck => j2k_core::Colorspace::Ycck,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_full_matches_dimensions() {
        let r = Rect::full((1024, 768));
        assert_eq!(
            r,
            Rect {
                x: 0,
                y: 0,
                w: 1024,
                h: 768
            }
        );
    }

    #[test]
    fn rect_is_within_accepts_contained_rect() {
        assert!(Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100
        }
        .is_within((100, 100)));
        assert!(Rect {
            x: 10,
            y: 20,
            w: 30,
            h: 40
        }
        .is_within((100, 100)));
    }

    #[test]
    fn rect_is_within_rejects_overflowing_rect() {
        assert!(!Rect {
            x: 50,
            y: 50,
            w: 60,
            h: 10
        }
        .is_within((100, 100)));
        assert!(!Rect {
            x: u32::MAX,
            y: 0,
            w: 1,
            h: 1
        }
        .is_within((100, 100)));
    }

    #[test]
    fn output_format_bytes_per_pixel_matches_spec() {
        assert_eq!(OutputFormat::Rgb8.bytes_per_pixel(), 3);
        assert_eq!(
            OutputFormat::Rgb8Scaled {
                factor: DownscaleFactor::Quarter
            }
            .bytes_per_pixel(),
            3
        );
        assert_eq!(OutputFormat::Rgba8 { alpha: 255 }.bytes_per_pixel(), 4);
        assert_eq!(
            OutputFormat::Rgba8Scaled {
                alpha: 255,
                factor: DownscaleFactor::Half,
            }
            .bytes_per_pixel(),
            4
        );
        assert_eq!(OutputFormat::Gray8.bytes_per_pixel(), 1);
        assert_eq!(
            OutputFormat::Gray8Scaled {
                factor: DownscaleFactor::Half
            }
            .bytes_per_pixel(),
            1
        );
        assert_eq!(OutputFormat::Gray16.bytes_per_pixel(), 2);
        assert_eq!(
            OutputFormat::Gray16Scaled {
                factor: DownscaleFactor::Half
            }
            .bytes_per_pixel(),
            2
        );
        assert_eq!(OutputFormat::Rgb16.bytes_per_pixel(), 6);
        assert_eq!(
            OutputFormat::Rgb16Scaled {
                factor: DownscaleFactor::Half
            }
            .bytes_per_pixel(),
            6
        );
        assert_eq!(
            OutputFormat::Rgba16 { alpha: u16::MAX }.bytes_per_pixel(),
            8
        );
        assert_eq!(
            OutputFormat::Rgba16Scaled {
                alpha: u16::MAX,
                factor: DownscaleFactor::Half
            }
            .bytes_per_pixel(),
            8
        );
    }

    #[test]
    fn sampling_factors_store_components_without_heap_state() {
        let sampling =
            SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)]).expect("sampling");
        assert_eq!(sampling.len(), 3);
        assert_eq!(sampling.component(0), Some((2, 2)));
        assert_eq!(sampling.component(1), Some((1, 1)));
        assert_eq!(sampling.component(3), None);
        assert_eq!(sampling.components(), &[(2, 2), (1, 1), (1, 1)]);
        assert_eq!(sampling.max_h, 2);
        assert_eq!(sampling.max_v, 2);
    }

    #[test]
    fn sampling_factors_reject_empty_component_list() {
        assert!(matches!(
            SamplingFactors::from_components(&[]),
            Err(SamplingFactorsError::Empty)
        ));
    }

    #[test]
    fn sampling_factors_accept_supported_component_counts() {
        for components in [
            &[(1, 1)][..],
            &[(2, 2), (1, 1), (1, 1)][..],
            &[(1, 1), (1, 1), (1, 1), (1, 1)][..],
        ] {
            let sampling = SamplingFactors::from_components(components).expect("sampling");
            assert_eq!(sampling.len(), components.len());
            assert_eq!(sampling.components(), components);
        }
    }

    #[test]
    fn sampling_factors_reject_invalid_factors() {
        assert!(matches!(
            SamplingFactors::from_components(&[(0, 1)]),
            Err(SamplingFactorsError::InvalidSampling {
                component: 0,
                h: 0,
                v: 1
            })
        ));
        assert!(matches!(
            SamplingFactors::from_components(&[(1, 5)]),
            Err(SamplingFactorsError::InvalidSampling {
                component: 0,
                h: 1,
                v: 5
            })
        ));
    }

    #[test]
    fn sampling_factors_reject_more_than_four_components_without_panic() {
        assert!(matches!(
            SamplingFactors::from_components(&[(1, 1); 5]),
            Err(SamplingFactorsError::TooManyComponents { count: 5 })
        ));
    }

    #[test]
    fn info_to_core_info_preserves_metadata_for_device_adapters() {
        let info = Info {
            dimensions: (32, 16),
            color_space: ColorSpace::YCbCr,
            sampling: SamplingFactors::from_components(&[(2, 2), (1, 1), (1, 1)])
                .expect("sampling"),
            sof_kind: SofKind::Baseline8,
            bit_depth: 8,
            restart_interval: Some(2),
            mcu_geometry: McuGeometry {
                width: 16,
                height: 16,
                columns: 2,
                rows: 1,
                count: 2,
            },
            scan_count: 1,
        };

        let core = info.to_core_info();

        assert_eq!(core.dimensions, (32, 16));
        assert_eq!(core.components, 3);
        assert_eq!(core.colorspace, j2k_core::Colorspace::YCbCr);
        assert_eq!(core.bit_depth, 8);
        assert_eq!(core.tile_layout, None);
        assert_eq!(
            core.coded_unit_layout,
            Some(j2k_core::CodedUnitLayout {
                unit_width: 16,
                unit_height: 16,
                units_x: 2,
                units_y: 1,
            })
        );
        assert_eq!(core.restart_interval, Some(2));
        assert_eq!(core.resolution_levels, 1);
    }

    #[test]
    fn info_to_core_info_preserves_four_component_colorspaces() {
        for (color_space, core_colorspace) in [
            (ColorSpace::Cmyk, j2k_core::Colorspace::Cmyk),
            (ColorSpace::Ycck, j2k_core::Colorspace::Ycck),
        ] {
            let info = Info {
                dimensions: (64, 32),
                color_space,
                sampling: SamplingFactors::from_components(&[(1, 1), (1, 1), (1, 1), (1, 1)])
                    .expect("sampling"),
                sof_kind: SofKind::Baseline8,
                bit_depth: 8,
                restart_interval: None,
                mcu_geometry: McuGeometry {
                    width: 8,
                    height: 8,
                    columns: 8,
                    rows: 4,
                    count: 32,
                },
                scan_count: 1,
            };

            let core = info.to_core_info();

            assert_eq!(core.components, 4);
            assert_eq!(core.colorspace, core_colorspace);
            assert_eq!(
                core.coded_unit_layout,
                Some(j2k_core::CodedUnitLayout {
                    unit_width: 8,
                    unit_height: 8,
                    units_x: 8,
                    units_y: 4,
                })
            );
        }
    }
}
