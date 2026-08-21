// SPDX-License-Identifier: MIT OR Apache-2.0

//! Backend-neutral retained JPEG 2000 decode-plan contracts.

mod allocation;
mod referenced;

use alloc::vec::Vec;

use crate::{J2kCodeBlockSegment, J2kCodeBlockStyle, J2kSubBandType};

pub use allocation::DecodePlanAllocationError;
pub use referenced::{
    J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kReferencedImageGeometry,
    J2kReferencedPayloadRecordSpan, J2kReferencedTileGeometry, J2kReferencedTilePlan,
};

/// Default maximum simultaneously retained codec allocation in bytes.
pub const DEFAULT_MAX_CODEC_BYTES: usize = 512 * 1024 * 1024;

/// Default maximum retained decode allocation in bytes.
pub const DEFAULT_MAX_DECODE_BYTES: usize = DEFAULT_MAX_CODEC_BYTES;

/// Stable identifier for one device-owned grayscale coefficient band.
pub type J2kDirectBandId = u32;

/// Integer rectangle in component coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

impl J2kRect {
    /// Rectangle width in samples.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// Rectangle height in samples.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }
}

/// Wavelet transform used by retained decode geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kWaveletTransform {
    /// Reversible 5/3 transform.
    Reversible53,
    /// Irreversible 9/7 transform.
    Irreversible97,
}

/// Ordered direct-device decode step for one component.
#[derive(Debug)]
pub enum J2kDirectGrayscaleStep {
    /// Decode one classic JPEG 2000 subband.
    ClassicSubBand(J2kOwnedSubBandPlan),
    /// Decode one HTJ2K subband.
    HtSubBand(HtOwnedSubBandPlan),
    /// Apply one single-decomposition inverse transform.
    Idwt(J2kDirectIdwtStep),
    /// Store the final component plane.
    Store(J2kDirectStoreStep),
}

/// Direct-device decode plan for one component.
#[derive(Debug)]
pub struct J2kDirectGrayscalePlan {
    /// Final output dimensions.
    pub dimensions: (u32, u32),
    /// Final output bit depth.
    pub bit_depth: u8,
    /// Ordered execution steps.
    pub steps: Vec<J2kDirectGrayscaleStep>,
}

/// Direct-device RGB decode plan.
#[derive(Debug)]
pub struct J2kDirectColorPlan {
    /// Final output dimensions.
    pub dimensions: (u32, u32),
    /// Final output bit depths for the first three components.
    pub bit_depths: [u8; 3],
    /// Whether inverse MCT is required.
    pub mct: bool,
    /// Wavelet transform used by the color transform.
    pub transform: J2kWaveletTransform,
    /// Per-component plans in R, G, B order.
    pub component_plans: Vec<J2kDirectGrayscalePlan>,
}

/// Direct-device RGBA decode plan with RGB-only inverse MCT semantics.
#[derive(Debug)]
pub struct J2kDirectRgbaPlan {
    /// Final output dimensions.
    pub dimensions: (u32, u32),
    /// Final output bit depths in R, G, B, A order.
    pub bit_depths: [u8; 4],
    /// Whether inverse MCT is required for the first three components.
    pub mct: bool,
    /// Wavelet transform used by the RGB color transform.
    pub transform: J2kWaveletTransform,
    /// Per-component plans in R, G, B, A order.
    pub component_plans: Vec<J2kDirectGrayscalePlan>,
}

/// Owned classic JPEG 2000 subband decode plan.
#[derive(Debug)]
pub struct J2kOwnedSubBandPlan {
    /// Stable output-band identifier.
    pub band_id: J2kDirectBandId,
    /// Absolute subband rectangle.
    pub rect: J2kRect,
    /// Subband width in samples.
    pub width: u32,
    /// Subband height in samples.
    pub height: u32,
    /// Whether irreversible midpoint reconstruction is required.
    pub irreversible_midpoint: bool,
    /// Owned code-block jobs.
    pub jobs: Vec<J2kOwnedCodeBlockBatchJob>,
}

/// Owned HTJ2K subband decode plan.
#[derive(Debug)]
pub struct HtOwnedSubBandPlan {
    /// Stable output-band identifier.
    pub band_id: J2kDirectBandId,
    /// Absolute subband rectangle.
    pub rect: J2kRect,
    /// Subband width in samples.
    pub width: u32,
    /// Subband height in samples.
    pub height: u32,
    /// Whether irreversible midpoint reconstruction is required.
    pub irreversible_midpoint: bool,
    /// Owned code-block jobs.
    pub jobs: Vec<HtOwnedCodeBlockBatchJob>,
}

/// Owned classic JPEG 2000 code-block decode job.
#[derive(Debug)]
pub struct J2kOwnedCodeBlockBatchJob {
    /// X offset in the target subband.
    pub output_x: u32,
    /// Y offset in the target subband.
    pub output_y: u32,
    /// Combined bytes for every coded segment.
    pub data: Vec<u8>,
    /// Coded segments.
    pub segments: Vec<J2kCodeBlockSegment>,
    /// Code-block width.
    pub width: u32,
    /// Code-block height.
    pub height: u32,
    /// Output row stride in samples.
    pub output_stride: usize,
    /// Missing most-significant bit planes.
    pub missing_bit_planes: u8,
    /// Number of coding passes.
    pub number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent subband.
    pub total_bitplanes: u8,
    /// ROI maxshift value.
    pub roi_shift: u8,
    /// Parent subband type.
    pub sub_band_type: J2kSubBandType,
    /// Code-block style flags.
    pub style: J2kCodeBlockStyle,
    /// Whether strict validation is enabled.
    pub strict: bool,
    /// Dequantization step.
    pub dequantization_step: f32,
}

/// Owned HTJ2K code-block decode job.
#[derive(Debug)]
pub struct HtOwnedCodeBlockBatchJob {
    /// X offset in the target subband.
    pub output_x: u32,
    /// Y offset in the target subband.
    pub output_y: u32,
    /// Combined cleanup and refinement bytes.
    pub data: Vec<u8>,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Code-block width.
    pub width: u32,
    /// Code-block height.
    pub height: u32,
    /// Output row stride in samples.
    pub output_stride: usize,
    /// Missing most-significant bit planes.
    pub missing_bit_planes: u8,
    /// Number of coding passes.
    pub number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent subband.
    pub num_bitplanes: u8,
    /// ROI maxshift value.
    pub roi_shift: u8,
    /// Whether vertically causal contexts are enabled.
    pub stripe_causal: bool,
    /// Whether strict validation is enabled.
    pub strict: bool,
    /// Dequantization step.
    pub dequantization_step: f32,
}

/// One inverse-transform step in a direct-device plan.
#[derive(Debug, Clone, Copy)]
pub struct J2kDirectIdwtStep {
    /// Output coefficient-band identifier.
    pub output_band_id: J2kDirectBandId,
    /// Output rectangle.
    pub rect: J2kRect,
    /// Transform to apply.
    pub transform: J2kWaveletTransform,
    /// LL input-band identifier.
    pub ll_band_id: J2kDirectBandId,
    /// LL input rectangle.
    pub ll: J2kRect,
    /// HL input-band identifier.
    pub hl_band_id: J2kDirectBandId,
    /// HL input rectangle.
    pub hl: J2kRect,
    /// LH input-band identifier.
    pub lh_band_id: J2kDirectBandId,
    /// LH input rectangle.
    pub lh: J2kRect,
    /// HH input-band identifier.
    pub hh_band_id: J2kDirectBandId,
    /// HH input rectangle.
    pub hh: J2kRect,
}

/// One final component-store step in a direct-device plan.
#[derive(Debug, Clone, Copy)]
pub struct J2kDirectStoreStep {
    /// Input coefficient-band identifier.
    pub input_band_id: J2kDirectBandId,
    /// Input plane rectangle.
    pub input_rect: J2kRect,
    /// Source x offset.
    pub source_x: u32,
    /// Source y offset.
    pub source_y: u32,
    /// Samples copied per row.
    pub copy_width: u32,
    /// Rows copied.
    pub copy_height: u32,
    /// Destination row width.
    pub output_width: u32,
    /// Destination height.
    pub output_height: u32,
    /// Destination x offset.
    pub output_x: u32,
    /// Destination y offset.
    pub output_y: u32,
    /// Constant added to every copied sample.
    pub addend: f32,
}
