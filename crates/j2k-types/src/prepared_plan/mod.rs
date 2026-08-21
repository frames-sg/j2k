// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned, move-only prepared HTJ2K encode values.

use alloc::vec::Vec;
use core::ops::Range;

use crate::{EncodedHtJ2kCodeBlock, J2kForwardDwt53Output, J2kForwardDwt97Output, J2kSubBandType};

/// Precomputed reversible 5/3 wavelet coefficients for one component.
#[derive(Debug)]
pub struct PrecomputedHtj2k53Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Forward 5/3 DWT output, ordered as the encoder expects.
    pub dwt: J2kForwardDwt53Output,
}

/// Precomputed reversible 5/3 wavelet image.
#[derive(Debug)]
pub struct PrecomputedHtj2k53Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrecomputedHtj2k53Component>,
}

/// Precomputed irreversible 9/7 wavelet coefficients for one component.
#[derive(Debug)]
pub struct PrecomputedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Forward 9/7 DWT output, ordered as the encoder expects.
    pub dwt: J2kForwardDwt97Output,
}

/// Precomputed irreversible 9/7 wavelet image.
#[derive(Debug)]
pub struct PrecomputedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrecomputedHtj2k97Component>,
}

/// Prequantized irreversible 9/7 HTJ2K code-block image.
#[derive(Debug)]
pub struct PrequantizedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PrequantizedHtj2k97Component>,
}

/// Prequantized irreversible 9/7 HTJ2K component.
#[derive(Debug)]
pub struct PrequantizedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PrequantizedHtj2k97Resolution>,
}

/// One component resolution's prequantized HTJ2K subbands.
#[derive(Debug)]
pub struct PrequantizedHtj2k97Resolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PrequantizedHtj2k97Subband>,
}

/// One prequantized HTJ2K subband split into code blocks.
#[derive(Debug)]
pub struct PrequantizedHtj2k97Subband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code block in this subband.
    pub total_bitplanes: u8,
    /// Code-block coefficients in row-major code-block order.
    pub code_blocks: Vec<PrequantizedHtj2k97CodeBlock>,
}

/// One prequantized HTJ2K code block.
#[derive(Debug)]
pub struct PrequantizedHtj2k97CodeBlock {
    /// Quantized coefficients in row-major order.
    pub coefficients: Vec<i32>,
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
}

/// Preencoded irreversible 9/7 HTJ2K code-block image.
#[derive(Debug)]
pub struct PreencodedHtj2k97Image {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Components at their native resolution.
    pub components: Vec<PreencodedHtj2k97Component>,
}

/// Preencoded irreversible 9/7 HTJ2K component.
#[derive(Debug)]
pub struct PreencodedHtj2k97Component {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PreencodedHtj2k97Resolution>,
}

/// One component resolution's preencoded HTJ2K subbands.
#[derive(Debug)]
pub struct PreencodedHtj2k97Resolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PreencodedHtj2k97Subband>,
}

/// One preencoded HTJ2K subband split into code blocks.
#[derive(Debug)]
pub struct PreencodedHtj2k97Subband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code block in this subband.
    pub total_bitplanes: u8,
    /// Encoded code-block payloads in row-major code-block order.
    pub code_blocks: Vec<PreencodedHtj2k97CodeBlock>,
}

/// One preencoded HTJ2K code block.
#[derive(Debug)]
pub struct PreencodedHtj2k97CodeBlock {
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Encoded cleanup/refinement payload and packet metadata.
    pub encoded: EncodedHtJ2kCodeBlock,
}

/// Preencoded irreversible 9/7 HTJ2K image backed by one compact payload buffer.
#[derive(Debug)]
pub struct PreencodedHtj2k97CompactImage {
    /// Reference-grid image width.
    pub width: u32,
    /// Reference-grid image height.
    pub height: u32,
    /// Component precision in bits.
    pub bit_depth: u8,
    /// Whether component samples are signed.
    pub signed: bool,
    /// Contiguous encoded code-block payload bytes.
    pub payload: Vec<u8>,
    /// Components at their native resolution.
    pub components: Vec<PreencodedHtj2k97CompactComponent>,
}

/// Preencoded compact irreversible 9/7 HTJ2K component.
#[derive(Debug)]
pub struct PreencodedHtj2k97CompactComponent {
    /// Horizontal SIZ sampling factor (`XRsiz`).
    pub x_rsiz: u8,
    /// Vertical SIZ sampling factor (`YRsiz`).
    pub y_rsiz: u8,
    /// Resolution packets for this component, ordered from lowest to highest.
    pub resolutions: Vec<PreencodedHtj2k97CompactResolution>,
}

/// One component resolution's compact preencoded HTJ2K subbands.
#[derive(Debug)]
pub struct PreencodedHtj2k97CompactResolution {
    /// Subbands in packet order: LL for resolution 0, then HL/LH/HH.
    pub subbands: Vec<PreencodedHtj2k97CompactSubband>,
}

/// One compact preencoded HTJ2K subband split into code blocks.
#[derive(Debug)]
pub struct PreencodedHtj2k97CompactSubband {
    /// Subband kind.
    pub sub_band_type: J2kSubBandType,
    /// Number of code blocks in the x direction.
    pub num_cbs_x: u32,
    /// Number of code blocks in the y direction.
    pub num_cbs_y: u32,
    /// Total bitplanes declared for every code block in this subband.
    pub total_bitplanes: u8,
    /// Code-block metadata in row-major code-block order.
    pub code_blocks: Vec<PreencodedHtj2k97CompactCodeBlock>,
}

/// One compact preencoded HTJ2K code block.
#[derive(Debug)]
pub struct PreencodedHtj2k97CompactCodeBlock {
    /// Code-block width in coefficients.
    pub width: u32,
    /// Code-block height in coefficients.
    pub height: u32,
    /// Byte range into the image-level compact payload.
    pub payload_range: Range<usize>,
    /// HTJ2K cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes.
    pub refinement_length: u32,
    /// Number of coding passes in the encoded payload.
    pub num_coding_passes: u8,
    /// Number of missing most-significant bitplanes.
    pub num_zero_bitplanes: u8,
}

crate::move_only::assert_move_only!(
    PrecomputedHtj2k53Component,
    PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Component,
    PrecomputedHtj2k97Image,
    PrequantizedHtj2k97Image,
    PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution,
    PrequantizedHtj2k97Subband,
    PrequantizedHtj2k97CodeBlock,
    PreencodedHtj2k97Image,
    PreencodedHtj2k97Component,
    PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband,
    PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97CompactCodeBlock,
);
