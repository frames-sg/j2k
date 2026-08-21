// SPDX-License-Identifier: MIT OR Apache-2.0

//! Classic JPEG 2000 Tier-1 coding values.

use alloc::vec::Vec;

/// Classic JPEG 2000 subband kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kSubBandType {
    /// Low-low subband.
    LowLow,
    /// High-low subband.
    HighLow,
    /// Low-high subband.
    LowHigh,
    /// High-high subband.
    HighHigh,
}

/// Classic JPEG 2000 code-block style flags.
#[derive(Debug, Clone, Copy)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the five booleans model independent JPEG 2000 COD code-block style flags"
)]
pub struct J2kCodeBlockStyle {
    /// Selective arithmetic coding bypass was enabled.
    pub selective_arithmetic_coding_bypass: bool,
    /// Context probabilities reset after each pass.
    pub reset_context_probabilities: bool,
    /// Coding terminated after each pass.
    pub termination_on_each_pass: bool,
    /// Vertically causal context was enabled.
    pub vertically_causal_context: bool,
    /// Segmentation symbols were enabled.
    pub segmentation_symbols: bool,
}

/// One coded segment in a classic JPEG 2000 code block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct J2kCodeBlockSegment {
    /// Byte offset of this segment within the combined payload.
    pub data_offset: u32,
    /// Segment payload length in bytes.
    pub data_length: u32,
    /// First coding pass covered by this segment.
    pub start_coding_pass: u8,
    /// One-past-last coding pass covered by this segment.
    pub end_coding_pass: u8,
    /// Whether this segment is decoded through the arithmetic path.
    pub use_arithmetic: bool,
}

/// Encoded classic JPEG 2000 code-block payload.
#[derive(Debug)]
pub struct EncodedJ2kCodeBlock {
    /// Combined payload bytes for all coded segments in this code block.
    pub data: Vec<u8>,
    /// Coded segments for the code block.
    pub segments: Vec<J2kCodeBlockSegment>,
    /// Number of coding passes present for this code block.
    pub number_of_coding_passes: u8,
    /// Missing most-significant bit planes for this code block.
    pub missing_bit_planes: u8,
}

/// Classic JPEG 2000 Tier-1 code-block encode job.
#[derive(Debug, Clone, Copy)]
pub struct J2kTier1CodeBlockEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Subband kind containing this code block.
    pub sub_band_type: J2kSubBandType,
    /// Total bitplanes for this subband/code block.
    pub total_bitplanes: u8,
    /// Classic JPEG 2000 code-block style flags.
    pub style: J2kCodeBlockStyle,
}

crate::move_only::assert_move_only!(EncodedJ2kCodeBlock);
