// SPDX-License-Identifier: MIT OR Apache-2.0

//! HTJ2K Tier-1 coding values.

use alloc::vec::Vec;

/// Encoded HTJ2K cleanup/refinement code-block payload.
#[derive(Debug)]
pub struct EncodedHtJ2kCodeBlock {
    /// Combined cleanup/refinement bytes for this code block.
    pub data: Vec<u8>,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub refinement_length: u32,
    /// Number of coding passes present for this code block.
    pub num_coding_passes: u8,
    /// Number of zero most-significant bitplanes before first inclusion.
    pub num_zero_bitplanes: u8,
}

/// Encoded payload and exact pass boundaries for one expert HT set candidate.
#[doc(hidden)]
#[derive(Debug)]
pub struct EncodedHtJ2kCodeBlockSet {
    /// Combined cleanup, `SigProp`, and `MagRef` bytes.
    pub data: Vec<u8>,
    /// Cleanup segment length in bytes.
    pub cleanup_length: u32,
    /// `SigProp` prefix length within the refinement segment.
    pub sigprop_length: u32,
    /// `MagRef` suffix length within the refinement segment.
    pub magref_length: u32,
    /// Number of passes present from this set.
    pub num_coding_passes: u8,
    /// Missing most-significant planes for this cleanup pass.
    pub num_zero_bitplanes: u8,
}

/// HTJ2K code-block encode job.
#[derive(Debug, Clone, Copy)]
pub struct J2kHtCodeBlockEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Total bitplanes for this subband/code block.
    pub total_bitplanes: u8,
    /// Requested HT coding passes for this contribution.
    ///
    /// `1` is cleanup-only. `2` requests cleanup plus significance-propagation
    /// refinement on the native CPU path. `3` additionally requests one
    /// magnitude-refinement pass. Higher values require an accelerator and
    /// must not be silently reduced by CPU fallback.
    pub target_coding_passes: u8,
}

/// Expert HTJ2K job that selects one exact cleanup/refinement set.
///
/// This is used by bounded FBCOT candidate generation. Ordinary callers
/// should use [`J2kHtCodeBlockEncodeJob`].
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct J2kHtCodeBlockSetEncodeJob<'a> {
    /// Quantized coefficients in row-major order.
    pub coefficients: &'a [i32],
    /// Code-block width in samples.
    pub width: u32,
    /// Code-block height in samples.
    pub height: u32,
    /// Total bitplanes for this subband/code block.
    pub total_bitplanes: u8,
    /// Least-significant magnitude bitplane represented by the cleanup pass.
    pub cleanup_bitplane: u8,
    /// Number of passes from this set to encode, in the range 1 through 3.
    pub target_coding_passes: u8,
}

/// HTJ2K cleanup/refinement encode job for one unquantized subband.
#[derive(Debug, Clone, Copy)]
pub struct J2kHtSubbandEncodeJob<'a> {
    /// Source subband coefficients in row-major order.
    pub coefficients: &'a [f32],
    /// Subband width in samples.
    pub width: u32,
    /// Subband height in samples.
    pub height: u32,
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this subband.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
    /// Code-block width in samples.
    pub code_block_width: u32,
    /// Code-block height in samples.
    pub code_block_height: u32,
    /// Total coded bitplanes for this subband.
    pub total_bitplanes: u8,
}

crate::move_only::assert_move_only!(EncodedHtJ2kCodeBlock);
crate::move_only::assert_move_only!(EncodedHtJ2kCodeBlockSet);
