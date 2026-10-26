//! EBCOT Tier-1 encoder for JPEG 2000 (ITU-T T.800 Annex D).
//!
//! Encodes quantized wavelet coefficients into code-block bitstreams using:
//! - MQ arithmetic coding
//! - Context-dependent coding with the same 19 contexts as the decoder
//! - Three passes per bitplane: significance propagation, magnitude refinement, cleanup
//! - Column-stripe scanning order (4-row stripes)

use alloc::vec::Vec;

use super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::build::SubBandType;
use super::codestream::CodeBlockStyle;
use super::coefficient_view::{CoefficientBlockView, SignedCoefficient};
use super::encode::allocation::try_untracked_vec_filled;
use crate::math::bit_width_u64;
use crate::EncodeResult;

mod allocation;
mod distortion;
mod flag_passes;
mod passes;
mod preparation;
mod segments;
mod tokens;

pub(crate) use self::allocation::classic_worker_allocation;
use self::allocation::ClassicWorkerAllocation;
use self::passes::{
    cleanup_pass, clear_coded_in_current_pass, magnitude_refinement_pass,
    significance_propagation_pass,
};
use self::preparation::try_prepare_padded_coefficients_from_view;
use self::segments::{encode_segmentation_symbols, reset_contexts};
pub(crate) use self::tokens::try_pack_classic_selective_bypass_tier1_tokens;
#[cfg(test)]
pub(crate) use self::tokens::{
    pack_classic_selective_bypass_tier1_tokens, ClassicTier1TokenSegment,
};

/// Result of encoding a single code-block.
#[derive(Debug)]
pub(crate) struct EncodedCodeBlock {
    /// The compressed bitstream data.
    pub(crate) data: Vec<u8>,
    /// Number of coding passes actually generated.
    pub(crate) num_coding_passes: u8,
    /// Number of leading zero bitplanes (missing MSBs).
    pub(crate) num_zero_bitplanes: u8,
    /// HTJ2K cleanup segment length in bytes when this block uses HT coding.
    pub(crate) ht_cleanup_length: u32,
    /// HTJ2K refinement segment length in bytes when this block uses HT coding.
    pub(crate) ht_refinement_length: u32,
    /// Significance-propagation prefix length within the HT refinement segment.
    pub(crate) ht_sigprop_length: u32,
    /// Magnitude-refinement suffix length within the HT refinement segment.
    pub(crate) ht_magref_length: u32,
    /// Per-pass squared-error reductions used by HT post-compression rate control.
    pub(crate) ht_distortion_deltas: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EncodedCodeBlockSegment {
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) start_coding_pass: u8,
    pub(crate) end_coding_pass: u8,
    pub(crate) distortion_delta: f64,
    pub(crate) use_arithmetic: bool,
}

#[derive(Debug)]
pub(crate) struct EncodedCodeBlockWithSegments {
    pub(crate) data: Vec<u8>,
    pub(crate) segments: Vec<EncodedCodeBlockSegment>,
    pub(crate) num_coding_passes: u8,
    pub(crate) num_zero_bitplanes: u8,
}

/// Encode a single code-block's quantized coefficients.
///
/// `coefficients` are quantized i32 values in row-major order.
/// `width`, `height` are the code-block dimensions.
/// `sub_band_type` determines which zero-coding context table to use.
/// `total_bitplanes` is the JPEG 2000 `Mb` value for this subband/code-block.
pub(crate) fn try_encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    try_encode_code_block_view(coefficients, sub_band_type, total_bitplanes)
}

pub(crate) fn try_encode_code_block_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    try_encode_code_block_view(coefficients, sub_band_type, total_bitplanes)
}

pub(crate) fn try_encode_code_block_view<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    try_encode_code_block_with_style_view(
        coefficients,
        sub_band_type,
        total_bitplanes,
        &CodeBlockStyle::default(),
    )
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable Tier-1 entrypoint borrows the caller's code-block style"
)]
pub(crate) fn try_encode_code_block_with_style_view<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodeResult<EncodedCodeBlock> {
    let allocation =
        classic_worker_allocation(coefficients.width(), coefficients.height(), total_bitplanes)?;
    // Empty blocks and invalid magnitudes need no pass scratch.
    let (data, num_coding_passes, num_zero_bitplanes) =
        match coded_bitplanes(coefficients, total_bitplanes)? {
            None => (Vec::new(), 0, total_bitplanes),
            Some(num_bitplanes) => {
                let block = if !style.vertically_causal_context
                    && total_bitplanes <= flag_passes::MAX_BITPLANES
                {
                    flag_passes::try_encode_column_passes(
                        coefficients,
                        sub_band_type,
                        num_bitplanes,
                        style,
                        allocation.payload_bytes,
                    )?
                } else {
                    try_encode_neighbor_byte_passes(
                        coefficients,
                        sub_band_type,
                        num_bitplanes,
                        style,
                        allocation,
                    )?
                };
                (
                    block.data,
                    block.num_coding_passes,
                    total_bitplanes - num_bitplanes,
                )
            }
        };
    Ok(EncodedCodeBlock {
        data,
        num_coding_passes,
        num_zero_bitplanes,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        ht_sigprop_length: 0,
        ht_magref_length: 0,
        ht_distortion_deltas: [0.0; 3],
    })
}

/// The MQ payload of one block's coding passes.
struct CodedBlock {
    data: Vec<u8>,
    num_coding_passes: u8,
}

#[derive(Debug, Clone, Copy)]
enum CodingPass {
    SignificancePropagation,
    MagnitudeRefinement,
    Cleanup,
}

/// The coded bitplane count of a block, or `None` when every coefficient is
/// zero.
fn coded_bitplanes<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    total_bitplanes: u8,
) -> EncodeResult<Option<u8>> {
    let any = coefficients
        .rows()
        .flatten()
        .fold(0, |any, coefficient| any | coefficient.unsigned_magnitude());
    if any == 0 {
        return Ok(None);
    }
    let num_bitplanes = bit_width_u64(any);
    if num_bitplanes > total_bitplanes {
        return Err(crate::EncodeError::InvalidInput {
            what: "classic code-block magnitude exceeds configured bitplane count",
        });
    }
    Ok(Some(num_bitplanes))
}

/// Run `code` for every coding pass of `num_bitplanes` in T.800 D.3 order,
/// from the most significant bitplane, with the style's segmentation symbols
/// and context resets at the pass boundaries, and finish the MQ codeword.
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
fn code_passes(
    num_bitplanes: u8,
    style: &CodeBlockStyle,
    payload_bytes: usize,
    mut code: impl FnMut(CodingPass, u32, &mut ArithmeticEncoder, &mut [ArithmeticEncoderContext; 19]),
) -> EncodeResult<CodedBlock> {
    let mut encoder = ArithmeticEncoder::try_with_byte_limit(payload_bytes)?;
    let mut contexts = [ArithmeticEncoderContext::default(); 19];
    reset_contexts(&mut contexts);
    let mut num_coding_passes = 0u8;
    for bitplane in (0..u32::from(num_bitplanes)).rev() {
        // The most significant bitplane has only a cleanup pass.
        let passes: &[CodingPass] = if bitplane + 1 == u32::from(num_bitplanes) {
            &[CodingPass::Cleanup]
        } else {
            &[
                CodingPass::SignificancePropagation,
                CodingPass::MagnitudeRefinement,
                CodingPass::Cleanup,
            ]
        };
        for &pass in passes {
            code(pass, bitplane, &mut encoder, &mut contexts);
            if matches!(pass, CodingPass::Cleanup) && style.segmentation_symbols {
                encode_segmentation_symbols(&mut encoder, &mut contexts);
            }
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }
        }
    }
    Ok(CodedBlock {
        data: encoder.finish_checked()?,
        num_coding_passes,
    })
}

/// Code every pass of a block's `num_bitplanes` with one state and one
/// neighbor byte per padded sample, for vertically causal blocks and blocks
/// with more bitplanes than the column words hold.
#[expect(clippy::trivially_copy_pass_by_ref, reason = "shared style borrow")]
fn try_encode_neighbor_byte_passes<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    num_bitplanes: u8,
    style: &CodeBlockStyle,
    allocation: ClassicWorkerAllocation,
) -> EncodeResult<CodedBlock> {
    let w = coefficients.width();
    let h = coefficients.height();

    // Build padded coefficient magnitude and state arrays.
    let pw = w
        .checked_add(2)
        .ok_or(crate::EncodeError::ArithmeticOverflow {
            what: "classic Tier-1 padded width",
        })?;
    let (magnitudes, mut states) = try_prepare_padded_coefficients_from_view(coefficients, pw)?;
    let mut neighbors = try_untracked_vec_filled(
        allocation.padded_coefficients,
        0_u8,
        "classic Tier-1 neighbor states",
    )?;
    code_passes(
        num_bitplanes,
        style,
        allocation.payload_bytes,
        |pass, bitplane, encoder, contexts| {
            let bit_mask = 1u64 << bitplane;
            match pass {
                CodingPass::SignificancePropagation => significance_propagation_pass(
                    &magnitudes,
                    &mut states,
                    &mut neighbors,
                    encoder,
                    contexts,
                    w,
                    h,
                    pw,
                    bit_mask,
                    sub_band_type,
                    style,
                ),
                CodingPass::MagnitudeRefinement => magnitude_refinement_pass(
                    &magnitudes,
                    &mut states,
                    &mut neighbors,
                    encoder,
                    contexts,
                    w,
                    h,
                    pw,
                    bit_mask,
                    style,
                ),
                CodingPass::Cleanup => {
                    cleanup_pass(
                        &magnitudes,
                        &mut states,
                        &mut neighbors,
                        encoder,
                        contexts,
                        w,
                        h,
                        pw,
                        bit_mask,
                        sub_band_type,
                        style,
                    );
                    clear_coded_in_current_pass(&mut states);
                }
            }
        },
    )
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable segmented Tier-1 entrypoint borrows the caller's code-block style"
)]
pub(crate) fn try_encode_code_block_segments_with_style(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodeResult<EncodedCodeBlockWithSegments> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    segments::try_encode_segmented_code_block(coefficients, sub_band_type, total_bitplanes, style)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable segmented Tier-1 entrypoint borrows the caller's code-block style"
)]
pub(crate) fn try_encode_code_block_segments_with_style_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodeResult<EncodedCodeBlockWithSegments> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    segments::try_encode_segmented_code_block(coefficients, sub_band_type, total_bitplanes, style)
}

#[cfg(test)]
pub(crate) fn encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodedCodeBlock {
    try_encode_code_block(coefficients, width, height, sub_band_type, total_bitplanes)
        .expect("test classic i32 encode")
}

#[cfg(test)]
pub(crate) fn encode_code_block_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodedCodeBlock {
    try_encode_code_block_i64(coefficients, width, height, sub_band_type, total_bitplanes)
        .expect("test classic i64 encode")
}

#[cfg(test)]
pub(crate) fn encode_code_block_view<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
) -> EncodedCodeBlock {
    try_encode_code_block_view(coefficients, sub_band_type, total_bitplanes)
        .expect("test classic strided encode")
}

#[cfg(test)]
pub(crate) fn encode_code_block_segments_with_style(
    coefficients: &[i32],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: CodeBlockStyle,
) -> EncodedCodeBlockWithSegments {
    try_encode_code_block_segments_with_style(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        &style,
    )
    .expect("test classic segmented i32 encode")
}

#[cfg(test)]
pub(crate) fn encode_code_block_segments_with_style_i64(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: CodeBlockStyle,
) -> EncodedCodeBlockWithSegments {
    try_encode_code_block_segments_with_style_i64(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        &style,
    )
    .expect("test classic segmented i64 encode")
}

#[cfg(test)]
mod tests;
