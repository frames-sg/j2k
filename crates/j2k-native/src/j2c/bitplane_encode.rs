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
use super::encode::allocation::{try_untracked_vec, try_untracked_vec_filled};
use crate::math::bit_width_u64;
use crate::EncodeResult;

mod allocation;
mod distortion;
mod passes;
mod preparation;
mod segments;
mod tokens;

pub(crate) use self::allocation::classic_worker_allocation;
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
    clippy::too_many_lines,
    reason = "the cohesive pass-order loop is kept intact to protect Tier-1 byte ordering"
)]
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
    let w = coefficients.width();
    let h = coefficients.height();
    let allocation = classic_worker_allocation(w, h, total_bitplanes)?;

    // Determine maximum magnitude and number of bitplanes
    let mut max_magnitude = 0_u64;
    for row in coefficients.rows() {
        for &coefficient in row {
            max_magnitude = max_magnitude.max(coefficient.unsigned_magnitude());
        }
    }

    if max_magnitude == 0 {
        return Ok(EncodedCodeBlock {
            data: Vec::new(),
            num_coding_passes: 0,
            num_zero_bitplanes: total_bitplanes,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
        });
    }

    let num_bitplanes = bit_width_u64(max_magnitude);
    if num_bitplanes > total_bitplanes {
        return Err(crate::EncodeError::InvalidInput {
            what: "classic code-block magnitude exceeds configured bitplane count",
        });
    }
    let num_zero_bitplanes = total_bitplanes.saturating_sub(num_bitplanes);

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

    let mut encoder = ArithmeticEncoder::try_with_byte_limit(allocation.payload_bytes)?;
    let mut contexts = [ArithmeticEncoderContext::default(); 19];
    reset_contexts(&mut contexts);

    let mut num_coding_passes = 0u8;
    let mut coded_indices = try_untracked_vec(
        allocation.padded_coefficients,
        "classic Tier-1 coded-index scratch",
    )?;

    // Process bitplanes from MSB to LSB
    for bp in (0..num_bitplanes).rev() {
        let bit_mask = 1u64 << bp;
        let is_first_bitplane = bp == num_bitplanes - 1;

        if is_first_bitplane {
            // First bitplane: cleanup pass only
            cleanup_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                sub_band_type,
                style,
            );
            if style.segmentation_symbols {
                encode_segmentation_symbols(&mut encoder, &mut contexts);
            }
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }
        } else {
            // Subsequent bitplanes: SPP, MRP, Cleanup
            significance_propagation_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut coded_indices,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                sub_band_type,
                style,
            );
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }

            magnitude_refinement_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                style,
            );
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }

            cleanup_pass(
                &magnitudes,
                &mut states,
                &mut neighbors,
                &mut encoder,
                &mut contexts,
                w,
                h,
                pw,
                bit_mask,
                sub_band_type,
                style,
            );
            if style.segmentation_symbols {
                encode_segmentation_symbols(&mut encoder, &mut contexts);
            }
            num_coding_passes += 1;
            if style.reset_context_probabilities {
                reset_contexts(&mut contexts);
            }
        }

        clear_coded_in_current_pass(&mut states, &mut coded_indices);
    }

    let data = encoder.finish_checked()?;

    Ok(EncodedCodeBlock {
        data,
        num_coding_passes,
        num_zero_bitplanes,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
    })
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
