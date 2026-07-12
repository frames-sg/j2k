// SPDX-License-Identifier: MIT OR Apache-2.0

// Coding-pass scheduling and classic segment state management.

use alloc::vec::Vec;

use super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::super::build::SubBandType;
use super::super::codestream::CodeBlockStyle;
use super::super::coefficient_view::{CoefficientBlockView, SignedCoefficient};
use super::super::encode::allocation::{
    try_reserve_untracked, try_reserve_untracked_bounded, try_untracked_vec,
};
use crate::math::bit_width_u64;
use crate::{EncodeError, EncodeResult};

use super::allocation::classic_worker_allocation;
use super::distortion::segment_distortion_delta_view;
use super::{
    try_encode_code_block_with_style_view, EncodedCodeBlockSegment, EncodedCodeBlockWithSegments,
};

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "segmented scheduling shares one borrowed style across every coding pass"
)]
pub(super) fn try_encode_segmented_code_block<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodeResult<EncodedCodeBlockWithSegments> {
    let allocation =
        classic_worker_allocation(coefficients.width(), coefficients.height(), total_bitplanes)?;
    if let Some(encoded) =
        try_encode_unsegmented_code_block(coefficients, sub_band_type, total_bitplanes, style)?
    {
        return Ok(encoded);
    }

    let max_magnitude = coefficients
        .rows()
        .flatten()
        .map(|coefficient| coefficient.unsigned_magnitude())
        .max()
        .unwrap_or(0);
    if max_magnitude == 0 {
        return Ok(empty_segmented_code_block(total_bitplanes));
    }

    let num_bitplanes = bit_width_u64(max_magnitude);
    if num_bitplanes > total_bitplanes {
        return Err(EncodeError::InvalidInput {
            what: "classic code-block magnitude exceeds configured bitplane count",
        });
    }
    let num_zero_bitplanes = total_bitplanes.saturating_sub(num_bitplanes);
    SegmentedCodeBlockEncoder::try_new(
        coefficients,
        sub_band_type,
        num_bitplanes,
        style,
        allocation,
    )?
    .try_encode_all_passes(num_zero_bitplanes)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the adapter shares the caller's style with the stable encoder entrypoint"
)]
fn try_encode_unsegmented_code_block<T: SignedCoefficient>(
    coefficients: CoefficientBlockView<'_, T>,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodeResult<Option<EncodedCodeBlockWithSegments>> {
    if style.termination_on_each_pass || style.selective_arithmetic_coding_bypass {
        return Ok(None);
    }

    let encoded =
        try_encode_code_block_with_style_view(coefficients, sub_band_type, total_bitplanes, style)?;
    let segments = if encoded.num_coding_passes == 0 {
        Vec::new()
    } else {
        let mut segments = try_untracked_vec(1, "classic Tier-1 segment metadata")?;
        segments.push(EncodedCodeBlockSegment {
            data_offset: 0,
            data_length: u32::try_from(encoded.data.len()).map_err(|_| {
                EncodeError::InternalInvariant {
                    what: "classic Tier-1 payload length exceeds u32",
                }
            })?,
            start_coding_pass: 0,
            end_coding_pass: encoded.num_coding_passes,
            distortion_delta: segment_distortion_delta_view(
                coefficients,
                0,
                encoded.num_coding_passes,
                total_bitplanes,
            ),
            use_arithmetic: true,
        });
        segments
    };
    Ok(Some(EncodedCodeBlockWithSegments {
        data: encoded.data,
        segments,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    }))
}

fn empty_segmented_code_block(total_bitplanes: u8) -> EncodedCodeBlockWithSegments {
    EncodedCodeBlockWithSegments {
        data: Vec::new(),
        segments: Vec::new(),
        num_coding_passes: 0,
        num_zero_bitplanes: total_bitplanes,
    }
}

mod encoder;
use encoder::SegmentedCodeBlockEncoder;

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the pass loop keeps one borrowed style rather than copying it for each query"
)]
fn segment_index(style: &CodeBlockStyle, coding_pass: u8) -> u8 {
    if style.termination_on_each_pass {
        coding_pass
    } else if style.selective_arithmetic_coding_bypass {
        bypass_segment_idx(coding_pass)
    } else {
        0
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the pass loop keeps one borrowed style rather than copying it for each query"
)]
fn pass_uses_arithmetic(style: &CodeBlockStyle, coding_pass: u8) -> bool {
    !style.selective_arithmetic_coding_bypass || coding_pass <= 9 || coding_pass.is_multiple_of(3)
}

pub(super) fn reset_contexts(contexts: &mut [ArithmeticEncoderContext; 19]) {
    *contexts = [ArithmeticEncoderContext::default(); 19];
    contexts[0].reset_with_index(4);
    contexts[17].reset_with_index(3);
    contexts[18].reset_with_index(46);
}

pub(super) fn encode_segmentation_symbols(
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
) {
    encoder.encode(1, &mut contexts[18]);
    encoder.encode(0, &mut contexts[18]);
    encoder.encode(1, &mut contexts[18]);
    encoder.encode(0, &mut contexts[18]);
}

#[inline]
fn bypass_segment_idx(pass_idx: u8) -> u8 {
    if pass_idx < 10 {
        0
    } else {
        1 + (2 * ((pass_idx - 10) / 3)) + u8::from(((pass_idx - 10) % 3) == 2)
    }
}

pub(super) struct PendingSegment {
    pub(super) start_coding_pass: u8,
    pub(super) end_coding_pass: u8,
    pub(super) data: Vec<u8>,
    pub(super) distortion_delta: f64,
    pub(super) use_arithmetic: bool,
}

pub(super) fn try_push_segment(
    data: &mut Vec<u8>,
    segments: &mut Vec<EncodedCodeBlockSegment>,
    payload_limit: usize,
    segment_limit: usize,
    pending: PendingSegment,
) -> EncodeResult<()> {
    let PendingSegment {
        start_coding_pass,
        end_coding_pass,
        data: segment_data,
        distortion_delta,
        use_arithmetic,
    } = pending;
    let new_payload_len =
        data.len()
            .checked_add(segment_data.len())
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "classic Tier-1 accumulated payload length",
            })?;
    if new_payload_len > payload_limit {
        return Err(EncodeError::InternalInvariant {
            what: "classic Tier-1 payload exceeded its checked bound",
        });
    }
    if segments.len() >= segment_limit {
        return Err(EncodeError::InternalInvariant {
            what: "classic Tier-1 segment count exceeded its checked bound",
        });
    }
    let data_offset = u32::try_from(data.len()).map_err(|_| EncodeError::InternalInvariant {
        what: "classic Tier-1 data offset exceeds u32",
    })?;
    let data_length =
        u32::try_from(segment_data.len()).map_err(|_| EncodeError::InternalInvariant {
            what: "classic Tier-1 segment length exceeds u32",
        })?;
    try_reserve_untracked_bounded(
        data,
        segment_data.len(),
        payload_limit,
        "classic Tier-1 payload",
    )?;
    try_reserve_untracked(segments, 1, "classic Tier-1 segment metadata")?;
    data.extend(segment_data);
    segments.push(EncodedCodeBlockSegment {
        data_offset,
        data_length,
        start_coding_pass,
        end_coding_pass,
        distortion_delta,
        use_arithmetic,
    });
    Ok(())
}
