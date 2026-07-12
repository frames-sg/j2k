// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::bitplane_encode::EncodedCodeBlock;
use super::allocation::ht_worker_allocation;
use super::cleanup::{max_nonzero_magnitude_view, try_encode_cleanup_segment_from_view};
use super::refinement::try_encode_refinement_segment_view;
use crate::j2c::coefficient_view::CoefficientBlockView;
use crate::j2c::encode::allocation::try_untracked_vec;
use crate::{EncodeError, EncodeResult};

pub(super) const MAX_HT_BITPLANES: u8 = 31;

pub(crate) fn try_encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    try_encode_code_block_with_passes(coefficients, width, height, total_bitplanes, 1)
}

pub(crate) fn try_encode_code_block_with_passes(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    let coefficients =
        CoefficientBlockView::try_contiguous(coefficients, width as usize, height as usize)?;
    try_encode_code_block_view(coefficients, total_bitplanes, target_coding_passes)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a u32 magnitude has at most 32 bitplanes, which always fits the u8 metadata field"
)]
pub(crate) fn try_encode_code_block_view(
    coefficients: CoefficientBlockView<'_, i32>,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> EncodeResult<EncodedCodeBlock> {
    if total_bitplanes == 0 || total_bitplanes > MAX_HT_BITPLANES {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports 1..=31 bitplanes",
        });
    }
    if target_coding_passes == 0 || target_coding_passes > 3 {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K scalar encoder currently supports cleanup, sigprop, and one magref refinement pass",
        });
    }
    let allocation = ht_worker_allocation(
        coefficients.width(),
        coefficients.height(),
        target_coding_passes,
    )?;

    let Some(max_magnitude) = max_nonzero_magnitude_view(coefficients) else {
        return Ok(EncodedCodeBlock {
            data: Vec::new(),
            num_coding_passes: 0,
            num_zero_bitplanes: total_bitplanes,
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
        });
    };

    let block_bitplanes = (u32::BITS - max_magnitude.leading_zeros()) as u8;
    if block_bitplanes > total_bitplanes {
        return Err(EncodeError::InvalidInput {
            what: "HTJ2K block magnitude exceeds configured bitplane count",
        });
    }

    let effective_coding_passes = if target_coding_passes >= 2 && total_bitplanes > 1 {
        target_coding_passes
    } else {
        1
    };
    let cleanup_bitplanes = if effective_coding_passes >= 2 { 2 } else { 1 };
    let missing_msbs = total_bitplanes.saturating_sub(cleanup_bitplanes);
    let cleanup =
        try_encode_cleanup_segment_from_view(coefficients, missing_msbs, total_bitplanes)?;
    if cleanup.len() > allocation.cleanup_bytes {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K cleanup segment exceeded its checked bound",
        });
    }
    let ht_cleanup_length =
        u32::try_from(cleanup.len()).map_err(|_| EncodeError::InternalInvariant {
            what: "HTJ2K cleanup segment exceeds u32 length",
        })?;
    let refinement = if effective_coding_passes > 1 {
        try_encode_refinement_segment_view(
            coefficients,
            1_i32 << (cleanup_bitplanes - 1),
            effective_coding_passes,
            allocation,
        )?
    } else {
        Vec::new()
    };
    let ht_refinement_length =
        u32::try_from(refinement.len()).map_err(|_| EncodeError::InternalInvariant {
            what: "HTJ2K refinement segment exceeds u32 length",
        })?;
    let combined_len =
        cleanup
            .len()
            .checked_add(refinement.len())
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HTJ2K combined block payload",
            })?;
    if combined_len > allocation.output_bytes {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K block output exceeded its checked bound",
        });
    }
    let mut data = try_untracked_vec(combined_len, "HTJ2K block output")?;
    data.extend_from_slice(&cleanup);
    data.extend_from_slice(&refinement);

    Ok(EncodedCodeBlock {
        data,
        num_coding_passes: effective_coding_passes,
        num_zero_bitplanes: missing_msbs,
        ht_cleanup_length,
        ht_refinement_length,
    })
}

#[cfg(test)]
mod legacy;
#[cfg(test)]
pub(crate) use legacy::{encode_code_block, encode_code_block_view, encode_code_block_with_passes};
