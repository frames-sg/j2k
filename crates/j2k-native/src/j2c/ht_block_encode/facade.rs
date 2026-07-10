// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::bitplane_encode::EncodedCodeBlock;
use super::cleanup::{encode_cleanup_segment_from_coefficients, max_nonzero_magnitude};
use super::refinement::encode_refinement_segment;

pub(super) const MAX_HT_BITPLANES: u8 = 31;

pub(crate) fn encode_code_block(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
) -> Result<EncodedCodeBlock, &'static str> {
    encode_code_block_with_passes(coefficients, width, height, total_bitplanes, 1)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a u32 magnitude has at most 32 bitplanes, which always fits the u8 metadata field"
)]
pub(crate) fn encode_code_block_with_passes(
    coefficients: &[i32],
    width: u32,
    height: u32,
    total_bitplanes: u8,
    target_coding_passes: u8,
) -> Result<EncodedCodeBlock, &'static str> {
    if total_bitplanes == 0 || total_bitplanes > MAX_HT_BITPLANES {
        return Err("HTJ2K scalar encoder currently supports 1..=31 bitplanes");
    }
    if target_coding_passes == 0 || target_coding_passes > 3 {
        return Err("HTJ2K scalar encoder currently supports cleanup, sigprop, and one magref refinement pass");
    }

    let Some(max_magnitude) = max_nonzero_magnitude(coefficients) else {
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
        return Err("HTJ2K block magnitude exceeds configured bitplane count");
    }

    let effective_coding_passes = if target_coding_passes >= 2 && total_bitplanes > 1 {
        target_coding_passes
    } else {
        1
    };
    let cleanup_bitplanes = if effective_coding_passes >= 2 { 2 } else { 1 };
    let missing_msbs = total_bitplanes.saturating_sub(cleanup_bitplanes);
    let data = encode_cleanup_segment_from_coefficients(
        coefficients,
        missing_msbs,
        width as usize,
        height as usize,
        total_bitplanes,
    )?;
    let ht_cleanup_length =
        u32::try_from(data.len()).map_err(|_| "HTJ2K cleanup segment exceeds u32 length")?;
    let mut data = data;
    let ht_refinement_length = if effective_coding_passes > 1 {
        let refinement = encode_refinement_segment(
            coefficients,
            width as usize,
            height as usize,
            1_i32 << (cleanup_bitplanes - 1),
            effective_coding_passes,
        )?;
        let refinement_len = refinement.len();
        data.extend_from_slice(&refinement);
        u32::try_from(refinement_len).map_err(|_| "HTJ2K refinement segment exceeds u32 length")?
    } else {
        0_u32
    };

    Ok(EncodedCodeBlock {
        data,
        num_coding_passes: effective_coding_passes,
        num_zero_bitplanes: missing_msbs,
        ht_cleanup_length,
        ht_refinement_length,
    })
}
