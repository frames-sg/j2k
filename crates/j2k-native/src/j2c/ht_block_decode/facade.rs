// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::build::CodeBlock;
use super::super::decode::DecompositionStorage;
use super::collect_code_block_segments;
use super::pipeline::PHASE_LIMIT_MAGREF;
use super::state::{HtBlockDecodeContext, HtBlockDecodeStats};
use super::validation::decode_segments_validated_with_scratch_for_phase;
use crate::error::{bail, DecodingError, Result};

#[expect(
    clippy::cast_possible_wrap,
    reason = "the sign bit is masked before converting the at-most 31-bit coefficient magnitude"
)]
pub(crate) fn coefficient_to_i32(value: u32, k_max: u8) -> i32 {
    let shift = 31_u32.saturating_sub(u32::from(k_max));
    let magnitude = ((value & 0x7FFF_FFFF) >> shift) as i32;

    if (value & 0x8000_0000) != 0 {
        -magnitude
    } else {
        magnitude
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "this internal facade preserves the established decode configuration and observer inputs"
)]
pub(crate) fn decode_with_stats(
    code_block: &CodeBlock,
    total_bitplanes: u8,
    stripe_causal: bool,
    ctx: &mut HtBlockDecodeContext,
    storage: &DecompositionStorage<'_>,
    strict: bool,
    stats: Option<&mut HtBlockDecodeStats>,
    profile_enabled: bool,
) -> Result<()> {
    ctx.reset(code_block);

    if total_bitplanes == 0 {
        return Ok(());
    }

    if total_bitplanes > 31 {
        bail!(DecodingError::TooManyBitplanes);
    }

    let actual_bitplanes = if strict {
        total_bitplanes
            .checked_sub(code_block.missing_bit_planes)
            .ok_or(DecodingError::InvalidBitplaneCount)?
    } else {
        total_bitplanes.saturating_sub(code_block.missing_bit_planes)
    };

    let max_coding_passes = if actual_bitplanes == 0 {
        0
    } else {
        1 + 3 * (actual_bitplanes - 1)
    };

    if code_block.number_of_coding_passes > max_coding_passes && strict {
        bail!(DecodingError::TooManyCodingPasses);
    }

    if code_block.number_of_coding_passes == 0 || actual_bitplanes == 0 {
        return Ok(());
    }

    let segments = collect_code_block_segments(code_block, storage)?;
    decode_segments_validated_with_scratch_for_phase::<PHASE_LIMIT_MAGREF>(
        &segments,
        code_block.missing_bit_planes,
        total_bitplanes,
        code_block.number_of_coding_passes,
        stripe_causal,
        strict,
        &mut ctx.coefficients,
        code_block.rect.width(),
        code_block.rect.height(),
        code_block.rect.width(),
        &mut ctx.scratch,
        stats,
        profile_enabled,
    )
}
