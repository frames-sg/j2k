// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::build::CodeBlock;
use super::super::decode::DecompositionStorage;
use super::pipeline::PHASE_LIMIT_MAGREF;
use super::state::HtBlockDecodeContext;
use super::stats::HtBlockDecodeStats;
use super::validation::{
    decode_segments_validated_with_scratch_for_phase, validate_combined_decode,
};
use super::{collect_code_block_data_into, HtCodeBlockSegments};
use crate::error::Result;

#[cfg(feature = "simd")]
#[derive(Clone, Copy)]
pub(crate) struct PreparedDirectCleanup {
    cleanup_length: usize,
}

#[cfg(feature = "simd")]
pub(crate) fn prepare_direct_cleanup(
    code_block: &CodeBlock,
    total_bitplanes: u8,
    roi_shift: u8,
    ctx: &mut HtBlockDecodeContext,
    storage: &DecompositionStorage<'_>,
    strict: bool,
    profile_enabled: bool,
) -> Result<Option<PreparedDirectCleanup>> {
    if profile_enabled
        || roi_shift != 0
        || code_block.number_of_coding_passes != 1
        || !super::validation::supports_cleanup_to_f32(
            code_block.missing_bit_planes,
            code_block.rect.width(),
        )
        || !validate_combined_decode(
            code_block.missing_bit_planes,
            total_bitplanes,
            code_block.number_of_coding_passes,
            strict,
        )?
    {
        return Ok(None);
    }

    ctx.reset_for_direct_output(code_block)?;
    let (cleanup_length, refinement_length) =
        collect_code_block_data_into(code_block, storage, &mut ctx.segment_data)?;
    if refinement_length != 0 {
        return Ok(None);
    }
    Ok(Some(PreparedDirectCleanup {
        cleanup_length: usize::try_from(cleanup_length)
            .map_err(|_| crate::ValidationError::ImageTooLarge)?,
    }))
}

#[cfg(feature = "simd")]
#[expect(
    clippy::too_many_arguments,
    reason = "the prepared direct decode keeps final output conversion explicit"
)]
pub(crate) fn decode_prepared_cleanup_to_f32<const IRREVERSIBLE_MIDPOINT: bool>(
    prepared: PreparedDirectCleanup,
    code_block: &CodeBlock,
    total_bitplanes: u8,
    strict: bool,
    output: &mut [f32],
    output_stride: u32,
    dequantization_step: f32,
    ctx: &mut HtBlockDecodeContext,
    stats: &mut HtBlockDecodeStats,
) -> Result<()> {
    stats.record_block(prepared.cleanup_length, 0);
    let decoded = super::validation::decode_cleanup_validated_to_f32::<IRREVERSIBLE_MIDPOINT>(
        &ctx.segment_data[..prepared.cleanup_length],
        code_block.missing_bit_planes,
        total_bitplanes,
        strict,
        output,
        code_block.rect.width(),
        code_block.rect.height(),
        output_stride,
        dequantization_step,
        &mut ctx.scratch,
    )?;
    debug_assert!(decoded, "prepared cleanup metadata was validated");
    Ok(())
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
    ctx.reset_for_overwrite(code_block)?;

    // The same validation runs again below; a block it skips decodes to zeros.
    if !validate_combined_decode(
        code_block.missing_bit_planes,
        total_bitplanes,
        code_block.number_of_coding_passes,
        strict,
    )? {
        ctx.zero_coefficients();
        return Ok(());
    }

    let (cleanup_length, refinement_length) =
        collect_code_block_data_into(code_block, storage, &mut ctx.segment_data)?;
    let segments = HtCodeBlockSegments::from_combined_payload(
        &ctx.segment_data,
        cleanup_length,
        refinement_length,
    )?;
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
