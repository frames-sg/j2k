// SPDX-License-Identifier: MIT OR Apache-2.0

//! Validation shared by preencoded and accelerator-produced Tier-1 metadata.

use crate::{EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kCodeBlockSegment, J2kCodeBlockStyle};

const MAX_PACKET_CODING_PASSES: u8 = 164;

pub(super) fn validate_ht_code_block(
    block: &EncodedHtJ2kCodeBlock,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    validate_ht_code_block_metadata(
        block.data.len(),
        block.cleanup_length,
        block.refinement_length,
        block.num_coding_passes,
        block.num_zero_bitplanes,
        total_bitplanes,
    )
}

pub(super) fn validate_accelerated_ht_code_block(
    block: &EncodedHtJ2kCodeBlock,
    total_bitplanes: u8,
    requested_coding_passes: u8,
) -> Result<(), &'static str> {
    validate_ht_code_block(block, total_bitplanes)?;
    if block.num_coding_passes == 0 {
        return Ok(());
    }
    let expected_coding_passes = if total_bitplanes == 1 {
        1
    } else {
        requested_coding_passes
    };
    if block.num_coding_passes != expected_coding_passes {
        return Err("accelerated HT code-block coding pass count differs from request");
    }
    Ok(())
}

pub(super) fn validate_accelerated_ht_job_output(
    block: &EncodedHtJ2kCodeBlock,
    coefficients: &[i32],
    total_bitplanes: u8,
    requested_coding_passes: u8,
) -> Result<(), &'static str> {
    validate_accelerated_ht_code_block(block, total_bitplanes, requested_coding_passes)?;
    let input_is_zero = coefficients.iter().all(|&coefficient| coefficient == 0);
    if block.num_coding_passes == 0 && !input_is_zero {
        return Err("accelerated HT code-block omitted nonzero coefficients");
    }
    if block.num_coding_passes != 0 && input_is_zero {
        return Err("accelerated HT code-block encoded an all-zero input");
    }
    Ok(())
}

pub(super) fn validate_ht_code_block_metadata(
    data_len: usize,
    cleanup_length: u32,
    refinement_length: u32,
    num_coding_passes: u8,
    num_zero_bitplanes: u8,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    let data_len = u32::try_from(data_len).map_err(|_| "HTJ2K payload too large")?;
    if num_coding_passes == 0 {
        if data_len != 0 || cleanup_length != 0 || refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if num_coding_passes > MAX_PACKET_CODING_PASSES {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = cleanup_length
        .checked_add(refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    if cleanup_length == 0 {
        return Err("HTJ2K cleanup segment is missing");
    }
    if num_coding_passes == 1 {
        if refinement_length != 0 {
            return Err("single-pass HTJ2K code-block must not carry refinement bytes");
        }
        return Ok(());
    }
    if refinement_length == 0 {
        return Err("HTJ2K refinement segment is missing");
    }
    if !(2..65_535).contains(&cleanup_length) {
        return Err("HTJ2K cleanup segment length is out of range");
    }
    if refinement_length >= 2_047 {
        return Err("HTJ2K refinement segment length is out of range");
    }
    Ok(())
}

pub(super) fn validate_accelerated_classic_code_block(
    block: &EncodedJ2kCodeBlock,
    coefficients: &[i32],
    total_bitplanes: u8,
    style: J2kCodeBlockStyle,
) -> Result<(), &'static str> {
    let coding_passes = block.number_of_coding_passes;
    if coding_passes == 0 {
        if !block.data.is_empty() || !block.segments.is_empty() {
            return Err("empty accelerated classic code-block carries payload metadata");
        }
        if block.missing_bit_planes != total_bitplanes {
            return Err("empty accelerated classic code-block zero-bitplane count mismatch");
        }
        if coefficients.iter().any(|&coefficient| coefficient != 0) {
            return Err("accelerated classic code-block omitted nonzero coefficients");
        }
        return Ok(());
    }
    if coding_passes > MAX_PACKET_CODING_PASSES {
        return Err("accelerated classic code-block coding pass count out of range");
    }
    if block.missing_bit_planes >= total_bitplanes {
        return Err("accelerated classic code-block zero-bitplane count out of range");
    }
    let Some(max_magnitude) = coefficients.iter().map(|value| value.unsigned_abs()).max() else {
        return Err("accelerated classic code-block encoded an all-zero input");
    };
    if max_magnitude == 0 {
        return Err("accelerated classic code-block encoded an all-zero input");
    }
    let coded_bitplanes = u8::try_from(u32::BITS - max_magnitude.leading_zeros())
        .map_err(|_| "accelerated classic code-block bitplane count exceeds u8")?;
    let expected_missing = total_bitplanes
        .checked_sub(coded_bitplanes)
        .ok_or("accelerated classic code-block input exceeds configured bitplanes")?;
    if block.missing_bit_planes != expected_missing {
        return Err("accelerated classic code-block zero-bitplane metadata mismatch");
    }
    validate_classic_pass_count(block, total_bitplanes)?;
    if block.data.is_empty() {
        return Err("accelerated classic code-block payload is missing");
    }
    if block.segments.is_empty() {
        return Err("accelerated classic code-block segment metadata is missing");
    }
    validate_classic_segments(block, style)
}

fn validate_classic_pass_count(
    block: &EncodedJ2kCodeBlock,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    let coded_bitplanes = u16::from(total_bitplanes - block.missing_bit_planes);
    let expected = 1 + 3 * (coded_bitplanes - 1);
    if u16::from(block.number_of_coding_passes) != expected {
        return Err("accelerated classic code-block pass/bitplane metadata mismatch");
    }
    Ok(())
}

fn validate_classic_segments(
    block: &EncodedJ2kCodeBlock,
    style: J2kCodeBlockStyle,
) -> Result<(), &'static str> {
    if block.segments.len() > usize::from(block.number_of_coding_passes) {
        return Err("accelerated classic code-block has too many segments");
    }
    if !style.termination_on_each_pass
        && !style.selective_arithmetic_coding_bypass
        && block.segments.len() != 1
    {
        return Err("accelerated classic code-block segments do not match coding style");
    }

    let mut data_end = 0u32;
    let mut pass_end = 0u8;
    let mut previous_segment: Option<&J2kCodeBlockSegment> = None;
    for segment in &block.segments {
        if segment.data_offset != data_end {
            return Err(
                "accelerated classic code-block segments do not cover payload contiguously",
            );
        }
        data_end = segment
            .data_offset
            .checked_add(segment.data_length)
            .ok_or("accelerated classic code-block segment range overflow")?;
        if segment.start_coding_pass != pass_end
            || segment.start_coding_pass >= segment.end_coding_pass
            || segment.end_coding_pass > block.number_of_coding_passes
        {
            return Err("accelerated classic code-block segments do not cover passes contiguously");
        }
        validate_classic_segment_style(segment, previous_segment, style)?;
        pass_end = segment.end_coding_pass;
        previous_segment = Some(segment);
    }
    let payload_len = u32::try_from(block.data.len())
        .map_err(|_| "accelerated classic code-block payload exceeds u32")?;
    if data_end != payload_len {
        return Err("accelerated classic code-block segments do not cover payload contiguously");
    }
    if pass_end != block.number_of_coding_passes {
        return Err("accelerated classic code-block segments do not cover passes contiguously");
    }
    Ok(())
}

fn validate_classic_segment_style(
    segment: &J2kCodeBlockSegment,
    previous: Option<&J2kCodeBlockSegment>,
    style: J2kCodeBlockStyle,
) -> Result<(), &'static str> {
    if style.termination_on_each_pass && segment.end_coding_pass - segment.start_coding_pass != 1 {
        return Err("accelerated classic code-block segments do not match coding style");
    }
    if (segment.start_coding_pass..segment.end_coding_pass)
        .any(|pass| classic_pass_uses_arithmetic(pass, style) != segment.use_arithmetic)
    {
        return Err("accelerated classic code-block segment coding mode mismatch");
    }
    if !style.termination_on_each_pass
        && previous.is_some_and(|prior| prior.use_arithmetic == segment.use_arithmetic)
    {
        return Err("accelerated classic code-block segments do not match coding style");
    }
    Ok(())
}

fn classic_pass_uses_arithmetic(pass: u8, style: J2kCodeBlockStyle) -> bool {
    !style.selective_arithmetic_coding_bypass || pass <= 9 || pass.is_multiple_of(3)
}
