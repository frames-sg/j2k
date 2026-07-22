// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned classic code-block payload assembly for direct and batched decode plans.

use super::{bail, DecodeAllocationBudget, DecodingError, DecompositionStorage, Result};
use crate::error::ValidationError;
use crate::j2c::build::CodeBlock;
use crate::j2c::codestream::CodeBlockStyle;
use crate::{try_reserve_decode_elements, J2kCodeBlockSegment};
use alloc::vec::Vec;

#[derive(Clone, Copy)]
struct ClassicAllocationCounts {
    data_bytes: usize,
    segments: usize,
    fragments: usize,
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
pub(crate) fn collect_classic_code_block_data(
    code_block: &CodeBlock,
    style: &CodeBlockStyle,
    storage: &DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<(Vec<u8>, Vec<J2kCodeBlockSegment>)> {
    let counts = classic_allocation_counts(code_block, *style, storage)?;
    budget.include_elements::<u8>(counts.data_bytes)?;
    budget.include_elements::<J2kCodeBlockSegment>(counts.segments)?;

    let mut combined_data = Vec::new();
    let mut collected_segments = Vec::new();
    try_reserve_decode_elements(&mut combined_data, counts.data_bytes)?;
    try_reserve_decode_elements(&mut collected_segments, counts.segments)?;
    budget.include_capacity_overage::<u8>(counts.data_bytes, combined_data.capacity())?;
    budget.include_capacity_overage::<J2kCodeBlockSegment>(
        counts.segments,
        collected_segments.capacity(),
    )?;

    let mut last_segment_idx = 0u8;
    let mut segment_start_offset = 0usize;
    let mut segment_start_coding_pass = 0u8;
    let mut coding_passes = 0u8;
    let is_normal_mode =
        !style.selective_arithmetic_coding_bypass && !style.termination_on_each_pass;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            if segment.idx != last_segment_idx {
                validate_next_segment_index(last_segment_idx, segment.idx)?;
                if coding_passes > segment_start_coding_pass
                    || combined_data.len() > segment_start_offset
                {
                    collected_segments.push(classic_segment(
                        *style,
                        segment_start_offset,
                        combined_data.len(),
                        segment_start_coding_pass,
                        coding_passes,
                    )?);
                }
                segment_start_offset = combined_data.len();
                segment_start_coding_pass = coding_passes;
                last_segment_idx = segment.idx;
            }

            combined_data.extend_from_slice(segment.data);
            coding_passes = coding_passes.saturating_add(segment.coding_pases);
        }
    }

    if coding_passes > segment_start_coding_pass || combined_data.len() > segment_start_offset {
        collected_segments.push(classic_segment(
            *style,
            segment_start_offset,
            combined_data.len(),
            segment_start_coding_pass,
            coding_passes,
        )?);
    }

    if is_normal_mode {
        collected_segments.clear();
        collected_segments.push(J2kCodeBlockSegment {
            data_offset: 0,
            data_length: u32::try_from(combined_data.len())
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
            start_coding_pass: 0,
            end_coding_pass: coding_passes,
            use_arithmetic: true,
        });
    }

    if coding_passes != code_block.number_of_coding_passes {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    if combined_data.len() != counts.data_bytes || collected_segments.len() > counts.segments {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    Ok((combined_data, collected_segments))
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the stable codec boundary borrows shared Copy metadata used across nested calls"
)]
pub(crate) fn collect_referenced_classic_code_block_data(
    code_block: &CodeBlock,
    style: &CodeBlockStyle,
    storage: &DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
    mut append_fragment: impl FnMut(&[u8]) -> Result<()>,
) -> Result<(usize, usize, Vec<J2kCodeBlockSegment>)> {
    let counts = classic_allocation_counts(code_block, *style, storage)?;
    budget.include_elements::<J2kCodeBlockSegment>(counts.segments)?;

    let mut collected_segments = Vec::new();
    try_reserve_decode_elements(&mut collected_segments, counts.segments)?;
    budget.include_capacity_overage::<J2kCodeBlockSegment>(
        counts.segments,
        collected_segments.capacity(),
    )?;

    let mut data_bytes = 0usize;
    let mut fragment_count = 0usize;
    let mut last_segment_idx = 0u8;
    let mut segment_start_offset = 0usize;
    let mut segment_start_coding_pass = 0u8;
    let mut coding_passes = 0u8;
    let is_normal_mode =
        !style.selective_arithmetic_coding_bypass && !style.termination_on_each_pass;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            if segment.idx != last_segment_idx {
                validate_next_segment_index(last_segment_idx, segment.idx)?;
                if coding_passes > segment_start_coding_pass || data_bytes > segment_start_offset {
                    collected_segments.push(classic_segment(
                        *style,
                        segment_start_offset,
                        data_bytes,
                        segment_start_coding_pass,
                        coding_passes,
                    )?);
                }
                segment_start_offset = data_bytes;
                segment_start_coding_pass = coding_passes;
                last_segment_idx = segment.idx;
            }

            if !segment.data.is_empty() {
                append_fragment(segment.data)?;
                fragment_count = fragment_count
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
            data_bytes = data_bytes
                .checked_add(segment.data.len())
                .ok_or(ValidationError::ImageTooLarge)?;
            coding_passes = coding_passes.saturating_add(segment.coding_pases);
        }
    }

    if coding_passes > segment_start_coding_pass || data_bytes > segment_start_offset {
        collected_segments.push(classic_segment(
            *style,
            segment_start_offset,
            data_bytes,
            segment_start_coding_pass,
            coding_passes,
        )?);
    }

    if is_normal_mode {
        collected_segments.clear();
        collected_segments.push(J2kCodeBlockSegment {
            data_offset: 0,
            data_length: u32::try_from(data_bytes)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
            start_coding_pass: 0,
            end_coding_pass: coding_passes,
            use_arithmetic: true,
        });
    }

    if coding_passes != code_block.number_of_coding_passes
        || data_bytes != counts.data_bytes
        || fragment_count != counts.fragments
        || collected_segments.len() > counts.segments
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    Ok((fragment_count, data_bytes, collected_segments))
}

fn classic_allocation_counts(
    code_block: &CodeBlock,
    style: CodeBlockStyle,
    storage: &DecompositionStorage<'_>,
) -> Result<ClassicAllocationCounts> {
    let mut data_bytes = 0_usize;
    let mut segment_count = 0_usize;
    let mut last_segment_idx = 0_u8;
    let mut segment_start_offset = 0_usize;
    let mut segment_start_coding_pass = 0_u8;
    let mut coding_passes = 0_u8;
    let mut fragment_count = 0_usize;

    for layer in &storage.layers[code_block.layers.start..code_block.layers.end] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };
        for segment in &storage.segments[range] {
            if segment.idx != last_segment_idx {
                validate_next_segment_index(last_segment_idx, segment.idx)?;
                if coding_passes > segment_start_coding_pass || data_bytes > segment_start_offset {
                    segment_count = segment_count
                        .checked_add(1)
                        .ok_or(ValidationError::ImageTooLarge)?;
                }
                segment_start_offset = data_bytes;
                segment_start_coding_pass = coding_passes;
                last_segment_idx = segment.idx;
            }
            data_bytes = data_bytes
                .checked_add(segment.data.len())
                .ok_or(ValidationError::ImageTooLarge)?;
            if !segment.data.is_empty() {
                fragment_count = fragment_count
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
            coding_passes = coding_passes.saturating_add(segment.coding_pases);
        }
    }

    if coding_passes > segment_start_coding_pass || data_bytes > segment_start_offset {
        segment_count = segment_count
            .checked_add(1)
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    if !style.selective_arithmetic_coding_bypass && !style.termination_on_each_pass {
        segment_count = 1;
    }

    Ok(ClassicAllocationCounts {
        data_bytes,
        segments: segment_count,
        fragments: fragment_count,
    })
}

fn validate_next_segment_index(previous: u8, current: u8) -> Result<()> {
    if previous.checked_add(1) != Some(current) {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok(())
}

fn classic_segment(
    style: CodeBlockStyle,
    start_offset: usize,
    end_offset: usize,
    start_coding_pass: u8,
    end_coding_pass: u8,
) -> Result<J2kCodeBlockSegment> {
    let data_offset =
        u32::try_from(start_offset).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let data_length = u32::try_from(
        end_offset
            .checked_sub(start_offset)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?,
    )
    .map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
    let use_arithmetic = !style.selective_arithmetic_coding_bypass
        || start_coding_pass <= 9
        || start_coding_pass.is_multiple_of(3);
    Ok(J2kCodeBlockSegment {
        data_offset,
        data_length,
        start_coding_pass,
        end_coding_pass,
        use_arithmetic,
    })
}
