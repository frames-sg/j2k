// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned classic code-block payload assembly for direct and batched decode plans.

use super::{bail, DecodeAllocationBudget, DecodingError, DecompositionStorage, Result};
use crate::error::ValidationError;
use crate::j2c::build::{CodeBlock, Layer, Segment};
use crate::j2c::codestream::CodeBlockStyle;
use crate::{try_reserve_decode_elements, J2kCodeBlockSegment};
use alloc::vec::Vec;

#[derive(Clone, Copy)]
pub(crate) struct ClassicAllocationCounts {
    pub(crate) data_bytes: usize,
    /// Codeword segments.
    pub(crate) segments: usize,
    /// Segments that hold data.
    pub(crate) fragments: usize,
    /// Coding passes over all segments.
    #[cfg_attr(
        not(feature = "parallel"),
        expect(dead_code, reason = "only the tile-wide route reads the pass count")
    )]
    pub(crate) coding_passes: u8,
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

    let (data_bytes, _, coding_passes) = walk_classic_segments(
        code_block,
        *style,
        (&storage.layers, &storage.segments),
        |fragment| extend_reserved(&mut combined_data, fragment),
        |segment| push_reserved(&mut collected_segments, segment),
    )?;
    if coding_passes != code_block.number_of_coding_passes
        || data_bytes != counts.data_bytes
        || collected_segments.len() > counts.segments
    {
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
    append_fragment: impl FnMut(&[u8]) -> Result<()>,
) -> Result<(usize, usize, Vec<J2kCodeBlockSegment>)> {
    let counts = classic_allocation_counts(code_block, *style, storage)?;
    budget.include_elements::<J2kCodeBlockSegment>(counts.segments)?;

    let mut collected_segments = Vec::new();
    try_reserve_decode_elements(&mut collected_segments, counts.segments)?;
    budget.include_capacity_overage::<J2kCodeBlockSegment>(
        counts.segments,
        collected_segments.capacity(),
    )?;

    let (data_bytes, fragment_count, coding_passes) = walk_classic_segments(
        code_block,
        *style,
        (&storage.layers, &storage.segments),
        append_fragment,
        |segment| push_reserved(&mut collected_segments, segment),
    )?;
    if coding_passes != code_block.number_of_coding_passes
        || data_bytes != counts.data_bytes
        || fragment_count != counts.fragments
        || collected_segments.len() > counts.segments
    {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    Ok((fragment_count, data_bytes, collected_segments))
}

/// Payload bytes, segments, data fragments and coding passes of a classic
/// block, rejecting segment indices that skip one, without allocating.
pub(crate) fn classic_allocation_counts(
    code_block: &CodeBlock,
    style: CodeBlockStyle,
    storage: &DecompositionStorage<'_>,
) -> Result<ClassicAllocationCounts> {
    let mut segments = 0_usize;
    let (data_bytes, fragments, coding_passes) = walk_classic_segments(
        code_block,
        style,
        (&storage.layers, &storage.segments),
        |_| Ok(()),
        |_| {
            segments = segments
                .checked_add(1)
                .ok_or(ValidationError::ImageTooLarge)?;
            Ok(())
        },
    )?;
    Ok(ClassicAllocationCounts {
        data_bytes,
        segments,
        fragments,
        coding_passes,
    })
}

/// Walk the codeword segments of a classic block without allocating, as
/// [`collect_classic_code_block_data`] collects them: pass each nonempty
/// data fragment, layer by layer, to `fragment`, and each codeword segment,
/// its contributions across layers joined, to `segment`, at its offset into
/// the fragments joined. A segment that holds coding passes and no bytes is
/// kept; one that holds neither is not. A block of one codeword segment is
/// one arithmetic segment over all its passes. Returns the payload bytes and
/// data fragments, and the coding passes, rejecting segment indices that
/// skip one.
pub(crate) fn walk_classic_segments(
    code_block: &CodeBlock,
    style: CodeBlockStyle,
    (layers, segments): (&[Layer], &[Segment<'_>]),
    mut fragment: impl FnMut(&[u8]) -> Result<()>,
    mut segment: impl FnMut(J2kCodeBlockSegment) -> Result<()>,
) -> Result<(usize, usize, u8)> {
    let single = style.single_codeword_segment();
    let mut data_bytes = 0usize;
    let mut fragments = 0usize;
    let mut last_segment_idx = 0u8;
    let mut segment_start_offset = 0usize;
    let mut segment_start_coding_pass = 0u8;
    let mut coding_passes = 0u8;
    let layers = layers
        .get(code_block.layers.clone())
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    for layer in layers {
        let Some(range) = layer.segments() else {
            continue;
        };
        let records = segments
            .get(range)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        for record in records {
            if record.idx != last_segment_idx {
                validate_next_segment_index(last_segment_idx, record.idx)?;
                if !single
                    && (coding_passes > segment_start_coding_pass
                        || data_bytes > segment_start_offset)
                {
                    segment(classic_segment(
                        style,
                        segment_start_offset,
                        data_bytes,
                        segment_start_coding_pass,
                        coding_passes,
                    )?)?;
                }
                segment_start_offset = data_bytes;
                segment_start_coding_pass = coding_passes;
                last_segment_idx = record.idx;
            }
            if !record.data.is_empty() {
                fragment(record.data)?;
                fragments = fragments
                    .checked_add(1)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
            data_bytes = data_bytes
                .checked_add(record.data.len())
                .ok_or(ValidationError::ImageTooLarge)?;
            coding_passes = coding_passes.saturating_add(record.coding_pases);
        }
    }
    if single {
        segment(J2kCodeBlockSegment {
            data_offset: 0,
            data_length: u32::try_from(data_bytes)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
            start_coding_pass: 0,
            end_coding_pass: coding_passes,
            use_arithmetic: true,
        })?;
    } else if coding_passes > segment_start_coding_pass || data_bytes > segment_start_offset {
        segment(classic_segment(
            style,
            segment_start_offset,
            data_bytes,
            segment_start_coding_pass,
            coding_passes,
        )?)?;
    }
    Ok((data_bytes, fragments, coding_passes))
}

/// Append `data` to `values` within the capacity reserved for it.
pub(crate) fn extend_reserved(values: &mut Vec<u8>, data: &[u8]) -> Result<()> {
    if values.capacity() - values.len() < data.len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    values.extend_from_slice(data);
    Ok(())
}

/// Push `value` onto `values` within the capacity reserved for it.
pub(crate) fn push_reserved<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    if values.len() == values.capacity() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    values.push(value);
    Ok(())
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
