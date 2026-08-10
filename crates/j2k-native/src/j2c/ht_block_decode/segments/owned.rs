// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::{selected_segment_indices, CombinedCodeBlockData};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::build::CodeBlock;
use crate::j2c::decode::{DecodeAllocationBudget, DecompositionStorage};
use crate::try_reserve_decode_elements;

pub(crate) fn selected_code_block_segment_lengths(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'_>,
) -> Result<(usize, usize)> {
    let (cleanup_index, refinement_index) = selected_segment_indices(code_block)?;
    let mut cleanup_length = None;
    let mut refinement_length = 0usize;

    for layer in &storage.layers[code_block.layers.clone()] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };

        for segment in &storage.segments[range] {
            match segment.idx {
                idx if idx == cleanup_index && cleanup_length.is_none() => {
                    cleanup_length = Some(segment.data.len());
                }
                idx if idx == cleanup_index => {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
                idx if idx == refinement_index && cleanup_length.is_some() => {
                    refinement_length = refinement_length
                        .checked_add(segment.data.len())
                        .ok_or(ValidationError::ImageTooLarge)?;
                }
                idx if idx == refinement_index => {
                    bail!(DecodingError::CodeBlockDecodeFailure);
                }
                idx if idx < cleanup_index => {}
                _ => bail!(DecodingError::CodeBlockDecodeFailure),
            }
        }
    }

    let cleanup_length = cleanup_length.ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok((cleanup_length, refinement_length))
}

pub(crate) fn collect_code_block_data_into(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'_>,
    data: &mut Vec<u8>,
) -> Result<(u32, u32)> {
    let (cleanup_index, refinement_index) = selected_segment_indices(code_block)?;
    let (cleanup_length, refinement_length) =
        selected_code_block_segment_lengths(code_block, storage)?;
    let total_length = cleanup_length
        .checked_add(refinement_length)
        .ok_or(ValidationError::ImageTooLarge)?;
    if data.capacity() < total_length {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    data.clear();
    for layer in &storage.layers[code_block.layers.clone()] {
        let Some(range) = layer.segments.clone() else {
            continue;
        };
        for segment in &storage.segments[range] {
            if segment.idx == cleanup_index || segment.idx == refinement_index {
                data.extend_from_slice(segment.data);
            }
        }
    }

    if data.len() != total_length {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    Ok((
        u32::try_from(cleanup_length).map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
        u32::try_from(refinement_length).map_err(|_| DecodingError::CodeBlockDecodeFailure)?,
    ))
}

pub(crate) fn collect_code_block_data<'a>(
    code_block: &CodeBlock,
    storage: &'a DecompositionStorage<'a>,
    budget: &mut DecodeAllocationBudget,
) -> Result<CombinedCodeBlockData> {
    let (cleanup_length, refinement_length) =
        selected_code_block_segment_lengths(code_block, storage)?;
    let data_len = cleanup_length
        .checked_add(refinement_length)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    budget.include_elements::<u8>(data_len)?;
    let mut data = Vec::new();
    try_reserve_decode_elements(&mut data, data_len)?;
    budget.include_capacity_overage::<u8>(data_len, data.capacity())?;
    let (cleanup_length, refinement_length) =
        collect_code_block_data_into(code_block, storage, &mut data)?;

    Ok(CombinedCodeBlockData {
        data,
        cleanup_length,
        refinement_length,
    })
}
