// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible collection of owned code-block inputs for batched decoding.

use super::{
    add_roi_shift_to_bitplanes, code_block_required_by_index, collect_classic_code_block_data,
    ht_block_decode, ht_code_block_has_decodable_passes, ComponentInfo, DecodeAllocationBudget,
    DecompositionStorage, Header, Result, SubBand, Vec,
};
use crate::error::ValidationError;
use crate::J2kCodeBlockSegment;

pub(super) struct PendingHtBlock {
    pub(super) combined: ht_block_decode::CombinedCodeBlockData,
    pub(super) output_x: u32,
    pub(super) output_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) missing_bit_planes: u8,
    pub(super) number_of_coding_passes: u8,
}

pub(super) struct PendingClassicBlock {
    pub(super) combined_data: Vec<u8>,
    pub(super) segments: Vec<J2kCodeBlockSegment>,
    pub(super) output_x: u32,
    pub(super) output_y: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) missing_bit_planes: u8,
    pub(super) number_of_coding_passes: u8,
}

pub(super) fn count_classic_code_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> Result<usize> {
    let mut count = 0_usize;
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block_idx in precinct.code_blocks.clone() {
            let code_block = &storage.code_blocks[code_block_idx];
            if code_block_required_by_index(storage, sub_band_idx, code_block) {
                count = count.checked_add(1).ok_or(ValidationError::ImageTooLarge)?;
            }
        }
    }
    Ok(count)
}

pub(super) fn collect_pending_classic_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    component_info: &ComponentInfo,
    storage: &DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<PendingClassicBlock>> {
    let block_count = count_classic_code_blocks(sub_band_idx, sub_band, storage)?;
    let mut pending_blocks = Vec::new();
    budget.reserve_new(&mut pending_blocks, block_count)?;
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                continue;
            }
            let (combined_data, segments) = collect_classic_code_block_data(
                code_block,
                &component_info.coding_style.parameters.code_block_style,
                storage,
                budget,
            )?;
            pending_blocks.push(PendingClassicBlock {
                combined_data,
                segments,
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
            });
        }
    }
    Ok(pending_blocks)
}

pub(super) fn count_ht_code_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
) -> Result<usize> {
    let mut count = 0_usize;
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block_idx in precinct.code_blocks.clone() {
            let code_block = &storage.code_blocks[code_block_idx];
            if code_block_required_by_index(storage, sub_band_idx, code_block)
                && code_block.number_of_coding_passes > 0
            {
                count = count.checked_add(1).ok_or(ValidationError::ImageTooLarge)?;
            }
        }
    }
    Ok(count)
}

pub(super) fn collect_pending_ht_blocks(
    sub_band_idx: usize,
    sub_band: &SubBand,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    num_bitplanes: u8,
    roi_shift: u8,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<PendingHtBlock>> {
    let coded_bitplanes = add_roi_shift_to_bitplanes(num_bitplanes, roi_shift, 31)?;
    let block_count = count_ht_code_blocks(sub_band_idx, sub_band, storage)?;
    let mut pending_blocks = Vec::new();
    budget.reserve_new(&mut pending_blocks, block_count)?;
    for precinct in sub_band
        .precincts
        .clone()
        .map(|idx| &storage.precincts[idx])
    {
        for code_block in precinct
            .code_blocks
            .clone()
            .map(|idx| &storage.code_blocks[idx])
        {
            if !code_block_required_by_index(storage, sub_band_idx, code_block) {
                continue;
            }
            if !ht_code_block_has_decodable_passes(code_block, coded_bitplanes, header.strict)? {
                continue;
            }

            pending_blocks.push(PendingHtBlock {
                combined: ht_block_decode::collect_code_block_data(code_block, storage, budget)?,
                output_x: code_block.rect.x0 - sub_band.rect.x0,
                output_y: code_block.rect.y0 - sub_band.rect.y0,
                width: code_block.rect.width(),
                height: code_block.rect.height(),
                missing_bit_planes: code_block.missing_bit_planes,
                number_of_coding_passes: code_block.number_of_coding_passes,
            });
        }
    }
    Ok(pending_blocks)
}
