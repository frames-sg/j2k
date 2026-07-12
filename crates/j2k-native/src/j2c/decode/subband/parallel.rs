// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible parallel code-block output assembly.

use super::pending::{PendingClassicBlock, PendingHtBlock};
use super::{DecodeAllocationBudget, DecompositionStorage, SubBand};
use crate::error::{bail, DecodingError, Result, ValidationError};
use crate::j2c::bitplane::classic_decode_workspace_bytes;
use crate::j2c::ht_block_decode::ht_decode_workspace_bytes;
use crate::{
    decode_ht_code_block_scalar_with_workspace, decode_j2k_code_block_scalar_with_workspace,
    try_reserve_decode_elements, try_resize_decode_elements, HtCodeBlockDecodeJob,
    HtCodeBlockDecodeWorkspace, J2kCodeBlockDecodeJob, J2kCodeBlockDecodeWorkspace,
    J2kCodeBlockStyle, J2kSubBandType,
};
use alloc::vec::Vec;
use rayon::prelude::*;

pub(crate) struct DecodedClassicBlock {
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coefficients: Vec<f32>,
}

pub(crate) struct DecodedHtBlock {
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) coefficients: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(super) struct ClassicParallelParameters {
    pub(super) sub_band_type: J2kSubBandType,
    pub(super) style: J2kCodeBlockStyle,
    pub(super) strict: bool,
    pub(super) total_bitplanes: u8,
    pub(super) roi_shift: u8,
    pub(super) dequantization_step: f32,
}

trait DecodedSubBandBlock {
    fn output_x(&self) -> u32;
    fn output_y(&self) -> u32;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn coefficients(&self) -> &[f32];
}

impl DecodedSubBandBlock for DecodedClassicBlock {
    fn output_x(&self) -> u32 {
        self.output_x
    }

    fn output_y(&self) -> u32 {
        self.output_y
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

impl DecodedSubBandBlock for DecodedHtBlock {
    fn output_x(&self) -> u32 {
        self.output_x
    }

    fn output_y(&self) -> u32 {
        self.output_y
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

pub(super) fn decode_classic_sub_band_blocks_parallel(
    pending_blocks: &[PendingClassicBlock],
    parameters: ClassicParallelParameters,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedClassicBlock>> {
    let mut decoded_blocks = preallocate_classic_outputs(pending_blocks, budget)?;
    let mut workspaces = preallocate_classic_workspaces(pending_blocks, budget)?;
    decoded_blocks
        .par_iter_mut()
        .zip(pending_blocks.par_iter())
        .zip(workspaces.par_iter_mut())
        .try_for_each(|((decoded, pending), workspace)| -> Result<()> {
            decode_j2k_code_block_scalar_with_workspace(
                J2kCodeBlockDecodeJob {
                    data: &pending.combined_data,
                    segments: &pending.segments,
                    width: pending.width,
                    height: pending.height,
                    output_stride: pending.width as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    total_bitplanes: parameters.total_bitplanes,
                    roi_shift: parameters.roi_shift,
                    sub_band_type: parameters.sub_band_type,
                    style: parameters.style,
                    strict: parameters.strict,
                    dequantization_step: parameters.dequantization_step,
                },
                &mut decoded.coefficients,
                workspace,
            )
        })?;
    Ok(decoded_blocks)
}

fn preallocate_classic_outputs(
    pending_blocks: &[PendingClassicBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedClassicBlock>> {
    let total_coefficients = pending_coefficient_count(
        pending_blocks
            .iter()
            .map(|pending| (pending.width, pending.height)),
    )?;
    budget.include_elements::<f32>(total_coefficients)?;

    let mut decoded_blocks = Vec::new();
    budget.reserve_new(&mut decoded_blocks, pending_blocks.len())?;
    for pending in pending_blocks {
        let coefficient_count = block_coefficient_count(pending.width, pending.height)?;
        let mut coefficients = Vec::new();
        try_resize_decode_elements(&mut coefficients, coefficient_count, 0.0)?;
        budget.include_capacity_overage::<f32>(coefficient_count, coefficients.capacity())?;
        decoded_blocks.push(DecodedClassicBlock {
            output_x: pending.output_x,
            output_y: pending.output_y,
            width: pending.width,
            height: pending.height,
            coefficients,
        });
    }
    Ok(decoded_blocks)
}

fn preallocate_classic_workspaces(
    pending_blocks: &[PendingClassicBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<J2kCodeBlockDecodeWorkspace>> {
    let planned_bytes =
        pending_blocks
            .iter()
            .try_fold(0usize, |bytes, pending| -> Result<usize> {
                bytes
                    .checked_add(classic_decode_workspace_bytes(
                        pending.width,
                        pending.height,
                    )?)
                    .ok_or_else(|| ValidationError::ImageTooLarge.into())
            })?;
    budget.include_bytes(planned_bytes)?;
    let mut workspaces = Vec::new();
    budget.reserve_new(&mut workspaces, pending_blocks.len())?;
    for pending in pending_blocks {
        let planned = classic_decode_workspace_bytes(pending.width, pending.height)?;
        let mut workspace = J2kCodeBlockDecodeWorkspace::default();
        workspace.prepare(pending.width, pending.height)?;
        let actual = workspace.allocated_bytes()?;
        if actual > planned {
            budget.include_bytes(actual - planned)?;
        }
        workspaces.push(workspace);
    }
    Ok(workspaces)
}

pub(super) fn decode_ht_sub_band_blocks_parallel(
    pending_blocks: &[PendingHtBlock],
    strict: bool,
    num_bitplanes: u8,
    roi_shift: u8,
    stripe_causal: bool,
    dequantization_step: f32,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedHtBlock>> {
    let mut decoded_blocks = preallocate_ht_outputs(pending_blocks, budget)?;
    let mut workspaces = preallocate_ht_workspaces(pending_blocks, budget)?;
    decoded_blocks
        .par_iter_mut()
        .zip(pending_blocks.par_iter())
        .zip(workspaces.par_iter_mut())
        .try_for_each(|((decoded, pending), workspace)| -> Result<()> {
            initialize_reserved_coefficients(
                &mut decoded.coefficients,
                block_coefficient_count(pending.width, pending.height)?,
            )?;
            decode_ht_code_block_scalar_with_workspace(
                HtCodeBlockDecodeJob {
                    data: &pending.combined.data,
                    cleanup_length: pending.combined.cleanup_length,
                    refinement_length: pending.combined.refinement_length,
                    width: pending.width,
                    height: pending.height,
                    output_stride: pending.width as usize,
                    missing_bit_planes: pending.missing_bit_planes,
                    number_of_coding_passes: pending.number_of_coding_passes,
                    num_bitplanes,
                    roi_shift,
                    stripe_causal,
                    strict,
                    dequantization_step,
                },
                &mut decoded.coefficients,
                workspace,
            )
        })?;
    Ok(decoded_blocks)
}

fn preallocate_ht_outputs(
    pending_blocks: &[PendingHtBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<DecodedHtBlock>> {
    let total_coefficients = pending_coefficient_count(
        pending_blocks
            .iter()
            .map(|pending| (pending.width, pending.height)),
    )?;
    budget.include_elements::<f32>(total_coefficients)?;

    let mut decoded_blocks = Vec::new();
    budget.reserve_new(&mut decoded_blocks, pending_blocks.len())?;
    for pending in pending_blocks {
        let coefficient_count = block_coefficient_count(pending.width, pending.height)?;
        let mut coefficients = Vec::new();
        try_reserve_decode_elements(&mut coefficients, coefficient_count)?;
        budget.include_capacity_overage::<f32>(coefficient_count, coefficients.capacity())?;
        decoded_blocks.push(DecodedHtBlock {
            output_x: pending.output_x,
            output_y: pending.output_y,
            width: pending.width,
            height: pending.height,
            coefficients,
        });
    }
    Ok(decoded_blocks)
}

fn preallocate_ht_workspaces(
    pending_blocks: &[PendingHtBlock],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<HtCodeBlockDecodeWorkspace>> {
    let planned_bytes =
        pending_blocks
            .iter()
            .try_fold(0usize, |bytes, pending| -> Result<usize> {
                bytes
                    .checked_add(ht_decode_workspace_bytes(pending.width, pending.height)?)
                    .ok_or_else(|| ValidationError::ImageTooLarge.into())
            })?;
    budget.include_bytes(planned_bytes)?;
    let mut workspaces = Vec::new();
    budget.reserve_new(&mut workspaces, pending_blocks.len())?;
    for pending in pending_blocks {
        let planned = ht_decode_workspace_bytes(pending.width, pending.height)?;
        let mut workspace = HtCodeBlockDecodeWorkspace::default();
        workspace.reserve(pending.width, pending.height)?;
        let actual = workspace.allocated_bytes()?;
        if actual > planned {
            budget.include_bytes(actual - planned)?;
        }
        workspaces.push(workspace);
    }
    Ok(workspaces)
}

fn initialize_reserved_coefficients(coefficients: &mut Vec<f32>, len: usize) -> Result<()> {
    if coefficients.capacity() < len {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    coefficients.resize(len, 0.0);
    Ok(())
}

fn pending_coefficient_count(mut dimensions: impl Iterator<Item = (u32, u32)>) -> Result<usize> {
    dimensions.try_fold(0_usize, |total, (width, height)| {
        total
            .checked_add(block_coefficient_count(width, height)?)
            .ok_or(ValidationError::ImageTooLarge.into())
    })
}

fn block_coefficient_count(width: u32, height: u32) -> Result<usize> {
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(ValidationError::ImageTooLarge.into())
}

pub(crate) fn copy_decoded_classic_blocks_to_sub_band(
    decoded_blocks: &[DecodedClassicBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)
}

pub(crate) fn copy_decoded_ht_blocks_to_sub_band(
    decoded_blocks: &[DecodedHtBlock],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    copy_decoded_blocks_to_sub_band(decoded_blocks, sub_band, storage)
}

fn copy_decoded_blocks_to_sub_band<B: DecodedSubBandBlock>(
    decoded_blocks: &[B],
    sub_band: &SubBand,
    storage: &mut DecompositionStorage<'_>,
) -> Result<()> {
    let sub_band_width = sub_band.rect.width() as usize;
    let base_store = &mut storage.coefficients[sub_band.coefficients.clone()];
    for block in decoded_blocks {
        let output_x = block.output_x();
        let output_y = block.output_y();
        let width = block.width();
        let height = block.height();
        if output_x
            .checked_add(width)
            .is_none_or(|x1| x1 > sub_band.rect.width())
            || output_y
                .checked_add(height)
                .is_none_or(|y1| y1 > sub_band.rect.height())
        {
            bail!(DecodingError::CodeBlockDecodeFailure);
        }
        let block_width = width as usize;
        for row in 0..height as usize {
            let dst_start = (output_y as usize + row)
                .checked_mul(sub_band_width)
                .and_then(|offset| offset.checked_add(output_x as usize))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let dst_end = dst_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_start = row
                .checked_mul(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let src_end = src_start
                .checked_add(block_width)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            base_store[dst_start..dst_end]
                .copy_from_slice(&block.coefficients()[src_start..src_end]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{initialize_reserved_coefficients, pending_coefficient_count};
    use crate::error::{DecodeError, ValidationError};
    use crate::try_reserve_decode_elements;
    use alloc::vec::Vec;

    #[test]
    fn coefficient_total_rejects_overflow_before_output_allocation() {
        let error =
            pending_coefficient_count([(u32::MAX, u32::MAX), (u32::MAX, u32::MAX)].into_iter())
                .expect_err("aggregate coefficient count must reject overflow");
        assert!(matches!(
            error,
            DecodeError::Validation(ValidationError::ImageTooLarge)
        ));
    }

    #[test]
    fn reserved_output_initialization_does_not_grow_allocation() {
        let mut coefficients = Vec::new();
        try_reserve_decode_elements(&mut coefficients, 64 * 64)
            .expect("coefficient reservation should succeed");
        let reserved_capacity = coefficients.capacity();

        initialize_reserved_coefficients(&mut coefficients, 64 * 64)
            .expect("reserved coefficients should initialize");

        assert_eq!(coefficients.len(), 64 * 64);
        assert_eq!(coefficients.capacity(), reserved_capacity);
    }

    #[test]
    fn unreserved_output_initialization_fails_without_allocating() {
        let mut coefficients = Vec::new();

        initialize_reserved_coefficients(&mut coefficients, 64 * 64)
            .expect_err("parallel initialization must not allocate");

        assert_eq!(coefficients.capacity(), 0);
        assert!(coefficients.is_empty());
    }
}
