// SPDX-License-Identifier: MIT OR Apache-2.0

//! DCT extraction layout, fallible storage allocation, and lifecycle budgets.

use alloc::vec::Vec;

use super::super::PreparedDecodePlan;
use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    try_reserve_for_len_with_live_budget,
};
use crate::error::JpegError;
use j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES;

pub(super) struct DctDecodeStorage {
    pub(super) block_cols_by_component: Vec<u32>,
    block_rows_by_component: Vec<u32>,
    pub(super) quantized_blocks: Vec<Vec<[i16; 64]>>,
    pub(super) dequantized_blocks: Vec<Vec<[i16; 64]>>,
    pub(super) prev_dc: Vec<i32>,
}

pub(super) fn allocate_dct_decode_storage(
    plan: &PreparedDecodePlan,
    mcus_per_row: u32,
    mcu_rows: u32,
    retain_quantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<DctDecodeStorage, JpegError> {
    validate_dct_workspace(
        plan,
        mcus_per_row,
        mcu_rows,
        retain_quantized_blocks,
        lifecycle,
    )?;

    let component_count = plan.sampling.len();
    let mut live_bytes = 0;
    let mut block_cols_by_component = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut block_cols_by_component,
        component_count,
        &mut live_bytes,
        lifecycle.workspace_cap,
    )?;
    block_cols_by_component.resize(component_count, 0_u32);
    let mut block_rows_by_component = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut block_rows_by_component,
        component_count,
        &mut live_bytes,
        lifecycle.workspace_cap,
    )?;
    block_rows_by_component.resize(component_count, 0_u32);
    for component in &plan.components {
        block_cols_by_component[component.output_index] = mcus_per_row * u32::from(component.h);
        block_rows_by_component[component.output_index] = mcu_rows * u32::from(component.v);
    }

    let mut quantized_blocks = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut quantized_blocks,
        component_count,
        &mut live_bytes,
        lifecycle.workspace_cap,
    )?;
    let mut dequantized_blocks = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut dequantized_blocks,
        component_count,
        &mut live_bytes,
        lifecycle.workspace_cap,
    )?;
    for (&cols, &rows) in block_cols_by_component.iter().zip(&block_rows_by_component) {
        let block_count = checked_allocation_len::<[i16; 64]>(cols as usize, rows as usize)?;
        let quantized = if retain_quantized_blocks {
            let mut blocks = Vec::new();
            try_reserve_for_len_with_live_budget(
                &mut blocks,
                block_count,
                &mut live_bytes,
                lifecycle.workspace_cap,
            )?;
            blocks.resize(block_count, [0_i16; 64]);
            blocks
        } else {
            Vec::new()
        };
        quantized_blocks.push(quantized);
        let mut dequantized = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut dequantized,
            block_count,
            &mut live_bytes,
            lifecycle.workspace_cap,
        )?;
        dequantized.resize(block_count, [0_i16; 64]);
        dequantized_blocks.push(dequantized);
    }

    let mut prev_dc = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut prev_dc,
        component_count,
        &mut live_bytes,
        lifecycle.workspace_cap,
    )?;
    prev_dc.resize(component_count, 0_i32);
    Ok(DctDecodeStorage {
        block_cols_by_component,
        block_rows_by_component,
        quantized_blocks,
        dequantized_blocks,
        prev_dc,
    })
}

impl DctDecodeStorage {
    pub(super) fn finish(
        self,
        lifecycle: SequentialDctLifecycleMetadata,
    ) -> Result<DecodedDctBlocks, JpegError> {
        let Self {
            block_cols_by_component,
            block_rows_by_component,
            quantized_blocks,
            dequantized_blocks,
            prev_dc,
        } = self;
        drop(prev_dc);
        drop(block_cols_by_component);
        drop(block_rows_by_component);
        let decoded = DecodedDctBlocks {
            quantized: quantized_blocks,
            dequantized: dequantized_blocks,
        };
        validate_actual_dct_lifecycle(&decoded, lifecycle)?;
        Ok(decoded)
    }
}

fn validate_dct_workspace(
    plan: &PreparedDecodePlan,
    mcus_per_row: u32,
    mcu_rows: u32,
    retain_quantized_blocks: bool,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<(), JpegError> {
    let component_count = plan.sampling.len();
    let decoded_outer_bytes = checked_add_allocation_bytes(
        checked_allocation_bytes::<Vec<[i16; 64]>>(component_count)?,
        checked_allocation_bytes::<Vec<[i16; 64]>>(component_count)?,
    )?;
    let mut entropy_state_bytes = checked_allocation_bytes::<u32>(component_count)?;
    entropy_state_bytes = checked_add_allocation_bytes(
        entropy_state_bytes,
        checked_allocation_bytes::<u32>(component_count)?,
    )?;
    entropy_state_bytes = checked_add_allocation_bytes(entropy_state_bytes, decoded_outer_bytes)?;
    entropy_state_bytes = checked_add_allocation_bytes(
        entropy_state_bytes,
        checked_allocation_bytes::<i32>(component_count)?,
    )?;

    let mut plane_bytes = 0usize;
    for component in &plan.components {
        let block_cols = (mcus_per_row as usize)
            .checked_mul(usize::from(component.h))
            .ok_or_else(cap_overflow)?;
        let block_rows = (mcu_rows as usize)
            .checked_mul(usize::from(component.v))
            .ok_or_else(cap_overflow)?;
        let block_count = checked_allocation_len::<[i16; 64]>(block_cols, block_rows)?;
        plane_bytes = checked_add_allocation_bytes(
            plane_bytes,
            checked_allocation_bytes::<[i16; 64]>(block_count)?,
        )?;
    }
    let retained_plane_bytes = if retain_quantized_blocks {
        checked_add_allocation_bytes(plane_bytes, plane_bytes)?
    } else {
        plane_bytes
    };
    ensure_phase_capacity(
        entropy_state_bytes,
        retained_plane_bytes,
        lifecycle.workspace_cap,
    )?;

    let assembly_metadata =
        checked_add_allocation_bytes(decoded_outer_bytes, lifecycle.component_output_bytes)?;
    ensure_phase_capacity(
        retained_plane_bytes,
        assembly_metadata,
        lifecycle.workspace_cap,
    )?;
    let returned_metadata = checked_add_allocation_bytes(
        lifecycle.component_output_bytes,
        lifecycle.restart_index_bytes,
    )?;
    ensure_phase_capacity(
        retained_plane_bytes,
        returned_metadata,
        lifecycle.workspace_cap,
    )
}

fn validate_actual_dct_lifecycle(
    decoded: &DecodedDctBlocks,
    lifecycle: SequentialDctLifecycleMetadata,
) -> Result<(), JpegError> {
    ensure_phase_capacity(
        decoded.capacity_bytes()?,
        lifecycle.component_output_bytes,
        lifecycle.workspace_cap,
    )?;
    let returned_metadata = lifecycle
        .component_output_bytes
        .checked_add(lifecycle.restart_index_bytes)
        .ok_or_else(cap_overflow)?;
    ensure_phase_capacity(
        decoded.plane_capacity_bytes()?,
        returned_metadata,
        lifecycle.workspace_cap,
    )
}

fn ensure_phase_capacity(initial: usize, additional: usize, cap: usize) -> Result<(), JpegError> {
    let requested = initial
        .checked_add(additional)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct SequentialDctLifecycleMetadata {
    pub(crate) component_output_bytes: usize,
    pub(crate) restart_index_bytes: usize,
    pub(crate) workspace_cap: usize,
}

impl SequentialDctLifecycleMetadata {
    pub(crate) const fn new(
        component_output_bytes: usize,
        restart_index_bytes: usize,
        workspace_cap: usize,
    ) -> Self {
        Self {
            component_output_bytes,
            restart_index_bytes,
            workspace_cap,
        }
    }
}

fn cap_overflow() -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DecodedDctBlocks {
    pub(crate) quantized: Vec<Vec<[i16; 64]>>,
    pub(crate) dequantized: Vec<Vec<[i16; 64]>>,
}

impl DecodedDctBlocks {
    pub(crate) fn capacity_bytes(&self) -> Result<usize, JpegError> {
        let outer = checked_add_allocation_bytes(
            checked_allocation_bytes::<Vec<[i16; 64]>>(self.quantized.capacity())?,
            checked_allocation_bytes::<Vec<[i16; 64]>>(self.dequantized.capacity())?,
        )?;
        checked_add_allocation_bytes(outer, self.plane_capacity_bytes()?)
    }

    pub(crate) fn plane_capacity_bytes(&self) -> Result<usize, JpegError> {
        let mut total = 0;
        for plane in self.quantized.iter().chain(&self.dequantized) {
            total = checked_add_allocation_bytes(
                total,
                checked_allocation_bytes::<[i16; 64]>(plane.capacity())?,
            )?;
        }
        Ok(total)
    }
}
