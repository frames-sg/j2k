// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible classic Tier-1 decode workspace initialization.

use super::{
    BitPlaneDecodeContext, Coefficient, CoefficientState, NeighborSignificances,
    COEFFICIENTS_PADDING,
};
use crate::error::{Result, ValidationError};
use crate::{checked_decode_usize_product2, try_resize_decode_elements, DEFAULT_MAX_DECODE_BYTES};
use core::mem::size_of;

#[derive(Clone, Copy)]
struct ClassicWorkspaceLayout {
    padded_width: u32,
    coefficient_count: usize,
    scan_units: usize,
}

fn workspace_layout(width: u32, height: u32) -> Result<ClassicWorkspaceLayout> {
    let padding = COEFFICIENTS_PADDING
        .checked_mul(2)
        .ok_or(ValidationError::ImageTooLarge)?;
    let padded_width = width
        .checked_add(padding)
        .ok_or(ValidationError::ImageTooLarge)?;
    let padded_height = height
        .checked_add(padding)
        .ok_or(ValidationError::ImageTooLarge)?;
    let coefficient_count =
        checked_decode_usize_product2(padded_width as usize, padded_height as usize)?;
    let scan_units = checked_decode_usize_product2(width as usize, height.div_ceil(4) as usize)?;
    Ok(ClassicWorkspaceLayout {
        padded_width,
        coefficient_count,
        scan_units,
    })
}

pub(crate) fn classic_decode_workspace_bytes(width: u32, height: u32) -> Result<usize> {
    let layout = workspace_layout(width, height)?;
    let mut bytes = 0usize;
    include_elements::<Coefficient>(&mut bytes, layout.coefficient_count)?;
    include_elements::<NeighborSignificances>(&mut bytes, layout.coefficient_count)?;
    include_elements::<CoefficientState>(&mut bytes, layout.coefficient_count)?;
    include_elements::<u8>(&mut bytes, layout.scan_units)?;
    include_elements::<u8>(&mut bytes, layout.scan_units)?;
    Ok(bytes)
}

fn include_elements<T>(bytes: &mut usize, count: usize) -> Result<()> {
    let additional = count
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    if *bytes > DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

pub(super) fn reset_decode_buffers(
    context: &mut BitPlaneDecodeContext,
    width: u32,
    height: u32,
) -> Result<u32> {
    let layout = workspace_layout(width, height)?;

    context.coefficients.clear();
    try_resize_decode_elements(
        &mut context.coefficients,
        layout.coefficient_count,
        Coefficient::default(),
    )?;
    context.neighbor_significances.clear();
    try_resize_decode_elements(
        &mut context.neighbor_significances,
        layout.coefficient_count,
        NeighborSignificances::default(),
    )?;
    context.coefficient_states.clear();
    try_resize_decode_elements(
        &mut context.coefficient_states,
        layout.coefficient_count,
        CoefficientState::default(),
    )?;

    context.significant_scan_masks.clear();
    try_resize_decode_elements(&mut context.significant_scan_masks, layout.scan_units, 0)?;
    context.zero_coding_scan_masks.clear();
    try_resize_decode_elements(&mut context.zero_coding_scan_masks, layout.scan_units, 0)?;
    Ok(layout.padded_width)
}
