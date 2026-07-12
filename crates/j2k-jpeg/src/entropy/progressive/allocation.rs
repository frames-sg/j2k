// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive coefficient, component-image, and phase workspace accounting.

use alloc::vec::Vec;

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    try_reserve_for_len_with_live_budget,
};
use crate::error::JpegError;

use super::model::{PreparedProgressiveComponentPlan, PreparedProgressivePlan};

#[derive(Debug)]
pub(super) struct ComponentImage {
    pub(super) plane: Vec<u8>,
    pub(super) stride: usize,
}

pub(crate) const COMPONENT_IMAGE_METADATA_BYTES: usize = core::mem::size_of::<ComponentImage>();

pub(super) fn allocate_coefficients(
    plan: &PreparedProgressivePlan,
    initial_live_bytes: usize,
) -> Result<Vec<Vec<[i32; 64]>>, JpegError> {
    let planned_bytes = validate_coefficient_workspace(&plan.components)?;
    checked_phase_capacity(initial_live_bytes, planned_bytes, plan.scratch_bytes)?;

    let mut live_bytes = initial_live_bytes;
    let mut coeffs = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut coeffs,
        plan.components.len(),
        &mut live_bytes,
        plan.scratch_bytes,
    )?;
    for component in &plan.components {
        let blocks = checked_allocation_len::<[i32; 64]>(
            component.block_cols as usize,
            component.block_rows as usize,
        )?;
        let mut component_coeffs = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut component_coeffs,
            blocks,
            &mut live_bytes,
            plan.scratch_bytes,
        )?;
        component_coeffs.resize(blocks, [0i32; 64]);
        coeffs.push(component_coeffs);
    }
    Ok(coeffs)
}

pub(super) fn validate_coefficient_workspace(
    components: &[PreparedProgressiveComponentPlan],
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<Vec<[i32; 64]>>(components.len())?;
    for component in components {
        let blocks = checked_allocation_len::<[i32; 64]>(
            component.block_cols as usize,
            component.block_rows as usize,
        )?;
        total =
            checked_add_allocation_bytes(total, checked_allocation_bytes::<[i32; 64]>(blocks)?)?;
    }
    Ok(total)
}

pub(super) fn coefficient_capacity_bytes(
    outer_capacity: usize,
    coeffs: &[Vec<[i32; 64]>],
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<Vec<[i32; 64]>>(outer_capacity)?;
    for component_coeffs in coeffs {
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<[i32; 64]>(component_coeffs.capacity())?,
        )?;
    }
    Ok(total)
}

pub(super) fn checked_phase_capacity(
    initial: usize,
    additional: usize,
    cap: usize,
) -> Result<usize, JpegError> {
    let requested = initial
        .checked_add(additional)
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap,
        })?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(requested)
}

pub(super) fn allocate_component_images(
    plan: &PreparedProgressivePlan,
    initial_live_bytes: usize,
) -> Result<Vec<ComponentImage>, JpegError> {
    let planned_bytes = validate_component_image_workspace(&plan.components)?;
    checked_phase_capacity(initial_live_bytes, planned_bytes, plan.scratch_bytes)?;

    let mut live_bytes = initial_live_bytes;
    let mut images = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut images,
        plan.components.len(),
        &mut live_bytes,
        plan.scratch_bytes,
    )?;
    for component in &plan.components {
        let stride = checked_allocation_len::<u8>(component.block_cols as usize, 8)?;
        let rows = checked_allocation_len::<u8>(component.block_rows as usize, 8)?;
        let plane_len = checked_allocation_len::<u8>(stride, rows)?;
        let mut plane = Vec::new();
        try_reserve_for_len_with_live_budget(
            &mut plane,
            plane_len,
            &mut live_bytes,
            plan.scratch_bytes,
        )?;
        plane.resize(plane_len, 0u8);
        images.push(ComponentImage { plane, stride });
    }
    Ok(images)
}

fn validate_component_image_workspace(
    components: &[PreparedProgressiveComponentPlan],
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<ComponentImage>(components.len())?;
    for component in components {
        let stride = checked_allocation_len::<u8>(component.block_cols as usize, 8)?;
        let rows = checked_allocation_len::<u8>(component.block_rows as usize, 8)?;
        let plane_len = checked_allocation_len::<u8>(stride, rows)?;
        total = checked_add_allocation_bytes(total, checked_allocation_bytes::<u8>(plane_len)?)?;
    }
    Ok(total)
}

pub(super) fn component_image_capacity_bytes(
    outer_capacity: usize,
    images: &[ComponentImage],
) -> Result<usize, JpegError> {
    let mut total = checked_allocation_bytes::<ComponentImage>(outer_capacity)?;
    for image in images {
        total = checked_add_allocation_bytes(
            total,
            checked_allocation_bytes::<u8>(image.plane.capacity())?,
        )?;
    }
    Ok(total)
}
