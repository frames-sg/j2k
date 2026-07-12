// SPDX-License-Identifier: MIT OR Apache-2.0

//! Aggregate planning and reusable-capacity policy for decomposition storage.

use super::{CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand};
use crate::error::{Result, ValidationError};
use crate::j2c::decode::{DecompositionStorage, TileDecompositions};
use crate::j2c::rect::IntRect;
use crate::j2c::tag_tree::TagNode;
use crate::j2c::tile::Tile;
use crate::DEFAULT_MAX_DECODE_BYTES;
use core::mem::size_of;

mod plan;
mod reuse;

pub(super) use plan::{code_block_grid, tag_tree_node_count};
pub(super) use reuse::push_preallocated;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BuildWorkspace {
    DecodePixels { skipped_resolution_levels: u8 },
    CoefficientsOnly,
}

impl BuildWorkspace {
    pub(super) fn active_decomposition_count(self, total_count: usize) -> Option<usize> {
        match self {
            Self::DecodePixels {
                skipped_resolution_levels,
            } => Some(total_count.saturating_sub(usize::from(skipped_resolution_levels))),
            Self::CoefficientsOnly => None,
        }
    }
}

pub(super) struct DecompositionAllocationPlan {
    retained_baseline_bytes: usize,
    pub(super) total_bytes: usize,
    pub(super) coefficients: usize,
    pub(super) tile_decompositions: usize,
    pub(super) decompositions: usize,
    pub(super) sub_bands: usize,
    pub(super) precincts: usize,
    pub(super) code_blocks: usize,
    pub(super) layers: usize,
    pub(super) tag_tree_nodes: usize,
    exact_integer_decode: bool,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
}

impl DecompositionAllocationPlan {
    fn new(
        retained_baseline_bytes: usize,
        exact_integer_decode: bool,
        include_roi_workspace: bool,
        workspace: BuildWorkspace,
    ) -> Result<Self> {
        if retained_baseline_bytes > DEFAULT_MAX_DECODE_BYTES {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(Self {
            retained_baseline_bytes,
            total_bytes: retained_baseline_bytes,
            coefficients: 0,
            tile_decompositions: 0,
            decompositions: 0,
            sub_bands: 0,
            precincts: 0,
            code_blocks: 0,
            layers: 0,
            tag_tree_nodes: 0,
            exact_integer_decode,
            include_roi_workspace,
            workspace,
        })
    }

    fn add_coefficients(&mut self, count: usize) -> Result<()> {
        self.coefficients = self
            .coefficients
            .checked_add(count)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    }

    fn checked_add_count(target_count: &mut usize, added_count: usize) -> Result<()> {
        *target_count = target_count
            .checked_add(added_count)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    }

    /// Reject a logically impossible request before walking every precinct.
    fn validate_minimum_live_workspace(&self, idwt_workspace_elements: usize) -> Result<()> {
        let mut total_bytes = self.retained_baseline_bytes;
        checked_include_elements::<TileDecompositions>(&mut total_bytes, self.tile_decompositions)?;
        checked_include_elements::<Decomposition>(&mut total_bytes, self.decompositions)?;
        checked_include_elements::<SubBand>(&mut total_bytes, self.sub_bands)?;
        checked_include_elements::<Precinct>(&mut total_bytes, self.precincts)?;
        checked_include_elements::<CodeBlock>(&mut total_bytes, self.code_blocks)?;
        checked_include_elements::<Layer>(&mut total_bytes, self.layers)?;
        checked_include_elements::<TagNode>(&mut total_bytes, self.tag_tree_nodes)?;
        checked_include_elements::<f32>(&mut total_bytes, self.coefficients)?;
        if self.exact_integer_decode {
            checked_include_elements::<i64>(&mut total_bytes, self.coefficients)?;
        }

        if self.include_roi_workspace {
            checked_include_elements::<Option<IntRect>>(&mut total_bytes, self.sub_bands)?;
            checked_include_elements::<Option<IntRect>>(&mut total_bytes, self.decompositions)?;
            checked_include_elements::<Option<IntRect>>(
                &mut total_bytes,
                self.tile_decompositions,
            )?;
        }

        if matches!(self.workspace, BuildWorkspace::DecodePixels { .. }) {
            if self.exact_integer_decode {
                checked_include_elements::<i64>(&mut total_bytes, idwt_workspace_elements)?;
            } else {
                checked_include_elements::<f32>(&mut total_bytes, idwt_workspace_elements)?;
            }
        }
        Ok(())
    }

    fn account_live_workspace(
        &mut self,
        storage: &DecompositionStorage<'_>,
        idwt_workspace_elements: usize,
    ) -> Result<()> {
        let mut total_bytes = self.retained_baseline_bytes;
        checked_include_reusable_elements::<TileDecompositions>(
            &mut total_bytes,
            storage.tile_decompositions.capacity(),
            self.tile_decompositions,
        )?;
        checked_include_reusable_elements::<Decomposition>(
            &mut total_bytes,
            storage.decompositions.capacity(),
            self.decompositions,
        )?;
        checked_include_reusable_elements::<SubBand>(
            &mut total_bytes,
            storage.sub_bands.capacity(),
            self.sub_bands,
        )?;
        checked_include_reusable_elements::<Precinct>(
            &mut total_bytes,
            storage.precincts.capacity(),
            self.precincts,
        )?;
        checked_include_reusable_elements::<CodeBlock>(
            &mut total_bytes,
            storage.code_blocks.capacity(),
            self.code_blocks,
        )?;
        checked_include_reusable_elements::<Layer>(
            &mut total_bytes,
            storage.layers.capacity(),
            self.layers,
        )?;
        checked_include_reusable_elements::<TagNode>(
            &mut total_bytes,
            storage.tag_tree_nodes.capacity(),
            self.tag_tree_nodes,
        )?;
        checked_include_reusable_elements::<f32>(
            &mut total_bytes,
            storage.coefficients.capacity(),
            self.coefficients,
        )?;
        checked_include_reusable_elements::<i64>(
            &mut total_bytes,
            storage.coefficients_i64.capacity(),
            usize::from(self.exact_integer_decode) * self.coefficients,
        )?;

        if self.include_roi_workspace {
            checked_include_elements::<Option<IntRect>>(&mut total_bytes, self.sub_bands)?;
            checked_include_elements::<Option<IntRect>>(&mut total_bytes, self.decompositions)?;
            checked_include_elements::<Option<IntRect>>(
                &mut total_bytes,
                self.tile_decompositions,
            )?;
        }

        if matches!(self.workspace, BuildWorkspace::DecodePixels { .. }) {
            if self.exact_integer_decode {
                checked_include_elements::<i64>(&mut total_bytes, idwt_workspace_elements)?;
            } else {
                checked_include_elements::<f32>(&mut total_bytes, idwt_workspace_elements)?;
            }
        }

        // The packet parser owns segment growth under the remaining budget.
        // Validate retained capacity once, but keep it outside the structural
        // total so downstream phases add this owner exactly once.
        let mut total_with_segments = total_bytes;
        checked_include_elements::<Segment<'_>>(
            &mut total_with_segments,
            storage.segments.capacity(),
        )?;
        self.total_bytes = total_bytes;
        Ok(())
    }
}

pub(super) fn prepare_decomposition_storage(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    retained_baseline_bytes: usize,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
) -> Result<DecompositionAllocationPlan> {
    let (mut plan, idwt_workspace_elements) = plan::build_allocation_plan(
        tile,
        retained_baseline_bytes,
        storage.exact_integer_decode,
        include_roi_workspace,
        workspace,
    )?;

    // Capacity retention must never make a smaller valid tile fail in a later
    // phase. Keep exact/current-size buffers, discard stale excess, and start
    // packet metadata empty because its target is not known until parsing.
    reuse::discard_stale_capacity(storage, &plan);
    plan.account_live_workspace(storage, idwt_workspace_elements)?;
    reuse::reserve_decomposition_storage(&plan, storage, retained_baseline_bytes)?;
    plan.account_live_workspace(storage, idwt_workspace_elements)?;
    Ok(plan)
}

pub(crate) fn release_unused_roi_workspace(
    storage: &mut DecompositionStorage<'_>,
    component_count: usize,
) -> Result<()> {
    let roi_bytes = roi_workspace_bytes(
        storage.sub_bands.len(),
        storage.decompositions.len(),
        component_count,
    )?;
    release_roi_workspace_bytes(&mut storage.structural_workspace_bytes, roi_bytes)
}

fn roi_workspace_bytes(
    sub_band_count: usize,
    decomposition_count: usize,
    component_count: usize,
) -> Result<usize> {
    let mut roi_bytes = 0usize;
    checked_include_elements::<Option<IntRect>>(&mut roi_bytes, sub_band_count)?;
    checked_include_elements::<Option<IntRect>>(&mut roi_bytes, decomposition_count)?;
    checked_include_elements::<Option<IntRect>>(&mut roi_bytes, component_count)?;
    Ok(roi_bytes)
}

fn release_roi_workspace_bytes(
    structural_workspace_bytes: &mut usize,
    roi_bytes: usize,
) -> Result<()> {
    *structural_workspace_bytes = structural_workspace_bytes
        .checked_sub(roi_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(())
}

fn checked_include_elements<T>(total_bytes: &mut usize, count: usize) -> Result<()> {
    let bytes = size_of::<T>()
        .checked_mul(count)
        .ok_or(ValidationError::ImageTooLarge)?;
    *total_bytes = total_bytes
        .checked_add(bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    if *total_bytes > DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

fn checked_include_reusable_elements<T>(
    total_bytes: &mut usize,
    retained_capacity: usize,
    target_count: usize,
) -> Result<()> {
    checked_include_elements::<T>(total_bytes, retained_capacity.max(target_count))
}

#[cfg(test)]
mod tests;
