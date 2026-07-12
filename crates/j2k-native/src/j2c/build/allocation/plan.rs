// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact logical counts and IDWT high-water planning for one tile.

use super::{BuildWorkspace, DecompositionAllocationPlan};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::build::SubBandType;
use crate::j2c::idwt::idwt_buffer_size;
use crate::j2c::rect::IntRect;
use crate::j2c::tile::{ResolutionTile, Tile};

#[derive(Clone, Copy)]
pub(in crate::j2c::build) struct CodeBlockGrid {
    pub(in crate::j2c::build) area: IntRect,
    pub(in crate::j2c::build) columns: u32,
    pub(in crate::j2c::build) rows: u32,
}

pub(super) fn build_allocation_plan(
    tile: &Tile<'_>,
    retained_baseline_bytes: usize,
    exact_integer_decode: bool,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
) -> Result<(DecompositionAllocationPlan, usize)> {
    let mut plan = DecompositionAllocationPlan::new(
        retained_baseline_bytes,
        exact_integer_decode,
        include_roi_workspace,
        workspace,
    )?;
    let mut idwt_workspace_elements = 0_usize;

    for component_tile in tile.component_tiles() {
        let coefficient_count = checked_rect_elements(component_tile.rect)?;
        plan.add_coefficients(coefficient_count)?;
        DecompositionAllocationPlan::checked_add_count(&mut plan.tile_decompositions, 1)?;

        let mut resolution_tiles = component_tile.resolution_tiles();
        let ll_resolution_tile = resolution_tiles
            .next()
            .ok_or(DecodingError::InvalidPrecinct)?;
        include_sub_band_allocations(&ll_resolution_tile, SubBandType::LowLow, tile, &mut plan)?;

        let total_decomposition_count = usize::from(
            component_tile
                .component_info
                .coding_style
                .parameters
                .num_decomposition_levels,
        );
        let mut idwt_tracker = IdwtWorkspaceTracker::new(
            workspace.active_decomposition_count(total_decomposition_count),
        );
        let mut observed_decomposition_count = 0usize;
        for resolution_tile in resolution_tiles {
            idwt_tracker.observe(resolution_tile.rect);
            observed_decomposition_count = observed_decomposition_count
                .checked_add(1)
                .ok_or(ValidationError::ImageTooLarge)?;
            DecompositionAllocationPlan::checked_add_count(&mut plan.decompositions, 1)?;
            for sub_band_type in [
                SubBandType::HighLow,
                SubBandType::LowHigh,
                SubBandType::HighHigh,
            ] {
                include_sub_band_allocations(&resolution_tile, sub_band_type, tile, &mut plan)?;
            }
        }
        if observed_decomposition_count != total_decomposition_count {
            return Err(DecodingError::InvalidPrecinct.into());
        }

        if let Some(component_workspace) =
            idwt_tracker.finish(ll_resolution_tile.sub_band_rect(SubBandType::LowLow))?
        {
            idwt_workspace_elements = idwt_workspace_elements.max(component_workspace);
        }
    }

    plan.validate_minimum_live_workspace(idwt_workspace_elements)?;
    Ok((plan, idwt_workspace_elements))
}

fn include_sub_band_allocations(
    resolution_tile: &ResolutionTile<'_>,
    sub_band_type: SubBandType,
    tile: &Tile<'_>,
    plan: &mut DecompositionAllocationPlan,
) -> Result<()> {
    DecompositionAllocationPlan::checked_add_count(&mut plan.sub_bands, 1)?;

    let sub_band_rect = resolution_tile.sub_band_rect(sub_band_type);
    let precinct_count = usize::try_from(resolution_tile.num_precincts())
        .map_err(|_| ValidationError::ImageTooLarge)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.precincts, precinct_count)?;

    let code_block_count = global_code_block_count(resolution_tile, sub_band_rect)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.code_blocks, code_block_count)?;
    let layer_count = code_block_count
        .checked_mul(usize::from(tile.num_layers))
        .ok_or(ValidationError::ImageTooLarge)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.layers, layer_count)?;

    // Every tag tree has a root even when its code-block grid is empty. Count
    // both roots up front and reject a metadata bomb before walking precincts.
    let root_count = precinct_count
        .checked_mul(2)
        .ok_or(ValidationError::ImageTooLarge)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.tag_tree_nodes, root_count)?;
    plan.validate_minimum_live_workspace(0)?;

    let mut observed_precincts = 0usize;
    let mut observed_code_blocks = 0usize;
    for precinct_data in resolution_tile
        .precincts()
        .ok_or(DecodingError::InvalidPrecinct)?
    {
        observed_precincts = observed_precincts
            .checked_add(1)
            .ok_or(ValidationError::ImageTooLarge)?;
        let grid = code_block_grid(resolution_tile, sub_band_rect, precinct_data.rect)?;
        let block_count = (grid.columns as usize)
            .checked_mul(grid.rows as usize)
            .ok_or(ValidationError::ImageTooLarge)?;
        observed_code_blocks = observed_code_blocks
            .checked_add(block_count)
            .ok_or(ValidationError::ImageTooLarge)?;

        let nodes_per_tree = tag_tree_node_count(grid.columns, grid.rows)?;
        let additional_nodes = nodes_per_tree
            .checked_sub(1)
            .and_then(|count| count.checked_mul(2))
            .ok_or(ValidationError::ImageTooLarge)?;
        DecompositionAllocationPlan::checked_add_count(&mut plan.tag_tree_nodes, additional_nodes)?;
        // Stop an expensive tree walk as soon as the exact node count alone
        // makes the request impossible.
        plan.validate_minimum_live_workspace(0)?;
    }

    if observed_precincts != precinct_count || observed_code_blocks != code_block_count {
        return Err(DecodingError::InvalidPrecinct.into());
    }
    Ok(())
}

fn global_code_block_count(
    resolution_tile: &ResolutionTile<'_>,
    sub_band_rect: IntRect,
) -> Result<usize> {
    if sub_band_rect.is_empty() {
        return Ok(0);
    }
    let code_block_width = resolution_tile.code_block_width();
    let code_block_height = resolution_tile.code_block_height();
    let columns = sub_band_rect
        .x1
        .div_ceil(code_block_width)
        .checked_sub(sub_band_rect.x0 / code_block_width)
        .ok_or(DecodingError::InvalidPrecinct)?;
    let rows = sub_band_rect
        .y1
        .div_ceil(code_block_height)
        .checked_sub(sub_band_rect.y0 / code_block_height)
        .ok_or(DecodingError::InvalidPrecinct)?;
    (columns as usize)
        .checked_mul(rows as usize)
        .ok_or(ValidationError::ImageTooLarge.into())
}

pub(in crate::j2c::build) fn tag_tree_node_count(width: u32, height: u32) -> Result<usize> {
    if width == 0 || height == 0 {
        return Ok(1);
    }

    let mut level_width = width;
    let mut level_height = height;
    let mut count = 0usize;
    loop {
        let level_count = (level_width as usize)
            .checked_mul(level_height as usize)
            .ok_or(ValidationError::ImageTooLarge)?;
        count = count
            .checked_add(level_count)
            .ok_or(ValidationError::ImageTooLarge)?;
        if level_width == 1 && level_height == 1 {
            return Ok(count);
        }
        level_width = level_width.div_ceil(2);
        level_height = level_height.div_ceil(2);
    }
}

#[expect(
    clippy::similar_names,
    reason = "paired axis and code-block names follow JPEG 2000 specification notation"
)]
pub(in crate::j2c::build) fn code_block_grid(
    resolution_tile: &ResolutionTile<'_>,
    sub_band_rect: IntRect,
    precinct_rect: IntRect,
) -> Result<CodeBlockGrid> {
    let cb_width = resolution_tile.code_block_width();
    let cb_height = resolution_tile.code_block_height();

    let cb_x0 = (u32::max(precinct_rect.x0, sub_band_rect.x0) / cb_width) * cb_width;
    let cb_y0 = (u32::max(precinct_rect.y0, sub_band_rect.y0) / cb_height) * cb_height;
    let cb_x1 = u32::min(precinct_rect.x1, sub_band_rect.x1)
        .div_ceil(cb_width)
        .checked_mul(cb_width)
        .ok_or(ValidationError::ImageTooLarge)?;
    let cb_y1 = u32::min(precinct_rect.y1, sub_band_rect.y1)
        .div_ceil(cb_height)
        .checked_mul(cb_height)
        .ok_or(ValidationError::ImageTooLarge)?;
    if cb_x0 > cb_x1 || cb_y0 > cb_y1 {
        return Err(DecodingError::InvalidPrecinct.into());
    }

    let area = IntRect::from_ltrb(cb_x0, cb_y0, cb_x1, cb_y1);
    let columns = if sub_band_rect.width() == 0 {
        0
    } else {
        area.width() / cb_width
    };
    let rows = if sub_band_rect.height() == 0 {
        0
    } else {
        area.height() / cb_height
    };

    Ok(CodeBlockGrid {
        area,
        columns,
        rows,
    })
}

fn checked_rect_elements(rect: IntRect) -> Result<usize> {
    (rect.width() as usize)
        .checked_mul(rect.height() as usize)
        .ok_or(ValidationError::ImageTooLarge.into())
}

#[derive(Clone, Copy)]
pub(super) struct IdwtWorkspaceTracker {
    active_decomposition_count: Option<usize>,
    observed_count: usize,
    second_highest_active_rect: Option<IntRect>,
    highest_active_rect: Option<IntRect>,
}

impl IdwtWorkspaceTracker {
    pub(super) fn new(active_decomposition_count: Option<usize>) -> Self {
        Self {
            active_decomposition_count,
            observed_count: 0,
            second_highest_active_rect: None,
            highest_active_rect: None,
        }
    }

    pub(super) fn observe(&mut self, rect: IntRect) {
        if self
            .active_decomposition_count
            .is_some_and(|count| self.observed_count < count)
        {
            self.second_highest_active_rect = self.highest_active_rect;
            self.highest_active_rect = Some(rect);
        }
        self.observed_count = self.observed_count.saturating_add(1);
    }

    pub(super) fn finish(self, ll_rect: IntRect) -> Result<Option<usize>> {
        if self.active_decomposition_count.is_none() {
            return Ok(None);
        }
        let workspace = match self.highest_active_rect {
            Some(highest_rect) => {
                let output = idwt_buffer_size(highest_rect)?.1;
                let scratch = self
                    .second_highest_active_rect
                    .map(|rect| idwt_buffer_size(rect).map(|(_, maximum)| maximum))
                    .transpose()?
                    .unwrap_or(0);
                output
                    .checked_add(scratch)
                    .ok_or(ValidationError::ImageTooLarge)?
            }
            None => checked_rect_elements(ll_rect)?,
        };
        Ok(Some(workspace))
    }
}
