// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact logical counts and IDWT high-water planning for one tile.

use super::super::{PrecinctSelection, PrecinctUse};
use super::{BuildWorkspace, DecompositionAllocationPlan};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::build::SubBandType;
use crate::j2c::idwt::idwt_buffer_size;
use crate::j2c::rect::IntRect;
use crate::j2c::roi::RoiPlan;
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
    selection: Option<&PrecinctSelection>,
    roi_plan: Option<&RoiPlan>,
) -> Result<(DecompositionAllocationPlan, usize)> {
    let mut plan = DecompositionAllocationPlan::new(
        retained_baseline_bytes,
        exact_integer_decode,
        include_roi_workspace,
        workspace,
    )?;
    let mut idwt_workspace_elements = 0_usize;
    let mut resolution_run = 0usize;
    let keep = |run: usize| -> Result<Option<&[PrecinctUse]>> {
        match selection {
            Some(selection) => Ok(Some(
                selection.run(run).ok_or(DecodingError::InvalidPrecinct)?,
            )),
            None => Ok(None),
        }
    };

    for component_tile in tile.component_tiles() {
        let coefficient_count = checked_rect_elements(component_tile.rect)?;
        plan.add_component_coefficients(coefficient_count)?;
        DecompositionAllocationPlan::checked_add_count(&mut plan.tile_decompositions, 1)?;

        let mut resolution_tiles = component_tile.resolution_tiles();
        let ll_resolution_tile = resolution_tiles
            .next()
            .ok_or(DecodingError::InvalidPrecinct)?;
        include_sub_band_allocations(
            &ll_resolution_tile,
            SubBandType::LowLow,
            tile,
            keep(resolution_run)?,
            roi_plan,
            &mut plan,
        )?;
        resolution_run += 1;

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
            let resolution_keep = keep(resolution_run)?;
            resolution_run += 1;
            for sub_band_type in [
                SubBandType::HighLow,
                SubBandType::LowHigh,
                SubBandType::HighHigh,
            ] {
                include_sub_band_allocations(
                    &resolution_tile,
                    sub_band_type,
                    tile,
                    resolution_keep,
                    roi_plan,
                    &mut plan,
                )?;
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

/// Count one sub-band's precincts, code-blocks, layers and tag-tree nodes.
/// With `keep`, precincts it excludes are counted as `build` makes them: no
/// code-blocks and root-only tag trees. With `roi_plan`, only the code-blocks
/// it requires get layers, as `build` makes them.
fn include_sub_band_allocations(
    resolution_tile: &ResolutionTile<'_>,
    sub_band_type: SubBandType,
    tile: &Tile<'_>,
    keep: Option<&[PrecinctUse]>,
    roi_plan: Option<&RoiPlan>,
    plan: &mut DecompositionAllocationPlan,
) -> Result<()> {
    // Sub-bands are counted in build order, so this is the index `build`
    // gives the sub-band.
    let sub_band_idx = plan.sub_bands;
    DecompositionAllocationPlan::checked_add_count(&mut plan.sub_bands, 1)?;

    let sub_band_rect = resolution_tile.sub_band_rect(sub_band_type);
    let precinct_count = usize::try_from(resolution_tile.num_precincts())
        .map_err(|_| ValidationError::ImageTooLarge)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.precincts, precinct_count)?;

    // Without a selection, count every code-block up front to reject a
    // metadata bomb before walking precincts. With one, kept precincts are
    // counted as they are walked.
    let code_block_count = if keep.is_none() {
        global_code_block_count(resolution_tile, sub_band_rect)?
    } else {
        0
    };
    include_code_blocks(plan, code_block_count, code_block_count, tile)?;

    // Every tag tree has a root even when its code-block grid is empty. Count
    // both roots up front and reject a metadata bomb before walking precincts.
    let root_count = precinct_count
        .checked_mul(2)
        .ok_or(ValidationError::ImageTooLarge)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.tag_tree_nodes, root_count)?;
    plan.validate_minimum_live_workspace(0)?;

    let mut observed_precincts = 0usize;
    let mut observed_code_blocks = 0usize;
    for (precinct_idx, precinct_data) in resolution_tile
        .precincts()
        .ok_or(DecodingError::InvalidPrecinct)?
        .enumerate()
    {
        observed_precincts = observed_precincts
            .checked_add(1)
            .ok_or(ValidationError::ImageTooLarge)?;
        if keep.is_some_and(|keep| {
            keep.get(precinct_idx)
                .is_some_and(|precinct_use| !precinct_use.is_built())
        }) {
            continue;
        }
        let grid = code_block_grid(resolution_tile, sub_band_rect, precinct_data.rect)?;
        let block_count = (grid.columns as usize)
            .checked_mul(grid.rows as usize)
            .ok_or(ValidationError::ImageTooLarge)?;
        observed_code_blocks = observed_code_blocks
            .checked_add(block_count)
            .ok_or(ValidationError::ImageTooLarge)?;
        if keep.is_some() {
            let layered_blocks = roi_plan.map_or(block_count, |roi_plan| {
                required_code_block_count(
                    &grid,
                    resolution_tile,
                    sub_band_rect,
                    roi_plan.sub_band_window(sub_band_idx),
                )
            });
            include_code_blocks(plan, block_count, layered_blocks, tile)?;
        }

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

    if observed_precincts != precinct_count
        || (keep.is_none() && observed_code_blocks != code_block_count)
    {
        return Err(DecodingError::InvalidPrecinct.into());
    }
    Ok(())
}

/// Count `code_block_count` code-blocks, of which `layered_blocks` get one
/// layer record per quality layer.
fn include_code_blocks(
    plan: &mut DecompositionAllocationPlan,
    code_block_count: usize,
    layered_blocks: usize,
    tile: &Tile<'_>,
) -> Result<()> {
    DecompositionAllocationPlan::checked_add_count(&mut plan.code_blocks, code_block_count)?;
    let layer_count = layered_blocks
        .checked_mul(usize::from(tile.num_layers))
        .ok_or(ValidationError::ImageTooLarge)?;
    DecompositionAllocationPlan::checked_add_count(&mut plan.layers, layer_count)
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
    let block_end_x = u32::min(precinct_rect.x1, sub_band_rect.x1).div_ceil(cb_width);
    let block_end_y = u32::min(precinct_rect.y1, sub_band_rect.y1).div_ceil(cb_height);
    // Keep counts in block coordinates: the last grid cell can end at 2^32,
    // beyond every representable image/subband endpoint. Only its rectangle
    // is clipped; the last partial block must still contribute to the count.
    let cb_x1 = block_end_x.saturating_mul(cb_width);
    let cb_y1 = block_end_y.saturating_mul(cb_height);
    if cb_x0 > cb_x1 || cb_y0 > cb_y1 {
        return Err(DecodingError::InvalidPrecinct.into());
    }

    let area = IntRect::from_ltrb(cb_x0, cb_y0, cb_x1, cb_y1);
    let columns = if sub_band_rect.width() == 0 {
        0
    } else {
        block_end_x - cb_x0 / cb_width
    };
    let rows = if sub_band_rect.height() == 0 {
        0
    } else {
        block_end_y - cb_y0 / cb_height
    };

    Ok(CodeBlockGrid {
        area,
        columns,
        rows,
    })
}

/// The code-blocks of `grid` the ROI plan requires: those whose rectangles,
/// clipped to the sub-band, meet the sub-band's `window`. They form a run of
/// columns by a run of rows, so they are counted without walking the grid,
/// and an oversized precinct is still rejected from its block count alone.
fn required_code_block_count(
    grid: &CodeBlockGrid,
    resolution_tile: &ResolutionTile<'_>,
    sub_band_rect: IntRect,
    window: Option<IntRect>,
) -> usize {
    let Some(window) = window.filter(|window| window.intersects(sub_band_rect)) else {
        return 0;
    };
    // The cells that start at `start` and step by `size`, `count` of them, which
    // meet the span from `low` to `high` the way `IntRect::intersects` does.
    let cells = |start: u32, size: u32, count: u32, low: u32, high: u32| {
        let first = (low.saturating_sub(start) / size).min(count);
        let end = high.saturating_sub(start).div_ceil(size).min(count);
        end.saturating_sub(first) as usize
    };
    let columns = cells(
        grid.area.x0,
        resolution_tile.code_block_width(),
        grid.columns,
        window.x0,
        window.x1,
    );
    let rows = cells(
        grid.area.y0,
        resolution_tile.code_block_height(),
        grid.rows,
        window.y0,
        window.y1,
    );
    columns * rows
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
                let band_scratch = crate::j2c::idwt::planned_band_scratch_elements(
                    highest_rect.width() as usize,
                    highest_rect.height() as usize,
                )?;
                output
                    .checked_add(scratch)
                    .and_then(|elements| elements.checked_add(band_scratch))
                    .ok_or(ValidationError::ImageTooLarge)?
            }
            None => checked_rect_elements(ll_rect)?,
        };
        Ok(Some(workspace))
    }
}
