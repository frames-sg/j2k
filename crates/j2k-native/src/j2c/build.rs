//! Building and setting up decompositions, sub-bands, precincts and code-blocks.

use super::decode::{DecompositionStorage, TileDecompositions};
use super::rect::IntRect;
use super::tag_tree::TagTree;
use super::tile::{ResolutionTile, Tile};
use crate::error::{DecodingError, Result, ValidationError};
use crate::try_resize_decode_elements;
use core::{iter, ops::Range};

mod allocation;
use allocation::{
    code_block_grid, prepare_decomposition_storage, push_preallocated, tag_tree_node_count,
    DecompositionAllocationPlan,
};
pub(crate) use allocation::{release_unused_roi_workspace, BuildWorkspace};

/// Build and allocate all necessary structures to process the code-blocks
/// for a specific tile. Also parses the segments for each code-block.
pub(crate) fn build(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    retained_baseline_bytes: usize,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
) -> Result<()> {
    build_decompositions(
        tile,
        storage,
        retained_baseline_bytes,
        include_roi_workspace,
        workspace,
    )
}

fn build_decompositions(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    retained_baseline_bytes: usize,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
) -> Result<()> {
    if !build_storage_is_empty(storage) {
        return Err(DecodingError::InvalidPrecinct.into());
    }

    let plan = prepare_decomposition_storage(
        tile,
        storage,
        retained_baseline_bytes,
        include_roi_workspace,
        workspace,
    )?;
    try_resize_decode_elements(&mut storage.coefficients, plan.coefficients, 0.0)?;
    if storage.exact_integer_decode {
        try_resize_decode_elements(&mut storage.coefficients_i64, plan.coefficients, 0)?;
    }
    let mut coefficient_counter = 0usize;

    for (component_idx, component_tile) in tile.component_tiles().enumerate() {
        let d_start = storage.decompositions.len();
        let mut resolution_tiles = component_tile.resolution_tiles();

        let mut build_sub_band = |sub_band_type: SubBandType,
                                  resolution_tile: &ResolutionTile<'_>,
                                  storage: &mut DecompositionStorage<'_>|
         -> Result<usize> {
            let sub_band_rect = resolution_tile.sub_band_rect(sub_band_type);

            ltrace!(
                "r {} making sub-band {} for component {}",
                resolution_tile.resolution,
                sub_band_type as u8,
                component_idx
            );
            ltrace!(
                "Sub-band rect: [{},{} {}x{}], ll rect [{},{} {}x{}]",
                sub_band_rect.x0,
                sub_band_rect.y0,
                sub_band_rect.width(),
                sub_band_rect.height(),
                resolution_tile.rect.x0,
                resolution_tile.rect.y0,
                resolution_tile.rect.width(),
                resolution_tile.rect.height(),
            );

            let precincts = build_precincts(resolution_tile, sub_band_rect, tile, storage)?;

            let added_coefficients = (sub_band_rect.width() as usize)
                .checked_mul(sub_band_rect.height() as usize)
                .ok_or(ValidationError::ImageTooLarge)?;
            let coefficient_end = coefficient_counter
                .checked_add(added_coefficients)
                .ok_or(ValidationError::ImageTooLarge)?;
            if coefficient_end > storage.coefficients.len() {
                return Err(DecodingError::InvalidPrecinct.into());
            }
            let coefficients = coefficient_counter..coefficient_end;
            coefficient_counter = coefficient_end;

            let idx = storage.sub_bands.len();
            push_preallocated(
                &mut storage.sub_bands,
                SubBand {
                    sub_band_type,
                    rect: sub_band_rect,
                    precincts: precincts.clone(),
                    coefficients,
                },
            )?;

            Ok(idx)
        };

        // Resolution 0 always is the LL sub-band.
        let ll_resolution_tile = resolution_tiles
            .next()
            .ok_or(DecodingError::InvalidPrecinct)?;
        let first_ll_sub_band = build_sub_band(SubBandType::LowLow, &ll_resolution_tile, storage)?;

        for resolution_tile in resolution_tiles {
            let decomposition = Decomposition {
                sub_bands: [
                    build_sub_band(SubBandType::HighLow, &resolution_tile, storage)?,
                    build_sub_band(SubBandType::LowHigh, &resolution_tile, storage)?,
                    build_sub_band(SubBandType::HighHigh, &resolution_tile, storage)?,
                ],
                rect: resolution_tile.rect,
            };

            push_preallocated(&mut storage.decompositions, decomposition)?;
        }

        let d_end = storage.decompositions.len();

        push_preallocated(
            &mut storage.tile_decompositions,
            TileDecompositions {
                decompositions: d_start..d_end,
                first_ll_sub_band,
            },
        )?;
    }

    validate_built_storage(&plan, storage, coefficient_counter)?;
    storage.structural_workspace_bytes = plan.total_bytes;
    Ok(())
}

fn build_storage_is_empty(storage: &DecompositionStorage<'_>) -> bool {
    storage.segments.is_empty()
        && storage.layers.is_empty()
        && storage.code_blocks.is_empty()
        && storage.precincts.is_empty()
        && storage.tag_tree_nodes.is_empty()
        && storage.coefficients.is_empty()
        && storage.coefficients_i64.is_empty()
        && storage.sub_bands.is_empty()
        && storage.decompositions.is_empty()
        && storage.tile_decompositions.is_empty()
}

fn validate_built_storage(
    plan: &DecompositionAllocationPlan,
    storage: &DecompositionStorage<'_>,
    coefficient_count: usize,
) -> Result<()> {
    let coefficient_lengths_match = coefficient_count == storage.coefficients.len()
        && (!storage.exact_integer_decode || coefficient_count == storage.coefficients_i64.len());
    let structural_lengths_match = storage.tile_decompositions.len() == plan.tile_decompositions
        && storage.decompositions.len() == plan.decompositions
        && storage.sub_bands.len() == plan.sub_bands
        && storage.precincts.len() == plan.precincts
        && storage.code_blocks.len() == plan.code_blocks
        && storage.layers.len() == plan.layers
        && storage.tag_tree_nodes.len() == plan.tag_tree_nodes;
    if !coefficient_lengths_match || !structural_lengths_match {
        return Err(DecodingError::InvalidPrecinct.into());
    }
    Ok(())
}

fn build_precincts(
    resolution_tile: &ResolutionTile<'_>,
    sub_band_rect: IntRect,
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
) -> Result<Range<usize>> {
    let start = storage.precincts.len();

    for precinct_data in resolution_tile
        .precincts()
        .ok_or(DecodingError::InvalidPrecinct)?
    {
        let grid = code_block_grid(resolution_tile, sub_band_rect, precinct_data.rect)?;

        ltrace!(
            "Precinct rect: [{},{} {}x{}], num_code_blocks_wide: {}, num_code_blocks_high: {}",
            precinct_data.rect.x0,
            precinct_data.rect.y0,
            precinct_data.rect.width(),
            precinct_data.rect.height(),
            grid.columns,
            grid.rows
        );

        let blocks = build_code_blocks(
            grid.area,
            sub_band_rect,
            resolution_tile,
            grid.columns,
            grid.rows,
            tile,
            storage,
        )?;

        let tree_node_count = tag_tree_node_count(grid.columns, grid.rows)?;
        let tree_nodes_end = tree_node_count
            .checked_mul(2)
            .and_then(|count| storage.tag_tree_nodes.len().checked_add(count))
            .ok_or(ValidationError::ImageTooLarge)?;
        if tree_nodes_end > storage.tag_tree_nodes.capacity() {
            return Err(DecodingError::HostAllocationFailed.into());
        }
        let code_inclusion_tree =
            TagTree::new(grid.columns, grid.rows, &mut storage.tag_tree_nodes);
        let zero_bitplane_tree = TagTree::new(grid.columns, grid.rows, &mut storage.tag_tree_nodes);
        if storage.tag_tree_nodes.len() != tree_nodes_end {
            return Err(DecodingError::InvalidPrecinct.into());
        }

        push_preallocated(
            &mut storage.precincts,
            Precinct {
                code_blocks: blocks,
                code_inclusion_tree,
                zero_bitplane_tree,
            },
        )?;
    }

    let end = storage.precincts.len();

    Ok(start..end)
}

fn build_code_blocks(
    code_block_area: IntRect,
    sub_band_rect: IntRect,
    tile_instance: &ResolutionTile<'_>,
    code_blocks_x: u32,
    code_blocks_y: u32,
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
) -> Result<Range<usize>> {
    let mut y = code_block_area.y0;

    let code_block_width = tile_instance.code_block_width();
    let code_block_height = tile_instance.code_block_height();

    let start = storage.code_blocks.len();

    for y_idx in 0..code_blocks_y {
        let mut x = code_block_area.x0;

        for x_idx in 0..code_blocks_x {
            // "Code-blocks in the partition may extend beyond the boundaries of
            // the sub-band coefficients. When this happens, only the
            // coefficients lying within the sub-band are coded using the method
            // described in Annex D."
            let area = IntRect::from_xywh(x, y, code_block_width, code_block_height)
                .intersect(sub_band_rect);

            ltrace!(
                "Codeblock rect: [{},{} {}x{}]",
                area.x0,
                area.y0,
                area.width(),
                area.height(),
            );

            let start = storage.layers.len();
            let end = start
                .checked_add(usize::from(tile.num_layers))
                .ok_or(ValidationError::ImageTooLarge)?;
            if end > storage.layers.capacity() {
                return Err(DecodingError::HostAllocationFailed.into());
            }
            storage.layers.extend(iter::repeat_n(
                Layer {
                    // This will be updated once we actually read the
                    // layer segments.
                    segments: None,
                },
                tile.num_layers as usize,
            ));
            if storage.layers.len() != end {
                return Err(DecodingError::InvalidPrecinct.into());
            }

            push_preallocated(
                &mut storage.code_blocks,
                CodeBlock {
                    x_idx,
                    y_idx,
                    rect: area,
                    has_been_included: false,
                    missing_bit_planes: 0,
                    l_block: 3,
                    number_of_coding_passes: 0,
                    layers: start..end,
                    non_empty_layer_count: 0,
                },
            )?;

            x = x
                .checked_add(code_block_width)
                .ok_or(ValidationError::ImageTooLarge)?;
        }

        y = y
            .checked_add(code_block_height)
            .ok_or(ValidationError::ImageTooLarge)?;
    }

    let end = storage.code_blocks.len();

    Ok(start..end)
}

pub(crate) struct Decomposition {
    /// In the order low-high, high-low and high-high.
    pub(crate) sub_bands: [usize; 3],
    /// The rectangle of the decomposition.
    pub(crate) rect: IntRect,
}

#[derive(Clone)]
#[expect(
    clippy::struct_field_names,
    reason = "sub_band_type matches JPEG 2000 specification terminology throughout the codec"
)]
pub(crate) struct SubBand {
    pub(crate) sub_band_type: SubBandType,
    pub(crate) rect: IntRect,
    pub(crate) precincts: Range<usize>,
    pub(crate) coefficients: Range<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubBandType {
    LowLow = 0,
    HighLow = 1,
    LowHigh = 2,
    HighHigh = 3,
}

#[derive(Clone)]
pub(crate) struct Precinct {
    pub(crate) code_blocks: Range<usize>,
    pub(crate) code_inclusion_tree: TagTree,
    pub(crate) zero_bitplane_tree: TagTree,
}

pub(crate) struct PrecinctData {
    /// The x coordinate mapped back to the reference grid.
    pub(crate) r_x: u32,
    /// The y coordinate mapped back to the reference grid.
    pub(crate) r_y: u32,
    /// The actual rectangle of the precinct (in the sub-band coordinate
    /// system).
    pub(crate) rect: IntRect,
    /// The index of the precinct in the sub-band.
    pub(crate) idx: u64,
}

#[derive(Clone)]
pub(crate) struct CodeBlock {
    pub(crate) rect: IntRect,
    pub(crate) x_idx: u32,
    pub(crate) y_idx: u32,
    pub(crate) layers: Range<usize>,
    pub(crate) has_been_included: bool,
    pub(crate) missing_bit_planes: u8,
    pub(crate) number_of_coding_passes: u8,
    pub(crate) l_block: u32,
    pub(crate) non_empty_layer_count: u8,
}

pub(crate) struct Segment<'a> {
    pub(crate) idx: u8,
    pub(crate) coding_pases: u8,
    pub(crate) data_length: u32,
    pub(crate) data: &'a [u8],
}

#[derive(Clone)]
pub(crate) struct Layer {
    pub(crate) segments: Option<Range<usize>>,
}
