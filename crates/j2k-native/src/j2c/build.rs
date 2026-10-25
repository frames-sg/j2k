//! Building and setting up decompositions, sub-bands, precincts and code-blocks.

use super::decode::{DecompositionStorage, TileDecompositions};
use super::rect::IntRect;
use super::roi::RoiPlan;
use super::tag_tree::{TagNode, TagTree};
use super::tile::{ResolutionTile, Tile};
use crate::error::{DecodingError, Result, ValidationError};
use crate::try_resize_decode_elements;
use core::{iter, ops::Range};

mod allocation;
mod selection;
pub(crate) use allocation::{
    allocate_deferred_coefficients, release_unused_roi_workspace, BuildWorkspace, CoefficientLayout,
};
use allocation::{
    code_block_grid, prepare_decomposition_storage, push_preallocated, tag_tree_node_count,
    DecompositionAllocationPlan,
};
pub(crate) use selection::{select_region_precincts, PrecinctSelection, PrecinctUse};

/// Build and allocate all necessary structures to process the code-blocks
/// for a specific tile. Also parses the segments for each code-block.
pub(crate) fn build(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    retained_baseline_bytes: usize,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
    selection: Option<&PrecinctSelection>,
) -> Result<()> {
    build_decompositions(
        tile,
        storage,
        retained_baseline_bytes,
        include_roi_workspace,
        workspace,
        selection,
    )
}

fn build_decompositions(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    retained_baseline_bytes: usize,
    include_roi_workspace: bool,
    workspace: BuildWorkspace,
    selection: Option<&PrecinctSelection>,
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
        selection,
    )?;
    resize_planned_coefficients(storage, &plan)?;
    // Building reads the ROI plan while it fills the rest of storage, so it
    // is held aside until the components are built.
    let roi_plan = storage.roi_plan.take();
    let built = build_components(
        tile,
        storage,
        &plan,
        workspace,
        selection,
        selection.and(roi_plan.as_ref()),
    );
    storage.roi_plan = roi_plan;
    let coefficient_extent = built?;
    build_scratch_precincts(storage, &plan)?;

    validate_built_storage(&plan, storage, coefficient_extent)?;
    storage.structural_workspace_bytes = plan.total_bytes;
    Ok(())
}

/// Build each component's sub-bands, precincts and code-blocks in decode
/// order, and return the end of the furthest coefficient range. With
/// `roi_plan`, only the code-blocks it requires get layer records.
#[expect(
    clippy::too_many_lines,
    reason = "one pass builds each component's sub-band, precinct, and coefficient ranges in decode order"
)]
fn build_components(
    tile: &Tile<'_>,
    storage: &mut DecompositionStorage<'_>,
    plan: &DecompositionAllocationPlan,
    workspace: BuildWorkspace,
    selection: Option<&PrecinctSelection>,
    roi_plan: Option<&RoiPlan>,
) -> Result<usize> {
    let coefficient_layout = workspace.coefficient_layout();
    storage.coefficient_layout = coefficient_layout;
    let mut coefficient_counter = 0usize;
    // The end of the furthest coefficient range, which the plan sized storage to.
    let mut coefficient_extent = 0usize;
    // Index of the current (component, resolution) in build order, for the
    // precinct selection.
    let mut resolution_run = 0usize;
    let parsed_in_scratch = selection.is_some_and(PrecinctSelection::parsed_in_scratch);

    for (component_idx, component_tile) in tile.component_tiles().enumerate() {
        if coefficient_layout == CoefficientLayout::Component {
            coefficient_counter = 0;
        }
        let d_start = storage.decompositions.len();
        let mut resolution_tiles = component_tile.resolution_tiles();

        let mut build_sub_band = |sub_band_type: SubBandType,
                                  resolution_tile: &ResolutionTile<'_>,
                                  run: usize,
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

            let keep = selected_precincts(selection, run)?;
            let precincts = build_precincts(
                resolution_tile,
                sub_band_rect,
                tile,
                keep,
                parsed_in_scratch,
                roi_plan,
                storage,
            )?;

            let added_coefficients = (sub_band_rect.width() as usize)
                .checked_mul(sub_band_rect.height() as usize)
                .ok_or(ValidationError::ImageTooLarge)?;
            let coefficient_end = coefficient_counter
                .checked_add(added_coefficients)
                .ok_or(ValidationError::ImageTooLarge)?;
            if coefficient_end > plan.coefficients {
                return Err(DecodingError::InvalidPrecinct.into());
            }
            let coefficients = coefficient_counter..coefficient_end;
            coefficient_counter = coefficient_end;
            coefficient_extent = coefficient_extent.max(coefficient_end);

            let idx = storage.sub_bands.len();
            push_preallocated(
                &mut storage.sub_bands,
                SubBand {
                    sub_band_type,
                    rect: sub_band_rect,
                    precincts: precincts.clone(),
                    coefficients,
                    storage_rect: sub_band_rect,
                },
            )?;

            Ok(idx)
        };

        // Resolution 0 always is the LL sub-band.
        let ll_resolution_tile = resolution_tiles
            .next()
            .ok_or(DecodingError::InvalidPrecinct)?;
        let first_ll_sub_band = build_sub_band(
            SubBandType::LowLow,
            &ll_resolution_tile,
            resolution_run,
            storage,
        )?;
        resolution_run += 1;

        for resolution_tile in resolution_tiles {
            let run = resolution_run;
            let decomposition = Decomposition {
                sub_bands: [
                    build_sub_band(SubBandType::HighLow, &resolution_tile, run, storage)?,
                    build_sub_band(SubBandType::LowHigh, &resolution_tile, run, storage)?,
                    build_sub_band(SubBandType::HighHigh, &resolution_tile, run, storage)?,
                ],
                rect: resolution_tile.rect,
            };
            resolution_run += 1;
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

    Ok(coefficient_extent)
}

fn resize_planned_coefficients(
    storage: &mut DecompositionStorage<'_>,
    plan: &DecompositionAllocationPlan,
) -> Result<()> {
    try_resize_decode_elements(
        &mut storage.coefficients,
        plan.coefficient_storage_len(),
        0.0,
    )?;
    if storage.exact_integer_decode {
        try_resize_decode_elements(
            &mut storage.coefficients_i64,
            plan.coefficient_storage_len(),
            0,
        )?;
    }
    Ok(())
}

/// The precinct uses for the `run`-th resolution, when region decode selected
/// precincts.
fn selected_precincts(
    selection: Option<&PrecinctSelection>,
    run: usize,
) -> Result<Option<&[PrecinctUse]>> {
    selection
        .map(|selection| {
            selection
                .run(run)
                .ok_or(DecodingError::InvalidPrecinct.into())
        })
        .transpose()
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
    let stored_coefficients = plan.coefficient_storage_len();
    let coefficient_lengths_match = coefficient_count == plan.coefficients
        && stored_coefficients == storage.coefficients.len()
        && storage.coefficients_i64.len()
            == usize::from(storage.exact_integer_decode) * stored_coefficients;
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
    keep: Option<&[PrecinctUse]>,
    parsed_in_scratch: bool,
    roi_plan: Option<&RoiPlan>,
    storage: &mut DecompositionStorage<'_>,
) -> Result<Range<usize>> {
    let start = storage.precincts.len();
    // The sub-band is pushed after its precincts, at this index.
    let sub_band_idx = storage.sub_bands.len();
    let required =
        |rect: IntRect| roi_plan.is_none_or(|plan| plan.code_block_required(sub_band_idx, rect));

    for (precinct_idx, precinct_data) in resolution_tile
        .precincts()
        .ok_or(DecodingError::InvalidPrecinct)?
        .enumerate()
    {
        let mut grid = code_block_grid(resolution_tile, sub_band_rect, precinct_data.rect)?;
        let precinct_use = keep.and_then(|keep| keep.get(precinct_idx).copied());
        if precinct_use == Some(PrecinctUse::Parsed) && parsed_in_scratch {
            // Its packets are parsed in a scratch precinct, which only needs
            // its code-block grid.
            let start = storage.code_blocks.len();
            push_preallocated(
                &mut storage.precincts,
                Precinct {
                    code_blocks: start..start,
                    code_inclusion_tree: TagTree::unplaced(grid.columns, grid.rows),
                    zero_bitplane_tree: TagTree::unplaced(grid.columns, grid.rows),
                },
            )?;
            continue;
        }
        if precinct_use.is_some_and(|precinct_use| !precinct_use.is_built(parsed_in_scratch)) {
            // Region decode never parses this precinct's packets: build it
            // with no code-blocks and root-only tag trees.
            grid.columns = 0;
            grid.rows = 0;
        }

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
            &required,
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

/// Build a precinct's code-blocks. Only those `required` keeps get layer
/// records: packet parsing reads the others' contributions only to skip them.
#[expect(
    clippy::too_many_arguments,
    reason = "the code-block grid, its tile, and the region filter stay explicit beside the storage they fill"
)]
fn build_code_blocks(
    code_block_area: IntRect,
    sub_band_rect: IntRect,
    tile_instance: &ResolutionTile<'_>,
    code_blocks_x: u32,
    code_blocks_y: u32,
    tile: &Tile<'_>,
    required: &dyn Fn(IntRect) -> bool,
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
            let area = IntRect::from_ltrb(
                x,
                y,
                x.saturating_add(code_block_width),
                y.saturating_add(code_block_height),
            )
            .intersect(sub_band_rect);

            ltrace!(
                "Codeblock rect: [{},{} {}x{}]",
                area.x0,
                area.y0,
                area.width(),
                area.height(),
            );

            let start = storage.layers.len();
            let layer_count = if required(area) {
                usize::from(tile.num_layers)
            } else {
                0
            };
            let end = start
                .checked_add(layer_count)
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
                layer_count,
            ));
            if storage.layers.len() != end {
                return Err(DecodingError::InvalidPrecinct.into());
            }

            push_preallocated(
                &mut storage.code_blocks,
                CodeBlock::new(area, x_idx, y_idx, start..end),
            )?;

            if x_idx + 1 < code_blocks_x {
                x = x
                    .checked_add(code_block_width)
                    .ok_or(ValidationError::ImageTooLarge)?;
            }
        }

        if y_idx + 1 < code_blocks_y {
            y = y
                .checked_add(code_block_height)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
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
    /// Row-major coefficients covering `storage_rect`.
    pub(crate) coefficients: Range<usize>,
    /// The part of `rect` held in `coefficients`. It is the whole sub-band,
    /// except in region decode, where it covers only the code-blocks the
    /// region needs.
    pub(crate) storage_rect: IntRect,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CodeBlockCoding {
    Classic,
    HighThroughput,
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
    /// Total packet-header passes, including HT placeholder passes and skipped HT sets.
    pub(crate) ht_total_coding_passes: u8,
    /// Global pass index of the first non-empty HT cleanup contribution.
    pub(crate) ht_first_cleanup_pass: Option<u8>,
    /// Zero-based HT set currently selected for decoding.
    pub(crate) ht_selected_set: Option<u8>,
    /// Entropy coder selected from packet syntax; absent until a codeword is identified.
    pub(crate) coding: Option<CodeBlockCoding>,
    pub(crate) l_block: u32,
    pub(crate) non_empty_layer_count: u8,
}

impl CodeBlock {
    /// A code-block before any packet has included it.
    fn new(rect: IntRect, x_idx: u32, y_idx: u32, layers: Range<usize>) -> Self {
        Self {
            x_idx,
            y_idx,
            rect,
            has_been_included: false,
            missing_bit_planes: 0,
            l_block: 3,
            number_of_coding_passes: 0,
            ht_total_coding_passes: 0,
            ht_first_cleanup_pass: None,
            ht_selected_set: None,
            coding: None,
            layers,
            non_empty_layer_count: 0,
        }
    }
}

/// Scratch precincts a region decode's parsed-only precincts borrow, one per
/// sub-band of a packet, at the end of the code-block and tag-tree storage.
///
/// In progressions that read all of a precinct's layers in a row, a
/// precinct whose code-blocks the region doesn't use is parsed only to reach
/// later packets, and its parsing state is dead after its last packet. Such
/// precincts are built with no code-blocks or tag-tree nodes, and while their
/// packets are parsed, they borrow a scratch precinct, which starts afresh for
/// each. After parsing, every precinct has given its scratch back.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParseScratch {
    /// The first code-block of each scratch precinct.
    code_block_start: [usize; SCRATCH_PRECINCTS],
    /// Code-blocks per scratch precinct.
    code_blocks: [usize; SCRATCH_PRECINCTS],
    /// The first tag-tree node of each scratch precinct.
    node_start: [usize; SCRATCH_PRECINCTS],
    /// Tag-tree nodes per scratch precinct, for both of its trees.
    nodes: [usize; SCRATCH_PRECINCTS],
    /// The precinct, as an index into the storage's precincts, that borrows
    /// each scratch precinct.
    borrowers: [Option<usize>; SCRATCH_PRECINCTS],
}

/// One scratch precinct per sub-band of a packet.
pub(crate) const SCRATCH_PRECINCTS: usize = 3;

impl ParseScratch {
    /// Lend scratch precinct `slot` to `precinct`, unless it already has it.
    /// The precinct that held it gives it back, and its state is gone, so
    /// a precinct can only borrow scratch for its first packet, at `layer` 0.
    pub(crate) fn lend(
        &mut self,
        slot: usize,
        precinct: usize,
        layer: u8,
        precincts: &mut [Precinct],
        code_blocks: &mut [CodeBlock],
        tag_tree_nodes: &mut [TagNode],
    ) -> Option<()> {
        let borrower = self.borrowers.get_mut(slot)?;
        if *borrower == Some(precinct) {
            return Some(());
        }
        if layer != 0 {
            return None;
        }
        if let Some(previous) = borrower.take() {
            give_back(precincts.get_mut(previous)?, self.code_block_start[slot]);
        }

        let target = precincts.get_mut(precinct)?;
        let columns = target.code_inclusion_tree.width();
        let count = (columns as usize).checked_mul(target.code_inclusion_tree.height() as usize)?;
        let inclusion_nodes = target.code_inclusion_tree.node_count();
        if count > self.code_blocks[slot]
            || inclusion_nodes.checked_add(target.zero_bitplane_tree.node_count())?
                > self.nodes[slot]
        {
            return None;
        }
        let start = self.code_block_start[slot];
        let blocks = code_blocks.get_mut(start..start.checked_add(count)?)?;
        // A rectangle that meets no window: no scratch code-block is read.
        let unread = IntRect::from_ltrb(0, 0, 0, 0);
        for (block, index) in blocks.iter_mut().zip(0u32..) {
            *block = CodeBlock::new(unread, index % columns, index / columns, 0..0);
        }
        let root = self.node_start[slot];
        target.code_inclusion_tree.place(root, tag_tree_nodes)?;
        target
            .zero_bitplane_tree
            .place(root.checked_add(inclusion_nodes)?, tag_tree_nodes)?;
        target.code_blocks = start..start + count;
        *borrower = Some(precinct);
        Some(())
    }

    /// Have every borrowing precinct give its scratch back.
    pub(crate) fn give_all_back(&mut self, precincts: &mut [Precinct]) {
        for (slot, borrower) in self.borrowers.iter_mut().enumerate() {
            if let Some(precinct) = borrower.take().and_then(|index| precincts.get_mut(index)) {
                give_back(precinct, self.code_block_start[slot]);
            }
        }
    }

    /// The most code-blocks a borrowing precinct can have.
    pub(crate) fn code_blocks(&self) -> usize {
        self.code_blocks.iter().copied().max().unwrap_or(0)
    }
}

/// Leave `precinct` with no code-blocks and unplaced tag trees, as built.
fn give_back(precinct: &mut Precinct, code_block_start: usize) {
    precinct.code_blocks = code_block_start..code_block_start;
    precinct.code_inclusion_tree.unplace();
    precinct.zero_bitplane_tree.unplace();
}

/// Append the scratch precincts the plan sized, when precincts are parsed in
/// scratch.
fn build_scratch_precincts(
    storage: &mut DecompositionStorage<'_>,
    plan: &DecompositionAllocationPlan,
) -> Result<()> {
    storage.parse_scratch = None;
    if plan.scratch_tree_nodes == [0; SCRATCH_PRECINCTS] {
        return Ok(());
    }
    let mut code_block_start = [0; SCRATCH_PRECINCTS];
    let mut node_start = [0; SCRATCH_PRECINCTS];
    let mut nodes = [0; SCRATCH_PRECINCTS];
    let mut code_block_end = storage.code_blocks.len();
    let mut node_end = storage.tag_tree_nodes.len();
    for slot in 0..SCRATCH_PRECINCTS {
        code_block_start[slot] = code_block_end;
        node_start[slot] = node_end;
        nodes[slot] = plan.scratch_tree_nodes[slot]
            .checked_mul(2)
            .ok_or(ValidationError::ImageTooLarge)?;
        code_block_end = code_block_end
            .checked_add(plan.scratch_code_blocks[slot])
            .ok_or(ValidationError::ImageTooLarge)?;
        node_end = node_end
            .checked_add(nodes[slot])
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    if code_block_end > storage.code_blocks.capacity()
        || node_end > storage.tag_tree_nodes.capacity()
    {
        return Err(DecodingError::HostAllocationFailed.into());
    }
    let unread = CodeBlock::new(IntRect::from_ltrb(0, 0, 0, 0), 0, 0, 0..0);
    storage.code_blocks.resize(code_block_end, unread);
    storage.tag_tree_nodes.resize(node_end, TagNode::default());
    storage.parse_scratch = Some(ParseScratch {
        code_block_start,
        code_blocks: plan.scratch_code_blocks,
        node_start,
        nodes,
        borrowers: [None; SCRATCH_PRECINCTS],
    });
    Ok(())
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
