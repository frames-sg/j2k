// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region decode chooses which precincts need code-blocks and tag trees.
//!
//! A precinct's code-blocks and tag trees are only used when its packet
//! headers are parsed. Region decode parses a packet header when the packet
//! holds a code-block the ROI plan requires, or, without PLT lengths to skip
//! by, when the packet precedes the last required packet in progression order.
//! Every other precinct is built empty, which keeps precinct indices stable.

use alloc::vec::Vec;
use core::mem::size_of;

use super::allocation::code_block_grid;
use super::{Decomposition, Precinct, SubBand, SubBandType};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::codestream::Header;
use crate::j2c::decode::{DecompositionStorage, OutputRegion, TileDecompositions};
use crate::j2c::progression::progression_iterator;
use crate::j2c::rect::IntRect;
use crate::j2c::roi::RoiPlan;
use crate::j2c::tile::Tile;
use crate::{try_reserve_decode_elements, try_resize_decode_elements, DEFAULT_MAX_DECODE_BYTES};

/// Precincts to build with code-blocks and tag trees, in build order: each
/// component's resolutions from 0, each resolution's precincts in order.
pub(crate) struct PrecinctSelection {
    keep: Vec<bool>,
    /// Start of each (component, resolution) run in `keep`.
    runs: Vec<usize>,
}

impl PrecinctSelection {
    /// The keep flags for the `run`-th resolution in build order.
    pub(crate) fn run(&self, run: usize) -> Option<&[bool]> {
        let start = *self.runs.get(run)?;
        let end = self.runs.get(run + 1).copied().unwrap_or(self.keep.len());
        self.keep.get(start..end)
    }
}

/// Build the ROI plan for a region decode and choose the precincts the decode
/// will parse. Leaves `storage.roi_plan` set and the rest of `storage` empty.
/// Returns `None` when no ROI plan applies, so the whole tile is built.
pub(crate) fn select_region_precincts(
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'_>,
    output_region: OutputRegion,
) -> Result<Option<PrecinctSelection>> {
    let precinct_count = validate_precinct_count(tile)?;
    build_skeleton(tile, storage)?;
    storage.roi_plan = RoiPlan::build(tile, header, storage, output_region)?;
    let selection = match storage.roi_plan.as_ref() {
        Some(plan) => Some(select(tile, storage, plan, precinct_count)?),
        None => None,
    };
    storage.sub_bands.clear();
    storage.decompositions.clear();
    storage.tile_decompositions.clear();
    Ok(selection)
}

/// Reject a precinct grid the structural budget would reject anyway, before
/// allocating per-precinct selection state for it.
fn validate_precinct_count(tile: &Tile<'_>) -> Result<usize> {
    let mut count = 0usize;
    for component_tile in tile.component_tiles() {
        for resolution_tile in component_tile.resolution_tiles() {
            let precincts = usize::try_from(resolution_tile.num_precincts())
                .map_err(|_| ValidationError::ImageTooLarge)?;
            count = count
                .checked_add(precincts)
                .ok_or(ValidationError::ImageTooLarge)?;
        }
    }
    if count
        .checked_mul(size_of::<Precinct>())
        .is_none_or(|bytes| bytes > DEFAULT_MAX_DECODE_BYTES)
    {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(count)
}

/// Push the decompositions and sub-bands `build` would create, in the same
/// order, without precincts or coefficients, so the ROI plan's indices match
/// the full build.
fn build_skeleton(tile: &Tile<'_>, storage: &mut DecompositionStorage<'_>) -> Result<()> {
    let mut sub_bands = 0usize;
    let mut decompositions = 0usize;
    let mut components = 0usize;
    for component_tile in tile.component_tiles() {
        components += 1;
        let levels = component_tile.resolution_tiles().count().saturating_sub(1);
        decompositions = decompositions
            .checked_add(levels)
            .ok_or(ValidationError::ImageTooLarge)?;
        sub_bands = levels
            .checked_mul(3)
            .and_then(|count| count.checked_add(1))
            .and_then(|count| sub_bands.checked_add(count))
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    try_reserve_decode_elements(&mut storage.sub_bands, sub_bands)?;
    try_reserve_decode_elements(&mut storage.decompositions, decompositions)?;
    try_reserve_decode_elements(&mut storage.tile_decompositions, components)?;

    for component_tile in tile.component_tiles() {
        let start = storage.decompositions.len();
        let mut resolution_tiles = component_tile.resolution_tiles();
        let ll = resolution_tiles
            .next()
            .ok_or(DecodingError::InvalidPrecinct)?;
        let first_ll_sub_band =
            push_skeleton_sub_band(storage, ll.sub_band_rect(SubBandType::LowLow));
        for resolution_tile in resolution_tiles {
            let sub_bands = [
                SubBandType::HighLow,
                SubBandType::LowHigh,
                SubBandType::HighHigh,
            ]
            .map(|sub_band_type| {
                push_skeleton_sub_band(storage, resolution_tile.sub_band_rect(sub_band_type))
            });
            storage.decompositions.push(Decomposition {
                sub_bands,
                rect: resolution_tile.rect,
            });
        }
        storage.tile_decompositions.push(TileDecompositions {
            decompositions: start..storage.decompositions.len(),
            first_ll_sub_band,
        });
    }
    Ok(())
}

fn push_skeleton_sub_band(storage: &mut DecompositionStorage<'_>, rect: IntRect) -> usize {
    let idx = storage.sub_bands.len();
    storage.sub_bands.push(SubBand {
        // The type is not read before the full build replaces this entry.
        sub_band_type: SubBandType::LowLow,
        rect,
        precincts: 0..0,
        coefficients: 0..0,
        storage_rect: rect,
    });
    idx
}

fn select(
    tile: &Tile<'_>,
    storage: &DecompositionStorage<'_>,
    plan: &RoiPlan,
    precinct_count: usize,
) -> Result<PrecinctSelection> {
    let mut keep = Vec::new();
    try_reserve_decode_elements(&mut keep, precinct_count)?;
    let mut runs = Vec::new();
    // First run of each component, to map a packet's resolution to its run.
    let mut component_runs = Vec::new();
    try_reserve_decode_elements(&mut component_runs, storage.tile_decompositions.len())?;

    for (component_tile, tile_decompositions) in
        tile.component_tiles().zip(&storage.tile_decompositions)
    {
        component_runs.push(runs.len());
        for (resolution, resolution_tile) in component_tile.resolution_tiles().enumerate() {
            try_reserve_decode_elements(&mut runs, 1)?;
            runs.push(keep.len());
            let (sub_band_types, sub_band_indices): (&[SubBandType], [usize; 3]) =
                if resolution == 0 {
                    (
                        &[SubBandType::LowLow],
                        [tile_decompositions.first_ll_sub_band; 3],
                    )
                } else {
                    let decomposition = storage
                        .decompositions
                        .get(tile_decompositions.decompositions.start + resolution - 1)
                        .ok_or(DecodingError::InvalidPrecinct)?;
                    (
                        &[
                            SubBandType::HighLow,
                            SubBandType::LowHigh,
                            SubBandType::HighHigh,
                        ],
                        decomposition.sub_bands,
                    )
                };
            for precinct in resolution_tile
                .precincts()
                .ok_or(DecodingError::InvalidPrecinct)?
            {
                let mut required = false;
                for (sub_band_type, sub_band_idx) in sub_band_types.iter().zip(sub_band_indices) {
                    let Some(window) = plan.sub_band_window(sub_band_idx) else {
                        continue;
                    };
                    let rect = resolution_tile.sub_band_rect(*sub_band_type);
                    let grid = code_block_grid(&resolution_tile, rect, precinct.rect)?;
                    if grid.columns > 0
                        && grid.rows > 0
                        && window.intersects(grid.area.intersect(rect))
                    {
                        required = true;
                        break;
                    }
                }
                if keep.len() == keep.capacity() {
                    return Err(DecodingError::InvalidPrecinct.into());
                }
                keep.push(required);
            }
        }
    }

    if !packets_skippable_by_length(tile)? {
        keep_precincts_parsed_before_last_required(tile, &mut keep, &runs, &component_runs)?;
    }
    Ok(PrecinctSelection { keep, runs })
}

/// Whether every packet of the tile can be skipped by its PLT length.
fn packets_skippable_by_length(tile: &Tile<'_>) -> Result<bool> {
    let mut lengths = 0usize;
    for tile_part in &tile.tile_parts {
        let Some(count) = tile_part.skippable_packet_count() else {
            return Ok(false);
        };
        lengths = lengths
            .checked_add(count)
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    // Lengths that do not cover every packet would leave a packet the parser
    // cannot skip.
    Ok(lengths == progression_iterator(tile)?.count())
}

/// Without PLT, the parser reads every packet header up to the last required
/// packet. Keep each precinct with a packet in that range.
fn keep_precincts_parsed_before_last_required(
    tile: &Tile<'_>,
    keep: &mut [bool],
    runs: &[usize],
    component_runs: &[usize],
) -> Result<()> {
    let flat_index = |component: u16, resolution: u8, precinct: u64| -> Option<usize> {
        let run = component_runs
            .get(usize::from(component))?
            .checked_add(usize::from(resolution))?;
        let start = *runs.get(run)?;
        let end = runs.get(run + 1).copied().unwrap_or(keep.len());
        let index = start.checked_add(usize::try_from(precinct).ok()?)?;
        (index < end).then_some(index)
    };

    let mut last_required = None;
    for (position, packet) in progression_iterator(tile)?.enumerate() {
        if flat_index(packet.component, packet.resolution, packet.precinct)
            .is_none_or(|index| keep[index])
        {
            last_required = Some(position);
        }
    }
    let Some(last_required) = last_required else {
        return Ok(());
    };
    let mut parsed = Vec::new();
    try_resize_decode_elements(&mut parsed, keep.len(), false)?;
    for packet in progression_iterator(tile)?.take(last_required) {
        if let Some(index) = flat_index(packet.component, packet.resolution, packet.precinct) {
            parsed[index] = true;
        }
    }
    for (keep, parsed) in keep.iter_mut().zip(parsed) {
        *keep |= parsed;
    }
    Ok(())
}
