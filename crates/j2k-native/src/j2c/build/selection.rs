// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region decode chooses which precincts need code-blocks and tag trees.
//!
//! A precinct's code-blocks and tag trees are only used when its packet
//! headers are parsed. Region decode parses a packet header when the packet
//! holds a code-block the ROI plan requires, or, when the tile's packets
//! cannot all be skipped by PLT length, when the packet precedes the last
//! required packet in progression order. Every other precinct is built empty,
//! which keeps precinct indices stable.

use alloc::vec::Vec;

use super::allocation::{checked_include_elements, code_block_grid};
use super::{Decomposition, Precinct, SubBand, SubBandType};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::codestream::Header;
use crate::j2c::decode::{
    DecodeAllocationBudget, DecompositionStorage, OutputRegion, TileDecompositions,
};
use crate::j2c::progression::{progression_iterator, ProgressionData};
use crate::j2c::rect::IntRect;
use crate::j2c::roi::RoiPlan;
use crate::j2c::tag_tree::TagNode;
use crate::j2c::tile::Tile;
use crate::DEFAULT_MAX_DECODE_BYTES;
use core::mem::size_of;

/// How region decode uses one precinct of one component and resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum PrecinctUse {
    /// Never parsed: built with no code-blocks and root-only tag trees.
    Unused,
    /// Parsed only to reach later packets; its code-blocks are not decoded.
    Parsed,
    /// Holds a code-block the ROI plan requires.
    Required,
}

impl PrecinctUse {
    /// Whether `build` gives the precinct its code-blocks and tag trees.
    pub(crate) const fn is_built(self) -> bool {
        !matches!(self, Self::Unused)
    }
}

/// The precincts a region decode builds and parses, from selection through
/// packet parsing.
pub(crate) struct PrecinctSelection {
    /// One entry per precinct in build order: each component's resolutions
    /// from 0, each resolution's precincts in order.
    uses: Vec<PrecinctUse>,
    /// Start of each (component, resolution) run in `uses`.
    runs: Vec<usize>,
    /// First run of each component.
    component_runs: Vec<usize>,
    /// Whether every packet of the tile can be skipped by its PLT length.
    skip_by_length: bool,
    /// Packets of required precincts, all layers.
    required_packets: usize,
    /// Live bytes of the vectors above, charged while the selection lives.
    owned_bytes: usize,
    /// Progression positions walked to find the parsed precincts.
    scanned_packets: usize,
    /// Empty decomposition capacity discarded before this tile's selection.
    released_storage_bytes: usize,
}

impl PrecinctSelection {
    /// The precinct uses for the `run`-th resolution in build order.
    pub(crate) fn run(&self, run: usize) -> Option<&[PrecinctUse]> {
        let start = *self.runs.get(run)?;
        let end = self.runs.get(run + 1).copied().unwrap_or(self.uses.len());
        self.uses.get(start..end)
    }

    /// How the decode uses the packet's precinct, or `None` when the packet
    /// names a precinct the tile does not have.
    pub(crate) fn packet_use(&self, packet: &ProgressionData) -> Option<PrecinctUse> {
        self.index(packet).map(|index| self.uses[index])
    }

    /// Whether the parser skips every unused packet by its PLT length. When
    /// false, it parses every packet header up to the last required packet.
    pub(crate) const fn skip_by_length(&self) -> bool {
        self.skip_by_length
    }

    /// Packets the parser must read before the rest of the tile can be left.
    pub(crate) const fn required_packets(&self) -> usize {
        self.required_packets
    }

    /// Bytes owned by the selection, to charge while it is live.
    pub(crate) const fn owned_bytes(&self) -> usize {
        self.owned_bytes
    }

    /// Progression positions walked to find the parsed precincts.
    pub(crate) const fn scanned_packets(&self) -> usize {
        self.scanned_packets
    }

    pub(crate) const fn released_storage_bytes(&self) -> usize {
        self.released_storage_bytes
    }

    fn index(&self, packet: &ProgressionData) -> Option<usize> {
        let run = self
            .component_runs
            .get(usize::from(packet.component))?
            .checked_add(usize::from(packet.resolution))?;
        let start = *self.runs.get(run)?;
        let end = self.runs.get(run + 1).copied().unwrap_or(self.uses.len());
        let index = start.checked_add(usize::try_from(packet.precinct).ok()?)?;
        (index < end).then_some(index)
    }
}

/// Element counts the selection needs, known from the tile geometry alone.
struct SelectionCounts {
    components: usize,
    decompositions: usize,
    sub_bands: usize,
    runs: usize,
    /// Precincts per (component, resolution), summed.
    run_precincts: usize,
    /// Precincts per sub-band, summed: what `build` stores.
    sub_band_precincts: usize,
}

/// Build the ROI plan for a region decode and choose the precincts the decode
/// will parse. Leaves `storage.roi_plan` set and the rest of `storage` empty.
/// Returns `None` when no ROI plan applies, so the whole tile is built.
///
/// The selection's vectors stay live through packet parsing. The caller
/// charges [`PrecinctSelection::owned_bytes`] on top of
/// `retained_baseline_bytes` until it drops the selection.
pub(crate) fn select_region_precincts(
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &mut DecompositionStorage<'_>,
    output_region: OutputRegion,
    retained_baseline_bytes: usize,
) -> Result<Option<PrecinctSelection>> {
    // Region selection allocates a skeleton, ROI plan, selection map and, for
    // position-major orders, a progression workspace at the same time. A
    // previous tile's empty capacities are not useful during this phase and
    // must not remain live beside those owners.
    let released_storage_bytes = storage.retained_capacity_bytes()?;
    let exact_integer_decode = storage.exact_integer_decode;
    storage.release_all_allocations();
    storage.exact_integer_decode = exact_integer_decode;

    let counts = count(tile)?;
    let owned_bytes = selection_owned_bytes(&counts)?;
    preflight_unavoidable_minimum(&counts, retained_baseline_bytes, owned_bytes)?;

    let mut budget = DecodeAllocationBudget::from_live_bytes(retained_baseline_bytes)?;
    build_skeleton(tile, &counts, storage, &mut budget)?;
    storage.roi_plan = RoiPlan::build(tile, header, storage, output_region, &mut budget)?;
    let selection = match storage.roi_plan.as_ref() {
        Some(plan) => Some(select(
            tile,
            storage,
            plan,
            &counts,
            &mut budget,
            released_storage_bytes,
        )?),
        None => None,
    };
    storage.sub_bands.clear();
    storage.decompositions.clear();
    storage.tile_decompositions.clear();
    Ok(selection)
}

fn count(tile: &Tile<'_>) -> Result<SelectionCounts> {
    let mut counts = SelectionCounts {
        components: 0,
        decompositions: 0,
        sub_bands: 0,
        runs: 0,
        run_precincts: 0,
        sub_band_precincts: 0,
    };
    let add = |total: &mut usize, count: usize| -> Result<()> {
        *total = total
            .checked_add(count)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    };
    for component_tile in tile.component_tiles() {
        add(&mut counts.components, 1)?;
        for resolution_tile in component_tile.resolution_tiles() {
            let precincts = usize::try_from(resolution_tile.num_precincts())
                .map_err(|_| ValidationError::ImageTooLarge)?;
            let sub_bands = if resolution_tile.resolution == 0 {
                1
            } else {
                add(&mut counts.decompositions, 1)?;
                3
            };
            add(&mut counts.sub_bands, sub_bands)?;
            add(&mut counts.runs, 1)?;
            add(&mut counts.run_precincts, precincts)?;
            add(
                &mut counts.sub_band_precincts,
                precincts
                    .checked_mul(sub_bands)
                    .ok_or(ValidationError::ImageTooLarge)?,
            )?;
        }
    }
    Ok(counts)
}

fn selection_owned_bytes(counts: &SelectionCounts) -> Result<usize> {
    let mut bytes = 0usize;
    checked_include_elements::<PrecinctUse>(&mut bytes, counts.run_precincts)?;
    checked_include_elements::<usize>(&mut bytes, counts.runs)?;
    checked_include_elements::<usize>(&mut bytes, counts.components)?;
    Ok(bytes)
}

/// Reject a tile whose structure exceeds the cap however few precincts are
/// kept, before allocating or walking the progression.
///
/// `build` stores every sub-band precinct, each with two root tag-tree nodes
/// even when it has no code-blocks, plus the decomposition skeleton and the
/// ROI windows. All of that is live alongside the selection and the retained
/// baseline.
fn preflight_unavoidable_minimum(
    counts: &SelectionCounts,
    retained_baseline_bytes: usize,
    owned_bytes: usize,
) -> Result<()> {
    if retained_baseline_bytes > DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    let mut bytes = retained_baseline_bytes;
    let mut include_bytes = |count: usize| -> Result<()> {
        bytes = bytes
            .checked_add(count)
            .filter(|bytes| *bytes <= DEFAULT_MAX_DECODE_BYTES)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    };
    include_bytes(owned_bytes)?;
    let mut structure = 0usize;
    checked_include_elements::<TileDecompositions>(&mut structure, counts.components)?;
    checked_include_elements::<Decomposition>(&mut structure, counts.decompositions)?;
    checked_include_elements::<SubBand>(&mut structure, counts.sub_bands)?;
    checked_include_elements::<Option<IntRect>>(
        &mut structure,
        counts
            .sub_bands
            .checked_add(counts.decompositions)
            .and_then(|count| count.checked_add(counts.components))
            .ok_or(ValidationError::ImageTooLarge)?,
    )?;
    checked_include_elements::<Precinct>(&mut structure, counts.sub_band_precincts)?;
    checked_include_elements::<TagNode>(
        &mut structure,
        counts
            .sub_band_precincts
            .checked_mul(2)
            .ok_or(ValidationError::ImageTooLarge)?,
    )?;
    include_bytes(structure)
}

/// Push the decompositions and sub-bands `build` would create, in the same
/// order, without precincts or coefficients, so the ROI plan's indices match
/// the full build.
fn build_skeleton(
    tile: &Tile<'_>,
    counts: &SelectionCounts,
    storage: &mut DecompositionStorage<'_>,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    budget.reserve_new(&mut storage.sub_bands, counts.sub_bands)?;
    budget.reserve_new(&mut storage.decompositions, counts.decompositions)?;
    budget.reserve_new(&mut storage.tile_decompositions, counts.components)?;

    for component_tile in tile.component_tiles() {
        let start = storage.decompositions.len();
        let mut resolution_tiles = component_tile.resolution_tiles();
        let ll = resolution_tiles
            .next()
            .ok_or(DecodingError::InvalidPrecinct)?;
        let first_ll_sub_band =
            push_skeleton_sub_band(storage, ll.sub_band_rect(SubBandType::LowLow))?;
        for resolution_tile in resolution_tiles {
            let mut sub_bands = [0; 3];
            for (slot, sub_band_type) in sub_bands.iter_mut().zip([
                SubBandType::HighLow,
                SubBandType::LowHigh,
                SubBandType::HighHigh,
            ]) {
                *slot =
                    push_skeleton_sub_band(storage, resolution_tile.sub_band_rect(sub_band_type))?;
            }
            push_within_capacity(
                &mut storage.decompositions,
                Decomposition {
                    sub_bands,
                    rect: resolution_tile.rect,
                },
            )?;
        }
        push_within_capacity(
            &mut storage.tile_decompositions,
            TileDecompositions {
                decompositions: start..storage.decompositions.len(),
                first_ll_sub_band,
            },
        )?;
    }
    Ok(())
}

fn push_skeleton_sub_band(storage: &mut DecompositionStorage<'_>, rect: IntRect) -> Result<usize> {
    let idx = storage.sub_bands.len();
    push_within_capacity(
        &mut storage.sub_bands,
        SubBand {
            // The type is not read before the full build replaces this entry.
            sub_band_type: SubBandType::LowLow,
            rect,
            precincts: 0..0,
            coefficients: 0..0,
            storage_rect: rect,
        },
    )?;
    Ok(idx)
}

/// Push into capacity reserved from the up-front counts, never growing.
fn push_within_capacity<T>(values: &mut Vec<T>, value: T) -> Result<()> {
    if values.len() == values.capacity() {
        return Err(DecodingError::InvalidPrecinct.into());
    }
    values.push(value);
    Ok(())
}

fn select(
    tile: &Tile<'_>,
    storage: &DecompositionStorage<'_>,
    plan: &RoiPlan,
    counts: &SelectionCounts,
    budget: &mut DecodeAllocationBudget,
    released_storage_bytes: usize,
) -> Result<PrecinctSelection> {
    let mut uses = Vec::new();
    budget.reserve_new(&mut uses, counts.run_precincts)?;
    let mut runs = Vec::new();
    budget.reserve_new(&mut runs, counts.runs)?;
    let mut component_runs = Vec::new();
    budget.reserve_new(&mut component_runs, counts.components)?;

    let layers = usize::from(tile.num_layers);
    let mut required_packets = 0usize;
    for (component_tile, tile_decompositions) in
        tile.component_tiles().zip(&storage.tile_decompositions)
    {
        push_within_capacity(&mut component_runs, runs.len())?;
        for (resolution, resolution_tile) in component_tile.resolution_tiles().enumerate() {
            push_within_capacity(&mut runs, uses.len())?;
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
                if required {
                    required_packets = required_packets
                        .checked_add(layers)
                        .ok_or(ValidationError::ImageTooLarge)?;
                }
                push_within_capacity(
                    &mut uses,
                    if required {
                        PrecinctUse::Required
                    } else {
                        PrecinctUse::Unused
                    },
                )?;
            }
        }
    }

    let owned_bytes = uses
        .capacity()
        .checked_mul(size_of::<PrecinctUse>())
        .and_then(|bytes| {
            runs.capacity()
                .checked_mul(size_of::<usize>())
                .and_then(|run_bytes| bytes.checked_add(run_bytes))
        })
        .and_then(|bytes| {
            component_runs
                .capacity()
                .checked_mul(size_of::<usize>())
                .and_then(|component_bytes| bytes.checked_add(component_bytes))
        })
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut selection = PrecinctSelection {
        uses,
        runs,
        component_runs,
        skip_by_length: packets_skippable_by_length(tile, counts, layers)?,
        required_packets,
        owned_bytes,
        scanned_packets: 0,
        released_storage_bytes,
    };
    if !selection.skip_by_length {
        mark_precincts_parsed_before_last_required(tile, &mut selection, budget.live_bytes())?;
    }
    Ok(selection)
}

/// Whether every packet of the tile can be skipped by its PLT length: every
/// tile-part keeps its headers in-band and signals PLT lengths, and together
/// they cover each packet of the progression. The decision is tile-wide,
/// because a skipped packet leaves its precinct's tag trees and Lblock state
/// unread, so a later packet of that precinct could never be parsed.
fn packets_skippable_by_length(
    tile: &Tile<'_>,
    counts: &SelectionCounts,
    layers: usize,
) -> Result<bool> {
    // Progression changes can leave packets out of the progression, so its
    // length is only known by walking it. Parse those tiles instead.
    if !tile.progression_changes.is_empty() {
        return Ok(false);
    }
    let mut lengths = 0usize;
    for tile_part in &tile.tile_parts {
        let Some(count) = tile_part.skippable_packet_count() else {
            return Ok(false);
        };
        lengths = lengths
            .checked_add(count)
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    // Without progression changes, the progression visits each precinct of
    // each component and resolution once per layer.
    let packets = counts
        .run_precincts
        .checked_mul(layers)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok(lengths == packets)
}

/// Without PLT skipping, the parser reads every packet header up to the last
/// required packet. Mark each precinct with a packet in that range as parsed.
///
/// The walk stops once every required packet has been seen. It never goes past
/// the tile's packet-header bytes either: every packet header takes at least
/// one byte, so the parser cannot reach a later packet.
fn mark_precincts_parsed_before_last_required(
    tile: &Tile<'_>,
    selection: &mut PrecinctSelection,
    retained_live_bytes: usize,
) -> Result<()> {
    let mut remaining = selection.required_packets;
    if remaining == 0 {
        return Ok(());
    }
    let parseable_packets = tile
        .tile_parts
        .iter()
        .try_fold(0usize, |total, tile_part| {
            total.checked_add(tile_part.packet_header_bytes())
        })
        .ok_or(ValidationError::ImageTooLarge)?;
    for packet in progression_iterator(tile, retained_live_bytes)?.take(parseable_packets) {
        selection.scanned_packets += 1;
        let Some(index) = selection.index(&packet) else {
            // The parser rejects a packet for a precinct the tile lacks.
            continue;
        };
        match selection.uses[index] {
            PrecinctUse::Required => {
                remaining = remaining.saturating_sub(1);
                if remaining == 0 {
                    break;
                }
            }
            PrecinctUse::Unused => selection.uses[index] = PrecinctUse::Parsed,
            PrecinctUse::Parsed => {}
        }
    }
    Ok(())
}
