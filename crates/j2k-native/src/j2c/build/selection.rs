// SPDX-License-Identifier: MIT OR Apache-2.0

//! Region decode chooses which precincts need code-blocks and tag trees.
//!
//! A precinct's code-blocks and tag trees are only used when its packet
//! headers are parsed. Region decode parses a packet header when the packet
//! holds a code-block the ROI plan requires, or, when the tile's packets
//! cannot all be skipped by PLT length, when the packet precedes the last
//! required packet in progression order. Every other precinct is built empty,
//! which keeps precinct indices stable. In progressions that read all of a
//! precinct's layers in a row, a precinct that is only parsed is built empty
//! too, and borrows a shared scratch precinct while its packets are parsed.

use alloc::vec::Vec;

use super::allocation::{checked_include_elements, code_block_grid};
use super::{Decomposition, Precinct, SubBand, SubBandType};
use crate::error::{DecodingError, Result, ValidationError};
use crate::j2c::codestream::{Header, ProgressionOrder};
use crate::j2c::decode::{
    DecodeAllocationBudget, DecompositionStorage, OutputRegion, TileDecompositions,
};
use crate::j2c::progression::{progression_iterator, ProgressionData};
use crate::j2c::rect::IntRect;
use crate::j2c::roi::RoiPlan;
use crate::j2c::tag_tree::TagNode;
use crate::j2c::tile::Tile;
use crate::DEFAULT_MAX_DECODE_BYTES;

/// How region decode uses one precinct of one component and resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum PrecinctUse {
    /// Never parsed: `build` gives it no precinct record.
    Unused,
    /// Parsed only to reach later packets; its code-blocks are not decoded.
    /// When the selection parses these in scratch, they are built like
    /// unused precincts and borrow a shared scratch precinct while parsed.
    Parsed,
    /// Holds a code-block the ROI plan requires.
    Required,
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
    /// Precinct numbers for runs that omit records, in increasing order.
    /// Dense runs use direct indexing and contribute no entries.
    records: Vec<u32>,
    /// Start of each run in `records`, empty when all runs stay dense.
    /// An empty range is either dense or entirely unused; `uses` distinguishes them.
    record_runs: Vec<usize>,
    /// Whether every packet of the tile can be skipped by its PLT length.
    skip_by_length: bool,
    /// Whether parsed-only precincts share scratch state instead of their
    /// own: the progression reads all of a precinct's layers in a row, so
    /// its state is dead once its last packet is parsed.
    parsed_in_scratch: bool,
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
    /// Full RPCL header scan, using the same bounded scratch precincts as
    /// parsed-only ROI packets. No block gets a layer/segment graph.
    #[cfg(feature = "std")]
    pub(crate) fn for_packet_scan(tile: &Tile<'_>, live_bytes: usize) -> Result<Self> {
        let counts = count(tile)?;
        let mut budget = DecodeAllocationBudget::from_live_bytes(live_bytes)?;
        let mut uses = Vec::new();
        let mut runs = Vec::new();
        let mut component_runs = Vec::new();
        budget.resize_new(&mut uses, counts.run_precincts, PrecinctUse::Parsed)?;
        budget.reserve_new(&mut runs, counts.runs)?;
        budget.reserve_new(&mut component_runs, counts.components)?;
        let mut precincts = 0;
        for component in tile.component_tiles() {
            component_runs.push(runs.len());
            for resolution in component.resolution_tiles() {
                runs.push(precincts);
                precincts += usize::try_from(resolution.num_precincts())
                    .map_err(|_| ValidationError::ImageTooLarge)?;
            }
        }
        let mut selection = Self {
            uses,
            runs,
            component_runs,
            records: Vec::new(),
            record_runs: Vec::new(),
            skip_by_length: false,
            parsed_in_scratch: true,
            required_packets: counts
                .run_precincts
                .checked_mul(usize::from(tile.num_layers))
                .ok_or(ValidationError::ImageTooLarge)?,
            owned_bytes: 0,
            scanned_packets: 0,
            released_storage_bytes: 0,
        };
        index_records(&mut selection, &mut budget)?;
        Ok(selection)
    }

    /// This selection for storage that keeps every precinct's record: a
    /// packet's precinct is found at its own index, and a parsed-only
    /// precinct is parsed in its own record.
    fn with_every_record(mut self) -> Result<Self> {
        self.records = Vec::new();
        self.record_runs = Vec::new();
        self.parsed_in_scratch = false;
        let mut bytes = 0;
        checked_include_elements::<PrecinctUse>(&mut bytes, self.uses.capacity())?;
        checked_include_elements::<usize>(&mut bytes, self.runs.capacity())?;
        checked_include_elements::<usize>(&mut bytes, self.component_runs.capacity())?;
        self.owned_bytes = bytes;
        Ok(self)
    }

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

    /// Whether parsed-only precincts are built without code-blocks and
    /// parsed in a shared scratch precinct.
    pub(crate) const fn parsed_in_scratch(&self) -> bool {
        self.parsed_in_scratch
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

    /// Where the packet's precinct lies among its sub-bands' precinct
    /// records, or `None` when it has none.
    pub(crate) fn record_offset(&self, packet: &ProgressionData) -> Option<usize> {
        let precinct = usize::try_from(packet.precinct).ok()?;
        if self.record_runs.is_empty() {
            return Some(precinct);
        }
        let run = self
            .component_runs
            .get(usize::from(packet.component))?
            .checked_add(usize::from(packet.resolution))?;
        let start = *self.record_runs.get(run)?;
        let end = self
            .record_runs
            .get(run + 1)
            .copied()
            .unwrap_or(self.records.len());
        if start == end {
            return self
                .packet_use(packet)
                .filter(|usage| *usage != PrecinctUse::Unused)
                .map(|_| precinct);
        }
        let number = u32::try_from(packet.precinct).ok()?;
        self.records.get(start..end)?.binary_search(&number).ok()
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

/// The selection a decode that builds every precinct of the tile parses it
/// with: below full resolution, the one a region decode of the whole image
/// parses it with, so both read the same packets; otherwise `None`, every
/// packet being read. `live_bytes` are live beside `storage`, whose retained
/// capacity the build reuses.
pub(crate) fn select_whole_image_precincts(
    tile: &Tile<'_>,
    header: &Header<'_>,
    storage: &DecompositionStorage<'_>,
    live_bytes: usize,
) -> Result<Option<PrecinctSelection>> {
    if header.skipped_resolution_levels == 0 {
        return Ok(None);
    }
    // The build stores a record and two tag-tree roots for every precinct
    // beside the selection, in whatever capacity it reuses: a tile for which
    // that alone passes the cap fails here, before its precincts are walked.
    let counts = count(tile)?;
    preflight_structure(&counts, live_bytes, selection_owned_bytes(&counts)?, false)?;
    let retained_baseline_bytes = live_bytes
        .checked_add(storage.retained_capacity_bytes()?)
        .ok_or(ValidationError::ImageTooLarge)?;
    whole_image_selection(tile, header, &counts, retained_baseline_bytes)
}

/// The selection a region decode of the whole image parses a tile with, for a
/// decode that builds every precinct of the tile: the same packets are read,
/// skipped by their PLT lengths, or left after the last one the output needs,
/// so both decodes accept and reject the same codestreams. The region
/// decode's skeleton and ROI plan live only while the selection is made, in
/// their own storage beside the `retained_baseline_bytes` (the tile storage's
/// retained capacity included), which the later build reuses; every
/// precinct keeps its record.
///
/// Unlike [`select_region_precincts`], this does not reserve room for the
/// build's precinct records and tag-tree roots up front: the build makes
/// them later, in the retained capacity, after its own preflight. Only what
/// is allocated here is charged.
fn whole_image_selection(
    tile: &Tile<'_>,
    header: &Header<'_>,
    counts: &SelectionCounts,
    retained_baseline_bytes: usize,
) -> Result<Option<PrecinctSelection>> {
    let whole_image = OutputRegion {
        x: 0,
        y: 0,
        width: header.size_data.image_width(),
        height: header.size_data.image_height(),
    };
    let mut budget = DecodeAllocationBudget::from_live_bytes(retained_baseline_bytes)?;
    let mut skeleton = DecompositionStorage::default();
    build_skeleton(tile, counts, &mut skeleton, &mut budget)?;
    RoiPlan::build(tile, header, &skeleton, whole_image, &mut budget)?
        .map(|plan| select(tile, &skeleton, &plan, counts, &mut budget, 0)?.with_every_record())
        .transpose()
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
    preflight_structure(counts, retained_baseline_bytes, owned_bytes, true)
}

/// Reject a tile whose selection (`owned_bytes`) and build structure do not
/// fit the cap beside `retained_baseline_bytes`: the skeleton, every sub-band
/// precinct with two root tag-tree nodes, and, with `roi_windows`, the ROI
/// windows a region build adds.
fn preflight_structure(
    counts: &SelectionCounts,
    retained_baseline_bytes: usize,
    owned_bytes: usize,
    roi_windows: bool,
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
    if roi_windows {
        checked_include_elements::<Option<IntRect>>(
            &mut structure,
            counts
                .sub_bands
                .checked_add(counts.decompositions)
                .and_then(|count| count.checked_add(counts.components))
                .ok_or(ValidationError::ImageTooLarge)?,
        )?;
    }
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

#[expect(
    clippy::too_many_lines,
    reason = "one pass marks every component's and resolution's precincts in the order the use table indexes them"
)]
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
            // Each sub-band the ROI plan reads, in order, with its window.
            let windows: [Option<(IntRect, IntRect)>; 3] = core::array::from_fn(|slot| {
                let sub_band_type = sub_band_types.get(slot)?;
                let window = plan.sub_band_window(sub_band_indices[slot])?;
                Some((window, resolution_tile.sub_band_rect(*sub_band_type)))
            });
            let precincts = resolution_tile
                .precinct_grid()
                .ok_or(DecodingError::InvalidPrecinct)?;
            let precinct_count = usize::try_from(resolution_tile.num_precincts())
                .map_err(|_| ValidationError::ImageTooLarge)?;
            if windows.iter().all(Option::is_none) {
                // The plan reads none of this resolution's sub-bands (a
                // reduced decode's finer levels): every precinct is unused,
                // and only the index check of its rectangle remains.
                let rects = precincts.rect_count();
                for precinct_idx in 0..precinct_count {
                    if precinct_idx >= rects {
                        return Err(DecodingError::InvalidPrecinct.into());
                    }
                    push_within_capacity(&mut uses, PrecinctUse::Unused)?;
                }
                continue;
            }
            for precinct_idx in 0..precinct_count {
                let precinct_rect = precincts
                    .rect_at(precinct_idx)
                    .ok_or(DecodingError::InvalidPrecinct)?;
                let mut required = false;
                for (window, rect) in windows.iter().flatten().copied() {
                    let grid = code_block_grid(&resolution_tile, rect, precinct_rect)?;
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

    let skip_by_length = packets_skippable_by_length(tile, counts, layers)?;
    let mut selection = PrecinctSelection {
        uses,
        runs,
        component_runs,
        records: Vec::new(),
        record_runs: Vec::new(),
        skip_by_length,
        parsed_in_scratch: !skip_by_length && reads_precinct_layers_in_a_row(tile),
        required_packets,
        // Counted once the records are indexed.
        owned_bytes: 0,
        scanned_packets: 0,
        released_storage_bytes,
    };
    if !selection.skip_by_length {
        mark_precincts_parsed_before_last_required(tile, &mut selection, budget.live_bytes())?;
    }
    index_records(&mut selection, budget)?;
    Ok(selection)
}

/// List the precincts that get precinct records, every one but the unused,
/// for the parser to find a packet's precinct among them, and count the
/// bytes the selection owns.
fn index_records(
    selection: &mut PrecinctSelection,
    budget: &mut DecodeAllocationBudget,
) -> Result<()> {
    let mut count = 0;
    let mut sparse = false;
    for run in 0..selection.runs.len() {
        let uses = selection.run(run).ok_or(DecodingError::InvalidPrecinct)?;
        if uses.contains(&PrecinctUse::Unused) {
            sparse = true;
            count += uses
                .iter()
                .filter(|usage| **usage != PrecinctUse::Unused)
                .count();
        }
    }
    // A dense selection owns exactly the same vectors as before compaction.
    // In a mixed selection only runs that actually omit records get an index.
    if sparse {
        budget.reserve_new(&mut selection.records, count)?;
        budget.reserve_new(&mut selection.record_runs, selection.runs.len())?;
        for (run, &start) in selection.runs.iter().enumerate() {
            push_within_capacity(&mut selection.record_runs, selection.records.len())?;
            let end = selection
                .runs
                .get(run + 1)
                .copied()
                .unwrap_or(selection.uses.len());
            let uses = selection
                .uses
                .get(start..end)
                .ok_or(DecodingError::InvalidPrecinct)?;
            if !uses.contains(&PrecinctUse::Unused) {
                continue;
            }
            for (number, precinct_use) in uses.iter().enumerate() {
                if *precinct_use != PrecinctUse::Unused {
                    let number =
                        u32::try_from(number).map_err(|_| ValidationError::ImageTooLarge)?;
                    push_within_capacity(&mut selection.records, number)?;
                }
            }
        }
    }
    let mut bytes = 0;
    checked_include_elements::<PrecinctUse>(&mut bytes, selection.uses.capacity())?;
    checked_include_elements::<usize>(&mut bytes, selection.runs.capacity())?;
    checked_include_elements::<usize>(&mut bytes, selection.component_runs.capacity())?;
    checked_include_elements::<u32>(&mut bytes, selection.records.capacity())?;
    checked_include_elements::<usize>(&mut bytes, selection.record_runs.capacity())?;
    selection.owned_bytes = bytes;
    Ok(())
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

/// Whether the progression reads every layer of a precinct before any packet
/// of another precinct of the same component and resolution, and never comes
/// back to it: RPCL, PCRL and CPRL, where the layer is the innermost index.
/// Progression changes can split a precinct's layers, so they are excluded.
fn reads_precinct_layers_in_a_row(tile: &Tile<'_>) -> bool {
    tile.progression_changes.is_empty()
        && matches!(
            tile.progression_order,
            ProgressionOrder::ResolutionPositionComponentLayer
                | ProgressionOrder::PositionComponentResolutionLayer
                | ProgressionOrder::ComponentPositionResolutionLayer
        )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{encode, DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image};

    #[test]
    fn mixed_runs_index_only_precincts_in_sparse_runs() {
        use PrecinctUse::{Parsed, Required, Unused};
        let mut selection = PrecinctSelection {
            uses: vec![Required, Parsed, Required, Unused, Required, Unused, Unused],
            runs: vec![0, 3, 5],
            component_runs: vec![0, 2],
            records: Vec::new(),
            record_runs: Vec::new(),
            skip_by_length: false,
            parsed_in_scratch: false,
            required_packets: 3,
            owned_bytes: 0,
            scanned_packets: 0,
            released_storage_bytes: 0,
        };
        // Only the middle run needs one index entry. Dense and wholly unused
        // runs must fit without reserving entries for their precincts.
        let cap = core::mem::size_of::<u32>() + 3 * core::mem::size_of::<usize>();
        let mut budget = DecodeAllocationBudget::from_live_bytes_with_cap(0, cap).unwrap();
        index_records(&mut selection, &mut budget).expect("only the sparse run is indexed");
        for (component, resolution, expected) in [
            (0, 0, vec![Some(0), Some(1), Some(2)]),
            (0, 1, vec![None, Some(0)]),
            (1, 0, vec![None, None]),
        ] {
            for (precinct, offset) in expected.into_iter().enumerate() {
                let packet = ProgressionData {
                    component,
                    resolution,
                    precinct: precinct as u64,
                    layer_num: 0,
                };
                assert_eq!(selection.record_offset(&packet), offset, "{packet:?}");
            }
        }
    }

    /// A whole-image selection made beside a tile storage that keeps its
    /// capacity charges what it allocates, not the precinct records and tag
    /// trees the later build reuses from that capacity: it fits a baseline at
    /// which a region decode's selection, which reserves room for them,
    /// rejects the tile.
    #[test]
    fn whole_image_selection_charges_only_its_own_allocations() {
        const WIDTH: u32 = 120;
        const HEIGHT: u32 = 64;
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            precinct_exponents: vec![(2, 2)],
            ..EncodeOptions::default()
        };
        let bytes = encode(
            &vec![128; (WIDTH * HEIGHT) as usize],
            WIDTH,
            HEIGHT,
            1,
            8,
            false,
            &options,
        )
        .expect("encode fixture");
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        let tiles = crate::j2c::tile::parse(
            &mut crate::reader::BitReader::new(image.codestream),
            &image.header,
            0,
        )
        .expect("parse tile");
        let tile = &tiles[0];
        let counts = count(tile).unwrap();
        let owned = selection_owned_bytes(&counts).unwrap();
        let (mut accepted, mut rejected) = (0, DEFAULT_MAX_DECODE_BYTES);
        while rejected - accepted > 1 {
            let baseline = accepted + (rejected - accepted) / 2;
            if preflight_unavoidable_minimum(&counts, baseline, owned).is_ok() {
                accepted = baseline;
            } else {
                rejected = baseline;
            }
        }
        let whole_image = OutputRegion::from_tuple((0, 0, WIDTH, HEIGHT));
        assert!(select_region_precincts(
            tile,
            &image.header,
            &mut DecompositionStorage::default(),
            whole_image,
            rejected,
        )
        .is_err());
        let selection = whole_image_selection(tile, &image.header, &counts, rejected)
            .expect("the whole-image selection fits beside the retained storage")
            .expect("a selection");
        // Every 4 x 4 precinct of the 120 x 64 tile, one layer.
        assert_eq!(selection.required_packets(), 30 * 16);
    }

    /// A one-tile gray codestream header covering `[x0, x0 + width) x
    /// [y0, y0 + height)` with `precincts.len() - 1` decomposition levels and
    /// `precincts` as its COD precinct bytes, followed by an empty tile-part.
    fn tile_geometry_codestream(
        (x0, y0): (u32, u32),
        (width, height): (u32, u32),
        precincts: &[u8],
    ) -> Vec<u8> {
        let mut bytes = vec![0xff, 0x4f, 0xff, 0x51, 0, 41, 0, 0];
        for value in [x0 + width, y0 + height, x0, y0, width, height, x0, y0] {
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        bytes.extend_from_slice(&[0, 1, 7, 1, 1]);
        let levels = u8::try_from(precincts.len() - 1).unwrap();
        // COD: precincts given; LRCP, one layer; 4 x 4 blocks, 5/3.
        let mut cod = vec![1, 0, 0, 1, 0, levels, 0, 0, 0, 1];
        cod.extend_from_slice(precincts);
        bytes.extend_from_slice(&[0xff, 0x52]);
        bytes.extend_from_slice(&u16::try_from(cod.len() + 2).unwrap().to_be_bytes());
        bytes.extend_from_slice(&cod);
        // QCD: reversible, two guard bits.
        let sub_bands = 3 * usize::from(levels) + 1;
        bytes.extend_from_slice(&[0xff, 0x5c]);
        bytes.extend_from_slice(&u16::try_from(sub_bands + 3).unwrap().to_be_bytes());
        bytes.push(0x40);
        bytes.extend(core::iter::repeat_n(0x48, sub_bands));
        bytes.extend_from_slice(&[0xff, 0x90, 0, 10, 0, 0, 0, 0, 0, 15, 0, 1]);
        bytes.extend_from_slice(&[0xff, 0x93, 0, 0xff, 0xd9]);
        bytes
    }

    fn with_tile_and_header<R>(
        codestream: &[u8],
        settings: &DecodeSettings,
        check: impl FnOnce(&Tile<'_>, &Header<'_>) -> R,
    ) -> R {
        let image = Image::new(codestream, settings).expect("parse header");
        let tiles = crate::j2c::tile::parse(
            &mut crate::reader::BitReader::new(image.codestream),
            &image.header,
            0,
        )
        .expect("parse tile");
        check(&tiles[0], &image.header)
    }

    /// A tile whose build could not hold one record and two tag-tree roots per
    /// precinct within the cap is rejected before its whole-image selection
    /// is made: 16,383 x 16,383 samples, one decomposition level, one-sample
    /// precincts in the LL sub-band and two-sample ones above, 2^28 sub-band
    /// precincts, at half resolution, with nothing else live.
    #[test]
    fn whole_image_selection_rejects_a_tile_its_build_cannot_hold_first() {
        let codestream = tile_geometry_codestream((0, 0), (16_383, 16_383), &[0x00, 0x11]);
        let half = DecodeSettings {
            target_resolution: Some((8_192, 8_192)),
            ..DecodeSettings::default()
        };
        with_tile_and_header(&codestream, &half, |tile, header| {
            assert_eq!(header.skipped_resolution_levels, 1);
            assert_eq!(count(tile).unwrap().sub_band_precincts, 4 << 26);
            assert!(select_whole_image_precincts(
                tile,
                header,
                &DecompositionStorage::default(),
                0,
            )
            .is_err());
        });
    }

    #[test]
    fn dense_region_selection_keeps_the_existing_budget_boundary() {
        // A scaled-down 7680x4096 LRCP example with 4x4 precincts and blocks.
        // Empty packets need no segment storage. The old selection fits exactly;
        // a redundant record index must not turn that acceptance into rejection.
        const WIDTH: u32 = 120;
        const HEIGHT: u32 = 64;
        for use_ht_block_coding in [false, true] {
            for (order, write_plt, roi) in [
                (EncodeProgressionOrder::Lrcp, false, (0, 0, WIDTH, HEIGHT)),
                (EncodeProgressionOrder::Lrcp, true, (0, 0, WIDTH, HEIGHT)),
                (
                    EncodeProgressionOrder::Lrcp,
                    false,
                    (WIDTH - 1, HEIGHT - 1, 1, 1),
                ),
                (
                    EncodeProgressionOrder::Rpcl,
                    false,
                    (WIDTH - 1, HEIGHT - 1, 1, 1),
                ),
            ] {
                let options = EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 0,
                    code_block_width_exp: 0,
                    code_block_height_exp: 0,
                    precinct_exponents: vec![(2, 2)],
                    progression_order: order,
                    write_plt,
                    use_ht_block_coding,
                    ..EncodeOptions::default()
                };
                let bytes = encode(
                    &vec![128; (WIDTH * HEIGHT) as usize],
                    WIDTH,
                    HEIGHT,
                    1,
                    8,
                    false,
                    &options,
                )
                .expect("encode fixture");
                let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
                let tiles = crate::j2c::tile::parse(
                    &mut crate::reader::BitReader::new(image.codestream),
                    &image.header,
                    0,
                )
                .expect("parse tile");
                let tile = &tiles[0];
                let counts = count(tile).unwrap();
                let mut storage = DecompositionStorage::default();
                let mut budget = DecodeAllocationBudget::from_live_bytes(0).unwrap();
                build_skeleton(tile, &counts, &mut storage, &mut budget).unwrap();
                let plan = RoiPlan::build(
                    tile,
                    &image.header,
                    &storage,
                    OutputRegion::from_tuple(roi),
                    &mut budget,
                )
                .unwrap()
                .expect("region plan");
                let live = budget.live_bytes();
                let cap = live + selection_owned_bytes(&counts).unwrap();
                let mut exact =
                    DecodeAllocationBudget::from_live_bytes_with_cap(live, cap).unwrap();
                let selection = select(tile, &storage, &plan, &counts, &mut exact, 0)
                    .expect("dense selection fits the existing cap");
                assert_eq!(exact.live_bytes(), cap);
                assert_eq!(selection.owned_bytes(), cap - live);
                for packet in progression_iterator(tile, 0).unwrap() {
                    assert_eq!(
                        selection.record_offset(&packet),
                        usize::try_from(packet.precinct).ok()
                    );
                }
                let mut short =
                    DecodeAllocationBudget::from_live_bytes_with_cap(live, cap - 1).unwrap();
                assert!(matches!(
                    select(tile, &storage, &plan, &counts, &mut short, 0),
                    Err(crate::DecodeError::Validation(
                        ValidationError::ImageTooLarge
                    ))
                ));
                assert_eq!(
                    image.decode_native_region(roi).unwrap().data,
                    vec![128; (roi.2 * roi.3) as usize]
                );
            }
        }
    }
}
