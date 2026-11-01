// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned tile decode state and its allocation accounting.

use alloc::vec::Vec;
use core::mem::size_of;
use core::ops::Range;

#[cfg(feature = "parallel")]
use super::workspace;
use super::{DecodeDebugCounters, OutputRegion};
use crate::error::{DecodeError, Result, ValidationError};
use crate::j2c::bitplane::{BitPlaneDecodeBuffers, BitPlaneDecodeContext};
use crate::j2c::build::{self, CodeBlock, Decomposition, Layer, Precinct, Segment, SubBand};
use crate::j2c::ht_block_decode::HtBlockDecodeContext;
use crate::j2c::idwt::{self, IDWTOutput};
use crate::j2c::mct;
use crate::j2c::roi::RoiPlan;
use crate::j2c::tag_tree::TagNode;
use crate::j2c::ComponentData;
#[cfg(feature = "parallel")]
use crate::J2kCodeBlockDecodeWorkspace;

/// All decompositions for a single tile.
#[derive(Clone)]
pub(crate) struct TileDecompositions {
    pub(crate) first_ll_sub_band: usize,
    pub(crate) decompositions: Range<usize>,
}

impl TileDecompositions {
    pub(crate) fn sub_band_iter(
        &self,
        resolution: u8,
        decompositions: &[Decomposition],
    ) -> SubBandIter {
        let indices = if resolution == 0 {
            [
                self.first_ll_sub_band,
                self.first_ll_sub_band,
                self.first_ll_sub_band,
            ]
        } else {
            decompositions[self.decompositions.clone()][resolution as usize - 1].sub_bands
        };

        SubBandIter {
            next_idx: 0,
            indices,
            resolution,
        }
    }
}

#[derive(Clone)]
pub(crate) struct SubBandIter {
    resolution: u8,
    next_idx: usize,
    indices: [usize; 3],
}

impl Iterator for SubBandIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let value = if self.resolution == 0 {
            if self.next_idx > 0 {
                None
            } else {
                Some(self.indices[0])
            }
        } else if self.next_idx >= self.indices.len() {
            None
        } else {
            Some(self.indices[self.next_idx])
        };

        self.next_idx += 1;

        value
    }
}

/// Owned decomposition workspace for one active tile.
///
/// Within an image, reset retains capacities and the next build counts them in
/// its live baseline. Crossing to a new image drops every retained allocation.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the integer, stale-plane, in-channel last level, row-streamed IDWT and banded Tier-1 switches are independent"
)]
pub(crate) struct DecompositionStorage<'a> {
    pub(crate) segments: Vec<Segment<'a>>,
    pub(crate) layers: Vec<Layer>,
    pub(crate) code_blocks: Vec<CodeBlock>,
    pub(crate) precincts: Vec<Precinct>,
    pub(crate) tag_tree_nodes: Vec<TagNode>,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) coefficients_i64: Vec<i64>,
    pub(crate) sub_bands: Vec<SubBand>,
    pub(crate) decompositions: Vec<Decomposition>,
    pub(crate) tile_decompositions: Vec<TileDecompositions>,
    pub(crate) roi_plan: Option<RoiPlan>,
    pub(crate) exact_integer_decode: bool,
    /// How the built sub-bands' coefficient ranges share `coefficients`.
    pub(crate) coefficient_layout: build::CoefficientLayout,
    /// Whether `coefficients` still holds the previous component's samples.
    /// Tier-1 must zero what it does not decode before the IDWT reads it.
    pub(crate) stale_coefficients: bool,
    /// Whether the region IDWT writes each component's last level into the
    /// padding of its `channel_data` plane, for `store` to compact in place.
    pub(crate) last_idwt_level_in_channels: bool,
    /// Whether the region IDWT streams this tile's levels a row at a time.
    pub(crate) line_region_idwt: bool,
    /// Whether Tier-1 decodes each component in bands of code-block rows as
    /// the streamed IDWT reaches them. Each sub-band's `coefficients` then
    /// hold one band of its stored rectangle's rows.
    pub(crate) banded_tier1: bool,
    /// Whether Tier-1 decodes each component's bands ahead of the streamed
    /// IDWT on the worker pool (see [`parallel_bands`]). Each sub-band's
    /// `coefficients` then hold a ring of bands.
    pub(crate) parallel_bands: bool,
    /// Whether the band rings of `parallel_bands` hold one band more at
    /// every level, for a classic tile that streams whole rows at full
    /// resolution from a reader per component (see
    /// [`parallel_bands::deep_rings`] and [`parallel_bands::reads_whole_rows`]).
    pub(crate) deep_band_rings: bool,
    /// Whether the finest level's band rings of `parallel_bands` hold one
    /// band fewer, for a classic tile that streams full-width rows from a
    /// reader per component without deep rings (see
    /// [`parallel_bands::shallow_rings`]).
    pub(crate) shallow_band_rings: bool,
    /// Whether the band rings of `parallel_bands` hold two bands at every
    /// level and their bands stay whole, for a decode whose pool other images
    /// share, on a pool too small for deep rings (see
    /// [`parallel_bands::shared_rings`]).
    pub(crate) shared_pool_band_rings: bool,
    /// Whether a tile streaming its rows from `parallel_bands` reconstructs
    /// them through one reader whatever its pool's workers (see
    /// [`parallel_bands::one_reader`]), which holds one short batch of rows.
    pub(crate) one_synthesis_reader: bool,
    /// Whether every component's IDWT rings are live at once, because the
    /// tile streams its rows to a sink, finishing them in the rings.
    pub(crate) lockstep_lines: bool,
    /// The f32 samples of IDWT workspace the ROI plan charged for this tile.
    pub(crate) roi_idwt_charge_elements: usize,
    /// While a band is decoded, the first row of its code-blocks: Tier-1
    /// decodes only the code-blocks that start there, into the rows the
    /// sub-band's `storage_rect` then names.
    pub(crate) tier1_band: Option<u32>,
    /// Planned non-segment live bytes for the active tile and future decode scratch.
    pub(crate) structural_workspace_bytes: usize,
    /// Set when region decode deferred coefficient storage until the ROI
    /// plan is known. Holds the whole-plane IDWT workspace the plan left out,
    /// which must be charged if decode falls back to the whole tile.
    pub(crate) deferred_idwt_workspace_elements: Option<usize>,
    /// Allocation/cap failure raised from the option-based packet parser.
    pub(crate) packet_workspace_error: Option<DecodeError>,
    /// The scratch precincts parsed-only precincts borrow during parsing.
    pub(crate) parse_scratch: Option<build::ParseScratch>,
}

impl DecompositionStorage<'_> {
    pub(crate) fn reset_for_next_tile(&mut self) {
        self.segments.clear();
        self.layers.clear();
        self.code_blocks.clear();
        self.precincts.clear();
        self.tag_tree_nodes.clear();
        self.coefficients.clear();
        self.coefficients_i64.clear();
        self.sub_bands.clear();
        self.decompositions.clear();
        self.tile_decompositions.clear();
        self.roi_plan = None;
        self.exact_integer_decode = false;
        self.coefficient_layout = build::CoefficientLayout::Tile;
        self.stale_coefficients = false;
        self.last_idwt_level_in_channels = false;
        self.line_region_idwt = false;
        self.banded_tier1 = false;
        self.parallel_bands = false;
        self.deep_band_rings = false;
        self.shallow_band_rings = false;
        self.shared_pool_band_rings = false;
        self.one_synthesis_reader = false;
        self.lockstep_lines = false;
        self.roi_idwt_charge_elements = 0;
        self.tier1_band = None;
        self.structural_workspace_bytes = 0;
        self.deferred_idwt_workspace_elements = None;
        self.packet_workspace_error = None;
        self.parse_scratch = None;
    }

    pub(crate) fn release_all_allocations(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn retained_capacity_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        include_capacity::<Segment<'_>>(&mut bytes, self.segments.capacity())?;
        include_capacity::<Layer>(&mut bytes, self.layers.capacity())?;
        include_capacity::<CodeBlock>(&mut bytes, self.code_blocks.capacity())?;
        include_capacity::<Precinct>(&mut bytes, self.precincts.capacity())?;
        include_capacity::<TagNode>(&mut bytes, self.tag_tree_nodes.capacity())?;
        include_capacity::<f32>(&mut bytes, self.coefficients.capacity())?;
        include_capacity::<i64>(&mut bytes, self.coefficients_i64.capacity())?;
        include_capacity::<SubBand>(&mut bytes, self.sub_bands.capacity())?;
        include_capacity::<Decomposition>(&mut bytes, self.decompositions.capacity())?;
        include_capacity::<TileDecompositions>(&mut bytes, self.tile_decompositions.capacity())?;
        Ok(bytes)
    }
}

fn include_capacity<T>(bytes: &mut usize, capacity: usize) -> Result<()> {
    let additional = capacity
        .checked_mul(size_of::<T>())
        .ok_or(ValidationError::ImageTooLarge)?;
    *bytes = bytes
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge)?;
    if *bytes > crate::DEFAULT_MAX_DECODE_BYTES {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

/// A reusable context used during the decoding of a single tile.
///
/// Some of the fields are temporary in nature and reset after moving on to the
/// next tile, some contain global state.
#[derive(Default)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the rounding, deferred-MCT and row-streamed IDWT switches are independent, and test builds add switches that pin the per-sub-band HT, whole-code-block storage, separate last-level IDWT and staged RGB8 pack paths"
)]
pub(crate) struct TileDecodeContext {
    /// A reusable buffer for the IDWT output.
    pub(crate) idwt_output: IDWTOutput,
    /// A scratch buffer used during IDWT.
    pub(crate) idwt_scratch_buffer: Vec<f32>,
    /// Per-worker band buffers for the row-banded IDWT.
    pub(crate) idwt_band_scratch: Vec<f32>,
    /// A scratch buffer used during exact reversible integer IDWT.
    pub(crate) idwt_scratch_buffer_i64: Vec<i64>,
    /// The rings of rows the row-streamed region IDWT keeps for each level.
    pub(crate) idwt_line_rings: Vec<f32>,
    /// A reusable context for decoding code blocks.
    pub(crate) bit_plane_decode_context: BitPlaneDecodeContext,
    /// Reusable buffers for decoding bitplanes.
    pub(crate) bit_plane_decode_buffers: BitPlaneDecodeBuffers,
    /// A reusable context for decoding HTJ2K code blocks.
    pub(crate) ht_block_decode_context: HtBlockDecodeContext,
    /// Scratch an HTJ2K code-block that the sub-band stores only part of
    /// decodes into before its stored part is copied out, sized with the
    /// serial Tier-1 workspace.
    pub(crate) ht_clipped_block_scratch: Vec<f32>,
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) ht_task_workspaces: Vec<workspace::HtTaskWorkspace>,
    /// One classic workspace per worker of the tile-wide route.
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) classic_task_workspaces:
        Vec<workspace::TaskWorkspace<J2kCodeBlockDecodeWorkspace>>,
    #[cfg(feature = "parallel")]
    pub(in crate::j2c::decode) parallel_coefficients: Vec<f32>,
    /// Test-only switch that keeps tiles off the tile-wide route, on the
    /// per-sub-band path, so its slab and bank behavior stays covered.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) force_per_sub_band: bool,
    /// One-shot fault injection after a tile route has written coefficients.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) tile_route_post_decode: Option<fn(&mut DecompositionStorage<'_>) -> Result<()>>,
    /// Test-only switch that stores whole code-blocks in region decode, as a
    /// hooked decode does, so clipped storage can be compared against it.
    #[cfg(test)]
    pub(crate) store_whole_code_blocks: bool,
    /// Test-only switch that decodes Tier-1 over whole windows before the
    /// streamed IDWT, as a parallel decode does, so banded Tier-1 can be
    /// compared against it.
    #[cfg(test)]
    pub(crate) whole_window_tier1: bool,
    /// Test-only: the first pool worker to start a band abandons it, so the
    /// band is left running with no worker to finish it.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) strand_pool_band: bool,
    /// Test-only: each pool band splits into this many parts, as far as its
    /// blocks allow, whatever the pool's size.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) band_parts: Option<usize>,
    /// Test-only: the average work per band below which a component's
    /// bands decode over whole windows instead of on the pool; zero, so
    /// unit tests' tiny fixtures take the pool, unless a test sets it.
    #[cfg(all(test, feature = "parallel"))]
    pub(crate) band_work_cutoff: usize,
    /// Test-only capacity failure when building the optional band row index.
    #[cfg(test)]
    pub(crate) reject_band_index: bool,
    /// Test-only switch that keeps the region IDWT's last level in its own
    /// window, as a hooked decode does, so writing it into `channel_data`
    /// can be compared against it.
    #[cfg(test)]
    pub(crate) separate_last_idwt_level: bool,
    /// Test-only switch that keeps an RGB8 decode's last IDWT level and its
    /// pack apart, so the fused pass can be compared against them.
    #[cfg(test)]
    pub(crate) staged_rgb8_pack: bool,
    /// Test-only switch that lets an `Auto` decode of a small image run
    /// serially, as every other build does. Tests leave it off so their
    /// small fixtures still exercise the parallel routes.
    #[cfg(test)]
    pub(crate) serial_small_decodes: bool,
    /// Test-only switch that lets a parallel `decode_into` called from outside
    /// the pool move onto it, as every other build does. Tests leave it off so
    /// their per-thread counters see the decode.
    #[cfg(test)]
    pub(crate) enter_pool: bool,
    /// Test-only decode budget for the component planes, in place of the
    /// default: `reset` charges them against it, and a region decode
    /// compares them with it when it chooses whether its rows stream from
    /// serial bands (see `choose_region_idwt`).
    #[cfg(test)]
    pub(crate) planes_budget: Option<usize>,
    /// The raw, decoded samples for each channel.
    pub(crate) channel_data: Vec<ComponentData>,
    /// Optional output window for region-local decode storage.
    pub(crate) output_region: Option<OutputRegion>,
    /// Whether irreversible samples are being decoded for an integer output.
    pub(crate) round_irreversible_output: bool,
    /// Whether the inverse MCT is left to the caller's RGB8 pack, set
    /// around a decode like `round_irreversible_output`.
    pub(crate) defer_mct: bool,
    /// Whether this context's decodes share their pool with other images
    /// decoding at the same time (see `DecoderContext::set_shared_pool`).
    pub(crate) shared_pool: bool,
    /// The inverse MCT this decode left to the caller.
    pub(crate) deferred_mct: Option<mct::DeferredMct>,
    /// The last IDWT level this decode leaves to the RGB8 pack with the
    /// deferred MCT: the IDWT stops one level short, at the level's LL band,
    /// and each plane receives the level's other sub-bands instead of samples.
    pub(crate) fused_last_level: Option<idwt::LastLevel>,
    /// Samples, over all components, that `store` must write because the
    /// planes were reused without zeroing; zero when the planes were zeroed.
    pub(crate) unzeroed_plane_samples: usize,
    /// Component-plane samples `store` wrote during the current decode.
    pub(crate) stored_plane_samples: usize,
    /// The sample count of the last component's plane when `reset` left it
    /// unallocated for a fused RGB8 decode to fill with its coefficients.
    pub(crate) deferred_plane: Option<usize>,
    /// Set for a region decode whose component planes reserve room for this
    /// padded size, so a tile covering the whole region can write its last
    /// IDWT level into them.
    pub(crate) last_idwt_level_padding: Option<(u32, u32)>,
    /// Whether the current call may use the row-streamed region IDWT. Set
    /// around a decode by the outputs that allow it, like `output_region`.
    pub(crate) line_region_idwt_permitted: bool,
    /// Whether this decode streams the region IDWT a row at a time.
    pub(crate) line_region_idwt: bool,
    /// Whether this decode hands the region's samples to a row sink as the
    /// streamed IDWT finishes them, with no component planes.
    pub(crate) stream_rows: bool,
    /// Samples, over all components, handed to the row sink this decode.
    pub(crate) streamed_samples: usize,
    /// Debug counters for tests and ROI instrumentation.
    pub(crate) debug_counters: DecodeDebugCounters,
}

impl TileDecodeContext {
    pub(super) fn release_all_allocations(&mut self) {
        let output_region = self.output_region;
        let round_irreversible_output = self.round_irreversible_output;
        let defer_mct = self.defer_mct;
        let line_region_idwt_permitted = self.line_region_idwt_permitted;
        #[cfg(all(test, feature = "parallel"))]
        let force_per_sub_band = self.force_per_sub_band;
        #[cfg(all(test, feature = "parallel"))]
        let tile_route_post_decode = self.tile_route_post_decode;
        #[cfg(test)]
        let store_whole_code_blocks = self.store_whole_code_blocks;
        #[cfg(test)]
        let whole_window_tier1 = self.whole_window_tier1;
        #[cfg(test)]
        let reject_band_index = self.reject_band_index;
        #[cfg(test)]
        let separate_last_idwt_level = self.separate_last_idwt_level;
        #[cfg(test)]
        let staged_rgb8_pack = self.staged_rgb8_pack;
        #[cfg(test)]
        let serial_small_decodes = self.serial_small_decodes;
        #[cfg(test)]
        let enter_pool = self.enter_pool;
        #[cfg(test)]
        let planes_budget = self.planes_budget;
        *self = Self::default();
        self.output_region = output_region;
        self.round_irreversible_output = round_irreversible_output;
        self.defer_mct = defer_mct;
        self.line_region_idwt_permitted = line_region_idwt_permitted;
        #[cfg(all(test, feature = "parallel"))]
        {
            self.force_per_sub_band = force_per_sub_band;
            self.tile_route_post_decode = tile_route_post_decode;
        }
        #[cfg(test)]
        {
            self.store_whole_code_blocks = store_whole_code_blocks;
            self.whole_window_tier1 = whole_window_tier1;
            self.reject_band_index = reject_band_index;
            self.separate_last_idwt_level = separate_last_idwt_level;
            self.staged_rgb8_pack = staged_rgb8_pack;
            self.serial_small_decodes = serial_small_decodes;
            self.enter_pool = enter_pool;
            self.planes_budget = planes_budget;
        }
    }

    pub(super) fn release_tile_scratch_allocations(&mut self) {
        self.release_idwt_allocations();
        self.release_tier1_allocations();
    }

    pub(super) fn release_tier1_allocations(&mut self) {
        self.bit_plane_decode_context = BitPlaneDecodeContext::default();
        self.bit_plane_decode_buffers = BitPlaneDecodeBuffers::default();
        self.ht_block_decode_context = HtBlockDecodeContext::default();
        self.ht_clipped_block_scratch = Vec::new();
        #[cfg(feature = "parallel")]
        {
            self.ht_task_workspaces = Vec::new();
            self.classic_task_workspaces = Vec::new();
            self.parallel_coefficients = Vec::new();
        }
    }

    pub(crate) fn serial_tier1_capacity_bytes(&self) -> Result<usize> {
        let classic_bytes = self.bit_plane_decode_context.allocated_bytes()?;
        let buffer_bytes = self.bit_plane_decode_buffers.allocated_bytes()?;
        let ht_bytes = self.ht_block_decode_context.allocated_bytes()?;
        let clipped_bytes = self
            .ht_clipped_block_scratch
            .capacity()
            .checked_mul(size_of::<f32>())
            .ok_or(ValidationError::ImageTooLarge)?;
        classic_bytes
            .checked_add(buffer_bytes)
            .and_then(|bytes| bytes.checked_add(ht_bytes))
            .and_then(|bytes| bytes.checked_add(clipped_bytes))
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    pub(crate) fn tier1_capacity_bytes(&self) -> Result<usize> {
        let serial_bytes = self.serial_tier1_capacity_bytes()?;
        #[cfg(feature = "parallel")]
        {
            let classic_bytes = workspace::task_workspace_bytes(&self.classic_task_workspaces)?;
            serial_bytes
                .checked_add(workspace::task_workspace_bytes(&self.ht_task_workspaces)?)
                .and_then(|bytes| bytes.checked_add(classic_bytes))
                .and_then(|bytes| {
                    self.parallel_coefficients
                        .capacity()
                        .checked_mul(size_of::<f32>())
                        .and_then(|coefficient_bytes| bytes.checked_add(coefficient_bytes))
                })
                .ok_or(ValidationError::ImageTooLarge.into())
        }
        #[cfg(not(feature = "parallel"))]
        {
            Ok(serial_bytes)
        }
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn ht_parallel_workspace_count(&self) -> usize {
        self.ht_task_workspaces.len()
    }

    #[cfg(all(test, feature = "parallel"))]
    pub(crate) fn ht_parallel_workspace_bytes(&self) -> Result<usize> {
        workspace::task_workspace_bytes(&self.ht_task_workspaces)
    }

    pub(crate) fn idwt_capacity_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        include_capacity::<f32>(&mut bytes, self.idwt_output.coefficients.capacity())?;
        include_capacity::<i64>(&mut bytes, self.idwt_output.coefficients_i64.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_scratch_buffer.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_band_scratch.capacity())?;
        include_capacity::<i64>(&mut bytes, self.idwt_scratch_buffer_i64.capacity())?;
        include_capacity::<f32>(&mut bytes, self.idwt_line_rings.capacity())?;
        Ok(bytes)
    }

    pub(crate) fn retained_scratch_bytes(&self) -> Result<usize> {
        self.tier1_capacity_bytes()?
            .checked_add(self.idwt_capacity_bytes()?)
            .ok_or(ValidationError::ImageTooLarge.into())
    }

    fn release_idwt_allocations(&mut self) {
        self.idwt_output = IDWTOutput::default();
        self.idwt_scratch_buffer = Vec::new();
        self.idwt_band_scratch = Vec::new();
        self.idwt_scratch_buffer_i64 = Vec::new();
        self.idwt_line_rings = Vec::new();
    }
}
