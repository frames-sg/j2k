// SPDX-License-Identifier: MIT OR Apache-2.0

//! Banded Tier-1 for the row-streamed region IDWT.
//!
//! The streamed IDWT reads each stored sub-band one row at a time, top to
//! bottom, and never returns to a row it has read. So Tier-1 can decode a
//! sub-band one band at a time, the code-blocks of one code-block row, just
//! before the IDWT reads the band's first row, and keep only that band.
//!
//! Every band of every sub-band is decoded once and in order, including bands
//! the IDWT never reads, so a decode visits, decodes and rejects the same
//! code-blocks as the whole-window path. Only the order differs, so when a
//! banded decode fails, Tier-1 reruns over whole windows to report the error
//! the whole-window path would.

use alloc::vec::Vec;
use core::ops::Range;

use super::subband::decode_sub_band_bitplanes;
use super::{
    decode_component_tile_bit_planes_budgeted, tier1, ComponentInfo, CpuDecodeParallelism,
    DecodeAllocationBudget, DecodeProfileTimings, DecompositionStorage, Header, Result, Tile,
    TileDecodeContext,
};
use super::{record_workspace, store};
use crate::error::{DecodeError, DecodingError, ValidationError};
use crate::j2c::build;
use crate::j2c::codestream::WaveletTransform;
use crate::j2c::idwt::{ComponentLines, LinePlan, LineReconstruction, SubBandRows};
use crate::j2c::rect::IntRect;
use crate::profile;
use crate::{try_resize_decode_elements, ComponentPlane, ComponentRowSink};

/// Whether a tile's Tier-1 decodes in bands, given that it feeds the
/// streamed IDWT: in stored windows clipped to it, on the calling thread
/// alone, so banding costs it no parallelism, and every component has a level
/// to reconstruct. Everything here is known before the tile is laid out, so
/// `streams_rows` can require it of every tile up front.
pub(super) fn applies(
    header: &Header<'_>,
    tile: &Tile<'_>,
    parallelism: CpuDecodeParallelism,
    clip_to_idwt_windows: bool,
) -> bool {
    clip_to_idwt_windows
        && tier1_runs_serially(parallelism)
        && tile.component_infos.iter().all(|component| {
            component.num_decomposition_levels() > header.skipped_resolution_levels
        })
}

pub(super) fn tier1_runs_serially(parallelism: CpuDecodeParallelism) -> bool {
    #[cfg(feature = "parallel")]
    {
        parallelism != CpuDecodeParallelism::Auto || rayon::current_num_threads() < 2
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = parallelism;
        true
    }
}

/// Decode component `component_idx` of a banded tile: run the streamed IDWT,
/// decoding each band of its sub-bands as the IDWT reaches it, then the bands
/// it never reached.
#[expect(
    clippy::too_many_arguments,
    reason = "the banded stage shares decode_tile's state and profiling explicitly"
)]
pub(super) fn decode_component<'a>(
    tile: &Tile<'a>,
    component_idx: usize,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    transform: WaveletTransform,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<()> {
    let stage_start = profile::profile_now(profile_enabled);
    let workspace = tier1::prepare_tier1_workspace(
        tile,
        component_idx..component_idx + 1,
        header,
        tile_ctx,
        storage,
    )?;
    let mut tier1_us = 0;
    let result = decode_bands(
        tile,
        component_idx,
        header,
        tile_ctx,
        storage,
        transform,
        profile_enabled,
        &mut tier1_us,
    );
    tier1::release_tier1_workspace(tile_ctx, storage, &workspace)?;
    let elapsed = profile::elapsed_us(stage_start);
    profile_timings.codeblock_us += tier1_us;
    profile_timings.idwt_us += elapsed.saturating_sub(tier1_us);
    result.map_err(|banded_error| {
        whole_window_error(tile, component_idx, header, tile_ctx, storage, banded_error)
    })
}

/// Decode every component of a banded tile together and hand the region's
/// rows to `sink` as they are finished: each output row of every component,
/// transformed as [`store::transform_streamed_rows`] transforms the planes,
/// then the bands the IDWT never reached.
pub(super) fn decode_tile_rows<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    sink: &mut dyn ComponentRowSink,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<()> {
    // `streams_rows` required every tile to band, by the same `tile_bands`
    // check that laid this one out.
    debug_assert!(storage.banded_tier1, "a streamed tile must band");
    if !storage.banded_tier1 {
        return Err(
            DecodingError::UnsupportedFeature("row streaming for a tile without bands").into(),
        );
    }
    let stage_start = profile::profile_now(profile_enabled);
    let components = tile.component_infos.len();
    let workspace = tier1::prepare_tier1_workspace(tile, 0..components, header, tile_ctx, storage)?;
    let mut tier1_us = 0;
    let result = stream_rows(
        tile,
        header,
        tile_ctx,
        storage,
        sink,
        profile_enabled,
        &mut tier1_us,
    );
    tier1::release_tier1_workspace(tile_ctx, storage, &workspace)?;
    let elapsed = profile::elapsed_us(stage_start);
    profile_timings.codeblock_us += tier1_us;
    profile_timings.idwt_us += elapsed.saturating_sub(tier1_us);
    result.map_err(|banded_error| {
        whole_window_tile_error(tile, header, tile_ctx, storage, banded_error)
    })?;
    record_workspace(tile_ctx, storage, 0)
}

fn stream_rows<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    sink: &mut dyn ComponentRowSink,
    profile_enabled: bool,
    tier1_us: &mut u128,
) -> Result<()> {
    let components = tile.component_infos.len();
    let mut bands = Vec::new();
    let line_plans = plan_tile_lines(tile, header, tile_ctx, storage, &mut bands)?;
    let empty = IntRect::from_xywh(0, 0, 0, 0);
    let (copy, region) = line_plans
        .first()
        .map_or((empty, empty), |plan| (plan.copy, plan.region));
    let scratch_len = (copy.width() as usize)
        .checked_mul(components)
        .ok_or(ValidationError::ImageTooLarge)?;
    // The index first: once the ring buffer is out of `tile_ctx`, every
    // error path below must put it back.
    let (precinct_rows, index_bytes) = index_bands(storage, tile_ctx, &bands)?;
    let (mut buffer, ring_lens) =
        match take_line_buffer(tile_ctx, storage, &line_plans, scratch_len) {
            Ok(taken) => taken,
            Err(error) => {
                storage.structural_workspace_bytes -= index_bytes;
                return Err(error);
            }
        };
    let mut output = RowOutput {
        tile,
        sink,
        round_irreversible_output: tile_ctx.round_irreversible_output,
        depths: core::array::from_fn(|index| {
            tile_ctx
                .channel_data
                .get(index)
                .map_or((0, false), |component| {
                    (component.bit_depth, component.signed)
                })
        }),
        copy,
        region,
        streamed: 0,
    };
    let mut output_samples = 0usize;
    let result = (|| {
        let rings_len = buffer.len() - scratch_len;
        let (mut rings, scratch) = buffer.split_at_mut(rings_len);
        let mut lines = Vec::new();
        lines
            .try_reserve_exact(line_plans.len())
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        for (plan, len) in line_plans.iter().zip(&ring_lens) {
            let (ring, tail) = core::mem::take(&mut rings).split_at_mut(*len);
            rings = tail;
            lines.push(plan.reconstruction(ring)?);
        }
        let mut rows = BandedRows {
            header,
            component_infos: &tile.component_infos,
            tile_ctx: &mut *tile_ctx,
            storage: &mut *storage,
            bands,
            precinct_rows,
            profile_enabled,
            tier1_us: 0,
        };
        let result = output
            .write(&mut lines, &mut rows, scratch)
            .and_then(|()| rows.decode_remaining_bands());
        *tier1_us = rows.tier1_us;
        output_samples = lines.iter().map(LineReconstruction::output_samples).sum();
        result
    })();
    storage.structural_workspace_bytes -= index_bytes;
    tile_ctx.idwt_line_rings = buffer;
    tile_ctx.streamed_samples = tile_ctx.streamed_samples.saturating_add(output.streamed);
    tile_ctx.debug_counters.idwt_output_samples = tile_ctx
        .debug_counters
        .idwt_output_samples
        .saturating_add(output_samples);
    result
}

/// Add every component's bands to `bands` and plan its reconstruction.
/// Returns no plans when no component has rows to reconstruct.
fn plan_tile_lines(
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    bands: &mut Vec<SubBandBands>,
) -> Result<Vec<LinePlan>> {
    let mut line_plans = Vec::new();
    line_plans
        .try_reserve_exact(tile.component_infos.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut missing = 0;
    for (component_idx, component_info) in tile.component_infos.iter().enumerate() {
        let active_levels = push_component_bands(storage, header, component_idx, bands)?;
        match LinePlan::new(
            storage,
            tile_ctx,
            component_idx,
            active_levels,
            header,
            component_info.wavelet_transform(),
        )? {
            Some(plan) => line_plans.push(plan),
            None => missing += 1,
        }
    }
    if missing == tile.component_infos.len() {
        line_plans.clear();
        return Ok(line_plans);
    }
    // Every component of a 1x1-sampled tile reconstructs the same rows.
    if missing > 0
        || line_plans
            .iter()
            .any(|plan| plan.copy != line_plans[0].copy || plan.region != line_plans[0].region)
    {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    Ok(line_plans)
}

/// Take the context's ring buffer, grown to hold every plan's rings and
/// `scratch_len` more samples, and return it with each plan's ring length.
fn take_line_buffer(
    tile_ctx: &mut TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    line_plans: &[LinePlan],
    scratch_len: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    let mut ring_lens = Vec::new();
    ring_lens
        .try_reserve_exact(line_plans.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    for plan in line_plans {
        ring_lens.push(plan.ring_len()?);
    }
    let total = ring_lens
        .iter()
        .try_fold(scratch_len, |total, len| total.checked_add(*len))
        .ok_or(ValidationError::ImageTooLarge)?;
    debug_assert!(
        total <= storage.roi_idwt_charge_elements,
        "the ROI plan charged {} IDWT samples for {total}",
        storage.roi_idwt_charge_elements
    );
    let mut buffer = core::mem::take(&mut tile_ctx.idwt_line_rings);
    let resized = if buffer.len() < total {
        try_resize_decode_elements(&mut buffer, total, 0.0)
    } else {
        Ok(())
    };
    tile_ctx.idwt_line_rings = buffer;
    resized?;
    record_workspace(tile_ctx, storage, 0)?;
    let mut buffer = core::mem::take(&mut tile_ctx.idwt_line_rings);
    buffer.truncate(total);
    Ok((buffer, ring_lens))
}

/// Where a tile's reconstructed rows go: transformed, then to the sink.
struct RowOutput<'o, 't> {
    tile: &'o Tile<'t>,
    sink: &'o mut dyn ComponentRowSink,
    round_irreversible_output: bool,
    /// Each component's bit depth and signedness.
    depths: [(u8, bool); 4],
    /// The rows, in the coordinates of `region`, that the tile reconstructs.
    copy: IntRect,
    region: IntRect,
    /// Samples handed to the sink.
    streamed: usize,
}

impl RowOutput<'_, '_> {
    /// Advance every component's reconstruction a row at a time, copying
    /// each row into `scratch`, one row per component, and hand them over.
    fn write(
        &mut self,
        lines: &mut [LineReconstruction<'_>],
        rows: &mut BandedRows<'_, '_>,
        scratch: &mut [f32],
    ) -> Result<()> {
        let (copy, components) = (self.copy, lines.len());
        let width = copy.width() as usize;
        if lines.is_empty() || width == 0 {
            return Ok(());
        }
        for y in copy.y0..copy.y1 {
            let mut outputs: [&mut [f32]; 4] = Default::default();
            for ((line, output), chunk) in lines
                .iter_mut()
                .zip(outputs.iter_mut())
                .zip(scratch.chunks_exact_mut(width))
            {
                let (row_y, row) = line
                    .next_row(rows)?
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                if row_y != y || row.len() != width {
                    return Err(DecodingError::CodeBlockDecodeFailure.into());
                }
                chunk.copy_from_slice(row);
                *output = chunk;
            }
            let outputs = &mut outputs[..components];
            store::transform_streamed_rows(self.tile, self.round_irreversible_output, outputs)?;
            let component_rows: [ComponentPlane<'_>; 4] =
                core::array::from_fn(|index| ComponentPlane {
                    samples: outputs.get(index).map_or(&[][..], |row| &**row),
                    dimensions: (copy.width(), 1),
                    bit_depth: self.depths[index].0,
                    signed: self.depths[index].1,
                    sampling: (1, 1),
                });
            self.sink.write_rows(
                copy.x0 - self.region.x0,
                y - self.region.y0,
                &component_rows[..components],
            );
            self.streamed = self.streamed.saturating_add(width * components);
        }
        Ok(())
    }
}

/// When a lockstep decode fails, rerun Tier-1 over whole windows one
/// component at a time, in the order the planes path decodes them, and
/// return the first error. If the whole windows don't fit the cap, or the
/// rerun succeeds, keep the banded error.
fn whole_window_tile_error<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    banded_error: DecodeError,
) -> DecodeError {
    storage.coefficient_layout = build::CoefficientLayout::Component;
    if build::lay_out_whole_region_coefficients(storage).is_err() {
        return banded_error;
    }
    for component_idx in 0..tile.component_infos.len() {
        storage.stale_coefficients = true;
        if let Err(error) = decode_component_tile_bit_planes_budgeted(
            tile,
            component_idx..component_idx + 1,
            tile_ctx,
            storage,
            header,
            &mut None,
            CpuDecodeParallelism::Serial,
            false,
        ) {
            return error;
        }
    }
    banded_error
}

#[expect(
    clippy::too_many_arguments,
    reason = "the banded stage shares decode_tile's state and profiling explicitly"
)]
fn decode_bands<'a>(
    tile: &Tile<'a>,
    component_idx: usize,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    transform: WaveletTransform,
    profile_enabled: bool,
    tier1_us: &mut u128,
) -> Result<()> {
    let mut bands = Vec::new();
    let active_levels = push_component_bands(storage, header, component_idx, &mut bands)?;
    let (precinct_rows, index_bytes) = index_bands(storage, tile_ctx, &bands)?;

    tile_ctx.idwt_output.placement = crate::j2c::idwt::IdwtPlacement::Coefficients;
    tile_ctx.idwt_output.coefficients = Vec::new();
    tile_ctx.idwt_output.rect = IntRect::from_xywh(0, 0, 0, 0);
    let mut lines = match ComponentLines::prepare(
        storage,
        tile_ctx,
        component_idx,
        active_levels,
        header,
        transform,
    ) {
        Ok(lines) => lines,
        Err(error) => {
            storage.structural_workspace_bytes -= index_bytes;
            return Err(error);
        }
    };

    let mut rows = BandedRows {
        header,
        component_infos: &tile.component_infos,
        tile_ctx: &mut *tile_ctx,
        storage: &mut *storage,
        bands,
        precinct_rows,
        profile_enabled,
        tier1_us: 0,
    };
    let result = lines.as_mut().map_or(Ok(()), |lines| lines.run(&mut rows));
    let result = result.and_then(|()| rows.decode_remaining_bands());
    *tier1_us = rows.tier1_us;
    drop(rows);
    let result = lines.map_or(result, |lines| lines.finish(tile_ctx, result));
    storage.structural_workspace_bytes -= index_bytes;
    result
}

/// Add the bands of each active resolution's sub-bands of `component_idx`,
/// in the order the whole-window path decodes them, and return how many
/// levels the component reconstructs.
fn push_component_bands(
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    component_idx: usize,
    bands: &mut Vec<SubBandBands>,
) -> Result<usize> {
    let tile_decompositions = &storage.tile_decompositions[component_idx];
    let active_levels = tile_decompositions
        .decompositions
        .len()
        .checked_sub(usize::from(header.skipped_resolution_levels))
        .filter(|levels| *levels > 0)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    bands
        .try_reserve(1 + 3 * active_levels)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    for resolution in 0..=active_levels {
        let resolution =
            u8::try_from(resolution).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
        for sub_band in tile_decompositions.sub_band_iter(resolution, &storage.decompositions) {
            let stored = storage.sub_bands[sub_band].storage_rect;
            bands.push(SubBandBands {
                sub_band,
                component: component_idx,
                resolution,
                stored,
                slot: IntRect::from_ltrb(stored.x0, stored.y0, stored.x1, stored.y0),
                last_band: None,
            });
        }
    }
    Ok(active_levels)
}

/// Build the row index of `bands` and charge it to the structural
/// workspace, returning it with its bytes, which the caller releases.
fn index_bands(
    storage: &mut DecompositionStorage<'_>,
    tile_ctx: &TileDecodeContext,
    bands: &[SubBandBands],
) -> Result<(Option<Vec<Vec<PrecinctRow>>>, usize)> {
    // The row index is optional. If building it fails, use the original scans.
    // A built index stays charged while the bands decode and allocate scratch.
    let mut index_budget = DecodeAllocationBudget::for_storage(storage)?;
    let baseline_bytes = index_budget.live_bytes();
    #[cfg(test)]
    let precinct_rows = if tile_ctx.reject_band_index {
        Err(ValidationError::ImageTooLarge.into())
    } else {
        build_precinct_rows(storage, bands, &mut index_budget)
    }
    .ok();
    #[cfg(not(test))]
    let precinct_rows = {
        let _ = tile_ctx;
        build_precinct_rows(storage, bands, &mut index_budget).ok()
    };
    let index_bytes = if precinct_rows.is_some() {
        index_budget.live_bytes() - baseline_bytes
    } else {
        0
    };
    storage.structural_workspace_bytes = storage
        .structural_workspace_bytes
        .checked_add(index_bytes)
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok((precinct_rows, index_bytes))
}

/// When a banded decode fails, rerun the component's Tier-1 over whole
/// windows, the old way, and return its error, which is the one the
/// whole-window path reports. If the whole windows don't fit the cap, or the
/// rerun succeeds, keep the banded error.
pub(super) fn whole_window_error<'a>(
    tile: &Tile<'a>,
    component_idx: usize,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    banded_error: DecodeError,
) -> DecodeError {
    if build::lay_out_whole_region_coefficients(storage).is_err() {
        return banded_error;
    }
    storage.stale_coefficients = true;
    match decode_component_tile_bit_planes_budgeted(
        tile,
        component_idx..component_idx + 1,
        tile_ctx,
        storage,
        header,
        &mut None,
        CpuDecodeParallelism::Serial,
        false,
    ) {
        Err(error) => error,
        Ok(()) => banded_error,
    }
}

/// The bands of one sub-band.
struct SubBandBands {
    sub_band: usize,
    component: usize,
    resolution: u8,
    /// The sub-band's stored rectangle.
    stored: IntRect,
    /// The stored rows of the band its coefficients hold.
    slot: IntRect,
    /// The first row of the last band decoded.
    last_band: Option<u32>,
}

struct PrecinctRow {
    first_y: u32,
    last_y: u32,
    precincts: Range<usize>,
}

fn build_precinct_rows(
    storage: &DecompositionStorage<'_>,
    bands: &[SubBandBands],
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<Vec<PrecinctRow>>> {
    let mut indexed = Vec::new();
    budget.reserve_new(&mut indexed, bands.len())?;
    for band in bands {
        indexed.push(collect_precinct_rows(storage, band.sub_band, budget)?);
    }
    Ok(indexed)
}

fn collect_precinct_rows(
    storage: &DecompositionStorage<'_>,
    sub_band: usize,
    budget: &mut DecodeAllocationBudget,
) -> Result<Vec<PrecinctRow>> {
    let mut rows: Vec<PrecinctRow> = Vec::new();
    for index in storage.sub_bands[sub_band].precincts.clone() {
        let precinct = &storage.precincts[index];
        let blocks = &storage.code_blocks[precinct.code_blocks.clone()];
        let Some((first, last)) = blocks.first().zip(blocks.last()) else {
            continue;
        };
        if let Some(row) = rows.last_mut().filter(|row| row.first_y == first.rect.y0) {
            row.last_y = row.last_y.max(last.rect.y0);
            row.precincts.end = index + 1;
        } else {
            budget.reserve_additional(&mut rows, 1)?;
            rows.push(PrecinctRow {
                first_y: first.rect.y0,
                last_y: last.rect.y0,
                precincts: index..index + 1,
            });
        }
    }
    Ok(rows)
}

/// Decodes the bands of a component's sub-bands as the streamed IDWT asks
/// for their rows.
struct BandedRows<'r, 'a> {
    header: &'r Header<'r>,
    component_infos: &'r [ComponentInfo],
    tile_ctx: &'r mut TileDecodeContext,
    storage: &'r mut DecompositionStorage<'a>,
    bands: Vec<SubBandBands>,
    precinct_rows: Option<Vec<Vec<PrecinctRow>>>,
    profile_enabled: bool,
    tier1_us: u128,
}

impl BandedRows<'_, '_> {
    fn position(&self, sub_band: usize) -> Result<usize> {
        self.bands
            .iter()
            .position(|bands| bands.sub_band == sub_band)
            .ok_or(DecodingError::CodeBlockDecodeFailure.into())
    }

    /// The precincts whose code-block rows can contain `y`. Build visits
    /// precincts from top to bottom, then left to right within a row.
    fn band_precincts(&self, index: usize, y: u32) -> Result<Range<usize>> {
        let rows = self
            .precinct_rows
            .as_ref()
            .and_then(|indexed| indexed.get(index))
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let at = rows.partition_point(|row| row.last_y < y);
        rows.get(at)
            .filter(|row| row.first_y <= y)
            .map(|row| row.precincts.clone())
            .ok_or(DecodingError::CodeBlockDecodeFailure.into())
    }

    /// The rows of the sub-band's first band below `last_band`, or `None`
    /// when no code-block starts below it. A code-block row of a sub-band
    /// starts and ends on the same rows for all of its code-blocks.
    fn next_band(&self, index: usize) -> Option<Range<u32>> {
        let bands = &self.bands[index];
        let Some(indexed) = self.precinct_rows.as_ref() else {
            let sub_band = &self.storage.sub_bands[bands.sub_band];
            return self.storage.precincts[sub_band.precincts.clone()]
                .iter()
                .flat_map(|precinct| &self.storage.code_blocks[precinct.code_blocks.clone()])
                .map(|block| block.rect)
                .filter(|rect| bands.last_band.is_none_or(|last| rect.y0 > last))
                .min_by_key(|rect| rect.y0)
                .map(|rect| rect.y0..rect.y1);
        };
        let rows = &indexed[index];
        let start = bands
            .last_band
            .map_or(0, |last| rows.partition_point(|row| row.last_y <= last));
        for row in &rows[start..] {
            let mut next: Option<IntRect> = None;
            for index in row.precincts.clone() {
                let precinct = &self.storage.precincts[index];
                let blocks = &self.storage.code_blocks[precinct.code_blocks.clone()];
                let at = blocks.partition_point(|block| {
                    bands.last_band.is_some_and(|last| block.rect.y0 <= last)
                });
                if let Some(block) = blocks.get(at) {
                    if next.is_none_or(|rect| block.rect.y0 < rect.y0) {
                        next = Some(block.rect);
                    }
                }
            }
            if let Some(rect) = next {
                return Some(rect.y0..rect.y1);
            }
        }
        None
    }

    /// Decode the code-blocks that start at `rows.start` into the
    /// sub-band's coefficients, which then hold the band's stored rows.
    fn decode_band(&mut self, index: usize, rows: Range<u32>) -> Result<()> {
        let start = profile::profile_now(self.profile_enabled);
        let bands = &self.bands[index];
        let (sub_band, resolution, stored) = (bands.sub_band, bands.resolution, bands.stored);
        let component_info = self
            .component_infos
            .get(bands.component)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?;
        let top = rows.start.clamp(stored.y0, stored.y1);
        let bottom = rows.end.clamp(top, stored.y1);
        let slot = IntRect::from_ltrb(stored.x0, top, stored.x1, bottom);
        let original_precincts = self.storage.sub_bands[sub_band].precincts.clone();
        let band_precincts = self
            .precinct_rows
            .as_ref()
            .map(|_| self.band_precincts(index, rows.start))
            .transpose()?;
        let coefficients = self.storage.sub_bands[sub_band].coefficients.clone();
        self.storage
            .coefficients
            .get_mut(coefficients)
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
            .fill(0.0);
        self.storage.sub_bands[sub_band].storage_rect = slot;
        if let Some(precincts) = band_precincts {
            self.storage.sub_bands[sub_band].precincts = precincts;
        }
        self.storage.tier1_band = Some(rows.start);
        let result = decode_sub_band_bitplanes(
            sub_band,
            resolution,
            component_info,
            self.tile_ctx,
            self.storage,
            self.header,
            &mut None,
            CpuDecodeParallelism::Serial,
            self.profile_enabled,
        );
        self.storage.tier1_band = None;
        self.storage.sub_bands[sub_band].storage_rect = stored;
        self.storage.sub_bands[sub_band].precincts = original_precincts;
        let bands = &mut self.bands[index];
        bands.slot = slot;
        bands.last_band = Some(rows.start);
        self.tier1_us += profile::elapsed_us(start);
        result
    }

    /// Decode every band the IDWT left, so that each code-block the
    /// whole-window path decodes is decoded here too.
    fn decode_remaining_bands(&mut self) -> Result<()> {
        for index in 0..self.bands.len() {
            while let Some(rows) = self.next_band(index) {
                self.decode_band(index, rows)?;
            }
        }
        Ok(())
    }
}

impl SubBandRows for BandedRows<'_, '_> {
    fn prepare_row(&mut self, sub_band: usize, y: u32) -> Result<()> {
        let index = self.position(sub_band)?;
        let stored = self.bands[index].stored;
        if y < stored.y0 || y >= stored.y1 {
            // Outside the stored rows the sub-band reads as zero.
            return Ok(());
        }
        while self.bands[index].slot.y1 <= y {
            match self.next_band(index) {
                Some(rows) if rows.start <= y => self.decode_band(index, rows)?,
                // No code-block covers the row, so it reads as zero.
                _ => {
                    let bands = &mut self.bands[index];
                    bands.slot = IntRect::from_ltrb(stored.x0, y, stored.x1, y);
                    break;
                }
            }
        }
        if y < self.bands[index].slot.y0 {
            // Rows are read in order, so a band already left can't be read
            // again.
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        Ok(())
    }

    fn samples(&self, sub_band: usize) -> (&[f32], IntRect, u32) {
        let Ok(index) = self.position(sub_band) else {
            return (&[], IntRect::from_xywh(0, 0, 0, 0), 0);
        };
        let bands = &self.bands[index];
        let coefficients = self.storage.sub_bands[sub_band].coefficients.clone();
        (
            &self.storage.coefficients[coefficients],
            bands.slot,
            bands.stored.width(),
        )
    }
}
