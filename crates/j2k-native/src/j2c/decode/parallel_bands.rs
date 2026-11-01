// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tier-1 decoded band by band on the worker pool, ahead of the streamed IDWT.
//!
//! A serial region decode on the streamed IDWT decodes each band of
//! code-block rows just before the IDWT reads it (see `banded_tier1`). With
//! several threads, the tile route instead decodes every code-block of a
//! component before the IDWT starts. Here the route's code-block rows are
//! bands that pool workers decode ahead of the IDWT, which runs on the
//! calling thread, so the two overlap:
//!
//! - Each sub-band holds a ring of band slots. Its band `k` can start once the
//!   IDWT has read all of band `k - slots`, which frees that band's slot.
//! - Workers take the band the IDWT will read first. A worker that finds no
//!   band it can start returns to the pool rather than wait, and the IDWT
//!   sends workers again when it frees slots.
//! - The IDWT waits only for a band a worker is decoding. A band nobody has
//!   started it decodes itself, so it never depends on a pool thread being
//!   free, even when the pool is busy with other work.
//! - When the finest level's slots ahead of the IDWT hold fewer bands than
//!   the pool has threads, as with one component, a classic band splits into
//!   parts, runs of its blocks that threads decode at once into the band's
//!   slot, so every thread has work without holding more bands.
//!
//! Every band is decoded once, including bands the IDWT never reads, so a
//! decode visits, decodes and rejects the code-blocks the whole-window route
//! does. Workers fail in any order, so a failure reruns whole-window Tier-1
//! serially for the error the whole-window path reports.
//!
//! A region decode that streams its rows to a sink runs every component's
//! bands on one executor instead, while the IDWT advances the components
//! together, a row of each at a time, as the serial bands of
//! `banded_tier1::decode_tile_rows` do. The pool then holds every
//! component's next bands at once, and the region needs no planes.
//!
//! A running band always belongs to a worker that is still active. A wait
//! that finds a running band with no worker active would otherwise last
//! forever, so it gives up the pool instead, and the component decodes over
//! whole windows; a debug build asserts.

use alloc::collections::BinaryHeap;
use alloc::vec::Vec;
use core::cmp::Reverse;
use core::mem::size_of;
use core::ops::Range;
use std::sync::{Condvar, Mutex, MutexGuard};

use super::banded_tier1::{self, RowOutput};
use super::subband::{
    plan_band_route, plan_band_route_beside, pool_entry_bytes, routes_tile, BandOutput, BandRoute,
    BandRouting, BlockScratch, PayloadSource, LOCK_BYTES,
};
use super::workspace::{TaskScratch, TaskWorkspace};
use super::{
    clips_to_idwt_windows, record_workspace, CpuDecodeParallelism, DecodeAllocationBudget,
    DecodeProfileTimings, DecompositionStorage, Header, Result, Tile, TileDecodeContext,
};
use crate::error::{DecodeError, DecodingError, ValidationError};
use crate::j2c::build::{self, CodeBlockCoding};
use crate::j2c::codestream::WaveletTransform;
use crate::j2c::idwt::{
    streamed_batch_rows, ComponentLines, LinePlan, LineReconstruction, SubBandRows,
};
use crate::j2c::rect::IntRect;
use crate::profile;
use crate::ComponentRowSink;

/// Sub-band rows of support on each side of the rows a level's lifting
/// reads, in rows of that level, for ordering bands: a 9/7 lift reads two
/// rows past each side, and both of its stages do.
const SUPPORT_ROWS: i64 = 4;

/// How often a wait for a running band checks that a worker is still active
/// to finish it.
const STRANDED_CHECK: std::time::Duration = std::time::Duration::from_millis(50);

/// Whether a tile's components decode their bands on the worker pool: it
/// feeds the streamed IDWT, in stored windows clipped to it, under `Auto`
/// with two threads or more, every component has a level to reconstruct, and
/// the tile route decodes its code-blocks. The serial bands of
/// [`banded_tier1`] cover the other streamed tiles.
pub(super) fn applies(
    header: &Header<'_>,
    tile: &Tile<'_>,
    tile_ctx: &TileDecodeContext,
    hooked: bool,
    parallelism: CpuDecodeParallelism,
) -> bool {
    let bands = tile_ctx.line_region_idwt
        && parallelism == CpuDecodeParallelism::Auto
        && rayon::current_num_threads() >= 2
        && clips_to_idwt_windows(tile_ctx, hooked)
        && tile.component_infos.iter().all(|component| {
            component.num_decomposition_levels() > header.skipped_resolution_levels
        })
        && routes_tile(tile);
    #[cfg(test)]
    let bands = bands && !tile_ctx.whole_window_tier1 && !tile_ctx.force_per_sub_band;
    bands
}

/// Decode component `component_idx`: its bands on the pool while the
/// streamed IDWT reads them, then the bands it never read. Returns `false`
/// when the route declines the component's bands; its sub-bands then hold
/// whole windows again, for the whole-window path.
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
) -> Result<bool> {
    let stage_start = profile::profile_now(profile_enabled);
    let routing = plan_band_route(tile, component_idx, tile_ctx, storage, header, false);
    let route = match routing {
        // Planning can fail in a later band before earlier blocks decode.
        Err(error) => {
            return Err(banded_tier1::whole_window_error(
                tile,
                component_idx,
                header,
                tile_ctx,
                storage,
                error,
            ));
        }
        Ok(BandRouting::Route(route)) => route,
        Ok(BandRouting::Serial) => {
            // A ring's first slot holds a band, as the serial bands need.
            banded_tier1::decode_component(
                tile,
                component_idx,
                header,
                tile_ctx,
                storage,
                transform,
                profile_enabled,
                profile_timings,
            )?;
            return Ok(true);
        }
        Ok(BandRouting::WholeWindows) => {
            build::lay_out_whole_region_coefficients(storage)?;
            storage.stale_coefficients = true;
            return Ok(false);
        }
    };
    let mut tier1_us = 0;
    let result = decode_route(
        &route,
        component_idx,
        header,
        tile_ctx,
        storage,
        transform,
        profile_enabled,
        &mut tier1_us,
    )
    .and_then(|completed| {
        if completed == Some(true) {
            route.record(tile_ctx, storage)?;
        }
        Ok(completed)
    });
    // Neither fallback's budget includes this plan's live allocations.
    drop(route);
    let elapsed = profile::elapsed_us(stage_start);
    profile_timings.codeblock_us += tier1_us;
    profile_timings.idwt_us += elapsed.saturating_sub(tier1_us);
    match result {
        Ok(Some(true)) => {
            tile_ctx.debug_counters.pool_band_components += 1;
            Ok(true)
        }
        Ok(None) => {
            // The executor did not fit beside the plan, before anything
            // decoded; the serial bands need neither.
            banded_tier1::decode_component(
                tile,
                component_idx,
                header,
                tile_ctx,
                storage,
                transform,
                profile_enabled,
                profile_timings,
            )?;
            Ok(true)
        }
        Ok(Some(false)) => {
            // A band was left running with no worker to finish it.
            #[cfg(test)]
            let simulated = tile_ctx.strand_pool_band;
            #[cfg(not(test))]
            let simulated = false;
            debug_assert!(simulated, "a running band had no active worker");
            tile_ctx.debug_counters.pool_band_fallbacks += 1;
            build::lay_out_whole_region_coefficients(storage)?;
            storage.stale_coefficients = true;
            Ok(false)
        }
        Err(parallel_error) => Err(banded_tier1::whole_window_error(
            tile,
            component_idx,
            header,
            tile_ctx,
            storage,
            parallel_error,
        )),
    }
}

/// Decode the bands of `route`'s component on the pool while the streamed
/// IDWT reads them. Returns `None`, before anything decodes, when the
/// rings and bands, the executor's records or its threads' scratch do not
/// fit beside the route's plan, and `Some(false)` when a band was left
/// running with no worker to finish it.
#[expect(
    clippy::too_many_arguments,
    reason = "the banded stage shares decode_tile's state and profiling explicitly"
)]
fn decode_route(
    route: &BandRoute<'_>,
    component_idx: usize,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'_>,
    transform: WaveletTransform,
    profile_enabled: bool,
    tier1_us: &mut u128,
) -> Result<Option<bool>> {
    // The rings and bands live beside the route's plan and scratch while
    // the bands decode; planning them holds their row scales beside them.
    let mut budget = route.budget();
    let (planned, scales) = ring_plan_bytes(route, storage, header, component_idx)?;
    let mut planning = budget;
    if declines(planning.include_bytes(planned + scales))?
        || declines(budget.include_bytes(planned))?
    {
        return Ok(None);
    }
    let (rings, mut bands, active_levels) = plan_rings(route, storage, header, component_idx)?;
    #[cfg(test)]
    count_ring_slots(tile_ctx, &rings);
    if route.coding() == CodeBlockCoding::Classic {
        split_bands(
            tile_ctx,
            &rings,
            &mut bands,
            route.workers() + 1,
            |band, parts| route.strip_parts(band.strip, parts),
        )?;
    }
    // So do the executor's records and each thread's scratch, the route's
    // workers' and this thread's.
    if declines(budget.include_bytes(executor_bytes(&rings, &bands, (route.workers(), 1))?))? {
        return Ok(None);
    }
    let mut scratches =
        match BlockScratch::for_threads(route.workers() + 1, route.scratch_sizes()?, &mut budget) {
            Err(error) if super::reuse::is_capacity_error(&error) => return Ok(None),
            scratches => scratches?,
        };
    tile_ctx.idwt_output.placement = crate::j2c::idwt::IdwtPlacement::Coefficients;
    tile_ctx.idwt_output.coefficients = Vec::new();
    tile_ctx.idwt_output.rect = IntRect::from_xywh(0, 0, 0, 0);
    let mut lines = ComponentLines::prepare(
        storage,
        tile_ctx,
        component_idx,
        active_levels,
        header,
        transform,
    )?;
    #[cfg(test)]
    let strand_once = tile_ctx.strand_pool_band;
    #[cfg(not(test))]
    let strand_once = false;
    let mut coefficients = core::mem::take(&mut storage.coefficients);
    let slots = split_slots(&rings, &bands, storage, &mut coefficients);
    let source = PayloadSource::new(storage);
    let result = slots.and_then(|slots| {
        let run = |rows: &mut [&mut (dyn SubBandRows + Send)]| -> Result<()> {
            match (lines.as_mut(), rows.first_mut()) {
                (Some(lines), Some(rows)) => lines.run(&mut **rows),
                _ => Ok(()),
            }
        };
        match route.coding() {
            CodeBlockCoding::Classic => run_bands(
                (route.workers(), 1, false),
                &rings,
                &bands,
                slots,
                (&mut tile_ctx.classic_task_workspaces, &mut scratches),
                |index, output, workspace, scratch| {
                    route.decode_classic(index, output, workspace, (scratch, source))
                },
                run,
                profile_enabled,
                tier1_us,
                strand_once,
            ),
            CodeBlockCoding::HighThroughput => run_bands(
                (route.workers(), 1, false),
                &rings,
                &bands,
                slots,
                (&mut tile_ctx.ht_task_workspaces, &mut scratches),
                |index, output, workspace, scratch| {
                    route.decode_ht(index, output, workspace, (scratch, source))
                },
                run,
                profile_enabled,
                tier1_us,
                strand_once,
            ),
        }
    });
    storage.coefficients = coefficients;
    // Each band zeroes its slot before it decodes, and the last bands leave
    // theirs written.
    storage.stale_coefficients = true;
    let Some(lines) = lines else {
        return result.map(Some);
    };
    match result {
        Ok(true) => lines.finish(tile_ctx, Ok(())).map(|()| Some(true)),
        Ok(false) => {
            // Return the rings and plane; the whole-window path refills them.
            match lines.finish(tile_ctx, Err(DecodingError::CodeBlockDecodeFailure.into())) {
                Ok(()) | Err(_) => Ok(Some(false)),
            }
        }
        Err(error) => lines.finish(tile_ctx, Err(error)).map(|()| Some(true)),
    }
}

/// Whether `result` failed for want of budget, which declines the pool's
/// bands; other errors pass on.
fn declines(result: Result<()>) -> Result<bool> {
    match result {
        Err(error) if super::reuse::is_capacity_error(&error) => Ok(true),
        result => result.map(|()| false),
    }
}

/// Decode every component of a tile whose rows stream to a sink: the bands
/// of all its components on the pool at once, while the streamed IDWT
/// advances the components together on this thread and hands their rows
/// over, as `banded_tier1::decode_tile_rows` does with serial bands.
/// Returns `false`, before anything is decoded or streamed, when the route
/// declines a component, or when a band was stranded before any row was
/// streamed; the tile's bands then decode serially.
#[expect(
    clippy::too_many_lines,
    reason = "the streamed tile keeps its plans, executor, rings, and failure handling together"
)]
pub(super) fn decode_tile_rows<'a>(
    tile: &Tile<'a>,
    header: &Header<'_>,
    tile_ctx: &mut TileDecodeContext,
    storage: &mut DecompositionStorage<'a>,
    sink: &mut dyn ComponentRowSink,
    profile_enabled: bool,
    profile_timings: &mut DecodeProfileTimings,
) -> Result<bool> {
    let stage_start = profile::profile_now(profile_enabled);
    let components = tile.component_infos.len();
    // Every component's route stays live beside the others' while the
    // executor runs them all, so each is planned in the budget the last one
    // left, as one budget, which starts with the list of routes.
    // A budget too small for the pool leaves the tile to serial bands.
    let mut live = DecodeAllocationBudget::for_storage(storage)?;
    let mut routes = Vec::new();
    if declines(live.reserve_new(&mut routes, components))? {
        return Ok(false);
    }
    for component_idx in 0..components {
        match plan_band_route_beside(
            tile,
            component_idx,
            tile_ctx,
            storage,
            header,
            Some(live),
            true,
        ) {
            Ok(BandRouting::Route(route)) => {
                live = route.budget();
                routes.push(route);
            }
            Ok(BandRouting::Serial | BandRouting::WholeWindows) => return Ok(false),
            // Planning can fail in a later band before earlier blocks decode.
            Err(error) => {
                // The whole-window rerun's budget leaves the plans out.
                drop(routes);
                return Err(banded_tier1::whole_window_tile_error(
                    tile, header, tile_ctx, storage, error,
                ));
            }
        }
    }
    // Every component's rings and bands live beside the plans while the
    // bands decode; planning each holds its own rings, bands and row scales
    // beside them.
    let (planned, planning) = tile_ring_plan_bytes(&routes, storage, header)?;
    let mut peak = live;
    if declines(peak.include_bytes(planned + planning))? || declines(live.include_bytes(planned))? {
        return Ok(false);
    }
    let (rings, mut bands, strip_starts, active_levels) =
        plan_tile_rings(&routes, storage, header)?;
    // So do the streamed IDWT's plans, its rings' lengths and its
    // reconstructions, one of each per component.
    let line_lists = components
        .checked_mul(
            size_of::<LinePlan>() + size_of::<usize>() + size_of::<LineReconstruction<'_>>(),
        )
        .ok_or(ValidationError::ImageTooLarge)?;
    if declines(live.include_bytes(line_lists))? {
        return Ok(false);
    }
    let line_plans = plan_tile_lines(tile, header, tile_ctx, storage, &active_levels)?;
    let empty = IntRect::from_xywh(0, 0, 0, 0);
    let (copy, region) = line_plans
        .first()
        .map_or((empty, empty), |plan| (plan.copy, plan.region));
    // Every route plans the tile's one block coding (see `routes_tile`).
    let coding = routes
        .first()
        .map_or(CodeBlockCoding::Classic, |route| route.coding());
    // Each component's reconstruction reads through its own view of the
    // bands, with a workspace of its own from the bank the routes prepared
    // for their workers and the calling thread.
    let bank = match coding {
        CodeBlockCoding::Classic => tile_ctx.classic_task_workspaces.len(),
        CodeBlockCoding::HighThroughput => tile_ctx.ht_task_workspaces.len(),
    };
    let route_workers = routes
        .iter()
        .map(|route| route.workers())
        .max()
        .unwrap_or(0);
    let samples = (copy.width() as usize).saturating_mul(copy.height() as usize);
    let readers = if parallel_synthesis(components, coding, bank, route_workers, samples) {
        components
    } else {
        1
    };
    let workers = route_workers.min(bank.saturating_sub(readers));
    #[cfg(test)]
    count_ring_slots(tile_ctx, &rings);
    if coding == CodeBlockCoding::Classic {
        split_bands(
            tile_ctx,
            &rings,
            &mut bands,
            workers + readers,
            |band, parts| {
                let (route, strip) = route_strip(&routes, &strip_starts, band.strip)?;
                route.strip_parts(strip, parts)
            },
        )?;
    }
    // The executor's records live beside every route's plan and scratch
    // while the bands decode.
    if declines(live.include_bytes(executor_bytes(&rings, &bands, (workers, readers))?))? {
        return Ok(false);
    }
    // Each thread's scratch serves every component's bands, so it holds,
    // and is charged once for, the largest of the routes'; the serial bands
    // need none.
    let threads = workers + readers;
    #[cfg(test)]
    let unscratched = live.live_bytes();
    let Some(mut scratches) = BlockScratch::charged_for_threads(
        threads,
        BandRoute::shared_scratch(routes.iter().map(|route| &**route))?,
        &mut live,
    )?
    else {
        return Ok(false);
    };
    #[cfg(test)]
    {
        tile_ctx.debug_counters.thread_scratch_charged += live.live_bytes() - unscratched;
    }
    // Rows reconstruct in batches, charged by the region's width (see
    // `roi_idwt_peak_elements`): two of them with a reader per component of
    // an HTJ2K tile, so one finishes while the next reconstructs; a classic
    // tile's readers wait on Tier-1 instead, and hold one batch of half the
    // rows, with deep band rings when they read whole rows. One reader holds
    // one short batch, which it finishes before it reconstructs the next.
    let pipelined = readers > 1 && coding == CodeBlockCoding::HighThroughput;
    let batch = streamed_batch_rows(
        region.width() as usize,
        copy.height() as usize,
        readers > 1 && !pipelined,
        readers == 1,
    );
    let scratch_len = (copy.width() as usize)
        .checked_mul(components)
        .and_then(|samples| samples.checked_mul(batch))
        .and_then(|samples| samples.checked_mul(if pipelined { 2 } else { 1 }))
        .ok_or(ValidationError::ImageTooLarge)?;
    let (mut buffer, ring_lens) =
        banded_tier1::take_line_buffer(tile_ctx, storage, &line_plans, scratch_len)?;
    #[cfg(test)]
    {
        let held = ring_lens
            .iter()
            .fold(scratch_len, |held, len| held.saturating_add(*len));
        let counters = &mut tile_ctx.debug_counters;
        counters.pool_band_readers = counters.pool_band_readers.max(readers);
        counters.uncharged_line_samples = counters
            .uncharged_line_samples
            .max(held.saturating_sub(storage.roi_idwt_charge_elements));
    }
    let mut output = RowOutput::new(tile, sink, tile_ctx, copy, region);
    // One reader for every component of a classic tile leaves the pool's
    // threads to Tier-1, and decodes bands itself while it waits.
    let helps = readers == 1 && coding == CodeBlockCoding::Classic;
    #[cfg(test)]
    let strand_once = tile_ctx.strand_pool_band;
    #[cfg(not(test))]
    let strand_once = false;
    let mut tier1_us = 0;
    let mut output_samples = 0usize;
    let mut coefficients = core::mem::take(&mut storage.coefficients);
    let slots = split_slots(&rings, &bands, storage, &mut coefficients);
    let source = PayloadSource::new(storage);
    let result = slots.and_then(|slots| {
        let rings_len = buffer.len() - scratch_len;
        let (mut line_rings, scratch) = buffer.split_at_mut(rings_len);
        let mut lines = Vec::new();
        lines
            .try_reserve_exact(line_plans.len())
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        for (plan, len) in line_plans.iter().zip(&ring_lens) {
            let (ring, tail) = core::mem::take(&mut line_rings).split_at_mut(*len);
            line_rings = tail;
            lines.push(plan.reconstruction(ring)?);
        }
        let run = |readers: &mut [&mut (dyn SubBandRows + Send)]| -> Result<()> {
            let written = output.write_batches(&mut lines, readers, scratch, batch);
            output_samples = lines.iter().map(LineReconstruction::output_samples).sum();
            written
        };
        match coding {
            CodeBlockCoding::Classic => run_bands(
                (workers, readers, helps),
                &rings,
                &bands,
                slots,
                (&mut tile_ctx.classic_task_workspaces, &mut scratches),
                |strip, output, workspace, scratch| {
                    let (route, strip) = route_strip(&routes, &strip_starts, strip)?;
                    route.decode_classic(strip, output, workspace, (scratch, source))
                },
                run,
                profile_enabled,
                &mut tier1_us,
                strand_once,
            ),
            CodeBlockCoding::HighThroughput => run_bands(
                (workers, readers, helps),
                &rings,
                &bands,
                slots,
                (&mut tile_ctx.ht_task_workspaces, &mut scratches),
                |strip, output, workspace, scratch| {
                    let (route, strip) = route_strip(&routes, &strip_starts, strip)?;
                    route.decode_ht(strip, output, workspace, (scratch, source))
                },
                run,
                profile_enabled,
                &mut tier1_us,
                strand_once,
            ),
        }
    });
    storage.coefficients = coefficients;
    // Each band zeroes its slot before it decodes, and the last bands leave
    // theirs written.
    storage.stale_coefficients = true;
    tile_ctx.idwt_line_rings = buffer;
    #[cfg(test)]
    {
        let counters = &mut tile_ctx.debug_counters;
        counters.thread_scratch_held += BlockScratch::held_bytes(&scratches);
        counters.thread_block_samples = counters
            .thread_block_samples
            .max(BlockScratch::block_samples(&scratches));
    }
    let streamed = output.streamed;
    tile_ctx.streamed_samples = tile_ctx.streamed_samples.saturating_add(streamed);
    tile_ctx.debug_counters.idwt_output_samples = tile_ctx
        .debug_counters
        .idwt_output_samples
        .saturating_add(output_samples);
    let elapsed = profile::elapsed_us(stage_start);
    profile_timings.codeblock_us += tier1_us;
    profile_timings.idwt_us += elapsed.saturating_sub(tier1_us);
    let result = result.and_then(|completed| {
        if completed {
            BandRoute::record_live(
                routes.iter().map(|route| &**route),
                threads,
                tile_ctx,
                storage,
            )?;
        }
        Ok(completed)
    });
    // Neither the serial bands' nor the whole-window rerun's budget includes
    // the plans, the executor's records or the threads' scratch, so they go
    // first.
    drop((
        routes,
        rings,
        bands,
        strip_starts,
        active_levels,
        line_plans,
        ring_lens,
        scratches,
    ));
    match result {
        Ok(true) => {
            tile_ctx.debug_counters.pool_band_components += components;
            record_workspace(tile_ctx, storage, 0)?;
            Ok(true)
        }
        Ok(false) => {
            // A band was left running with no worker to finish it.
            #[cfg(test)]
            let simulated = tile_ctx.strand_pool_band;
            #[cfg(not(test))]
            let simulated = false;
            debug_assert!(simulated, "a running band had no active worker");
            tile_ctx.debug_counters.pool_band_fallbacks += 1;
            // Rows the sink holds can't be taken back.
            if streamed > 0 {
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
            Ok(false)
        }
        Err(error) => Err(banded_tier1::whole_window_tile_error(
            tile, header, tile_ctx, storage, error,
        )),
    }
}

/// Whether a tile streaming its rows from pool bands may hold deeper band
/// rings (see `DecompositionStorage::deep_band_rings`): a classic tile of
/// several components on a pool of [`MIN_CLASSIC_SYNTHESIS_THREADS`] threads
/// or more, which reconstructs each on its own reader (see
/// [`parallel_synthesis`]). Its readers wait on classic Tier-1, which more
/// bands in flight keep busier when it reads whole rows (see
/// [`reads_whole_rows`]); it holds one batch of rows instead of two. HTJ2K
/// bands keep up with the shallower rings, and its readers gain more from
/// the second batch.
pub(super) fn deep_rings(tile: &Tile<'_>) -> bool {
    tile.component_infos.len() > 1
        && rayon::current_num_threads() >= MIN_CLASSIC_SYNTHESIS_THREADS
        && tile.component_infos.iter().all(|component| {
            !component
                .coding_style
                .parameters
                .code_block_style
                .uses_high_throughput_block_coding()
        })
}

/// Whether a tile reads whole rows at full resolution: no resolution level
/// skipped, and full-width windows (see [`reads_full_width`]). Only such
/// tiles hold deep band rings: a reduced or narrower read decodes as fast
/// without the extra band in every ring.
pub(super) fn reads_whole_rows(header: &Header<'_>, storage: &DecompositionStorage<'_>) -> bool {
    header.skipped_resolution_levels == 0 && reads_full_width(storage)
}

/// Whether every sub-band window a tile's IDWT reads spans its sub-band.
fn reads_full_width(storage: &DecompositionStorage<'_>) -> bool {
    storage.roi_plan.as_ref().is_some_and(|plan| {
        storage
            .sub_bands
            .iter()
            .enumerate()
            .all(|(index, sub_band)| {
                plan.sub_band_window(index)
                    .is_none_or(|window| window.width() == sub_band.rect.width())
            })
    })
}

/// Whether a tile streaming its rows from pool bands holds one band fewer in
/// each ring of its finest level (see
/// `DecompositionStorage::shallow_band_rings`): a classic tile of several
/// components that reconstructs each on its own reader without deep band
/// rings, as on 10 and 11 threads or at a reduced resolution, reading
/// full-width windows, on a pool with more threads than the tile has
/// finest-level sub-bands. Its readers wait on classic Tier-1. With two slots
/// the bands ahead of them are then fewer than the pool's threads, so the
/// bands split into parts (see [`plan_parts`]) that every thread takes, which
/// keeps the pool as busy as a band more in every ring did. Narrower regions
/// keep their rings.
pub(super) fn shallow_rings(tile: &Tile<'_>, storage: &DecompositionStorage<'_>) -> bool {
    let components = tile.component_infos.len();
    storage.parallel_bands
        && storage.lockstep_lines
        && !storage.deep_band_rings
        && !storage.one_synthesis_reader
        && components > 1
        && 3 * components < rayon::current_num_threads()
        && tile.component_infos.iter().all(|component| {
            !component
                .coding_style
                .parameters
                .code_block_style
                .uses_high_throughput_block_coding()
        })
        && reads_full_width(storage)
}

/// Whether a tile streaming its rows from pool bands reconstructs them
/// through one reader whatever workers its routes plan (see
/// [`parallel_synthesis`]): it has one component, or classic code-blocks on
/// too few pool threads for its size to reconstruct each component on its
/// own (see [`reader_per_component`]). The size is the largest final window
/// of a component, which holds the rows [`parallel_synthesis`] counts.
pub(super) fn one_reader(tile: &Tile<'_>, storage: &DecompositionStorage<'_>) -> bool {
    let components = tile.component_infos.len();
    let classic = tile.component_infos.iter().all(|component| {
        !component
            .coding_style
            .parameters
            .code_block_style
            .uses_high_throughput_block_coding()
    });
    let coding = if classic {
        CodeBlockCoding::Classic
    } else {
        CodeBlockCoding::HighThroughput
    };
    let samples = storage.roi_plan.as_ref().map_or(usize::MAX, |plan| {
        (0..components)
            .filter_map(|component_idx| plan.final_window(component_idx))
            .map(|window| (window.width() as usize).saturating_mul(window.height() as usize))
            .max()
            .unwrap_or(0)
    });
    !reader_per_component(components, coding, rayon::current_num_threads(), samples)
}

/// Pool threads from which a classic tile reconstructs each component on its
/// own reader whatever its size, and its band rings are deep.
const MIN_CLASSIC_SYNTHESIS_THREADS: usize = 12;

/// Pool threads from which a classic tile of [`MIN_CLASSIC_READER_SAMPLES`]
/// or more reconstructs each component on its own reader below
/// [`MIN_CLASSIC_SYNTHESIS_THREADS`]. A reader waiting for a band holds its
/// thread, and classic Tier-1 keeps readers waiting, so a smaller pool does
/// better with one reader that decodes bands while it waits (4-7% on 8
/// threads). From 10 threads that one reader, which reconstructs every
/// component and packs every row alone, bounds a large tile: a reader per
/// component decodes 18000 x 12000 RGB 11-19% and 5661 x 8287 RGB 8-13%
/// faster on 10 and 11 threads.
const MIN_CLASSIC_READER_THREADS: usize = 10;

/// Samples of each component from which a classic tile takes a reader per
/// component on [`MIN_CLASSIC_READER_THREADS`] threads. Smaller tiles spend
/// more of their time in Tier-1, from which the readers take two pool
/// threads: on 10 threads a 4096 x 2048 lossless RGB image and the 2830 x
/// 4143 to 708 x 1036 reduced decodes of a 5661 x 8287 one were 1-11% slower
/// with them.
#[cfg(not(test))]
const MIN_CLASSIC_READER_SAMPLES: usize = 1 << 24;
/// The unit tests' fixtures are far below the size, so their gate lies
/// between a 203 x 157 component and its 102 x 79 half.
#[cfg(test)]
const MIN_CLASSIC_READER_SAMPLES: usize = 1 << 14;

/// Whether a streamed tile's `components` reconstruct on the pool, one per
/// task, given its block coding, a workspace bank of `bank`, routes planned
/// for `route_workers` pool threads and `samples` reconstructed per component.
fn parallel_synthesis(
    components: usize,
    coding: CodeBlockCoding,
    bank: usize,
    route_workers: usize,
    samples: usize,
) -> bool {
    route_workers >= 2
        && bank > components
        && reader_per_component(components, coding, rayon::current_num_threads(), samples)
}

/// Whether a streamed tile of `components` of `coding` blocks may reconstruct
/// each component on its own reader on a pool of `threads`, with `samples`
/// reconstructed per component: HTJ2K tiles on any pool, classic ones from
/// [`MIN_CLASSIC_SYNTHESIS_THREADS`], or from [`MIN_CLASSIC_READER_THREADS`]
/// at [`MIN_CLASSIC_READER_SAMPLES`] or more. [`parallel_synthesis`] decides
/// by it and [`one_reader`] charges by it, so a tile charged for one reader
/// never runs several.
fn reader_per_component(
    components: usize,
    coding: CodeBlockCoding,
    threads: usize,
    samples: usize,
) -> bool {
    components > 1
        && (coding == CodeBlockCoding::HighThroughput
            || threads >= MIN_CLASSIC_SYNTHESIS_THREADS
            || (threads >= MIN_CLASSIC_READER_THREADS && samples >= MIN_CLASSIC_READER_SAMPLES))
}

/// The route of a band's `strip`, counted over every component's route from
/// `strip_starts`, and the strip in that route.
fn route_strip<'r, 'a>(
    routes: &'r [Box<BandRoute<'a>>],
    strip_starts: &[usize],
    strip: usize,
) -> Result<(&'r BandRoute<'a>, usize)> {
    let component = strip_starts
        .partition_point(|start| *start <= strip)
        .checked_sub(1)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let route = routes
        .get(component)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    Ok((route, strip - strip_starts[component]))
}

/// Every component's rings and bands, in component order, as one executor
/// takes them, with the first strip of each component's route among the
/// strips of all routes, and the levels each component reconstructs.
type TileRings = (Vec<Ring>, Vec<Band>, Vec<usize>, Vec<usize>);

fn plan_tile_rings(
    routes: &[Box<BandRoute<'_>>],
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
) -> Result<TileRings> {
    let (mut rings, mut bands) = (Vec::new(), Vec::new());
    let (mut ring_count, mut band_count) = (0usize, 0usize);
    for (component_idx, route) in routes.iter().enumerate() {
        let (component_rings, component_bands) =
            ring_counts(route, storage, header, component_idx)?;
        ring_count = ring_count.saturating_add(component_rings);
        band_count = band_count.saturating_add(component_bands);
    }
    rings
        .try_reserve_exact(ring_count)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    bands
        .try_reserve_exact(band_count)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut strip_starts = Vec::new();
    strip_starts
        .try_reserve_exact(routes.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut active_levels = Vec::new();
    active_levels
        .try_reserve_exact(routes.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut strips = 0usize;
    for (component_idx, route) in routes.iter().enumerate() {
        let (component_rings, component_bands, levels) =
            plan_rings(route, storage, header, component_idx)?;
        let (ring_offset, band_offset) = (rings.len(), bands.len());
        if component_rings.len() > rings.capacity() - ring_offset
            || component_bands.len() > bands.capacity() - band_offset
        {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
        rings.extend(component_rings.into_iter().map(|ring| Ring {
            bands: ring.bands.start + band_offset..ring.bands.end + band_offset,
            ..ring
        }));
        bands.extend(component_bands.into_iter().map(|band| Band {
            strip: band.strip + strips,
            ring: band.ring + ring_offset,
            ..band
        }));
        strip_starts.push(strips);
        strips += route.strips().count();
        active_levels.push(levels);
    }
    Ok((rings, bands, strip_starts, active_levels))
}

/// Plan every component's streamed reconstruction of its `active_levels`.
/// No plans when no component has rows to reconstruct; every component of
/// a 1x1-sampled tile reconstructs the same rows.
fn plan_tile_lines(
    tile: &Tile<'_>,
    header: &Header<'_>,
    tile_ctx: &TileDecodeContext,
    storage: &DecompositionStorage<'_>,
    active_levels: &[usize],
) -> Result<Vec<LinePlan>> {
    let mut line_plans = Vec::new();
    line_plans
        .try_reserve_exact(active_levels.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    for (component_idx, (component_info, levels)) in
        tile.component_infos.iter().zip(active_levels).enumerate()
    {
        if let Some(plan) = LinePlan::new(
            storage,
            tile_ctx,
            component_idx,
            *levels,
            header,
            component_info.wavelet_transform(),
        )? {
            line_plans.push(plan);
        }
    }
    if line_plans.is_empty() {
        return Ok(line_plans);
    }
    if line_plans.len() != tile.component_infos.len()
        || line_plans
            .iter()
            .any(|plan| plan.copy != line_plans[0].copy || plan.region != line_plans[0].region)
    {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    Ok(line_plans)
}

/// The bytes the executor holds for `rings` and `bands` while `workers`
/// pool threads and `readers` readers decode them, besides the rings and
/// bands themselves (see [`ring_plan_bytes`]) and the slots' samples: each
/// band's status, held and owned slot, part counters, decoded-list entry
/// and shared output; a queue entry for each part of each band, as many as
/// its parts put in it; each ring's cursors, slot list and slots; each
/// reader's view, with its band being read and next band in every ring, and
/// its entry in the readers' list; each thread's workspace and scratch
/// pair; the pool's job for each worker running or queued, and for each
/// pool thread whose worker finished before the pool freed its job; what
/// the shared output of each band decoding in parts, the shared state's
/// lock and its condition variable allocate when first used; and what the
/// pool takes when entered from outside it.
fn executor_bytes(
    rings: &[Ring],
    bands: &[Band],
    (workers, readers): (usize, usize),
) -> Result<usize> {
    let per_band = size_of::<Status>()
        + 3 * size_of::<usize>()
        + 2 * size_of::<Option<&mut [f32]>>()
        + size_of::<Mutex<&mut [f32]>>();
    let per_ring = 2 * size_of::<usize>() + size_of::<Vec<&mut [f32]>>();
    // A view holds references to its workspace, scratch and decode,
    // whatever their types.
    let per_reader = rings
        .len()
        .checked_mul(size_of::<usize>() + size_of::<Option<(usize, &mut [f32])>>())
        .ok_or(ValidationError::ImageTooLarge)?
        + size_of::<PoolRows<'_, '_, '_, '_, (), ()>>()
        + size_of::<&mut (dyn SubBandRows + Send)>();
    let per_thread = size_of::<(&mut (), &mut BlockScratch)>();
    let slots: usize = rings.iter().map(|ring| ring.slots).sum();
    [
        bands.len().checked_mul(per_band),
        queue_entries(bands).checked_mul(size_of::<QueueEntry>()),
        rings.len().checked_mul(per_ring),
        slots.checked_mul(size_of::<&mut [f32]>()),
        readers.checked_mul(per_reader),
        workers
            .checked_add(readers)
            .and_then(|threads| threads.checked_mul(per_thread)),
        workers
            .checked_add(rayon::current_num_threads())
            .and_then(|jobs| jobs.checked_mul(SENT_WORKER_BYTES)),
        // Called from outside the pool, its workers queue on the pool's
        // shared queue, and this thread waits for them to finish.
        Some(pool_entry_bytes()),
        bands
            .iter()
            .filter(|band| band.parts > 1)
            .count()
            .checked_add(2)
            .and_then(|locks| locks.checked_mul(LOCK_BYTES)),
    ]
    .into_iter()
    .try_fold(0usize, |bytes, part| bytes.checked_add(part?))
    .ok_or(ValidationError::ImageTooLarge.into())
}

/// An entry of the executor's queue of bands that can start or hand out a
/// part: soonest read first, then largest first, and the band.
type QueueEntry = Reverse<(i64, Reverse<usize>, usize)>;

/// The most entries the executor's queue holds for `bands`: a band enters
/// it once when its ring has a slot for it and again for each part after
/// one it hands out, and a reader may start a band without taking its
/// entry, so a band puts as many entries in it as it has parts.
fn queue_entries(bands: &[Band]) -> usize {
    bands
        .iter()
        .fold(0usize, |entries, band| entries.saturating_add(band.parts))
}

/// Bytes of the job the pool boxes for each worker the executor sends: the
/// worker's closure, its two references, with the scope it runs in and the
/// job's context.
const SENT_WORKER_BYTES: usize = 4 * size_of::<usize>();

/// The bytes the rings and bands of the component of `route` hold once
/// planned, and the rows' scales planning them holds beside them.
fn ring_plan_bytes(
    route: &BandRoute<'_>,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    component_idx: usize,
) -> Result<(usize, usize)> {
    let (rings, bands) = ring_counts(route, storage, header, component_idx)?;
    let planned = rings
        .checked_mul(size_of::<Ring>())
        .zip(bands.checked_mul(size_of::<Band>()))
        .and_then(|(rings, bands)| rings.checked_add(bands))
        .ok_or(ValidationError::ImageTooLarge)?;
    let scales = rings
        .checked_mul(size_of::<i64>())
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok((planned, scales))
}

/// The bytes every route's rings and bands hold once [`plan_tile_rings`]
/// planned them, with each component's first strip and levels, and the
/// most it holds beside them while it plans one component: that one's own
/// rings, bands and row scales.
fn tile_ring_plan_bytes(
    routes: &[Box<BandRoute<'_>>],
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
) -> Result<(usize, usize)> {
    let mut planned = routes
        .len()
        .checked_mul(2 * size_of::<usize>())
        .ok_or(ValidationError::ImageTooLarge)?;
    let mut planning = 0usize;
    for (component_idx, route) in routes.iter().enumerate() {
        let (own, scales) = ring_plan_bytes(route, storage, header, component_idx)?;
        planned = planned
            .checked_add(own)
            .ok_or(ValidationError::ImageTooLarge)?;
        planning = planning.max(own.saturating_add(scales));
    }
    Ok((planned, planning))
}

/// The rings and bands [`plan_rings`] plans for the component of `route`:
/// one ring for each sub-band it reconstructs, one band for each strip.
fn ring_counts(
    route: &BandRoute<'_>,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    component_idx: usize,
) -> Result<(usize, usize)> {
    let levels = active_levels(storage, header, component_idx)?;
    let rings = levels
        .checked_mul(3)
        .and_then(|rings| rings.checked_add(1))
        .ok_or(ValidationError::ImageTooLarge)?;
    Ok((rings, route.strips().count()))
}

/// The levels the component reconstructs: its decompositions, but the
/// resolution levels the decode skips.
fn active_levels(
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    component_idx: usize,
) -> Result<usize> {
    storage
        .tile_decompositions
        .get(component_idx)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?
        .decompositions
        .len()
        .checked_sub(usize::from(header.skipped_resolution_levels))
        .filter(|levels| *levels > 0)
        .ok_or_else(|| DecodingError::CodeBlockDecodeFailure.into())
}

/// A sub-band's ring of band slots and its bands, in row order.
struct Ring {
    sub_band: usize,
    stored: IntRect,
    slots: usize,
    /// Whether the sub-band belongs to the finest level the IDWT reads.
    finest: bool,
    /// Whether the ring holds the sub-band's whole window, because its bands
    /// fit in fewer rows than its slots would: each band then decodes into
    /// its own rows of the window, and all of them can be in flight at once.
    whole_window: bool,
    bands: Range<usize>,
}

/// A band: one strip of the route, in its ring's order.
struct Band {
    strip: usize,
    ring: usize,
    /// The band's rows, in absolute sub-band coordinates.
    rows: Range<u32>,
    /// Where the IDWT first reads the band, in rows of the reconstructed
    /// component, approximately: bands it reads sooner start sooner.
    order: i64,
    work: usize,
    /// Runs of its blocks the band decodes in, on as many threads at once.
    parts: usize,
    /// The count its blocks were split for, which gives those `parts` runs.
    split: usize,
}

/// The component's rings, its bands grouped by ring in row order, and the
/// number of levels it reconstructs.
fn plan_rings(
    route: &BandRoute<'_>,
    storage: &DecompositionStorage<'_>,
    header: &Header<'_>,
    component_idx: usize,
) -> Result<(Vec<Ring>, Vec<Band>, usize)> {
    let tile_decompositions = &storage.tile_decompositions[component_idx];
    let active_levels = active_levels(storage, header, component_idx)?;
    let mut rings = Vec::new();
    rings
        .try_reserve_exact(1 + 3 * active_levels)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    // A sub-band row at resolution `r` spans 2^(levels + 1 - r) rows of the
    // reconstructed component, and the LL band's 2^levels.
    let mut scales = Vec::new();
    scales
        .try_reserve_exact(rings.capacity())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    for resolution in 0..=active_levels {
        let level = u8::try_from(resolution).map_err(|_| DecodingError::CodeBlockDecodeFailure)?;
        let shift = if resolution == 0 {
            active_levels
        } else {
            active_levels + 1 - resolution
        };
        for sub_band in tile_decompositions.sub_band_iter(level, &storage.decompositions) {
            let stored = storage.sub_bands[sub_band].storage_rect;
            let coefficients = storage.sub_bands[sub_band].coefficients.len();
            let band_len = route
                .strips()
                .filter(|strip| strip.sub_band == sub_band)
                .map(|strip| strip.rows * stored.width() as usize)
                .max()
                .unwrap_or(0);
            let window = stored.width() as usize * stored.height() as usize;
            let whole_window = band_len > 0 && coefficients >= window;
            let slots = coefficients.checked_div(band_len).unwrap_or(0);
            rings.push(Ring {
                sub_band,
                stored,
                slots,
                finest: resolution == active_levels,
                whole_window,
                bands: 0..0,
            });
            scales.push(1_i64 << shift.min(62));
        }
    }
    let mut bands = Vec::new();
    bands
        .try_reserve_exact(route.strips().count())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    for (index, ring) in rings.iter_mut().enumerate() {
        let start = bands.len();
        for (strip_index, strip) in route
            .strips()
            .enumerate()
            .filter(|(_, strip)| strip.sub_band == ring.sub_band)
        {
            let first = u32::try_from(strip.first_row)
                .ok()
                .and_then(|row| ring.stored.y0.checked_add(row))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let last = u32::try_from(strip.rows)
                .ok()
                .and_then(|rows| first.checked_add(rows))
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            bands.push(Band {
                strip: strip_index,
                ring: index,
                rows: first..last,
                order: (i64::from(first) - SUPPORT_ROWS) * scales[index],
                work: strip.work,
                parts: 1,
                split: 1,
            });
        }
        bands[start..].sort_unstable_by_key(|band| band.rows.start);
        ring.bands = start..bands.len();
        if ring.whole_window {
            ring.slots = ring.bands.len();
        }
        if !ring.bands.is_empty() && ring.slots == 0 {
            return Err(DecodingError::CodeBlockDecodeFailure.into());
        }
    }
    if bands.len() != route.strips().count() {
        // A strip of a sub-band outside the component.
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    Ok((rings, bands, active_levels))
}

/// Count the deepest of `rings` that hold no whole window, at the finest
/// level and at the others, in the decode's debug counters.
#[cfg(test)]
fn count_ring_slots(tile_ctx: &mut TileDecodeContext, rings: &[Ring]) {
    let counters = &mut tile_ctx.debug_counters;
    for ring in rings.iter().filter(|ring| !ring.whole_window) {
        let deepest = if ring.finest {
            &mut counters.pool_band_finest_slots
        } else {
            &mut counters.pool_band_other_slots
        };
        *deepest = (*deepest).max(ring.slots);
    }
}

/// Most parts a band splits into.
const MAX_BAND_PARTS: usize = 8;

/// Parts of finest-level bands that a split aims at per thread: with two, a
/// thread that finishes a part finds another while the band's last parts
/// finish. One part per thread left 5792 x 5792 gray 14% slower (80 against
/// 70 ms on 18 threads).
const BAND_PARTS_PER_THREAD: usize = 2;

/// Split a classic tile's bands into parts for `threads` threads (see
/// [`plan_parts`]), and count the bands that split.
fn split_bands(
    tile_ctx: &mut TileDecodeContext,
    rings: &[Ring],
    bands: &mut [Band],
    threads: usize,
    split: impl Fn(&Band, usize) -> Result<usize>,
) -> Result<()> {
    #[cfg(test)]
    let forced = tile_ctx.band_parts;
    #[cfg(not(test))]
    let forced = None;
    plan_parts(rings, bands, threads, forced, split)?;
    tile_ctx.debug_counters.pool_band_parts += bands.iter().filter(|band| band.parts > 1).count();
    Ok(())
}

/// Split `bands` into parts, through `split` (which returns the runs a band
/// splits into for a wanted count), when the finest level's slots ahead of
/// the bands being read hold fewer bands than `threads`: each finest band
/// into as many parts as give [`BAND_PARTS_PER_THREAD`] parts per thread, and
/// every other band into parts of about the same work. A tile whose rings
/// hold bands enough keeps them whole. `forced` splits every band into that
/// many parts instead.
fn plan_parts(
    rings: &[Ring],
    bands: &mut [Band],
    threads: usize,
    forced: Option<usize>,
    split: impl Fn(&Band, usize) -> Result<usize>,
) -> Result<()> {
    if let Some(parts) = forced {
        for band in bands {
            band.split = parts.max(1);
            band.parts = split(band, band.split)?;
        }
        return Ok(());
    }
    let ahead: usize = rings
        .iter()
        .filter(|ring| ring.finest && !ring.whole_window)
        .map(|ring| ring.slots.saturating_sub(1))
        .sum();
    if ahead == 0 || ahead >= threads {
        return Ok(());
    }
    let parts = threads
        .saturating_mul(BAND_PARTS_PER_THREAD)
        .div_ceil(ahead)
        .min(MAX_BAND_PARTS);
    let (finest_work, finest_bands) = rings
        .iter()
        .filter(|ring| ring.finest)
        .flat_map(|ring| &bands[ring.bands.clone()])
        .fold((0usize, 0usize), |(work, count), band| {
            (work.saturating_add(band.work), count + 1)
        });
    let part_work = (finest_work / finest_bands.max(1) / parts).max(1);
    for band in bands {
        let wanted = band.work.div_ceil(part_work).clamp(1, parts);
        if wanted > 1 {
            band.split = wanted;
            band.parts = split(band, wanted)?;
        }
    }
    Ok(())
}

/// Each ring's free slots and the slot each band of a whole-window ring
/// owns.
type Slots<'c> = (Vec<Vec<&'c mut [f32]>>, Vec<Option<&'c mut [f32]>>);

/// Split each ring's coefficients into its slots: equal slots as long as its
/// longest band, or in a whole-window ring, each band's own rows.
fn split_slots<'c>(
    rings: &[Ring],
    bands: &[Band],
    storage: &DecompositionStorage<'_>,
    coefficients: &'c mut [f32],
) -> Result<Slots<'c>> {
    let mut slots = Vec::new();
    slots
        .try_reserve_exact(rings.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let mut owned = try_filled_with(bands.len(), || None)?;
    // Sub-band ranges of a component don't overlap and ascend in ring order.
    let mut rest = coefficients;
    let mut consumed = 0usize;
    for ring in rings {
        let range = storage.sub_bands[ring.sub_band].coefficients.clone();
        let mut ring_slots = Vec::new();
        ring_slots
            .try_reserve_exact(ring.slots)
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        if ring.slots > 0 {
            let skip = range
                .start
                .checked_sub(consumed)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let tail = core::mem::take(&mut rest);
            let (_, tail) = tail
                .split_at_mut_checked(skip)
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            let (mut ring_samples, tail) = tail
                .split_at_mut_checked(range.len())
                .ok_or(DecodingError::CodeBlockDecodeFailure)?;
            rest = tail;
            consumed = range.end;
            if ring.whole_window {
                // Bands of a sub-band cover disjoint rows, in order.
                let width = ring.stored.width() as usize;
                let mut offset = 0usize;
                for index in ring.bands.clone() {
                    let rows = &bands[index].rows;
                    let start = (rows.start - ring.stored.y0) as usize * width;
                    let len = (rows.end - rows.start) as usize * width;
                    let gap = start
                        .checked_sub(offset)
                        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                    let (_, tail) = core::mem::take(&mut ring_samples)
                        .split_at_mut_checked(gap)
                        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                    let (slot, tail) = tail
                        .split_at_mut_checked(len)
                        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                    owned[index] = Some(slot);
                    ring_samples = tail;
                    offset = start + len;
                }
            } else {
                // Each slot holds whole rows of the sub-band, as many as the
                // ring's share of them: at least its longest band's, which
                // can be fewer than the rows the ring was laid out for.
                let width = ring.stored.width() as usize;
                let slot_len = range
                    .len()
                    .checked_div(ring.slots)
                    .and_then(|len| len.checked_div(width))
                    .and_then(|rows| rows.checked_mul(width))
                    .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                for _ in 0..ring.slots {
                    let (slot, tail) = core::mem::take(&mut ring_samples).split_at_mut(slot_len);
                    ring_slots.push(slot);
                    ring_samples = tail;
                }
            }
        }
        slots.push(ring_slots);
    }
    Ok((slots, owned))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Not started; ready to start once its ring has a free slot.
    Waiting,
    /// Being decoded.
    Running,
    /// Decoded, holding its slot until the IDWT has read it.
    Done,
    /// Read, its slot back in its ring.
    Freed,
}

/// What the IDWT and the workers share, behind one lock.
struct State<'c, 'w, W> {
    status: Vec<Status>,
    /// The slot of each band that holds one and isn't in a thread's hands.
    held: Vec<Option<&'c mut [f32]>>,
    free_slots: Vec<Vec<&'c mut [f32]>>,
    /// The rows of the window each band of a whole-window ring decodes into,
    /// until it starts.
    owned: Vec<Option<&'c mut [f32]>>,
    /// Bands that can start, or hand out another part, soonest read first,
    /// then largest first. Entries for bands with nothing left to hand out
    /// are skipped.
    ready: BinaryHeap<QueueEntry>,
    /// Each ring's next band to make ready.
    next_ready: Vec<usize>,
    /// Parts of each band handed out, and parts still decoding or to hand
    /// out.
    next_part: Vec<usize>,
    parts_left: Vec<usize>,
    /// How many of each ring's bands, from its first, have been freed.
    freed_prefix: Vec<usize>,
    /// Bands that finished decoding, for the drain to free; bands the IDWT
    /// has since read are skipped.
    decoded: Vec<usize>,
    running: usize,
    freed: usize,
    workspaces: Vec<(&'w mut W, &'w mut BlockScratch)>,
    /// Workers on the pool.
    active: usize,
    error: Option<DecodeError>,
    /// Whether a wait found a running band with no worker active.
    stranded: bool,
}

struct Shared<'p, 'c, 'w, W> {
    rings: &'p [Ring],
    bands: &'p [Band],
    /// The slot of each band decoding in parts, which its parts share.
    outputs: Vec<Mutex<&'c mut [f32]>>,
    max_workers: usize,
    /// Test-only: the first worker to start a band abandons it.
    #[cfg(test)]
    strand_once: core::sync::atomic::AtomicBool,
    state: Mutex<State<'c, 'w, W>>,
    /// Signalled when a band is decoded or a decode fails.
    decoded: Condvar,
}

impl<'c, 'w, W> Shared<'_, 'c, 'w, W> {
    fn lock(&self) -> Result<MutexGuard<'_, State<'c, 'w, W>>> {
        self.state
            .lock()
            .map_err(|_| DecodingError::CodeBlockDecodeFailure.into())
    }

    /// Make ring `ring`'s next bands ready while it has free slots.
    fn refill(&self, state: &mut State<'c, 'w, W>, ring: usize) {
        let ring_bands = &self.rings[ring].bands;
        while state.next_ready[ring] < ring_bands.end
            && state.next_ready[ring] - ring_bands.start
                < state.freed_prefix[ring] + self.rings[ring].slots
        {
            let index = state.next_ready[ring];
            let band = &self.bands[index];
            state
                .ready
                .push(Reverse((band.order, Reverse(band.work), index)));
            state.next_ready[ring] += 1;
        }
    }

    /// Take the next ready band, marked running, with a slot to decode
    /// into, or the next part of a running band.
    fn start_next(&self, state: &mut State<'c, 'w, W>) -> Result<Option<Task<'c>>> {
        while let Some(Reverse((_, _, index))) = state.ready.pop() {
            match state.status[index] {
                Status::Waiting => return self.start(state, index).map(Some),
                Status::Running if self.has_part(state, index) => {
                    return Ok(Some(self.next_part(state, index)));
                }
                _ => {}
            }
        }
        Ok(None)
    }

    fn start(&self, state: &mut State<'c, 'w, W>, index: usize) -> Result<Task<'c>> {
        let slot = match state.owned[index].take() {
            Some(slot) => slot,
            None => state.free_slots[self.bands[index].ring]
                .pop()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
        };
        state.status[index] = Status::Running;
        state.running += 1;
        let parts = self.bands[index].parts;
        if parts <= 1 {
            return Ok(Task::Whole(index, slot));
        }
        *self.output(index)? = slot;
        state.parts_left[index] = parts;
        Ok(self.next_part(state, index))
    }

    /// Whether running band `index` decodes in parts and has parts left to
    /// hand out.
    fn has_part(&self, state: &State<'c, 'w, W>, index: usize) -> bool {
        let parts = self.bands[index].parts;
        parts > 1 && state.next_part[index] < parts
    }

    /// Hand out the next part of running band `index`, leaving the band
    /// ready while it has more.
    fn next_part(&self, state: &mut State<'c, 'w, W>, index: usize) -> Task<'c> {
        let part = state.next_part[index];
        state.next_part[index] += 1;
        if self.has_part(state, index) {
            let band = &self.bands[index];
            state
                .ready
                .push(Reverse((band.order, Reverse(band.work), index)));
        }
        Task::Part(index, part)
    }

    fn output(&self, index: usize) -> Result<MutexGuard<'_, &'c mut [f32]>> {
        self.outputs[index]
            .lock()
            .map_err(|_| DecodingError::CodeBlockDecodeFailure.into())
    }

    /// Record a part of band `index`, decoded outside the lock; the last
    /// part to finish leaves the band decoded.
    fn finish_part(&self, index: usize, result: Result<()>) -> Result<()> {
        let mut state = self.lock()?;
        match result {
            Ok(()) => {
                state.parts_left[index] -= 1;
                if state.parts_left[index] == 0 {
                    let slot = core::mem::take(&mut *self.output(index)?);
                    state.held[index] = Some(slot);
                    Self::decoded(&mut state, index);
                }
            }
            Err(error) => {
                state.error.get_or_insert(error);
            }
        }
        drop(state);
        self.decoded.notify_all();
        Ok(())
    }

    /// Band `index`, running, has decoded.
    fn decoded(state: &mut State<'c, 'w, W>, index: usize) {
        state.status[index] = Status::Done;
        state.running -= 1;
        state.decoded.push(index);
    }

    /// Record band `index`'s decode, which its decoder ran outside the lock.
    fn finish(&self, index: usize, slot: &'c mut [f32], result: Result<()>) -> Result<()> {
        let mut state = self.lock()?;
        match result {
            Ok(()) => {
                state.held[index] = Some(slot);
                Self::decoded(&mut state, index);
            }
            Err(error) => {
                state.error.get_or_insert(error);
            }
        }
        drop(state);
        self.decoded.notify_all();
        Ok(())
    }

    /// Return band `index`'s slot to its ring and make the ring's next band
    /// ready.
    fn free(&self, state: &mut State<'c, 'w, W>, index: usize, slot: &'c mut [f32]) {
        let ring = self.bands[index].ring;
        // A whole-window ring's slots belong to their bands.
        if !self.rings[ring].whole_window {
            state.free_slots[ring].push(slot);
        }
        state.status[index] = Status::Freed;
        state.freed += 1;
        let ring_bands = &self.rings[ring].bands;
        while ring_bands.start + state.freed_prefix[ring] < ring_bands.end
            && state.status[ring_bands.start + state.freed_prefix[ring]] == Status::Freed
        {
            state.freed_prefix[ring] += 1;
        }
        self.refill(state, ring);
    }
}

/// A thread's next decode: a band of one part, straight into its slot, or
/// part `.1` of band `.0`, into the slot its parts share.
enum Task<'c> {
    Whole(usize, &'c mut [f32]),
    Part(usize, usize),
}

/// Decode `bands` into `slots` on up to `workers` pool threads while `run`
/// drives the IDWT through `readers` views of them, then decode the bands it
/// didn't read. Each view reads its own sub-bands and decodes a band nobody
/// has started with a workspace and scratch of its own, so views may run on
/// different threads. Views that `help` decode other bands while they wait.
/// Returns `false` when a band was stranded.
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the executor takes the plan, its storage and its two drivers explicitly, and sets up its shared state and runs its views in one scope"
)]
fn run_bands<W: TaskScratch>(
    (workers, readers, helps): (usize, usize, bool),
    rings: &[Ring],
    bands: &[Band],
    slots: Slots<'_>,
    (bank, scratches): (&mut [TaskWorkspace<W>], &mut [BlockScratch]),
    decode: impl Fn(usize, BandOutput<'_, '_>, &mut W, &mut BlockScratch) -> Result<()> + Sync,
    run: impl FnOnce(&mut [&mut (dyn SubBandRows + Send)]) -> Result<()>,
    profile_enabled: bool,
    tier1_us: &mut u128,
    strand_once: bool,
) -> Result<bool> {
    #[cfg(not(test))]
    let _ = strand_once;
    let workspaces_needed = workers + readers.max(1);
    let mut workspaces: Vec<(&mut W, &mut BlockScratch)> = Vec::new();
    workspaces
        .try_reserve_exact(workspaces_needed)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    workspaces.extend(
        bank.iter_mut()
            .map(|slot| &mut slot.workspace)
            .zip(scratches.iter_mut())
            .take(workspaces_needed),
    );
    if workspaces.len() != workspaces_needed {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let reader_workspaces = workspaces.split_off(workers);
    let mut state = State {
        status: try_filled(bands.len(), Status::Waiting)?,
        held: try_filled_with(bands.len(), || None)?,
        free_slots: slots.0,
        owned: slots.1,
        ready: BinaryHeap::new(),
        next_ready: try_collect(rings.iter().map(|ring| ring.bands.start), rings.len())?,
        next_part: try_filled(bands.len(), 0)?,
        parts_left: try_filled(bands.len(), 0)?,
        freed_prefix: try_filled(rings.len(), 0)?,
        decoded: Vec::new(),
        running: 0,
        freed: 0,
        workspaces,
        active: 0,
        error: None,
        stranded: false,
    };
    state
        .ready
        .try_reserve_exact(queue_entries(bands))
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    state
        .decoded
        .try_reserve_exact(bands.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let shared = Shared {
        rings,
        bands,
        outputs: try_filled_with(bands.len(), || Mutex::new(<&mut [f32]>::default()))?,
        max_workers: workers,
        #[cfg(test)]
        strand_once: core::sync::atomic::AtomicBool::new(strand_once),
        state: Mutex::new(state),
        decoded: Condvar::new(),
    };
    // Where locks allocate when first used, the state's is first locked
    // below, and the condition variable is first used here, before any
    // worker races to allocate it.
    shared.decoded.notify_all();
    {
        let mut state = shared.lock()?;
        for ring in 0..rings.len() {
            shared.refill(&mut state, ring);
        }
    }
    let decode = &decode;
    let shared = &shared;
    let scoped = rayon::in_place_scope(|scope| -> Result<()> {
        let mut views = Vec::new();
        views
            .try_reserve_exact(reader_workspaces.len())
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        for (workspace, scratch) in reader_workspaces {
            views.push(PoolRows {
                shared,
                scope,
                decode,
                workspace,
                scratch,
                current: try_filled_with(rings.len(), || None)?,
                next: try_collect(rings.iter().map(|ring| ring.bands.start), rings.len())?,
                helps,
                profile_enabled,
                tier1_us: 0,
            });
        }
        views
            .first()
            .ok_or(DecodingError::CodeBlockDecodeFailure)?
            .send_workers()?;
        let mut readers: Vec<&mut (dyn SubBandRows + Send)> = Vec::new();
        readers
            .try_reserve_exact(views.len())
            .map_err(|_| DecodingError::HostAllocationFailed)?;
        readers.extend(
            views
                .iter_mut()
                .map(|view| view as &mut (dyn SubBandRows + Send)),
        );
        let result = run(&mut readers).and_then(|()| {
            // Every view hands its bands back before the first drains.
            for view in &mut views[1..] {
                view.release_all()?;
            }
            views[0].drain()
        });
        *tier1_us = views.iter().map(|view| view.tier1_us).sum();
        if result.is_err() {
            // Let the workers stop at their next band.
            if let Ok(mut state) = shared.lock() {
                state.ready.clear();
            }
        }
        result
    });
    let mut state = shared.lock()?;
    if state.stranded {
        return Ok(false);
    }
    let error = state.error.take();
    drop(state);
    scoped?;
    error.map_or(Ok(true), Err)
}

fn try_filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    values.resize(len, value);
    Ok(values)
}

fn try_collect<T>(values: impl Iterator<Item = T>, len: usize) -> Result<Vec<T>> {
    let mut collected = Vec::new();
    collected
        .try_reserve_exact(len)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    collected.extend(values);
    Ok(collected)
}

fn try_filled_with<T>(len: usize, value: impl FnMut() -> T) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    values.extend(core::iter::repeat_with(value).take(len));
    Ok(values)
}

/// Run `task` with `workspace` and `scratch`, and record it.
fn run_task<'c, W>(
    shared: &Shared<'_, 'c, '_, W>,
    decode: &(impl Fn(usize, BandOutput<'_, '_>, &mut W, &mut BlockScratch) -> Result<()> + Sync),
    task: Task<'c>,
    workspace: &mut W,
    scratch: &mut BlockScratch,
) -> Result<()> {
    match task {
        Task::Whole(index, slot) => {
            slot.fill(0.0);
            let strip = shared.bands[index].strip;
            let result = decode(strip, BandOutput::Whole(&mut *slot), workspace, scratch);
            shared.finish(index, slot, result)
        }
        Task::Part(index, part) => {
            let band = &shared.bands[index];
            let output = BandOutput::Part {
                part,
                parts: band.split,
                slot: &shared.outputs[index],
            };
            let result = decode(band.strip, output, workspace, scratch);
            shared.finish_part(index, result)
        }
    }
}

/// A pool worker: decode ready bands until none is left to start.
fn work<W: TaskScratch>(
    shared: &Shared<'_, '_, '_, W>,
    decode: &(impl Fn(usize, BandOutput<'_, '_>, &mut W, &mut BlockScratch) -> Result<()> + Sync),
) {
    let Ok(mut state) = shared.lock() else {
        return;
    };
    let Some((workspace, scratch)) = state.workspaces.pop() else {
        state.active -= 1;
        return;
    };
    loop {
        let next = if state.error.is_some() {
            Ok(None)
        } else {
            shared.start_next(&mut state)
        };
        let task = match next {
            Ok(Some(task)) => task,
            Ok(None) => break,
            Err(error) => {
                state.error.get_or_insert(error);
                break;
            }
        };
        drop(state);
        #[cfg(test)]
        if shared
            .strand_once
            .swap(false, core::sync::atomic::Ordering::Relaxed)
        {
            // Leave the band running, as the stranding bug would.
            if let Ok(mut state) = shared.lock() {
                state.active -= 1;
            }
            return;
        }
        if run_task(shared, decode, task, workspace, scratch).is_err() {
            return;
        }
        state = match shared.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
    }
    state.workspaces.push((workspace, scratch));
    state.active -= 1;
    drop(state);
    shared.decoded.notify_all();
}

/// The IDWT's view of the bands: each sub-band's current band, which it
/// holds while it reads it.
struct PoolRows<'s, 'p, 'c, 'w, W, D> {
    shared: &'p Shared<'p, 'c, 'w, W>,
    scope: &'s rayon::Scope<'p>,
    decode: &'p D,
    workspace: &'w mut W,
    scratch: &'w mut BlockScratch,
    /// Each ring's band being read, with its slot.
    current: Vec<Option<(usize, &'c mut [f32])>>,
    /// Each ring's next band to read.
    next: Vec<usize>,
    /// Whether a wait for a running band first decodes the next band
    /// nobody has started.
    helps: bool,
    profile_enabled: bool,
    tier1_us: u128,
}

impl<'p, 'c, W, D> PoolRows<'_, 'p, 'c, '_, W, D>
where
    W: TaskScratch + 'p,
    D: Fn(usize, BandOutput<'_, '_>, &mut W, &mut BlockScratch) -> Result<()> + Sync,
    'c: 'p,
{
    /// Send pool workers, up to the plan's count, while bands are ready.
    fn send_workers(&self) -> Result<()> {
        let sent = {
            let mut state = self.shared.lock()?;
            let wanted = self.shared.max_workers.min(state.ready.len());
            let sent = wanted.saturating_sub(state.active);
            state.active += sent;
            sent
        };
        for _ in 0..sent {
            let (shared, decode) = (self.shared, self.decode);
            self.scope.spawn(move |_| work(shared, decode));
        }
        Ok(())
    }

    fn ring(&self, sub_band: usize) -> Result<usize> {
        self.shared
            .rings
            .iter()
            .position(|ring| ring.sub_band == sub_band)
            .ok_or(DecodingError::CodeBlockDecodeFailure.into())
    }

    /// Hand back the band ring `ring` is reading.
    fn release(&mut self, ring: usize) -> Result<()> {
        if let Some((index, slot)) = self.current[ring].take() {
            let mut state = self.shared.lock()?;
            self.shared.free(&mut state, index, slot);
            drop(state);
            self.send_workers()?;
        }
        Ok(())
    }

    /// Band `index`, decoded, with its slot: wait for a worker decoding it,
    /// or decode it here if nobody has started it.
    fn acquire(&mut self, index: usize) -> Result<&'c mut [f32]> {
        let mut state = self.shared.lock()?;
        loop {
            // The error stays for the other readers, whose bands it may
            // have left running.
            if let Some(error) = state.error {
                return Err(error);
            }
            let task = match state.status[index] {
                Status::Done => {
                    return state.held[index]
                        .take()
                        .ok_or(DecodingError::CodeBlockDecodeFailure.into());
                }
                Status::Freed => return Err(DecodingError::CodeBlockDecodeFailure.into()),
                // A band nobody has started, or its first part, decodes
                // here; its other parts go to the pool.
                Status::Waiting => Some(self.shared.start(&mut state, index)?),
                // A reader decodes a part of this band nobody has started,
                // and one that helps the next band or part nobody has
                // started, while this one runs.
                Status::Running if self.shared.has_part(&state, index) => {
                    Some(self.shared.next_part(&mut state, index))
                }
                Status::Running if self.helps => self.shared.start_next(&mut state)?,
                Status::Running => None,
            };
            if let Some(task) = task {
                // A reader decoding a band counts as an active worker for
                // anyone waiting on it.
                state.active += 1;
                drop(state);
                self.send_workers()?;
                let decoded = self.decode_for_pool(task);
                state = self.shared.lock()?;
                state.active -= 1;
                decoded?;
                continue;
            }
            state = self
                .shared
                .decoded
                .wait_timeout(state, STRANDED_CHECK)
                .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                .0;
            // A failed band stays running; its error ends the wait.
            if state.error.is_none() && state.status[index] == Status::Running && state.active == 0
            {
                state.stranded = true;
                drop(state);
                return Err(DecodingError::CodeBlockDecodeFailure.into());
            }
        }
    }

    /// Decode `task`, a band or part this reader, another one or the drain
    /// will take, and hand it over as a worker does.
    fn decode_for_pool(&mut self, task: Task<'c>) -> Result<()> {
        let start = profile::profile_now(self.profile_enabled);
        let result = run_task(self.shared, self.decode, task, self.workspace, self.scratch);
        self.tier1_us += profile::elapsed_us(start);
        result
    }

    fn decode_here(&mut self, index: usize, slot: &mut [f32]) -> Result<()> {
        let start = profile::profile_now(self.profile_enabled);
        slot.fill(0.0);
        let result = (self.decode)(
            self.shared.bands[index].strip,
            BandOutput::Whole(slot),
            self.workspace,
            self.scratch,
        );
        self.tier1_us += profile::elapsed_us(start);
        result?;
        let mut state = self.shared.lock()?;
        Shared::decoded(&mut state, index);
        drop(state);
        Ok(())
    }

    /// Hand back every band this view is reading.
    fn release_all(&mut self) -> Result<()> {
        for ring in 0..self.current.len() {
            self.release(ring)?;
        }
        Ok(())
    }

    /// After the IDWT, free its bands and decode every band it didn't read,
    /// here and on the pool.
    fn drain(&mut self) -> Result<()> {
        self.release_all()?;
        loop {
            let mut state = self.shared.lock()?;
            // The error stays for the other readers, whose bands it may
            // have left running.
            if let Some(error) = state.error {
                return Err(error);
            }
            // Free every decoded band, which lets its ring's next band start.
            while let Some(index) = state.decoded.pop() {
                if state.status[index] == Status::Done {
                    let slot = state.held[index]
                        .take()
                        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
                    self.shared.free(&mut state, index, slot);
                }
            }
            if state.freed == self.shared.bands.len() {
                return Ok(());
            }
            let next = self.shared.start_next(&mut state)?;
            let running = state.running > 0;
            match next {
                Some(Task::Whole(index, slot)) => {
                    drop(state);
                    self.send_workers()?;
                    self.decode_here(index, slot)?;
                    self.shared.lock()?.held[index] = Some(slot);
                }
                Some(part @ Task::Part(..)) => {
                    drop(state);
                    self.send_workers()?;
                    self.decode_for_pool(part)?;
                }
                None if running => {
                    let mut state = self
                        .shared
                        .decoded
                        .wait_timeout(state, STRANDED_CHECK)
                        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                        .0;
                    let stranded = state.error.is_none() && state.active == 0 && state.running > 0;
                    state.stranded |= stranded;
                    drop(state);
                    if stranded {
                        return Err(DecodingError::CodeBlockDecodeFailure.into());
                    }
                }
                None => {
                    drop(state);
                    // Bands left that no ring can start.
                    return Err(DecodingError::CodeBlockDecodeFailure.into());
                }
            }
        }
    }
}

impl<'p, 'c, W, D> SubBandRows for PoolRows<'_, 'p, 'c, '_, W, D>
where
    W: TaskScratch + 'p,
    D: Fn(usize, BandOutput<'_, '_>, &mut W, &mut BlockScratch) -> Result<()> + Sync,
    'c: 'p,
{
    fn prepare_row(&mut self, sub_band: usize, y: u32) -> Result<()> {
        let ring = self.ring(sub_band)?;
        let ring_bands = self.shared.rings[ring].bands.clone();
        let stored = self.shared.rings[ring].stored;
        if y < stored.y0 || y >= stored.y1 {
            // Outside the stored rows the sub-band reads as zero.
            return Ok(());
        }
        loop {
            if let Some((index, _)) = self.current[ring] {
                let rows = &self.shared.bands[index].rows;
                if y < rows.start {
                    // Rows are read in order, so a band already left can't be
                    // read again.
                    return Err(DecodingError::CodeBlockDecodeFailure.into());
                }
                if y < rows.end {
                    return Ok(());
                }
                self.release(ring)?;
            }
            let index = self.next[ring];
            if index >= ring_bands.end || self.shared.bands[index].rows.start > y {
                // No code-block covers the row, so it reads as zero.
                return Ok(());
            }
            let slot = self.acquire(index)?;
            self.current[ring] = Some((index, slot));
            self.next[ring] = index + 1;
        }
    }

    fn samples(&self, sub_band: usize) -> (&[f32], IntRect, u32) {
        let Ok(ring) = self.ring(sub_band) else {
            return (&[], IntRect::from_xywh(0, 0, 0, 0), 0);
        };
        let stored = self.shared.rings[ring].stored;
        match &self.current[ring] {
            Some((index, slot)) => {
                let rows = &self.shared.bands[*index].rows;
                (
                    slot,
                    IntRect::from_ltrb(stored.x0, rows.start, stored.x1, rows.end),
                    stored.width(),
                )
            }
            None => (&[], IntRect::from_xywh(0, 0, 0, 0), stored.width()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::decode::{DecodeAllocationBudget, DecoderContext, WorkspacePeak};
    use crate::j2c::tile;
    use crate::reader::BitReader;
    use crate::{encode, DecodeSettings, EncodeOptions, Image, DEFAULT_MAX_DECODE_BYTES};

    fn budget_fixture(width: u32, height: u32, components: u16, block_exp: u8) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let pixels: Vec<u8> = (0..width * height * u32::from(components))
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[3]
            })
            .collect();
        encode(
            &pixels,
            width,
            height,
            components,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 4,
                code_block_width_exp: block_exp,
                code_block_height_exp: block_exp,
                ..EncodeOptions::default()
            },
        )
        .expect("encode budget fixture")
    }

    struct CountingSink(usize);

    impl ComponentRowSink for CountingSink {
        fn write_rows(&mut self, _x: u32, _y: u32, planes: &[crate::ComponentPlane<'_>]) {
            self.0 += planes
                .iter()
                .map(|plane| plane.samples().len())
                .sum::<usize>();
        }
    }

    /// Every live component plan and the scratch of the threads that serve
    /// them must fit beside the executor, including the shared executor that
    /// streams RGB rows. Simulated retained memory leaves room for either
    /// owner, but not both.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one budget matrix checks the plans, the executor and the shared RGB executor together"
    )]
    fn pool_bands_respect_the_combined_plan_and_executor_budget() {
        let (width, height) = (203, 157);
        // A component in 16 x 16 blocks plans its route in less room than
        // its executor then needs beside it; 4 x 4 blocks plan in more.
        for (components, block_exp) in [(1_u16, 2), (3, 0)] {
            let bytes = budget_fixture(width, height, components, block_exp);
            let image =
                Image::new(&bytes, &DecodeSettings::default()).expect("parse budget fixture");
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .expect("build budget test pool")
                .install(|| {
                    let mut context = DecoderContext::default();
                    context.set_output_region(Some((0, 0, width, height)));
                    context.tile_decode_context.line_region_idwt = true;
                    context.tile_decode_context.stream_rows = components > 1;
                    let tiles =
                        tile::parse(&mut BitReader::new(image.codestream), &image.header, 0)
                            .expect("parse tile");
                    let baseline = context
                        .reset(
                            &image.header,
                            &tiles[0],
                            tiles.structural_workspace_bytes(),
                            0,
                            false,
                            None,
                        )
                        .expect("prepare output planes");
                    let (tile_ctx, storage) =
                        (&mut context.tile_decode_context, &mut context.storage);
                    let mut sink = CountingSink(0);
                    // Warm the retained bank and leave parsed block storage
                    // available at the tile boundary, before applying the cap.
                    let output = if components == 1 {
                        super::super::TileOutput::Planes
                    } else {
                        super::super::TileOutput::Sink(&mut sink)
                    };
                    super::super::decode_tile(
                        &tiles[0],
                        &image.header,
                        tile_ctx,
                        storage,
                        &mut None,
                        CpuDecodeParallelism::Auto,
                        false,
                        &mut DecodeProfileTimings::default(),
                        baseline,
                        output,
                    )
                    .expect("decode with ample headroom");
                    assert_eq!(
                        tile_ctx.debug_counters.pool_band_components,
                        usize::from(components)
                    );
                    let (route_bytes, executor) = {
                        let mut routes = Vec::new();
                        let mut live = None;
                        for component in 0..usize::from(components) {
                            let BandRouting::Route(route) = plan_band_route_beside(
                                &tiles[0],
                                component,
                                tile_ctx,
                                storage,
                                &image.header,
                                live,
                                components > 1,
                            )
                            .expect("plan control route") else {
                                panic!("control fixture must take the pool route");
                            };
                            live = Some(route.budget());
                            routes.push(route);
                        }
                        // Reuse the existing peak accounting for the routes'
                        // owned tables and their threads' scratch, excluding
                        // the retained bank: the workers and one reader.
                        let threads = routes
                            .iter()
                            .map(|route| route.workers() + 1)
                            .max()
                            .unwrap_or(0);
                        let retained = tile_ctx.tier1_capacity_bytes().expect("bank bytes");
                        tile_ctx.debug_counters.workspace_peak = WorkspacePeak::default();
                        BandRoute::record_live(
                            routes.iter().map(|route| &**route),
                            threads,
                            tile_ctx,
                            storage,
                        )
                        .expect("record route owners");
                        let route_bytes =
                            tile_ctx.debug_counters.workspace_peak.bytes().tier1 - retained;
                        let (rings, bands, _, _) = plan_tile_rings(&routes, storage, &image.header)
                            .expect("plan control rings");
                        (
                            route_bytes,
                            executor_bytes(&rings, &bands, (threads - 1, 1))
                                .expect("executor bytes"),
                        )
                    };
                    let headroom = route_bytes + executor / 2;
                    assert!(headroom >= executor, "each owner must fit separately");
                    let baseline = DecodeAllocationBudget::for_storage(storage)
                        .expect("control budget")
                        .live_bytes();
                    let structural = storage.structural_workspace_bytes;
                    storage.structural_workspace_bytes +=
                        DEFAULT_MAX_DECODE_BYTES - baseline - headroom;
                    sink.0 = 0;
                    let result = if components == 1 {
                        // The least room the component's route plans in,
                        // which leaves none for its executor.
                        let (mut short, mut room) = (route_bytes, 4 * headroom);
                        let mut plans = |headroom: usize| {
                            storage.structural_workspace_bytes =
                                structural + DEFAULT_MAX_DECODE_BYTES - baseline - headroom;
                            match plan_band_route(
                                &tiles[0],
                                0,
                                tile_ctx,
                                storage,
                                &image.header,
                                false,
                            ) {
                                Ok(BandRouting::Route(route)) => Some(route),
                                Ok(BandRouting::Serial | BandRouting::WholeWindows) => None,
                                Err(error) => panic!("planning within {headroom}: {error:?}"),
                            }
                        };
                        assert!(plans(short).is_none() && plans(room).is_some());
                        while room - short > 1 {
                            let middle = short + (room - short) / 2;
                            if plans(middle).is_some() {
                                room = middle;
                            } else {
                                short = middle;
                            }
                        }
                        let route = plans(room).expect("the route plans");
                        let declined = decode_route(
                            &route,
                            0,
                            &image.header,
                            tile_ctx,
                            storage,
                            tiles[0].component_infos[0].wavelet_transform(),
                            false,
                            &mut 0,
                        );
                        assert!(
                            matches!(declined, Ok(None)),
                            "the executor must decline before decoding: {declined:?}"
                        );
                        drop(route);
                        // Where the executor does not fit, the serial bands
                        // decode the component as the pool does with room.
                        let mut decode = |headroom: usize| {
                            storage.structural_workspace_bytes =
                                structural + DEFAULT_MAX_DECODE_BYTES - baseline - headroom;
                            tile_ctx.channel_data[0].container.fill(f32::NAN);
                            let pooled = tile_ctx.debug_counters.pool_band_components;
                            let decoded = decode_component(
                                &tiles[0],
                                0,
                                &image.header,
                                tile_ctx,
                                storage,
                                tiles[0].component_infos[0].wavelet_transform(),
                                false,
                                &mut DecodeProfileTimings::default(),
                            );
                            assert!(
                                matches!(decoded, Ok(true)),
                                "the component decodes within {headroom}: {decoded:?}"
                            );
                            let bits: Vec<u32> = tile_ctx.channel_data[0]
                                .container
                                .iter()
                                .map(|sample| sample.to_bits())
                                .collect();
                            (tile_ctx.debug_counters.pool_band_components - pooled, bits)
                        };
                        let (pooled, with_room) = decode(4 * headroom);
                        assert_eq!(pooled, 1, "the pool decodes the component with room");
                        let (pooled, serial) = decode(room);
                        assert_eq!(pooled, 0, "the serial bands decode it within {room}");
                        assert!(
                            serial == with_room,
                            "the serial bands decode it bit for bit"
                        );
                        return;
                    } else {
                        decode_tile_rows(
                            &tiles[0],
                            &image.header,
                            tile_ctx,
                            storage,
                            &mut sink,
                            false,
                            &mut DecodeProfileTimings::default(),
                        )
                    };
                    match result {
                        Err(error) if super::super::reuse::is_capacity_error(&error) => {}
                        Ok(false) => {}
                        other => panic!(
                            "pool route must decline or reject combined live owners: \
                             components={components}, headroom={headroom}, route={route_bytes}, \
                             executor={executor}, excess={}, result={other:?}",
                            route_bytes + executor - headroom,
                        ),
                    }
                    assert_eq!(
                        sink.0, 0,
                        "budget rejection must precede streaming any rows"
                    );
                });
        }
    }

    /// Collects the sample bits a sink receives, by component, and how often
    /// each sample arrives.
    struct Collected {
        width: usize,
        planes: Vec<Vec<u32>>,
        arrivals: Vec<u8>,
    }

    impl Collected {
        fn new(width: u32, height: u32) -> Self {
            Self {
                width: width as usize,
                planes: Vec::new(),
                arrivals: alloc::vec![0; width as usize * height as usize],
            }
        }
    }

    impl ComponentRowSink for Collected {
        fn write_rows(&mut self, x: u32, y: u32, planes: &[crate::ComponentPlane<'_>]) {
            if self.planes.is_empty() {
                self.planes = alloc::vec![alloc::vec![0; self.arrivals.len()]; planes.len()];
            }
            let (width, height) = planes[0].dimensions();
            for (component, plane) in planes.iter().enumerate() {
                for row in 0..height as usize {
                    for column in 0..width as usize {
                        let at = (y as usize + row) * self.width + x as usize + column;
                        self.planes[component][at] =
                            plane.samples()[row * width as usize + column].to_bits();
                        if component == 0 {
                            self.arrivals[at] += 1;
                        }
                    }
                }
            }
        }
    }

    /// The bytes a streamed tile's pool bands charge beyond the decode's
    /// baseline, in the order they charge them: the list of routes, every
    /// route's plan, the rings and bands, with the most planning them holds
    /// beside them, the executor's records and the threads' scratch.
    #[derive(Clone, Copy, Debug)]
    struct Steps {
        routes: usize,
        plans: usize,
        rings: usize,
        planning: usize,
        /// The streamed IDWT's plan, ring length and reconstruction of each
        /// component.
        lines: usize,
        executor: usize,
        scratch: usize,
        /// The tile's readers, beside its workers.
        readers: usize,
    }

    /// How a streamed tile decoded: on the pool's bands or, declined there,
    /// on serial bands, and to what rows or capacity error.
    struct Streamed {
        pooled: bool,
        rows: Result<Collected>,
    }

    /// Decode the one tile of `image` on a fresh context, streaming its rows
    /// to a sink: first with room to spare, which warms the workspace bank
    /// and gives the rows and the [`Steps`], measured from what the routes,
    /// records and scratch hold; then again as `decode_tile` does, on the
    /// pool's bands or, when they decline the tile, on serial bands, with all
    /// but `headroom` bytes beyond the baseline taken as if retained, which
    /// every budget of the decode sees.
    #[expect(
        clippy::too_many_lines,
        reason = "the warm pass measures the steps the budgeted pass is then checked against, on one context"
    )]
    fn decode_streamed(image: &Image<'_>, headroom: usize) -> (Steps, Collected, Streamed) {
        let (width, height) = (image.width(), image.height());
        let mut context = DecoderContext::default();
        context.set_output_region(Some((0, 0, width, height)));
        context.tile_decode_context.line_region_idwt = true;
        context.tile_decode_context.stream_rows = true;
        let tiles = tile::parse(&mut BitReader::new(image.codestream), &image.header, 0)
            .expect("parse tile");
        let baseline = context
            .reset(
                &image.header,
                &tiles[0],
                tiles.structural_workspace_bytes(),
                0,
                false,
                None,
            )
            .expect("prepare output planes");
        let (tile_ctx, storage) = (&mut context.tile_decode_context, &mut context.storage);
        let mut warm = Collected::new(width, height);
        super::super::decode_tile(
            &tiles[0],
            &image.header,
            tile_ctx,
            storage,
            &mut None,
            CpuDecodeParallelism::Auto,
            false,
            &mut DecodeProfileTimings::default(),
            baseline,
            super::super::TileOutput::Sink(&mut warm),
        )
        .expect("decode with room to spare");
        let components = tiles[0].component_infos.len();
        assert_eq!(tile_ctx.debug_counters.pool_band_components, components);
        assert!(warm.arrivals.iter().all(|&count| count == 1));
        let start = DecodeAllocationBudget::for_storage(storage)
            .expect("baseline")
            .live_bytes();
        let routes_bytes = components * size_of::<Box<BandRoute<'_>>>();
        let mut live = DecodeAllocationBudget::for_storage(storage).expect("baseline");
        live.include_bytes(routes_bytes).expect("routes list");
        let mut routes = Vec::new();
        for component in 0..components {
            let BandRouting::Route(route) = plan_band_route_beside(
                &tiles[0],
                component,
                tile_ctx,
                storage,
                &image.header,
                Some(live),
                true,
            )
            .expect("plan route") else {
                panic!("the tile takes the pool's bands");
            };
            live = route.budget();
            routes.push(route);
        }
        let plans = live.live_bytes() - start - routes_bytes;
        // With the bank warm, planning charges no workspace; it charges the
        // plans' tables, and a task per strip.
        let recorded = tile_ctx.debug_counters.pool_band_plan_bytes;
        BandRoute::record_live(routes.iter().map(|route| &**route), 0, tile_ctx, storage)
            .expect("record plans");
        let tables = tile_ctx.debug_counters.pool_band_plan_bytes - recorded;
        assert!(
            plans >= tables,
            "planning charged {plans} for {tables} of tables"
        );
        let (_, planning) =
            tile_ring_plan_bytes(&routes, storage, &image.header).expect("planning bytes");
        let (rings, mut bands, strip_starts, active_levels) =
            plan_tile_rings(&routes, storage, &image.header).expect("plan rings");
        let ring_bytes = rings.capacity() * size_of::<Ring>()
            + bands.capacity() * size_of::<Band>()
            + (strip_starts.capacity() + active_levels.capacity()) * size_of::<usize>();
        // The threads the tile streamed on: the most workers its routes
        // planned that the bank holds beside its readers, which split its
        // bands.
        let readers = tile_ctx.debug_counters.pool_band_readers;
        let route_workers = routes
            .iter()
            .map(|route| route.workers())
            .max()
            .unwrap_or(0);
        let workers = route_workers.min(tile_ctx.classic_task_workspaces.len() - readers);
        split_bands(
            tile_ctx,
            &rings,
            &mut bands,
            workers + readers,
            |band, parts| {
                let (route, strip) = route_strip(&routes, &strip_starts, band.strip)?;
                route.strip_parts(strip, parts)
            },
        )
        .expect("split bands");
        let steps = Steps {
            routes: routes_bytes,
            plans,
            rings: ring_bytes,
            planning,
            lines: components
                * (size_of::<LinePlan>()
                    + size_of::<usize>()
                    + size_of::<LineReconstruction<'_>>()),
            executor: executor_records(&rings, &bands, workers, readers),
            scratch: tile_ctx.debug_counters.thread_scratch_held,
            readers,
        };
        drop((routes, rings, bands, strip_starts, active_levels));
        storage.structural_workspace_bytes += DEFAULT_MAX_DECODE_BYTES - start - headroom;
        let mut sink = Collected::new(width, height);
        let pooled = decode_tile_rows(
            &tiles[0],
            &image.header,
            tile_ctx,
            storage,
            &mut sink,
            false,
            &mut DecodeProfileTimings::default(),
        )
        .unwrap_or_else(|error| panic!("the pool's bands decline instead: {error:?}"));
        if pooled {
            return (
                steps,
                warm,
                Streamed {
                    pooled,
                    rows: Ok(sink),
                },
            );
        }
        assert!(
            sink.arrivals.iter().all(|&count| count == 0),
            "no row before the pool declines"
        );
        storage.parallel_bands = false;
        storage.banded_tier1 = true;
        let rows = banded_tier1::decode_tile_rows(
            &tiles[0],
            &image.header,
            tile_ctx,
            storage,
            &mut sink,
            false,
            &mut DecodeProfileTimings::default(),
        )
        .map(|()| sink);
        (steps, warm, Streamed { pooled, rows })
    }

    /// A tile of blocks of several codeword segments streams its rows from
    /// pool bands, with one reader or one per component, under exactly the
    /// bytes they charge: the list of routes, the plans, the executor's
    /// records and the threads' scratch, as their sizes give them, the
    /// records' from their types and counts. A byte short of the list, of the
    /// plans, of the records or of the scratch, the pool declines the tile,
    /// before any row reaches the sink, and under the same retained bytes
    /// serial bands stream the same rows when they fit and fail with a
    /// capacity error when they don't.
    #[test]
    fn streamed_pool_bands_take_a_segmented_tile_at_exactly_their_bytes() {
        let bytes = include_bytes!("../../../fixtures/multi_segment/bypass_termall_rgb_53.j2k");
        let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
        // One reader for the tile on 4 threads, one per component on 12.
        for (threads, expected_readers) in [(4, 1), (12, 3)] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("build pool")
                .install(|| take_at_exactly_their_bytes(&image, expected_readers));
        }
    }

    fn take_at_exactly_their_bytes(image: &Image<'_>, expected_readers: usize) {
        let (steps, _, _) = decode_streamed(image, DEFAULT_MAX_DECODE_BYTES / 2);
        let Steps {
            routes,
            plans,
            rings,
            planning,
            lines,
            executor,
            scratch,
            readers,
        } = steps;
        assert_eq!(readers, expected_readers, "{steps:?}");
        assert!(
            routes > 0 && plans > 0 && rings > 0 && executor > 0 && scratch > 0,
            "{steps:?}"
        );
        // Planning the rings holds more beside them only until the executor
        // and the scratch take its place.
        let all = routes + plans + rings + planning.max(lines + executor + scratch);
        let (_, warm, exact) = decode_streamed(image, all);
        assert!(exact.pooled, "pool bands within {all} bytes: {steps:?}");
        let rows = exact.rows.expect("pool bands decode");
        assert!(rows.arrivals.iter().all(|&count| count == 1));
        assert!(rows.planes == warm.planes, "pool bands' rows");
        let mut serial = Vec::new();
        for headroom in [
            routes - 1,
            routes + plans - 1,
            routes + plans + rings - 1,
            routes + plans + rings + lines - 1,
            routes + plans + rings + lines + executor - 1,
            all - 1,
        ] {
            let (_, _, short) = decode_streamed(image, headroom);
            assert!(
                !short.pooled,
                "pool bands within {headroom} bytes: {steps:?}"
            );
            match short.rows {
                Ok(rows) => {
                    assert!(rows.arrivals.iter().all(|&count| count == 1));
                    assert!(rows.planes == warm.planes, "serial rows at {headroom}");
                    serial.push(true);
                }
                Err(error) => {
                    assert!(
                        super::super::reuse::is_capacity_error(&error),
                        "serial bands at {headroom}: {error:?}"
                    );
                    serial.push(false);
                }
            }
        }
        // Serial bands need less than the pool's bands, and more than
        // their list of routes.
        assert_eq!(serial.first(), Some(&false), "{steps:?}");
        assert_eq!(serial.last(), Some(&true), "{steps:?}");
        assert!(
            serial.windows(2).all(|pair| pair[0] <= pair[1]),
            "serial bands fit more as the headroom grows: {serial:?}"
        );
    }

    /// The bytes of the records the band executor allocates while `workers`
    /// and `readers` threads decode `bands` of `rings`, from their types: per
    /// band its status, held and owned slot, part counters, decoded entry
    /// and shared output; a queue entry per part; per ring its readiness and
    /// freed counters, slot list and slots, and each reader's band being read
    /// and next band; per thread its workspace pair; per reader its view and
    /// its entry in the readers' list; and the pool's job per worker and per
    /// pool thread that may retire one, and what the locks of bands in parts
    /// and of the shared state allocate, which
    /// `band_executor_allocates_within_its_charge` measures. It decodes in a
    /// pool, so its workers take no block of the pool's shared queue.
    fn executor_records(rings: &[Ring], bands: &[Band], workers: usize, readers: usize) -> usize {
        let per_band = size_of::<Status>()
            + 2 * size_of::<Option<&mut [f32]>>()
            + 3 * size_of::<usize>()
            + size_of::<Mutex<&mut [f32]>>();
        let entries: usize = bands.iter().map(|band| band.parts).sum();
        let per_ring = 2 * size_of::<usize>()
            + size_of::<Vec<&mut [f32]>>()
            + readers * (size_of::<Option<(usize, &mut [f32])>>() + size_of::<usize>());
        let slots: usize = rings.iter().map(|ring| ring.slots).sum();
        let per_thread = size_of::<(&mut (), &mut BlockScratch)>();
        let per_reader = size_of::<PoolRows<'_, '_, '_, '_, (), ()>>()
            + size_of::<&mut (dyn crate::j2c::idwt::SubBandRows + Send)>();
        let locks = bands.iter().filter(|band| band.parts > 1).count() + 2;
        bands.len() * per_band
            + entries * size_of::<Reverse<(i64, Reverse<usize>, usize)>>()
            + rings.len() * per_ring
            + slots * size_of::<&mut [f32]>()
            + (workers + readers) * per_thread
            + readers * per_reader
            + (workers + rayon::current_num_threads()) * SENT_WORKER_BYTES
            + locks * LOCK_BYTES
    }

    /// What the band executor allocated for a tile, measured under the
    /// counting allocator, beside what the decode charges for it.
    #[derive(Debug)]
    struct Measured {
        /// The most planning the routes held at once, and what they kept,
        /// against what their budgets charged.
        routing: (usize, usize),
        routing_charge: usize,
        /// The most planning the rings and bands held at once, and what it
        /// kept, against the bytes charged for them and for the most beside
        /// them.
        planning: (usize, usize),
        planning_charge: (usize, usize),
        /// The most the executor held at once while the bands decoded,
        /// against its charge.
        running: usize,
        running_charge: usize,
        readers: usize,
        parted: usize,
        bands: usize,
        deep: bool,
        shallow: bool,
        /// The deepest finest-level and coarser-level rings that hold no
        /// whole window.
        slots: (usize, usize),
    }

    /// Read every row of the rings that `ring_readers` gives to `readers`,
    /// the first of which is reader `first`: each reader on a thread of its
    /// own in a pool, and one after another outside one, where handing them
    /// to the pool would allocate.
    fn read_rings(
        rings: &[Ring],
        ring_readers: &[usize],
        readers: &mut [&mut (dyn SubBandRows + Send)],
        first: usize,
    ) -> Result<()> {
        let Some((reader, rest)) = readers.split_first_mut() else {
            return Ok(());
        };
        let mut read = || -> Result<()> {
            for (ring, _) in rings
                .iter()
                .zip(ring_readers)
                .filter(|(_, reader)| **reader == first)
            {
                for y in ring.stored.y0..ring.stored.y1 {
                    reader.prepare_row(ring.sub_band, y)?;
                }
            }
            Ok(())
        };
        if rest.is_empty() {
            return read();
        }
        let mut next = || read_rings(rings, ring_readers, rest, first + 1);
        if rayon::current_thread_index().is_none() {
            return read().and_then(|()| next());
        }
        let (mine, theirs) = rayon::join(read, next);
        mine.and(theirs)
    }

    /// Decode the tile of `bytes`, or its `roi`, streaming its rows on the
    /// current pool, with its classic bands in `parts` parts when given:
    /// once to warm the bank, then, under the counting allocator, planning
    /// its routes, rings and bands and running the band executor as the
    /// streamed tile drives it, reading every row of every ring, each
    /// component's through its own reader when it has one.
    #[expect(
        clippy::too_many_lines,
        reason = "the planning and executor phases are measured in order under the counting allocator, on one warm bank"
    )]
    fn measure_executor(
        bytes: &[u8],
        roi: Option<(u32, u32, u32, u32)>,
        parts: Option<usize>,
    ) -> Measured {
        let image = Image::new(bytes, &DecodeSettings::default()).expect("parse fixture");
        let roi = roi.unwrap_or((0, 0, image.width(), image.height()));
        let mut context = DecoderContext::default();
        context.set_output_region(Some(roi));
        context.tile_decode_context.line_region_idwt = true;
        context.tile_decode_context.stream_rows = true;
        context.tile_decode_context.band_parts = parts;
        let tiles = tile::parse(&mut BitReader::new(image.codestream), &image.header, 0)
            .expect("parse tile");
        let baseline = context
            .reset(
                &image.header,
                &tiles[0],
                tiles.structural_workspace_bytes(),
                0,
                false,
                None,
            )
            .expect("prepare output planes");
        let (tile_ctx, storage) = (&mut context.tile_decode_context, &mut context.storage);
        super::super::decode_tile(
            &tiles[0],
            &image.header,
            tile_ctx,
            storage,
            &mut None,
            CpuDecodeParallelism::Auto,
            false,
            &mut DecodeProfileTimings::default(),
            baseline,
            super::super::TileOutput::Sink(&mut CountingSink(0)),
        )
        .expect("decode with room to spare");
        let components = tiles[0].component_infos.len();
        assert_eq!(tile_ctx.debug_counters.pool_band_components, components);
        let mut routes = Vec::with_capacity(components);
        let mut live = DecodeAllocationBudget::for_storage(storage).expect("baseline");
        let start = live.live_bytes();
        settle();
        let base = j2k_alloc_probe::live_bytes();
        j2k_alloc_probe::reset_peak();
        for component in 0..components {
            let BandRouting::Route(route) = plan_band_route_beside(
                &tiles[0],
                component,
                tile_ctx,
                storage,
                &image.header,
                Some(live),
                true,
            )
            .expect("plan route") else {
                panic!("the tile takes the pool's bands");
            };
            live = route.budget();
            routes.push(route);
        }
        let routing = (
            j2k_alloc_probe::peak_bytes() - base,
            j2k_alloc_probe::live_bytes() - base,
        );
        let routing_charge = live.live_bytes() - start;
        let planning_charge =
            tile_ring_plan_bytes(&routes, storage, &image.header).expect("planning bytes");
        let base = j2k_alloc_probe::live_bytes();
        j2k_alloc_probe::reset_peak();
        let (rings, mut bands, strip_starts, active_levels) =
            plan_tile_rings(&routes, storage, &image.header).expect("plan rings");
        let planning = (
            j2k_alloc_probe::peak_bytes() - base,
            j2k_alloc_probe::live_bytes() - base,
        );
        let line_plans =
            plan_tile_lines(&tiles[0], &image.header, tile_ctx, storage, &active_levels)
                .expect("plan lines");
        let copy = line_plans
            .first()
            .map_or(IntRect::from_xywh(0, 0, 0, 0), |plan| plan.copy);
        let coding = routes[0].coding();
        let bank = match coding {
            CodeBlockCoding::Classic => tile_ctx.classic_task_workspaces.len(),
            CodeBlockCoding::HighThroughput => tile_ctx.ht_task_workspaces.len(),
        };
        let route_workers = routes
            .iter()
            .map(|route| route.workers())
            .max()
            .unwrap_or(0);
        let samples = copy.width() as usize * copy.height() as usize;
        let readers = if parallel_synthesis(components, coding, bank, route_workers, samples) {
            components
        } else {
            1
        };
        let workers = route_workers.min(bank - readers);
        if coding == CodeBlockCoding::Classic {
            split_bands(
                tile_ctx,
                &rings,
                &mut bands,
                workers + readers,
                |band, parts| {
                    let (route, strip) = route_strip(&routes, &strip_starts, band.strip)?;
                    route.strip_parts(strip, parts)
                },
            )
            .expect("split bands");
        }
        let running_charge =
            executor_bytes(&rings, &bands, (workers, readers)).expect("executor bytes");
        let mut scratches = BlockScratch::for_threads(
            workers + readers,
            BandRoute::shared_scratch(routes.iter().map(|route| &**route)).expect("sizes"),
            &mut DecodeAllocationBudget::for_storage(storage).expect("budget"),
        )
        .expect("thread scratch");
        let helps = readers == 1 && coding == CodeBlockCoding::Classic;
        let mut coefficients = core::mem::take(&mut storage.coefficients);
        // Each component's rings read through its own reader, when it has
        // one.
        let mut ring_readers = Vec::new();
        for (component_idx, route) in routes.iter().enumerate() {
            let (component_rings, _) =
                ring_counts(route, storage, &image.header, component_idx).expect("rings");
            let reader = if readers == components {
                component_idx
            } else {
                0
            };
            ring_readers.extend(core::iter::repeat_n(reader, component_rings));
        }
        let read = |readers: &mut [&mut (dyn SubBandRows + Send)]| -> Result<()> {
            read_rings(&rings, &ring_readers, readers, 0)
        };
        let base = j2k_alloc_probe::live_bytes();
        j2k_alloc_probe::reset_peak();
        let slots = split_slots(&rings, &bands, storage, &mut coefficients).expect("slots");
        let source = PayloadSource::new(storage);
        let decoded = match coding {
            CodeBlockCoding::Classic => run_bands(
                (workers, readers, helps),
                &rings,
                &bands,
                slots,
                (&mut tile_ctx.classic_task_workspaces, &mut scratches),
                |strip, output, workspace, scratch| {
                    let (route, strip) = route_strip(&routes, &strip_starts, strip)?;
                    route.decode_classic(strip, output, workspace, (scratch, source))
                },
                read,
                false,
                &mut 0,
                false,
            ),
            CodeBlockCoding::HighThroughput => run_bands(
                (workers, readers, helps),
                &rings,
                &bands,
                slots,
                (&mut tile_ctx.ht_task_workspaces, &mut scratches),
                |strip, output, workspace, scratch| {
                    let (route, strip) = route_strip(&routes, &strip_starts, strip)?;
                    route.decode_ht(strip, output, workspace, (scratch, source))
                },
                read,
                false,
                &mut 0,
                false,
            ),
        };
        let running = j2k_alloc_probe::peak_bytes() - base;
        assert_eq!(decoded.ok(), Some(true), "the executor decodes every band");
        storage.coefficients = coefficients;
        Measured {
            routing,
            routing_charge,
            planning,
            planning_charge,
            running,
            running_charge,
            readers,
            parted: bands.iter().filter(|band| band.parts > 1).count(),
            bands: bands.len(),
            deep: storage.deep_band_rings,
            shallow: storage.shallow_band_rings,
            slots: deepest_rings(&rings),
        }
    }

    /// The most slots of a finest-level ring, and of a coarser-level one,
    /// that holds no whole window.
    fn deepest_rings(rings: &[Ring]) -> (usize, usize) {
        let deepest = |finest: bool| {
            rings
                .iter()
                .filter(|ring| ring.finest == finest && !ring.whole_window)
                .map(|ring| ring.slots)
                .max()
                .unwrap_or(0)
        };
        (deepest(true), deepest(false))
    }

    /// Whether this process is the one that measures the test `name`: the
    /// allocator's high-water mark is the process's, so when it is not,
    /// this runs the test alone in a process of its own and checks that it
    /// passed there.
    fn measures_alone(name: &str) -> bool {
        const PROBE: &str = "J2K_NATIVE_BAND_EXECUTOR_PROBE";
        if std::env::var_os(PROBE).is_some() {
            return true;
        }
        let module = module_path!()
            .split_once("::")
            .map(|(_, path)| path)
            .expect("module path");
        let output = std::process::Command::new(std::env::current_exe().expect("test binary"))
            .args([
                &format!("{module}::{name}"),
                "--exact",
                "--test-threads=1",
                "--nocapture",
            ])
            .env(PROBE, "1")
            .output()
            .expect("run the measuring process");
        let report = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(output.status.success(), "{report}");
        assert!(report.contains("1 passed"), "{report}");
        false
    }

    /// Under the counting allocator, the whole-window tile route, entered
    /// from a thread outside the pool that never entered it before, holds
    /// at most what it charges for its plan, tasks and threads' scratch,
    /// the locks it shares and what the pool takes for it.
    #[test]
    fn tile_route_allocates_within_its_charge() {
        use core::sync::atomic::{AtomicBool, Ordering};

        static ROUTED: AtomicBool = AtomicBool::new(false);
        if !measures_alone("tile_route_allocates_within_its_charge") {
            return;
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .expect("the measuring process's first use of the global pool");
        rayon::broadcast(|_| ());
        let bytes = budget_fixture(203, 157, 1, 2);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse fixture");
        let mut context = DecoderContext::default();
        let tiles = tile::parse(&mut BitReader::new(image.codestream), &image.header, 0)
            .expect("parse tile");
        let baseline = context
            .reset(
                &image.header,
                &tiles[0],
                tiles.structural_workspace_bytes(),
                0,
                false,
                None,
            )
            .expect("prepare output planes");
        let (tile_ctx, storage) = (&mut context.tile_decode_context, &mut context.storage);
        // Warm the bank on this thread, so the route reuses it.
        super::super::decode_tile(
            &tiles[0],
            &image.header,
            tile_ctx,
            storage,
            &mut None,
            CpuDecodeParallelism::Auto,
            false,
            &mut DecodeProfileTimings::default(),
            baseline,
            super::super::TileOutput::Planes,
        )
        .expect("decode with room to spare");
        tile_ctx.tile_route_post_decode = Some(|_| {
            ROUTED.store(true, Ordering::Relaxed);
            Ok(())
        });
        let retained = tile_ctx.tier1_capacity_bytes().expect("bank bytes");
        tile_ctx.debug_counters.workspace_peak = WorkspacePeak::default();
        let held = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    settle();
                    let base = j2k_alloc_probe::live_bytes();
                    j2k_alloc_probe::reset_peak();
                    super::super::subband::decode_component_tile_bit_planes(
                        &tiles[0],
                        0..1,
                        tile_ctx,
                        storage,
                        &image.header,
                        &mut None,
                        CpuDecodeParallelism::Auto,
                        false,
                    )
                    .expect("decode the tile");
                    j2k_alloc_probe::peak_bytes() - base
                })
                .join()
                .expect("the measuring thread")
        });
        assert!(ROUTED.load(Ordering::Relaxed), "the tile route decodes it");
        let charged = tile_ctx.debug_counters.workspace_peak.bytes().tier1 - retained;
        assert!(
            held <= charged,
            "the tile route held {held} of {charged} bytes"
        );
    }

    /// Wait until the process has held the same bytes for 100 ms, so the
    /// pools' threads have gone idle: a thread allocates its queue's state
    /// and the locks it sleeps on when it first idles.
    fn settle() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let (mut last, mut still) = (j2k_alloc_probe::live_bytes(), 0);
        while still < 20 {
            assert!(
                std::time::Instant::now() < deadline,
                "the process's allocations never settled"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
            let now = j2k_alloc_probe::live_bytes();
            still = if now == last { still + 1 } else { 0 };
            last = now;
        }
    }

    /// Pools of `threads` threads each, every thread of which has run a
    /// job, so none allocates its first job's state inside a measurement.
    fn warmed_pools(threads: &[usize]) -> Vec<rayon::ThreadPool> {
        threads
            .iter()
            .map(|&threads| {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("build pool");
                pool.broadcast(|_| ());
                pool
            })
            .collect()
    }

    /// Under the counting allocator, the band executor holds at most what
    /// the decode charges for it, planning its routes, rings and bands and
    /// while they decode: for classic and HTJ2K tiles, whole and in a
    /// region, with bands in parts, on pools of 2, 4, 10, 11 and 12 threads
    /// and from outside a pool, with one reader and with one per component,
    /// with shallow rings and deep ones.
    #[test]
    fn band_executor_allocates_within_its_charge() {
        if !measures_alone("band_executor_allocates_within_its_charge") {
            return;
        }
        // The decode from outside a pool streams on the global pool, of
        // enough threads whatever the machine.
        rayon::ThreadPoolBuilder::new()
            .num_threads(12)
            .build_global()
            .expect("the measuring process's first use of the global pool");
        let classic = include_bytes!("../../../fixtures/multi_segment/bypass_termall_rgb_53.j2k");
        let ht = ht_fixture();
        let wide = shallow_fixture();
        let (mut readers, mut parted, mut clipped, mut deep) = (Vec::new(), 0, false, false);
        let mut shallow = Vec::new();
        let mut check = |case: &str, measured: Measured, whole: &Measured| {
            assert!(
                measured.routing.0 <= measured.routing_charge,
                "{case}: {measured:?}"
            );
            let (peak, kept) = measured.planning;
            let (planned, beside) = measured.planning_charge;
            assert!(
                kept <= planned && peak <= planned + beside,
                "{case}: {measured:?}"
            );
            assert!(
                measured.running <= measured.running_charge,
                "{case}: {measured:?}"
            );
            if measured.shallow {
                assert_eq!(measured.slots.0, 2, "{case}: shallow finest rings");
                shallow.push((case.to_owned(), measured.slots, measured.parted));
            }
            readers.push(measured.readers);
            parted += measured.parted;
            clipped |= measured.bands < whole.bands;
            deep |= measured.deep;
        };
        // The pools, and the global one, stay up until every measurement
        // ends, so no pool's teardown frees bytes inside another's window.
        // Every thread of each runs a job before any measurement, and every
        // measurement waits until they idle (see `settle`).
        let pools = warmed_pools(&[2, 4, 10, 11, 12]);
        rayon::broadcast(|_| ());
        for pool in &pools {
            let threads = pool.current_num_threads();
            pool.install(|| {
                for (label, bytes, splits) in [
                    ("classic", &classic[..], &[None, Some(4)][..]),
                    ("HTJ2K", &ht[..], &[None][..]),
                    ("wide classic", &wide[..], &[None, Some(4)][..]),
                ] {
                    let whole = measure_executor(bytes, None, None);
                    for &roi in &[None, Some((48, 48, 16, 16))] {
                        for &parts in splits {
                            let case = format!("{label} {roi:?} parts {parts:?} on {threads}");
                            check(&case, measure_executor(bytes, roi, parts), &whole);
                        }
                    }
                }
            });
        }
        // Called from outside a pool, the executor sends its workers through
        // the global pool's shared queue.
        let whole = measure_executor(classic, None, None);
        check(
            "classic outside the pool",
            measure_executor(classic, None, Some(4)),
            &whole,
        );
        drop(pools);
        assert!(readers.contains(&1) && readers.contains(&3), "{readers:?}");
        assert!(
            parted > 0 && clipped && deep,
            "parts {parted}, region {clipped}, deep {deep}"
        );
        // The wide tile at full resolution on 10 and 11 threads holds the
        // shallow finest rings, and its bands split into parts by themselves.
        assert!(
            shallow
                .iter()
                .any(|(case, slots, parted)| case.contains("parts None")
                    && slots.1 == 2
                    && *parted > 0),
            "{shallow:?}"
        );
    }

    /// A classic RGB tile of noise wide and tall enough for the shallow
    /// finest rings: a reader per component on 10 threads (the unit tests'
    /// sample gate), finest-level bands of several blocks, and coarser
    /// levels with more bands than slots.
    fn shallow_fixture() -> Vec<u8> {
        let mut state = 0x9e37_79b9_u32;
        let pixels: Vec<u8> = (0..384 * 192 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[3]
            })
            .collect();
        encode(
            &pixels,
            384,
            192,
            3,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 4,
                code_block_height_exp: 2,
                ..EncodeOptions::default()
            },
        )
        .expect("encode shallow ring fixture")
    }

    /// A three-component 64 x 64 HTJ2K image in 16 x 16 code-blocks.
    fn ht_fixture() -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let pixels: Vec<u8> = (0..64 * 64 * 3)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state.to_le_bytes()[3]
            })
            .collect();
        encode(
            &pixels,
            64,
            64,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 2,
                code_block_width_exp: 2,
                code_block_height_exp: 2,
                use_ht_block_coding: true,
                ..EncodeOptions::default()
            },
        )
        .expect("encode HTJ2K fixture")
    }
}
