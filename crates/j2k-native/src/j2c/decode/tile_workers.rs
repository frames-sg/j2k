// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concurrent decode of the tiles of one image.
//!
//! Under [`CpuDecodeParallelism::Auto`], a full-image decode of several tiles
//! whose components are all sampled 1x1 hands the tiles to workers that live
//! only for this call. Each worker owns its Tier-1, IDWT and decomposition
//! scratch and stores into its own row slices of the component planes, so no
//! two workers write the same sample and the order tiles finish in cannot
//! change a bit. The workers split the bytes the decode cap leaves equally.
//! Any failure, a capacity error included, leaves the caller's counts as they
//! were, and the serial tile loop decodes the image again.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use rayon::prelude::*;

use super::store::{store_rect, TileRows};
use super::{
    decode_tile, tile_requires_exact_integer_decode, CpuDecodeParallelism, DecodeAllocationBudget,
    DecodeDebugCounters, DecodeProfileTimings, DecompositionStorage, Header, Tile,
    TileDecodeContext, WorkspaceBytes,
};
use crate::j2c::rect::IntRect;
use crate::DEFAULT_MAX_DECODE_BYTES;

#[cfg(test)]
mod tests;

/// Below this many output samples per worker, forking a worker costs more
/// than its tiles take to decode.
const MIN_SAMPLES_PER_WORKER: usize = 8 * 1024;
/// Planned bytes one worker needs per full-resolution sample of the largest
/// tile, plus fixed bytes per worker and per pool thread for Tier-1
/// workspaces. Profiled 512x512 and larger tiles peak near 9.4 bytes per
/// sample, 64x64 tiles near 15.
const EST_BYTES_PER_TILE_SAMPLE: usize = 16;
const EST_FIXED_WORKER_BYTES: usize = 256 * 1024;
const EST_BYTES_PER_POOL_THREAD: usize = 64 * 1024;

/// Decode every tile on workers, storing into `tile_ctx`'s planes and adding
/// the workers' counts to its own. Returns `None`, with those counts
/// untouched, when the image does not qualify or any tile fails.
pub(super) fn decode_tiles(
    header: &Header<'_>,
    tiles: &[Tile<'_>],
    tile_ctx: &mut TileDecodeContext,
    retained_decode_baseline: usize,
) -> Option<()> {
    let threads = rayon::current_num_threads();
    let width = header.size_data.image_width() as usize;
    let height = header.size_data.image_height() as usize;
    let samples = width.checked_mul(height)?;
    let max_workers = threads
        .min(tiles.len())
        .min(samples / MIN_SAMPLES_PER_WORKER);
    if max_workers < 2 || !eligible(header, tiles, tile_ctx, samples) {
        return None;
    }
    // The job tables are charged first; the workers share what is left.
    let mut budget = DecodeAllocationBudget::from_live_bytes(retained_decode_baseline).ok()?;
    let (mut jobs, largest_tile) = plan_jobs(header, tiles, (width, height), &mut budget)?;
    let remaining = DEFAULT_MAX_DECODE_BYTES.checked_sub(budget.live_bytes())?;
    let worker_bytes = EST_BYTES_PER_TILE_SAMPLE
        .saturating_mul(largest_tile)
        .saturating_add(EST_FIXED_WORKER_BYTES)
        .saturating_add(EST_BYTES_PER_POOL_THREAD.saturating_mul(threads));
    // Workers the cap holds below `max_workers` run inner Auto, which can
    // outgrow a worker's share and leave the serial loop to decode the image
    // again, so such images decode serially from the start.
    if remaining / worker_bytes < max_workers {
        return None;
    }
    let workers = max_workers;
    let worker_baseline = DEFAULT_MAX_DECODE_BYTES - remaining / workers;
    // The caller's owners and the job tables stay live beside the workers.
    let shared_bytes = WorkspaceBytes {
        channels: tile_ctx.retained_channel_bytes().ok()?,
        idwt: tile_ctx.idwt_capacity_bytes().ok()?,
        tier1: tile_ctx.tier1_capacity_bytes().ok()?,
        structure: budget.live_bytes() - retained_decode_baseline,
        coefficients: 0,
    };
    let round_irreversible_output = tile_ctx.round_irreversible_output;
    #[cfg(test)]
    let force_per_sub_band = tile_ctx.force_per_sub_band;
    for plane in &mut tile_ctx.channel_data {
        split_rows(plane.container.get_mut(..samples)?, width, &mut jobs)?;
    }

    let queue = Mutex::new(jobs.into_iter());
    let failed = AtomicBool::new(false);
    let totals = (0..workers)
        .into_par_iter()
        .with_max_len(1)
        .map(|_| {
            let mut ctx = TileDecodeContext {
                #[cfg(test)]
                force_per_sub_band,
                round_irreversible_output,
                ..TileDecodeContext::default()
            };
            let mut storage = DecompositionStorage::default();
            while !failed.load(Ordering::Relaxed) {
                let Some((mut job, unclaimed)) = queue
                    .lock()
                    .ok()
                    .and_then(|mut queue| Some((queue.next()?, queue.len())))
                else {
                    break;
                };
                // With a thread for every worker and a tile for each, inner
                // fork-joins only add overhead.
                let inner = if workers < threads || unclaimed + 1 < workers {
                    CpuDecodeParallelism::Auto
                } else {
                    CpuDecodeParallelism::Serial
                };
                let decoded = tiles.get(job.index).is_some_and(|tile| {
                    decode_tile(
                        tile,
                        header,
                        &mut ctx,
                        &mut storage,
                        &mut None,
                        inner,
                        false,
                        &mut DecodeProfileTimings::default(),
                        worker_baseline,
                        Some(&mut job),
                    )
                    .is_ok()
                });
                if !decoded {
                    failed.store(true, Ordering::Relaxed);
                    return None;
                }
            }
            Some(Totals {
                peak: ctx.debug_counters.workspace_peak.bytes(),
                counters: ctx.debug_counters,
                stored_plane_samples: ctx.stored_plane_samples,
            })
        })
        .try_reduce(Totals::default, |a, b| Some(a.plus(b)))?;
    // A poisoned queue stops the workers before every tile is claimed.
    if queue.into_inner().ok()?.len() != 0 {
        return None;
    }

    add_counts(&mut tile_ctx.debug_counters, &totals.counters);
    tile_ctx
        .debug_counters
        .workspace_peak
        .record(shared_bytes.plus(totals.peak));
    tile_ctx.stored_plane_samples = tile_ctx
        .stored_plane_samples
        .saturating_add(totals.stored_plane_samples);
    Some(())
}

/// Whether a full-image float decode of 1x1 sampled components stores every tile
/// through the fast path of `store` into SIMD-padded planes of at least `samples`.
fn eligible(
    header: &Header<'_>,
    tiles: &[Tile<'_>],
    tile_ctx: &TileDecodeContext,
    samples: usize,
) -> bool {
    let components = header.component_infos.len();
    tile_ctx.output_region.is_none()
        && tile_ctx.last_idwt_level_padding.is_none()
        && tile_ctx.channel_data.len() == components
        && tile_ctx
            .channel_data
            .iter()
            .all(|plane| plane.integer_container.is_none() && plane.container.len() >= samples)
        && header.component_infos.iter().all(|info| {
            info.size_info.horizontal_resolution == 1 && info.size_info.vertical_resolution == 1
        })
        && tiles.iter().all(|tile| {
            tile.component_infos.len() == components && !tile_requires_exact_integer_decode(tile)
        })
}

/// One job per tile with its row-slice table reserved within `budget`, and
/// the samples of the largest tile.
fn plan_jobs<'p>(
    header: &Header<'_>,
    tiles: &[Tile<'_>],
    (width, height): (usize, usize),
    budget: &mut DecodeAllocationBudget,
) -> Option<(Vec<TileRows<'p>>, usize)> {
    let mut jobs = Vec::new();
    budget.reserve_new(&mut jobs, tiles.len()).ok()?;
    let mut largest_tile = 0_usize;
    for (index, tile) in tiles.iter().enumerate() {
        let rect = plan_rect(header, tile, width, height)?;
        let slices = (rect.height() as usize).checked_mul(header.component_infos.len())?;
        let mut rows = Vec::new();
        budget.reserve_new(&mut rows, slices).ok()?;
        push_reserved(&mut jobs, TileRows { index, rect, rows })?;
        let tile_samples = u64::from(tile.rect.width()) * u64::from(tile.rect.height());
        largest_tile = largest_tile.max(usize::try_from(tile_samples).ok()?);
    }
    Some((jobs, largest_tile))
}

/// The plane rectangle every component of `tile` stores, or `None` when the
/// components disagree, or it is empty or leaves the planes.
fn plan_rect(header: &Header<'_>, tile: &Tile<'_>, width: usize, height: usize) -> Option<IntRect> {
    let mut rects = header
        .component_infos
        .iter()
        .map(|info| store_rect(tile, info, header));
    let rect = rects.next()??;
    let fits = rect.x0 < rect.x1
        && rect.y0 < rect.y1
        && rect.x1 as usize <= width
        && rect.y1 as usize <= height;
    (fits && rects.all(|other| other == Some(rect))).then_some(rect)
}

/// Split one plane into the row slices of `jobs`, whose rectangles must come
/// in raster order: bands of equal rows with increasing columns, each band
/// below the one before.
fn split_rows<'p>(plane: &'p mut [f32], width: usize, jobs: &mut [TileRows<'p>]) -> Option<()> {
    let mut lines = plane.chunks_exact_mut(width).enumerate();
    let mut start = 0;
    while let Some(first) = jobs.get(start).map(|job| job.rect) {
        let len = jobs[start..]
            .iter()
            .take_while(|job| (job.rect.y0, job.rect.y1) == (first.y0, first.y1))
            .count();
        let band = &mut jobs[start..start + len];
        for y in first.y0 as usize..first.y1 as usize {
            let (_, mut line) = lines.find(|(row, _)| *row == y)?;
            let mut column = 0;
            for job in band.iter_mut() {
                let (x0, x1) = (job.rect.x0 as usize, job.rect.x1 as usize);
                let (_, rest) = line.split_at_mut_checked(x0.checked_sub(column)?)?;
                let (segment, rest) = rest.split_at_mut_checked(x1 - x0)?;
                push_reserved(&mut job.rows, segment)?;
                (line, column) = (rest, x1);
            }
        }
        start += len;
    }
    Some(())
}

/// Push into capacity reserved, and budgeted, up front.
fn push_reserved<T>(values: &mut Vec<T>, value: T) -> Option<()> {
    if values.len() == values.capacity() {
        return None;
    }
    values.push(value);
    Some(())
}

/// What the workers add to the caller's counts. They run at once, so their
/// workspace peaks add up.
#[derive(Default, Clone, Copy)]
struct Totals {
    counters: DecodeDebugCounters,
    peak: WorkspaceBytes,
    stored_plane_samples: usize,
}

impl Totals {
    fn plus(mut self, other: Self) -> Self {
        add_counts(&mut self.counters, &other.counters);
        self.peak = self.peak.plus(other.peak);
        self.stored_plane_samples = self
            .stored_plane_samples
            .saturating_add(other.stored_plane_samples);
        self
    }
}

fn add_counts(into: &mut DecodeDebugCounters, from: &DecodeDebugCounters) {
    into.decoded_code_blocks += from.decoded_code_blocks;
    into.skipped_code_blocks += from.skipped_code_blocks;
    into.idwt_output_samples += from.idwt_output_samples;
    into.stored_coefficients += from.stored_coefficients;
}
