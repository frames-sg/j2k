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
//!
//! Every band is decoded once, including bands the IDWT never reads, so a
//! decode visits, decodes and rejects the code-blocks the whole-window route
//! does. Workers fail in any order, so a failure reruns whole-window Tier-1
//! serially for the error the whole-window path reports.
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

use super::banded_tier1;
use super::subband::{plan_band_route, routes_tile, BandRoute, BandRouting};
use super::workspace::{TaskScratch, TaskWorkspace};
use super::{
    clips_to_idwt_windows, CpuDecodeParallelism, DecodeProfileTimings, DecompositionStorage,
    Header, Result, Tile, TileDecodeContext,
};
use crate::error::{DecodeError, DecodingError, ValidationError};
use crate::j2c::build::{self, CodeBlockCoding};
use crate::j2c::codestream::WaveletTransform;
use crate::j2c::idwt::{ComponentLines, SubBandRows};
use crate::j2c::rect::IntRect;
use crate::profile;

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
    let routing = plan_band_route(tile, component_idx, tile_ctx, storage, header);
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
        if completed {
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
        Ok(true) => {
            tile_ctx.debug_counters.pool_band_components += 1;
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
) -> Result<bool> {
    let (rings, bands, active_levels) = plan_rings(route, storage, header, component_idx)?;
    // The executor's records live beside the route's plan and scratch while
    // the bands decode.
    route
        .budget()
        .include_bytes(executor_bytes(&rings, &bands)?)?;
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
    let result = slots.and_then(|slots| {
        let run = |rows: &mut dyn SubBandRows| -> Result<()> {
            lines.as_mut().map_or(Ok(()), |lines| lines.run(rows))
        };
        match route.coding() {
            CodeBlockCoding::Classic => run_bands(
                route,
                &rings,
                &bands,
                slots,
                &mut tile_ctx.classic_task_workspaces,
                |index, output, workspace, scratch| {
                    route.decode_classic(index, output, workspace, scratch)
                },
                run,
                profile_enabled,
                tier1_us,
                strand_once,
            ),
            CodeBlockCoding::HighThroughput => run_bands(
                route,
                &rings,
                &bands,
                slots,
                &mut tile_ctx.ht_task_workspaces,
                |index, output, workspace, scratch| {
                    route.decode_ht(index, output, workspace, scratch)
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
        return result;
    };
    match result {
        Ok(true) => lines.finish(tile_ctx, Ok(())).map(|()| true),
        Ok(false) => {
            // Return the rings and plane; the whole-window path refills them.
            match lines.finish(tile_ctx, Err(DecodingError::CodeBlockDecodeFailure.into())) {
                Ok(()) | Err(_) => Ok(false),
            }
        }
        Err(error) => lines.finish(tile_ctx, Err(error)).map(|()| true),
    }
}

/// The bytes the executor holds for `rings` and `bands`, besides the slots:
/// the records, each band's status, slot, queue entry and decoded-list entry,
/// and each ring's cursors and slot list.
fn executor_bytes(rings: &[Ring], bands: &[Band]) -> Result<usize> {
    let per_band = size_of::<Band>()
        + size_of::<Status>()
        + size_of::<usize>()
        + 2 * size_of::<Option<&mut [f32]>>()
        + size_of::<Reverse<(i64, Reverse<usize>, usize)>>();
    let per_ring = size_of::<Ring>()
        + 3 * size_of::<usize>()
        + size_of::<Vec<&mut [f32]>>()
        + size_of::<Option<(usize, &mut [f32])>>();
    let slots: usize = rings.iter().map(|ring| ring.slots).sum();
    bands
        .len()
        .checked_mul(per_band)
        .zip(rings.len().checked_mul(per_ring))
        .and_then(|(bands, rings)| bands.checked_add(rings))
        .and_then(|bytes| bytes.checked_add(slots.checked_mul(size_of::<&mut [f32]>())?))
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// A sub-band's ring of band slots and its bands, in row order.
struct Ring {
    sub_band: usize,
    stored: IntRect,
    slots: usize,
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
    let active_levels = tile_decompositions
        .decompositions
        .len()
        .checked_sub(usize::from(header.skipped_resolution_levels))
        .filter(|levels| *levels > 0)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
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
                let slot_len = range
                    .len()
                    .checked_div(ring.slots)
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
    /// Bands that can start, soonest read first, then largest first.
    /// Entries for bands since started are skipped.
    ready: BinaryHeap<Reverse<(i64, Reverse<usize>, usize)>>,
    /// Each ring's next band to make ready.
    next_ready: Vec<usize>,
    /// How many of each ring's bands, from its first, have been freed.
    freed_prefix: Vec<usize>,
    /// Bands that finished decoding, for the drain to free; bands the IDWT
    /// has since read are skipped.
    decoded: Vec<usize>,
    running: usize,
    freed: usize,
    workspaces: Vec<&'w mut W>,
    /// Workers on the pool.
    active: usize,
    error: Option<DecodeError>,
    /// Whether a wait found a running band with no worker active.
    stranded: bool,
}

struct Shared<'p, 'c, 'w, W> {
    rings: &'p [Ring],
    bands: &'p [Band],
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

    /// Take the next ready band, marked running, with a slot to decode into.
    fn start_next(&self, state: &mut State<'c, 'w, W>) -> Result<Option<(usize, &'c mut [f32])>> {
        while let Some(Reverse((_, _, index))) = state.ready.pop() {
            if state.status[index] == Status::Waiting {
                return self.start(state, index).map(Some);
            }
        }
        Ok(None)
    }

    fn start(&self, state: &mut State<'c, 'w, W>, index: usize) -> Result<(usize, &'c mut [f32])> {
        let slot = match state.owned[index].take() {
            Some(slot) => slot,
            None => state.free_slots[self.bands[index].ring]
                .pop()
                .ok_or(DecodingError::CodeBlockDecodeFailure)?,
        };
        state.status[index] = Status::Running;
        state.running += 1;
        Ok((index, slot))
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

/// Decode `bands` into `slots` on up to `route.workers()` pool threads
/// while `run` drives the IDWT through them on this thread, then decode the
/// bands it didn't read. Returns `false` when a band was stranded.
#[expect(
    clippy::too_many_arguments,
    reason = "the executor takes the plan, its storage and its two drivers explicitly"
)]
fn run_bands<W: TaskScratch>(
    route: &BandRoute<'_>,
    rings: &[Ring],
    bands: &[Band],
    slots: Slots<'_>,
    bank: &mut [TaskWorkspace<W>],
    decode: impl Fn(usize, &mut [f32], &mut W, &mut Vec<f32>) -> Result<()> + Sync,
    run: impl FnOnce(&mut dyn SubBandRows) -> Result<()>,
    profile_enabled: bool,
    tier1_us: &mut u128,
    strand_once: bool,
) -> Result<bool> {
    #[cfg(not(test))]
    let _ = strand_once;
    let workspaces_needed = route.workers() + 1;
    let mut workspaces: Vec<&mut W> = Vec::new();
    workspaces
        .try_reserve_exact(workspaces_needed)
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    workspaces.extend(
        bank.iter_mut()
            .take(workspaces_needed)
            .map(|slot| &mut slot.workspace),
    );
    if workspaces.len() != workspaces_needed {
        return Err(DecodingError::CodeBlockDecodeFailure.into());
    }
    let own_workspace = workspaces
        .pop()
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let mut state = State {
        status: try_filled(bands.len(), Status::Waiting)?,
        held: try_filled_with(bands.len(), || None)?,
        free_slots: slots.0,
        owned: slots.1,
        ready: BinaryHeap::new(),
        next_ready: try_collect(rings.iter().map(|ring| ring.bands.start), rings.len())?,
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
        .try_reserve(bands.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    state
        .decoded
        .try_reserve_exact(bands.len())
        .map_err(|_| DecodingError::HostAllocationFailed)?;
    let shared = Shared {
        rings,
        bands,
        max_workers: route.workers(),
        #[cfg(test)]
        strand_once: core::sync::atomic::AtomicBool::new(strand_once),
        state: Mutex::new(state),
        decoded: Condvar::new(),
    };
    {
        let mut state = shared.lock()?;
        for ring in 0..rings.len() {
            shared.refill(&mut state, ring);
        }
    }
    let decode = &decode;
    let shared = &shared;
    let scoped = rayon::in_place_scope(|scope| -> Result<()> {
        let mut rows = PoolRows {
            shared,
            scope,
            decode,
            workspace: own_workspace,
            clipped_scratch: Vec::new(),
            current: try_filled_with(rings.len(), || None)?,
            next: try_collect(rings.iter().map(|ring| ring.bands.start), rings.len())?,
            profile_enabled,
            tier1_us: 0,
        };
        rows.send_workers()?;
        let result = run(&mut rows).and_then(|()| rows.drain());
        *tier1_us = rows.tier1_us;
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

/// A pool worker: decode ready bands until none is left to start.
fn work<W: TaskScratch>(
    shared: &Shared<'_, '_, '_, W>,
    decode: &(impl Fn(usize, &mut [f32], &mut W, &mut Vec<f32>) -> Result<()> + Sync),
) {
    let Ok(mut state) = shared.lock() else {
        return;
    };
    let Some(workspace) = state.workspaces.pop() else {
        state.active -= 1;
        return;
    };
    let mut clipped_scratch = Vec::new();
    loop {
        let next = if state.error.is_some() {
            Ok(None)
        } else {
            shared.start_next(&mut state)
        };
        let (index, slot) = match next {
            Ok(Some(started)) => started,
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
        slot.fill(0.0);
        let result = decode(
            shared.bands[index].strip,
            slot,
            workspace,
            &mut clipped_scratch,
        );
        if shared.finish(index, slot, result).is_err() {
            return;
        }
        state = match shared.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
    }
    state.workspaces.push(workspace);
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
    clipped_scratch: Vec<f32>,
    /// Each ring's band being read, with its slot.
    current: Vec<Option<(usize, &'c mut [f32])>>,
    /// Each ring's next band to read.
    next: Vec<usize>,
    profile_enabled: bool,
    tier1_us: u128,
}

impl<'p, 'c, W, D> PoolRows<'_, 'p, 'c, '_, W, D>
where
    W: TaskScratch + 'p,
    D: Fn(usize, &mut [f32], &mut W, &mut Vec<f32>) -> Result<()> + Sync,
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
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            match state.status[index] {
                Status::Done => {
                    return state.held[index]
                        .take()
                        .ok_or(DecodingError::CodeBlockDecodeFailure.into());
                }
                Status::Running => {
                    state = self
                        .shared
                        .decoded
                        .wait_timeout(state, STRANDED_CHECK)
                        .map_err(|_| DecodingError::CodeBlockDecodeFailure)?
                        .0;
                    // A failed band stays running; its error ends the wait.
                    if state.error.is_none()
                        && state.status[index] == Status::Running
                        && state.active == 0
                    {
                        state.stranded = true;
                        return Err(DecodingError::CodeBlockDecodeFailure.into());
                    }
                }
                Status::Waiting => {
                    let (index, slot) = self.shared.start(&mut state, index)?;
                    drop(state);
                    self.decode_here(index, slot)?;
                    return Ok(slot);
                }
                Status::Freed => return Err(DecodingError::CodeBlockDecodeFailure.into()),
            }
        }
    }

    fn decode_here(&mut self, index: usize, slot: &mut [f32]) -> Result<()> {
        let start = profile::profile_now(self.profile_enabled);
        slot.fill(0.0);
        let result = (self.decode)(
            self.shared.bands[index].strip,
            slot,
            self.workspace,
            &mut self.clipped_scratch,
        );
        self.tier1_us += profile::elapsed_us(start);
        result?;
        let mut state = self.shared.lock()?;
        Shared::decoded(&mut state, index);
        drop(state);
        Ok(())
    }

    /// After the IDWT, free its bands and decode every band it didn't read,
    /// here and on the pool.
    fn drain(&mut self) -> Result<()> {
        for ring in 0..self.current.len() {
            self.release(ring)?;
        }
        loop {
            let mut state = self.shared.lock()?;
            if let Some(error) = state.error.take() {
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
                Some((index, slot)) => {
                    drop(state);
                    self.send_workers()?;
                    self.decode_here(index, slot)?;
                    self.shared.lock()?.held[index] = Some(slot);
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
    D: Fn(usize, &mut [f32], &mut W, &mut Vec<f32>) -> Result<()> + Sync,
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

    fn budget_fixture(width: u32, height: u32) -> Vec<u8> {
        let mut state = 0x2545_f491_u32;
        let pixels: Vec<u8> = (0..width * height)
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
            1,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 4,
                code_block_width_exp: 0,
                code_block_height_exp: 0,
                ..EncodeOptions::default()
            },
        )
        .expect("encode budget fixture")
    }

    /// The route's plan and clipped scratch stay live beside the executor.
    /// Leave enough simulated headroom for either alone, but not both. No
    /// large retained allocation is needed to exercise the cap boundary.
    #[test]
    fn pool_bands_respect_the_combined_plan_and_executor_budget() {
        let (width, height) = (203, 157);
        let bytes = budget_fixture(width, height);
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("parse budget fixture");
        rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("build budget test pool")
            .install(|| {
                let mut context = DecoderContext::default();
                context.set_output_region(Some((0, 0, width, height)));
                context.tile_decode_context.line_region_idwt = true;
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
                // Stop at the tile boundary so its parsed block storage stays
                // available. This also prepares the retained worker bank, so
                // the admission check below needs no replacement bank.
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
                .expect("decode with ample headroom");
                assert_eq!(tile_ctx.debug_counters.pool_band_components, 1);
                let (route_bytes, executor) = {
                    let BandRouting::Route(route) =
                        plan_band_route(&tiles[0], 0, tile_ctx, storage, &image.header)
                            .expect("plan control route")
                    else {
                        panic!("control fixture must take the pool route");
                    };
                    // Reuse the existing peak accounting for the route's owned
                    // tables and clipped scratch, excluding the retained bank.
                    let retained_tier1 = tile_ctx.tier1_capacity_bytes().expect("bank bytes");
                    tile_ctx.debug_counters.workspace_peak = WorkspacePeak::default();
                    route
                        .record(tile_ctx, storage)
                        .expect("record route owners");
                    let route_bytes =
                        tile_ctx.debug_counters.workspace_peak.bytes().tier1 - retained_tier1;
                    let (rings, bands, _) =
                        plan_rings(&route, storage, &image.header, 0).expect("plan control rings");
                    (
                        route_bytes,
                        executor_bytes(&rings, &bands).expect("executor bytes"),
                    )
                };
                let headroom = route_bytes + executor / 2;
                assert!(headroom >= executor, "each owner must fit separately");
                let baseline = DecodeAllocationBudget::for_storage(storage)
                    .expect("control budget")
                    .live_bytes();
                storage.structural_workspace_bytes +=
                    DEFAULT_MAX_DECODE_BYTES - baseline - headroom;
                let route = match plan_band_route(&tiles[0], 0, tile_ctx, storage, &image.header) {
                    Ok(BandRouting::Route(route)) => route,
                    Ok(BandRouting::Serial | BandRouting::WholeWindows) => return,
                    Err(error) if super::super::reuse::is_capacity_error(&error) => return,
                    Err(error) => panic!("unexpected planning error: {error:?}"),
                };
                let result = decode_route(
                    &route,
                    0,
                    &image.header,
                    tile_ctx,
                    storage,
                    tiles[0].component_infos[0].wavelet_transform(),
                    false,
                    &mut 0,
                );
                match result {
                    Err(error) if super::super::reuse::is_capacity_error(&error) => {}
                    Ok(false) => {}
                    other => panic!(
                        "pool route must decline or reject combined live owners: \
                         headroom={headroom}, route={route_bytes}, executor={executor}, \
                         excess={}, result={other:?}",
                        route_bytes + executor - headroom,
                    ),
                }
            });
    }
}
