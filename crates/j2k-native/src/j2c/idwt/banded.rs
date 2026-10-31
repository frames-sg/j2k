// SPDX-License-Identifier: MIT OR Apache-2.0

//! Streamed `2D_SR` (Figure F.6).
//!
//! The plain path interleaves a whole level, filters every row, and then
//! sweeps the whole plane once per vertical lifting step. For large planes each
//! sweep goes to DRAM. Here a level is reconstructed top to bottom: when a row
//! is synthesized horizontally, every vertical lifting step whose inputs are now
//! complete runs on the row it updates, so only a few rows are live and they
//! stay in cache. On one thread the rows are reconstructed in the output itself.
//! In parallel, each worker streams a strip of rows through a small ring buffer,
//! starting `lifting steps` rows above the strip and ending as many below it.
//! Those halo rows lack a neighbor at the strip edge and are left stale; each
//! step can only spread that error by one row, so the strip's rows see exactly
//! the neighbors the plane-wide sweep would. Every sample goes through the same
//! expressions in the same order as the plain path, so the output is
//! bit-identical.

use alloc::vec::Vec;
use core::ops::Range;

use super::super::build::Decomposition;
use super::super::codestream::WaveletTransform;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::filter_common::{periodic_symmetric_extension_left, periodic_symmetric_extension_right};
use super::horizontal::filter_row;
use super::interleave::interleave_row;
use super::model::{IDWTInput, IDWTTempOutput};
use super::try_grow_level_buffer;
#[cfg(feature = "parallel")]
use crate::error::{bail, DecodingError};
use crate::math::{self, dispatch, f32x8, Level, Simd, SIMD_WIDTH};
use crate::{checked_decode_usize_product2, Result, ValidationError};
use j2k_codec_math::dwt;

mod last_level;

pub(crate) use last_level::{pack_last_level, LastLevel};

/// Halo rows needed on each side of a band: one per vertical lifting step.
const HALO_ROWS_53: usize = 2;
const HALO_ROWS_97: usize = 4;
const MAX_HALO_ROWS: usize = HALO_ROWS_97;

/// Levels smaller than this run the plain path; banding saves nothing there.
const MIN_BANDED_SAMPLES: usize = 64 * 64;
/// Levels with fewer samples than this stay on one thread.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SAMPLES: usize = 256 * 256;

/// Halo rows a strip worker keeps: one per lifting step, shared by the rows
/// above and below the strip, with room for both neighbors of a row.
const RING_ROWS: usize = MAX_HALO_ROWS + 2;

/// Strips shorter than this would spend too much on their halos.
#[cfg(feature = "parallel")]
const MIN_STRIP_ROWS: usize = 64;

/// Scratch elements for the two horizontally lifted bands of one row.
fn horizontal_scratch_elements(width: usize) -> Result<usize> {
    if width < MIN_SIMD_ROW_WIDTH {
        return Ok(0);
    }
    width
        .checked_add(4)
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// Scratch elements one parallel strip worker needs: its vertical ring and
/// the two ghosted horizontal bands it reuses for every synthesized row.
fn worker_scratch_elements(width: usize) -> Result<usize> {
    let ring = RING_ROWS
        .checked_mul(width)
        .ok_or(ValidationError::ImageTooLarge)?;
    ring.checked_add(horizontal_scratch_elements(width)?)
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// Banded scratch one execution needs for this level. Serial reconstruction
/// keeps only the two ghosted horizontal bands; parallel reconstruction keeps
/// one complete worker slab for every strip that can run concurrently.
pub(crate) fn banded_scratch_elements_for_execution(
    width: usize,
    height: usize,
    parallel: bool,
) -> Result<usize> {
    if !use_banded(width, height) {
        return Ok(0);
    }
    let Some(layout) = parallel_strip_layout(width, height, parallel)? else {
        return horizontal_scratch_elements(width);
    };
    worker_scratch_elements(width)?
        .checked_mul(layout.concurrent_strips)
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// Scratch the build plan reserves for a component whose largest active level
/// is `width` x `height`. The normal decoder may use the Rayon pool, so plan
/// for parallel execution when that path is available.
pub(crate) fn planned_band_scratch_elements(width: usize, height: usize) -> Result<usize> {
    banded_scratch_elements_for_execution(width, height, true)
}

/// Band scratch the build plan reserves for a component whose two finest
/// active levels `inner` and `outer`, whose LL band `inner` reconstructs,
/// `idwt::apply` runs as one pass on the pool: what [`filter_levels_pair`]
/// grows its scratch to. `None` when the levels do not pair. Planned for
/// parallel execution, as [`planned_band_scratch_elements`] is.
pub(crate) fn planned_pair_band_scratch_elements(
    inner: IntRect,
    outer: IntRect,
    transform: WaveletTransform,
) -> Result<Option<usize>> {
    #[cfg(feature = "parallel")]
    {
        if !pair_runs_on_pool(inner, outer, true) {
            return Ok(None);
        }
        let (width, height) = (outer.width() as usize, outer.height() as usize);
        let Some(layout) = parallel_strip_layout(width, height, true)? else {
            return Ok(None);
        };
        let steps = lifting_steps(transform);
        PairScratch::new(inner.width() as usize, width, layout, steps).map(|pair| Some(pair.len))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (inner, outer, transform);
        Ok(None)
    }
}

/// Workers that may each hold ring scratch at once.
#[cfg(feature = "parallel")]
fn band_workers() -> usize {
    rayon::current_num_threads().max(1)
}

#[derive(Clone, Copy)]
#[cfg_attr(
    not(feature = "parallel"),
    expect(
        dead_code,
        reason = "the serial build retains the parallel strip result type for one shared planner signature"
    )
)]
struct ParallelStripLayout {
    strip_len: usize,
    concurrent_strips: usize,
}

/// Parallel strip geometry and the maximum number of streams Rayon can run
/// at once. `None` selects the serial stream.
#[cfg_attr(
    not(feature = "parallel"),
    expect(
        clippy::unnecessary_wraps,
        reason = "the serial stub preserves the fallible parallel geometry signature"
    )
)]
fn parallel_strip_layout(
    width: usize,
    height: usize,
    parallel: bool,
) -> Result<Option<ParallelStripLayout>> {
    #[cfg(feature = "parallel")]
    {
        let required_len = width
            .checked_mul(height)
            .ok_or(ValidationError::ImageTooLarge)?;
        let workers = if parallel && required_len >= MIN_PARALLEL_SAMPLES {
            band_workers().min(height / MIN_STRIP_ROWS)
        } else {
            1
        };
        if workers <= 1 {
            return Ok(None);
        }
        let strip_rows = height.div_ceil(2 * workers).max(MIN_STRIP_ROWS);
        let strip_len = strip_rows
            .checked_mul(width)
            .ok_or(ValidationError::ImageTooLarge)?;
        let strip_count = height.div_ceil(strip_rows);
        Ok(Some(ParallelStripLayout {
            strip_len,
            concurrent_strips: band_workers().min(strip_count),
        }))
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (width, height, parallel);
        Ok(None)
    }
}

/// Whether `filter_2d_banded` streams a level of this size.
pub(super) fn use_banded(width: usize, height: usize) -> bool {
    width >= SIMD_WIDTH
        && height > 2 * MAX_HALO_ROWS
        && width.saturating_mul(height) >= MIN_BANDED_SAMPLES
}

/// Where the rows of one parity come from: the two sub-bands interleaved
/// along the row, and the first output row they fill.
#[derive(Clone, Copy)]
struct RowSource<'a> {
    first: &'a [f32],
    second: &'a [f32],
    first_width: usize,
    second_width: usize,
    /// Band rows missing from the front of `first` and `second`.
    first_skip: usize,
    second_skip: usize,
    start_row: usize,
    rows: usize,
}

impl<'a> RowSource<'a> {
    fn row(&self, global_row: usize) -> Option<(&'a [f32], &'a [f32])> {
        let offset = global_row.checked_sub(self.start_row)?;
        if offset % 2 != 0 {
            return None;
        }
        let index = offset / 2;
        if index >= self.rows {
            return None;
        }
        let band_row = |band: &'a [f32], width: usize, skip: usize| {
            band.get(index.checked_sub(skip)? * width..)?.get(..width)
        };
        Some((
            band_row(self.first, self.first_width, self.first_skip)?,
            band_row(self.second, self.second_width, self.second_skip)?,
        ))
    }
}

#[derive(Clone, Copy)]
struct Level2d<'a> {
    even: RowSource<'a>,
    odd: RowSource<'a>,
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    transform: WaveletTransform,
    high_pass: f32,
}

impl<'a> Level2d<'a> {
    /// The level `rect` of packed LL, HL, LH, and HH bands (row-major, each
    /// as wide as its rectangle in [`packed_band_rects`]).
    #[expect(
        clippy::similar_names,
        reason = "paired LL, HL, LH, and HH band names follow JPEG 2000 specification notation"
    )]
    fn new(
        rect: IntRect,
        [ll, hl, lh, hh]: [&'a [f32]; 4],
        transform: WaveletTransform,
        high_pass: f32,
    ) -> Self {
        // See Figure F.8 and `interleave_samples`.
        let (u0, u1, v0, v1) = (rect.x0, rect.x1, rect.y0, rect.y1);
        let num_u_low = (u1.div_ceil(2) - u0.div_ceil(2)) as usize;
        let num_u_high = (u1 / 2 - u0 / 2) as usize;
        let num_v_low = (v1.div_ceil(2) - v0.div_ceil(2)) as usize;
        let num_v_high = (v1 / 2 - v0 / 2) as usize;
        let (first_width, second_width) = if u0 % 2 == 0 {
            (num_u_low, num_u_high)
        } else {
            (num_u_high, num_u_low)
        };
        let (first_even, second_even) = if u0 % 2 == 0 { (ll, hl) } else { (hl, ll) };
        let (first_odd, second_odd) = if u0 % 2 == 0 { (lh, hh) } else { (hh, lh) };
        Self {
            even: RowSource {
                first: first_even,
                second: second_even,
                first_width,
                second_width,
                first_skip: 0,
                second_skip: 0,
                start_row: usize::from(v0 % 2 != 0),
                rows: num_v_low,
            },
            odd: RowSource {
                first: first_odd,
                second: second_odd,
                first_width,
                second_width,
                first_skip: 0,
                second_skip: 0,
                start_row: usize::from(v0 % 2 == 0),
                rows: num_v_high,
            },
            width: rect.width() as usize,
            height: rect.height() as usize,
            x0: u0 as usize,
            y0: v0 as usize,
            transform,
            high_pass,
        }
    }

    /// This level with its LL band rows `first_row..` taken from `rows`.
    #[cfg(feature = "parallel")]
    fn with_ll_rows(mut self, rows: &'a [f32], first_row: usize) -> Self {
        if self.x0.is_multiple_of(2) {
            self.even.first = rows;
            self.even.first_skip = first_row;
        } else {
            self.even.second = rows;
            self.even.second_skip = first_row;
        }
        self
    }
}

pub(super) fn filter_2d_banded(
    input: IDWTInput<'_>,
    decomposition: &Decomposition,
    target: &mut Vec<f32>,
    band_scratch: &mut Vec<f32>,
    transform: WaveletTransform,
    storage: &DecompositionStorage<'_>,
    parallel: bool,
) -> Result<IDWTTempOutput> {
    let band = |index: usize| {
        &storage.coefficients[storage.sub_bands[decomposition.sub_bands[index]]
            .coefficients
            .clone()]
    };
    let required_len = checked_decode_usize_product2(
        decomposition.rect.width() as usize,
        decomposition.rect.height() as usize,
    )?;
    try_grow_level_buffer(target, required_len)?;
    filter_2d_banded_from_bands(
        decomposition.rect,
        [input.coefficients, band(0), band(1), band(2)],
        target,
        band_scratch,
        transform,
        super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
        parallel,
    )?;
    Ok(IDWTTempOutput {
        rect: decomposition.rect,
    })
}

/// Rectangles of the packed LL, HL, LH, and HH sub-bands of a level (Figure
/// F.8), in the order [`filter_2d_banded_from_bands`] takes them.
pub(super) fn packed_band_rects(rect: IntRect) -> [IntRect; 4] {
    let (u0, u1, v0, v1) = (rect.x0, rect.x1, rect.y0, rect.y1);
    let low = |a: u32, b: u32| (a.div_ceil(2), b.div_ceil(2));
    let high = |a: u32, b: u32| (a / 2, b / 2);
    let band = |(x0, x1): (u32, u32), (y0, y1): (u32, u32)| IntRect::from_ltrb(x0, y0, x1, y1);
    [
        band(low(u0, u1), low(v0, v1)),
        band(high(u0, u1), low(v0, v1)),
        band(low(u0, u1), high(v0, v1)),
        band(high(u0, u1), high(v0, v1)),
    ]
}

/// Reconstruct the level `rect` from its packed LL, HL, LH, and HH bands
/// (row-major, each as wide as its rectangle in [`packed_band_rects`]).
pub(super) fn filter_2d_banded_from_bands(
    rect: IntRect,
    bands: [&[f32]; 4],
    target: &mut [f32],
    band_scratch: &mut Vec<f32>,
    transform: WaveletTransform,
    high_pass: f32,
    parallel: bool,
) -> Result<()> {
    let width = rect.width() as usize;
    let height = rect.height() as usize;
    let required_len = checked_decode_usize_product2(width, height)?;
    let level = Level2d::new(rect, bands, transform, high_pass);

    let target = &mut target[..required_len];
    let Some(parallel_layout) = parallel_strip_layout(width, height, parallel)? else {
        let scratch_len = horizontal_scratch_elements(width)?;
        try_grow_level_buffer(band_scratch, scratch_len)?;
        dispatch!(Level::new(), simd => stream_level(
            simd,
            level,
            target,
            &mut band_scratch[..scratch_len],
        ));
        return Ok(());
    };
    #[cfg(not(feature = "parallel"))]
    let _ = parallel_layout;

    #[cfg(feature = "parallel")]
    {
        let per_worker = worker_scratch_elements(width)?;
        let scratch_len = per_worker
            .checked_mul(parallel_layout.concurrent_strips)
            .ok_or(ValidationError::ImageTooLarge)?;
        try_grow_level_buffer(band_scratch, scratch_len)?;
        dispatch!(Level::new(), simd => run_strips(
            simd,
            level,
            target,
            parallel_layout.strip_len,
            &mut band_scratch[..scratch_len],
            per_worker,
        ));
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn run_strips<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    target: &mut [f32],
    strip_len: usize,
    scratch: &mut [f32],
    per_worker: usize,
) {
    use rayon::prelude::*;
    use std::sync::Mutex;

    // Each worker takes the next strip as it finishes the last, so slower
    // cores do not hold up a wave of equal strips.
    let strips = Mutex::new(target.chunks_mut(strip_len).enumerate());
    let next = || strips.lock().ok().and_then(|mut strips| strips.next());
    scratch
        .par_chunks_mut(per_worker)
        .with_max_len(1)
        .for_each(|worker_scratch| {
            // Rayon invokes this closure outside the caller's target-feature
            // scope. Re-enter it so AVX2 operations inline into the strip,
            // instead of making an out-of-line call for each vector operation.
            #[cfg(feature = "simd")]
            simd.vectorize(
                #[inline(always)]
                || {
                    while let Some((strip, output)) = next() {
                        let first_row = strip * strip_len / level.width;
                        stream_strip(simd, level, first_row, output, worker_scratch);
                    }
                },
            );
            #[cfg(not(feature = "simd"))]
            while let Some((strip, output)) = next() {
                let first_row = strip * strip_len / level.width;
                stream_strip(simd, level, first_row, output, worker_scratch);
            }
        });
}

/// Whether [`filter_2d_banded_pair`] can reconstruct `outer`, whose LL band
/// `inner` reconstructs, together with `inner` on the pool.
#[cfg(feature = "parallel")]
pub(super) fn pair_runs_on_pool(inner: IntRect, outer: IntRect, parallel: bool) -> bool {
    let size = |rect: IntRect| (rect.width() as usize, rect.height() as usize);
    let ((inner_width, inner_height), (width, height)) = (size(inner), size(outer));
    size(packed_band_rects(outer)[0]) == (inner_width, inner_height)
        && use_banded(inner_width, inner_height)
        && use_banded(width, height)
        && matches!(parallel_strip_layout(width, height, parallel), Ok(Some(_)))
}

/// Reconstruct `inner` and then `outer`, whose LL band `inner` reconstructs,
/// into `target` in one pass on the pool, without waiting for all of
/// `inner` before `outer` starts: each strip of `outer` first streams the
/// rows of `inner` it reads into its worker's scratch, as a strip of
/// [`stream_strip`] with its own halo, and then streams its own rows. Those
/// `inner` rows go through the same expressions as in a pass over the whole
/// level, so the output is bit-identical to two level passes.
#[cfg(feature = "parallel")]
pub(super) fn filter_2d_banded_pair(
    input: IDWTInput<'_>,
    [inner, outer]: [&Decomposition; 2],
    target: &mut [f32],
    band_scratch: &mut Vec<f32>,
    transform: WaveletTransform,
    storage: &DecompositionStorage<'_>,
) -> Result<IDWTTempOutput> {
    let band = |decomposition: &Decomposition, index: usize| {
        &storage.coefficients[storage.sub_bands[decomposition.sub_bands[index]]
            .coefficients
            .clone()]
    };
    let high_pass = super::OPENJPEG_NORMALIZED_HIGH_PASS_F32;
    let inner_level = Level2d::new(
        inner.rect,
        [
            input.coefficients,
            band(inner, 0),
            band(inner, 1),
            band(inner, 2),
        ],
        transform,
        high_pass,
    );
    let outer_level = Level2d::new(
        outer.rect,
        [&[], band(outer, 0), band(outer, 1), band(outer, 2)],
        transform,
        high_pass,
    );
    filter_levels_pair([inner_level, outer_level], target, band_scratch)?;
    Ok(IDWTTempOutput { rect: outer.rect })
}

/// [`filter_2d_banded_pair`] of `outer`, whose LL band `inner` reconstructs.
#[cfg(feature = "parallel")]
fn filter_levels_pair(
    [inner_level, outer_level]: [Level2d<'_>; 2],
    target: &mut [f32],
    band_scratch: &mut Vec<f32>,
) -> Result<()> {
    let (width, height) = (outer_level.width, outer_level.height);
    let required_len = checked_decode_usize_product2(width, height)?;
    let target = target
        .get_mut(..required_len)
        .ok_or(DecodingError::OutputBufferTooSmall)?;
    let Some(layout) = parallel_strip_layout(width, height, true)? else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    let steps = lifting_steps(outer_level.transform);
    let scratch = PairScratch::new(inner_level.width, width, layout, steps)?;
    try_grow_level_buffer(band_scratch, scratch.len)?;
    dispatch!(Level::new(), simd => run_pair_strips(
        simd,
        [inner_level, outer_level],
        target,
        layout.strip_len,
        &mut band_scratch[..scratch.len],
        [scratch.per_worker, scratch.rows_len],
    ));
    Ok(())
}

/// The band scratch of [`filter_levels_pair`] for an outer level `width`
/// wide on `layout`, whose LL band is a level `inner_width` wide, with
/// `steps` vertical lifting steps: every concurrent strip's worker holds the
/// LL rows its strip reads and the wider level's worker scratch.
#[cfg(feature = "parallel")]
#[derive(Clone, Copy)]
struct PairScratch {
    /// One worker's elements.
    per_worker: usize,
    /// The LL rows at the front of each worker's elements.
    rows_len: usize,
    /// Every concurrent worker's elements.
    len: usize,
}

#[cfg(feature = "parallel")]
impl PairScratch {
    fn new(
        inner_width: usize,
        width: usize,
        layout: ParallelStripLayout,
        steps: usize,
    ) -> Result<Self> {
        let rows_len = pair_ll_rows(layout.strip_len / width, steps)
            .checked_mul(inner_width)
            .ok_or(ValidationError::ImageTooLarge)?;
        let per_worker = worker_scratch_elements(width)?
            .max(worker_scratch_elements(inner_width)?)
            .checked_add(rows_len)
            .ok_or(ValidationError::ImageTooLarge)?;
        let len = per_worker
            .checked_mul(layout.concurrent_strips)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(Self {
            per_worker,
            rows_len,
            len,
        })
    }
}

/// Rows of the LL band a strip of `strip_rows` rows reads, at most: its
/// arrivals start and end `count` rows beyond it, two output rows per band
/// row, and one band row of slack at each end.
#[cfg(feature = "parallel")]
const fn pair_ll_rows(strip_rows: usize, count: usize) -> usize {
    strip_rows.div_ceil(2) + count + 3
}

#[cfg(feature = "parallel")]
#[inline(always)]
fn run_pair_strips<S: Simd>(
    simd: S,
    levels: [Level2d<'_>; 2],
    target: &mut [f32],
    strip_len: usize,
    scratch: &mut [f32],
    [per_worker, rows_len]: [usize; 2],
) {
    use rayon::prelude::*;
    use std::sync::Mutex;

    let strips = Mutex::new(target.chunks_mut(strip_len).enumerate());
    let next = || strips.lock().ok().and_then(|mut strips| strips.next());
    let width = levels[1].width;
    scratch
        .par_chunks_mut(per_worker)
        .with_max_len(1)
        .for_each(|worker_scratch| {
            // As in `run_strips`.
            #[cfg(feature = "simd")]
            simd.vectorize(
                #[inline(always)]
                || {
                    while let Some((strip, output)) = next() {
                        let first_row = strip * strip_len / width;
                        pair_strip(simd, levels, first_row, output, worker_scratch, rows_len);
                    }
                },
            );
            #[cfg(not(feature = "simd"))]
            while let Some((strip, output)) = next() {
                let first_row = strip * strip_len / width;
                pair_strip(simd, levels, first_row, output, worker_scratch, rows_len);
            }
        });
}

/// Reconstruct rows `first_row..` of `outer` into `output`, after the rows
/// of `inner` they read.
#[cfg(feature = "parallel")]
#[expect(
    clippy::inline_always,
    reason = "the strip kernel is specialized at the architecture dispatch boundary"
)]
#[inline(always)]
fn pair_strip<S: Simd>(
    simd: S,
    [inner, outer]: [Level2d<'_>; 2],
    first_row: usize,
    output: &mut [f32],
    worker_scratch: &mut [f32],
    rows_len: usize,
) {
    let (_, count) = vertical_steps(outer);
    let end = first_row + output.len() / outer.width;
    // The arrivals of `stream_strip` and the LL band rows they read.
    let lo = first_row.saturating_sub(count);
    let hi = (end + count).min(outer.height);
    let first_ll = (lo / 2).saturating_sub(1);
    let end_ll = (hi.div_ceil(2) + 1).min(inner.height);
    // `pair_ll_rows` bounds `end_ll - first_ll` for every strip.
    let (rows, strip_scratch) = worker_scratch.split_at_mut(rows_len);
    let rows = &mut rows[..(end_ll - first_ll) * inner.width];
    stream_strip(simd, inner, first_ll, rows, strip_scratch);
    stream_strip(
        simd,
        outer.with_ll_rows(rows, first_ll),
        first_row,
        output,
        strip_scratch,
    );
}

/// One vertical lifting step: the rows it updates have parity `parity`.
#[derive(Clone, Copy)]
enum VerticalStep {
    /// Equation (F-5).
    Reversible53Low,
    /// Equation (F-6).
    Reversible53High,
    /// A 9-7I lifting step with this coefficient.
    Irreversible97(f32),
}

/// The vertical lifting steps of a level in order, each with the parity of
/// the rows it updates, and how many there are.
fn vertical_steps(level: Level2d<'_>) -> ([(usize, VerticalStep); MAX_HALO_ROWS], usize) {
    let low = level.y0 % 2;
    let high = 1 - low;
    match level.transform {
        WaveletTransform::Reversible53 => (
            [
                (low, VerticalStep::Reversible53Low),
                (high, VerticalStep::Reversible53High),
                (low, VerticalStep::Reversible53Low),
                (high, VerticalStep::Reversible53High),
            ],
            HALO_ROWS_53,
        ),
        WaveletTransform::Irreversible97 => (
            [
                (low, VerticalStep::Irreversible97(dwt::IDWT97_NEG_DELTA_F32)),
                (
                    high,
                    VerticalStep::Irreversible97(dwt::IDWT97_NEG_GAMMA_F32),
                ),
                (low, VerticalStep::Irreversible97(dwt::IDWT97_NEG_BETA_F32)),
                (
                    high,
                    VerticalStep::Irreversible97(dwt::IDWT97_NEG_ALPHA_F32),
                ),
            ],
            HALO_ROWS_97,
        ),
    }
}

/// How many vertical lifting steps [`vertical_steps`] gives a level of
/// `transform`.
#[cfg(feature = "parallel")]
const fn lifting_steps(transform: WaveletTransform) -> usize {
    match transform {
        WaveletTransform::Reversible53 => HALO_ROWS_53,
        WaveletTransform::Irreversible97 => HALO_ROWS_97,
    }
}

/// Synthesize level row `index` horizontally into `row`, and for 9-7I apply
/// its whole-row scaling (steps 1 and 2).
#[expect(
    clippy::inline_always,
    reason = "specialized with the streaming kernels"
)]
#[inline(always)]
fn synthesize_level_row<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    index: usize,
    row: &mut [f32],
    low: &mut [f32],
    high: &mut [f32],
) {
    let row_scale = if level.transform == WaveletTransform::Irreversible97 {
        if index % 2 == level.y0 % 2 {
            dwt::DWT97_KAPPA_F32
        } else {
            level.high_pass
        }
    } else {
        1.0
    };
    let source = if index % 2 == level.even.start_row % 2 {
        level.even.row(index)
    } else {
        level.odd.row(index)
    };
    match source {
        Some((first, second)) => {
            synthesize_row(simd, level, first, second, row, low, high, row_scale);
        }
        // Lifting maps an all-zero row to zeros.
        None => row.fill(0.0),
    }
}

/// [`synthesize_level_row`] restricted to `columns`, one of the level's
/// [`column_strips`], into `out`.
#[expect(
    clippy::inline_always,
    reason = "specialized with the streaming kernels"
)]
#[inline(always)]
fn synthesize_level_columns<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    index: usize,
    out: &mut [f32],
    columns: Range<usize>,
    low: &mut [f32],
    high: &mut [f32],
) {
    if columns.len() == level.width {
        synthesize_level_row(simd, level, index, out, low, high);
        return;
    }
    // Strips are only made of even-origin levels (`column_strips`).
    let row_scale = if level.transform == WaveletTransform::Reversible53 {
        1.0
    } else if index % 2 == level.y0 % 2 {
        dwt::DWT97_KAPPA_F32
    } else {
        level.high_pass
    };
    let source = if index % 2 == level.even.start_row % 2 {
        level.even.row(index)
    } else {
        level.odd.row(index)
    };
    match source {
        Some((low_source, high_source)) => synthesize_row_columns(
            simd,
            level,
            low_source,
            high_source,
            out,
            columns,
            low,
            high,
            row_scale,
        ),
        None => out.fill(0.0),
    }
}

/// Reconstruct the whole level top to bottom in `target` on one thread.
///
/// When row `n` arrives it is synthesized horizontally (and scaled, for 9-7I),
/// then vertical step `k` (1-based) runs on row `n - k` if that row has the
/// step's parity. Step `k` on a row needs both neighbors after step `k - 1`
/// and before step `k + 1`: the lower neighbor reached step `k - 1` one
/// arrival earlier, and the upper one takes step `k + 1` later in this same
/// arrival, so every step reads exactly what the plane-wide sweep reads,
/// including the mirrored rows at both edges. Only the last few rows are
/// live, so they stay in cache whatever the width, and no row is synthesized
/// twice or copied.
#[expect(
    clippy::inline_always,
    reason = "the streaming kernel is specialized at the architecture dispatch boundary"
)]
#[inline(always)]
fn stream_level<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    target: &mut [f32],
    horizontal_scratch: &mut [f32],
) {
    let width = level.width;
    let height = level.height;
    let (_, count) = vertical_steps(level);
    let (low, high) = split_horizontal_scratch(level, horizontal_scratch);
    let mut rows = RingRows {
        rows: target,
        width,
        stride: width,
        slots: height,
    };
    for first in (0..height + count).step_by(WAVEFRONT_ARRIVALS) {
        let arrivals = first..(first + WAVEFRONT_ARRIVALS).min(height + count);
        for arrival in arrivals.start..arrivals.end.min(height) {
            synthesize_level_row(simd, level, arrival, rows.row_mut(arrival), low, high);
        }
        lift_arrivals(simd, level, arrivals, 0..height, &mut rows);
    }
}

/// Reconstruct level rows `[first_row, first_row + output.len() / width)` in
/// `output` with the schedule of [`stream_level`], starting and ending one
/// halo row per lifting step beyond the strip. A halo row whose neighbor is
/// outside the halo skips that step; the resulting error stays inside the
/// halo. The strip's own rows are reconstructed in place; halo rows live in
/// the worker's scratch.
#[cfg(feature = "parallel")]
#[expect(
    clippy::inline_always,
    reason = "the strip kernel is specialized at the architecture dispatch boundary"
)]
#[inline(always)]
fn stream_strip<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    first_row: usize,
    output: &mut [f32],
    worker_scratch: &mut [f32],
) {
    let width = level.width;
    let height = level.height;
    let (_, count) = vertical_steps(level);
    let end = first_row + output.len() / width;
    let lo = first_row.saturating_sub(count);
    let hi = (end + count).min(height);
    let (halo, horizontal_scratch) = worker_scratch.split_at_mut(RING_ROWS * width);
    let (low, high) = split_horizontal_scratch(level, horizontal_scratch);
    let mut rows = StripRows {
        output,
        halo,
        first_row,
        end,
        lo,
        width,
    };
    for first in (lo..hi + count).step_by(WAVEFRONT_ARRIVALS) {
        let arrivals = first..(first + WAVEFRONT_ARRIVALS).min(hi + count);
        for arrival in arrivals.start..arrivals.end.min(hi) {
            synthesize_level_row(simd, level, arrival, rows.row_mut(arrival), low, high);
        }
        lift_arrivals(simd, level, arrivals, lo..hi, &mut rows);
    }
}

/// Rows `lo..` of one parallel strip: its own rows `first_row..end` in
/// `output`, and its halo rows in `halo`. The rows above and below the strip
/// share halo slots: every strip with rows below it spans at least
/// [`MIN_STRIP_ROWS`] rows, more than [`WAVEFRONT_ARRIVALS`] plus the halo,
/// so the rows above have taken their last step before the batch that
/// synthesizes the first row below.
#[cfg(feature = "parallel")]
struct StripRows<'a> {
    output: &'a mut [f32],
    halo: &'a mut [f32],
    first_row: usize,
    end: usize,
    lo: usize,
    width: usize,
}

#[cfg(feature = "parallel")]
impl StripRows<'_> {
    /// Whether `row` lies in `output`, and its index in that buffer.
    fn locate(&self, row: usize) -> (bool, usize) {
        if row < self.first_row {
            (false, row - self.lo)
        } else if row < self.end {
            (true, row - self.first_row)
        } else {
            (false, row - self.end)
        }
    }

    fn row_mut(&mut self, row: usize) -> &mut [f32] {
        let (in_output, index) = self.locate(row);
        let rows = if in_output {
            &mut *self.output
        } else {
            &mut *self.halo
        };
        &mut rows[index * self.width..][..self.width]
    }
}

#[cfg(feature = "parallel")]
impl LevelRows for StripRows<'_> {
    fn three(
        &mut self,
        current: usize,
        above: usize,
        below: usize,
    ) -> (&mut [f32], &[f32], &[f32]) {
        let width = self.width;
        let locate = |row| self.locate(row);
        let ((current_in_output, index), above, below) =
            (locate(current), locate(above), locate(below));
        let (same, other): (&mut [f32], &[f32]) = if current_in_output {
            (&mut *self.output, &*self.halo)
        } else {
            (&mut *self.halo, &*self.output)
        };
        let (before, rest) = same.split_at_mut(index * width);
        let (current_row, after) = rest.split_at_mut(width);
        let (before, after): (&[f32], &[f32]) = (before, after);
        let neighbor = |(in_output, neighbor): (bool, usize)| -> &[f32] {
            if in_output != current_in_output {
                &other[neighbor * width..][..width]
            } else if neighbor < index {
                &before[neighbor * width..][..width]
            } else {
                &after[(neighbor - index - 1) * width..][..width]
            }
        };
        (current_row, neighbor(above), neighbor(below))
    }

    fn window<const ROWS: usize>(&mut self, first: usize) -> Option<[&mut [f32]; ROWS]> {
        let width = self.width;
        let rows = first..first + ROWS;
        // The halo rows above the strip, its own rows, and the halo rows
        // below it. The two halo runs share slots, so at most one of them
        // may take part.
        let above = rows.start..rows.end.clamp(rows.start, self.first_row.max(rows.start));
        let own = rows.start.max(self.first_row)..rows.end.min(self.end);
        let below = rows.start.max(self.end)..rows.end.max(self.end);
        let (above, below): (&mut [f32], &mut [f32]) = match (above.is_empty(), below.is_empty()) {
            (true, true) => (&mut [], &mut []),
            (false, true) => (
                self.halo
                    .get_mut((above.start - self.lo) * width..(above.end - self.lo) * width)?,
                &mut [],
            ),
            (true, false) => (
                &mut [],
                self.halo
                    .get_mut((below.start - self.end) * width..(below.end - self.end) * width)?,
            ),
            (false, false) => return None,
        };
        let own: &mut [f32] = if own.is_empty() {
            &mut []
        } else {
            self.output
                .get_mut((own.start - self.first_row) * width..(own.end - self.first_row) * width)?
        };
        filled_window(
            above
                .chunks_exact_mut(width)
                .chain(own.chunks_exact_mut(width))
                .chain(below.chunks_exact_mut(width)),
        )
    }
}

/// Rows of `width` samples, `stride` apart, where level row `n` occupies
/// slot `n % slots`: a whole level in place when `slots` is its height, or
/// the ring of the last level's rows.
struct RingRows<'a> {
    rows: &'a mut [f32],
    width: usize,
    stride: usize,
    slots: usize,
}

impl RingRows<'_> {
    fn row_mut(&mut self, row: usize) -> &mut [f32] {
        &mut self.rows[row % self.slots * self.stride..][..self.width]
    }
}

impl LevelRows for RingRows<'_> {
    fn three(
        &mut self,
        current: usize,
        above: usize,
        below: usize,
    ) -> (&mut [f32], &[f32], &[f32]) {
        let width = self.width;
        let slot = |row: usize| row % self.slots;
        let (current, above, below) = three_rows(
            self.rows,
            self.stride,
            slot(current),
            slot(above),
            slot(below),
        );
        (&mut current[..width], &above[..width], &below[..width])
    }

    fn window<const ROWS: usize>(&mut self, first: usize) -> Option<[&mut [f32]; ROWS]> {
        if ROWS > self.slots {
            return None;
        }
        let (width, stride) = (self.width, self.stride);
        // The rows from the first one's slot to the end, then from slot zero.
        let (wrapped, from_first) = self.rows.split_at_mut(first % self.slots * stride);
        filled_window(
            from_first
                .chunks_mut(stride)
                .chain(wrapped.chunks_mut(stride))
                .filter_map(|row| row.get_mut(..width)),
        )
    }
}

/// The first `ROWS` rows of `rows`, if it has as many.
fn filled_window<'rows, const ROWS: usize>(
    mut rows: impl Iterator<Item = &'rows mut [f32]>,
) -> Option<[&'rows mut [f32]; ROWS]> {
    let mut filled = 0;
    let window = core::array::from_fn(|_| match rows.next() {
        Some(row) => {
            filled += 1;
            row
        }
        None => &mut [],
    });
    (filled == ROWS).then_some(window)
}

/// Rows of a level under reconstruction, by level row index.
trait LevelRows {
    /// Row `current` mutably, with rows `above` and `below`, both different
    /// from `current`, shared.
    fn three(&mut self, current: usize, above: usize, below: usize)
        -> (&mut [f32], &[f32], &[f32]);

    /// The `ROWS` consecutive level rows from `first`, mutably, or `None`
    /// when they cannot all be borrowed at once.
    fn window<const ROWS: usize>(&mut self, first: usize) -> Option<[&mut [f32]; ROWS]>;
}

/// Arrivals [`stream_level`] and [`stream_strip`] synthesize before they run
/// those arrivals' vertical steps, so one sweep of the columns runs the
/// steps of several wavefronts.
const WAVEFRONT_ARRIVALS: usize = 16;

/// Wavefronts one sweep of the columns runs, sharing their rows in
/// registers.
const SWEEP_WAVEFRONTS: usize = 2;

/// Run the vertical lifting steps of `arrivals`, in order, on `rows`, of
/// which only `live` take part (see [`stream_strip`]). Every arrival must
/// already be synthesized.
///
/// An arrival whose row above has the first step's parity runs every step,
/// one row further up each (a wavefront); the others run none. Away from
/// the edges of `live`, those steps read only unmirrored rows inside it, so
/// consecutive wavefronts run together in [`lift_wavefronts`], which gives
/// every sample the same operations in the same order. Arrivals near the
/// edges run their steps one row at a time, as before.
#[expect(
    clippy::inline_always,
    reason = "the vertical schedule is specialized with the streaming kernels"
)]
#[inline(always)]
fn lift_arrivals<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    arrivals: Range<usize>,
    live: Range<usize>,
    rows: &mut impl LevelRows,
) {
    let (steps, count) = vertical_steps(level);
    let steps = &steps[..count];
    // From `lo + count + 1` the deepest step's upper neighbor lies in
    // `live`; before `hi` the first step's lower neighbor does.
    let start = (live.start + count + 1).clamp(arrivals.start, arrivals.end);
    let end = live.end.clamp(start, arrivals.end);
    for arrival in arrivals.start..start {
        lift_arrival(simd, steps, arrival, level.height, &live, rows);
    }
    // `start` is at least one; its row above has its parity.
    let mut wavefront = start + usize::from((start + 1) % 2 != steps[0].0);
    while wavefront < end {
        let wavefronts = if (end - wavefront).div_ceil(2) >= SWEEP_WAVEFRONTS {
            SWEEP_WAVEFRONTS
        } else {
            1
        };
        let first = wavefront - count - 1;
        let lifted = match (level.transform, wavefronts) {
            (WaveletTransform::Reversible53, 1) => lift_wavefronts::<S, 2, 4>(simd, rows, first),
            (WaveletTransform::Reversible53, _) => {
                lift_wavefronts::<S, 2, { 2 * SWEEP_WAVEFRONTS + 2 }>(simd, rows, first)
            }
            (WaveletTransform::Irreversible97, 1) => lift_wavefronts::<S, 4, 6>(simd, rows, first),
            (WaveletTransform::Irreversible97, _) => {
                lift_wavefronts::<S, 4, { 2 * SWEEP_WAVEFRONTS + 4 }>(simd, rows, first)
            }
        };
        if !lifted {
            for arrival in (wavefront..wavefront + 2 * wavefronts).step_by(2) {
                lift_arrival(simd, steps, arrival, level.height, &live, rows);
            }
        }
        wavefront += 2 * wavefronts;
    }
    for arrival in end..arrivals.end {
        lift_arrival(simd, steps, arrival, level.height, &live, rows);
    }
}

/// Run the vertical lifting steps of one arrival a row at a time: step `k`
/// (1-based) on row `arrival - k` when that row has the step's parity and it
/// and both its neighbors (mirrored at the level's edges) lie in `live`.
#[expect(
    clippy::inline_always,
    reason = "the vertical schedule is specialized with the streaming kernels"
)]
#[inline(always)]
fn lift_arrival<S: Simd>(
    simd: S,
    steps: &[(usize, VerticalStep)],
    arrival: usize,
    height: usize,
    live: &Range<usize>,
    rows: &mut impl LevelRows,
) {
    for (index, &(parity, step)) in steps.iter().enumerate() {
        let Some(row) = arrival.checked_sub(index + 1) else {
            break;
        };
        if !live.contains(&row) || row % 2 != parity {
            continue;
        }
        let above = periodic_symmetric_extension_left(row, 1);
        let below = periodic_symmetric_extension_right(row, 1, height);
        if above < live.start || below >= live.end {
            continue;
        }
        let (current, above, below) = rows.three(row, above, below);
        apply_vertical_step(simd, step, current, above, below);
    }
}

/// Run `(ROWS - STEPS) / 2` consecutive wavefronts of a transform with
/// `STEPS` vertical steps on the `ROWS` level rows from `first`, all in the
/// level's interior: wavefront `j` runs step `k` (0-based) on row
/// `first + STEPS + 2 * j - k` from that row's two neighbors. Each column
/// runs the wavefronts in order with their rows in registers, so every
/// sample sees the operations of [`apply_vertical_step`] in the same order
/// while each row is loaded and stored once. Returns `false`, having changed
/// nothing, when the rows cannot be borrowed together.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernel is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_wavefronts<S: Simd, const STEPS: usize, const ROWS: usize>(
    simd: S,
    rows: &mut impl LevelRows,
    first: usize,
) -> bool {
    let Some(mut window) = rows.window::<ROWS>(first) else {
        return false;
    };
    if STEPS == HALO_ROWS_53 {
        lift_window::<S, STEPS, ROWS>(
            simd,
            &mut window,
            #[inline(always)]
            |step, current, above, below| {
                if step == 0 {
                    lift_53_low_simd(current, above, below)
                } else {
                    lift_53_high_simd(current, above, below)
                }
            },
            #[inline(always)]
            |step, current, above, below| {
                if step == 0 {
                    lift_53_low(current, above, below)
                } else {
                    lift_53_high(current, above, below)
                }
            },
        );
    } else {
        let coefficients = [
            dwt::IDWT97_NEG_DELTA_F32,
            dwt::IDWT97_NEG_GAMMA_F32,
            dwt::IDWT97_NEG_BETA_F32,
            dwt::IDWT97_NEG_ALPHA_F32,
        ];
        let coefficients_simd = coefficients.map(|value| f32x8::splat(simd, value));
        lift_window::<S, STEPS, ROWS>(
            simd,
            &mut window,
            #[inline(always)]
            |step, current, above, below| {
                lift_97_simd(current, above, below, coefficients_simd[step])
            },
            #[inline(always)]
            |step, current, above, below| lift_97(current, above, below, coefficients[step]),
        );
    }
    true
}

/// [`lift_wavefronts`] on rows of equal width, `step(k, current, above,
/// below)` being step `k`. Every loop over rows has a constant trip count,
/// so the window and the row pointers stay in registers.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernel is specialized inside each SIMD dispatch target"
)]
#[expect(
    clippy::needless_range_loop,
    reason = "constant-index loops over the window unroll into registers; iterator adaptors keep it in memory"
)]
#[inline(always)]
fn lift_window<S: Simd, const STEPS: usize, const ROWS: usize>(
    simd: S,
    rows: &mut [&mut [f32]; ROWS],
    step_simd: impl Fn(usize, f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    step_scalar: impl Fn(usize, f32, f32, f32) -> f32,
) {
    let wavefronts = (ROWS - STEPS) / 2;
    let width = rows[0].len();
    let blocks = width / SIMD_WIDTH;
    // The scalar tail columns first: columns are independent.
    for column in blocks * SIMD_WIDTH..width {
        let mut window = [0.0; ROWS];
        for row in 0..ROWS {
            window[row] = rows[row][column];
        }
        for wavefront in 0..wavefronts {
            for step in 0..STEPS {
                let row = 2 * wavefront + STEPS - step;
                window[row] = step_scalar(step, window[row], window[row - 1], window[row + 1]);
            }
        }
        // The first and last rows are only read.
        for row in 1..ROWS - 1 {
            rows[row][column] = window[row];
        }
    }
    // Every row as exactly `blocks` SIMD blocks, so no access needs a check.
    let blocks_of = rows
        .each_mut()
        .map(|row| &mut row.as_chunks_mut::<SIMD_WIDTH>().0[..blocks]);
    for block in 0..blocks {
        let mut window = [f32x8::splat(simd, 0.0); ROWS];
        for row in 0..ROWS {
            window[row] = f32x8::from_slice(simd, &blocks_of[row][block]);
        }
        for wavefront in 0..wavefronts {
            for step in 0..STEPS {
                let row = 2 * wavefront + STEPS - step;
                window[row] = step_simd(step, window[row], window[row - 1], window[row + 1]);
            }
        }
        for row in 1..ROWS - 1 {
            window[row].store(&mut blocks_of[row][block]);
        }
    }
}

/// Split one worker's exact horizontal scratch into low- and high-pass bands,
/// each with one ghost sample on either side.
fn split_horizontal_scratch<'scratch>(
    level: Level2d<'_>,
    scratch: &'scratch mut [f32],
) -> (&'scratch mut [f32], &'scratch mut [f32]) {
    if level.width < MIN_SIMD_ROW_WIDTH {
        let (low, rest) = scratch.split_at_mut(0);
        let (high, _) = rest.split_at_mut(0);
        return (low, high);
    }
    let (low_width, high_width) = if level.x0.is_multiple_of(2) {
        (level.even.first_width, level.even.second_width)
    } else {
        (level.even.second_width, level.even.first_width)
    };
    let (low, rest) = scratch.split_at_mut(low_width + 2);
    let (high, _) = rest.split_at_mut(high_width + 2);
    (low, high)
}

/// Row `current` of `rows` (rows of `width` samples) mutably, with rows
/// `above` and `below`, both different from `current`, shared.
fn three_rows(
    rows: &mut [f32],
    width: usize,
    current: usize,
    above: usize,
    below: usize,
) -> (&mut [f32], &[f32], &[f32]) {
    let (before, rest) = rows.split_at_mut(current * width);
    let (current_row, after) = rest.split_at_mut(width);
    let (before, after): (&[f32], &[f32]) = (before, after);
    let neighbor = |index: usize| -> &[f32] {
        if index < current {
            &before[index * width..][..width]
        } else {
            &after[(index - current - 1) * width..][..width]
        }
    };
    (current_row, neighbor(above), neighbor(below))
}

/// Apply one vertical lifting step to `current` from its neighbors.
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn apply_vertical_step<S: Simd>(
    simd: S,
    step: VerticalStep,
    current: &mut [f32],
    above: &[f32],
    below: &[f32],
) {
    match step {
        VerticalStep::Reversible53Low => {
            lift_rows(simd, current, above, below, lift_53_low_simd, lift_53_low);
        }
        VerticalStep::Reversible53High => {
            lift_rows(simd, current, above, below, lift_53_high_simd, lift_53_high);
        }
        VerticalStep::Irreversible97(coefficient) => {
            let coefficient_simd = f32x8::splat(simd, coefficient);
            lift_rows(
                simd,
                current,
                above,
                below,
                #[inline(always)]
                |s1, s2, s3| lift_97_simd(s1, s2, s3, coefficient_simd),
                #[inline(always)]
                |s1, s2, s3| lift_97(s1, s2, s3, coefficient),
            );
        }
    }
}

/// Equation (F-5) on one sample from its two neighbors: the SIMD form.
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_53_low_simd<S: Simd>(s1: f32x8<S>, s2: f32x8<S>, s3: f32x8<S>) -> f32x8<S> {
    s1 - ((s2 + s3 + 2.0) * 0.25).floor()
}

/// Equation (F-5): the scalar form of [`lift_53_low_simd`].
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_53_low(s1: f32, s2: f32, s3: f32) -> f32 {
    s1 - math::floor_f32(math::mul_add(s2 + s3, 0.25, 0.5))
}

/// Equation (F-6) on one sample from its two neighbors: the SIMD form.
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_53_high_simd<S: Simd>(s1: f32x8<S>, s2: f32x8<S>, s3: f32x8<S>) -> f32x8<S> {
    s1 + ((s2 + s3) * 0.5).floor()
}

/// Equation (F-6): the scalar form of [`lift_53_high_simd`].
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_53_high(s1: f32, s2: f32, s3: f32) -> f32 {
    s1 + math::floor_half_sum_f32(s2, s3)
}

/// One 9-7I lifting step on one sample from its two neighbors: the SIMD form.
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_97_simd<S: Simd>(
    s1: f32x8<S>,
    s2: f32x8<S>,
    s3: f32x8<S>,
    coefficient: f32x8<S>,
) -> f32x8<S> {
    (s2 + s3).mul_add(coefficient, s1)
}

/// One 9-7I lifting step: the scalar form of [`lift_97_simd`].
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_97(s1: f32, s2: f32, s3: f32, coefficient: f32) -> f32 {
    math::mul_add(s2 + s3, coefficient, s1)
}

/// Multiply every sample of `samples` by `scale`.
#[expect(
    clippy::inline_always,
    reason = "specialized with the streaming kernels"
)]
#[inline(always)]
pub(in crate::j2c) fn scale_samples<S: Simd>(simd: S, samples: &mut [f32], scale: f32) {
    let scale_simd = f32x8::splat(simd, scale);
    let simd_len = samples.len() / SIMD_WIDTH * SIMD_WIDTH;
    for index in (0..simd_len).step_by(SIMD_WIDTH) {
        let values = f32x8::from_slice(simd, &samples[index..][..SIMD_WIDTH]);
        (values * scale_simd).store(&mut samples[index..][..SIMD_WIDTH]);
    }
    for value in &mut samples[simd_len..] {
        *value *= scale;
    }
}

/// `current[i] = f(current[i], above[i], below[i])` for every sample.
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
pub(in crate::j2c) fn lift_rows<S: Simd>(
    simd: S,
    current: &mut [f32],
    above: &[f32],
    below: &[f32],
    f_simd: impl Fn(f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    f_scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let width = current.len();
    let simd_width = width / SIMD_WIDTH * SIMD_WIDTH;
    for column in (0..simd_width).step_by(SIMD_WIDTH) {
        let s1 = f32x8::from_slice(simd, &current[column..][..SIMD_WIDTH]);
        let s2 = f32x8::from_slice(simd, &above[column..][..SIMD_WIDTH]);
        let s3 = f32x8::from_slice(simd, &below[column..][..SIMD_WIDTH]);
        f_simd(s1, s2, s3).store(&mut current[column..][..SIMD_WIDTH]);
    }
    for column in simd_width..width {
        current[column] = f_scalar(current[column], above[column], below[column]);
    }
}

/// Rows narrower than this use the scalar interleaved `1D_SR`.
const MIN_SIMD_ROW_WIDTH: usize = 2 * SIMD_WIDTH;

/// Horizontal `1D_SR` (F.3.6) of one row from its two band rows, written
/// interleaved into `row`.
///
/// The band rows already hold the row's low-pass (even global column) and
/// high-pass samples apart, so each lifting step runs as contiguous SIMD
/// over one of them. A step updates one parity from the other, which it
/// does not change, so vectorizing it keeps every per-sample expression of
/// `horizontal::filter_row`. Each buffer carries a ghost sample at both ends
/// that mirrors its edge sample: that is the symmetric extension (F.3.7) of
/// the interleaved row seen from the other parity.
#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    reason = "the row synthesis is specialized inside each SIMD dispatch target and keeps its geometry, two sources, output, and exact scratch explicit"
)]
#[inline(always)]
fn synthesize_row<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    first: &[f32],
    second: &[f32],
    row: &mut [f32],
    low: &mut [f32],
    high: &mut [f32],
    row_scale: f32,
) {
    let width = level.width;
    if width < MIN_SIMD_ROW_WIDTH {
        interleave_row(simd, first, second, row);
        filter_row(row, width, level.x0, level.transform, level.high_pass);
        if level.transform == WaveletTransform::Irreversible97 {
            scale_samples(simd, row, row_scale);
        }
        return;
    }
    // Column `x0` is low-pass when it is even; the first band row holds the
    // samples of local column zero.
    let low_first = level.x0.is_multiple_of(2);
    let (low_source, high_source) = if low_first {
        (first, second)
    } else {
        (second, first)
    };
    // Ghost-relative neighbor offsets: a low sample's high neighbors sit at
    // (k - 1, k) when the row starts low and at (k, k + 1) otherwise; a high
    // sample's low neighbors are the other way around.
    let (low_reads, high_reads) = if low_first {
        ((0, 1), (1, 2))
    } else {
        ((1, 2), (0, 1))
    };

    match level.transform {
        WaveletTransform::Reversible53 => {
            // As for 9-7I below, with `lift_row_53_window`.
            let window = if low_first {
                window_pairs_97(high_source.len())
            } else {
                0
            };
            let start = window.saturating_sub(WINDOW_HALO_PAIRS);
            synthesize_53_tail(
                simd,
                [low_source, high_source],
                [low, high],
                [low_reads, high_reads],
                low_first,
                &mut row[2 * start..],
                start,
            );
            if window > 0 {
                let row = &mut row[..2 * window];
                lift_row_53_window(simd, low_source, high_source, row, 0);
            }
        }
        WaveletTransform::Irreversible97 => {
            // An even-origin row lifts its first `window` pairs in one pass
            // (`lift_row_97_window`); the steps below cover the rest, from
            // `WINDOW_HALO_PAIRS` pairs earlier, so their wrong left edge
            // lands only on pairs that pass then overwrites.
            let window = if low_first {
                window_pairs_97(high_source.len())
            } else {
                0
            };
            let start = window.saturating_sub(WINDOW_HALO_PAIRS);
            synthesize_97_tail(
                simd,
                [low_source, high_source],
                [low, high],
                [low_reads, high_reads],
                low_first,
                &mut row[2 * start..],
                start,
                level.high_pass,
                row_scale,
            );
            if window > 0 {
                let row = &mut row[..2 * window];
                lift_row_97_window(
                    simd,
                    low_source,
                    high_source,
                    row,
                    0,
                    level.high_pass,
                    row_scale,
                );
            }
        }
    }
}

/// The multi-pass 5-3R synthesis of a row's pairs from `start` on into
/// `tail`, through ghosted copies of the band rows' remaining samples.
#[expect(
    clippy::inline_always,
    reason = "the 5/3 tail is specialized with the row synthesis inside each SIMD dispatch target"
)]
#[inline(always)]
fn synthesize_53_tail<S: Simd>(
    simd: S,
    [low_source, high_source]: [&[f32]; 2],
    [low, high]: [&mut [f32]; 2],
    [low_reads, high_reads]: [(usize, usize); 2],
    low_first: bool,
    tail: &mut [f32],
    start: usize,
) {
    let low = &mut low[..low_source.len() - start + 2];
    let high = &mut high[..high_source.len() - start + 2];
    load_ghosted(low, &low_source[start..]);
    load_ghosted(high, &high_source[start..]);
    // Equation (F-5), then (F-6).
    let quarter = f32x8::splat(simd, 0.25);
    let half = f32x8::splat(simd, 0.5);
    lift_row(
        simd,
        low,
        high,
        low_reads,
        #[inline(always)]
        |s, a, b| s - (a + b).mul_add(quarter, half).floor(),
        #[inline(always)]
        |s, a, b| s - math::floor_f32(math::mul_add(a + b, 0.25, 0.5)),
    );
    lift_high_to_row::<S, false>(
        simd,
        high,
        low,
        high_reads,
        low_first,
        tail,
        1.0,
        #[inline(always)]
        |s, a, b| s + ((a + b) * 0.5).floor(),
        #[inline(always)]
        |s, a, b| s + math::floor_half_sum_f32(a, b),
    );
}

/// The horizontal 5-3R synthesis of pairs `SIMD_WIDTH * first_block` on, as
/// many as `row` has room for, of an even-origin row from its `low` and
/// `high` band rows, in one pass of `SIMD_WIDTH`-pair blocks: equation (F-5)
/// a block ahead of (F-6), neighbors taken across blocks by `slide`, each
/// block interleaved and stored. A pass from block zero mirrors the row's
/// left edge; one from a later block first lifts that block's low samples
/// from the block before it. Each sample sees the operations of the
/// multi-pass steps in [`synthesize_53_tail`].
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_row_53_window<S: Simd>(
    simd: S,
    low: &[f32],
    high: &[f32],
    row: &mut [f32],
    first_block: usize,
) {
    let steps = [f32x8::splat(simd, 0.25), f32x8::splat(simd, 0.5)];
    let mut high0 = band_block(simd, high, first_block);
    let mut low1 = if first_block == 0 {
        // T.800 F.3.7: high sample -1 mirrors high sample 0.
        let low0 = band_block(simd, low, 0);
        update_53(high0.slide::<1>(high0), high0, low0, steps)
    } else {
        let previous = band_block(simd, high, first_block - 1);
        update_53(previous, high0, band_block(simd, low, first_block), steps)
    };
    for (index, out) in row
        .as_chunks_mut::<{ 2 * SIMD_WIDTH }>()
        .0
        .iter_mut()
        .enumerate()
    {
        let block = first_block + index + 1;
        let next_high0 = band_block(simd, high, block);
        let next_low1 = update_53(high0, next_high0, band_block(simd, low, block), steps);
        let high1 = predict_53(low1, next_low1, high0);
        let (first, second) = out.split_at_mut(SIMD_WIDTH);
        low1.zip_low(high1).store(first);
        low1.zip_high(high1).store(second);
        (high0, low1) = (next_high0, next_low1);
    }
}

/// Equation (F-5) on a block of low samples from the high block before them
/// (`previous`) and their own: low sample `k` lies between high samples
/// `k - 1` and `k`. `[quarter, half]` are splats of 0.25 and 0.5.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn update_53<S: Simd>(
    previous: f32x8<S>,
    high: f32x8<S>,
    low: f32x8<S>,
    [quarter, half]: [f32x8<S>; 2],
) -> f32x8<S> {
    low - (previous.slide::<7>(high) + high)
        .mul_add(quarter, half)
        .floor()
}

/// Equation (F-6) on a block of high samples from the lifted low block with
/// them and the next: high sample `k` lies between low samples `k` and
/// `k + 1`.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn predict_53<S: Simd>(low: f32x8<S>, next: f32x8<S>, high: f32x8<S>) -> f32x8<S> {
    high + ((low + low.slide::<1>(next)) * 0.5).floor()
}

/// Block `block` of `SIMD_WIDTH` samples of `band`.
#[expect(
    clippy::inline_always,
    reason = "the block load is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn band_block<S: Simd>(simd: S, band: &[f32], block: usize) -> f32x8<S> {
    f32x8::from_slice(simd, &band[block * SIMD_WIDTH..][..SIMD_WIDTH])
}

/// The multi-pass 9-7I synthesis of a row's pairs from `start` on into
/// `tail`, through ghosted copies of the band rows' remaining samples.
#[expect(
    clippy::too_many_arguments,
    reason = "the tail keeps its two sources, two scratch bands, reads and scales explicit"
)]
#[inline(always)]
fn synthesize_97_tail<S: Simd>(
    simd: S,
    [low_source, high_source]: [&[f32]; 2],
    [low, high]: [&mut [f32]; 2],
    [low_reads, high_reads]: [(usize, usize); 2],
    low_first: bool,
    tail: &mut [f32],
    start: usize,
    high_pass: f32,
    row_scale: f32,
) {
    const NEG_ALPHA: f32 = dwt::IDWT97_NEG_ALPHA_F32;
    const NEG_BETA: f32 = dwt::IDWT97_NEG_BETA_F32;
    const NEG_GAMMA: f32 = dwt::IDWT97_NEG_GAMMA_F32;
    const NEG_DELTA: f32 = dwt::IDWT97_NEG_DELTA_F32;
    const KAPPA: f32 = dwt::DWT97_KAPPA_F32;
    let low = &mut low[..low_source.len() - start + 2];
    let high = &mut high[..high_source.len() - start + 2];
    load_ghosted(low, &low_source[start..]);
    load_ghosted(high, &high_source[start..]);
    // Steps 1 and 2.
    scale_row(simd, low, KAPPA);
    scale_row(simd, high, high_pass);
    // Steps 3 to 5.
    for (update_low, coefficient) in [(true, NEG_DELTA), (false, NEG_GAMMA), (true, NEG_BETA)] {
        let coefficient_simd = f32x8::splat(simd, coefficient);
        let (target, source, reads) = if update_low {
            (&mut *low, &mut *high, low_reads)
        } else {
            (&mut *high, &mut *low, high_reads)
        };
        lift_row(
            simd,
            target,
            source,
            reads,
            #[inline(always)]
            |s, a, b| (a + b).mul_add(coefficient_simd, s),
            #[inline(always)]
            |s, a, b| math::mul_add(a + b, coefficient, s),
        );
    }
    // Step 6, the interleave, and the vertical scale share their final loads
    // and store. The multiply still follows the final lifting expression for
    // every sample.
    let coefficient_simd = f32x8::splat(simd, NEG_ALPHA);
    lift_high_to_row::<S, true>(
        simd,
        high,
        low,
        high_reads,
        low_first,
        tail,
        row_scale,
        #[inline(always)]
        |s, a, b| (a + b).mul_add(coefficient_simd, s),
        #[inline(always)]
        |s, a, b| math::mul_add(a + b, NEG_ALPHA, s),
    );
}

/// Columns `columns` of [`synthesize_row`] of an even-origin row that
/// [`column_strips`] split, into `out`: the window blocks the columns cover
/// and, for the row's last strip, its multi-pass tail.
#[expect(
    clippy::too_many_arguments,
    reason = "the strip synthesis keeps its geometry, two sources, output, and exact scratch explicit"
)]
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn synthesize_row_columns<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    low_source: &[f32],
    high_source: &[f32],
    out: &mut [f32],
    columns: Range<usize>,
    low: &mut [f32],
    high: &mut [f32],
    row_scale: f32,
) {
    let first_block = columns.start / (2 * SIMD_WIDTH);
    let window = window_pairs_97(high_source.len());
    let reversible = level.transform == WaveletTransform::Reversible53;
    if columns.end < level.width {
        lift_row_window(
            simd,
            level,
            [low_source, high_source],
            out,
            first_block,
            row_scale,
        );
        return;
    }
    let start = window - WINDOW_HALO_PAIRS;
    // As in `synthesize_row`, the tail first, then the window over its left
    // edge.
    let tail = &mut out[2 * start - columns.start..];
    if reversible {
        synthesize_53_tail(
            simd,
            [low_source, high_source],
            [low, high],
            [(0, 1), (1, 2)],
            true,
            tail,
            start,
        );
    } else {
        synthesize_97_tail(
            simd,
            [low_source, high_source],
            [low, high],
            [(0, 1), (1, 2)],
            true,
            tail,
            start,
            level.high_pass,
            row_scale,
        );
    }
    lift_row_window(
        simd,
        level,
        [low_source, high_source],
        &mut out[..2 * window - columns.start],
        first_block,
        row_scale,
    );
}

/// The register-window pass of `level`'s transform over `out` from
/// `first_block` ([`lift_row_53_window`], [`lift_row_97_window`]).
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_row_window<S: Simd>(
    simd: S,
    level: Level2d<'_>,
    [low_source, high_source]: [&[f32]; 2],
    out: &mut [f32],
    first_block: usize,
    row_scale: f32,
) {
    if level.transform == WaveletTransform::Reversible53 {
        lift_row_53_window(simd, low_source, high_source, out, first_block);
    } else {
        lift_row_97_window(
            simd,
            low_source,
            high_source,
            out,
            first_block,
            level.high_pass,
            row_scale,
        );
    }
}

/// Columns of each strip the last level reconstructs on its own when it
/// streams on one thread ([`last_level`]), so that the strip's rows of all
/// three components stay in the core's L1 cache. Tests use narrow strips so
/// that small images have several.
const STRIP_COLUMNS: usize = if cfg!(test) { 64 } else { 1024 };

/// The column ranges a level's rows are synthesized and lifted in: strips
/// of [`STRIP_COLUMNS`] for wide even-origin levels, whose last strip
/// takes the remaining columns and the row's multi-pass tail, or else the
/// whole row.
fn column_strips(level: Level2d<'_>) -> (usize, usize) {
    let width = level.width;
    let strips = width / STRIP_COLUMNS;
    let low_first = level.x0.is_multiple_of(2);
    let high_len = if low_first {
        width / 2
    } else {
        width.div_ceil(2)
    };
    // The last strip starts before its window ends and before the tail does.
    let last_start = strips.saturating_sub(1) * STRIP_COLUMNS;
    let window = window_pairs_97(high_len);
    if !low_first
        || strips < 2
        || width < MIN_SIMD_ROW_WIDTH
        || last_start + 2 * SIMD_WIDTH > 2 * window.saturating_sub(WINDOW_HALO_PAIRS)
    {
        return (width, 1);
    }
    (STRIP_COLUMNS, strips)
}

/// Strip `index` of `strips` strips of `strip_columns` columns each, the
/// last taking every remaining column of a row of `width`.
fn strip_range(
    index: usize,
    (strip_columns, strips): (usize, usize),
    width: usize,
) -> Range<usize> {
    let start = index * strip_columns;
    start..if index + 1 == strips {
        width
    } else {
        start + strip_columns
    }
}

/// Pairs before the row's remaining pairs that their multi-pass lifting
/// starts from: a wrong left edge spreads one pair per two lifting steps.
const WINDOW_HALO_PAIRS: usize = 2;

/// Pairs that [`lift_row_97_window`] lifts in a row with `high_len` high-pass
/// samples: whole blocks of `SIMD_WIDTH` pairs, each followed by the two
/// blocks its window reads.
fn window_pairs_97(high_len: usize) -> usize {
    high_len.saturating_sub(2 * SIMD_WIDTH) / SIMD_WIDTH * SIMD_WIDTH
}

/// The horizontal 9-7I synthesis of pairs `SIMD_WIDTH * first_block` on, as
/// many as `row` has room for (`row.len() / 2`), of an even-origin row from
/// its `low` and `high` band rows, in one pass of `SIMD_WIDTH`-pair blocks:
/// both bands are scaled as they are loaded, the delta, gamma, beta and alpha
/// steps run a block apart with their neighbors taken across blocks by
/// `slide`, and each block is interleaved, scaled by `row_scale` and stored.
/// A pass from block zero mirrors the row's left edge; one from a later block
/// (at least two) first lifts the two blocks before it as far as their
/// outputs need. Each sample sees the operations of the multi-pass steps in
/// [`synthesize_row`].
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_row_97_window<S: Simd>(
    simd: S,
    low: &[f32],
    high: &[f32],
    row: &mut [f32],
    first_block: usize,
    high_pass: f32,
    row_scale: f32,
) {
    let [kappa, high_pass, row_scale] =
        [dwt::DWT97_KAPPA_F32, high_pass, row_scale].map(|value| f32x8::splat(simd, value));
    let [delta, gamma, beta, alpha] = [
        dwt::IDWT97_NEG_DELTA_F32,
        dwt::IDWT97_NEG_GAMMA_F32,
        dwt::IDWT97_NEG_BETA_F32,
        dwt::IDWT97_NEG_ALPHA_F32,
    ]
    .map(|value| f32x8::splat(simd, value));
    let bands = (low, high);
    let scales = (kappa, high_pass);
    // Low sample `k` lies between high samples `k - 1` and `k`; high sample
    // `k` between low samples `k` and `k + 1`.
    let update =
        |previous: f32x8<S>, high, low, c| (previous.slide::<7>(high) + high).mul_add(c, low);
    let predict =
        |low: f32x8<S>, next: f32x8<S>, high, c| (low + low.slide::<1>(next)).mul_add(c, high);
    // The pipeline before the first output block: its doubly lifted low
    // samples, its once lifted high samples, the next block's once lifted
    // low samples and the next block's scaled high samples.
    let (mut low2, mut high1, mut next_low1, mut next_high0) = if first_block == 0 {
        // T.800 F.3.7: high sample -1 mirrors high sample 0, which a block
        // rotated by one lane holds in its last lane.
        let mirror = |high: f32x8<S>| high.slide::<1>(high);
        let (low0, high0) = scaled_blocks(simd, bands, scales, 0);
        let (next_low0, next_high0) = scaled_blocks(simd, bands, scales, 1);
        let low1 = update(mirror(high0), high0, low0, delta);
        let next_low1 = update(high0, next_high0, next_low0, delta);
        let high1 = predict(low1, next_low1, high0, gamma);
        let low2 = update(mirror(high1), high1, low1, beta);
        (low2, high1, next_low1, next_high0)
    } else {
        let (_, before_high0) = scaled_blocks(simd, bands, scales, first_block - 2);
        let (previous_low0, previous_high0) = scaled_blocks(simd, bands, scales, first_block - 1);
        let (low0, high0) = scaled_blocks(simd, bands, scales, first_block);
        let (next_low0, next_high0) = scaled_blocks(simd, bands, scales, first_block + 1);
        let previous_low1 = update(before_high0, previous_high0, previous_low0, delta);
        let low1 = update(previous_high0, high0, low0, delta);
        let next_low1 = update(high0, next_high0, next_low0, delta);
        let previous_high1 = predict(previous_low1, low1, previous_high0, gamma);
        let high1 = predict(low1, next_low1, high0, gamma);
        let low2 = update(previous_high1, high1, low1, beta);
        (low2, high1, next_low1, next_high0)
    };
    for (index, out) in row
        .as_chunks_mut::<{ 2 * SIMD_WIDTH }>()
        .0
        .iter_mut()
        .enumerate()
    {
        let (block_low0, block_high0) = scaled_blocks(simd, bands, scales, first_block + index + 2);
        let block_low1 = update(next_high0, block_high0, block_low0, delta);
        let next_high1 = predict(next_low1, block_low1, next_high0, gamma);
        let next_low2 = update(high1, next_high1, next_low1, beta);
        let high2 = predict(low2, next_low2, high1, alpha);
        let (first, second) = out.split_at_mut(SIMD_WIDTH);
        (low2.zip_low(high2) * row_scale).store(first);
        (low2.zip_high(high2) * row_scale).store(second);
        (next_high0, next_low1, high1, low2) = (block_high0, block_low1, next_high1, next_low2);
    }
}

/// Block `block` of the low and high bands, each times its scale. A
/// function rather than a closure in [`lift_row_97_window`]: called several
/// times, a closure is outlined, and an outlined closure does not inherit the
/// dispatch's target features.
#[expect(
    clippy::inline_always,
    reason = "the block load is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn scaled_blocks<S: Simd>(
    simd: S,
    (low, high): (&[f32], &[f32]),
    (low_scale, high_scale): (f32x8<S>, f32x8<S>),
    block: usize,
) -> (f32x8<S>, f32x8<S>) {
    let start = block * SIMD_WIDTH;
    (
        f32x8::from_slice(simd, &low[start..][..SIMD_WIDTH]) * low_scale,
        f32x8::from_slice(simd, &high[start..][..SIMD_WIDTH]) * high_scale,
    )
}

/// Copy `source` into `buffer[1..]`, leaving room for a ghost at each end.
fn load_ghosted(buffer: &mut [f32], source: &[f32]) {
    debug_assert_eq!(buffer.len(), source.len() + 2);
    buffer[0] = 0.0;
    buffer[1..=source.len()].copy_from_slice(source);
    buffer[source.len() + 1] = 0.0;
}

/// One horizontal lifting step: `target[k] = f(target[k], source[k + a],
/// source[k + b])` for every sample, with ghost-relative offsets.
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_row<S: Simd>(
    simd: S,
    target: &mut [f32],
    source: &mut [f32],
    (first_offset, second_offset): (usize, usize),
    f_simd: impl Fn(f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    f_scalar: impl Fn(f32, f32, f32) -> f32,
) {
    // Refresh the source ghosts from its current edge samples.
    let source_len = source.len() - 2;
    source[0] = source[1];
    source[source_len + 1] = source[source_len];
    let len = target.len() - 2;
    let target = &mut target[1..=len];
    let first = &source[first_offset..first_offset + len];
    let second = &source[second_offset..second_offset + len];
    let simd_len = len / SIMD_WIDTH * SIMD_WIDTH;
    for index in (0..simd_len).step_by(SIMD_WIDTH) {
        let s = f32x8::from_slice(simd, &target[index..][..SIMD_WIDTH]);
        let x = f32x8::from_slice(simd, &first[index..][..SIMD_WIDTH]);
        let y = f32x8::from_slice(simd, &second[index..][..SIMD_WIDTH]);
        f_simd(s, x, y).store(&mut target[index..][..SIMD_WIDTH]);
    }
    for index in simd_len..len {
        target[index] = f_scalar(target[index], first[index], second[index]);
    }
}

/// Apply the final high-pass lifting step and interleave its result with the
/// low-pass samples directly into `row`.
///
/// The low sample paired with each high sample is already one of the two
/// neighbor vectors loaded for the lift. For 9-7I, scaling the packed vectors
/// here preserves the original lift-then-scale order while avoiding another
/// whole-row load and store.
#[expect(
    clippy::too_many_arguments,
    reason = "the fused final lift keeps its geometry and scalar/SIMD operations explicit"
)]
#[expect(
    clippy::inline_always,
    reason = "the lifting step is specialized with its scalar and SIMD operations"
)]
#[inline(always)]
fn lift_high_to_row<S: Simd, const SCALE: bool>(
    simd: S,
    high: &mut [f32],
    low: &mut [f32],
    (first_offset, second_offset): (usize, usize),
    low_first: bool,
    row: &mut [f32],
    scale: f32,
    f_simd: impl Fn(f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    f_scalar: impl Fn(f32, f32, f32) -> f32,
) {
    let low_len = low.len() - 2;
    let high_len = high.len() - 2;
    // A high-first odd row can end with an unpaired high sample. If that
    // sample shares a SIMD vector with seven pairs, preserve the old vector
    // lift and scalar interleave boundary on this uncommon width.
    if high_len > low_len && high_len.is_multiple_of(SIMD_WIDTH) {
        lift_row(
            simd,
            high,
            low,
            (first_offset, second_offset),
            f_simd,
            f_scalar,
        );
        interleave_row(simd, &high[1..=high_len], &low[1..=low_len], row);
        if SCALE {
            scale_samples(simd, row, scale);
        }
        return;
    }

    // Refresh the low-pass ghosts before taking the neighbor slices.
    low[0] = low[1];
    low[low_len + 1] = low[low_len];
    let high = &high[1..=high_len];
    let first = &low[first_offset..first_offset + high_len];
    let second = &low[second_offset..second_offset + high_len];
    let pairs = low_len.min(high_len);
    let simd_pairs = pairs / SIMD_WIDTH * SIMD_WIDTH;
    let scale_simd = f32x8::splat(simd, scale);

    for index in (0..simd_pairs).step_by(SIMD_WIDTH) {
        let s = f32x8::from_slice(simd, &high[index..][..SIMD_WIDTH]);
        let a = f32x8::from_slice(simd, &first[index..][..SIMD_WIDTH]);
        let b = f32x8::from_slice(simd, &second[index..][..SIMD_WIDTH]);
        let updated_high = f_simd(s, a, b);
        let paired_low = if low_first { a } else { b };
        let (first_values, second_values) = if low_first {
            (paired_low, updated_high)
        } else {
            (updated_high, paired_low)
        };
        let mut packed_low = first_values.zip_low(second_values);
        let mut packed_high = first_values.zip_high(second_values);
        if SCALE {
            packed_low = packed_low * scale_simd;
            packed_high = packed_high * scale_simd;
        }
        packed_low.store(&mut row[index * 2..]);
        packed_high.store(&mut row[index * 2 + SIMD_WIDTH..]);
    }

    for index in simd_pairs..high_len {
        let updated_high = f_scalar(high[index], first[index], second[index]);
        let updated_high = if SCALE {
            updated_high * scale
        } else {
            updated_high
        };
        if index < pairs {
            let paired_low = if low_first {
                first[index]
            } else {
                second[index]
            };
            let paired_low = if SCALE {
                paired_low * scale
            } else {
                paired_low
            };
            if low_first {
                row[index * 2] = paired_low;
                row[index * 2 + 1] = updated_high;
            } else {
                row[index * 2] = updated_high;
                row[index * 2 + 1] = paired_low;
            }
        } else {
            row[index * 2] = updated_high;
        }
    }
    if low_len > high_len {
        let last_low = low[low_len];
        row[row.len() - 1] = if SCALE { last_low * scale } else { last_low };
    }
}

#[expect(clippy::inline_always, reason = "specialized with the band kernel")]
#[inline(always)]
fn scale_row<S: Simd>(simd: S, buffer: &mut [f32], scale: f32) {
    let len = buffer.len() - 2;
    let samples = &mut buffer[1..=len];
    let scale_simd = f32x8::splat(simd, scale);
    let simd_len = len / SIMD_WIDTH * SIMD_WIDTH;
    for index in (0..simd_len).step_by(SIMD_WIDTH) {
        let values = f32x8::from_slice(simd, &samples[index..][..SIMD_WIDTH]);
        (values * scale_simd).store(&mut samples[index..][..SIMD_WIDTH]);
    }
    for value in &mut samples[simd_len..] {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Deterministic xorshift samples: small integers for 5/3, fractions for 9/7.
    #[expect(
        clippy::cast_precision_loss,
        reason = "the drawn value is below 20001 and exact in f32"
    )]
    fn samples(count: usize, seed: u64, integral: bool) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let value = (state % 20_001) as f32 - 10_000.0;
                if integral {
                    value
                } else {
                    value / 7.0
                }
            })
            .collect()
    }

    /// The vertical steps of `arrivals` in batches of `batch` through
    /// [`lift_arrivals`] give every sample the bits of one arrival at a
    /// time through [`lift_arrival`], whatever the width, height, origin
    /// parity, live rows or batch.
    #[test]
    fn wavefront_batches_match_one_arrival_at_a_time() {
        let mut seed = 7_u64;
        for transform in [
            WaveletTransform::Reversible53,
            WaveletTransform::Irreversible97,
        ] {
            let integral = transform == WaveletTransform::Reversible53;
            for y0 in [0_u32, 1, 6] {
                for (width, height) in [(1, 9), (7, 12), (8, 13), (9, 40), (17, 31), (33, 70)] {
                    let rect = IntRect::from_xywh(3, y0, width, height);
                    let level = Level2d::new(rect, [&[]; 4], transform, 1.25);
                    let (steps, count) = vertical_steps(level);
                    let height = height as usize;
                    let width = width as usize;
                    // The whole level, and strip-like windows with halos.
                    for live in [0..height, 0..height.min(11), 2..height, 3..height - 2] {
                        for batch in [1, 2, 3, 5, 16, 37] {
                            seed += 1;
                            let source = samples(width * height, seed, integral);
                            let mut expected = source.clone();
                            let mut rows = RingRows {
                                rows: &mut expected,
                                width,
                                stride: width,
                                slots: height,
                            };
                            for arrival in live.start..live.end + count {
                                dispatch!(Level::new(), simd => lift_arrival(
                                    simd,
                                    &steps[..count],
                                    arrival,
                                    height,
                                    &live,
                                    &mut rows,
                                ));
                            }
                            let mut actual = source;
                            let mut rows = RingRows {
                                rows: &mut actual,
                                width,
                                stride: width,
                                slots: height,
                            };
                            for first in (live.start..live.end + count).step_by(batch) {
                                let arrivals = first..(first + batch).min(live.end + count);
                                dispatch!(Level::new(), simd => lift_arrivals(
                                    simd,
                                    level,
                                    arrivals.clone(),
                                    live.clone(),
                                    &mut rows,
                                ));
                            }
                            assert_eq!(
                                actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                                expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                                "{transform:?} y0={y0} {width}x{height} live={live:?} batch={batch}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Two levels in one pool pass give the bits of the inner level's pass
    /// and then the outer level's, whatever the transform, origin parity
    /// and size.
    #[cfg(feature = "parallel")]
    #[test]
    #[expect(
        clippy::similar_names,
        reason = "paired LL, HL, LH, and HH band names follow JPEG 2000 specification notation"
    )]
    fn level_pair_matches_two_level_passes() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("pool");
        let mut seed = 11_u64;
        for transform in [
            WaveletTransform::Reversible53,
            WaveletTransform::Irreversible97,
        ] {
            let integral = transform == WaveletTransform::Reversible53;
            for (x0, y0, width, height) in [(0, 0, 256, 300), (3, 5, 301, 263), (2, 1, 290, 517)] {
                let outer_rect = IntRect::from_xywh(x0, y0, width, height);
                let inner_rect = packed_band_rects(outer_rect)[0];
                let mut bands = |rect: IntRect| {
                    packed_band_rects(rect).map(|band| {
                        seed += 1;
                        samples((band.width() * band.height()) as usize, seed, integral)
                    })
                };
                let [ll, hl, lh, hh] = bands(inner_rect);
                let [_, outer_hl, outer_lh, outer_hh] = bands(outer_rect);
                let size = |rect: IntRect| (rect.width() * rect.height()) as usize;
                let mut scratch = Vec::new();
                let mut inner = alloc::vec![0.0; size(inner_rect)];
                filter_2d_banded_from_bands(
                    inner_rect,
                    [&ll, &hl, &lh, &hh],
                    &mut inner,
                    &mut scratch,
                    transform,
                    1.25,
                    false,
                )
                .expect("inner level");
                let mut expected = alloc::vec![0.0; size(outer_rect)];
                filter_2d_banded_from_bands(
                    outer_rect,
                    [&inner, &outer_hl, &outer_lh, &outer_hh],
                    &mut expected,
                    &mut scratch,
                    transform,
                    1.25,
                    false,
                )
                .expect("outer level");
                let levels = [
                    Level2d::new(inner_rect, [&ll, &hl, &lh, &hh], transform, 1.25),
                    Level2d::new(
                        outer_rect,
                        [&[], &outer_hl, &outer_lh, &outer_hh],
                        transform,
                        1.25,
                    ),
                ];
                assert!(pool.install(|| pair_runs_on_pool(inner_rect, outer_rect, true)));
                let mut actual = alloc::vec![f32::NAN; size(outer_rect)];
                let mut scratch = Vec::new();
                pool.install(|| filter_levels_pair(levels, &mut actual, &mut scratch))
                    .expect("level pair");
                assert_eq!(
                    actual
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "{transform:?} {outer_rect:?}"
                );
            }
        }
    }

    /// The band scratch the build plan charges for two paired levels is the
    /// scratch their pass grows to, whatever the pool, transform and size.
    #[cfg(feature = "parallel")]
    #[test]
    #[expect(
        clippy::similar_names,
        reason = "paired LL, HL, LH, and HH band names follow JPEG 2000 specification notation"
    )]
    fn level_pair_band_scratch_is_what_the_plan_charges() {
        for threads in [2, 3, 4, 8, 18] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            for transform in [
                WaveletTransform::Reversible53,
                WaveletTransform::Irreversible97,
            ] {
                for (x0, y0, width, height) in [
                    (0, 0, 256, 300),
                    (3, 5, 301, 263),
                    (2, 1, 290, 517),
                    (0, 0, 640, 1300),
                    (1, 0, 1031, 777),
                ] {
                    let outer_rect = IntRect::from_xywh(x0, y0, width, height);
                    let inner_rect = packed_band_rects(outer_rect)[0];
                    let zeros = |rect: IntRect| {
                        packed_band_rects(rect)
                            .map(|band| alloc::vec![0.0; (band.width() * band.height()) as usize])
                    };
                    let [ll, hl, lh, hh] = zeros(inner_rect);
                    let [_, outer_hl, outer_lh, outer_hh] = zeros(outer_rect);
                    let levels = [
                        Level2d::new(inner_rect, [&ll, &hl, &lh, &hh], transform, 1.25),
                        Level2d::new(
                            outer_rect,
                            [&[], &outer_hl, &outer_lh, &outer_hh],
                            transform,
                            1.25,
                        ),
                    ];
                    let label = format!("{threads} threads, {transform:?} {outer_rect:?}");
                    let mut target =
                        alloc::vec![0.0; (outer_rect.width() * outer_rect.height()) as usize];
                    let mut scratch = Vec::new();
                    let charged = pool.install(|| {
                        assert!(pair_runs_on_pool(inner_rect, outer_rect, true), "{label}");
                        filter_levels_pair(levels, &mut target, &mut scratch).expect("level pair");
                        planned_pair_band_scratch_elements(inner_rect, outer_rect, transform)
                            .expect("planned scratch")
                    });
                    assert_eq!(Some(scratch.len()), charged, "{label}");
                }
            }
        }
    }
}
