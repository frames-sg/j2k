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

use super::super::build::Decomposition;
use super::super::codestream::WaveletTransform;
use super::super::decode::DecompositionStorage;
use super::super::rect::IntRect;
use super::filter_common::{periodic_symmetric_extension_left, periodic_symmetric_extension_right};
use super::horizontal::filter_row;
use super::interleave::interleave_row;
use super::model::{IDWTInput, IDWTTempOutput};
use crate::math::{self, dispatch, f32x8, Level, Simd, SIMD_WIDTH};
use crate::{checked_decode_usize_product2, try_resize_decode_elements, Result, ValidationError};
use j2k_codec_math::dwt;

/// Halo rows needed on each side of a band: one per vertical lifting step.
const HALO_ROWS_53: usize = 2;
const HALO_ROWS_97: usize = 4;
const MAX_HALO_ROWS: usize = HALO_ROWS_97;

/// Levels smaller than this run the plain path; banding saves nothing there.
const MIN_BANDED_SAMPLES: usize = 64 * 64;
/// Levels with fewer samples than this stay on one thread.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SAMPLES: usize = 256 * 256;

/// Ring rows a strip worker keeps: the row being lifted with both of its
/// neighbors, plus one row per lifting step still to finish.
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
        Some((
            self.first
                .get(index * self.first_width..)?
                .get(..self.first_width)?,
            self.second
                .get(index * self.second_width..)?
                .get(..self.second_width)?,
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
#[expect(
    clippy::similar_names,
    reason = "paired LL, HL, LH, and HH band names follow JPEG 2000 specification notation"
)]
pub(super) fn filter_2d_banded_from_bands(
    rect: IntRect,
    [ll, hl, lh, hh]: [&[f32]; 4],
    target: &mut Vec<f32>,
    band_scratch: &mut Vec<f32>,
    transform: WaveletTransform,
    high_pass: f32,
    parallel: bool,
) -> Result<()> {
    let width = rect.width() as usize;
    let height = rect.height() as usize;
    let required_len = checked_decode_usize_product2(width, height)?;
    try_resize_decode_elements(target, required_len, 0.0)?;

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
    let level = Level2d {
        even: RowSource {
            first: first_even,
            second: second_even,
            first_width,
            second_width,
            start_row: usize::from(v0 % 2 != 0),
            rows: num_v_low,
        },
        odd: RowSource {
            first: first_odd,
            second: second_odd,
            first_width,
            second_width,
            start_row: usize::from(v0 % 2 == 0),
            rows: num_v_high,
        },
        width,
        height,
        x0: u0 as usize,
        y0: v0 as usize,
        transform,
        high_pass,
    };

    let target = &mut target[..required_len];
    let Some(parallel_layout) = parallel_strip_layout(width, height, parallel)? else {
        let scratch_len = horizontal_scratch_elements(width)?;
        try_resize_decode_elements(band_scratch, scratch_len, 0.0)?;
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
        try_resize_decode_elements(band_scratch, scratch_len, 0.0)?;
        let wave_len = parallel_layout
            .strip_len
            .checked_mul(parallel_layout.concurrent_strips)
            .ok_or(ValidationError::ImageTooLarge)?;
        dispatch!(Level::new(), simd => run_strips(
            simd,
            level,
            target,
            parallel_layout.strip_len,
            wave_len,
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
    wave_len: usize,
    scratch: &mut [f32],
    per_worker: usize,
) {
    use rayon::prelude::*;

    let strips_per_wave = scratch.len() / per_worker;
    for (wave, outputs) in target.chunks_mut(wave_len).enumerate() {
        outputs
            .par_chunks_mut(strip_len)
            .zip(scratch.par_chunks_mut(per_worker))
            .enumerate()
            .for_each(|(index, (output, worker_scratch))| {
                let strip = wave * strips_per_wave + index;
                let first_row = strip * strip_len / level.width;
                stream_strip(simd, level, first_row, output, worker_scratch);
            });
    }
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
    let (steps, count) = vertical_steps(level);
    let (low, high) = split_horizontal_scratch(level, horizontal_scratch);
    for arrival in 0..height + count {
        if arrival < height {
            let row = &mut target[arrival * width..][..width];
            synthesize_level_row(simd, level, arrival, row, low, high);
        }
        for (index, &(parity, step)) in steps[..count].iter().enumerate() {
            let Some(row) = arrival.checked_sub(index + 1) else {
                break;
            };
            if row >= height || row % 2 != parity {
                continue;
            }
            let above = periodic_symmetric_extension_left(row, 1);
            let below = periodic_symmetric_extension_right(row, 1, height);
            let (current, above, below) = three_rows(target, width, row, above, below);
            apply_vertical_step(simd, step, current, above, below);
        }
    }
}

/// Reconstruct level rows `[first_row, first_row + output.len() / width)` into
/// `output` with the schedule of [`stream_level`], keeping the live rows in
/// `ring` and starting and ending one halo row per lifting step beyond the
/// strip. A halo row whose neighbor is outside the halo skips that step; the
/// resulting error stays inside the halo.
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
    let (steps, count) = vertical_steps(level);
    let end = first_row + output.len() / width;
    let lo = first_row.saturating_sub(count);
    let hi = (end + count).min(height);
    let ring_rows = count + 2;
    let ring_capacity = RING_ROWS * width;
    let (ring, horizontal_scratch) = worker_scratch.split_at_mut(ring_capacity);
    let ring = &mut ring[..ring_rows * width];
    let (low, high) = split_horizontal_scratch(level, horizontal_scratch);
    for arrival in lo..hi + count {
        if arrival < hi {
            let row = &mut ring[(arrival % ring_rows) * width..][..width];
            synthesize_level_row(simd, level, arrival, row, low, high);
        }
        for (index, &(parity, step)) in steps[..count].iter().enumerate() {
            let Some(row) = arrival.checked_sub(index + 1) else {
                break;
            };
            if row < lo || row >= hi || row % 2 != parity {
                continue;
            }
            let above = periodic_symmetric_extension_left(row, 1);
            let below = periodic_symmetric_extension_right(row, 1, height);
            if above < lo || below >= hi {
                continue;
            }
            let (current, above, below) = three_rows(
                ring,
                width,
                row % ring_rows,
                above % ring_rows,
                below % ring_rows,
            );
            apply_vertical_step(simd, step, current, above, below);
        }
        // The row `count` arrivals back has taken its last step, and its
        // ring slot is not reused before two more arrivals.
        if let Some(done) = arrival.checked_sub(count) {
            if (first_row..end).contains(&done) {
                output[(done - first_row) * width..][..width]
                    .copy_from_slice(&ring[(done % ring_rows) * width..][..width]);
            }
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
        VerticalStep::Reversible53Low => lift_rows(
            simd,
            current,
            above,
            below,
            #[inline(always)]
            |s1, s2, s3| s1 - ((s2 + s3 + 2.0) * 0.25).floor(),
            #[inline(always)]
            |s1, s2, s3| s1 - math::floor_f32(math::mul_add(s2 + s3, 0.25, 0.5)),
        ),
        VerticalStep::Reversible53High => lift_rows(
            simd,
            current,
            above,
            below,
            #[inline(always)]
            |s1, s2, s3| s1 + ((s2 + s3) * 0.5).floor(),
            #[inline(always)]
            |s1, s2, s3| s1 + math::floor_half_sum_f32(s2, s3),
        ),
        VerticalStep::Irreversible97(coefficient) => {
            let coefficient_simd = f32x8::splat(simd, coefficient);
            lift_rows(
                simd,
                current,
                above,
                below,
                #[inline(always)]
                |s1, s2, s3| (s2 + s3).mul_add(coefficient_simd, s1),
                #[inline(always)]
                |s1, s2, s3| math::mul_add(s2 + s3, coefficient, s1),
            );
        }
    }
}

/// Multiply every sample of `samples` by `scale`.
#[expect(
    clippy::inline_always,
    reason = "specialized with the streaming kernels"
)]
#[inline(always)]
fn scale_samples<S: Simd>(simd: S, samples: &mut [f32], scale: f32) {
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
fn lift_rows<S: Simd>(
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
    clippy::too_many_arguments,
    reason = "the row synthesis keeps its geometry, two sources, output, and exact scratch explicit"
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
    const NEG_ALPHA: f32 = dwt::IDWT97_NEG_ALPHA_F32;
    const NEG_BETA: f32 = dwt::IDWT97_NEG_BETA_F32;
    const NEG_GAMMA: f32 = dwt::IDWT97_NEG_GAMMA_F32;
    const NEG_DELTA: f32 = dwt::IDWT97_NEG_DELTA_F32;
    const KAPPA: f32 = dwt::DWT97_KAPPA_F32;
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
    load_ghosted(low, low_source);
    load_ghosted(high, high_source);
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
                row,
                1.0,
                #[inline(always)]
                |s, a, b| s + ((a + b) * 0.5).floor(),
                #[inline(always)]
                |s, a, b| s + math::floor_half_sum_f32(a, b),
            );
        }
        WaveletTransform::Irreversible97 => {
            // Steps 1 and 2.
            scale_row(simd, low, KAPPA);
            scale_row(simd, high, level.high_pass);
            // Steps 3 to 5.
            for (update_low, coefficient) in
                [(true, NEG_DELTA), (false, NEG_GAMMA), (true, NEG_BETA)]
            {
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
            // Step 6, the interleave, and the vertical scale share their
            // final loads and store. The multiply still follows the final
            // lifting expression for every sample.
            let coefficient_simd = f32x8::splat(simd, NEG_ALPHA);
            lift_high_to_row::<S, true>(
                simd,
                high,
                low,
                high_reads,
                low_first,
                row,
                row_scale,
                #[inline(always)]
                |s, a, b| (a + b).mul_add(coefficient_simd, s),
                #[inline(always)]
                |s, a, b| math::mul_add(a + b, NEG_ALPHA, s),
            );
        }
    }
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
