// SPDX-License-Identifier: MIT OR Apache-2.0

//! SIMD lifting of the packed `f32` forward DWT.
//!
//! Columns are lifted in strips: a strip is copied into a scratch block of its
//! low-pass rows followed by its high-pass rows. Rows are split into their
//! packed halves in place. A lifting step then updates contiguous rows of one
//! band from two neighboring rows of the other band, which it does not write,
//! so every sample sees the operands and IEEE operations of the line transform
//! in `fdwt.rs` and the output is bit-identical for any strip width.

#[cfg(feature = "parallel")]
use alloc::vec::Vec;

use super::packed::{usize_from_u32, LevelRegion};
use super::{ALPHA, BETA, DELTA, GAMMA, INV_KAPPA, KAPPA};
use crate::j2c::idwt::{lift_rows, scale_samples};
use crate::math::{f32x8, floor_f32, floor_half_sum_f32, mul_add, Simd, SIMD_WIDTH};
use crate::{EncodeError, EncodeResult};

/// Columns per vertical strip: one 128-byte cache line of `f32` per row. Rows
/// at least 8 KiB apart get double-width strips, which halves the strip walks
/// that outrun the page-local prefetchers (4096-wide planes are 30% faster).
fn strip_columns(stride: usize) -> usize {
    if stride >= 2048 {
        64
    } else {
        32
    }
}

/// Scratch samples that give every strip its full width:
/// `max(width, min(strip, width) * height)`.
pub(crate) fn packed_dwt_scratch_len(width: u32, height: u32) -> EncodeResult<usize> {
    let width = usize_from_u32(width, "packed forward DWT scratch width")?;
    let height = usize_from_u32(height, "packed forward DWT scratch height")?;
    width
        .min(strip_columns(width))
        .checked_mul(height)
        .map(|strip| strip.max(width))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "packed forward DWT strip scratch",
        })
}

/// Scratch samples per worker that [`transform_planes_from`] needs for
/// `planes` planes: [`packed_dwt_scratch_len`] or every plane's band block,
/// whichever is longer.
pub(crate) fn packed_planes_dwt_scratch_len(
    width: u32,
    height: u32,
    planes: usize,
) -> EncodeResult<usize> {
    let single = packed_dwt_scratch_len(width, height)?;
    let width = usize_from_u32(width, "packed forward DWT scratch width")?;
    let height = usize_from_u32(height, "packed forward DWT scratch height")?;
    let block_rows = height.min(2 * (BAND_ROWS + 2 * HALO_ROWS));
    width
        .min(strip_columns(width))
        .checked_mul(block_rows)
        .and_then(|block| block.checked_mul(planes))
        .map(|blocks| blocks.max(single))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "packed forward DWT band block scratch",
        })
}

/// Levels with fewer samples than this stay on one thread.
#[cfg(feature = "parallel")]
const MIN_PARALLEL_SAMPLES: usize = 256 * 256;

/// Rows a horizontal-pass worker takes at a time.
#[cfg(feature = "parallel")]
const ROW_BATCH: usize = 8;

/// Lift one level's columns in strips, then its rows. `scratch` holds
/// `workers` equal parts of at least `max(width, height)` samples each; a part
/// shorter than [`packed_dwt_scratch_len`] only narrows the strips.
#[expect(
    clippy::inline_always,
    reason = "the level transform is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
pub(super) fn transform_level<S: Simd>(
    simd: S,
    plane: &mut [f32],
    scratch: &mut [f32],
    workers: usize,
    level: LevelRegion,
    reversible: bool,
) -> EncodeResult<()> {
    if level.width == 0 || level.height == 0 {
        return Ok(());
    }
    let per_worker = scratch.len() / workers.max(1);
    let strip = strip_columns(level.stride)
        .min(level.width)
        .min(per_worker / level.height);
    if strip == 0 {
        return Err(EncodeError::InternalInvariant {
            what: "packed forward DWT scratch is shorter than a column",
        });
    }
    #[cfg(feature = "parallel")]
    if workers > 1 && level.width * level.height >= MIN_PARALLEL_SAMPLES {
        transform_level_parallel(simd, plane, scratch, per_worker, level, strip, reversible);
        return Ok(());
    }
    for first in (0..level.width).step_by(strip) {
        let columns = strip.min(level.width - first);
        let block = &mut scratch[..level.height * columns];
        let rows = plane[first..].chunks(level.stride).take(level.height);
        gather_strip(simd, block, columns, rows.map(|row| &row[..columns]), level);
        lift_strip(simd, block, columns, level, reversible);
        let rows = plane[first..].chunks_mut(level.stride);
        scatter_strip(simd, block, columns, rows.map(|row| &mut row[..columns]));
    }
    let line = &mut scratch[..level.width];
    for row in plane.chunks_mut(level.stride).take(level.height) {
        lift_row(simd, &mut row[..level.width], line, level, reversible);
    }
    Ok(())
}

/// Core rows of each band and pass in a band of [`transform_level_into`].
const BAND_ROWS: usize = 16;
/// Rows of each band lifted above and below a band's core. The lifting steps
/// reach two rows into the other band, so core rows see no block edge.
const HALO_ROWS: usize = 2;

/// One level out of place in horizontal bands. A band's column strips, with
/// a halo, are gathered from `input` and lifted in a cache-sized block, its
/// core rows go to their packed rows of `output`, and those rows are lifted
/// while still cached. Each sample is read about once and written once,
/// against two full passes for [`transform_level`], and every core sample sees
/// the same operations. `scratch` must hold `workers` parts of at least
/// [`packed_dwt_scratch_len`] samples.
#[expect(
    clippy::inline_always,
    reason = "the level transform is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
pub(super) fn transform_level_into<S: Simd>(
    simd: S,
    input: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
    workers: usize,
    level: LevelRegion,
    reversible: bool,
) -> EncodeResult<()> {
    let read_row = |y: usize, first: usize, [row]: [&mut [f32]; 1]| {
        copy_row(simd, row, &input[y * level.stride + first..][..row.len()]);
    };
    transform_planes_from(
        simd,
        &read_row,
        [output],
        scratch,
        workers,
        level,
        reversible,
    )
}

/// [`transform_level_into`] for `N` planes of one geometry, with input rows
/// from `read_rows(y, first, rows)`, which fills each plane's row `y` from
/// column `first`. A band's strips of all planes are gathered together, so
/// an interleaved source is read once for all of them.
#[expect(
    clippy::inline_always,
    reason = "the level transform is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
pub(super) fn transform_planes_from<S, R, const N: usize>(
    simd: S,
    read_rows: &R,
    outputs: [&mut [f32]; N],
    scratch: &mut [f32],
    workers: usize,
    level: LevelRegion,
    reversible: bool,
) -> EncodeResult<()>
where
    S: Simd,
    R: Fn(usize, usize, [&mut [f32]; N]) + Sync + ?Sized,
{
    if level.width == 0 || level.height == 0 {
        return Ok(());
    }
    let per_worker = scratch.len() / workers.max(1);
    let block_rows = level.height.min(2 * (BAND_ROWS + 2 * HALO_ROWS));
    let strip = strip_columns(level.stride)
        .min(level.width)
        .min(per_worker / (N * block_rows));
    if strip == 0 || per_worker < level.width {
        return Err(EncodeError::InternalInvariant {
            what: "packed forward DWT scratch is shorter than a band strip or row",
        });
    }
    let high_height = level.height - level.low_height;
    let mut plane_bands = outputs.map(|output| {
        let (low_rows, high_rows) =
            output[..level.height * level.stride].split_at_mut(level.low_height * level.stride);
        (
            low_rows.chunks_mut(BAND_ROWS * level.stride),
            high_rows.chunks_mut(BAND_ROWS * level.stride),
        )
    });
    let bands = (0..level.low_height.max(high_height).div_ceil(BAND_ROWS)).map(|band| {
        let planes = plane_bands.each_mut().map(|(low, high)| {
            (
                low.next().unwrap_or_default(),
                high.next().unwrap_or_default(),
            )
        });
        (band * BAND_ROWS, planes)
    });
    #[cfg(feature = "parallel")]
    if workers > 1 && level.width * level.height >= MIN_PARALLEL_SAMPLES {
        use rayon::prelude::*;
        let queue = std::sync::Mutex::new(bands);
        scratch
            .par_chunks_mut(per_worker)
            .with_max_len(1)
            .for_each(|scratch| {
                in_simd_scope(
                    simd,
                    #[inline(always)]
                    || {
                        while let Some(band) = queue.lock().ok().and_then(|mut bands| bands.next())
                        {
                            transform_band(
                                simd, read_rows, band, scratch, strip, level, reversible,
                            );
                        }
                    },
                );
            });
        return Ok(());
    }
    for band in bands {
        transform_band(simd, read_rows, band, scratch, strip, level, reversible);
    }
    Ok(())
}

/// One band's low-pass and high-pass rows in each plane.
type PlaneBand<'a, const N: usize> = [(&'a mut [f32], &'a mut [f32]); N];

/// Lift one band `(first core row, each plane's low-pass and high-pass
/// rows)` of [`transform_planes_from`] in strips of `strip` columns.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn transform_band<S, R, const N: usize>(
    simd: S,
    read_rows: &R,
    (first_row, mut planes): (usize, PlaneBand<'_, N>),
    scratch: &mut [f32],
    strip: usize,
    level: LevelRegion,
    reversible: bool,
) where
    S: Simd,
    R: Fn(usize, usize, [&mut [f32]; N]) + ?Sized,
{
    let top = first_row.saturating_sub(HALO_ROWS);
    let end = first_row + BAND_ROWS + HALO_ROWS;
    let low_count = end.min(level.low_height).saturating_sub(top);
    let high_count = end.min(level.height - level.low_height).saturating_sub(top);
    for first in (0..level.width).step_by(strip) {
        let columns = strip.min(level.width - first);
        let size = (low_count + high_count) * columns;
        let mut chunks = scratch[..N * size].chunks_exact_mut(size);
        let mut blocks: [&mut [f32]; N] =
            core::array::from_fn(|_| chunks.next().unwrap_or_default());
        for y in 2 * top..level.height.min(2 * end) {
            let k = y / 2 - top;
            let band_row = if (y % 2 == 1) == level.odd_y {
                k
            } else {
                low_count + k
            };
            let rows = blocks
                .each_mut()
                .map(|block| &mut block[band_row * columns..][..columns]);
            read_rows(y, first, rows);
        }
        for (block, (low_rows, high_rows)) in blocks.iter_mut().zip(planes.iter_mut()) {
            let (low, high) = block.split_at_mut(low_count * columns);
            lift(
                simd,
                low,
                high,
                columns,
                level.height,
                level.odd_y,
                reversible,
            );
            for (rows, lifted) in [(&mut **low_rows, &*low), (&mut **high_rows, &*high)] {
                for (k, row) in rows.chunks_mut(level.stride).enumerate() {
                    let core = &lifted[(first_row + k - top) * columns..][..columns];
                    copy_row(simd, &mut row[first..][..columns], core);
                }
            }
        }
    }
    let line = &mut scratch[..level.width];
    for (low_rows, high_rows) in planes {
        for row in low_rows
            .chunks_mut(level.stride)
            .chain(high_rows.chunks_mut(level.stride))
        {
            lift_row(simd, &mut row[..level.width], line, level, reversible);
        }
    }
}

/// The level on several workers. Each takes the next column strip, peeled off
/// every row, and then the next batch of rows as it finishes the last, so
/// slower cores do not hold the level up. Every strip and row goes through
/// the same operations as on one thread.
#[cfg(feature = "parallel")]
fn transform_level_parallel<S: Simd>(
    simd: S,
    plane: &mut [f32],
    scratch: &mut [f32],
    per_worker: usize,
    level: LevelRegion,
    strip: usize,
    reversible: bool,
) {
    use rayon::prelude::*;
    use std::sync::Mutex;

    let rows = plane.chunks_mut(level.stride).take(level.height);
    let remaining = Mutex::new(rows.map(|row| &mut row[..level.width]).collect::<Vec<_>>());
    scratch
        .par_chunks_mut(per_worker)
        .with_max_len(1)
        .for_each(|block| {
            let mut segments = Vec::new();
            in_simd_scope(
                simd,
                #[inline(always)]
                || {
                    while let Some(columns) = next_strip(&remaining, strip, &mut segments) {
                        let block = &mut block[..level.height * columns];
                        let rows = segments.iter().map(|row| &**row);
                        gather_strip(simd, block, columns, rows, level);
                        lift_strip(simd, block, columns, level, reversible);
                        let rows = segments.iter_mut().map(|row| &mut **row);
                        scatter_strip(simd, block, columns, rows);
                    }
                },
            );
        });
    let batches =
        Mutex::new(plane[..level.height * level.stride].chunks_mut(level.stride * ROW_BATCH));
    scratch
        .par_chunks_mut(per_worker)
        .with_max_len(1)
        .for_each(|line| {
            let line = &mut line[..level.width];
            in_simd_scope(
                simd,
                #[inline(always)]
                || {
                    while let Some(batch) = batches.lock().ok().and_then(|mut rows| rows.next()) {
                        for row in batch.chunks_mut(level.stride) {
                            lift_row(simd, &mut row[..level.width], line, level, reversible);
                        }
                    }
                },
            );
        });
}

/// Peel the next strip of up to `strip` columns off every remaining row into
/// `segments`; `None` once the rows are used up.
#[cfg(feature = "parallel")]
fn next_strip<'p>(
    remaining: &std::sync::Mutex<Vec<&'p mut [f32]>>,
    strip: usize,
    segments: &mut Vec<&'p mut [f32]>,
) -> Option<usize> {
    let mut rows = remaining.lock().ok()?;
    let columns = strip.min(rows.first()?.len());
    segments.clear();
    segments.extend(rows.iter_mut().map(|row| {
        let (head, tail) = core::mem::take(row).split_at_mut(columns);
        *row = tail;
        head
    }));
    drop(rows);
    (columns > 0).then_some(columns)
}

/// Run `work` inside `simd`'s target-feature scope. Rayon calls workers
/// outside the caller's scope, so the lifting kernels would not inline.
#[cfg(feature = "parallel")]
#[expect(
    clippy::inline_always,
    reason = "the worker body must inline into the SIMD target-feature scope"
)]
#[inline(always)]
fn in_simd_scope<S: Simd>(simd: S, work: impl FnOnce()) {
    #[cfg(feature = "simd")]
    simd.vectorize(work);
    #[cfg(not(feature = "simd"))]
    {
        let _ = simd;
        work();
    }
}

/// Copy a strip's rows into `block`: its low-pass rows, then its high-pass rows.
#[expect(
    clippy::inline_always,
    reason = "strip copies are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn gather_strip<'a, S: Simd>(
    simd: S,
    block: &mut [f32],
    columns: usize,
    rows: impl Iterator<Item = &'a [f32]>,
    level: LevelRegion,
) {
    for (y, row) in rows.enumerate() {
        let band_row = if (y % 2 == 1) == level.odd_y {
            y / 2
        } else {
            level.low_height + y / 2
        };
        copy_row(simd, &mut block[band_row * columns..][..columns], row);
    }
}

#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_strip<S: Simd>(
    simd: S,
    block: &mut [f32],
    columns: usize,
    level: LevelRegion,
    reversible: bool,
) {
    let (low, high) = block.split_at_mut(level.low_height * columns);
    lift(
        simd,
        low,
        high,
        columns,
        level.height,
        level.odd_y,
        reversible,
    );
}

#[expect(
    clippy::inline_always,
    reason = "strip copies are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn scatter_strip<'a, S: Simd>(
    simd: S,
    block: &[f32],
    columns: usize,
    rows: impl Iterator<Item = &'a mut [f32]>,
) {
    for (row, packed) in rows.zip(block.chunks_exact(columns)) {
        copy_row(simd, row, packed);
    }
}

/// Split one row into its packed halves through `line`, then lift it. An
/// even-origin 9/7 row lifts its first [`fused_pairs`] pairs in one
/// register-window pass, which leaves their low samples in place and their
/// high samples in `line`; the rest is split into `line` and lifted from
/// [`HALO_ROWS`] pairs earlier, so its window's left edge reaches no sample it
/// keeps.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_row<S: Simd>(
    simd: S,
    row: &mut [f32],
    line: &mut [f32],
    level: LevelRegion,
    reversible: bool,
) {
    let fused = if reversible || level.odd_x {
        0
    } else {
        fused_pairs(level.width)
    };
    if fused == 0 {
        line.copy_from_slice(row);
        let (low, high) = row.split_at_mut(level.low_width);
        split_pairs(line, low, high, level.odd_x);
        lift(simd, low, high, 1, level.width, level.odd_x, reversible);
        return;
    }
    let (fused_high, rest) = line.split_at_mut(fused);
    lift_row_97(simd, row, fused_high);
    let start = fused - HALO_ROWS;
    let (rest_low, rest_high) = rest.split_at_mut(level.low_width - start);
    let rest_high = &mut rest_high[..level.width - level.low_width - start];
    split_pairs(&row[2 * start..], rest_low, rest_high, false);
    let samples = level.width - 2 * start;
    lift(simd, rest_low, rest_high, 1, samples, false, false);
    let (low, high) = row.split_at_mut(level.low_width);
    low[fused..].copy_from_slice(&rest_low[HALO_ROWS..]);
    high[..fused].copy_from_slice(fused_high);
    high[fused..].copy_from_slice(&rest_high[HALO_ROWS..]);
}

/// Split the interleaved `line` into its `low` and `high` samples; with
/// `odd_origin` its first sample is high.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn split_pairs(line: &[f32], low: &mut [f32], high: &mut [f32], odd_origin: bool) {
    let (even, odd) = if odd_origin { (high, low) } else { (low, high) };
    let (pairs, tail) = line.as_chunks::<2>();
    for ((even, odd), &[first, second]) in even.iter_mut().zip(odd.iter_mut()).zip(pairs) {
        *even = first;
        *odd = second;
    }
    if let ([last], Some(even)) = (tail, even.last_mut()) {
        *even = *last;
    }
}

/// Pairs that [`lift_row_97`] lifts in a row of `width` samples: whole blocks
/// of `SIMD_WIDTH` pairs, each followed by the two blocks its window reads.
fn fused_pairs(width: usize) -> usize {
    width.saturating_sub(4 * SIMD_WIDTH) / (2 * SIMD_WIDTH) * SIMD_WIDTH
}

/// The 9/7 lifting steps and scaling of the first `high.len()` pairs of an
/// even-origin interleaved `row` in one pass. Each block of pairs is split into
/// its low and high samples once, and every step takes its neighbors across
/// blocks with `slide`; alpha and beta run a block ahead of gamma and delta,
/// whose predict reads the next block's beta output. Low samples are stored at
/// the start of `row`, behind the samples still to be read, and high samples
/// in `high`. Each sample sees the operations of [`lift`].
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift_row_97<S: Simd>(simd: S, row: &mut [f32], high: &mut [f32]) {
    let split = |row: &[f32], block: usize| {
        let samples = &row[2 * SIMD_WIDTH * block..][..2 * SIMD_WIDTH];
        let (first, second) = samples.split_at(SIMD_WIDTH);
        f32x8::from_slice(simd, first).deinterleave(f32x8::from_slice(simd, second))
    };
    let [alpha, beta, gamma, delta] = [ALPHA, BETA, GAMMA, DELTA].map(|c| f32x8::splat(simd, c));
    // High sample `k` lies between low samples `k` and `k + 1`.
    let predict =
        |low: f32x8<S>, next: f32x8<S>, high, c| (low + low.slide::<1>(next)).mul_add(c, high);
    let update =
        |previous: f32x8<S>, high, low, c| (previous.slide::<7>(high) + high).mul_add(c, low);
    // T.800 F.4.7: low sample 0 mirrors onto high sample 0, which a block
    // rotated by one lane holds in its last lane.
    let mirror = |high: f32x8<S>| high.slide::<1>(high);
    let (first_low, first_high) = split(row, 0);
    let (mut low0, mut high0) = split(row, 1);
    let mut high1 = predict(first_low, low0, first_high, alpha);
    let mut low1 = update(mirror(high1), high1, first_low, beta);
    // Lane 0 of the first gamma output reads only lanes 0 and 1 of `low1`.
    let mut high2 = mirror(predict(low1, low1, high1, gamma));
    let (low_scale, high_scale) = (f32x8::splat(simd, INV_KAPPA), f32x8::splat(simd, KAPPA));
    for (index, high_out) in high.as_chunks_mut::<SIMD_WIDTH>().0.iter_mut().enumerate() {
        let (next_low0, next_high0) = split(row, index + 2);
        let next_high1 = predict(low0, next_low0, high0, alpha);
        let next_low1 = update(high1, next_high1, low0, beta);
        let block_high2 = predict(low1, next_low1, high1, gamma);
        let block_low2 = update(high2, block_high2, low1, delta);
        (block_low2 * low_scale).store(&mut row[index * SIMD_WIDTH..]);
        (block_high2 * high_scale).store(high_out);
        (low0, high0, high1, low1, high2) =
            (next_low0, next_high0, next_high1, next_low1, block_high2);
    }
}

/// Lift `columns` parallel lines held as their low-pass rows `low` and their
/// high-pass rows `high`. At an even origin high sample `k` lies between low
/// samples `k` and `k + 1`; at an odd origin, between `k - 1` and `k`. The
/// rows may be a window of longer lines of `samples` samples each, whose
/// length selects the reversible edge forms.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn lift<S: Simd>(
    simd: S,
    low: &mut [f32],
    high: &mut [f32],
    columns: usize,
    samples: usize,
    odd_origin: bool,
    reversible: bool,
) {
    if samples < 2 {
        // T.800 F.4.6: a single sample at an odd coordinate is high-pass and
        // doubled; at an even one it is low-pass and unchanged.
        scale_samples(simd, high, 2.0);
        return;
    }
    if reversible {
        // The even-length, even-origin line transform writes its edge steps
        // with one operand; they equal the two-operand forms unless `a + a`
        // overflows.
        let closed = !odd_origin && samples.is_multiple_of(2);
        step(
            simd,
            (high, low),
            columns,
            !odd_origin,
            |s, a, b| s - ((a + b) * 0.5).floor(),
            |s, a, b| s - floor_half_sum_f32(a, b),
            |s, a| {
                s - if closed {
                    floor_f32(a)
                } else {
                    floor_half_sum_f32(a, a)
                }
            },
        );
        step(
            simd,
            (low, high),
            columns,
            odd_origin,
            |s, a, b| s + ((a + b) * 0.25 + 0.5).floor(),
            |s, a, b| s + floor_f32((a + b) * 0.25 + 0.5),
            |s, a| s + floor_f32(if closed { a * 0.5 } else { (a + a) * 0.25 } + 0.5),
        );
        return;
    }
    // T.800 F-11: alpha and gamma predict the high band, beta and delta update
    // the low band, then both bands are normalized.
    for (coefficient, predict) in [(ALPHA, true), (BETA, false), (GAMMA, true), (DELTA, false)] {
        let bands = if predict {
            (&mut *high, &*low)
        } else {
            (&mut *low, &*high)
        };
        let vector = f32x8::splat(simd, coefficient);
        step(
            simd,
            bands,
            columns,
            predict != odd_origin,
            move |s, a, b| (a + b).mul_add(vector, s),
            move |s, a, b| mul_add(coefficient, a + b, s),
            move |s, a| mul_add(coefficient, a + a, s),
        );
    }
    scale_samples(simd, low, INV_KAPPA);
    scale_samples(simd, high, KAPPA);
}

/// One lifting step over `(target, source)`: target row `k` reads source rows
/// `k` and `k + 1` when `next`, else rows `k - 1` and `k`. A neighbor outside
/// the source mirrors onto the other one (T.800 F.4.7), and those edge rows
/// use `edge(target, neighbor)`.
#[expect(
    clippy::inline_always,
    reason = "the lifting kernels are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn step<S: Simd>(
    simd: S,
    (target, source): (&mut [f32], &[f32]),
    columns: usize,
    next: bool,
    f_simd: impl Fn(f32x8<S>, f32x8<S>, f32x8<S>) -> f32x8<S>,
    f_scalar: impl Fn(f32, f32, f32) -> f32,
    edge: impl Fn(f32, f32) -> f32,
) {
    let (first, end) = if next {
        (0, target.len().min(source.len() - columns))
    } else {
        (columns, target.len().min(source.len()))
    };
    let (leading, rest) = target.split_at_mut(first);
    let (interior, trailing) = rest.split_at_mut(end - first);
    let samples = interior.len();
    lift_rows(
        simd,
        interior,
        &source[..samples],
        &source[columns..][..samples],
        f_simd,
        f_scalar,
    );
    let last = &source[source.len() - columns..];
    for (rows, neighbor) in [(leading, &source[..columns]), (trailing, last)] {
        for (sample, &a) in rows.iter_mut().zip(neighbor) {
            *sample = edge(*sample, a);
        }
    }
}

/// Copy one strip row with vector moves, which stay inline.
#[expect(
    clippy::inline_always,
    reason = "strip row copies are specialized inside each SIMD dispatch target"
)]
#[inline(always)]
fn copy_row<S: Simd>(simd: S, destination: &mut [f32], source: &[f32]) {
    let vectors = destination.len() / SIMD_WIDTH * SIMD_WIDTH;
    for index in (0..vectors).step_by(SIMD_WIDTH) {
        f32x8::from_slice(simd, &source[index..][..SIMD_WIDTH])
            .store(&mut destination[index..][..SIMD_WIDTH]);
    }
    for (sample, &value) in destination[vectors..].iter_mut().zip(&source[vectors..]) {
        *sample = value;
    }
}
