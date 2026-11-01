// SPDX-License-Identifier: MIT OR Apache-2.0

//! The last level of an RGB8 decode, streamed straight into the pack.
//!
//! [`super::filter_2d_banded`] writes each component's last level as a whole
//! plane, which the inverse MCT and the RGB8 pack then read back. Here the
//! three components' last levels are reconstructed together with the schedule
//! of [`super::stream_strip`], each in a ring of a few rows, and a row is
//! handed to the pack as soon as no lifting step reads it any more, so the
//! level is never written to memory. Every sample goes through the same
//! expressions as the banded path, so the output is bit-identical.

use alloc::vec::Vec;
use core::ops::Range;

use super::super::super::codestream::WaveletTransform;
use super::super::super::rect::IntRect;
use super::{
    column_strips, horizontal_scratch_elements, lift_arrivals, packed_band_rects,
    parallel_strip_layout, split_horizontal_scratch, strip_range, synthesize_level_columns,
    use_banded, vertical_steps, Level2d, RingRows, RING_ROWS, STRIP_COLUMNS, WAVEFRONT_ARRIVALS,
};
use crate::error::{bail, DecodingError};
use crate::math::{dispatch, Level, Simd};
use crate::{try_resize_decode_elements, Result, ValidationError};

/// A last level whose high bands remain in their component plane. The
/// first LL stays in the IDWT output; each later LL occupies the previous
/// plane's consumed coarse-coefficient prefix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LastLevel {
    rect: IntRect,
    /// Where the HL, LH, and HH bands start in each plane.
    offsets: [usize; 3],
    transform: WaveletTransform,
}

impl LastLevel {
    /// The level `rect` whose HL, LH, and HH bands have these rectangles and
    /// coefficient ranges. `None` unless each band is packed, the LL band fits
    /// in front of them, and the banded path would reconstruct the level.
    pub(crate) fn new(
        rect: IntRect,
        bands: [(IntRect, Range<usize>); 3],
        transform: WaveletTransform,
    ) -> Option<Self> {
        if !use_banded(rect.width() as usize, rect.height() as usize) {
            return None;
        }
        let packed = packed_band_rects(rect);
        let mut offsets = [0; 3];
        for ((offset, packed), (band, range)) in offsets.iter_mut().zip(&packed[1..]).zip(bands) {
            if band != *packed || range.len() != band_len(band)? {
                return None;
            }
            *offset = range.start;
        }
        if band_len(packed[0])? > *offsets.iter().min()? {
            return None;
        }
        Some(Self {
            rect,
            offsets,
            transform,
        })
    }

    /// The LL band reconstructed into the previous plane's dead prefix,
    /// except for the first component, which keeps the IDWT output.
    pub(crate) fn ll_rect(&self) -> IntRect {
        packed_band_rects(self.rect)[0]
    }

    fn level<'a>(&self, plane: &'a [f32], ll: &'a [f32]) -> Option<Level2d<'a>> {
        let packed = packed_band_rects(self.rect);
        let mut bands = [ll.get(..band_len(packed[0])?)?; 4];
        for ((band, rect), offset) in bands[1..].iter_mut().zip(&packed[1..]).zip(self.offsets) {
            *band = plane.get(offset..offset.checked_add(band_len(*rect)?)?)?;
        }
        Some(Level2d::new(
            self.rect,
            bands,
            self.transform,
            super::super::OPENJPEG_NORMALIZED_HIGH_PASS_F32,
        ))
    }
}

/// Arrivals each component takes in turn on one thread, and on the pool,
/// whose workers share the cache.
const SERIAL_BATCH_ROWS: usize = 16;
#[cfg(feature = "parallel")]
const PARALLEL_BATCH_ROWS: usize = 8;

/// Samples between ring rows beyond the row itself: rows a power-of-two
/// apart in memory share cache sets.
const RING_ROW_PAD: usize = 16;

fn band_len(rect: IntRect) -> Option<usize> {
    (rect.width() as usize).checked_mul(rect.height() as usize)
}

/// Rows each component's ring holds for `batch` arrivals: the batch and
/// every row a lifting step of it may read or that waits to be emitted.
const fn ring_rows(batch: usize) -> usize {
    batch + RING_ROWS - 1
}

/// Scratch one worker needs: three rings for each column strip a level of
/// this width can have when `split`, and the horizontal bands.
fn worker_scratch_elements(width: usize, batch: usize, split: bool) -> Result<usize> {
    let strips = if split {
        (width / STRIP_COLUMNS).max(1)
    } else {
        1
    };
    strips
        .checked_mul(RING_ROW_PAD)
        .and_then(|pads| pads.checked_add(width))
        .and_then(|row| row.checked_mul(3 * ring_rows(batch)))
        .and_then(|rings| rings.checked_add(horizontal_scratch_elements(width).ok()?))
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// The three components' rings of column strip `index` in `rings`, each of
/// `ring_rows` rows `stride` apart: every strip before it is `strips.0`
/// columns wide.
fn strip_rings(
    rings: &mut [f32],
    index: usize,
    strips: (usize, usize),
    stride: usize,
    ring_rows: usize,
) -> [&mut [f32]; 3] {
    let ring_len = ring_rows * stride;
    let start = index * 3 * ring_rows * (strips.0 + RING_ROW_PAD);
    let (r, rest) = rings[start..].split_at_mut(ring_len);
    let (g, rest) = rest.split_at_mut(ring_len);
    [r, g, &mut rest[..ring_len]]
}

/// Reconstruct `level` of the three components from their `planes` and the
/// first component's `first_ll` band, and pass every row to `emit` with the
/// three components' samples, which it may overwrite, and that row of
/// `output`, three bytes per sample.
pub(crate) fn pack_last_level(
    level: LastLevel,
    planes: [&[f32]; 3],
    first_ll: &[f32],
    scratch: [&mut [f32]; 2],
    output: &mut [u8],
    parallel: bool,
    emit: impl Fn([&mut [f32]; 3], &mut [u8]) + Sync,
) -> Result<()> {
    let lls = [first_ll, planes[0], planes[1]];
    let levels = [0, 1, 2].map(|component| level.level(planes[component], lls[component]));
    let [Some(r), Some(g), Some(b)] = levels else {
        bail!(DecodingError::CodeBlockDecodeFailure);
    };
    let levels = [r, g, b];
    let (width, height) = (r.width, r.height);
    let row_bytes = 3 * width;
    if output.len() != row_bytes * height {
        bail!(DecodingError::OutputBufferTooSmall);
    }
    let Some(layout) = parallel_strip_layout(width, height, parallel)? else {
        let batch = SERIAL_BATCH_ROWS;
        let mut scratch = Vec::new();
        try_resize_decode_elements(
            &mut scratch,
            worker_scratch_elements(width, batch, true)?,
            0.0,
        )?;
        dispatch!(Level::new(), simd => stream_rows(
            simd, &levels, 0, batch, true, output, &mut scratch, &emit,
        ));
        return Ok(());
    };
    #[cfg(feature = "parallel")]
    {
        let strip_bytes = layout.strip_len / width * row_bytes;
        let workers = layout.concurrent_strips;
        let per_worker = worker_scratch_elements(width, PARALLEL_BATCH_ROWS, false)?;
        let complete = dispatch!(Level::new(), simd => run_strips(
            simd, &levels, output, strip_bytes, workers, per_worker, scratch, &emit,
        ));
        if !complete {
            bail!(DecodingError::HostAllocationFailed);
        }
    }
    #[cfg(not(feature = "parallel"))]
    let _ = (layout, scratch);
    Ok(())
}

/// Stream strips of `strip_bytes` of `output` on `workers` workers, each
/// with its own scratch. Returns `false` when a worker could not allocate it.
#[cfg(feature = "parallel")]
#[inline(always)]
#[expect(
    clippy::too_many_arguments,
    reason = "the strip driver keeps geometry and reused scratch explicit"
)]
fn run_strips<S: Simd>(
    simd: S,
    levels: &[Level2d<'_>; 3],
    output: &mut [u8],
    strip_bytes: usize,
    workers: usize,
    per_worker: usize,
    scratch: [&mut [f32]; 2],
    emit: &(impl Fn([&mut [f32]; 3], &mut [u8]) + Sync),
) -> bool {
    use core::sync::atomic::{AtomicBool, Ordering};
    use rayon::prelude::*;
    use std::sync::Mutex;

    let row_bytes = 3 * levels[0].width;
    // As in `super::run_strips`, each worker takes the next strip when it
    // finishes the last.
    let strips = Mutex::new(output.chunks_mut(strip_bytes).enumerate());
    let next = || strips.lock().ok().and_then(|mut strips| strips.next());
    // Coarse reconstruction has finished. Its initialized scratch can
    // serve pack workers without changing the context's retained capacity.
    let [coarse, bands] = scratch;
    let spare = Mutex::new(
        coarse
            .chunks_exact_mut(per_worker)
            .chain(bands.chunks_exact_mut(per_worker)),
    );
    let failed = AtomicBool::new(false);
    (0..workers).into_par_iter().with_max_len(1).for_each(|_| {
        let mut owned = Vec::new();
        let scratch = if let Some(scratch) = spare.lock().ok().and_then(|mut spare| spare.next()) {
            scratch
        } else {
            if try_resize_decode_elements(&mut owned, per_worker, 0.0).is_err() {
                failed.store(true, Ordering::Relaxed);
                return;
            }
            &mut owned
        };
        // Re-enter the caller's target-feature scope, as `super::run_strips`.
        #[cfg(feature = "simd")]
        simd.vectorize(
            #[inline(always)]
            || {
                while let Some((strip, output)) = next() {
                    let first_row = strip * strip_bytes / row_bytes;
                    stream_rows(
                        simd,
                        levels,
                        first_row,
                        PARALLEL_BATCH_ROWS,
                        false,
                        output,
                        scratch,
                        emit,
                    );
                }
            },
        );
        #[cfg(not(feature = "simd"))]
        while let Some((strip, output)) = next() {
            let first_row = strip * strip_bytes / row_bytes;
            stream_rows(
                simd,
                levels,
                first_row,
                PARALLEL_BATCH_ROWS,
                false,
                output,
                scratch,
                emit,
            );
        }
    });
    !failed.load(Ordering::Relaxed)
}

/// Reconstruct the level rows from `first_row` that `output` holds, of all
/// three components, with the schedule and halo rows of
/// [`super::stream_strip`], and emit each row.
///
/// Each component takes `batch` arrivals in turn, one column strip
/// ([`column_strips`]) at a time, so the few rows its lifting steps share
/// stay in cache: lifting is per column, so the strips are independent but
/// for their horizontal synthesis, which reads past their edges. At an
/// arrival, lifting step `k`
/// (1-based) updates the row `k` above it, reading that row's two
/// neighbors, so after a batch no row more than `count + 1` above its last
/// arrival is read again: those rows are emitted, and the ring refills
/// their slots from the next batch on. Rows left at the end are final.
#[expect(
    clippy::inline_always,
    reason = "the strip kernel is specialized at the architecture dispatch boundary"
)]
#[expect(
    clippy::too_many_arguments,
    reason = "the row stream keeps its geometry, batch, strip choice, and reused scratch explicit"
)]
#[inline(always)]
fn stream_rows<S: Simd>(
    simd: S,
    levels: &[Level2d<'_>; 3],
    first_row: usize,
    batch: usize,
    split: bool,
    output: &mut [u8],
    scratch: &mut [f32],
    emit: &impl Fn([&mut [f32]; 3], &mut [u8]),
) {
    let width = levels[0].width;
    let height = levels[0].height;
    let row_bytes = 3 * width;
    let end = first_row + output.len() / row_bytes;
    let (_, count) = vertical_steps(levels[0].lifting());
    let lo = first_row.saturating_sub(count);
    let hi = (end + count).min(height);
    let ring_rows = ring_rows(batch);
    // Column strips pay on one thread. On the pool, where several cores
    // stream band rows from memory at once, their shorter row segments cost
    // more than the strips save.
    let strips = if split {
        column_strips(levels[0])
    } else {
        (width, 1)
    };
    let rings_len = (0..strips.1)
        .map(|index| 3 * ring_rows * (strip_range(index, strips, width).len() + RING_ROW_PAD))
        .sum();
    let (rings, horizontal_scratch) = scratch.split_at_mut(rings_len);
    let (low, high) = split_horizontal_scratch(levels[0], horizontal_scratch);
    let mut emit_rows = |rings: &mut [f32], index: usize, rows: Range<usize>| {
        let columns = strip_range(index, strips, width);
        let stride = columns.len() + RING_ROW_PAD;
        let mut rings = strip_rings(rings, index, strips, stride, ring_rows);
        for row in rows {
            let start = row % ring_rows * stride;
            let samples = rings
                .each_mut()
                .map(|ring| &mut ring[start..][..columns.len()]);
            let output = &mut output[(row - first_row) * row_bytes..][..row_bytes];
            emit(samples, &mut output[3 * columns.start..3 * columns.end]);
        }
    };
    let mut emitted = first_row;
    for first in (lo..hi + count).step_by(batch) {
        let arrivals = first..(first + batch).min(hi + count);
        // The batch's last arrival read its rows `count + 1` above it.
        let unread = (arrivals.end - 1).saturating_sub(count).min(end);
        // Each column strip runs the batch of all three components and
        // emits its part of the finished rows, while its rings are in cache.
        for index in 0..strips.1 {
            let columns = strip_range(index, strips, width);
            let stride = columns.len() + RING_ROW_PAD;
            let component_rings = strip_rings(rings, index, strips, stride, ring_rows);
            for (level, ring) in levels.iter().zip(component_rings) {
                let mut rows = RingRows {
                    rows: ring,
                    width: columns.len(),
                    stride,
                    slots: ring_rows,
                };
                // The ring holds every row the batch's steps read besides
                // the batch itself, so arrivals can be synthesized ahead.
                for first in arrivals.clone().step_by(WAVEFRONT_ARRIVALS) {
                    let ahead = first..(first + WAVEFRONT_ARRIVALS).min(arrivals.end);
                    for arrival in ahead.start..ahead.end.min(hi) {
                        let row = rows.row_mut(arrival);
                        synthesize_level_columns(
                            simd,
                            *level,
                            arrival,
                            row,
                            columns.clone(),
                            low,
                            high,
                        );
                    }
                    lift_arrivals(simd, level.lifting(), ahead, lo..hi, &mut rows);
                }
            }
            emit_rows(rings, index, emitted..unread.max(emitted));
        }
        emitted = emitted.max(unread);
    }
    for index in 0..strips.1 {
        emit_rows(rings, index, emitted..end);
    }
}
