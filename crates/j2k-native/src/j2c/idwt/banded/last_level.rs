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
    apply_vertical_step, horizontal_scratch_elements, packed_band_rects, parallel_strip_layout,
    periodic_symmetric_extension_left, periodic_symmetric_extension_right,
    split_horizontal_scratch, synthesize_level_row, three_rows, use_banded, vertical_steps,
    Level2d, RING_ROWS,
};
use crate::error::{bail, DecodingError};
use crate::math::{dispatch, Level, Simd};
use crate::{try_resize_decode_elements, Result, ValidationError};

/// A last level that each component keeps as its packed sub-bands: the LL
/// band apart, and the HL, LH, and HH bands in the component's plane, where
/// the decode built them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LastLevel {
    rect: IntRect,
    /// Where the HL, LH, and HH bands start in each plane.
    offsets: [usize; 3],
    transform: WaveletTransform,
}

impl LastLevel {
    /// The level `rect` whose HL, LH, and HH bands have these rectangles and
    /// coefficient ranges. `None` unless each band is packed and the banded
    /// path would reconstruct the level.
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
        Some(Self {
            rect,
            offsets,
            transform,
        })
    }

    /// The LL band, which each component keeps apart from its plane.
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

/// Scratch one worker needs: three rings and the horizontal bands.
fn worker_scratch_elements(width: usize, batch: usize) -> Result<usize> {
    (width + RING_ROW_PAD)
        .checked_mul(3 * ring_rows(batch))
        .and_then(|rings| rings.checked_add(horizontal_scratch_elements(width).ok()?))
        .ok_or(ValidationError::ImageTooLarge.into())
}

/// Reconstruct `level` of the three components from their `planes` and
/// `ll_bands`, and pass every row to `emit` with the three components'
/// samples, which it may overwrite, and that row of `output`, three bytes
/// per sample.
pub(crate) fn pack_last_level(
    level: LastLevel,
    planes: [&[f32]; 3],
    ll_bands: &[Vec<f32>; 3],
    scratch: [&mut [f32]; 2],
    output: &mut [u8],
    parallel: bool,
    emit: impl Fn([&mut [f32]; 3], &mut [u8]) + Sync,
) -> Result<()> {
    let levels = [0, 1, 2].map(|component| level.level(planes[component], &ll_bands[component]));
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
        try_resize_decode_elements(&mut scratch, worker_scratch_elements(width, batch)?, 0.0)?;
        dispatch!(Level::new(), simd => stream_rows(
            simd, &levels, 0, batch, output, &mut scratch, &emit,
        ));
        return Ok(());
    };
    #[cfg(feature = "parallel")]
    {
        let strip_bytes = layout.strip_len / width * row_bytes;
        let workers = layout.concurrent_strips;
        let per_worker = worker_scratch_elements(width, PARALLEL_BATCH_ROWS)?;
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
/// Each component takes `batch` arrivals in turn, so the few rows
/// its lifting steps share stay in cache. At an arrival, lifting step `k`
/// (1-based) updates the row `k` above it, reading that row's two
/// neighbors, so after a batch no row more than `count + 1` above its last
/// arrival is read again: those rows are emitted, and the ring refills
/// their slots from the next batch on. Rows left at the end are final.
#[expect(
    clippy::inline_always,
    reason = "the strip kernel is specialized at the architecture dispatch boundary"
)]
#[inline(always)]
fn stream_rows<S: Simd>(
    simd: S,
    levels: &[Level2d<'_>; 3],
    first_row: usize,
    batch: usize,
    output: &mut [u8],
    scratch: &mut [f32],
    emit: &impl Fn([&mut [f32]; 3], &mut [u8]),
) {
    let width = levels[0].width;
    let height = levels[0].height;
    let row_bytes = 3 * width;
    let end = first_row + output.len() / row_bytes;
    let (steps, count) = vertical_steps(levels[0]);
    let lo = first_row.saturating_sub(count);
    let hi = (end + count).min(height);
    let stride = width + RING_ROW_PAD;
    let ring_rows = ring_rows(batch);
    let ring_len = ring_rows * stride;
    let (rings, horizontal_scratch) = scratch.split_at_mut(3 * ring_len);
    let (low, high) = split_horizontal_scratch(levels[0], horizontal_scratch);
    let (r, rest) = rings.split_at_mut(ring_len);
    let (g, b) = rest.split_at_mut(ring_len);
    let mut rings = [r, g, b];
    let slot = |row: usize| row % ring_rows;
    let mut emitted = first_row;
    let mut emit_rows = |rings: &mut [&mut [f32]; 3], rows_end: usize| {
        for row in emitted..rows_end {
            let start = slot(row) * stride;
            let samples = rings.each_mut().map(|ring| &mut ring[start..][..width]);
            emit(
                samples,
                &mut output[(row - first_row) * row_bytes..][..row_bytes],
            );
        }
        emitted = emitted.max(rows_end);
    };
    for first in (lo..hi + count).step_by(batch) {
        let arrivals = first..(first + batch).min(hi + count);
        for (level, ring) in levels.iter().zip(rings.iter_mut()) {
            for arrival in arrivals.clone() {
                if arrival < hi {
                    let row = &mut ring[slot(arrival) * stride..][..width];
                    synthesize_level_row(simd, *level, arrival, row, low, high);
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
                    let (current, above, below) =
                        three_rows(ring, stride, slot(row), slot(above), slot(below));
                    let (current, above, below) =
                        (&mut current[..width], &above[..width], &below[..width]);
                    apply_vertical_step(simd, step, current, above, below);
                }
            }
        }
        // The batch's last arrival read its rows `count + 1` above it.
        let unread = (arrivals.end - 1).saturating_sub(count);
        emit_rows(&mut rings, unread.min(end));
    }
    emit_rows(&mut rings, end);
}
