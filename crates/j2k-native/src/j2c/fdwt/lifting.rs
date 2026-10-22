// SPDX-License-Identifier: MIT OR Apache-2.0

//! SIMD lifting of the packed `f32` forward DWT.
//!
//! Columns are lifted in strips: a strip is copied into a scratch block of its
//! low-pass rows followed by its high-pass rows. Rows are split into their
//! packed halves in place. A lifting step then updates contiguous rows of one
//! band from two neighboring rows of the other band, which it does not write,
//! so every sample sees the operands and IEEE operations of the line transform
//! in `fdwt.rs` and the output is bit-identical for any strip width.

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

/// Lift one level's columns in strips, then its rows. `scratch` holds at
/// least `max(width, height)` samples; a shorter one than
/// [`packed_dwt_scratch_len`] only narrows the strips.
#[expect(
    clippy::inline_always,
    reason = "the level transform is specialized inside each SIMD dispatch target"
)]
#[inline(always)]
pub(super) fn transform_level<S: Simd>(
    simd: S,
    plane: &mut [f32],
    scratch: &mut [f32],
    level: LevelRegion,
    reversible: bool,
) -> EncodeResult<()> {
    if level.width == 0 || level.height == 0 {
        return Ok(());
    }
    let strip = strip_columns(level.stride)
        .min(level.width)
        .min(scratch.len() / level.height);
    if strip == 0 {
        return Err(EncodeError::InternalInvariant {
            what: "packed forward DWT scratch is shorter than a column",
        });
    }
    for first in (0..level.width).step_by(strip) {
        let columns = strip.min(level.width - first);
        let block = &mut scratch[..level.height * columns];
        let rows = plane[first..].chunks(level.stride).take(level.height);
        for (y, row) in rows.enumerate() {
            let band_row = if (y % 2 == 1) == level.odd_y {
                y / 2
            } else {
                level.low_height + y / 2
            };
            copy_row(
                simd,
                &mut block[band_row * columns..][..columns],
                &row[..columns],
            );
        }
        let (low, high) = block.split_at_mut(level.low_height * columns);
        lift(simd, low, high, columns, level.odd_y, reversible);
        for (row, packed) in plane[first..]
            .chunks_mut(level.stride)
            .zip(block.chunks_exact(columns))
        {
            copy_row(simd, &mut row[..columns], packed);
        }
    }

    let line = &mut scratch[..level.width];
    for row in plane.chunks_mut(level.stride).take(level.height) {
        let row = &mut row[..level.width];
        line.copy_from_slice(row);
        let (low, high) = row.split_at_mut(level.low_width);
        let (even, odd) = if level.odd_x {
            (&mut *high, &mut *low)
        } else {
            (&mut *low, &mut *high)
        };
        let (pairs, tail) = line.as_chunks::<2>();
        for ((even, odd), &[first, second]) in even.iter_mut().zip(odd.iter_mut()).zip(pairs) {
            *even = first;
            *odd = second;
        }
        if let ([last], Some(even)) = (tail, even.last_mut()) {
            *even = *last;
        }
        lift(simd, low, high, 1, level.odd_x, reversible);
    }
    Ok(())
}

/// Lift `columns` parallel lines held as their low-pass rows `low` and their
/// high-pass rows `high`. At an even origin high sample `k` lies between low
/// samples `k` and `k + 1`; at an odd origin, between `k - 1` and `k`.
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
    odd_origin: bool,
    reversible: bool,
) {
    let samples = (low.len() + high.len()) / columns;
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
