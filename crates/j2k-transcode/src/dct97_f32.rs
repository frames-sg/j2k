// SPDX-License-Identifier: MIT OR Apache-2.0

//! Canonical binary32 DCT-grid to irreversible 9/7 transform.
//!
//! This is the default-precision 9/7 transcode transform. It evaluates the
//! operation sequence defined by [`j2k_codec_math::transcode97`], so its
//! coefficients are bit-identical to the Metal and CUDA transcode kernels.
//! Lifting runs on deinterleaved low/high halves: each lifting step reads only
//! the other half, which makes every step an element-wise update that can be
//! split across threads without changing a single operation.

use j2k_codec_math::transcode97::{
    idct8x8, lift97_update, Transcode97Arithmetic, DWT97_HIGH_SCALE_F32, DWT97_LIFT_STEPS_F32,
    DWT97_LOW_SCALE_F32,
};
use rayon::prelude::*;

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, checked_allocation_len,
    checked_capacity_bytes, try_vec_filled, try_vec_resize_with, TranscodeAllocationError,
};
use crate::dct_grid::{high_len, low_len, validate_dct_block_grid};
use crate::{DctTransformError, Dwt97TwoDimensional};

/// Planes at least this large split block rows across Rayon workers for the
/// IDCT, which is one dispatch of compute-bound work.
const PARALLEL_IDCT_MIN_SAMPLES: usize = 256 * 256;

/// Planes at least this large split lifting rows across Rayon workers. Column
/// lifting synchronises once per lifting step, so smaller planes run faster
/// on one thread (laptop M4 Pro, 224x224 4:2:0 tile: 1.9 ms parallel at a
/// 64x64 threshold, 0.40 ms serial).
const PARALLEL_LIFT_MIN_SAMPLES: usize = 1024 * 1024;

/// Host binary32 primitives for the canonical sequence.
///
/// Rust never contracts `*` and `+`, and `f32::mul_add` is a single-rounding
/// fused multiply-add on every target. Baseline x86-64 has no FMA
/// instruction, so every loop that uses these runs inside
/// [`j2k_native::at_simd_level`], where AVX2 and FMA are enabled and the
/// loops vectorise; NEON always has FMA. SIMD lanes round exactly like the
/// scalar operations, so every level produces the same bits.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostTranscode97Arithmetic;

impl Transcode97Arithmetic for HostTranscode97Arithmetic {
    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "target features reach only code inlined into the SIMD dispatch"
    )]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        a.mul_add(b, c)
    }

    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "target features reach only code inlined into the SIMD dispatch"
    )]
    fn mul(a: f32, b: f32) -> f32 {
        a * b
    }

    #[inline(always)]
    #[expect(
        clippy::inline_always,
        reason = "target features reach only code inlined into the SIMD dispatch"
    )]
    fn add(a: f32, b: f32) -> f32 {
        a + b
    }
}

type Host = HostTranscode97Arithmetic;

/// Reusable scratch for repeated canonical 9/7 transforms.
#[derive(Debug, Default)]
pub(crate) struct Dct97F32Scratch {
    spatial: Vec<f32>,
    row_low: Vec<f32>,
    row_high: Vec<f32>,
}

impl Dct97F32Scratch {
    pub(crate) fn retained_bytes(&self) -> Result<usize, TranscodeAllocationError> {
        let mut total = checked_capacity_bytes::<f32>(self.spatial.capacity())?;
        for capacity in [self.row_low.capacity(), self.row_high.capacity()] {
            total = checked_add_allocation_bytes(total, checked_capacity_bytes::<f32>(capacity)?)?;
        }
        Ok(total)
    }
}

/// Canonical DCT-grid to one-level 9/7 transform from dequantized JPEG blocks.
pub(crate) fn dct_blocks_then_dwt97_f32(
    blocks: &[[i16; 64]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct97F32Scratch,
) -> Result<Dwt97TwoDimensional<f32>, DctTransformError> {
    validate_dct_block_grid(blocks.len(), block_cols, block_rows, width, height)?;
    let sample_count = checked_allocation_len::<f32>(width, height)?;
    validate_workspace(sample_count, true)?;
    try_vec_resize_with(&mut scratch.spatial, sample_count, || 0.0)?;
    idct_blocks_to_plane(blocks, block_cols, width, &mut scratch.spatial);
    let spatial = core::mem::take(&mut scratch.spatial);
    let transformed = lift97_plane_with_rows(&spatial, width, height, scratch);
    scratch.spatial = spatial;
    transformed
}

/// Canonical one-level 9/7 transform of a sample plane, used for every
/// decomposition level after the first.
pub(crate) fn lift97_plane_f32(
    samples: &[f32],
    width: usize,
    height: usize,
    scratch: &mut Dct97F32Scratch,
) -> Result<Dwt97TwoDimensional<f32>, DctTransformError> {
    if width == 0 || height == 0 {
        return Err(DctTransformError::InvalidSamplePlaneDimensions { width, height });
    }
    let sample_count = checked_allocation_len::<f32>(width, height)?;
    if samples.len() != sample_count {
        return Err(DctTransformError::SamplePlaneLengthMismatch {
            sample_count: samples.len(),
            width,
            height,
        });
    }
    validate_workspace(sample_count, false)?;
    lift97_plane_with_rows(samples, width, height, scratch)
}

/// The spatial plane (when present), the row bands, and the output bands each
/// hold one plane of samples.
fn validate_workspace(sample_count: usize, with_spatial: bool) -> Result<(), DctTransformError> {
    let plane_bytes = checked_allocation_bytes::<f32>(sample_count)?;
    let planes = if with_spatial { 3 } else { 2 };
    (0..planes).try_fold(0, |total, _| {
        checked_add_allocation_bytes(total, plane_bytes)
    })?;
    Ok(())
}

/// Reconstruct the `width`-wide sample plane covered by the block grid.
fn idct_blocks_to_plane(blocks: &[[i16; 64]], block_cols: usize, width: usize, plane: &mut [f32]) {
    let block_row_samples = width * 8;
    let block_row = |block_y: usize| &blocks[block_y * block_cols..(block_y + 1) * block_cols];
    if plane.len() >= PARALLEL_IDCT_MIN_SAMPLES {
        plane
            .par_chunks_mut(block_row_samples)
            .enumerate()
            .for_each(|(block_y, rows)| {
                let blocks = block_row(block_y);
                j2k_native::at_simd_level(
                    #[inline(always)]
                    || idct_block_row(blocks, width, rows),
                );
            });
    } else {
        j2k_native::at_simd_level(
            #[inline(always)]
            || {
                for (block_y, rows) in plane.chunks_mut(block_row_samples).enumerate() {
                    idct_block_row(block_row(block_y), width, rows);
                }
            },
        );
    }
}

/// Reconstruct one row of blocks into `rows`, the up to eight plane rows it
/// covers.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn idct_block_row(blocks: &[[i16; 64]], width: usize, rows: &mut [f32]) {
    let output_rows = rows.len() / width;
    for (block_x, block) in blocks.iter().take(width.div_ceil(8)).enumerate() {
        let mut coefficients = [0.0f32; 64];
        for (target, &value) in coefficients.iter_mut().zip(block) {
            *target = f32::from(value);
        }
        let samples = idct8x8::<Host>(&coefficients);
        let x0 = block_x * 8;
        let output_cols = (width - x0).min(8);
        for local_y in 0..output_rows {
            let row = &mut rows[local_y * width + x0..local_y * width + x0 + output_cols];
            row.copy_from_slice(&samples[local_y * 8..local_y * 8 + output_cols]);
        }
    }
}

fn lift97_plane_with_rows(
    samples: &[f32],
    width: usize,
    height: usize,
    scratch: &mut Dct97F32Scratch,
) -> Result<Dwt97TwoDimensional<f32>, DctTransformError> {
    let parallel = samples.len() >= PARALLEL_LIFT_MIN_SAMPLES;
    lift97_plane_in_mode(samples, width, height, scratch, parallel)
}

fn lift97_plane_in_mode(
    samples: &[f32],
    width: usize,
    height: usize,
    scratch: &mut Dct97F32Scratch,
    parallel: bool,
) -> Result<Dwt97TwoDimensional<f32>, DctTransformError> {
    let low_width = low_len(width);
    let high_width = high_len(width);
    let low_height = low_len(height);
    let high_height = high_len(height);

    try_vec_resize_with(
        &mut scratch.row_low,
        checked_allocation_len::<f32>(height, low_width)?,
        || 0.0,
    )?;
    try_vec_resize_with(
        &mut scratch.row_high,
        checked_allocation_len::<f32>(height, high_width)?,
        || 0.0,
    )?;
    lift_rows(
        samples,
        width,
        &mut scratch.row_low,
        &mut scratch.row_high,
        parallel,
    );

    let mut ll = try_vec_filled(checked_allocation_len::<f32>(low_width, low_height)?, 0.0)?;
    let mut lh = try_vec_filled(checked_allocation_len::<f32>(low_width, high_height)?, 0.0)?;
    split_rows(&scratch.row_low, low_width, &mut ll, &mut lh);
    lift_columns(&mut ll, &mut lh, low_width, parallel);

    let mut hl = try_vec_filled(checked_allocation_len::<f32>(high_width, low_height)?, 0.0)?;
    let mut hh = try_vec_filled(checked_allocation_len::<f32>(high_width, high_height)?, 0.0)?;
    split_rows(&scratch.row_high, high_width, &mut hl, &mut hh);
    lift_columns(&mut hl, &mut hh, high_width, parallel);

    Ok(Dwt97TwoDimensional {
        ll,
        hl,
        lh,
        hh,
        low_width,
        low_height,
        high_width,
        high_height,
    })
}

/// Deinterleave and lift every row into the row-low and row-high planes.
fn lift_rows(
    samples: &[f32],
    width: usize,
    row_low: &mut [f32],
    row_high: &mut [f32],
    parallel: bool,
) {
    let low_width = low_len(width);
    let high_width = high_len(width);
    if high_width == 0 {
        // One-sample rows have no lifting partner and are copied unchanged.
        row_low.copy_from_slice(samples);
        return;
    }
    if parallel {
        samples
            .par_chunks(width)
            .zip(row_low.par_chunks_mut(low_width))
            .zip(row_high.par_chunks_mut(high_width))
            .for_each(|((row, low), high)| {
                j2k_native::at_simd_level(
                    #[inline(always)]
                    || lift_row(row, low, high),
                );
            });
    } else {
        j2k_native::at_simd_level(
            #[inline(always)]
            || {
                for ((row, low), high) in samples
                    .chunks(width)
                    .zip(row_low.chunks_mut(low_width))
                    .zip(row_high.chunks_mut(high_width))
                {
                    lift_row(row, low, high);
                }
            },
        );
    }
}

/// Deinterleave one row into its even and odd samples and lift them.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn lift_row(row: &[f32], low: &mut [f32], high: &mut [f32]) {
    for (target, value) in low.iter_mut().zip(row.iter().step_by(2)) {
        *target = *value;
    }
    for (target, value) in high.iter_mut().zip(row.iter().skip(1).step_by(2)) {
        *target = *value;
    }
    lift_split(low, high);
}

/// Canonical lifting of one deinterleaved signal: `low` holds the even
/// samples and `high` the odd samples of the original signal, so
/// `low.len()` is `high.len()` or `high.len() + 1`.
///
/// Interior updates run as straight element-wise loops; the boundary samples,
/// whose symmetric extension repeats the left neighbour, are updated apart.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn lift_split(low: &mut [f32], high: &mut [f32]) {
    let high_count = high.len();
    if high_count == 0 {
        return;
    }
    // Odd steps read low[k] and low[k + 1]; the last odd sample of an
    // even-length signal has no right neighbour.
    let paired_high = high_count.min(low.len() - 1);
    for (step_index, &step) in DWT97_LIFT_STEPS_F32.iter().enumerate() {
        if step_index % 2 == 0 {
            for ((value, &left), &right) in high[..paired_high]
                .iter_mut()
                .zip(&low[..paired_high])
                .zip(&low[1..=paired_high])
            {
                *value = lift97_update::<Host>(*value, left, right, step);
            }
            if paired_high < high_count {
                let left = low[paired_high];
                high[paired_high] = lift97_update::<Host>(high[paired_high], left, left, step);
            }
        } else {
            // Even steps read high[k - 1] and high[k]; low[0] mirrors high[0]
            // and a trailing even sample mirrors the last odd sample.
            low[0] = lift97_update::<Host>(low[0], high[0], high[0], step);
            for ((value, &left), &right) in low[1..high_count]
                .iter_mut()
                .zip(&high[..high_count - 1])
                .zip(&high[1..])
            {
                *value = lift97_update::<Host>(*value, left, right, step);
            }
            if low.len() > high_count {
                let left = high[high_count - 1];
                low[high_count] = lift97_update::<Host>(low[high_count], left, left, step);
            }
        }
    }
    scale_row(low, DWT97_LOW_SCALE_F32);
    scale_row(high, DWT97_HIGH_SCALE_F32);
}

/// Copy even rows of `plane` into `low` and odd rows into `high`.
fn split_rows(plane: &[f32], band_width: usize, low: &mut [f32], high: &mut [f32]) {
    if band_width == 0 {
        return;
    }
    for (row_index, row) in plane.chunks(band_width).enumerate() {
        let start = (row_index / 2) * band_width;
        let target = if row_index % 2 == 0 {
            &mut low[start..start + band_width]
        } else {
            &mut high[start..start + band_width]
        };
        target.copy_from_slice(row);
    }
}

/// Canonical column lifting on row-deinterleaved planes: `low` holds the
/// even rows and `high` the odd rows, each `band_width` samples wide.
fn lift_columns(low: &mut [f32], high: &mut [f32], band_width: usize, parallel: bool) {
    if band_width == 0 || high.is_empty() {
        return;
    }
    if parallel {
        lift_columns_parallel(low, high, band_width);
    } else {
        j2k_native::at_simd_level(
            #[inline(always)]
            || lift_columns_serial(low, high, band_width),
        );
    }
}

/// Rows each parallel column-lifting task updates.
const PARALLEL_LIFT_ROWS_PER_TASK: usize = 16;

#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn lift_columns_serial(low: &mut [f32], high: &mut [f32], band_width: usize) {
    for (step_index, &step) in DWT97_LIFT_STEPS_F32.iter().enumerate() {
        if step_index % 2 == 0 {
            for (k, row) in high.chunks_mut(band_width).enumerate() {
                let (left, right) = odd_step_neighbours(low, band_width, k);
                update_row(row, left, right, step);
            }
        } else {
            for (k, row) in low.chunks_mut(band_width).enumerate() {
                let (left, right) = even_step_neighbours(high, band_width, k);
                update_row(row, left, right, step);
            }
        }
    }
    scale_row(low, DWT97_LOW_SCALE_F32);
    scale_row(high, DWT97_HIGH_SCALE_F32);
}

fn lift_columns_parallel(low: &mut [f32], high: &mut [f32], band_width: usize) {
    let task_samples = band_width * PARALLEL_LIFT_ROWS_PER_TASK;
    for (step_index, &step) in DWT97_LIFT_STEPS_F32.iter().enumerate() {
        let (rows, neighbours, odd) = if step_index % 2 == 0 {
            (&mut *high, &*low, true)
        } else {
            (&mut *low, &*high, false)
        };
        rows.par_chunks_mut(task_samples)
            .enumerate()
            .for_each(|(task, chunk)| {
                j2k_native::at_simd_level(
                    #[inline(always)]
                    || {
                        let first_row = task * PARALLEL_LIFT_ROWS_PER_TASK;
                        for (offset, row) in chunk.chunks_mut(band_width).enumerate() {
                            let k = first_row + offset;
                            let (left, right) = if odd {
                                odd_step_neighbours(neighbours, band_width, k)
                            } else {
                                even_step_neighbours(neighbours, band_width, k)
                            };
                            update_row(row, left, right, step);
                        }
                    },
                );
            });
    }
    for (plane, scale) in [
        (&mut *low, DWT97_LOW_SCALE_F32),
        (&mut *high, DWT97_HIGH_SCALE_F32),
    ] {
        plane.par_chunks_mut(task_samples).for_each(|chunk| {
            j2k_native::at_simd_level(
                #[inline(always)]
                || scale_row(chunk, scale),
            );
        });
    }
}

/// Rows `k` and `k + 1` of the row-low plane, the neighbours of odd row `k`;
/// a missing right neighbour repeats the left one.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn odd_step_neighbours(low: &[f32], band_width: usize, k: usize) -> (&[f32], &[f32]) {
    let left = &low[k * band_width..(k + 1) * band_width];
    let right = low
        .get((k + 1) * band_width..(k + 2) * band_width)
        .unwrap_or(left);
    (left, right)
}

/// Rows `k - 1` and `k` of the row-high plane, the neighbours of even row
/// `k`; row 0 mirrors high row 0 and a trailing row mirrors the last one.
#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn even_step_neighbours(high: &[f32], band_width: usize, k: usize) -> (&[f32], &[f32]) {
    let left_row = k.saturating_sub(1);
    let left = &high[left_row * band_width..(left_row + 1) * band_width];
    let right = high
        .get(k * band_width..(k + 1) * band_width)
        .unwrap_or(left);
    (left, right)
}

#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn update_row(row: &mut [f32], left: &[f32], right: &[f32], step: f32) {
    for ((value, &left), &right) in row.iter_mut().zip(left).zip(right) {
        *value = lift97_update::<Host>(*value, left, right, step);
    }
}

#[inline(always)]
#[expect(
    clippy::inline_always,
    reason = "target features reach only code inlined into the SIMD dispatch"
)]
fn scale_row(row: &mut [f32], scale: f32) {
    for value in row {
        *value = Host::mul(*value, scale);
    }
}

#[cfg(test)]
mod tests {
    use j2k_codec_math::transcode97::forward_lift97;

    use super::*;

    fn next_unit(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = u32::try_from(*state >> 41).expect("23-bit fraction");
        f32::from_bits(0x3f80_0000 | bits) - 1.5
    }

    /// Reference: canonical interleaved lifting of each row, then each column.
    fn reference_plane(samples: &[f32], width: usize, height: usize) -> Dwt97TwoDimensional<f32> {
        let mut plane = samples.to_vec();
        for row in plane.chunks_mut(width) {
            forward_lift97::<Host>(row);
        }
        for x in 0..width {
            let mut column: Vec<f32> = (0..height).map(|y| plane[y * width + x]).collect();
            forward_lift97::<Host>(&mut column);
            for (y, value) in column.into_iter().enumerate() {
                plane[y * width + x] = value;
            }
        }
        let (low_width, high_width) = (low_len(width), high_len(width));
        let (low_height, high_height) = (low_len(height), high_len(height));
        let band = |x_parity: usize, y_parity: usize, band_width: usize, band_height: usize| {
            let mut band = Vec::with_capacity(band_width * band_height);
            for y in 0..band_height {
                for x in 0..band_width {
                    band.push(plane[(2 * y + y_parity) * width + 2 * x + x_parity]);
                }
            }
            band
        };
        Dwt97TwoDimensional {
            ll: band(0, 0, low_width, low_height),
            hl: band(1, 0, high_width, low_height),
            lh: band(0, 1, low_width, high_height),
            hh: band(1, 1, high_width, high_height),
            low_width,
            low_height,
            high_width,
            high_height,
        }
    }

    fn assert_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(a.to_bits(), e.to_bits(), "{label}[{index}]: {a} vs {e}");
        }
    }

    #[test]
    fn deinterleaved_lifting_is_bit_identical_to_interleaved_reference() {
        let mut state = 0x0123_4567_89ab_cdef_u64;
        for (width, height) in [
            (1usize, 1usize),
            (1, 7),
            (7, 1),
            (2, 2),
            (3, 5),
            (8, 8),
            (9, 7),
            (16, 23),
            (65, 64),
            (130, 67),
            (37, 200),
        ] {
            let samples: Vec<f32> = (0..width * height)
                .map(|_| next_unit(&mut state) * 300.0)
                .collect();
            let expected = reference_plane(&samples, width, height);
            for parallel in [false, true] {
                let mut scratch = Dct97F32Scratch::default();
                let actual = lift97_plane_in_mode(&samples, width, height, &mut scratch, parallel)
                    .expect("small plane fits the workspace");
                let label = format!("{width}x{height} parallel={parallel}");
                assert_bits_eq(&actual.ll, &expected.ll, &format!("{label} ll"));
                assert_bits_eq(&actual.hl, &expected.hl, &format!("{label} hl"));
                assert_bits_eq(&actual.lh, &expected.lh, &format!("{label} lh"));
                assert_bits_eq(&actual.hh, &expected.hh, &format!("{label} hh"));
            }
        }
    }

    #[test]
    fn block_grid_transform_matches_per_block_idct_then_reference_lifting() {
        let (block_cols, block_rows, width, height) = (5usize, 4usize, 37usize, 29usize);
        let blocks: Vec<[i16; 64]> = (0..block_cols * block_rows)
            .map(|block| {
                core::array::from_fn(|index| {
                    i16::try_from((block * 37 + index * 11) % 97).expect("small") - 48
                })
            })
            .collect();
        let mut plane = vec![0.0f32; width * height];
        for y in 0..height {
            for x in 0..width {
                let block = blocks[(y / 8) * block_cols + x / 8].map(f32::from);
                plane[y * width + x] = idct8x8::<Host>(&block)[(y % 8) * 8 + x % 8];
            }
        }
        let expected = reference_plane(&plane, width, height);
        let mut scratch = Dct97F32Scratch::default();
        let actual =
            dct_blocks_then_dwt97_f32(&blocks, block_cols, block_rows, width, height, &mut scratch)
                .expect("covered grid");
        assert_bits_eq(&actual.ll, &expected.ll, "ll");
        assert_bits_eq(&actual.hl, &expected.hl, "hl");
        assert_bits_eq(&actual.lh, &expected.lh, "lh");
        assert_bits_eq(&actual.hh, &expected.hh, "hh");
    }

    #[test]
    fn rejects_uncovered_grid_and_mismatched_plane() {
        let mut scratch = Dct97F32Scratch::default();
        assert!(dct_blocks_then_dwt97_f32(&[[0; 64]], 1, 1, 9, 8, &mut scratch).is_err());
        assert!(matches!(
            lift97_plane_f32(&[0.0; 5], 2, 3, &mut scratch),
            Err(DctTransformError::SamplePlaneLengthMismatch { .. })
        ));
        assert!(matches!(
            lift97_plane_f32(&[], 0, 3, &mut scratch),
            Err(DctTransformError::InvalidSamplePlaneDimensions { .. })
        ));
    }
}
