// SPDX-License-Identifier: Apache-2.0

//! Optional acceleration hooks for coefficient-domain transform stages.
//!
//! These hooks are intentionally narrow: accelerated backends may replace the
//! direct DCT-grid to one-level wavelet projection, while the scalar path
//! remains the default oracle and fallback.

use crate::dct53_2d::Dwt53TwoDimensional;
use crate::dct97_2d::Dwt97TwoDimensional;
use rayon::prelude::*;
use signinum_jpeg::transcode::idct_islow_block;

const REVERSIBLE_DWT53_UNSUPPORTED_GRID: &str =
    "reversible DCT 5/3 job has unsupported grid geometry";

/// Direct DCT-grid to one-level reversible integer 5/3 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToReversibleDwt53Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub dequantized_blocks: &'a [[i16; 64]],
    /// Number of DCT block columns in `dequantized_blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `dequantized_blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// One separable single-level reversible integer 5/3 transform result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReversibleDwt53FirstLevel {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<i32>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<i32>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<i32>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<i32>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Direct DCT-grid to one-level 5/3 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToDwt53Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// Direct DCT-grid to one-level 9/7 projection job.
#[derive(Debug, Clone, Copy)]
pub struct DctGridToDwt97Job<'a> {
    /// Natural-order, dequantized 8x8 DCT blocks.
    pub blocks: &'a [[[f64; 8]; 8]],
    /// Number of DCT block columns in `blocks`.
    pub block_cols: usize,
    /// Number of DCT block rows in `blocks`.
    pub block_rows: usize,
    /// Logical component width in samples.
    pub width: usize,
    /// Logical component height in samples.
    pub height: usize,
}

/// Optional backend for SIMD, GPU, or other accelerated transform stages.
pub trait DctToWaveletStageAccelerator {
    /// Optionally compute the direct DCT-grid to one-level reversible integer
    /// 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job bit-exactly
    /// relative to signinum's scalar integer oracle. Return `Ok(None)` to use
    /// the scalar fallback.
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        _job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        Ok(None)
    }

    /// Optionally compute a same-geometry batch of direct DCT-grid to
    /// one-level reversible integer 5/3 projections.
    ///
    /// Backends should return outputs in the same order as `jobs`. Return
    /// `Ok(None)` to use the scalar per-component fallback.
    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        _jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 5/3 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt53(
        &mut self,
        _job: DctGridToDwt53Job<'_>,
    ) -> Result<Option<Dwt53TwoDimensional<f64>>, &'static str> {
        Ok(None)
    }

    /// Optionally compute the direct DCT-grid to one-level 9/7 projection.
    ///
    /// Return `Ok(Some(output))` when the backend handled the job. Return
    /// `Ok(None)` to use the scalar fallback.
    fn dct_grid_to_dwt97(
        &mut self,
        _job: DctGridToDwt97Job<'_>,
    ) -> Result<Option<Dwt97TwoDimensional<f64>>, &'static str> {
        Ok(None)
    }
}

/// Accelerator that always uses the scalar CPU fallback.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuOnlyDctToWaveletStageAccelerator;

impl DctToWaveletStageAccelerator for CpuOnlyDctToWaveletStageAccelerator {}

/// CPU/Rayon accelerator for the exact reversible integer 5/3 first level.
///
/// This backend keeps signinum's scalar ISLOW IDCT semantics as the oracle:
/// each 8x8 block is decoded with `signinum-jpeg`, level-shifted to signed
/// component samples, then transformed with reversible integer 5/3 lifting.
#[derive(Debug, Default, Clone)]
pub struct RayonReversibleDwt53Accelerator {
    attempts: usize,
    dispatches: usize,
    batch_attempts: usize,
    batch_dispatches: usize,
}

impl RayonReversibleDwt53Accelerator {
    /// Number of reversible 5/3 jobs offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_attempts(&self) -> usize {
        self.attempts
    }

    /// Number of reversible 5/3 jobs handled by this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_dispatches(&self) -> usize {
        self.dispatches
    }

    /// Number of reversible 5/3 batches offered to this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_attempts(&self) -> usize {
        self.batch_attempts
    }

    /// Number of reversible 5/3 batches handled by this accelerator.
    #[must_use]
    pub const fn reversible_dwt53_batch_dispatches(&self) -> usize {
        self.batch_dispatches
    }
}

impl DctToWaveletStageAccelerator for RayonReversibleDwt53Accelerator {
    fn dct_grid_to_reversible_dwt53(
        &mut self,
        job: DctGridToReversibleDwt53Job<'_>,
    ) -> Result<Option<ReversibleDwt53FirstLevel>, &'static str> {
        self.attempts = self.attempts.saturating_add(1);
        let output = reversible_dwt53_first_level_rayon(job)?;
        self.dispatches = self.dispatches.saturating_add(1);
        Ok(Some(output))
    }

    fn dct_grid_to_reversible_dwt53_batch(
        &mut self,
        jobs: &[DctGridToReversibleDwt53Job<'_>],
    ) -> Result<Option<Vec<ReversibleDwt53FirstLevel>>, &'static str> {
        self.batch_attempts = self.batch_attempts.saturating_add(1);
        let mut output = Vec::with_capacity(jobs.len());
        for job in jobs {
            output.push(reversible_dwt53_first_level_rayon(*job)?);
        }
        self.batch_dispatches = self.batch_dispatches.saturating_add(1);
        Ok(Some(output))
    }
}

/// Decode the job's dequantized DCT blocks into signinum's signed integer
/// component sample blocks.
///
/// This is public so hybrid GPU backends can keep JPEG parsing and exact IDCT
/// on CPU while offloading the reversible 5/3 projection.
pub fn idct_blocks_to_signed_samples_rayon(blocks: &[[i16; 64]]) -> Vec<[i32; 64]> {
    blocks
        .par_iter()
        .map(|block| {
            let decoded = idct_islow_block(block);
            decoded.map(|sample| i32::from(sample) - 128)
        })
        .collect()
}

/// Compute one exact reversible integer 5/3 level from already decoded
/// block-local signed samples.
pub fn reversible_dwt53_first_level_from_block_samples(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<ReversibleDwt53FirstLevel, &'static str> {
    validate_reversible_grid(block_samples.len(), block_cols, block_rows, width, height)?;

    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;

    let low_rows: Vec<(Vec<i32>, Vec<i32>)> = (0..low_height)
        .into_par_iter()
        .map(|output_y| {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                row.push(vertical_low_53_i32_at(
                    block_samples,
                    block_cols,
                    width,
                    height,
                    x,
                    output_y,
                ));
            }
            reversible_lift_53_i32(&mut row);
            (
                row.iter().step_by(2).copied().collect(),
                row.iter().skip(1).step_by(2).copied().collect(),
            )
        })
        .collect();
    let high_rows: Vec<(Vec<i32>, Vec<i32>)> = (0..high_height)
        .into_par_iter()
        .map(|output_y| {
            let mut row = Vec::with_capacity(width);
            for x in 0..width {
                row.push(vertical_high_53_i32_at(
                    block_samples,
                    block_cols,
                    width,
                    height,
                    x,
                    output_y,
                ));
            }
            reversible_lift_53_i32(&mut row);
            (
                row.iter().step_by(2).copied().collect(),
                row.iter().skip(1).step_by(2).copied().collect(),
            )
        })
        .collect();

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut hl = Vec::with_capacity(high_width * low_height);
    for (low, high) in low_rows {
        ll.extend(low);
        hl.extend(high);
    }

    let mut lh = Vec::with_capacity(low_width * high_height);
    let mut hh = Vec::with_capacity(high_width * high_height);
    for (low, high) in high_rows {
        lh.extend(low);
        hh.extend(high);
    }

    Ok(ReversibleDwt53FirstLevel {
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

fn reversible_dwt53_first_level_rayon(
    job: DctGridToReversibleDwt53Job<'_>,
) -> Result<ReversibleDwt53FirstLevel, &'static str> {
    validate_reversible_grid(
        job.dequantized_blocks.len(),
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )?;
    let block_samples = idct_blocks_to_signed_samples_rayon(job.dequantized_blocks);
    reversible_dwt53_first_level_from_block_samples(
        &block_samples,
        job.block_cols,
        job.block_rows,
        job.width,
        job.height,
    )
}

fn validate_reversible_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), &'static str> {
    let expected_blocks = block_cols
        .checked_mul(block_rows)
        .ok_or(REVERSIBLE_DWT53_UNSUPPORTED_GRID)?;
    let covered_width = block_cols
        .checked_mul(8)
        .ok_or(REVERSIBLE_DWT53_UNSUPPORTED_GRID)?;
    let covered_height = block_rows
        .checked_mul(8)
        .ok_or(REVERSIBLE_DWT53_UNSUPPORTED_GRID)?;

    if block_count != expected_blocks
        || width == 0
        || height == 0
        || width > covered_width
        || height > covered_height
    {
        return Err(REVERSIBLE_DWT53_UNSUPPORTED_GRID);
    }
    Ok(())
}

fn vertical_low_53_i32_at(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    low_idx: usize,
) -> i32 {
    let even_idx = low_idx * 2;
    let current = component_sample_i32(block_samples, block_cols, width, height, x, even_idx);
    if height < 2 {
        return current;
    }

    if height.is_multiple_of(2) {
        let right = vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx);
        if low_idx == 0 {
            return current + floor_div_i32(right + 1, 2);
        }
        let left =
            vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx - 1);
        return current + floor_div_i32(left + right + 2, 4);
    }

    let high_len = height / 2;
    if high_len == 0 {
        return current;
    }
    let left = if low_idx > 0 {
        vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx - 1)
    } else {
        vertical_high_53_i32_at(block_samples, block_cols, width, height, x, 0)
    };
    let right = if low_idx < high_len {
        vertical_high_53_i32_at(block_samples, block_cols, width, height, x, low_idx)
    } else {
        left
    };
    current + floor_div_i32(left + right + 2, 4)
}

fn vertical_high_53_i32_at(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    high_idx: usize,
) -> i32 {
    let odd_idx = high_idx * 2 + 1;
    let current = component_sample_i32(block_samples, block_cols, width, height, x, odd_idx);
    let left = component_sample_i32(block_samples, block_cols, width, height, x, odd_idx - 1);
    if height.is_multiple_of(2) && odd_idx + 1 == height {
        return current - left;
    }

    let right_idx = if odd_idx + 1 < height {
        odd_idx + 1
    } else {
        height - 1
    };
    let right = component_sample_i32(block_samples, block_cols, width, height, x, right_idx);
    current - floor_div_i32(left + right, 2)
}

fn component_sample_i32(
    block_samples: &[[i32; 64]],
    block_cols: usize,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> i32 {
    debug_assert!(x < width);
    debug_assert!(y < height);
    let block_x = x / 8;
    let block_y = y / 8;
    let block_idx = block_y * block_cols + block_x;
    let local_idx = (y % 8) * 8 + (x % 8);
    block_samples[block_idx][local_idx]
}

fn reversible_lift_53_i32(values: &mut [i32]) {
    let n = values.len();
    if n < 2 {
        return;
    }

    if n.is_multiple_of(2) {
        for i in (1..n - 1).step_by(2) {
            values[i] -= floor_div_i32(values[i - 1] + values[i + 1], 2);
        }
        values[n - 1] -= values[n - 2];

        values[0] += floor_div_i32(values[1] + 1, 2);
        for i in (2..n).step_by(2) {
            values[i] += floor_div_i32(values[i - 1] + values[i + 1] + 2, 4);
        }
        return;
    }

    let last_even = n - 1;
    for i in (1..n).step_by(2) {
        let right = values.get(i + 1).copied().unwrap_or(values[last_even]);
        values[i] -= floor_div_i32(values[i - 1] + right, 2);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { values[i - 1] } else { values[1] };
        let right = values.get(i + 1).copied().unwrap_or(left);
        values[i] += floor_div_i32(left + right + 2, 4);
    }
}

fn floor_div_i32(numerator: i32, denominator: i32) -> i32 {
    numerator.div_euclid(denominator)
}
