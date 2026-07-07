// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constrained 2D DCT to 5/3 wavelet experiments.
//!
//! The direct float path projects an 8x8 DCT block into one separable
//! single-level 5/3 result without first storing the 8x8 spatial samples. The
//! reference path materializes samples to keep the oracle easy to audit.

use crate::dct_grid::{high_len, idct8_basis, low_len, validate_dct_block_grid};
use crate::{DctGridError, Dwt53TwoDimensional};

#[cfg(test)]
impl Dwt53TwoDimensional<f64> {
    /// Maximum absolute coefficient difference across matching bands.
    #[must_use]
    pub(crate) fn max_abs_diff(&self, other: &Self) -> f64 {
        assert_eq!(self.low_width, other.low_width);
        assert_eq!(self.low_height, other.low_height);
        assert_eq!(self.high_width, other.high_width);
        assert_eq!(self.high_height, other.high_height);

        self.ll
            .iter()
            .zip(other.ll.iter())
            .chain(self.hl.iter().zip(other.hl.iter()))
            .chain(self.lh.iter().zip(other.lh.iter()))
            .chain(self.hh.iter().zip(other.hh.iter()))
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max)
    }
}

/// Scratch storage for repeated DCT-grid to 5/3 projection calls.
///
/// Reuse one value per worker when transforming many components or tiles with
/// matching geometry. The scratch caches linearized 5/3 weight rows; it does
/// not store spatial samples.
#[derive(Debug, Default)]
pub(crate) struct Dct53GridScratch {
    x_weights: Dwt53WeightRows,
    y_weights: Dwt53WeightRows,
}

impl Dct53GridScratch {
    #[cfg(test)]
    fn weight_row_capacity(&self) -> usize {
        self.x_weights.weight_capacity() + self.y_weights.weight_capacity()
    }
}

/// Map an adjacent 8x8 DCT block grid directly into a linearized one-level 2D
/// 5/3 result for the logical component dimensions.
///
/// Padded JPEG edge samples outside `width x height` are ignored.
pub fn dct8x8_blocks_to_dwt53_float_linear(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<Dwt53TwoDimensional<f64>, DctGridError> {
    let mut scratch = Dct53GridScratch::default();
    dct8x8_blocks_to_dwt53_float_linear_with_scratch(
        blocks,
        block_cols,
        block_rows,
        width,
        height,
        &mut scratch,
    )
}

/// Map an adjacent 8x8 DCT block grid directly into a linearized one-level 2D
/// 5/3 result using caller-owned scratch for reusable weight rows.
pub(crate) fn dct8x8_blocks_to_dwt53_float_linear_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct53GridScratch,
) -> Result<Dwt53TwoDimensional<f64>, DctGridError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;

    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);
    scratch.x_weights.ensure_sample_len(width);
    scratch.y_weights.ensure_sample_len(height);
    let x_weights = &scratch.x_weights;
    let y_weights = &scratch.y_weights;

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut hl = Vec::with_capacity(high_width * low_height);
    let mut lh = Vec::with_capacity(low_width * high_height);
    let mut hh = Vec::with_capacity(high_width * high_height);

    for y in 0..low_height {
        for x in 0..low_width {
            ll.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.low[y].taps,
                &x_weights.low[x].taps,
            ));
        }
        for x in 0..high_width {
            hl.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.low[y].taps,
                &x_weights.high[x].taps,
            ));
        }
    }

    for y in 0..high_height {
        for x in 0..low_width {
            lh.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.high[y].taps,
                &x_weights.low[x].taps,
            ));
        }
        for x in 0..high_width {
            hh.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.high[y].taps,
                &x_weights.high[x].taps,
            ));
        }
    }

    Ok(Dwt53TwoDimensional {
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

/// Reference path for a DCT block grid:
/// DCT coefficients -> float IDCT samples -> separable linearized 5/3.
pub fn dct8x8_blocks_then_dwt53_float(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<Dwt53TwoDimensional<f64>, DctGridError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;

    let mut samples = Vec::with_capacity(width * height);
    for y in 0..height {
        let block_y = y / 8;
        let local_y = y % 8;
        for x in 0..width {
            let block_x = x / 8;
            let local_x = x % 8;
            let block = &blocks[block_y * block_cols + block_x];
            samples.push(idct8x8_sample(block, local_x, local_y));
        }
    }

    Ok(linearized_53_2d_from_plane(&samples, width, height))
}

fn project_dct_grid(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    y_weights: &[SparseWeightTap],
    x_weights: &[SparseWeightTap],
) -> f64 {
    let mut output = 0.0;

    for &SparseWeightTap {
        sample_idx: sample_y,
        weight: y_weight,
    } in y_weights
    {
        let block_y = sample_y / 8;
        let local_y = sample_y % 8;

        for &SparseWeightTap {
            sample_idx: sample_x,
            weight: x_weight,
        } in x_weights
        {
            let block_x = sample_x / 8;
            let local_x = sample_x % 8;
            let block = &blocks[block_y * block_cols + block_x];
            let sample_weight = y_weight * x_weight;

            for (freq_y, coefficient_row) in block.iter().enumerate() {
                let y_basis = idct8_basis(local_y, freq_y);
                for (freq_x, coefficient) in coefficient_row.iter().copied().enumerate() {
                    output += sample_weight * y_basis * idct8_basis(local_x, freq_x) * coefficient;
                }
            }
        }
    }

    output
}

fn idct8x8_sample(block: &[[f64; 8]; 8], x: usize, y: usize) -> f64 {
    let mut sample = 0.0;
    for (freq_y, row) in block.iter().enumerate() {
        let y_basis = idct8_basis(y, freq_y);
        for (freq_x, coefficient) in row.iter().copied().enumerate() {
            sample += coefficient * y_basis * idct8_basis(x, freq_x);
        }
    }
    sample
}

pub(crate) fn linearized_53_2d_from_plane(
    samples: &[f64],
    width: usize,
    height: usize,
) -> Dwt53TwoDimensional<f64> {
    debug_assert_eq!(samples.len(), width * height);

    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);

    let mut row_low = Vec::with_capacity(height * low_width);
    let mut row_high = Vec::with_capacity(height * high_width);
    for y in 0..height {
        let start = y * width;
        let row = &samples[start..start + width];
        let transformed = linearized_53_from_sample_slice(row);
        row_low.extend(transformed.low);
        row_high.extend(transformed.high);
    }

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut lh = Vec::with_capacity(low_width * high_height);
    for x in 0..low_width {
        let column = column_from_rows(&row_low, low_width, x, height);
        let transformed = linearized_53_from_sample_slice(&column);
        ll.extend(transformed.low);
        lh.extend(transformed.high);
    }

    let mut hl = Vec::with_capacity(high_width * low_height);
    let mut hh = Vec::with_capacity(high_width * high_height);
    for x in 0..high_width {
        let column = column_from_rows(&row_high, high_width, x, height);
        let transformed = linearized_53_from_sample_slice(&column);
        hl.extend(transformed.low);
        hh.extend(transformed.high);
    }

    Dwt53TwoDimensional {
        ll: transpose_band(&ll, low_height, low_width),
        hl: transpose_band(&hl, low_height, high_width),
        lh: transpose_band(&lh, high_height, low_width),
        hh: transpose_band(&hh, high_height, high_width),
        low_width,
        low_height,
        high_width,
        high_height,
    }
}

fn column_from_rows(rows: &[f64], stride: usize, x: usize, height: usize) -> Vec<f64> {
    (0..height).map(|y| rows[y * stride + x]).collect()
}

fn transpose_band(column_major: &[f64], height: usize, width: usize) -> Vec<f64> {
    let mut row_major = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            row_major.push(column_major[x * height + y]);
        }
    }
    row_major
}

fn linearized_53_from_sample_slice(samples: &[f64]) -> Dwt53OneDimensional {
    let mut high = Vec::with_capacity(high_len(samples.len()));
    for odd_idx in (1..samples.len()).step_by(2) {
        let left = samples[odd_idx - 1];
        let right = samples.get(odd_idx + 1).copied().unwrap_or(left);
        high.push(samples[odd_idx] - ((left + right) * 0.5));
    }

    let mut low = Vec::with_capacity(low_len(samples.len()));
    for even_idx in (0..samples.len()).step_by(2) {
        let current = samples[even_idx];
        let even_output_idx = even_idx / 2;
        let left_high = even_output_idx.checked_sub(1).and_then(|idx| high.get(idx));
        let right_high = high.get(even_output_idx);
        let update = match (left_high, right_high) {
            (Some(left), Some(right)) => (*left + *right) * 0.25,
            (None, Some(right)) => *right * 0.5,
            (Some(left), None) => *left * 0.5,
            (None, None) => 0.0,
        };
        low.push(current + update);
    }

    Dwt53OneDimensional { low, high }
}

fn validate_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), DctGridError> {
    validate_dct_block_grid(block_count, block_cols, block_rows, width, height)
}

#[derive(Debug, Default)]
struct Dwt53WeightRows {
    sample_len: Option<usize>,
    low: Vec<SparseWeightRow>,
    high: Vec<SparseWeightRow>,
}

impl Dwt53WeightRows {
    fn ensure_sample_len(&mut self, sample_len: usize) {
        if self.sample_len == Some(sample_len) {
            return;
        }

        resize_weight_rows(&mut self.low, low_len(sample_len), 5);
        resize_weight_rows(&mut self.high, high_len(sample_len), 3);

        for sample_idx in 0..sample_len {
            let mut basis = vec![0.0; sample_len];
            basis[sample_idx] = 1.0;
            let transformed = linearized_53_from_sample_slice(&basis);
            for (row, &weight) in self.low.iter_mut().zip(transformed.low.iter()) {
                if weight != 0.0 {
                    row.taps.push(SparseWeightTap { sample_idx, weight });
                }
            }
            for (row, &weight) in self.high.iter_mut().zip(transformed.high.iter()) {
                if weight != 0.0 {
                    row.taps.push(SparseWeightTap { sample_idx, weight });
                }
            }
        }

        self.sample_len = Some(sample_len);
    }

    #[cfg(test)]
    fn weight_capacity(&self) -> usize {
        self.low
            .iter()
            .map(|row| row.taps.capacity())
            .sum::<usize>()
            + self
                .high
                .iter()
                .map(|row| row.taps.capacity())
                .sum::<usize>()
    }
}

fn resize_weight_rows(rows: &mut Vec<SparseWeightRow>, row_count: usize, max_taps: usize) {
    if rows.len() < row_count {
        rows.resize_with(row_count, SparseWeightRow::default);
    }
    for row in rows.iter_mut().take(row_count) {
        row.taps.clear();
        if row.taps.capacity() < max_taps {
            row.taps.reserve_exact(max_taps - row.taps.capacity());
        }
    }
    rows.truncate(row_count);
}

#[derive(Debug, Default)]
struct SparseWeightRow {
    taps: Vec<SparseWeightTap>,
}

#[derive(Debug, Clone, Copy)]
struct SparseWeightTap {
    sample_idx: usize,
    weight: f64,
}

struct Dwt53OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dct8x8_grid_scratch_reuses_weight_rows_for_same_geometry() {
        let blocks = synthetic_grid_blocks(2, 2);
        let mut scratch = Dct53GridScratch::default();

        let direct =
            dct8x8_blocks_to_dwt53_float_linear_with_scratch(&blocks, 2, 2, 13, 11, &mut scratch)
                .expect("valid DCT grid");
        let stateless =
            dct8x8_blocks_to_dwt53_float_linear(&blocks, 2, 2, 13, 11).expect("valid DCT grid");
        let capacity_after_first = scratch.weight_row_capacity();

        let repeated =
            dct8x8_blocks_to_dwt53_float_linear_with_scratch(&blocks, 2, 2, 13, 11, &mut scratch)
                .expect("valid DCT grid");

        assert!(capacity_after_first > 0);
        assert_eq!(scratch.weight_row_capacity(), capacity_after_first);
        assert!(direct.max_abs_diff(&stateless) <= 1.0e-9);
        assert!(repeated.max_abs_diff(&stateless) <= 1.0e-9);
    }

    #[test]
    fn dct8x8_grid_scratch_uses_sparse_weight_rows_for_wsi_tile() {
        let dim = 224_usize;
        let block_cols = dim / 8;
        let block_rows = dim / 8;
        let blocks = vec![[[0.0; 8]; 8]; block_cols * block_rows];
        let mut scratch = Dct53GridScratch::default();

        dct8x8_blocks_to_dwt53_float_linear_with_scratch(
            &blocks,
            block_cols,
            block_rows,
            dim,
            dim,
            &mut scratch,
        )
        .expect("valid DCT grid");

        assert!(
            scratch.weight_row_capacity() <= dim * 10,
            "5/3 grid weights should stay sparse at WSI tile sizes, got capacity {}",
            scratch.weight_row_capacity()
        );
    }

    fn synthetic_grid_blocks(block_cols: usize, block_rows: usize) -> Vec<[[f64; 8]; 8]> {
        let mut blocks = Vec::with_capacity(block_cols * block_rows);
        for block_y in 0..block_rows {
            for block_x in 0..block_cols {
                let mut block = [[0.0; 8]; 8];
                block[0][0] = 192.0 + (block_x * 17 + block_y * 23) as f64;
                block[0][1] = -31.0 + block_x as f64;
                block[1][0] = 27.0 - block_y as f64;
                block[2][3] = 9.0;
                block[7][7] = -6.0;
                blocks.push(block);
            }
        }
        blocks
    }
}
