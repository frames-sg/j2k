// SPDX-License-Identifier: Apache-2.0

//! Constrained 2D DCT to irreversible 9/7 wavelet experiments.
//!
//! The direct float path projects an 8x8 DCT block grid into one separable
//! single-level 9/7 result without storing the spatial component plane. The
//! reference path materializes samples only for validation.

use core::f64::consts::PI;
use core::fmt;

const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
const KAPPA: f64 = 1.230_174_104_914_001;
const INV_KAPPA: f64 = 1.0 / KAPPA;

/// One separable single-level 2D 9/7 transform result.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt97TwoDimensional<T> {
    /// Low-horizontal, low-vertical band.
    pub ll: Vec<T>,
    /// High-horizontal, low-vertical band.
    pub hl: Vec<T>,
    /// Low-horizontal, high-vertical band.
    pub lh: Vec<T>,
    /// High-horizontal, high-vertical band.
    pub hh: Vec<T>,
    /// Width of horizontally low-pass bands.
    pub low_width: usize,
    /// Height of vertically low-pass bands.
    pub low_height: usize,
    /// Width of horizontally high-pass bands.
    pub high_width: usize,
    /// Height of vertically high-pass bands.
    pub high_height: usize,
}

/// Error returned when a DCT block grid cannot cover the requested component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dct97GridError {
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
}

impl Dct97GridError {
    /// Number of supplied 8x8 DCT blocks.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.block_count
    }

    /// Declared block columns.
    #[must_use]
    pub const fn block_cols(self) -> usize {
        self.block_cols
    }

    /// Declared block rows.
    #[must_use]
    pub const fn block_rows(self) -> usize {
        self.block_rows
    }

    /// Requested component width.
    #[must_use]
    pub const fn width(self) -> usize {
        self.width
    }

    /// Requested component height.
    #[must_use]
    pub const fn height(self) -> usize {
        self.height
    }
}

impl fmt::Display for Dct97GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DCT grid has {} blocks for {}x{} grid covering requested {}x{} samples",
            self.block_count, self.block_cols, self.block_rows, self.width, self.height
        )
    }
}

impl std::error::Error for Dct97GridError {}

/// Scratch storage for repeated DCT-grid to 9/7 projection calls.
#[derive(Debug, Default)]
pub struct Dct97GridScratch {
    x_weights: Dwt97WeightRows,
    y_weights: Dwt97WeightRows,
}

impl Dct97GridScratch {
    /// Aggregate capacity of cached weight rows.
    #[must_use]
    pub fn weight_row_capacity(&self) -> usize {
        self.x_weights.weight_capacity() + self.y_weights.weight_capacity()
    }
}

/// Map an adjacent 8x8 DCT block grid directly into a linearized one-level 2D
/// 9/7 result for the logical component dimensions.
///
/// Padded JPEG edge samples outside `width x height` are ignored.
pub fn dct8x8_blocks_to_dwt97_float_linear_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct97GridScratch,
) -> Result<Dwt97TwoDimensional<f64>, Dct97GridError> {
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
                &y_weights.low[y],
                &x_weights.low[x],
            ));
        }
        for x in 0..high_width {
            hl.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.low[y],
                &x_weights.high[x],
            ));
        }
    }

    for y in 0..high_height {
        for x in 0..low_width {
            lh.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.high[y],
                &x_weights.low[x],
            ));
        }
        for x in 0..high_width {
            hh.push(project_dct_grid(
                blocks,
                block_cols,
                &y_weights.high[y],
                &x_weights.high[x],
            ));
        }
    }

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

/// Reference path for a DCT block grid:
/// DCT coefficients -> float IDCT samples -> separable linearized 9/7.
pub fn dct8x8_blocks_then_dwt97_float(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<Dwt97TwoDimensional<f64>, Dct97GridError> {
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

    Ok(linearized_97_2d_from_plane(&samples, width, height))
}

pub(crate) fn linearized_97_2d_from_plane(
    samples: &[f64],
    width: usize,
    height: usize,
) -> Dwt97TwoDimensional<f64> {
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
        let transformed = linearized_97_from_sample_slice(row);
        row_low.extend(transformed.low);
        row_high.extend(transformed.high);
    }

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut lh = Vec::with_capacity(low_width * high_height);
    for x in 0..low_width {
        let column = column_from_rows(&row_low, low_width, x, height);
        let transformed = linearized_97_from_sample_slice(&column);
        ll.extend(transformed.low);
        lh.extend(transformed.high);
    }

    let mut hl = Vec::with_capacity(high_width * low_height);
    let mut hh = Vec::with_capacity(high_width * high_height);
    for x in 0..high_width {
        let column = column_from_rows(&row_high, high_width, x, height);
        let transformed = linearized_97_from_sample_slice(&column);
        hl.extend(transformed.low);
        hh.extend(transformed.high);
    }

    Dwt97TwoDimensional {
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

fn project_dct_grid(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    y_weights: &[f64],
    x_weights: &[f64],
) -> f64 {
    let mut output = 0.0;

    for (sample_y, &y_weight) in y_weights.iter().enumerate() {
        if y_weight == 0.0 {
            continue;
        }
        let block_y = sample_y / 8;
        let local_y = sample_y % 8;

        for (sample_x, &x_weight) in x_weights.iter().enumerate() {
            if x_weight == 0.0 {
                continue;
            }
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

fn linearized_97_from_sample_slice(samples: &[f64]) -> Dwt97OneDimensional {
    let mut lifted = samples.to_vec();
    forward_lift_97(&mut lifted);

    Dwt97OneDimensional {
        low: lifted.iter().step_by(2).copied().collect(),
        high: lifted.iter().skip(1).step_by(2).copied().collect(),
    }
}

fn forward_lift_97(data: &mut [f64]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += ALPHA * (left + right);
    }

    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n { data[i + 1] } else { data[i - 1] };
        data[i] += BETA * (left + right);
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += GAMMA * (left + right);
    }

    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n { data[i + 1] } else { data[i - 1] };
        data[i] += DELTA * (left + right);
    }

    for i in (0..n).step_by(2) {
        data[i] *= KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= INV_KAPPA;
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

fn idct8_basis(sample_idx: usize, freq: usize) -> f64 {
    debug_assert!(sample_idx < 8);
    debug_assert!(freq < 8);

    let scale = if freq == 0 {
        (1.0_f64 / 8.0).sqrt()
    } else {
        (2.0_f64 / 8.0).sqrt()
    };
    scale * (((sample_idx as f64 + 0.5) * freq as f64 * PI) / 8.0).cos()
}

fn low_len(sample_len: usize) -> usize {
    sample_len.div_ceil(2)
}

fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}

fn validate_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), Dct97GridError> {
    let expected_blocks = block_cols.saturating_mul(block_rows);
    let covered_width = block_cols.saturating_mul(8);
    let covered_height = block_rows.saturating_mul(8);
    if block_count != expected_blocks
        || width == 0
        || height == 0
        || width > covered_width
        || height > covered_height
    {
        return Err(Dct97GridError {
            block_count,
            block_cols,
            block_rows,
            width,
            height,
        });
    }

    Ok(())
}

#[derive(Debug, Default)]
struct Dwt97WeightRows {
    sample_len: Option<usize>,
    low: Vec<Vec<f64>>,
    high: Vec<Vec<f64>>,
}

impl Dwt97WeightRows {
    fn ensure_sample_len(&mut self, sample_len: usize) {
        if self.sample_len == Some(sample_len) {
            return;
        }

        resize_weight_rows(&mut self.low, low_len(sample_len), sample_len);
        resize_weight_rows(&mut self.high, high_len(sample_len), sample_len);

        for sample_idx in 0..sample_len {
            let mut basis = vec![0.0; sample_len];
            basis[sample_idx] = 1.0;
            let transformed = linearized_97_from_sample_slice(&basis);
            for (row, &weight) in self.low.iter_mut().zip(transformed.low.iter()) {
                row[sample_idx] = weight;
            }
            for (row, &weight) in self.high.iter_mut().zip(transformed.high.iter()) {
                row[sample_idx] = weight;
            }
        }

        self.sample_len = Some(sample_len);
    }

    fn weight_capacity(&self) -> usize {
        self.low.iter().map(Vec::capacity).sum::<usize>()
            + self.high.iter().map(Vec::capacity).sum::<usize>()
    }
}

fn resize_weight_rows(rows: &mut Vec<Vec<f64>>, row_count: usize, sample_len: usize) {
    if rows.len() < row_count {
        rows.resize_with(row_count, Vec::new);
    }
    for row in rows.iter_mut().take(row_count) {
        row.clear();
        row.resize(sample_len, 0.0);
    }
    rows.truncate(row_count);
}

struct Dwt97OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}
