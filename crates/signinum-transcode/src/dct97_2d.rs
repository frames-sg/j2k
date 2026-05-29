// SPDX-License-Identifier: Apache-2.0

//! Constrained 2D DCT to irreversible 9/7 wavelet transforms.
//!
//! The production float path performs a separable 8x8 IDCT into a reusable
//! spatial plane, then applies the separable single-level 9/7 transform.

use core::f64::consts::PI;
use core::fmt;
use std::sync::LazyLock;

use rayon::prelude::*;

const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
const KAPPA: f64 = 1.230_174_104_914_001;
const INV_KAPPA: f64 = 1.0 / KAPPA;
const PARALLEL_IDCT_MIN_SAMPLES: usize = 64 * 64;

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
#[non_exhaustive]
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

/// Scratch storage for repeated DCT-grid to 9/7 transform calls.
#[derive(Debug, Default)]
pub struct Dct97GridScratch {
    spatial_samples: Vec<f64>,
    plane: Dwt97PlaneScratch,
}

#[derive(Debug, Default)]
struct Dwt97PlaneScratch {
    row_low: Vec<f64>,
    row_high: Vec<f64>,
    lift_workspace: Vec<f64>,
}

impl Dct97GridScratch {
    /// Capacity of the reusable spatial-sample buffer used by the IDCT-then
    /// 9/7 path.
    #[must_use]
    pub fn spatial_sample_capacity(&self) -> usize {
        self.spatial_samples.capacity()
    }
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

/// Reference 9/7 path with caller-owned spatial-sample scratch:
/// DCT coefficients -> float IDCT samples -> separable linearized 9/7.
pub fn dct8x8_blocks_then_dwt97_float_with_scratch(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    scratch: &mut Dct97GridScratch,
) -> Result<Dwt97TwoDimensional<f64>, Dct97GridError> {
    validate_grid(blocks.len(), block_cols, block_rows, width, height)?;

    let sample_count = width.saturating_mul(height);
    scratch.spatial_samples.clear();
    scratch.spatial_samples.resize(sample_count, 0.0);
    idct8x8_blocks_to_samples(
        blocks,
        block_cols,
        width,
        height,
        &mut scratch.spatial_samples,
    );

    Ok(linearized_97_2d_from_plane_with_plane_scratch(
        &scratch.spatial_samples,
        width,
        height,
        &mut scratch.plane,
    ))
}

pub(crate) fn linearized_97_2d_from_plane(
    samples: &[f64],
    width: usize,
    height: usize,
) -> Dwt97TwoDimensional<f64> {
    let mut scratch = Dct97GridScratch::default();
    linearized_97_2d_from_plane_with_scratch(samples, width, height, &mut scratch)
}

pub(crate) fn linearized_97_2d_from_plane_with_scratch(
    samples: &[f64],
    width: usize,
    height: usize,
    scratch: &mut Dct97GridScratch,
) -> Dwt97TwoDimensional<f64> {
    linearized_97_2d_from_plane_with_plane_scratch(samples, width, height, &mut scratch.plane)
}

fn linearized_97_2d_from_plane_with_plane_scratch(
    samples: &[f64],
    width: usize,
    height: usize,
    scratch: &mut Dwt97PlaneScratch,
) -> Dwt97TwoDimensional<f64> {
    debug_assert_eq!(samples.len(), width * height);

    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);

    scratch.row_low.clear();
    scratch.row_low.resize(height * low_width, 0.0);
    scratch.row_high.clear();
    scratch.row_high.resize(height * high_width, 0.0);

    for y in 0..height {
        let start = y * width;
        let row = &samples[start..start + width];
        let low_start = y * low_width;
        let high_start = y * high_width;
        linearized_97_split_contiguous_into(
            row,
            &mut scratch.row_low[low_start..low_start + low_width],
            &mut scratch.row_high[high_start..high_start + high_width],
            &mut scratch.lift_workspace,
        );
    }

    let mut ll = vec![0.0; low_width * low_height];
    let mut lh = vec![0.0; low_width * high_height];
    for x in 0..low_width {
        linearized_97_split_strided_into(
            &scratch.row_low,
            low_width,
            x,
            height,
            &mut ll,
            &mut lh,
            low_width,
            &mut scratch.lift_workspace,
        );
    }

    let mut hl = vec![0.0; high_width * low_height];
    let mut hh = vec![0.0; high_width * high_height];
    for x in 0..high_width {
        linearized_97_split_strided_into(
            &scratch.row_high,
            high_width,
            x,
            height,
            &mut hl,
            &mut hh,
            high_width,
            &mut scratch.lift_workspace,
        );
    }

    Dwt97TwoDimensional {
        ll,
        hl,
        lh,
        hh,
        low_width,
        low_height,
        high_width,
        high_height,
    }
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

fn idct8x8_blocks_to_samples(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    width: usize,
    height: usize,
    samples: &mut [f64],
) {
    debug_assert_eq!(samples.len(), width * height);
    let basis = idct8_basis_table();
    let active_block_cols = width.div_ceil(8);
    let active_block_rows = height.div_ceil(8);

    if width * height >= PARALLEL_IDCT_MIN_SAMPLES {
        samples
            .par_chunks_mut(width * 8)
            .enumerate()
            .take(active_block_rows)
            .for_each(|(block_y, sample_rows)| {
                idct8x8_block_row_to_samples(
                    blocks,
                    block_cols,
                    width,
                    height,
                    basis,
                    active_block_cols,
                    block_y,
                    sample_rows,
                );
            });
    } else {
        for block_y in 0..active_block_rows {
            let block_sample_y = block_y * 8;
            let output_rows = (height - block_sample_y).min(8);
            let row_start = block_sample_y * width;
            let row_end = row_start + output_rows * width;
            idct8x8_block_row_to_samples(
                blocks,
                block_cols,
                width,
                height,
                basis,
                active_block_cols,
                block_y,
                &mut samples[row_start..row_end],
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn idct8x8_block_row_to_samples(
    blocks: &[[[f64; 8]; 8]],
    block_cols: usize,
    width: usize,
    height: usize,
    basis: &[[f64; 8]; 8],
    active_block_cols: usize,
    block_y: usize,
    sample_rows: &mut [f64],
) {
    let block_sample_y = block_y * 8;
    let output_rows = (height - block_sample_y).min(8);
    for block_x in 0..active_block_cols {
        let block_sample_x = block_x * 8;
        let output_cols = (width - block_sample_x).min(8);
        let block = &blocks[block_y * block_cols + block_x];
        let mut vertical = [[0.0; 8]; 8];

        for (local_y, basis_row) in basis.iter().enumerate() {
            for freq_x in 0..8 {
                let mut sum = 0.0;
                for (freq_y, block_row) in block.iter().enumerate() {
                    sum += basis_row[freq_y] * block_row[freq_x];
                }
                vertical[local_y][freq_x] = sum;
            }
        }

        for (local_y, vertical_row) in vertical.iter().enumerate().take(output_rows) {
            let row_offset = local_y * width + block_sample_x;
            for local_x in 0..output_cols {
                let mut sample = 0.0;
                for (freq_x, vertical_value) in vertical_row.iter().enumerate() {
                    sample += *vertical_value * basis[local_x][freq_x];
                }
                sample_rows[row_offset + local_x] = sample;
            }
        }
    }
}

#[cfg(test)]
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

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };

    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += ALPHA * (left + right);
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += BETA * (left + right);
    }

    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += GAMMA * (left + right);
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += DELTA * (left + right);
    }

    for i in (0..n).step_by(2) {
        data[i] *= INV_KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= KAPPA;
    }
}

fn linearized_97_split_contiguous_into(
    samples: &[f64],
    low: &mut [f64],
    high: &mut [f64],
    workspace: &mut Vec<f64>,
) {
    debug_assert_eq!(low.len(), low_len(samples.len()));
    debug_assert_eq!(high.len(), high_len(samples.len()));

    workspace.clear();
    workspace.extend_from_slice(samples);
    forward_lift_97(workspace);

    for (target, value) in low.iter_mut().zip(workspace.iter().step_by(2)) {
        *target = *value;
    }
    for (target, value) in high.iter_mut().zip(workspace.iter().skip(1).step_by(2)) {
        *target = *value;
    }
}

#[allow(clippy::too_many_arguments)]
fn linearized_97_split_strided_into(
    samples: &[f64],
    stride: usize,
    x: usize,
    height: usize,
    low: &mut [f64],
    high: &mut [f64],
    band_width: usize,
    workspace: &mut Vec<f64>,
) {
    debug_assert_eq!(low.len(), band_width * low_len(height));
    debug_assert_eq!(high.len(), band_width * high_len(height));

    workspace.clear();
    workspace.extend((0..height).map(|y| samples[y * stride + x]));
    forward_lift_97(workspace);

    for (low_y, value) in workspace.iter().step_by(2).enumerate() {
        low[low_y * band_width + x] = *value;
    }
    for (high_y, value) in workspace.iter().skip(1).step_by(2).enumerate() {
        high[high_y * band_width + x] = *value;
    }
}

fn idct8_basis(sample_idx: usize, freq: usize) -> f64 {
    debug_assert!(sample_idx < 8);
    debug_assert!(freq < 8);

    idct8_basis_table()[sample_idx][freq]
}

fn idct8_basis_table() -> &'static [[f64; 8]; 8] {
    static BASIS: LazyLock<[[f64; 8]; 8]> = LazyLock::new(|| {
        let mut basis = [[0.0; 8]; 8];
        for (sample_idx, row) in basis.iter_mut().enumerate() {
            for (freq, value) in row.iter_mut().enumerate() {
                *value = idct8_basis_uncached(sample_idx, freq);
            }
        }
        basis
    });
    &BASIS
}

fn idct8_basis_uncached(sample_idx: usize, freq: usize) -> f64 {
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

#[cfg(test)]
struct Dwt97OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_all_close(values: &[f64], expected: f64, epsilon: f64) {
        for &value in values {
            assert!(
                (value - expected).abs() < epsilon,
                "value={value} expected={expected} values={values:?}"
            );
        }
    }

    #[test]
    fn linearized_97_from_constant_signal_places_dc_in_low_pass() {
        for len in [2usize, 3, 8, 9, 64, 65] {
            let samples = vec![50.0; len];

            let transformed = linearized_97_from_sample_slice(&samples);

            assert_all_close(&transformed.low, 50.0, 0.001);
            assert_all_close(&transformed.high, 0.0, 0.001);
        }
    }

    #[test]
    fn linearized_97_2d_from_constant_plane_places_dc_in_ll() {
        for (width, height) in [(8usize, 8usize), (9, 7)] {
            let samples = vec![50.0; width * height];

            let transformed = linearized_97_2d_from_plane(&samples, width, height);

            assert_all_close(&transformed.ll, 50.0, 0.001);
            assert_all_close(&transformed.hl, 0.0, 0.001);
            assert_all_close(&transformed.lh, 0.0, 0.001);
            assert_all_close(&transformed.hh, 0.0, 0.001);
        }
    }
}
