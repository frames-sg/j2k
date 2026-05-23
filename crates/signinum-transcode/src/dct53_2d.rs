// SPDX-License-Identifier: Apache-2.0

//! Constrained 2D DCT to 5/3 wavelet experiments.
//!
//! The direct float path projects an 8x8 DCT block into one separable
//! single-level 5/3 result without first storing the 8x8 spatial samples. The
//! reference path materializes samples to keep the oracle easy to audit.

use core::f64::consts::PI;

/// One separable single-level 2D 5/3 transform result.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt53TwoDimensional<T> {
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

impl Dwt53TwoDimensional<f64> {
    /// Maximum absolute coefficient difference across matching bands.
    #[must_use]
    pub fn max_abs_diff(&self, other: &Self) -> f64 {
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

/// Map one 8x8 DCT block directly into a linearized one-level 2D 5/3 result.
#[must_use]
pub fn dct8x8_to_dwt53_float_linear(block: [[f64; 8]; 8]) -> Dwt53TwoDimensional<f64> {
    let width = 8;
    let height = 8;
    let low_width = low_len(width);
    let low_height = low_len(height);
    let high_width = high_len(width);
    let high_height = high_len(height);

    let mut ll = Vec::with_capacity(low_width * low_height);
    let mut hl = Vec::with_capacity(high_width * low_height);
    let mut lh = Vec::with_capacity(low_width * high_height);
    let mut hh = Vec::with_capacity(high_width * high_height);

    for y in 0..low_height {
        for x in 0..low_width {
            ll.push(project_dct_block(&block, true, y, true, x));
        }
        for x in 0..high_width {
            hl.push(project_dct_block(&block, true, y, false, x));
        }
    }

    for y in 0..high_height {
        for x in 0..low_width {
            lh.push(project_dct_block(&block, false, y, true, x));
        }
        for x in 0..high_width {
            hh.push(project_dct_block(&block, false, y, false, x));
        }
    }

    Dwt53TwoDimensional {
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

/// Reference path for the 2D experiment:
/// DCT coefficients -> float IDCT samples -> separable linearized 5/3.
#[must_use]
pub fn idct8x8_then_dwt53_float(block: [[f64; 8]; 8]) -> Dwt53TwoDimensional<f64> {
    let mut samples = Vec::with_capacity(64);
    for y in 0..8 {
        for x in 0..8 {
            samples.push(idct8x8_sample(&block, x, y));
        }
    }

    linearized_53_2d_from_plane(&samples, 8, 8)
}

fn project_dct_block(
    block: &[[f64; 8]; 8],
    vertical_low: bool,
    output_y: usize,
    horizontal_low: bool,
    output_x: usize,
) -> f64 {
    let mut output = 0.0;

    for sample_y in 0..8 {
        let y_weight = linearized_53_sample_weight(8, vertical_low, output_y, sample_y);
        if y_weight == 0.0 {
            continue;
        }

        for sample_x in 0..8 {
            let x_weight = linearized_53_sample_weight(8, horizontal_low, output_x, sample_x);
            if x_weight == 0.0 {
                continue;
            }

            let sample_weight = y_weight * x_weight;
            for (freq_y, coefficient_row) in block.iter().enumerate() {
                let y_basis = idct8_basis(sample_y, freq_y);
                for (freq_x, coefficient) in coefficient_row.iter().copied().enumerate() {
                    output += sample_weight * y_basis * idct8_basis(sample_x, freq_x) * coefficient;
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

fn linearized_53_sample_weight(
    sample_len: usize,
    is_low: bool,
    output_idx: usize,
    sample_idx: usize,
) -> f64 {
    let mut basis = vec![0.0; sample_len];
    basis[sample_idx] = 1.0;
    let row = linearized_53_from_sample_slice(&basis);
    if is_low {
        row.low[output_idx]
    } else {
        row.high[output_idx]
    }
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

struct Dwt53OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}
