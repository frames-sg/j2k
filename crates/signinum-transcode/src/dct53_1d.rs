// SPDX-License-Identifier: Apache-2.0

//! Constrained 1D DCT to 5/3 wavelet experiments.
//!
//! This module intentionally works on one synthetic 8-coefficient DCT block and
//! one single-level 5/3 transform. The float path is a linear composition of
//! the inverse DCT basis with a linearized 5/3 analysis step. The reversible
//! path is bit-exact against the rounded-IDCT reference, but it is piecewise
//! integer arithmetic rather than a single linear matrix.

use core::f64::consts::PI;
use core::fmt;

/// One single-level 5/3 transform result for an 8-sample 1D signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dwt53OneLevel<T> {
    /// Low-pass samples, corresponding to even positions after lifting.
    pub low: [T; 4],
    /// High-pass samples, corresponding to odd positions after lifting.
    pub high: [T; 4],
}

/// One single-level 5/3 transform result for an arbitrary-length 1D row.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt53Row<T> {
    /// Low-pass samples, corresponding to even positions after lifting.
    pub low: Vec<T>,
    /// High-pass samples, corresponding to odd positions after lifting.
    pub high: Vec<T>,
}

/// Error returned when a logical row length cannot be covered by DCT blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dct53RowLengthError {
    sample_len: usize,
    capacity: usize,
}

impl Dct53RowLengthError {
    /// Requested logical sample length.
    #[must_use]
    pub const fn sample_len(self) -> usize {
        self.sample_len
    }

    /// Number of samples covered by the provided 8-point DCT blocks.
    #[must_use]
    pub const fn capacity(self) -> usize {
        self.capacity
    }
}

impl fmt::Display for Dct53RowLengthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "row length {} exceeds DCT block sample capacity {}",
            self.sample_len, self.capacity
        )
    }
}

impl std::error::Error for Dct53RowLengthError {}

/// Map one 8-point DCT coefficient vector directly into a linearized one-level
/// 5/3 wavelet result.
///
/// This proves the linear composition:
///
/// `DWT53_linear * IDCT8 * dct_coefficients`
#[must_use]
pub fn dct8_to_dwt53_float_linear(coefficients: [f64; 8]) -> Dwt53OneLevel<f64> {
    let rows = linearized_53_rows();
    let mut low = [0.0; 4];
    let mut high = [0.0; 4];

    for (dst, row) in low.iter_mut().zip(rows[..4].iter()) {
        *dst = dct_row_projection(row, &coefficients);
    }
    for (dst, row) in high.iter_mut().zip(rows[4..].iter()) {
        *dst = dct_row_projection(row, &coefficients);
    }

    Dwt53OneLevel { low, high }
}

/// Reference path for the linearized 1D experiment:
/// DCT coefficients -> float IDCT samples -> linearized 5/3.
#[must_use]
pub fn idct8_then_dwt53_float(coefficients: [f64; 8]) -> Dwt53OneLevel<f64> {
    let mut samples = [0.0; 8];
    for (idx, sample) in samples.iter_mut().enumerate() {
        *sample = idct8_sample(&coefficients, idx);
    }
    linearized_53_from_samples(samples)
}

/// Map adjacent 8-point DCT blocks directly into a linearized one-level 5/3
/// wavelet row.
///
/// This keeps the production direction honest: output coefficients are
/// projected from the DCT basis without first creating a row of spatial samples.
#[must_use]
pub fn dct8_blocks_to_dwt53_float_linear(blocks: &[[f64; 8]]) -> Dwt53Row<f64> {
    dct8_blocks_to_dwt53_float_linear_with_len(blocks, blocks.len() * 8)
        .expect("full row length is always covered by the provided blocks")
}

/// Map adjacent 8-point DCT blocks directly into a linearized one-level 5/3
/// wavelet row using a logical sample length.
///
/// Use this for image component rows whose JPEG block storage includes padded
/// samples beyond the real component width.
pub fn dct8_blocks_to_dwt53_float_linear_with_len(
    blocks: &[[f64; 8]],
    sample_len: usize,
) -> Result<Dwt53Row<f64>, Dct53RowLengthError> {
    validate_sample_len(blocks, sample_len)?;

    let low_len = low_len(sample_len);
    let high_len = high_len(sample_len);
    let mut low = Vec::with_capacity(low_len);
    let mut high = Vec::with_capacity(high_len);

    for output_idx in 0..low_len {
        low.push(project_blocks_with_linearized_53_weights(
            blocks, sample_len, true, output_idx,
        ));
    }
    for output_idx in 0..high_len {
        high.push(project_blocks_with_linearized_53_weights(
            blocks, sample_len, false, output_idx,
        ));
    }

    Ok(Dwt53Row { low, high })
}

/// Reference path for an arbitrary-length row:
/// DCT coefficients -> float IDCT samples -> linearized 5/3.
#[must_use]
pub fn idct8_blocks_then_dwt53_float(blocks: &[[f64; 8]]) -> Dwt53Row<f64> {
    idct8_blocks_then_dwt53_float_with_len(blocks, blocks.len() * 8)
        .expect("full row length is always covered by the provided blocks")
}

/// Reference path for a logical row length:
/// DCT coefficients -> float IDCT samples -> linearized 5/3.
pub fn idct8_blocks_then_dwt53_float_with_len(
    blocks: &[[f64; 8]],
    sample_len: usize,
) -> Result<Dwt53Row<f64>, Dct53RowLengthError> {
    validate_sample_len(blocks, sample_len)?;

    let mut samples = Vec::with_capacity(sample_len);
    for sample_idx in 0..sample_len {
        let block = &blocks[sample_idx / 8];
        samples.push(idct8_sample(block, sample_idx % 8));
    }

    Ok(linearized_53_from_sample_slice(&samples))
}

/// Map one 8-point integer DCT coefficient vector into one reversible 5/3
/// wavelet result after rounded IDCT sample evaluation.
///
/// This path is not a linear matrix. It keeps the integer rounding points that
/// the reversible 5/3 path requires, while avoiding a reusable spatial-domain
/// image buffer.
#[must_use]
pub fn dct8_to_dwt53_reversible_i16(coefficients: [i16; 8]) -> Dwt53OneLevel<i32> {
    let x0 = rounded_idct8_sample(&coefficients, 0);
    let x1 = rounded_idct8_sample(&coefficients, 1);
    let x2 = rounded_idct8_sample(&coefficients, 2);
    let x3 = rounded_idct8_sample(&coefficients, 3);
    let x4 = rounded_idct8_sample(&coefficients, 4);
    let x5 = rounded_idct8_sample(&coefficients, 5);
    let x6 = rounded_idct8_sample(&coefficients, 6);
    let x7 = rounded_idct8_sample(&coefficients, 7);

    let h0 = x1 - floor_div(x0 + x2, 2);
    let h1 = x3 - floor_div(x2 + x4, 2);
    let h2 = x5 - floor_div(x4 + x6, 2);
    let h3 = x7 - x6;

    let l0 = x0 + floor_div(h0 + 1, 2);
    let l1 = x2 + floor_div(h0 + h1 + 2, 4);
    let l2 = x4 + floor_div(h1 + h2 + 2, 4);
    let l3 = x6 + floor_div(h2 + h3 + 2, 4);

    Dwt53OneLevel {
        low: [l0, l1, l2, l3],
        high: [h0, h1, h2, h3],
    }
}

/// Reference path for the reversible 1D experiment:
/// DCT coefficients -> rounded IDCT samples -> reversible 5/3.
#[must_use]
pub fn idct8_rounded_then_dwt53_reversible(coefficients: [i16; 8]) -> Dwt53OneLevel<i32> {
    let mut samples = [0; 8];
    for (idx, sample) in samples.iter_mut().enumerate() {
        *sample = rounded_idct8_sample(&coefficients, idx);
    }
    reversible_53_from_samples(samples)
}

fn dct_row_projection(sample_weights: &[f64; 8], coefficients: &[f64; 8]) -> f64 {
    let mut coefficient_weights = [0.0; 8];
    for (sample_idx, sample_weight) in sample_weights.iter().copied().enumerate() {
        for (freq, coefficient_weight) in coefficient_weights.iter_mut().enumerate() {
            *coefficient_weight += sample_weight * idct8_basis(sample_idx, freq);
        }
    }

    coefficient_weights
        .iter()
        .zip(coefficients.iter())
        .map(|(weight, coefficient)| weight * coefficient)
        .sum()
}

fn project_blocks_with_linearized_53_weights(
    blocks: &[[f64; 8]],
    sample_len: usize,
    is_low: bool,
    output_idx: usize,
) -> f64 {
    let mut output = 0.0;

    for sample_idx in 0..sample_len {
        let sample_weight = linearized_53_sample_weight(sample_len, is_low, output_idx, sample_idx);
        if sample_weight == 0.0 {
            continue;
        }

        let block_idx = sample_idx / 8;
        let local_sample_idx = sample_idx % 8;
        let block = &blocks[block_idx];
        for (freq, coefficient) in block.iter().copied().enumerate() {
            output += sample_weight * idct8_basis(local_sample_idx, freq) * coefficient;
        }
    }

    output
}

fn idct8_sample(coefficients: &[f64; 8], sample_idx: usize) -> f64 {
    coefficients
        .iter()
        .enumerate()
        .map(|(freq, coefficient)| coefficient * idct8_basis(sample_idx, freq))
        .sum()
}

fn rounded_idct8_sample(coefficients: &[i16; 8], sample_idx: usize) -> i32 {
    let float_coefficients = coefficients.map(f64::from);
    idct8_sample(&float_coefficients, sample_idx).round() as i32
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

fn linearized_53_from_samples(samples: [f64; 8]) -> Dwt53OneLevel<f64> {
    let row = linearized_53_from_sample_slice(&samples);
    Dwt53OneLevel {
        low: row
            .low
            .try_into()
            .expect("8 samples produce exactly 4 low-pass outputs"),
        high: row
            .high
            .try_into()
            .expect("8 samples produce exactly 4 high-pass outputs"),
    }
}

fn linearized_53_from_sample_slice(samples: &[f64]) -> Dwt53Row<f64> {
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

    Dwt53Row { low, high }
}

fn reversible_53_from_samples(mut samples: [i32; 8]) -> Dwt53OneLevel<i32> {
    samples[1] -= floor_div(samples[0] + samples[2], 2);
    samples[3] -= floor_div(samples[2] + samples[4], 2);
    samples[5] -= floor_div(samples[4] + samples[6], 2);
    samples[7] -= samples[6];

    samples[0] += floor_div(samples[1] + 1, 2);
    samples[2] += floor_div(samples[1] + samples[3] + 2, 4);
    samples[4] += floor_div(samples[3] + samples[5] + 2, 4);
    samples[6] += floor_div(samples[5] + samples[7] + 2, 4);

    Dwt53OneLevel {
        low: [samples[0], samples[2], samples[4], samples[6]],
        high: [samples[1], samples[3], samples[5], samples[7]],
    }
}

fn floor_div(numerator: i32, denominator: i32) -> i32 {
    numerator.div_euclid(denominator)
}

fn validate_sample_len(blocks: &[[f64; 8]], sample_len: usize) -> Result<(), Dct53RowLengthError> {
    let capacity = blocks.len() * 8;
    if sample_len > capacity {
        return Err(Dct53RowLengthError {
            sample_len,
            capacity,
        });
    }

    Ok(())
}

fn low_len(sample_len: usize) -> usize {
    sample_len.div_ceil(2)
}

fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}

fn linearized_53_rows() -> [[f64; 8]; 8] {
    [
        [0.75, 0.5, -0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.125, 0.25, 0.75, 0.25, -0.125, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.125, 0.25, 0.75, 0.25, -0.125, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.125, 0.25, 0.625, 0.25],
        [-0.5, 1.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -0.5, 1.0, -0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, -0.5, 1.0, -0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
    ]
}
