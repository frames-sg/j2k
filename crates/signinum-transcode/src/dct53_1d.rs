// SPDX-License-Identifier: Apache-2.0

//! Constrained 1D DCT to 5/3 wavelet experiments.
//!
//! This module intentionally works on one synthetic 8-coefficient DCT block and
//! one single-level 5/3 transform. The float path is a linear composition of
//! the inverse DCT basis with a linearized 5/3 analysis step. The reversible
//! path is bit-exact against the rounded-IDCT reference, but it is piecewise
//! integer arithmetic rather than a single linear matrix.

use core::f64::consts::PI;

/// One single-level 5/3 transform result for an 8-sample 1D signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dwt53OneLevel<T> {
    /// Low-pass samples, corresponding to even positions after lifting.
    pub low: [T; 4],
    /// High-pass samples, corresponding to odd positions after lifting.
    pub high: [T; 4],
}

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
    let rows = linearized_53_rows();
    let mut low = [0.0; 4];
    let mut high = [0.0; 4];

    for (dst, row) in low.iter_mut().zip(rows[..4].iter()) {
        *dst = row
            .iter()
            .zip(samples.iter())
            .map(|(weight, sample)| weight * sample)
            .sum();
    }
    for (dst, row) in high.iter_mut().zip(rows[4..].iter()) {
        *dst = row
            .iter()
            .zip(samples.iter())
            .map(|(weight, sample)| weight * sample)
            .sum();
    }

    Dwt53OneLevel { low, high }
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
