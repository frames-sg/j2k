// SPDX-License-Identifier: Apache-2.0

//! Scalar-derived wavelet projection weight rows for Metal kernels.

const ALPHA: f64 = -1.586_134_342_059_924;
const BETA: f64 = -0.052_980_118_572_961;
const GAMMA: f64 = 0.882_911_075_530_934;
const DELTA: f64 = 0.443_506_852_043_971;
const KAPPA: f64 = 1.230_174_104_914_001;
const INV_KAPPA: f64 = 1.0 / KAPPA;

/// One-dimensional 9/7 projection weights for every output row.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt97WeightRows {
    /// Low-pass output rows, each indexed by input sample position.
    pub low: Vec<Vec<f32>>,
    /// High-pass output rows, each indexed by input sample position.
    pub high: Vec<Vec<f32>>,
}

impl Dwt97WeightRows {
    /// Build deterministic 9/7 projection rows for a one-dimensional sample
    /// extent.
    #[must_use]
    pub fn for_len(sample_len: usize) -> Self {
        let mut low = vec![vec![0.0; sample_len]; low_len(sample_len)];
        let mut high = vec![vec![0.0; sample_len]; high_len(sample_len)];

        for sample_idx in 0..sample_len {
            let mut basis = vec![0.0; sample_len];
            basis[sample_idx] = 1.0;
            let transformed = linearized_97_from_sample_slice(&basis);

            for (row, &weight) in low.iter_mut().zip(transformed.low.iter()) {
                row[sample_idx] = weight as f32;
            }
            for (row, &weight) in high.iter_mut().zip(transformed.high.iter()) {
                row[sample_idx] = weight as f32;
            }
        }

        Self { low, high }
    }
}

/// One-dimensional 5/3 projection weights for every output row.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt53WeightRows {
    /// Low-pass output rows, each indexed by input sample position.
    pub low: Vec<Vec<f32>>,
    /// High-pass output rows, each indexed by input sample position.
    pub high: Vec<Vec<f32>>,
}

impl Dwt53WeightRows {
    /// Build deterministic 5/3 projection rows for a one-dimensional sample
    /// extent.
    #[must_use]
    pub fn for_len(sample_len: usize) -> Self {
        let mut low = vec![vec![0.0; sample_len]; low_len(sample_len)];
        let mut high = vec![vec![0.0; sample_len]; high_len(sample_len)];

        for sample_idx in 0..sample_len {
            let mut basis = vec![0.0; sample_len];
            basis[sample_idx] = 1.0;
            let transformed = linearized_53_from_sample_slice(&basis);

            for (row, &weight) in low.iter_mut().zip(transformed.low.iter()) {
                row[sample_idx] = weight as f32;
            }
            for (row, &weight) in high.iter_mut().zip(transformed.high.iter()) {
                row[sample_idx] = weight as f32;
            }
        }

        Self { low, high }
    }
}

/// Sparse one-dimensional 9/7 projection rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseDwt97WeightRows {
    /// Low-pass sparse output rows.
    pub low: Vec<SparseWeightRow>,
    /// High-pass sparse output rows.
    pub high: Vec<SparseWeightRow>,
}

impl SparseDwt97WeightRows {
    /// Build sparse 9/7 projection rows for a one-dimensional sample extent.
    #[must_use]
    pub fn for_len(sample_len: usize) -> Self {
        let dense = Dwt97WeightRows::for_len(sample_len);
        Self {
            low: sparse_rows_from_dense(&dense.low),
            high: sparse_rows_from_dense(&dense.high),
        }
    }

    /// Largest tap count across low-pass and high-pass rows.
    #[must_use]
    pub fn max_taps_per_row(&self) -> usize {
        self.low
            .iter()
            .chain(self.high.iter())
            .map(|row| row.taps.len())
            .max()
            .unwrap_or(0)
    }
}

/// Sparse one-dimensional 5/3 projection rows.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseDwt53WeightRows {
    /// Low-pass sparse output rows.
    pub low: Vec<SparseWeightRow>,
    /// High-pass sparse output rows.
    pub high: Vec<SparseWeightRow>,
}

impl SparseDwt53WeightRows {
    /// Build sparse 5/3 projection rows for a one-dimensional sample extent.
    #[must_use]
    pub fn for_len(sample_len: usize) -> Self {
        let dense = Dwt53WeightRows::for_len(sample_len);
        Self {
            low: sparse_rows_from_dense(&dense.low),
            high: sparse_rows_from_dense(&dense.high),
        }
    }

    /// Largest tap count across low-pass and high-pass rows.
    #[must_use]
    pub fn max_taps_per_row(&self) -> usize {
        self.low
            .iter()
            .chain(self.high.iter())
            .map(|row| row.taps.len())
            .max()
            .unwrap_or(0)
    }
}

/// Sparse row of sample-position weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseWeightRow {
    /// Nonzero taps in sample-index order.
    pub taps: Vec<SparseWeightTap>,
}

/// One nonzero sample-position weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseWeightTap {
    /// Input sample index.
    pub sample_idx: usize,
    /// Weight applied to that sample.
    pub weight: f32,
}

fn sparse_rows_from_dense(rows: &[Vec<f32>]) -> Vec<SparseWeightRow> {
    rows.iter()
        .map(|row| SparseWeightRow {
            taps: row
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, weight)| weight.to_bits() != 0)
                .map(|(sample_idx, weight)| SparseWeightTap { sample_idx, weight })
                .collect(),
        })
        .collect()
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

fn linearized_97_from_sample_slice(samples: &[f64]) -> Dwt97OneDimensional {
    let mut lifted = samples.to_vec();
    forward_lift_97(&mut lifted);

    Dwt97OneDimensional {
        low: lifted.iter().step_by(2).copied().collect(),
        high: lifted.iter().skip(1).step_by(2).copied().collect(),
    }
}

fn forward_lift_97(data: &mut [f64]) {
    let sample_count = data.len();
    if sample_count < 2 {
        return;
    }

    let last_even = if sample_count.is_multiple_of(2) {
        sample_count - 2
    } else {
        sample_count - 1
    };

    for sample_idx in (1..sample_count).step_by(2) {
        let left = data[sample_idx - 1];
        let right = if sample_idx + 1 < sample_count {
            data[sample_idx + 1]
        } else {
            data[last_even]
        };
        data[sample_idx] += ALPHA * (left + right);
    }

    for sample_idx in (0..sample_count).step_by(2) {
        let left = if sample_idx > 0 {
            data[sample_idx - 1]
        } else {
            data[1]
        };
        let right = if sample_idx + 1 < sample_count {
            data[sample_idx + 1]
        } else {
            left
        };
        data[sample_idx] += BETA * (left + right);
    }

    for sample_idx in (1..sample_count).step_by(2) {
        let left = data[sample_idx - 1];
        let right = if sample_idx + 1 < sample_count {
            data[sample_idx + 1]
        } else {
            data[last_even]
        };
        data[sample_idx] += GAMMA * (left + right);
    }

    for sample_idx in (0..sample_count).step_by(2) {
        let left = if sample_idx > 0 {
            data[sample_idx - 1]
        } else {
            data[1]
        };
        let right = if sample_idx + 1 < sample_count {
            data[sample_idx + 1]
        } else {
            left
        };
        data[sample_idx] += DELTA * (left + right);
    }

    for sample_idx in (0..sample_count).step_by(2) {
        data[sample_idx] *= INV_KAPPA;
    }
    for sample_idx in (1..sample_count).step_by(2) {
        data[sample_idx] *= KAPPA;
    }
}

const fn low_len(sample_len: usize) -> usize {
    sample_len.div_ceil(2)
}

const fn high_len(sample_len: usize) -> usize {
    sample_len / 2
}

struct Dwt97OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}

struct Dwt53OneDimensional {
    low: Vec<f64>,
    high: Vec<f64>,
}
