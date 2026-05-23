// SPDX-License-Identifier: Apache-2.0

//! Scalar-derived 9/7 projection weight rows for Metal kernels.

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
        data[sample_idx] += ALPHA * (left + right);
    }

    for sample_idx in (1..sample_count).step_by(2) {
        let left = data[sample_idx - 1];
        let right = if sample_idx + 1 < sample_count {
            data[sample_idx + 1]
        } else {
            data[sample_idx - 1]
        };
        data[sample_idx] += BETA * (left + right);
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
        data[sample_idx] += GAMMA * (left + right);
    }

    for sample_idx in (1..sample_count).step_by(2) {
        let left = data[sample_idx - 1];
        let right = if sample_idx + 1 < sample_count {
            data[sample_idx + 1]
        } else {
            data[sample_idx - 1]
        };
        data[sample_idx] += DELTA * (left + right);
    }

    for sample_idx in (0..sample_count).step_by(2) {
        data[sample_idx] *= KAPPA;
    }
    for sample_idx in (1..sample_count).step_by(2) {
        data[sample_idx] *= INV_KAPPA;
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
