// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{try_host_vec_filled, try_host_vec_with_capacity};

use super::{
    budget::validate_dense_weight_budget,
    error::allocation_error,
    shared::{high_len, low_len, WaveletKind},
    transform::try_linearized_from_sample_slice,
    SparseWeightRowsError,
};

type DenseWeightTable = Vec<Vec<f32>>;

/// One-dimensional 9/7 projection weights for every output row.
#[derive(Debug, Clone, PartialEq)]
pub struct Dwt97WeightRows {
    /// Low-pass output rows, each indexed by input sample position.
    pub low: Vec<Vec<f32>>,
    /// High-pass output rows, each indexed by input sample position.
    pub high: Vec<Vec<f32>>,
}

impl Dwt97WeightRows {
    /// Build deterministic 9/7 projection rows for a one-dimensional sample extent.
    pub fn for_len(sample_len: usize) -> Result<Self, SparseWeightRowsError> {
        let (low, high) = dense_rows_for_len(sample_len, WaveletKind::Irreversible97)?;
        Ok(Self { low, high })
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
    /// Build deterministic 5/3 projection rows for a one-dimensional sample extent.
    pub fn for_len(sample_len: usize) -> Result<Self, SparseWeightRowsError> {
        let (low, high) = dense_rows_for_len(sample_len, WaveletKind::Reversible53)?;
        Ok(Self { low, high })
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "Metal projection tables intentionally store scalar f64 weights in the f32 shader ABI"
)]
fn dense_rows_for_len(
    sample_len: usize,
    wavelet: WaveletKind,
) -> Result<(DenseWeightTable, DenseWeightTable), SparseWeightRowsError> {
    validate_dense_weight_budget(sample_len)?;
    let mut low = try_dense_rows(low_len(sample_len), sample_len)?;
    let mut high = try_dense_rows(high_len(sample_len), sample_len)?;
    let mut basis = try_host_vec_filled(sample_len, 0.0).map_err(allocation_error)?;

    for sample_idx in 0..sample_len {
        basis[sample_idx] = 1.0;
        let transformed = try_linearized_from_sample_slice(&basis, wavelet)?;
        for (row, &weight) in low.iter_mut().zip(&transformed.low) {
            row[sample_idx] = weight as f32;
        }
        for (row, &weight) in high.iter_mut().zip(&transformed.high) {
            row[sample_idx] = weight as f32;
        }
        basis[sample_idx] = 0.0;
    }

    Ok((low, high))
}

fn try_dense_rows(
    row_count: usize,
    sample_len: usize,
) -> Result<DenseWeightTable, SparseWeightRowsError> {
    let mut rows = try_host_vec_with_capacity(row_count).map_err(allocation_error)?;
    for _ in 0..row_count {
        rows.push(try_host_vec_filled(sample_len, 0.0).map_err(allocation_error)?);
    }
    Ok(rows)
}
