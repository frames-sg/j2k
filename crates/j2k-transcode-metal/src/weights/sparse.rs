// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::try_host_vec_with_capacity;

#[cfg(target_os = "macos")]
use super::budget::metal_sparse_weight_budget;
use super::{
    budget::bounded_sparse_weight_budget,
    error::allocation_error,
    shared::{high_len, low_len, WaveletBand, WaveletKind},
    symbolic::write_symbolic_row,
    SparseWeightRowsError,
};

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
    pub fn for_len(sample_len: usize) -> Result<Self, SparseWeightRowsError> {
        let rows = sparse_rows_for_len(sample_len, WaveletKind::Irreversible97)?;
        Ok(Self {
            low: rows.low,
            high: rows.high,
        })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn allocation_bytes_for_len(
        sample_len: usize,
    ) -> Result<usize, SparseWeightRowsError> {
        sparse_allocation_bytes(sample_len, WaveletKind::Irreversible97)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal_bytes_for_len(sample_len: usize) -> Result<usize, SparseWeightRowsError> {
        sparse_metal_bytes(sample_len, WaveletKind::Irreversible97)
    }

    /// Largest tap count across low-pass and high-pass rows.
    #[must_use]
    pub fn max_taps_per_row(&self) -> usize {
        max_taps_per_row(&self.low, &self.high)
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
    pub fn for_len(sample_len: usize) -> Result<Self, SparseWeightRowsError> {
        let rows = sparse_rows_for_len(sample_len, WaveletKind::Reversible53)?;
        Ok(Self {
            low: rows.low,
            high: rows.high,
        })
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn allocation_bytes_for_len(
        sample_len: usize,
    ) -> Result<usize, SparseWeightRowsError> {
        sparse_allocation_bytes(sample_len, WaveletKind::Reversible53)
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn metal_bytes_for_len(sample_len: usize) -> Result<usize, SparseWeightRowsError> {
        sparse_metal_bytes(sample_len, WaveletKind::Reversible53)
    }

    /// Largest tap count across low-pass and high-pass rows.
    #[must_use]
    pub fn max_taps_per_row(&self) -> usize {
        max_taps_per_row(&self.low, &self.high)
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

struct SparseRows {
    low: Vec<SparseWeightRow>,
    high: Vec<SparseWeightRow>,
}

fn sparse_rows_for_len(
    sample_len: usize,
    wavelet: WaveletKind,
) -> Result<SparseRows, SparseWeightRowsError> {
    let max_taps = wavelet.max_taps_per_row();
    bounded_sparse_weight_budget(sample_len, max_taps)?;
    let mut low = empty_sparse_rows(low_len(sample_len), max_taps)?;
    let mut high = empty_sparse_rows(high_len(sample_len), max_taps)?;

    for (output_idx, row) in low.iter_mut().enumerate() {
        write_symbolic_row(row, sample_len, output_idx, WaveletBand::Low, wavelet)?;
    }
    for (output_idx, row) in high.iter_mut().enumerate() {
        write_symbolic_row(row, sample_len, output_idx, WaveletBand::High, wavelet)?;
    }

    Ok(SparseRows { low, high })
}

fn empty_sparse_rows(
    count: usize,
    max_taps_per_row: usize,
) -> Result<Vec<SparseWeightRow>, SparseWeightRowsError> {
    let mut rows = try_host_vec_with_capacity(count).map_err(allocation_error)?;
    for _ in 0..count {
        rows.push(SparseWeightRow {
            taps: try_host_vec_with_capacity(max_taps_per_row).map_err(allocation_error)?,
        });
    }
    Ok(rows)
}

#[cfg(target_os = "macos")]
fn sparse_allocation_bytes(
    sample_len: usize,
    wavelet: WaveletKind,
) -> Result<usize, SparseWeightRowsError> {
    bounded_sparse_weight_budget(sample_len, wavelet.max_taps_per_row())
}

#[cfg(target_os = "macos")]
fn sparse_metal_bytes(
    sample_len: usize,
    wavelet: WaveletKind,
) -> Result<usize, SparseWeightRowsError> {
    metal_sparse_weight_budget(sample_len, wavelet.max_taps_per_row())
}

fn max_taps_per_row(low: &[SparseWeightRow], high: &[SparseWeightRow]) -> usize {
    low.iter()
        .chain(high)
        .map(|row| row.taps.len())
        .max()
        .unwrap_or(0)
}
