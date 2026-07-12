// SPDX-License-Identifier: MIT OR Apache-2.0

//! Scalar-derived wavelet projection weight rows for Metal kernels.

mod budget;
mod dense;
mod error;
mod shared;
mod sparse;
mod symbolic;
mod transform;

pub use dense::{Dwt53WeightRows, Dwt97WeightRows};
pub use error::SparseWeightRowsError;
pub use sparse::{SparseDwt53WeightRows, SparseDwt97WeightRows, SparseWeightRow, SparseWeightTap};

#[cfg(test)]
mod tests;
