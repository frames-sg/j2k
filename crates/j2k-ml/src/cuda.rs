// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA codec decode into staged or Burn-owned device storage.

mod batch;
mod direct;

pub use batch::{CudaBurnDecoder, SubmittedCudaBurnBatch};
pub use direct::CudaDirectBurnDecoder;

/// Explicit name for [`CudaBurnDecoder`]'s staged host-memory upload behavior.
///
/// This alias has the same explicitly staged behavior; it does not restore the
/// former direct-destination implementation.
pub type CudaUploadBurnDecoder = CudaBurnDecoder;

/// Explicit name for the staged upload represented by [`SubmittedCudaBurnBatch`].
pub type SubmittedCudaUploadBurnBatch = SubmittedCudaBurnBatch;
