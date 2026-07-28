// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA codec decode followed by an explicit staged Burn upload.

mod batch;

pub use batch::{CudaBurnDecoder, SubmittedCudaBurnBatch};

/// Explicit name for [`CudaBurnDecoder`]'s staged host-memory upload behavior.
///
/// This alias has the same explicitly staged behavior; it does not restore the
/// former direct-destination implementation.
pub type CudaUploadBurnDecoder = CudaBurnDecoder;

/// Explicit name for the staged upload represented by [`SubmittedCudaBurnBatch`].
pub type SubmittedCudaUploadBurnBatch = SubmittedCudaBurnBatch;
