// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA codec decode followed by an explicit staged Burn upload.

mod batch;

pub use batch::{CudaUploadBurnDecoder, SubmittedCudaUploadBurnBatch};

/// Compatibility alias for [`CudaUploadBurnDecoder`].
///
/// This alias has the same explicitly staged behavior; it does not restore the
/// former direct-destination implementation.
#[deprecated(
    since = "0.7.6",
    note = "use CudaUploadBurnDecoder to make staging explicit"
)]
pub type CudaBurnDecoder = CudaUploadBurnDecoder;

/// Compatibility alias for [`SubmittedCudaUploadBurnBatch`].
#[deprecated(
    since = "0.7.6",
    note = "use SubmittedCudaUploadBurnBatch to make staging explicit"
)]
pub type SubmittedCudaBurnBatch = SubmittedCudaUploadBurnBatch;
