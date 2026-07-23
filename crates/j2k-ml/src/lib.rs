// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin Burn tensor adapter for the `j2k` owned batch codec.
//!
//! The codec crates own parsing, grouping, decoding, and accelerator execution.
//! This crate materializes CPU groups or stages completed CUDA and Metal codec
//! output through host memory before an ordinary Burn upload. Casting and
//! normalization stay in ordinary Burn tensor operations after decode.

#![deny(missing_docs)]

use burn_core::tensor::{backend::Backend, Int, Tensor};
use j2k::{BatchGroupInfo, IndexedBatchError, J2kDecodeWarning, Rect};

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    all(feature = "metal", target_os = "macos")
))]
mod batch_contract;
#[cfg(test)]
mod completion;
mod error;
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
mod staging;

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cpu")]
pub use cpu::CpuBurnDecoder;
#[cfg(feature = "cuda")]
pub use cuda::{CudaUploadBurnDecoder, SubmittedCudaUploadBurnBatch};
pub use error::BurnDecodeError;
#[cfg(feature = "metal")]
pub use metal::{MetalUploadBurnDecoder, SubmittedMetalUploadBurnBatch};

/// Compatibility alias for [`CudaUploadBurnDecoder`].
#[cfg(feature = "cuda")]
#[deprecated(
    since = "0.7.6",
    note = "use CudaUploadBurnDecoder to make staging explicit"
)]
pub type CudaBurnDecoder = CudaUploadBurnDecoder;

/// Compatibility alias for [`SubmittedCudaUploadBurnBatch`].
#[cfg(feature = "cuda")]
#[deprecated(
    since = "0.7.6",
    note = "use SubmittedCudaUploadBurnBatch to make staging explicit"
)]
pub type SubmittedCudaBurnBatch = SubmittedCudaUploadBurnBatch;

/// Compatibility alias for [`MetalUploadBurnDecoder`].
#[cfg(feature = "metal")]
#[deprecated(
    since = "0.7.6",
    note = "use MetalUploadBurnDecoder to make staging explicit"
)]
pub type MetalBurnDecoder = MetalUploadBurnDecoder;

/// Compatibility alias for [`SubmittedMetalUploadBurnBatch`].
#[cfg(feature = "metal")]
#[deprecated(
    since = "0.7.6",
    note = "use SubmittedMetalUploadBurnBatch to make staging explicit"
)]
pub type SubmittedMetalBurnBatch = SubmittedMetalUploadBurnBatch;

/// Ordinary rank-4 Burn integer tensor tagged with its exact codec sample type.
#[derive(Debug)]
pub enum BurnBatchTensor<B: Backend> {
    /// Unsigned samples with precision at most eight bits.
    U8(Tensor<B, 4, Int>),
    /// Unsigned samples with precision from nine through sixteen bits.
    U16(Tensor<B, 4, Int>),
    /// Signed samples with precision at most sixteen bits.
    I16(Tensor<B, 4, Int>),
}

impl<B: Backend> BurnBatchTensor<B> {
    /// Borrow the ordinary Burn integer tensor regardless of its codec dtype tag.
    #[must_use]
    pub const fn tensor(&self) -> &Tensor<B, 4, Int> {
        match self {
            Self::U8(tensor) | Self::U16(tensor) | Self::I16(tensor) => tensor,
        }
    }

    /// Consume the codec dtype tag and return the ordinary Burn integer tensor.
    #[must_use]
    pub fn into_tensor(self) -> Tensor<B, 4, Int> {
        match self {
            Self::U8(tensor) | Self::U16(tensor) | Self::I16(tensor) => tensor,
        }
    }
}

/// One homogeneous decoded tensor group and its codec metadata.
#[derive(Debug)]
pub struct BurnBatchGroup<B: Backend> {
    /// Decoded rank-4 integer tensor.
    pub tensor: BurnBatchTensor<B>,
    /// Exact codec and output metadata shared by the group.
    pub info: BatchGroupInfo,
    /// Original caller indices in tensor batch order.
    pub source_indices: Vec<usize>,
    /// Actual decoded source rectangle for each tensor item.
    pub decoded_rects: Vec<Rect>,
    /// Non-fatal codec warnings for each tensor item.
    pub warnings: Vec<Vec<J2kDecodeWarning>>,
}

/// Failure while submitting or completing one homogeneous Burn tensor group.
///
/// No partially written tensor from the affected group is exposed. Other
/// homogeneous groups may still succeed when the retained codec and framework
/// sessions remain usable.
#[derive(Debug, thiserror::Error)]
#[error("Burn batch group containing source indices {source_indices:?} failed: {source}")]
pub struct BurnBatchGroupError {
    source_indices: Vec<usize>,
    #[source]
    source: BurnDecodeError,
}

impl BurnBatchGroupError {
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos"), test))]
    pub(crate) fn new(source_indices: Vec<usize>, source: BurnDecodeError) -> Self {
        Self {
            source_indices,
            source,
        }
    }

    /// Original input indices whose dense tensor group was discarded.
    #[must_use]
    pub fn source_indices(&self) -> &[usize] {
        &self.source_indices
    }

    /// Structured codec, framework, or interop failure for this group.
    #[must_use]
    pub const fn source(&self) -> &BurnDecodeError {
        &self.source
    }

    /// Consume the group failure into affected indices and its source.
    #[must_use]
    pub fn into_parts(self) -> (Vec<usize>, BurnDecodeError) {
        (self.source_indices, self.source)
    }
}

/// Successful tensor groups plus indexed preparation and homogeneous execution failures.
#[derive(Debug)]
pub struct BurnBatchDecode<B: Backend> {
    /// Successfully decoded homogeneous tensor groups.
    pub groups: Vec<BurnBatchGroup<B>>,
    /// Structured preparation or decode failures in original input order.
    pub errors: Vec<IndexedBatchError>,
    /// Homogeneous groups discarded after adapter submission or completion failed.
    pub group_errors: Vec<BurnBatchGroupError>,
}
