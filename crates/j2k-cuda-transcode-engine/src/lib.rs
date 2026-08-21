// SPDX-License-Identifier: MIT OR Apache-2.0

//! Coefficient-domain transcode operations over the low-level CUDA runtime.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(unreachable_pub)]

#[macro_use]
mod macros;
mod allocation;
mod build_flags;
mod bytes;
mod context;
mod error;
mod execution;
mod kernels;
mod memory;
#[cfg(test)]
mod tests;
mod transcode;

pub use build_flags::transcode_kernels_built;
pub use transcode::{
    CudaDwt97BatchGeometry, CudaDwt97BatchStageTimings, CudaDwt97BatchWithPoolRequest,
    CudaHtj2k97CodeblockBands, CudaHtj2k97CodeblockBatchWithPoolRequest,
    CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97I16CodeblockBatchWithPoolRequest,
    CudaHtj2k97QuantizeParams, CudaTranscodeDwt97Bands, CudaTranscodeReversible53Bands,
};

pub(crate) use j2k_cuda_runtime::CudaContext;

/// Borrowed coefficient-domain transcode operations over one CUDA context.
#[derive(Clone)]
pub struct CudaTranscodeEngine<'context> {
    pub(crate) context: &'context CudaContext,
}

impl<'context> CudaTranscodeEngine<'context> {
    /// Bind transcode operations to `context` without changing its ownership.
    #[must_use]
    pub const fn new(context: &'context CudaContext) -> Self {
        Self { context }
    }

    /// Return the borrowed low-level context.
    #[must_use]
    pub const fn context(&self) -> &'context CudaContext {
        self.context
    }
}

#[cfg(all(test, feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
pub(crate) use bytes::i32_slice_as_bytes_mut;
#[cfg(all(test, feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
pub(crate) use j2k_cuda_runtime::CudaPooledDeviceBuffer;
#[cfg(test)]
pub(crate) use j2k_cuda_runtime::{CudaBufferPool, CudaError};
#[cfg(test)]
pub(crate) use transcode::validate_dct_block_grid;
