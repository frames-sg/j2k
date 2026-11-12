// SPDX-License-Identifier: MIT OR Apache-2.0

mod abi;
mod bytes;
mod launch;
mod prepare;
mod queued;
#[cfg(test)]
mod tests;

pub use abi::{
    CudaClassicCodeBlockJob, CudaClassicDecodeStageTimings, CudaClassicDecodeTableResources,
    CudaClassicDecodeTarget, CudaClassicSegment, CudaClassicStatus,
};
pub use queued::CudaQueuedClassicDecode;

/// The classic Tier-1 decode kernel's name, as its launches report it in
/// [`j2k_cuda_runtime::CudaError::KernelJobStatus`].
#[doc(hidden)]
pub const CUDA_CLASSIC_DECODE_KERNEL_NAME: &str = "j2k_decode_classic_codeblocks_multi";
