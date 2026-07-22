// SPDX-License-Identifier: MIT OR Apache-2.0

mod api;
mod completion;
mod context_validation;
mod launch;
pub(crate) mod output_regions;
mod planning;
mod queued;
mod status;
mod status_group;
mod types;

pub(crate) use self::planning::htj2k_decode_needs_zero_fill;
#[cfg(test)]
pub(crate) use self::planning::{
    htj2k_decode_multi_cleanup_dequant_kernel_for_jobs, htj2k_decode_multi_kernel_for_jobs,
};
pub use self::queued::CudaQueuedHtj2kCleanup;
pub use self::status_group::CudaQueuedHtj2kCleanupGroup;
pub use self::types::{
    htj2k_cleanup_multi_descriptor_bytes, CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob,
    CudaHtj2kDecodeOutput, CudaHtj2kDecodeResources, CudaHtj2kDecodeStageTimings,
    CudaHtj2kDecodeTableResources, CudaHtj2kDecodeTables, CudaHtj2kDequantizeTarget,
    CudaHtj2kStatus, CudaPooledHtj2kDecodeOutput,
};
pub(crate) use self::types::{
    CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCodeBlockKernelJob, CudaHtj2kDecodePayload,
    CudaHtj2kDequantizeKernelJob, HTJ2K_STATUS_OK, HTJ2K_STATUS_UNSUPPORTED,
};
