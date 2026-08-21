// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA codec engine and Driver API runtime used by J2K CUDA adapter crates.

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(unreachable_pub)]
#[macro_use]
mod macros;
mod allocation;
mod build_flags;
mod bytes;
mod context;
mod driver;
mod error;
mod execution;
mod kernel;
mod kernels;
mod memory;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use context::CudaKernelName;
pub use context::{
    CudaContext, CudaContextDiagnostics, CudaExternalHostOwner, CudaExternalHostReservation,
};
pub use error::CudaError;
#[doc(hidden)]
pub use error::{select_resource_release_error, select_uncertain_completion_error};
pub use execution::{
    cuda_kernel_param, elapsed_event_us_ceil, CudaEvent, CudaExecutionStats, CudaKernelBatchOutput,
    CudaKernelContiguousBatchOutput, CudaKernelOutput, CudaKernelParam, CudaPooledKernelOutput,
    CudaQueuedExecution, CudaSynchronizationOutcome,
};
pub use kernel::CudaKernelSpec;
pub use kernels::CudaLaunchGeometry;
#[doc(hidden)]
pub use memory::checked_image_words;
pub use memory::{
    CheckedDeviceBufferRanges, CudaBufferPool, CudaBufferPoolDiagnostics, CudaBufferPoolLimits,
    CudaBufferPoolReuseGuard, CudaBufferPoolTakeTrace, CudaDeviceBuffer, CudaDeviceBufferRange,
    CudaDeviceBufferView, CudaDeviceBufferViewMut, CudaExternalDeviceBufferViewMut,
    CudaPinnedUploadOperationGuard, CudaPinnedUploadStagingCheckout,
    CudaPinnedUploadStagingPoolDiagnostics, CudaPinnedUploadStagingPoolLimits,
    CudaPooledDeviceBuffer,
};

#[cfg(test)]
pub(crate) use bytes::{f32_slice_as_bytes_mut, i32_slice_as_bytes_mut};
#[cfg(test)]
pub(crate) use memory::{copy_pooled_bytes_to_vec_uninit, pool_fit_buffer_index_by_len};
