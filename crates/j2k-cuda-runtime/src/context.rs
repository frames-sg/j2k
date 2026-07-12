// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::{
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    htj2k_decode::{
        htj2k_decode_needs_zero_fill, CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput,
        CudaHtj2kDecodeStageTimings, CudaQueuedHtj2kCleanup,
    },
    memory::pooled_device_buffer,
};

mod band_transfer;
mod compact;
mod device;
mod inner;
mod kernel_cache;
mod kernel_dispatch;
mod lifecycle;
mod operations;
mod pinned_host;
mod resource_creation;
#[cfg(test)]
mod test_kernels;

pub use self::compact::{CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks};
#[cfg(test)]
pub(crate) use self::pinned_host::validate_non_null_pinned_host_allocation;
#[cfg(test)]
pub(crate) use self::test_kernels::{CudaKernelModule, CudaKernelName};
pub(crate) use self::{
    band_transfer::cuda_idwt_trace_enabled,
    compact::HTJ2K_UVLC_ENCODE_TABLE_BYTES,
    inner::ContextInner,
    kernel_cache::{CompiledKernel, CompiledKernelKey},
    lifecycle::ContextResourceLifecycle,
    operations::ensure_context_ownership,
    pinned_host::PinnedUploadStaging,
    resource_creation::{validate_device_allocation, validate_resource_handle},
};

/// CUDA driver context shared by J2K CUDA adapter crates.
#[derive(Clone)]
pub struct CudaContext {
    pub(crate) inner: Arc<ContextInner>,
}

impl CudaContext {
    /// Returns whether both handles own the same CUDA driver context.
    #[doc(hidden)]
    #[must_use]
    pub fn is_same_context(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Dequantize HTJ2K cleanup outputs using the metadata buffer already held
    /// live by a queued cleanup launch.
    #[doc(hidden)]
    pub fn j2k_dequantize_queued_htj2k_cleanup_with_pool(
        &self,
        cleanup: &CudaQueuedHtj2kCleanup,
    ) -> Result<CudaExecutionStats, CudaError> {
        if !self.is_same_context(&cleanup.context) {
            return Err(CudaError::InvalidArgument {
                message: "queued HTJ2K cleanup belongs to a different CUDA context".to_string(),
            });
        }
        self.inner.set_current()?;
        if cleanup.status_count == 0 {
            return Ok(CudaExecutionStats::default());
        }
        let Some(jobs_buffer) = cleanup.resources.first() else {
            return Err(CudaError::InvalidArgument {
                message: "queued HTJ2K cleanup has no metadata buffer".to_string(),
            });
        };
        self.launch_j2k_dequantize_htj2k_cleanup_jobs_multi(
            pooled_device_buffer(jobs_buffer)?,
            cleanup.status_count,
            CudaLaunchMode::Sync,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }

    pub(crate) fn decode_empty_htj2k_codeblocks(
        &self,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        self.inner.set_current()?;
        let output_bytes = output_words
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(CudaError::LengthTooLarge { len: output_words })?;
        let coefficients = self.allocate(output_bytes)?;
        if htj2k_decode_needs_zero_fill(jobs, output_words)? {
            self.memset_d32(&coefficients, 0, output_words)?;
            self.synchronize()?;
        }
        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats::default(),
            statuses: Vec::new(),
            stage_timings: CudaHtj2kDecodeStageTimings::default(),
        })
    }
}

impl std::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod structure_tests;
