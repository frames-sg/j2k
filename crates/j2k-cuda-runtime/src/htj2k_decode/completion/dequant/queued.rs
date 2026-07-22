// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    memory::pooled_device_buffer,
};

use super::super::super::queued::CudaQueuedHtj2kCleanup;

impl CudaContext {
    /// Dequantize HTJ2K cleanup outputs using the metadata buffer already held
    /// live by a queued cleanup launch.
    #[doc(hidden)]
    pub fn j2k_dequantize_queued_htj2k_cleanup_with_pool(
        &self,
        cleanup: &CudaQueuedHtj2kCleanup,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_queued_htj2k_cleanup(cleanup, CudaLaunchMode::Sync)
    }

    /// Enqueue dequantization using metadata retained by queued HTJ2K cleanup.
    ///
    /// # Safety
    ///
    /// `cleanup`, all coefficient targets referenced by its metadata, and the
    /// owning pool must remain live and unavailable for reuse until a later
    /// same-context completion point.
    #[doc(hidden)]
    pub unsafe fn j2k_dequantize_queued_htj2k_cleanup_enqueue(
        &self,
        cleanup: &CudaQueuedHtj2kCleanup,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_queued_htj2k_cleanup(cleanup, CudaLaunchMode::Async)
    }

    fn j2k_dequantize_queued_htj2k_cleanup(
        &self,
        cleanup: &CudaQueuedHtj2kCleanup,
        mode: CudaLaunchMode,
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
            mode,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }
}
