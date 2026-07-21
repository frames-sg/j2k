// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::htj2k_dequantize_jobs_as_bytes,
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    memory::{pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer},
};

use super::super::{
    context_validation::validate_dequantize_context,
    planning::htj2k_dequantize_kernel_jobs_with_live_host_bytes, types::CudaHtj2kDequantizeTarget,
};

mod queued;

impl CudaContext {
    pub(super) fn submit_htj2k_dequantize_htj2k_codeblocks(
        &self,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<u128, CudaError> {
        if collect_stage_timings {
            let ((), dequant_us) =
                self.time_default_stream_named_us("j2k.htj2k.decode.dequantize", || {
                    self.launch_j2k_dequantize_htj2k_codeblocks(
                        coefficients,
                        jobs_buffer,
                        job_count,
                        CudaLaunchMode::Sync,
                    )
                })?;
            return Ok(dequant_us);
        }
        // SAFETY: the owning decode method retains coefficients and job
        // metadata until the ordered status D2H after this launch completes.
        unsafe {
            self.submit_default_stream_named("j2k.htj2k.decode.dequantize", || {
                self.launch_j2k_dequantize_htj2k_codeblocks(
                    coefficients,
                    jobs_buffer,
                    job_count,
                    CudaLaunchMode::Async,
                )
            })?;
        }
        Ok(0)
    }

    /// Dequantize HTJ2K code-block outputs that live in multiple device buffers
    /// with one completed CUDA dispatch, reusing caller-owned transient storage.
    ///
    /// The completion boundary keeps the locally uploaded job descriptor alive.
    /// Asynchronous decode pipelines must use
    /// [`Self::j2k_dequantize_queued_htj2k_cleanup_enqueue`], whose typed cleanup
    /// guard retains the descriptor through the later group completion.
    #[doc(hidden)]
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes(
            targets, pool, 0,
        )
    }

    /// Dequantize batched outputs while accounting caller-live host metadata.
    #[doc(hidden)]
    pub fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_and_live_host_bytes(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(
            targets,
            pool,
            live_host_bytes,
        )
    }

    fn j2k_dequantize_htj2k_codeblocks_multi_device_with_pool_impl(
        &self,
        targets: &[CudaHtj2kDequantizeTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaExecutionStats, CudaError> {
        validate_dequantize_context(self, targets, pool)?;
        let kernel_jobs =
            htj2k_dequantize_kernel_jobs_with_live_host_bytes(targets, live_host_bytes)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        self.inner.set_current()?;
        let jobs_buffer = pool.upload(htj2k_dequantize_jobs_as_bytes(&kernel_jobs))?;
        self.launch_j2k_dequantize_htj2k_codeblocks_multi(
            pooled_device_buffer(&jobs_buffer)?,
            kernel_jobs.len(),
            CudaLaunchMode::Sync,
        )?;
        Ok(CudaExecutionStats {
            kernel_dispatches: 1,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 1,
            hardware_decode: false,
        })
    }
}
