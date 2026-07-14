// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Instant;

mod dequant;

use crate::{
    allocation::HostPhaseBudget,
    bytes::{
        htj2k_cleanup_multi_jobs_as_bytes, htj2k_jobs_as_bytes, htj2k_statuses_as_bytes_mut,
        htj2k_statuses_byte_len,
    },
    context::CudaContext,
    error::CudaError,
    execution::{CudaExecutionStats, CudaLaunchMode},
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer},
};

use super::{
    context_validation::validate_cleanup_context,
    planning::{
        htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes,
        htj2k_decode_multi_cleanup_dequant_kernel_for_jobs, htj2k_decode_multi_kernel_for_jobs,
        htj2k_kernel_jobs,
    },
    queued::CudaQueuedHtj2kCleanup,
    status::{first_status_error, select_status_release_result},
    types::{
        htj2k_decode_kernel_tables, CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCleanupTarget,
        CudaHtj2kCodeBlockJob, CudaHtj2kDecodeOutput, CudaHtj2kDecodeResources,
        CudaHtj2kDecodeStageTimings, CudaHtj2kStatus, Htj2kDecodeCodeblocksLaunch,
        Htj2kDecodeCodeblocksMultiLaunch,
    },
};

impl CudaContext {
    /// Enqueue HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch. The returned value must be kept live until `finish` validates
    /// the kernel statuses after the default stream has completed.
    ///
    /// # Safety
    ///
    /// Every target coefficient buffer must remain allocated and must not be
    /// mutated or reused until the returned cleanup is finished or dropped.
    /// Target allocations and each target's job write regions must be
    /// pairwise disjoint; both conditions are validated before launch.
    /// The decode payload and table resources must remain live for the same
    /// duration. The resources, targets, and pool must belong to this context
    /// (validated at runtime), and all pool clones must remain confined to this
    /// context's default stream until that completion point.
    #[doc(hidden)]
    pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        // SAFETY: this wrapper preserves the caller's target and pool lifetime
        // requirements and contributes no additional caller-live host owners.
        unsafe {
            self.decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(
                resources,
                targets,
                pool,
                0,
            )
        }
    }

    /// Enqueue HTJ2K cleanup while accounting caller-live host metadata.
    ///
    /// # Safety
    ///
    /// The target buffers, resources, and pool must satisfy the same lifetime,
    /// aliasing, context, and stream-confinement requirements as
    /// [`Self::decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool`].
    #[doc(hidden)]
    pub unsafe fn decode_htj2k_codeblocks_cleanup_multi_enqueue_with_resources_and_pool_and_live_host_bytes(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaQueuedHtj2kCleanup, CudaError> {
        validate_cleanup_context(self, resources, targets, pool)?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes(
            targets,
            resources.payload_len,
            live_host_bytes,
        )?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedHtj2kCleanup {
                context: self.clone(),
                resources: Vec::new(),
                status_buffer: None,
                status_count: 0,
                kernel_name: "j2k_htj2k_decode_codeblocks_multi",
                execution: CudaExecutionStats::default(),
                pool_reuse_guard: None,
                finish_host_live_bytes: 0,
            });
        }
        self.inner.set_current()?;
        let (decode_kernel, decode_kernel_name) = htj2k_decode_multi_kernel_for_jobs(&kernel_jobs);
        let tables = htj2k_decode_kernel_tables(resources)?;

        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA queued HTJ2K cleanup metadata",
            live_host_bytes,
        )?;
        host_budget.account_vec(&kernel_jobs)?;
        let mut queued_resources = host_budget.try_vec_with_capacity(1)?;
        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(&kernel_jobs))?;
        queued_resources.push(jobs_buffer);
        let mut finish_budget = HostPhaseBudget::with_live_bytes(
            "CUDA queued HTJ2K cleanup retained metadata",
            live_host_bytes,
        )?;
        finish_budget.account_vec(&queued_resources)?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        let payload_buffer = resources.payload.buffer()?;
        let jobs_device_buffer = pooled_device_buffer(&queued_resources[0])?;
        let status_device_buffer = pooled_device_buffer(&status_buffer)?;
        let pool_reuse_guard = pool.defer_reuse()?;
        let launch_result =
            self.launch_htj2k_decode_codeblocks_multi(Htj2kDecodeCodeblocksMultiLaunch {
                kernel: decode_kernel,
                payload: payload_buffer,
                jobs: jobs_device_buffer,
                tables,
                statuses: status_device_buffer,
                job_count: kernel_jobs.len(),
                mode: CudaLaunchMode::Async,
            });
        if let Err(error) = launch_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }

        Ok(CudaQueuedHtj2kCleanup {
            context: self.clone(),
            resources: queued_resources,
            status_buffer: Some(status_buffer),
            status_count: kernel_jobs.len(),
            kernel_name: decode_kernel_name,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            pool_reuse_guard: Some(pool_reuse_guard),
            finish_host_live_bytes: finish_budget.live_bytes(),
        })
    }

    fn run_htj2k_cleanup_multi_kernel(
        &self,
        resources: &CudaHtj2kDecodeResources,
        kernel_jobs: &[CudaHtj2kCleanupMultiKernelJob],
        pool: &CudaBufferPool,
        selected_kernel: (CudaKernel, &'static str),
        collect_stage_timings: bool,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        let (decode_kernel, decode_kernel_name) = selected_kernel;
        let mut statuses =
            host_budget.try_vec_filled(kernel_jobs.len(), CudaHtj2kStatus::default())?;
        let jobs_buffer = pool.upload(htj2k_cleanup_multi_jobs_as_bytes(kernel_jobs))?;
        let status_buffer = pool.take(htj2k_statuses_byte_len(kernel_jobs.len())?)?;
        let tables = htj2k_decode_kernel_tables(resources)?;
        let payload_buffer = resources.payload.buffer()?;
        let jobs_device_buffer = pooled_device_buffer(&jobs_buffer)?;
        let status_device_buffer = pooled_device_buffer(&status_buffer)?;
        let pool_reuse_guard = pool.defer_reuse()?;
        let mode = if collect_stage_timings {
            CudaLaunchMode::Sync
        } else {
            CudaLaunchMode::Async
        };
        let launch_result =
            self.launch_htj2k_decode_codeblocks_multi(Htj2kDecodeCodeblocksMultiLaunch {
                kernel: decode_kernel,
                payload: payload_buffer,
                jobs: jobs_device_buffer,
                tables,
                statuses: status_device_buffer,
                job_count: kernel_jobs.len(),
                mode,
            });
        if let Err(error) = launch_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }
        let mut pending_pool_reuse = Some(pool_reuse_guard);

        let status_d2h_start = collect_stage_timings.then(Instant::now);
        if let Err(error) = status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses)) {
            if let Some(pool_reuse_guard) = pending_pool_reuse.take() {
                return pool_reuse_guard.release_after_recoverable_operation_error(error);
            }
            return Err(error);
        }
        // A successful synchronous device-to-host copy completes the
        // preceding default-stream cleanup launch.
        let status_d2h_us = status_d2h_start.map_or(0, |start| start.elapsed().as_micros());
        let release_result = pending_pool_reuse
            .take()
            .ok_or_else(|| CudaError::StatePoisoned {
                message: "HTJ2K cleanup pool guard disappeared before release".to_string(),
            })?
            .release();
        let execution = select_status_release_result(
            CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 1,
                hardware_decode: false,
            },
            first_status_error(&statuses, decode_kernel_name),
            release_result,
        )?;

        Ok((
            execution,
            CudaHtj2kDecodeStageTimings {
                status_d2h_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        ))
    }

    /// Decode HTJ2K cleanup passes for multiple output buffers with one CUDA
    /// dispatch and return optional host-side timing splits.
    ///
    /// Dequantization is left to a later dispatch. When `collect_stage_timings`
    /// is false, the cleanup kernel launch is left asynchronous and the
    /// mandatory status readback remains the completion point.
    #[doc(hidden)]
    pub fn decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        self.decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed_and_live_host_bytes(
            resources,
            targets,
            pool,
            collect_stage_timings,
            0,
        )
    }

    /// Decode batched cleanup while accounting caller-live host metadata.
    #[doc(hidden)]
    pub fn decode_htj2k_codeblocks_cleanup_multi_with_resources_and_pool_timed_and_live_host_bytes(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
        live_host_bytes: usize,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        validate_cleanup_context(self, resources, targets, pool)?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes(
            targets,
            resources.payload_len,
            live_host_bytes,
        )?;
        if kernel_jobs.is_empty() {
            return Ok((
                CudaExecutionStats::default(),
                CudaHtj2kDecodeStageTimings::default(),
            ));
        }
        self.inner.set_current()?;
        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA HTJ2K cleanup completion metadata",
            live_host_bytes,
        )?;
        host_budget.account_vec(&kernel_jobs)?;

        self.run_htj2k_cleanup_multi_kernel(
            resources,
            &kernel_jobs,
            pool,
            htj2k_decode_multi_kernel_for_jobs(&kernel_jobs),
            collect_stage_timings,
            &mut host_budget,
        )
    }

    /// Decode HTJ2K cleanup-only passes and dequantize their coefficients in
    /// one CUDA dispatch. Targets containing refinement passes are rejected so
    /// callers can fall back to cleanup followed by dequantization.
    #[doc(hidden)]
    pub fn decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        self.decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed_and_live_host_bytes(
            resources,
            targets,
            pool,
            collect_stage_timings,
            0,
        )
    }

    /// Decode fused cleanup/dequantization while accounting caller-live host metadata.
    #[doc(hidden)]
    pub fn decode_htj2k_codeblocks_cleanup_dequantize_multi_with_resources_and_pool_timed_and_live_host_bytes(
        &self,
        resources: &CudaHtj2kDecodeResources,
        targets: &[CudaHtj2kCleanupTarget<'_>],
        pool: &CudaBufferPool,
        collect_stage_timings: bool,
        live_host_bytes: usize,
    ) -> Result<(CudaExecutionStats, CudaHtj2kDecodeStageTimings), CudaError> {
        validate_cleanup_context(self, resources, targets, pool)?;
        let kernel_jobs = htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes(
            targets,
            resources.payload_len,
            live_host_bytes,
        )?;
        if kernel_jobs.is_empty() {
            return Ok((
                CudaExecutionStats::default(),
                CudaHtj2kDecodeStageTimings::default(),
            ));
        }
        let Some((decode_kernel, decode_kernel_name)) =
            htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(&kernel_jobs)
        else {
            return Err(CudaError::InvalidArgument {
                message: "fused HTJ2K cleanup/dequantize requires cleanup-only jobs".to_string(),
            });
        };
        self.inner.set_current()?;
        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA HTJ2K fused cleanup completion metadata",
            live_host_bytes,
        )?;
        host_budget.account_vec(&kernel_jobs)?;

        self.run_htj2k_cleanup_multi_kernel(
            resources,
            &kernel_jobs,
            pool,
            (decode_kernel, decode_kernel_name),
            collect_stage_timings,
            &mut host_budget,
        )
    }

    pub(super) fn decode_htj2k_codeblocks_with_resources_impl(
        &self,
        resources: &CudaHtj2kDecodeResources,
        jobs: &[CudaHtj2kCodeBlockJob],
        output_words: usize,
        collect_stage_timings: bool,
    ) -> Result<CudaHtj2kDecodeOutput, CudaError> {
        if !resources.is_owned_by(self)? {
            return Err(CudaError::InvalidArgument {
                message: "HTJ2K decode resources must belong to the launch context".to_string(),
            });
        }
        let validated_jobs = htj2k_kernel_jobs(jobs, resources.payload_len, output_words)?;
        let output_layout = validated_jobs.output_layout;
        let kernel_jobs = validated_jobs.jobs;
        self.inner.set_current()?;
        let coefficients = self.allocate(output_layout.output_bytes)?;
        if output_layout.needs_zero_fill {
            self.memset_d32(&coefficients, 0, output_words)?;
        }
        if kernel_jobs.is_empty() {
            if output_layout.needs_zero_fill {
                self.synchronize()?;
            }
            return Ok(CudaHtj2kDecodeOutput {
                coefficients,
                execution: CudaExecutionStats::default(),
                statuses: Vec::new(),
                stage_timings: CudaHtj2kDecodeStageTimings::default(),
            });
        }

        let mut host_budget = HostPhaseBudget::new("CUDA HTJ2K decode completion metadata");
        host_budget.account_vec(&kernel_jobs)?;
        let mut statuses = host_budget.try_vec_filled(jobs.len(), CudaHtj2kStatus::default())?;
        let jobs_buffer = self.upload(htj2k_jobs_as_bytes(&kernel_jobs))?;
        let status_buffer = self.allocate(htj2k_statuses_byte_len(jobs.len())?)?;

        let has_refinement = jobs
            .iter()
            .any(|job| job.refinement_length > 0 || job.number_of_coding_passes > 1);
        let (ht_cleanup_us, dequant_us) = self.submit_htj2k_decode_and_dequantize(
            resources,
            &coefficients,
            &jobs_buffer,
            &status_buffer,
            jobs.len(),
            collect_stage_timings,
        )?;

        status_buffer.copy_to_host(htj2k_statuses_as_bytes_mut(&mut statuses))?;
        if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
            return Err(CudaError::KernelStatus {
                kernel: "j2k_htj2k_decode_codeblocks",
                code: status.code,
                detail: status.detail,
            });
        }

        Ok(CudaHtj2kDecodeOutput {
            coefficients,
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
            statuses,
            stage_timings: CudaHtj2kDecodeStageTimings {
                ht_cleanup_us,
                ht_refine_us: if has_refinement { ht_cleanup_us } else { 0 },
                dequant_us,
                ..CudaHtj2kDecodeStageTimings::default()
            },
        })
    }

    fn submit_htj2k_decode_and_dequantize(
        &self,
        resources: &CudaHtj2kDecodeResources,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        status_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<(u128, u128), CudaError> {
        let ht_cleanup_us = self.submit_htj2k_decode_cleanup(
            resources,
            coefficients,
            jobs_buffer,
            status_buffer,
            job_count,
            collect_stage_timings,
        )?;
        let dequant_us = self.submit_htj2k_dequantize_htj2k_codeblocks(
            coefficients,
            jobs_buffer,
            job_count,
            collect_stage_timings,
        )?;
        Ok((ht_cleanup_us, dequant_us))
    }

    fn submit_htj2k_decode_cleanup(
        &self,
        resources: &CudaHtj2kDecodeResources,
        coefficients: &CudaDeviceBuffer,
        jobs_buffer: &CudaDeviceBuffer,
        status_buffer: &CudaDeviceBuffer,
        job_count: usize,
        collect_stage_timings: bool,
    ) -> Result<u128, CudaError> {
        let tables = htj2k_decode_kernel_tables(resources)?;
        if collect_stage_timings {
            let ((), ht_cleanup_us) =
                self.time_default_stream_named_us("j2k.htj2k.decode.cleanup", || {
                    self.launch_htj2k_decode_codeblocks(Htj2kDecodeCodeblocksLaunch {
                        payload: resources.payload.buffer()?,
                        coefficients,
                        jobs: jobs_buffer,
                        tables,
                        statuses: status_buffer,
                        job_count,
                        mode: CudaLaunchMode::Sync,
                    })
                })?;
            return Ok(ht_cleanup_us);
        }
        // SAFETY: the owning decode method retains payload, tables, jobs,
        // statuses, and coefficients until an ordered status D2H establishes
        // completion before any resource is released.
        unsafe {
            self.submit_default_stream_named("j2k.htj2k.decode.cleanup", || {
                self.launch_htj2k_decode_codeblocks(Htj2kDecodeCodeblocksLaunch {
                    payload: resources.payload.buffer()?,
                    coefficients,
                    jobs: jobs_buffer,
                    tables,
                    statuses: status_buffer,
                    job_count,
                    mode: CudaLaunchMode::Async,
                })
            })?;
        }
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ht_status_timing_excludes_pool_release() {
        let source = include_str!("completion.rs");
        let status_copy = source
            .find("status_buffer.copy_to_host")
            .expect("HT status copy");
        let status_timing = source
            .find("let status_d2h_us")
            .expect("HT status timing result");
        let pool_release = source
            .find("let release_result = pending_pool_reuse")
            .expect("HT pool release");
        assert!(status_copy < status_timing && status_timing < pool_release);
    }
}
