// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    context_validation::validate_idwt_sequence_enqueue_context,
    launch_validation::plan_idwt_batch_launch,
};
use crate::{
    allocation::HostPhaseBudget,
    bytes::idwt_multi_jobs_as_bytes,
    context::{cuda_idwt_trace_enabled, CudaContext},
    error::CudaError,
    execution::{elapsed_event_us_ceil, CudaExecutionStats, CudaQueuedExecution},
    memory::{pooled_device_buffer, CudaBufferPool},
};

use super::super::{
    append_j2k_idwt_multi_kernel_jobs, format_idwt_batch_trace_row, idwt_batch_trace_row,
    types::{CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget},
    CudaJ2kIdwtBatchKernelMode,
};

impl CudaContext {
    /// Enqueue a sequence of batched inverse JPEG 2000 DWT stages while
    /// uploading all stage job metadata in one device buffer.
    ///
    /// # Safety
    ///
    /// Every target buffer must remain allocated and must not be mutated or
    /// reused until the returned execution is finished, dropped, or released
    /// after this context has completed the queued work. Within each stage,
    /// output allocations must be pairwise disjoint and may not overlap any
    /// concurrently read input allocation; dependencies may alias only across
    /// ordered stages. These rules and context ownership are validated at
    /// runtime. All pool clones must remain confined to that stream until the
    /// same completion point.
    #[doc(hidden)]
    pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        // SAFETY: this wrapper preserves the caller's target and pool lifetime
        // requirements and contributes no additional caller-live host owners.
        unsafe {
            self.j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(
                target_batches,
                pool,
                0,
            )
        }
    }

    /// Enqueue an IDWT sequence while accounting caller-live host metadata.
    ///
    /// # Safety
    ///
    /// The target buffers and pool must satisfy the same lifetime, aliasing,
    /// context, and stream-confinement requirements as
    /// [`Self::j2k_inverse_dwt_batch_sequence_enqueue_with_pool`].
    #[doc(hidden)]
    #[expect(
        clippy::too_many_lines,
        reason = "batched IDWT sequence preserves metadata upload and CUDA launch ordering"
    )]
    pub unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
    ) -> Result<CudaQueuedExecution, CudaError> {
        validate_idwt_sequence_enqueue_context(self, target_batches, pool)?;
        let total_target_count = target_batches.iter().try_fold(0usize, |count, targets| {
            count
                .checked_add(targets.len())
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
        })?;
        let mut host_budget =
            HostPhaseBudget::with_live_bytes("CUDA J2K IDWT sequence metadata", live_host_bytes)?;
        let mut all_jobs = host_budget.try_vec_with_capacity(total_target_count)?;
        let mut batches = host_budget.try_vec_with_capacity(target_batches.len())?;
        for targets in target_batches {
            let start = all_jobs.len();
            append_j2k_idwt_multi_kernel_jobs(targets, &mut all_jobs)?;
            let count = all_jobs.len().saturating_sub(start);
            if count == 0 {
                continue;
            }
            let Some(plan) = plan_idwt_batch_launch(&all_jobs[start..])? else {
                continue;
            };
            batches.push((start, count, plan));
        }
        if all_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
                pool_reuse_guard: None,
            });
        }
        self.inner.set_current()?;

        let mut queued_resources = host_budget.try_vec_with_capacity(1)?;
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&all_jobs))?;
        queued_resources.push(jobs_buffer);
        let jobs_base = pooled_device_buffer(&queued_resources[0])?.device_ptr();
        let job_size = std::mem::size_of::<CudaJ2kIdwtMultiKernelJob>();
        let mut kernel_dispatches = 0usize;
        let trace_enabled = cuda_idwt_trace_enabled();
        let pool_reuse_guard = pool.defer_reuse()?;
        let sequence_result = (|| -> Result<(), CudaError> {
            for (stage_index, (start, count, plan)) in batches.into_iter().enumerate() {
                let max_width = plan.max_width;
                let max_height = plan.max_height;
                let kernel_mode = plan.kernel_mode;
                let byte_offset = start
                    .checked_mul(job_size)
                    .ok_or(CudaError::LengthTooLarge { len: start })?;
                let jobs_ptr = jobs_base
                    .checked_add(byte_offset as u64)
                    .ok_or(CudaError::LengthTooLarge { len: byte_offset })?;
                let trace_start = if trace_enabled {
                    let event = self.create_event()?;
                    event.record_default_stream()?;
                    Some(event)
                } else {
                    None
                };
                match kernel_mode {
                    CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                        .launch_j2k_idwt_interleave_horizontal_53_multi_ptr(
                            jobs_ptr,
                            max_height as usize,
                            count,
                            false,
                        ),
                    CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                        .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                            jobs_ptr,
                            max_width as usize,
                            max_height as usize,
                            count,
                            false,
                        ),
                    CudaJ2kIdwtBatchKernelMode::Generic => self
                        .launch_j2k_idwt_interleave_horizontal_multi_ptr(
                            jobs_ptr,
                            max_height as usize,
                            count,
                            false,
                        ),
                }?;
                kernel_dispatches = kernel_dispatches.saturating_add(1);

                match kernel_mode {
                    CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                        .launch_j2k_idwt_vertical_53_multi_ptr(
                            jobs_ptr,
                            max_width as usize,
                            count,
                            false,
                        ),
                    CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                        .launch_j2k_idwt_vertical_97_multi_ptr(
                            jobs_ptr,
                            max_width as usize,
                            max_height as usize,
                            count,
                            false,
                        ),
                    CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi_ptr(
                        jobs_ptr,
                        max_width as usize,
                        count,
                        false,
                    ),
                }?;
                kernel_dispatches = kernel_dispatches.saturating_add(1);
                if let Some(trace_start) = trace_start {
                    let trace_end = self.create_event()?;
                    trace_end.record_default_stream()?;
                    trace_end.synchronize()?;
                    let elapsed_us = elapsed_event_us_ceil(&trace_start, &trace_end)?;
                    let end = start.saturating_add(count);
                    let row = idwt_batch_trace_row(
                        stage_index,
                        &all_jobs[start..end],
                        max_width,
                        max_height,
                        kernel_mode,
                        elapsed_us,
                    );
                    eprintln!("{}", format_idwt_batch_trace_row(row));
                }
            }
            Ok(())
        })();
        if let Err(error) = sequence_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }

        Ok(CudaQueuedExecution {
            resources: queued_resources,
            execution: CudaExecutionStats {
                kernel_dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: kernel_dispatches,
                hardware_decode: false,
            },
            pool_reuse_guard: Some(pool_reuse_guard),
        })
    }
}
