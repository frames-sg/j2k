// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    context_validation::validate_idwt_sequence_enqueue_context,
    launch_validation::{plan_idwt_batch_launch, IdwtBatchLaunchPlan},
};
use crate::{
    allocation::HostPhaseBudget,
    bytes::idwt_multi_jobs_as_bytes,
    context::cuda_idwt_trace_enabled,
    driver::CuDevicePtr,
    error::CudaError,
    execution::{CudaExecutionStats, CudaQueuedExecution},
    memory::{pooled_device_buffer, CudaBufferPool},
};

use super::super::{
    append_j2k_idwt_multi_kernel_jobs, format_idwt_batch_trace_row, idwt_batch_trace_row,
    types::{CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget},
};
use super::{normalized_idwt_job, CudaJ2kIdwtBatchStageProfile, CudaJ2kIdwtNormalization};

#[derive(Clone, Copy)]
struct IdwtSequenceStageLaunch {
    stage_index: usize,
    stage_count: usize,
    start: usize,
    count: usize,
    plan: IdwtBatchLaunchPlan,
    jobs_base: CuDevicePtr,
    job_size: usize,
    trace_enabled: bool,
    collect_stage_profile: bool,
}

impl crate::J2kCudaEngine<'_> {
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

    pub(super) unsafe fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool_and_live_host_bytes_impl(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
        live_host_bytes: usize,
        normalization: CudaJ2kIdwtNormalization,
        collect_stage_profile: bool,
    ) -> Result<(CudaQueuedExecution, CudaJ2kIdwtBatchStageProfile), CudaError> {
        validate_idwt_sequence_enqueue_context(self.context, target_batches, pool)?;
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
            for kernel_job in &mut all_jobs[start..] {
                kernel_job.job = normalized_idwt_job(kernel_job.job, normalization);
            }
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
            return Ok((
                CudaQueuedExecution {
                    resources: Vec::new(),
                    execution: CudaExecutionStats::default(),
                    pool_reuse_guard: None,
                },
                CudaJ2kIdwtBatchStageProfile::default(),
            ));
        }
        self.prepare_operation()?;

        let mut queued_resources = host_budget.try_vec_with_capacity(1)?;
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&all_jobs))?;
        queued_resources.push(jobs_buffer);
        let jobs_base = pooled_device_buffer(&queued_resources[0])?.device_ptr();
        let job_size = std::mem::size_of::<CudaJ2kIdwtMultiKernelJob>();
        let mut kernel_dispatches = 0usize;
        let trace_enabled = cuda_idwt_trace_enabled();
        let stage_count = batches.len();
        let mut final_stage_profile = CudaJ2kIdwtBatchStageProfile::default();
        let pool_reuse_guard = pool.defer_reuse()?;
        let sequence_result = (|| -> Result<(), CudaError> {
            for (stage_index, (start, count, plan)) in batches.into_iter().enumerate() {
                let stage_profile = self.launch_idwt_sequence_stage(
                    IdwtSequenceStageLaunch {
                        stage_index,
                        stage_count,
                        start,
                        count,
                        plan,
                        jobs_base,
                        job_size,
                        trace_enabled,
                        collect_stage_profile,
                    },
                    &all_jobs,
                )?;
                kernel_dispatches = kernel_dispatches.saturating_add(2);
                if stage_profile.final_stage {
                    final_stage_profile = stage_profile;
                }
            }
            Ok(())
        })();
        if let Err(error) = sequence_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }

        Ok((
            CudaQueuedExecution {
                resources: queued_resources,
                execution: CudaExecutionStats {
                    kernel_dispatches,
                    copy_kernel_dispatches: 0,
                    decode_kernel_dispatches: kernel_dispatches,
                    hardware_decode: false,
                },
                pool_reuse_guard: Some(pool_reuse_guard),
            },
            final_stage_profile,
        ))
    }

    fn launch_idwt_sequence_stage(
        &self,
        launch: IdwtSequenceStageLaunch,
        all_jobs: &[CudaJ2kIdwtMultiKernelJob],
    ) -> Result<CudaJ2kIdwtBatchStageProfile, CudaError> {
        let byte_offset = launch
            .start
            .checked_mul(launch.job_size)
            .ok_or(CudaError::LengthTooLarge { len: launch.start })?;
        let jobs_ptr = launch
            .jobs_base
            .checked_add(byte_offset as u64)
            .ok_or(CudaError::LengthTooLarge { len: byte_offset })?;
        let final_stage = launch.stage_index.saturating_add(1) == launch.stage_count;
        let profile_stage = launch.trace_enabled || (launch.collect_stage_profile && final_stage);
        let profile = if profile_stage {
            self.profile_j2k_idwt_batch_mode_ptr(
                launch.plan.kernel_mode,
                jobs_ptr,
                launch.plan.max_width as usize,
                launch.plan.max_height as usize,
                launch.count,
                final_stage,
            )?
        } else {
            self.launch_j2k_idwt_batch_mode_ptr(
                launch.plan.kernel_mode,
                jobs_ptr,
                launch.plan.max_width as usize,
                launch.plan.max_height as usize,
                launch.count,
                false,
            )?;
            CudaJ2kIdwtBatchStageProfile {
                final_stage,
                ..CudaJ2kIdwtBatchStageProfile::default()
            }
        };
        if launch.trace_enabled {
            let end = launch.start.saturating_add(launch.count);
            let row = idwt_batch_trace_row(
                launch.stage_index,
                &all_jobs[launch.start..end],
                launch.plan.max_width,
                launch.plan.max_height,
                launch.plan.kernel_mode,
                profile,
            );
            eprintln!("{}", format_idwt_batch_trace_row(row));
        }
        Ok(profile)
    }
}
