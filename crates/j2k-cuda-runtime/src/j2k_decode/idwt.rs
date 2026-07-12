// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::{idwt_job_as_bytes, idwt_multi_jobs_as_bytes},
    context::CudaContext,
    error::CudaError,
    execution::{
        CudaExecutionStats, CudaKernelOutput, CudaLaunchMode, CudaPooledKernelOutput,
        CudaQueuedExecution,
    },
    kernels::CudaKernel,
    memory::{pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer},
};

use super::{
    j2k_idwt_multi_kernel_jobs,
    types::{CudaJ2kIdwtJob, CudaJ2kIdwtTarget},
    CudaJ2kIdwtBatchKernelMode,
};

mod context_validation;
pub(super) mod job_validation;
pub(in crate::j2k_decode) mod launch_validation;
mod preflight;
mod sequence;

use context_validation::{idwt_inputs_belong_to_context, validate_idwt_enqueue_context};
use job_validation::validate_idwt_job;
use launch_validation::{plan_idwt_batch_launch, validate_idwt_single_launch};
use preflight::validate_idwt_single_request;

#[derive(Clone, Copy)]
struct J2kInverseDwtSinglePoolRequest<'a> {
    bands: [&'a CudaDeviceBuffer; 4],
    job: CudaJ2kIdwtJob,
    synchronize_each_launch: bool,
    pool: &'a CudaBufferPool,
}

impl CudaContext {
    /// Apply one inverse JPEG 2000 DWT decomposition to device coefficient bands.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_single_device(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_impl(ll, hl, lh, hh, job)
    }

    /// Apply one inverse JPEG 2000 DWT decomposition with caller-owned
    /// transient buffer reuse.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_single_device_with_pool(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(J2kInverseDwtSinglePoolRequest {
            bands: [ll, hl, lh, hh],
            job,
            synchronize_each_launch: true,
            pool,
        })
    }

    /// Apply one inverse JPEG 2000 DWT decomposition with caller-owned
    /// transient buffer reuse and without per-kernel synchronizes.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_single_device_untimed_with_pool(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
        pool: &CudaBufferPool,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        self.j2k_inverse_dwt_single_device_with_pool_impl(J2kInverseDwtSinglePoolRequest {
            bands: [ll, hl, lh, hh],
            job,
            synchronize_each_launch: false,
            pool,
        })
    }

    /// Apply inverse JPEG 2000 DWT decompositions for multiple independent
    /// targets using one dispatch per parallel stage.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_batch_device_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, pool)
    }

    /// Enqueue batched inverse JPEG 2000 DWT decompositions without
    /// synchronizing.
    ///
    /// # Safety
    ///
    /// Every target buffer must remain allocated and must not be mutated or
    /// reused until the returned execution is finished, dropped, or released
    /// after this context has completed the queued work. Within the batch,
    /// output allocations must be pairwise disjoint and may not overlap any
    /// concurrently read input allocation; both conditions are validated at
    /// runtime. The supplied pool and every target must belong to this context
    /// (also validated), and all pool clones must remain confined to that
    /// stream until the same completion point.
    #[doc(hidden)]
    pub unsafe fn j2k_inverse_dwt_batch_device_enqueue_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        validate_idwt_enqueue_context(self, targets, pool)?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        let Some(plan) = plan_idwt_batch_launch(&kernel_jobs)? else {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
                pool_reuse_guard: None,
            });
        };
        self.inner.set_current()?;
        let mut host_budget = HostPhaseBudget::new("CUDA J2K IDWT queued metadata");
        host_budget.account_vec(&kernel_jobs)?;
        let mut queued_resources = host_budget.try_vec_with_capacity(1)?;
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        queued_resources.push(jobs_buffer);
        let jobs_device = pooled_device_buffer(&queued_resources[0])?;
        let max_width = plan.max_width;
        let max_height = plan.max_height;
        let kernel_mode = plan.kernel_mode;
        let pool_reuse_guard = pool.defer_reuse()?;
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
        };
        if let Err(error) = interleave_horizontal_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                false,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    false,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                false,
            ),
        };
        if let Err(error) = vertical_result {
            return pool_reuse_guard.synchronize_then_error(error);
        }

        Ok(CudaQueuedExecution {
            resources: queued_resources,
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
            pool_reuse_guard: Some(pool_reuse_guard),
        })
    }

    fn j2k_inverse_dwt_batch_device_with_pool_impl(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaExecutionStats, CudaError> {
        validate_idwt_enqueue_context(self, targets, pool)?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        let Some(plan) = plan_idwt_batch_launch(&kernel_jobs)? else {
            return Ok(CudaExecutionStats::default());
        };
        self.inner.set_current()?;
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = plan.max_width;
        let max_height = plan.max_height;
        let kernel_mode = plan.kernel_mode;
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    true,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    true,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    true,
                ),
        };
        interleave_horizontal_result?;
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                true,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    true,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                true,
            ),
        };
        vertical_result?;

        Ok(CudaExecutionStats {
            kernel_dispatches: 2,
            copy_kernel_dispatches: 0,
            decode_kernel_dispatches: 2,
            hardware_decode: false,
        })
    }

    fn j2k_inverse_dwt_single_device_impl(
        &self,
        ll: &CudaDeviceBuffer,
        hl: &CudaDeviceBuffer,
        lh: &CudaDeviceBuffer,
        hh: &CudaDeviceBuffer,
        job: CudaJ2kIdwtJob,
    ) -> Result<CudaKernelOutput, CudaError> {
        let validated = validate_idwt_single_request(self, [ll, hl, lh, hh], job)?;
        let output = self.allocate(validated.output_bytes)?;
        if validated.is_empty() {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        let width = validated.width;
        let height = validated.height;

        let job_buffer = self.upload(idwt_job_as_bytes(&job))?;
        let (horizontal_kernel, vertical_kernel) = if job.irreversible97 == 0 {
            (
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtVertical53,
            )
        } else {
            (
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVertical97,
            )
        };
        self.launch_j2k_idwt_interleave(
            [ll, hl, lh, hh],
            &output,
            &job_buffer,
            width,
            height,
            CudaLaunchMode::Sync,
        )?;
        self.launch_j2k_idwt_horizontal(
            horizontal_kernel,
            &output,
            &job_buffer,
            height as usize,
            CudaLaunchMode::Sync,
        )?;
        self.launch_j2k_idwt_vertical(
            vertical_kernel,
            &output,
            &job_buffer,
            width as usize,
            CudaLaunchMode::Sync,
        )?;
        Ok(CudaKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 3,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 3,
                hardware_decode: false,
            },
        })
    }

    fn j2k_inverse_dwt_single_device_with_pool_impl(
        &self,
        request: J2kInverseDwtSinglePoolRequest<'_>,
    ) -> Result<CudaPooledKernelOutput, CudaError> {
        let [ll, hl, lh, hh] = request.bands;
        let job = request.job;
        let synchronize_each_launch = request.synchronize_each_launch;
        let pool = request.pool;
        if !pool.is_owned_by(self) || !idwt_inputs_belong_to_context(self, [ll, hl, lh, hh]) {
            return Err(CudaError::InvalidArgument {
                message: "IDWT buffers and pool must belong to the launch context".to_string(),
            });
        }
        let validated = validate_idwt_job([ll, hl, lh, hh], None, job)?;
        validate_idwt_single_launch(validated.width, validated.height)?;
        let output = pool.take(validated.output_bytes)?;
        let output_buffer = pooled_device_buffer(&output)?;
        if validated.is_empty() {
            return Ok(CudaPooledKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }
        let width = validated.width;
        let height = validated.height;

        let job_buffer = pool.upload(idwt_job_as_bytes(&job))?;
        let job_device_buffer = pooled_device_buffer(&job_buffer)?;
        let (horizontal_kernel, vertical_kernel) = if job.irreversible97 == 0 {
            (
                CudaKernel::J2kIdwtHorizontal53,
                CudaKernel::J2kIdwtVertical53,
            )
        } else {
            (
                CudaKernel::J2kIdwtHorizontal97,
                CudaKernel::J2kIdwtVertical97,
            )
        };
        if synchronize_each_launch {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                output_buffer,
                job_device_buffer,
                width,
                height,
                CudaLaunchMode::Sync,
            )?;
            self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                output_buffer,
                job_device_buffer,
                height as usize,
                CudaLaunchMode::Sync,
            )?;
            self.launch_j2k_idwt_vertical(
                vertical_kernel,
                output_buffer,
                job_device_buffer,
                width as usize,
                CudaLaunchMode::Sync,
            )?;
        } else {
            let pool_reuse_guard = pool.defer_reuse()?;
            let launch_result = (|| {
                self.launch_j2k_idwt_interleave(
                    [ll, hl, lh, hh],
                    output_buffer,
                    job_device_buffer,
                    width,
                    height,
                    CudaLaunchMode::Async,
                )?;
                self.launch_j2k_idwt_horizontal(
                    horizontal_kernel,
                    output_buffer,
                    job_device_buffer,
                    height as usize,
                    CudaLaunchMode::Async,
                )?;
                self.launch_j2k_idwt_vertical(
                    vertical_kernel,
                    output_buffer,
                    job_device_buffer,
                    width as usize,
                    CudaLaunchMode::Async,
                )
            })();
            if let Err(error) = launch_result {
                return pool_reuse_guard.synchronize_then_error(error);
            }
            pool_reuse_guard.synchronize_and_release()?;
        }
        Ok(CudaPooledKernelOutput {
            buffer: output,
            execution: CudaExecutionStats {
                kernel_dispatches: 3,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 3,
                hardware_decode: false,
            },
        })
    }
}

#[cfg(test)]
mod tests;
