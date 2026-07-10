// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    bytes::{idwt_job_as_bytes, idwt_multi_jobs_as_bytes},
    context::{cuda_idwt_trace_enabled, CudaContext},
    error::CudaError,
    execution::{
        elapsed_event_us_ceil, CudaExecutionStats, CudaKernelOutput, CudaLaunchMode,
        CudaPooledKernelOutput, CudaQueuedExecution,
    },
    kernels::CudaKernel,
    memory::{checked_image_words, pooled_device_buffer, CudaBufferPool, CudaDeviceBuffer},
};

use super::{
    checked_f32_words_byte_len, format_idwt_batch_trace_row, idwt_batch_kernel_mode,
    idwt_batch_trace_row, j2k_idwt_multi_kernel_jobs,
    types::{CudaJ2kIdwtJob, CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget},
    CudaJ2kIdwtBatchKernelMode,
};

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
        self.j2k_inverse_dwt_single_device_impl(ll, hl, lh, hh, job, true)
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
        self.j2k_inverse_dwt_batch_device_with_pool_impl(targets, pool, true)
    }

    /// Enqueue batched inverse JPEG 2000 DWT decompositions without
    /// synchronizing. The returned value must be kept live until the default
    /// stream has been synchronized by the caller.
    #[doc(hidden)]
    pub fn j2k_inverse_dwt_batch_device_enqueue_with_pool(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = kernel_jobs
            .iter()
            .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
            .max()
            .unwrap_or(0);
        let max_height = kernel_jobs
            .iter()
            .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
            .max()
            .unwrap_or(0);
        let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
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
            let _ = self.synchronize();
            return Err(error);
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
            let _ = self.synchronize();
            return Err(error);
        }

        Ok(CudaQueuedExecution {
            resources: vec![jobs_buffer],
            execution: CudaExecutionStats {
                kernel_dispatches: 2,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: 2,
                hardware_decode: false,
            },
        })
    }

    /// Enqueue a sequence of batched inverse JPEG 2000 DWT stages while
    /// uploading all stage job metadata in one device buffer. The returned
    /// value must be kept live until the default stream has been synchronized
    /// by the caller.
    #[doc(hidden)]
    #[allow(clippy::too_many_lines)]
    pub fn j2k_inverse_dwt_batch_sequence_enqueue_with_pool(
        &self,
        target_batches: &[&[CudaJ2kIdwtTarget<'_>]],
        pool: &CudaBufferPool,
    ) -> Result<CudaQueuedExecution, CudaError> {
        self.inner.set_current()?;
        let mut all_jobs = Vec::new();
        let mut batches = Vec::new();
        for targets in target_batches {
            let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
            if kernel_jobs.is_empty() {
                continue;
            }
            let start = all_jobs.len();
            let count = kernel_jobs.len();
            let max_width = kernel_jobs
                .iter()
                .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
                .max()
                .unwrap_or(0);
            let max_height = kernel_jobs
                .iter()
                .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
                .max()
                .unwrap_or(0);
            let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
            all_jobs.extend(kernel_jobs);
            batches.push((start, count, max_width, max_height, kernel_mode));
        }
        if all_jobs.is_empty() {
            return Ok(CudaQueuedExecution {
                resources: Vec::new(),
                execution: CudaExecutionStats::default(),
            });
        }

        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&all_jobs))?;
        let jobs_base = pooled_device_buffer(&jobs_buffer)?.device_ptr();
        let job_size = std::mem::size_of::<CudaJ2kIdwtMultiKernelJob>();
        let mut kernel_dispatches = 0usize;
        let trace_enabled = cuda_idwt_trace_enabled();
        for (stage_index, (start, count, max_width, max_height, kernel_mode)) in
            batches.into_iter().enumerate()
        {
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
            let interleave_horizontal_result = match kernel_mode {
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
            };
            if let Err(error) = interleave_horizontal_result {
                let _ = self.synchronize();
                return Err(error);
            }
            kernel_dispatches = kernel_dispatches.saturating_add(1);

            let vertical_result = match kernel_mode {
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
            };
            if let Err(error) = vertical_result {
                let _ = self.synchronize();
                return Err(error);
            }
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

        Ok(CudaQueuedExecution {
            resources: vec![jobs_buffer],
            execution: CudaExecutionStats {
                kernel_dispatches,
                copy_kernel_dispatches: 0,
                decode_kernel_dispatches: kernel_dispatches,
                hardware_decode: false,
            },
        })
    }

    fn j2k_inverse_dwt_batch_device_with_pool_impl(
        &self,
        targets: &[CudaJ2kIdwtTarget<'_>],
        pool: &CudaBufferPool,
        synchronize_each_launch: bool,
    ) -> Result<CudaExecutionStats, CudaError> {
        self.inner.set_current()?;
        let kernel_jobs = j2k_idwt_multi_kernel_jobs(targets)?;
        if kernel_jobs.is_empty() {
            return Ok(CudaExecutionStats::default());
        }
        let jobs_buffer = pool.upload(idwt_multi_jobs_as_bytes(&kernel_jobs))?;
        let jobs_device = pooled_device_buffer(&jobs_buffer)?;
        let max_width = kernel_jobs
            .iter()
            .map(|job| job.job.rect.x1.saturating_sub(job.job.rect.x0))
            .max()
            .unwrap_or(0);
        let max_height = kernel_jobs
            .iter()
            .map(|job| job.job.rect.y1.saturating_sub(job.job.rect.y0))
            .max()
            .unwrap_or(0);
        let kernel_mode = idwt_batch_kernel_mode(&kernel_jobs, max_width, max_height);
        let interleave_horizontal_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self
                .launch_j2k_idwt_interleave_horizontal_53_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_interleave_horizontal_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self
                .launch_j2k_idwt_interleave_horizontal_multi(
                    jobs_device,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
        };
        if let Err(error) = interleave_horizontal_result {
            if !synchronize_each_launch {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        let vertical_result = match kernel_mode {
            CudaJ2kIdwtBatchKernelMode::Cooperative53 => self.launch_j2k_idwt_vertical_53_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                synchronize_each_launch,
            ),
            CudaJ2kIdwtBatchKernelMode::Cooperative97 => self
                .launch_j2k_idwt_vertical_97_multi_ptr(
                    jobs_device.device_ptr(),
                    max_width as usize,
                    max_height as usize,
                    kernel_jobs.len(),
                    synchronize_each_launch,
                ),
            CudaJ2kIdwtBatchKernelMode::Generic => self.launch_j2k_idwt_vertical_multi(
                jobs_device,
                max_width as usize,
                kernel_jobs.len(),
                synchronize_each_launch,
            ),
        };
        if let Err(error) = vertical_result {
            if !synchronize_each_launch {
                let _ = self.synchronize();
            }
            return Err(error);
        }
        if !synchronize_each_launch {
            self.synchronize()?;
        }

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
        synchronize_each_launch: bool,
    ) -> Result<CudaKernelOutput, CudaError> {
        let width = job.rect.x1.saturating_sub(job.rect.x0);
        let height = job.rect.y1.saturating_sub(job.rect.y0);
        let output_words = checked_image_words(width, height, 1)?;
        let output = self.allocate(checked_f32_words_byte_len(output_words)?)?;
        if output_words == 0 {
            return Ok(CudaKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

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
        if synchronize_each_launch {
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
        } else {
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                &output,
                &job_buffer,
                width,
                height,
                CudaLaunchMode::Async,
            )?;
            if let Err(error) = self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                &output,
                &job_buffer,
                height as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            if let Err(error) = self.launch_j2k_idwt_vertical(
                vertical_kernel,
                &output,
                &job_buffer,
                width as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            self.synchronize()?;
        }
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
        let width = job.rect.x1.saturating_sub(job.rect.x0);
        let height = job.rect.y1.saturating_sub(job.rect.y0);
        let output_words = checked_image_words(width, height, 1)?;
        let output = pool.take(checked_f32_words_byte_len(output_words)?)?;
        let output_buffer = pooled_device_buffer(&output)?;
        if output_words == 0 {
            return Ok(CudaPooledKernelOutput {
                buffer: output,
                execution: CudaExecutionStats::default(),
            });
        }

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
            self.launch_j2k_idwt_interleave(
                [ll, hl, lh, hh],
                output_buffer,
                job_device_buffer,
                width,
                height,
                CudaLaunchMode::Async,
            )?;
            if let Err(error) = self.launch_j2k_idwt_horizontal(
                horizontal_kernel,
                output_buffer,
                job_device_buffer,
                height as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            if let Err(error) = self.launch_j2k_idwt_vertical(
                vertical_kernel,
                output_buffer,
                job_device_buffer,
                width as usize,
                CudaLaunchMode::Async,
            ) {
                let _ = self.synchronize();
                return Err(error);
            }
            self.synchronize()?;
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
