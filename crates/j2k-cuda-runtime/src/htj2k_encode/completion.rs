// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::{
        htj2k_encode_compact_jobs_as_bytes, htj2k_encode_jobs_as_bytes,
        htj2k_encode_multi_input_jobs_as_bytes, htj2k_encode_statuses_as_bytes_mut,
        htj2k_encode_statuses_byte_len,
    },
    context::{CudaContext, CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kCompactEncodedCodeBlocks},
    error::CudaError,
    execution::CudaExecutionStats,
    memory::{
        copy_pooled_bytes_to_vec_uninit_with_budget, pooled_device_buffer, CudaBufferPool,
        CudaDeviceBuffer,
    },
};

use super::{
    planning::{htj2k_encode_compact_jobs, htj2k_encode_compact_jobs_multi_input},
    types::{
        CudaHtj2kEncodeCodeblocksLaunch, CudaHtj2kEncodeKernelJob,
        CudaHtj2kEncodeMultiInputKernelJob, CudaHtj2kEncodeMultiInputLaunch,
        CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeStatus,
        CudaHtj2kEncodedCodeBlock, CudaHtj2kEncodedCodeBlocks,
    },
};

impl CudaContext {
    #[expect(
        clippy::too_many_lines,
        reason = "pooled HT encode keeps CUDA buffer lifetimes, compaction, and timings atomic"
    )]
    pub(super) fn encode_htj2k_kernel_jobs_device_with_resources_and_pool(
        &self,
        coefficient_buffer: &CudaDeviceBuffer,
        kernel_jobs: &[CudaHtj2kEncodeKernelJob],
        kernel_jobs_capacity: usize,
        caller_live_host_bytes: usize,
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kEncodedCodeBlocks, CudaError> {
        let output_bytes = kernel_jobs
            .last()
            .map(|job| {
                (job.output_offset as usize)
                    .checked_add(job.output_capacity as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })
            .transpose()?
            .unwrap_or(0);

        let jobs_buffer = pool.upload(htj2k_encode_jobs_as_bytes(kernel_jobs))?;
        let output_buffer = pool.take(output_bytes)?;
        let status_buffer = pool.take(htj2k_encode_statuses_byte_len(kernel_jobs.len())?)?;
        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA HTJ2K code-block completion",
            caller_live_host_bytes,
        )?;
        host_budget.account_capacity::<CudaHtj2kEncodeKernelJob>(kernel_jobs_capacity)?;
        let mut statuses =
            host_budget.try_vec_filled(kernel_jobs.len(), CudaHtj2kEncodeStatus::default())?;

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("j2k.htj2k.encode.codeblocks", || {
                self.launch_htj2k_encode_codeblocks(&CudaHtj2kEncodeCodeblocksLaunch {
                    coefficients: coefficient_buffer,
                    output: pooled_device_buffer(&output_buffer)?,
                    jobs: pooled_device_buffer(&jobs_buffer)?,
                    tables: resources.launch_tables(),
                    statuses: pooled_device_buffer(&status_buffer)?,
                    job_count: kernel_jobs.len(),
                })
            })?;
        let ((), status_readback_us) = self.time_default_stream_named_us(
            "j2k.htj2k.encode.codeblocks.status_readback",
            || {
                status_buffer.copy_to_host(htj2k_encode_statuses_as_bytes_mut(&mut statuses))?;
                if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
                    return Err(CudaError::KernelStatus {
                        kernel: "j2k_htj2k_encode_codeblocks",
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(())
            },
        )?;

        let (compact_jobs, compact_output_bytes) =
            htj2k_encode_compact_jobs(&statuses, kernel_jobs, &mut host_budget)?;
        let compact_output_buffer = pool.take(compact_output_bytes)?;
        let compact_dispatched = compact_output_bytes != 0;
        let compact_us = if compact_dispatched {
            let compact_jobs_buffer =
                pool.upload(htj2k_encode_compact_jobs_as_bytes(&compact_jobs))?;
            let ((), compact_us) =
                self.time_default_stream_named_us("j2k.htj2k.encode.codeblocks.compact", || {
                    self.launch_htj2k_compact_codeblocks(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&compact_output_buffer)?,
                        pooled_device_buffer(&compact_jobs_buffer)?,
                        compact_jobs.len(),
                    )
                })?;
            compact_us
        } else {
            0
        };
        let (output, output_readback_us) = if compact_output_bytes == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "j2k.htj2k.encode.codeblocks.output_readback",
                || {
                    copy_pooled_bytes_to_vec_uninit_with_budget(
                        &compact_output_buffer,
                        compact_output_bytes,
                        &mut host_budget,
                    )
                },
            )?
        };
        let mut code_blocks = host_budget.try_vec_with_capacity(statuses.len())?;
        for ((status, job), compact_job) in statuses
            .into_iter()
            .zip(kernel_jobs.iter())
            .zip(compact_jobs.iter())
        {
            let data_len = usize::try_from(status.data_len)
                .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
            if data_len > job.output_capacity as usize {
                return Err(CudaError::LengthTooLarge { len: data_len });
            }
            let start = compact_job.compact_offset as usize;
            let end = start
                .checked_add(data_len)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if end > compact_output_bytes {
                return Err(CudaError::LengthTooLarge { len: end });
            }
            let data = host_budget.try_vec_from_slice(&output[start..end])?;
            code_blocks.push(CudaHtj2kEncodedCodeBlock {
                data,
                status,
                execution: CudaExecutionStats {
                    kernel_dispatches: 1,
                    copy_kernel_dispatches: usize::from(compact_dispatched),
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            compact_us,
            output_readback_us,
        );
        for block in &mut code_blocks {
            block.stage_timings = stage_timings;
        }
        let copy_kernel_dispatches =
            usize::from(code_blocks.iter().any(|block| !block.data().is_empty()));

        Ok(CudaHtj2kEncodedCodeBlocks {
            code_blocks,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }

    #[expect(
        clippy::too_many_lines,
        reason = "multi-input HT compaction keeps job offsets and device buffers synchronized"
    )]
    pub(super) fn encode_htj2k_multi_input_kernel_jobs_device_compact_with_resources_and_pool(
        &self,
        kernel_jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
        kernel_jobs_capacity: usize,
        caller_live_host_bytes: usize,
        resources: &CudaHtj2kEncodeResources,
        pool: &CudaBufferPool,
    ) -> Result<CudaHtj2kCompactEncodedCodeBlocks, CudaError> {
        let output_bytes = kernel_jobs
            .last()
            .map(|job| {
                (job.output_offset as usize)
                    .checked_add(job.output_capacity as usize)
                    .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
            })
            .transpose()?
            .unwrap_or(0);

        let jobs_buffer = pool.upload(htj2k_encode_multi_input_jobs_as_bytes(kernel_jobs))?;
        let output_buffer = pool.take(output_bytes)?;
        let status_buffer = pool.take(htj2k_encode_statuses_byte_len(kernel_jobs.len())?)?;
        let mut host_budget = HostPhaseBudget::with_live_bytes(
            "CUDA HTJ2K multi-input code-block completion",
            caller_live_host_bytes,
        )?;
        host_budget.account_capacity::<CudaHtj2kEncodeMultiInputKernelJob>(kernel_jobs_capacity)?;
        let mut statuses =
            host_budget.try_vec_filled(kernel_jobs.len(), CudaHtj2kEncodeStatus::default())?;
        let cleanup_only = kernel_jobs.iter().all(|job| job.target_coding_passes == 1);
        let cleanup_only_64 = cleanup_only
            && kernel_jobs
                .iter()
                .all(|job| job.width == 64 && job.height == 64 && job.coefficient_stride == 64);
        let status_kernel = if cleanup_only_64 {
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64"
        } else if cleanup_only {
            "j2k_htj2k_encode_codeblocks_multi_input_cleanup"
        } else {
            "j2k_htj2k_encode_codeblocks_multi_input"
        };

        let ((), ht_encode_us) =
            self.time_default_stream_named_us("j2k.htj2k.encode.codeblocks.multi_input", || {
                if cleanup_only_64 {
                    self.launch_htj2k_encode_codeblocks_multi_input_cleanup_64(
                        &CudaHtj2kEncodeMultiInputLaunch {
                            output: pooled_device_buffer(&output_buffer)?,
                            jobs: pooled_device_buffer(&jobs_buffer)?,
                            tables: resources.launch_tables(),
                            statuses: pooled_device_buffer(&status_buffer)?,
                            job_count: kernel_jobs.len(),
                        },
                    )
                } else if cleanup_only {
                    self.launch_htj2k_encode_codeblocks_multi_input_cleanup(
                        &CudaHtj2kEncodeMultiInputLaunch {
                            output: pooled_device_buffer(&output_buffer)?,
                            jobs: pooled_device_buffer(&jobs_buffer)?,
                            tables: resources.launch_tables(),
                            statuses: pooled_device_buffer(&status_buffer)?,
                            job_count: kernel_jobs.len(),
                        },
                    )
                } else {
                    self.launch_htj2k_encode_codeblocks_multi_input(
                        &CudaHtj2kEncodeMultiInputLaunch {
                            output: pooled_device_buffer(&output_buffer)?,
                            jobs: pooled_device_buffer(&jobs_buffer)?,
                            tables: resources.launch_tables(),
                            statuses: pooled_device_buffer(&status_buffer)?,
                            job_count: kernel_jobs.len(),
                        },
                    )
                }
            })?;
        let ((), status_readback_us) = self.time_default_stream_named_us(
            "j2k.htj2k.encode.codeblocks.multi_input.status_readback",
            || {
                status_buffer.copy_to_host(htj2k_encode_statuses_as_bytes_mut(&mut statuses))?;
                if let Some(status) = statuses.iter().copied().find(|status| !status.is_ok()) {
                    return Err(CudaError::KernelStatus {
                        kernel: status_kernel,
                        code: status.code,
                        detail: status.detail,
                    });
                }
                Ok(())
            },
        )?;

        let (compact_jobs, compact_output_bytes) =
            htj2k_encode_compact_jobs_multi_input(&statuses, kernel_jobs, &mut host_budget)?;
        let compact_output_buffer = pool.take(compact_output_bytes)?;
        let compact_dispatched = compact_output_bytes != 0;
        let compact_us = if compact_dispatched {
            let compact_jobs_buffer =
                pool.upload(htj2k_encode_compact_jobs_as_bytes(&compact_jobs))?;
            let ((), compact_us) = self.time_default_stream_named_us(
                "j2k.htj2k.encode.codeblocks.multi_input.compact",
                || {
                    self.launch_htj2k_compact_codeblocks(
                        pooled_device_buffer(&output_buffer)?,
                        pooled_device_buffer(&compact_output_buffer)?,
                        pooled_device_buffer(&compact_jobs_buffer)?,
                        compact_jobs.len(),
                    )
                },
            )?;
            compact_us
        } else {
            0
        };
        let (output, output_readback_us) = if compact_output_bytes == 0 {
            (Vec::new(), 0)
        } else {
            self.time_default_stream_named_us(
                "j2k.htj2k.encode.codeblocks.multi_input.output_readback",
                || {
                    copy_pooled_bytes_to_vec_uninit_with_budget(
                        &compact_output_buffer,
                        compact_output_bytes,
                        &mut host_budget,
                    )
                },
            )?
        };
        let mut code_blocks = host_budget.try_vec_with_capacity(statuses.len())?;
        for ((status, job), compact_job) in statuses
            .into_iter()
            .zip(kernel_jobs.iter())
            .zip(compact_jobs.iter())
        {
            let data_len = usize::try_from(status.data_len)
                .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
            if data_len > job.output_capacity as usize {
                return Err(CudaError::LengthTooLarge { len: data_len });
            }
            let start = compact_job.compact_offset as usize;
            let end = start
                .checked_add(data_len)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if end > compact_output_bytes {
                return Err(CudaError::LengthTooLarge { len: end });
            }
            code_blocks.push(CudaHtj2kCompactEncodedCodeBlock {
                payload_range: start..end,
                status,
                execution: CudaExecutionStats {
                    kernel_dispatches: 1,
                    copy_kernel_dispatches: usize::from(compact_dispatched),
                    decode_kernel_dispatches: 0,
                    hardware_decode: false,
                },
                stage_timings: CudaHtj2kEncodeStageTimings::default(),
            });
        }
        let stage_timings = CudaHtj2kEncodeStageTimings::from_parts(
            ht_encode_us,
            status_readback_us,
            compact_us,
            output_readback_us,
        );
        for block in &mut code_blocks {
            block.stage_timings = stage_timings;
        }
        let copy_kernel_dispatches = usize::from(!output.is_empty());

        Ok(CudaHtj2kCompactEncodedCodeBlocks {
            payload: output,
            code_blocks,
            execution: CudaExecutionStats {
                kernel_dispatches: 1,
                copy_kernel_dispatches,
                decode_kernel_dispatches: 0,
                hardware_decode: false,
            },
            stage_timings,
        })
    }
}
