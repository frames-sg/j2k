// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::HostPhaseBudget, error::CudaError, memory::checked_image_words};

use super::types::{
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeKernelJob,
    CudaHtj2kEncodeMultiInputKernelJob, CudaHtj2kEncodeResidentTarget,
    HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES, HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH,
};

mod compact;

pub(crate) use compact::{htj2k_encode_compact_jobs, htj2k_encode_compact_jobs_multi_input};

pub(crate) const HTJ2K_ENCODE_OUTPUT_CAPACITY: usize = 24 * 1024;

pub(crate) fn htj2k_encode_kernel_jobs_with_live_host_bytes(
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
    coefficient_words: usize,
    live_host_bytes: usize,
) -> Result<Vec<CudaHtj2kEncodeKernelJob>, CudaError> {
    let mut output_offset = 0usize;
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K encode kernel jobs", live_host_bytes)?;
    let mut kernel_jobs = host_budget.try_vec_with_capacity(jobs.len())?;
    for job in jobs {
        validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
        let coefficient_offset = job.coefficient_offset as usize;
        let coefficient_len = checked_image_words(job.width, job.height, 1)?;
        let coefficient_end =
            coefficient_offset
                .checked_add(coefficient_len)
                .ok_or(CudaError::LengthTooLarge {
                    len: coefficient_words,
                })?;
        if coefficient_end > coefficient_words {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_end,
            });
        }

        let output_end = output_offset
            .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if output_end > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: output_end });
        }
        kernel_jobs.push(CudaHtj2kEncodeKernelJob {
            coefficient_offset: job.coefficient_offset,
            coefficient_stride: job.width,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: u32::from(job.target_coding_passes),
        });
        output_offset = output_end;
    }
    Ok(kernel_jobs)
}

pub(super) fn htj2k_encode_multi_input_kernel_jobs_with_live_host_bytes(
    targets: &[CudaHtj2kEncodeResidentTarget<'_>],
    live_host_bytes: usize,
) -> Result<Vec<CudaHtj2kEncodeMultiInputKernelJob>, CudaError> {
    let job_count = targets
        .iter()
        .try_fold(0usize, |sum, target| sum.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut output_offset = 0usize;
    let mut host_budget = HostPhaseBudget::with_live_bytes(
        "CUDA HTJ2K multi-input encode kernel jobs",
        live_host_bytes,
    )?;
    let mut kernel_jobs = host_budget.try_vec_with_capacity(job_count)?;
    for target in targets {
        let available_coefficients = target.coefficients.typed_view::<i32>()?.len();
        if available_coefficients < target.coefficient_count {
            return Err(CudaError::OutputTooSmall {
                required: target
                    .coefficient_count
                    .checked_mul(std::mem::size_of::<i32>())
                    .ok_or(CudaError::LengthTooLarge {
                        len: target.coefficient_count,
                    })?,
                have: target.coefficients.byte_len(),
            });
        }
        for job in target.jobs {
            validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
            let coefficient_offset = job.coefficient_offset as usize;
            let coefficient_len = checked_image_words(job.width, job.height, 1)?;
            let coefficient_end = coefficient_offset.checked_add(coefficient_len).ok_or(
                CudaError::LengthTooLarge {
                    len: target.coefficient_count,
                },
            )?;
            if coefficient_end > target.coefficient_count {
                return Err(CudaError::LengthTooLarge {
                    len: coefficient_end,
                });
            }

            let output_end = output_offset
                .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
                .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
            if output_end > u32::MAX as usize {
                return Err(CudaError::LengthTooLarge { len: output_end });
            }
            kernel_jobs.push(CudaHtj2kEncodeMultiInputKernelJob {
                coefficient_ptr: target.coefficients.device_ptr(),
                coefficient_offset: job.coefficient_offset,
                coefficient_stride: job.width,
                width: job.width,
                height: job.height,
                total_bitplanes: u32::from(job.total_bitplanes),
                output_offset: u32::try_from(output_offset)
                    .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
                output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                    CudaError::LengthTooLarge {
                        len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                    }
                })?,
                target_coding_passes: u32::from(job.target_coding_passes),
            });
            output_offset = output_end;
        }
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_encode_region_kernel_jobs_with_live_host_bytes(
    jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
    coefficient_words: usize,
    live_host_bytes: usize,
) -> Result<Vec<CudaHtj2kEncodeKernelJob>, CudaError> {
    let mut output_offset = 0usize;
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K region encode kernel jobs", live_host_bytes)?;
    let mut kernel_jobs = host_budget.try_vec_with_capacity(jobs.len())?;
    for job in jobs {
        validate_htj2k_encode_codeblock_shape(job.width, job.height)?;
        if job.width == 0 || job.height == 0 || job.coefficient_stride < job.width {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_words,
            });
        }
        let row_offset = (job.height as usize - 1)
            .checked_mul(job.coefficient_stride as usize)
            .ok_or(CudaError::LengthTooLarge {
                len: coefficient_words,
            })?;
        let coefficient_end = job
            .coefficient_offset
            .try_into()
            .ok()
            .and_then(|offset: usize| offset.checked_add(row_offset))
            .and_then(|offset| offset.checked_add(job.width as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: coefficient_words,
            })?;
        if coefficient_end > coefficient_words {
            return Err(CudaError::LengthTooLarge {
                len: coefficient_end,
            });
        }

        let output_end = output_offset
            .checked_add(HTJ2K_ENCODE_OUTPUT_CAPACITY)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if output_end > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge { len: output_end });
        }
        kernel_jobs.push(CudaHtj2kEncodeKernelJob {
            coefficient_offset: job.coefficient_offset,
            coefficient_stride: job.coefficient_stride,
            width: job.width,
            height: job.height,
            total_bitplanes: u32::from(job.total_bitplanes),
            output_offset: u32::try_from(output_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: output_offset })?,
            output_capacity: u32::try_from(HTJ2K_ENCODE_OUTPUT_CAPACITY).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: HTJ2K_ENCODE_OUTPUT_CAPACITY,
                }
            })?,
            target_coding_passes: u32::from(job.target_coding_passes),
        });
        output_offset = output_end;
    }
    Ok(kernel_jobs)
}

pub(crate) fn validate_htj2k_encode_codeblock_shape(
    width: u32,
    height: u32,
) -> Result<(), CudaError> {
    let samples = usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    if width == 0
        || height == 0
        || width > HTJ2K_ENCODE_MAX_CODEBLOCK_WIDTH
        || samples > HTJ2K_ENCODE_MAX_CODEBLOCK_SAMPLES
    {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode code-block dimensions exceed CUDA kernel limits".to_string(),
        });
    }
    Ok(())
}
