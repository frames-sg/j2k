// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::HostPhaseBudget, error::CudaError, kernels::CudaKernel};

use super::{
    output_regions::{validate_htj2k_output_layout, validate_htj2k_output_layout_with_live_bytes},
    types::{
        CudaHtj2kCleanupMultiKernelJob, CudaHtj2kCleanupTarget, CudaHtj2kCodeBlockJob,
        CudaHtj2kCodeBlockKernelJob, CudaHtj2kDequantizeKernelJob, CudaHtj2kDequantizeTarget,
        ValidatedHtj2kKernelJobs,
    },
};

pub(super) fn htj2k_kernel_jobs(
    jobs: &[CudaHtj2kCodeBlockJob],
    payload_len: usize,
    output_words: usize,
) -> Result<ValidatedHtj2kKernelJobs, CudaError> {
    let mut host_budget = HostPhaseBudget::new("CUDA HTJ2K kernel-job validation");
    htj2k_kernel_jobs_with_budget(jobs, payload_len, output_words, &mut host_budget)
}

fn htj2k_kernel_jobs_with_budget(
    jobs: &[CudaHtj2kCodeBlockJob],
    payload_len: usize,
    output_words: usize,
    host_budget: &mut HostPhaseBudget,
) -> Result<ValidatedHtj2kKernelJobs, CudaError> {
    let output_layout =
        validate_htj2k_output_layout_with_live_bytes(jobs, output_words, host_budget.live_bytes())?;
    let mut kernel_jobs = host_budget.try_vec_with_capacity(jobs.len())?;
    for job in jobs {
        let kernel_job = (|| {
            let payload_offset = usize::try_from(job.payload_offset)
                .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
            let payload_end = payload_offset
                .checked_add(job.payload_len as usize)
                .ok_or(CudaError::LengthTooLarge { len: payload_len })?;
            let expected_payload_len = job
                .cleanup_length
                .checked_add(job.refinement_length)
                .ok_or(CudaError::LengthTooLarge {
                    len: job.payload_len as usize,
                })?;
            if payload_end > payload_len || expected_payload_len != job.payload_len {
                return Err(CudaError::LengthTooLarge {
                    len: payload_len.max(output_words),
                });
            }
            Ok(CudaHtj2kCodeBlockKernelJob {
                coded_offset: u32::try_from(payload_offset)
                    .map_err(|_| CudaError::LengthTooLarge { len: payload_len })?,
                width: job.width,
                height: job.height,
                coded_len: job.payload_len,
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                missing_msbs: u32::from(job.missing_bit_planes),
                num_bitplanes: u32::from(job.num_bitplanes),
                number_of_coding_passes: u32::from(job.number_of_coding_passes),
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                dequantization_step: job.dequantization_step,
                stripe_causal: u32::from(job.stripe_causal),
            })
        })()?;
        kernel_jobs.push(kernel_job);
    }
    Ok(ValidatedHtj2kKernelJobs {
        jobs: kernel_jobs,
        output_layout,
    })
}

pub(crate) fn htj2k_dequantize_kernel_jobs_with_live_host_bytes(
    targets: &[CudaHtj2kDequantizeTarget<'_>],
    live_host_bytes: usize,
) -> Result<Vec<CudaHtj2kDequantizeKernelJob>, CudaError> {
    let total_jobs = targets
        .iter()
        .try_fold(0usize, |count, target| count.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K dequantization kernel jobs", live_host_bytes)?;
    let mut kernel_jobs = host_budget.try_vec_with_capacity(total_jobs)?;
    for target in targets {
        let output_layout = validate_htj2k_output_layout_with_live_bytes(
            target.jobs,
            target.output_words,
            host_budget.live_bytes(),
        )?;
        if output_layout.output_bytes > target.coefficients.byte_len() {
            return Err(CudaError::LengthTooLarge {
                len: output_layout.output_bytes,
            });
        }
        for job in target.jobs {
            kernel_jobs.push(CudaHtj2kDequantizeKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                width: job.width,
                height: job.height,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                num_bitplanes: u32::from(job.num_bitplanes),
                reserved: 0,
                dequantization_step: job.dequantization_step,
                reserved_tail: 0,
            });
        }
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_cleanup_multi_kernel_jobs_with_live_host_bytes(
    targets: &[CudaHtj2kCleanupTarget<'_>],
    payload_len: usize,
    live_host_bytes: usize,
) -> Result<Vec<CudaHtj2kCleanupMultiKernelJob>, CudaError> {
    let total_jobs = targets
        .iter()
        .try_fold(0usize, |count, target| count.checked_add(target.jobs.len()))
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K cleanup kernel jobs", live_host_bytes)?;
    let mut kernel_jobs = host_budget.try_vec_with_capacity(total_jobs)?;
    for target in targets {
        let output_bytes = target
            .output_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge {
                len: target.output_words,
            })?;
        if output_bytes > target.coefficients.byte_len() {
            return Err(CudaError::LengthTooLarge { len: output_bytes });
        }
        let mut target_budget = HostPhaseBudget::with_live_bytes(
            "CUDA HTJ2K cleanup target kernel jobs",
            host_budget.live_bytes(),
        )?;
        for job in htj2k_kernel_jobs_with_budget(
            target.jobs,
            payload_len,
            target.output_words,
            &mut target_budget,
        )?
        .jobs
        {
            kernel_jobs.push(CudaHtj2kCleanupMultiKernelJob {
                output_ptr: target.coefficients.device_ptr(),
                coded_offset: job.coded_offset,
                width: job.width,
                height: job.height,
                coded_len: job.coded_len,
                cleanup_length: job.cleanup_length,
                refinement_length: job.refinement_length,
                missing_msbs: job.missing_msbs,
                num_bitplanes: job.num_bitplanes,
                number_of_coding_passes: job.number_of_coding_passes,
                output_stride: job.output_stride,
                output_offset: job.output_offset,
                dequantization_step: job.dequantization_step,
                stripe_causal: job.stripe_causal,
                reserved_tail: 0,
            });
        }
    }
    Ok(kernel_jobs)
}

pub(crate) fn htj2k_decode_multi_kernel_for_jobs(
    jobs: &[CudaHtj2kCleanupMultiKernelJob],
) -> (CudaKernel, &'static str) {
    let cleanup_only = jobs
        .iter()
        .all(|job| job.refinement_length == 0 && job.number_of_coding_passes <= 1);
    if cleanup_only {
        (
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
            "j2k_htj2k_decode_codeblocks_multi_cleanup_only",
        )
    } else {
        (
            CudaKernel::Htj2kDecodeCodeblocksMulti,
            "j2k_htj2k_decode_codeblocks_multi",
        )
    }
}

pub(crate) fn htj2k_decode_multi_cleanup_dequant_kernel_for_jobs(
    jobs: &[CudaHtj2kCleanupMultiKernelJob],
) -> Option<(CudaKernel, &'static str)> {
    let cleanup_only = jobs
        .iter()
        .all(|job| job.refinement_length == 0 && job.number_of_coding_passes <= 1);
    cleanup_only.then_some((
        CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
        "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize",
    ))
}

pub(crate) fn htj2k_decode_needs_zero_fill(
    jobs: &[CudaHtj2kCodeBlockJob],
    output_words: usize,
) -> Result<bool, CudaError> {
    validate_htj2k_output_layout(jobs, output_words).map(|layout| layout.needs_zero_fill)
}
