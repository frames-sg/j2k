// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::HostPhaseBudget, error::CudaError};

use super::super::types::{
    CudaHtj2kEncodeCompactJob, CudaHtj2kEncodeKernelJob, CudaHtj2kEncodeMultiInputKernelJob,
    CudaHtj2kEncodeStatus,
};

trait Htj2kCompactPlanJob {
    fn output_offset(&self) -> u32;
    fn output_capacity(&self) -> u32;
}

impl Htj2kCompactPlanJob for CudaHtj2kEncodeKernelJob {
    #[inline]
    fn output_offset(&self) -> u32 {
        self.output_offset
    }

    #[inline]
    fn output_capacity(&self) -> u32 {
        self.output_capacity
    }
}

impl Htj2kCompactPlanJob for CudaHtj2kEncodeMultiInputKernelJob {
    #[inline]
    fn output_offset(&self) -> u32 {
        self.output_offset
    }

    #[inline]
    fn output_capacity(&self) -> u32 {
        self.output_capacity
    }
}

pub(crate) fn htj2k_encode_compact_jobs(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[CudaHtj2kEncodeKernelJob],
    host_budget: &mut HostPhaseBudget,
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    htj2k_encode_compact_jobs_impl(statuses, kernel_jobs, host_budget)
}

pub(crate) fn htj2k_encode_compact_jobs_multi_input(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[CudaHtj2kEncodeMultiInputKernelJob],
    host_budget: &mut HostPhaseBudget,
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    htj2k_encode_compact_jobs_impl(statuses, kernel_jobs, host_budget)
}

fn htj2k_encode_compact_jobs_impl<J: Htj2kCompactPlanJob>(
    statuses: &[CudaHtj2kEncodeStatus],
    kernel_jobs: &[J],
    host_budget: &mut HostPhaseBudget,
) -> Result<(Vec<CudaHtj2kEncodeCompactJob>, usize), CudaError> {
    if statuses.len() != kernel_jobs.len() {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K encode status count does not match job count".to_string(),
        });
    }

    let mut compact_offset = 0usize;
    let mut compact_jobs = host_budget.try_vec_with_capacity(kernel_jobs.len())?;
    for (status, job) in statuses.iter().zip(kernel_jobs) {
        let data_len = usize::try_from(status.data_len)
            .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
        let output_offset = job.output_offset();
        let output_capacity = job.output_capacity();
        if data_len > output_capacity as usize {
            return Err(CudaError::LengthTooLarge { len: data_len });
        }
        let source_end = (output_offset as usize)
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let job_output_end = (output_offset as usize)
            .checked_add(output_capacity as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if source_end > job_output_end {
            return Err(CudaError::LengthTooLarge { len: source_end });
        }
        compact_jobs.push(CudaHtj2kEncodeCompactJob {
            source_offset: output_offset,
            compact_offset: u32::try_from(compact_offset).map_err(|_| {
                CudaError::LengthTooLarge {
                    len: compact_offset,
                }
            })?,
            data_len: status.data_len,
            reserved: status.reserved2,
        });
        compact_offset = compact_offset
            .checked_add(data_len)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        if compact_offset > u32::MAX as usize {
            return Err(CudaError::LengthTooLarge {
                len: compact_offset,
            });
        }
    }

    Ok((compact_jobs, compact_offset))
}
