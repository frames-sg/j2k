// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::try_vec_with_capacity, error::CudaError, memory::CudaDeviceBuffer};

use super::types::{CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtTarget};

pub(crate) fn active_dwt53_buffers<'a>(
    buffer_a: &'a CudaDeviceBuffer,
    buffer_b: &'a CudaDeviceBuffer,
    active_is_a: bool,
) -> (&'a CudaDeviceBuffer, &'a CudaDeviceBuffer) {
    if active_is_a {
        (buffer_a, buffer_b)
    } else {
        (buffer_b, buffer_a)
    }
}

pub(crate) fn j2k_idwt_multi_kernel_jobs(
    targets: &[CudaJ2kIdwtTarget<'_>],
) -> Result<Vec<CudaJ2kIdwtMultiKernelJob>, CudaError> {
    let mut kernel_jobs = try_vec_with_capacity(targets.len())?;
    append_j2k_idwt_multi_kernel_jobs(targets, &mut kernel_jobs)?;
    Ok(kernel_jobs)
}

pub(crate) fn append_j2k_idwt_multi_kernel_jobs(
    targets: &[CudaJ2kIdwtTarget<'_>],
    kernel_jobs: &mut Vec<CudaJ2kIdwtMultiKernelJob>,
) -> Result<(), CudaError> {
    for target in targets {
        if super::idwt::job_validation::validate_idwt_target(target)? {
            continue;
        }
        kernel_jobs.push(CudaJ2kIdwtMultiKernelJob {
            ll_ptr: target.ll.device_ptr(),
            hl_ptr: target.hl.device_ptr(),
            lh_ptr: target.lh.device_ptr(),
            hh_ptr: target.hh.device_ptr(),
            output_ptr: target.output.device_ptr(),
            job: target.job,
            reserved_tail: 0,
        });
    }
    Ok(())
}

pub(crate) fn checked_f32_words_byte_len(words: usize) -> Result<usize, CudaError> {
    words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::LengthTooLarge { len: words })
}
