// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::{ensure_context_ownership, CudaContext},
    error::CudaError,
    memory::CudaDeviceBuffer,
};

use super::CudaJ2kQuantizeSubbandRegionJob;

pub(super) const ENCODE_CONTEXT_MISMATCH: &str =
    "J2K encode input buffers must belong to the launch context";

pub(super) fn validate_encode_context_matches(
    matches_context: impl IntoIterator<Item = bool>,
) -> Result<(), CudaError> {
    ensure_context_ownership(matches_context, ENCODE_CONTEXT_MISMATCH)
}

pub(crate) fn validate_encode_buffer_context<'a>(
    context: &CudaContext,
    buffers: impl IntoIterator<Item = &'a CudaDeviceBuffer>,
) -> Result<(), CudaError> {
    validate_encode_context_matches(
        buffers
            .into_iter()
            .map(|buffer| buffer.is_owned_by(context)),
    )
}

pub(crate) fn validate_quantize_region(
    job: CudaJ2kQuantizeSubbandRegionJob,
    available_samples: usize,
) -> Result<(), CudaError> {
    if job.width == 0 || job.height == 0 {
        return Ok(());
    }
    if job.stride == 0
        || job
            .x0
            .checked_add(job.width)
            .is_none_or(|end_x| end_x > job.stride)
    {
        return Err(CudaError::LengthTooLarge {
            len: available_samples,
        });
    }

    let last_sample = (job.y0 as usize)
        .checked_add(job.height as usize - 1)
        .and_then(|row| row.checked_mul(job.stride as usize))
        .and_then(|row| row.checked_add(job.x0 as usize))
        .and_then(|row| row.checked_add(job.width as usize))
        .ok_or(CudaError::LengthTooLarge {
            len: available_samples,
        })?;
    if last_sample > available_samples {
        return Err(CudaError::OutputTooSmall {
            required: last_sample
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge { len: last_sample })?,
            have: available_samples
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(CudaError::LengthTooLarge {
                    len: available_samples,
                })?,
        });
    }
    Ok(())
}
