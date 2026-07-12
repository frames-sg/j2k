// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaHtj2kCodeBlockJob;
use crate::{allocation::HostPhaseBudget, error::CudaError};
use sweep::{validate_disjoint_output_regions, Htj2kOutputRect, Htj2kOutputRegion};

mod sweep;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ValidatedHtj2kOutputLayout {
    pub(super) output_bytes: usize,
    pub(super) needs_zero_fill: bool,
}

fn output_rect(
    job: &CudaHtj2kCodeBlockJob,
    output_words: usize,
) -> Result<Option<Htj2kOutputRegion>, CudaError> {
    let output_offset = job.output_offset as usize;
    if job.width == 0 || job.height == 0 {
        if output_offset > output_words {
            return Err(CudaError::LengthTooLarge { len: output_words });
        }
        return Ok(None);
    }

    let output_stride = job.output_stride as usize;
    let width = job.width as usize;
    let height = job.height as usize;
    if output_stride == 0 || width > output_stride {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K output rows require a nonzero stride at least as wide as the block"
                .to_string(),
        });
    }
    let column_start = output_offset % output_stride;
    let column_end = column_start
        .checked_add(width)
        .ok_or(CudaError::LengthTooLarge { len: output_words })?;
    if column_end > output_stride {
        return Err(CudaError::InvalidArgument {
            message: "HTJ2K output block crosses its row stride".to_string(),
        });
    }
    let row_start = output_offset / output_stride;
    let row_end = row_start
        .checked_add(height)
        .ok_or(CudaError::LengthTooLarge { len: output_words })?;
    let output_end = output_offset
        .checked_add(
            output_stride
                .checked_mul(height - 1)
                .ok_or(CudaError::LengthTooLarge { len: output_words })?,
        )
        .and_then(|last_row| last_row.checked_add(width))
        .ok_or(CudaError::LengthTooLarge { len: output_words })?;
    if output_end > output_words {
        return Err(CudaError::LengthTooLarge { len: output_words });
    }
    Ok(Some(Htj2kOutputRegion {
        stride: output_stride,
        rect: Htj2kOutputRect {
            row_start,
            row_end,
            column_start,
            column_end,
        },
        linear_start: output_offset,
        linear_end: output_end,
    }))
}

pub(super) fn validate_disjoint_htj2k_job_outputs_with_live_bytes(
    jobs: &[CudaHtj2kCodeBlockJob],
    output_words: usize,
    live_host_bytes: usize,
) -> Result<(), CudaError> {
    let mut host_budget =
        HostPhaseBudget::with_live_bytes("CUDA HTJ2K output-region validation", live_host_bytes)?;
    let mut regions = host_budget.try_vec_with_capacity(jobs.len())?;
    for job in jobs {
        let Some(region) = output_rect(job, output_words)? else {
            continue;
        };
        regions.push(region);
    }
    validate_disjoint_output_regions(&mut regions, host_budget.live_bytes())
}

pub(super) fn validate_htj2k_output_layout(
    jobs: &[CudaHtj2kCodeBlockJob],
    output_words: usize,
) -> Result<ValidatedHtj2kOutputLayout, CudaError> {
    validate_htj2k_output_layout_with_live_bytes(jobs, output_words, 0)
}

pub(super) fn validate_htj2k_output_layout_with_live_bytes(
    jobs: &[CudaHtj2kCodeBlockJob],
    output_words: usize,
    live_host_bytes: usize,
) -> Result<ValidatedHtj2kOutputLayout, CudaError> {
    validate_disjoint_htj2k_job_outputs_with_live_bytes(jobs, output_words, live_host_bytes)?;
    let covered_words = jobs.iter().try_fold(0usize, |covered, job| {
        let area = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        covered
            .checked_add(area)
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })
    })?;
    let output_bytes = output_words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(CudaError::LengthTooLarge { len: output_words })?;
    Ok(ValidatedHtj2kOutputLayout {
        output_bytes,
        needs_zero_fill: covered_words != output_words,
    })
}

#[cfg(test)]
mod tests;
