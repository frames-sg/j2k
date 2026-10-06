// SPDX-License-Identifier: MIT OR Apache-2.0

use super::launch_geometry::{
    idwt_vertical_97_multi_launch_geometry, idwt_vertical_strip_launch_geometry,
};
use crate::{
    error::CudaError,
    j2k_decode::{
        idwt_launch::{generic_horizontal_idwt_route, generic_vertical_idwt_route},
        trace::MAX_COOPERATIVE_IDWT_DIMENSION,
    },
    kernels::{
        j2k_dwt53_launch_geometry, j2k_idwt_multi_coop_axis_launch_geometry,
        j2k_idwt_tiled_horizontal_block_count,
    },
};

use super::super::{
    idwt_batch_kernel_mode, types::CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtBatchKernelMode,
};

pub(super) const IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS: &str =
    "J2K IDWT geometry exceeds static CUDA launch limits";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::j2k_decode) struct IdwtBatchLaunchPlan {
    pub(in crate::j2k_decode) max_width: u32,
    pub(in crate::j2k_decode) max_height: u32,
    pub(in crate::j2k_decode) tiled_horizontal_blocks: usize,
    pub(in crate::j2k_decode) kernel_mode: CudaJ2kIdwtBatchKernelMode,
}

pub(super) fn validate_idwt_single_launch(width: u32, height: u32) -> Result<(), CudaError> {
    if width != 0 && height != 0 && j2k_dwt53_launch_geometry(width, height).is_none() {
        return Err(CudaError::InvalidArgument {
            message: format!("{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: single {width}x{height}"),
        });
    }
    Ok(())
}

pub(super) fn plan_idwt_batch_launch(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
) -> Result<Option<IdwtBatchLaunchPlan>, CudaError> {
    if kernel_jobs.is_empty() {
        return Ok(None);
    }
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
    let kernel_mode = idwt_batch_kernel_mode(kernel_jobs, max_width, max_height);
    let tiled_horizontal_blocks = kernel_jobs
        .iter()
        .try_fold(0usize, |maximum, kernel_job| {
            let width = kernel_job
                .job
                .rect
                .x1
                .saturating_sub(kernel_job.job.rect.x0) as usize;
            let height = kernel_job
                .job
                .rect
                .y1
                .saturating_sub(kernel_job.job.rect.y0) as usize;
            j2k_idwt_tiled_horizontal_block_count(width, height).map(|blocks| maximum.max(blocks))
        })
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    validate_idwt_batch_launch(
        max_width,
        max_height,
        tiled_horizontal_blocks,
        kernel_jobs.len(),
        kernel_mode,
    )?;
    Ok(Some(IdwtBatchLaunchPlan {
        max_width,
        max_height,
        tiled_horizontal_blocks,
        kernel_mode,
    }))
}

pub(super) fn validate_idwt_batch_launch(
    max_width: u32,
    max_height: u32,
    tiled_horizontal_blocks: usize,
    job_count: usize,
    kernel_mode: CudaJ2kIdwtBatchKernelMode,
) -> Result<(), CudaError> {
    if kernel_mode != CudaJ2kIdwtBatchKernelMode::Generic
        && (max_width > MAX_COOPERATIVE_IDWT_DIMENSION
            || max_height > MAX_COOPERATIVE_IDWT_DIMENSION)
    {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: batch jobs={job_count}, maximum={max_width}x{max_height}, mode={kernel_mode:?}"
            ),
        });
    }
    let horizontal = match kernel_mode {
        CudaJ2kIdwtBatchKernelMode::Generic => {
            generic_horizontal_idwt_route(
                max_width as usize,
                max_height as usize,
                tiled_horizontal_blocks,
                job_count,
            )
            .1
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative53 | CudaJ2kIdwtBatchKernelMode::Cooperative97 => {
            j2k_idwt_multi_coop_axis_launch_geometry(
                max_height as usize,
                max_width as usize,
                job_count,
            )
        }
    };
    let vertical = match kernel_mode {
        CudaJ2kIdwtBatchKernelMode::Generic => {
            generic_vertical_idwt_route(max_width as usize, max_height as usize, job_count).1
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative53 => {
            idwt_vertical_strip_launch_geometry(max_width as usize, max_height as usize, job_count)
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative97 => idwt_vertical_97_multi_launch_geometry(
            max_width as usize,
            max_height as usize,
            job_count,
        )
        .map(|pair| pair.1),
    };
    if horizontal.is_none() || vertical.is_none() {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "{IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS}: batch jobs={job_count}, maximum={max_width}x{max_height}, mode={kernel_mode:?}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
