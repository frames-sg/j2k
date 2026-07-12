// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    kernels::{
        j2k_dwt53_launch_geometry, j2k_idwt_multi_1d_launch_geometry,
        j2k_idwt_multi_coop_axis_launch_geometry, j2k_idwt_multi_coop_columns_launch_geometry,
        j2k_idwt_multi_coop_launch_geometry, CudaKernel, CudaLaunchGeometry,
    },
};

use super::super::{
    idwt_batch_kernel_mode, types::CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtBatchKernelMode,
};

pub(super) const IDWT_LAUNCH_GEOMETRY_EXCEEDS_LIMITS: &str =
    "J2K IDWT geometry exceeds static CUDA launch limits";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct IdwtBatchLaunchPlan {
    pub(super) max_width: u32,
    pub(super) max_height: u32,
    pub(super) kernel_mode: CudaJ2kIdwtBatchKernelMode,
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
    validate_idwt_batch_launch(max_width, max_height, kernel_jobs.len(), kernel_mode)?;
    Ok(Some(IdwtBatchLaunchPlan {
        max_width,
        max_height,
        kernel_mode,
    }))
}

pub(super) fn validate_idwt_batch_launch(
    max_width: u32,
    max_height: u32,
    job_count: usize,
    kernel_mode: CudaJ2kIdwtBatchKernelMode,
) -> Result<(), CudaError> {
    let horizontal = match kernel_mode {
        CudaJ2kIdwtBatchKernelMode::Generic => {
            j2k_idwt_multi_1d_launch_geometry(max_height as usize, job_count)
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative53 => {
            j2k_idwt_multi_coop_launch_geometry(max_height as usize, job_count)
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative97 => j2k_idwt_multi_coop_axis_launch_geometry(
            max_height as usize,
            max_width as usize,
            job_count,
        ),
    };
    let vertical = match kernel_mode {
        CudaJ2kIdwtBatchKernelMode::Generic => {
            j2k_idwt_multi_1d_launch_geometry(max_width as usize, job_count)
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative53 => {
            j2k_idwt_multi_coop_launch_geometry(max_width as usize, job_count)
        }
        CudaJ2kIdwtBatchKernelMode::Cooperative97 => idwt_vertical_97_multi_launch_geometry(
            max_width as usize,
            max_height as usize,
            job_count,
        )
        .map(|(_, geometry)| geometry),
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

pub(in crate::j2k_decode) fn idwt_vertical_97_multi_launch_geometry(
    max_columns: usize,
    max_height: usize,
    job_count: usize,
) -> Option<(CudaKernel, CudaLaunchGeometry)> {
    const COLUMNS_PER_BLOCK: usize = 4;
    const MIN_COLS4_JOBS: usize = 64;
    if job_count >= MIN_COLS4_JOBS && max_height <= 256 {
        let geometry = j2k_idwt_multi_coop_columns_launch_geometry(
            max_columns,
            max_height,
            job_count,
            COLUMNS_PER_BLOCK,
        )?;
        Some((CudaKernel::J2kIdwtVertical97MultiCols4, geometry))
    } else {
        let geometry =
            j2k_idwt_multi_coop_axis_launch_geometry(max_columns, max_height, job_count)?;
        Some((CudaKernel::J2kIdwtVertical97Multi, geometry))
    }
}

#[cfg(test)]
mod tests;
