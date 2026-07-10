// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::CudaJ2kIdwtMultiKernelJob, CudaJ2kIdwtBatchKernelMode, CudaJ2kIdwtBatchTraceRow,
};

pub(crate) fn idwt_batch_kernel_mode(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
) -> CudaJ2kIdwtBatchKernelMode {
    const MAX_COOPERATIVE_DIMENSION: u32 = 512;
    const MIN_COOPERATIVE_53_DIMENSION: u32 = 128;
    const MIN_COOPERATIVE_97_DIMENSION: u32 = 64;
    let bounded_cooperative_shape =
        max_width <= MAX_COOPERATIVE_DIMENSION && max_height <= MAX_COOPERATIVE_DIMENSION;
    if !bounded_cooperative_shape {
        return CudaJ2kIdwtBatchKernelMode::Generic;
    }
    if kernel_jobs.iter().all(|job| job.job.irreversible97 == 0) {
        if max_width >= MIN_COOPERATIVE_53_DIMENSION && max_height >= MIN_COOPERATIVE_53_DIMENSION {
            CudaJ2kIdwtBatchKernelMode::Cooperative53
        } else {
            CudaJ2kIdwtBatchKernelMode::Generic
        }
    } else if kernel_jobs.iter().all(|job| job.job.irreversible97 != 0) {
        if max_width >= MIN_COOPERATIVE_97_DIMENSION && max_height >= MIN_COOPERATIVE_97_DIMENSION {
            CudaJ2kIdwtBatchKernelMode::Cooperative97
        } else {
            CudaJ2kIdwtBatchKernelMode::Generic
        }
    } else {
        CudaJ2kIdwtBatchKernelMode::Generic
    }
}

pub(crate) fn idwt_batch_trace_row(
    stage_index: usize,
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
    mode: CudaJ2kIdwtBatchKernelMode,
    elapsed_us: u128,
) -> CudaJ2kIdwtBatchTraceRow {
    let mut min_width = u32::MAX;
    let mut min_height = u32::MAX;
    let mut total_pixels = 0u64;
    let mut irreversible_jobs = 0usize;
    for kernel_job in kernel_jobs {
        let width = kernel_job
            .job
            .rect
            .x1
            .saturating_sub(kernel_job.job.rect.x0);
        let height = kernel_job
            .job
            .rect
            .y1
            .saturating_sub(kernel_job.job.rect.y0);
        min_width = min_width.min(width);
        min_height = min_height.min(height);
        total_pixels =
            total_pixels.saturating_add(u64::from(width).saturating_mul(u64::from(height)));
        if kernel_job.job.irreversible97 != 0 {
            irreversible_jobs = irreversible_jobs.saturating_add(1);
        }
    }
    if kernel_jobs.is_empty() {
        min_width = 0;
        min_height = 0;
    }
    CudaJ2kIdwtBatchTraceRow {
        stage_index,
        mode,
        job_count: kernel_jobs.len(),
        max_width,
        max_height,
        min_width,
        min_height,
        total_pixels,
        irreversible_jobs,
        elapsed_us,
    }
}

pub(crate) fn format_idwt_batch_trace_row(row: CudaJ2kIdwtBatchTraceRow) -> String {
    format!(
        "j2k_profile codec=j2k op=cuda_idwt_batch path=decode \
         stage_index={} mode={:?} job_count={} max_width={} max_height={} \
         min_width={} min_height={} total_pixels={} irreversible_jobs={} elapsed_us={}",
        row.stage_index,
        row.mode,
        row.job_count,
        row.max_width,
        row.max_height,
        row.min_width,
        row.min_height,
        row.total_pixels,
        row.irreversible_jobs,
        row.elapsed_us
    )
}

#[cfg(test)]
pub(crate) fn idwt_batch_uses_cooperative_53(
    kernel_jobs: &[CudaJ2kIdwtMultiKernelJob],
    max_width: u32,
    max_height: u32,
) -> bool {
    idwt_batch_kernel_mode(kernel_jobs, max_width, max_height)
        == CudaJ2kIdwtBatchKernelMode::Cooperative53
}
