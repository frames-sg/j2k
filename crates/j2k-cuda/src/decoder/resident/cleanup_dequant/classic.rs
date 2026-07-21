// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    cuda_error, CudaBufferPool, CudaClassicDecodeTarget, CudaComponentDecodeWork,
    CudaHtj2kDecodeResources, Error, CUDA_HTJ2K_KERNELS_NOT_READY,
};
use super::super::buffer_access::pooled_cuda_buffer;
use crate::allocation::HostPhaseBudget;

mod queued;
pub(in crate::decoder) use queued::{
    enqueue_component_classic_batches, QueuedComponentClassicDecode,
};

pub(in crate::decoder) fn run_component_classic_batches(
    context: &j2k_cuda_runtime::CudaContext,
    decode_resources: &CudaHtj2kDecodeResources,
    component_work: &mut [CudaComponentDecodeWork],
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    live_host_bytes: usize,
) -> Result<(), Error> {
    let pending_count = component_work
        .iter()
        .map(|work| work.pending_classic_bands.len())
        .sum::<usize>();
    if pending_count == 0 {
        return Ok(());
    }
    let accounting_index = component_work
        .iter()
        .position(|work| !work.pending_classic_bands.is_empty())
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    let mut budget =
        HostPhaseBudget::with_live_bytes("j2k CUDA classic Tier-1 target phase", live_host_bytes)?;
    let mut targets = budget.try_vec_with_capacity(pending_count)?;
    for work in component_work.iter() {
        for pending in &work.pending_classic_bands {
            targets.push(CudaClassicDecodeTarget {
                coefficients: pooled_cuda_buffer(&work.bands[pending.band_index].buffer)?,
                jobs: &pending.jobs,
                segments: &pending.segments,
                output_words: pending.output_words,
            });
        }
    }
    let (_, runtime_timings) = context
        .decode_classic_codeblocks_multi_with_resources_and_pool_timed(
            decode_resources,
            &targets,
            pool,
            budget.live_bytes(),
            collect_stage_timings,
        )
        .map_err(cuda_error)?;
    let accounting = &mut component_work[accounting_index];
    accounting.timings.h2d = accounting
        .timings
        .h2d
        .saturating_add(runtime_timings.job_upload_us)
        .saturating_add(runtime_timings.table_upload_us);
    accounting.timings.job_upload = accounting
        .timings
        .job_upload
        .saturating_add(runtime_timings.job_upload_us);
    accounting.timings.table_upload = accounting
        .timings
        .table_upload
        .saturating_add(runtime_timings.table_upload_us);
    accounting.timings.status_d2h = accounting
        .timings
        .status_d2h
        .saturating_add(runtime_timings.status_d2h_us);
    accounting.timings.classic_tier1 = accounting
        .timings
        .classic_tier1
        .saturating_add(runtime_timings.kernel_us);
    accounting.timings.classic_dispatch_count =
        accounting.timings.classic_dispatch_count.saturating_add(1);
    accounting.dispatches = accounting.dispatches.saturating_add(1);
    accounting.decode_dispatches = accounting.decode_dispatches.saturating_add(1);
    for work in component_work {
        work.pending_classic_bands.clear();
    }
    Ok(())
}
