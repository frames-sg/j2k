// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    cuda_error, profile, CudaBufferPool, CudaCoefficientBand, CudaComponentDecodeWork,
    CudaDecodeStageTimings, CudaDecodedComponent, CudaHtj2kDecodePlan, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeTableResources, CudaPendingDequantBand, Error, CUDA_HTJ2K_KERNELS_NOT_READY,
    CUDA_HTJ2K_PLAN_INVARIANT_FAILED, CUDA_HTJ2K_STORE_UNSUPPORTED,
};
use super::cleanup_dequant::run_component_cleanup_dequant_batches;
use super::helpers::{checked_area, cuda_code_block_job_from_plan_block};
use super::idwt::run_cuda_component_idwt_steps;
use crate::allocation::{host_allocation_error, HostPhaseBudget};

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_cuda_component_plan(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    tables: &CudaHtj2kDecodeTableResources,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let resource_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = context
        .upload_htj2k_decode_resources_with_tables(plan.payload(), tables)
        .map_err(cuda_error)?;
    let resource_upload_us = profile::elapsed_us(resource_upload_start);
    let mut component = decode_cuda_component_plan_with_resources(
        context,
        plan,
        &decode_resources,
        pool,
        collect_stage_timings,
    )?;
    component.timings.h2d = component.timings.h2d.saturating_add(resource_upload_us);
    component.timings.payload_upload = component
        .timings
        .payload_upload
        .saturating_add(resource_upload_us);
    Ok(component)
}

#[cfg(feature = "cuda-runtime")]
fn decode_cuda_component_plan_with_resources(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    decode_resources: &CudaHtj2kDecodeResources,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let mut host_budget = HostPhaseBudget::new("j2k CUDA component decode owner graph");
    plan.account_host_owners(&mut host_budget)?;
    let mut work = decode_cuda_component_subbands_with_resources(
        context,
        plan,
        pool,
        collect_stage_timings,
        &mut host_budget,
    )?;
    run_component_cleanup_dequant_batches(
        context,
        decode_resources,
        std::slice::from_mut(&mut work),
        pool,
        collect_stage_timings,
        host_budget.live_bytes(),
    )?;
    run_cuda_component_idwt_steps(
        context,
        plan.idwt_steps(),
        &mut work,
        pool,
        collect_stage_timings,
    )?;
    finish_cuda_component_decode(work)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn decode_cuda_component_subbands_with_resources(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    host_budget: &mut HostPhaseBudget,
) -> Result<CudaComponentDecodeWork, Error> {
    let band_capacity = plan
        .subbands()
        .len()
        .checked_add(plan.idwt_steps().len())
        .ok_or_else(|| {
            host_allocation_error::<CudaCoefficientBand>(
                usize::MAX,
                "j2k CUDA decoded coefficient bands",
            )
        })?;
    let mut bands = host_budget.try_vec_with_capacity(band_capacity)?;
    let mut pending_dequant_bands = host_budget.try_vec_with_capacity(plan.subbands().len())?;
    let dispatches = 0usize;
    let decode_dispatches = 0usize;
    let mut timings = CudaDecodeStageTimings::default();

    for subband in plan.subbands() {
        let start = subband.code_block_start as usize;
        let end = start.checked_add(subband.code_block_count as usize).ok_or(
            Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            },
        )?;
        let code_blocks =
            plan.code_blocks()
                .get(start..end)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
                })?;
        let jobs = host_budget.try_collect_results_exact(
            code_blocks
                .iter()
                .map(|block| cuda_code_block_job_from_plan_block(block, subband.width)),
        )?;
        let output_words = checked_area(subband.width, subband.height)?;
        let allocate_start = profile::profile_now(collect_stage_timings);
        let output = context
            .allocate_htj2k_codeblock_coefficients_with_pool(&jobs, output_words, pool)
            .map_err(cuda_error)?;
        let allocate_wall_us = profile::elapsed_us(allocate_start);
        timings.h2d = timings.h2d.saturating_add(allocate_wall_us);
        let (buffer, _, _) = output.into_parts();
        let band_index = bands.len();
        bands.push(CudaCoefficientBand {
            band_id: subband.band_id,
            buffer,
        });
        if !jobs.is_empty() {
            pending_dequant_bands.push(CudaPendingDequantBand {
                band_index,
                jobs,
                output_words,
            });
        }
    }

    let [store] = plan.store_steps() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_STORE_UNSUPPORTED,
        });
    };

    Ok(CudaComponentDecodeWork {
        bands,
        pending_dequant_bands,
        store: *store,
        dispatches,
        decode_dispatches,
        timings,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn finish_cuda_component_decode(
    mut work: CudaComponentDecodeWork,
) -> Result<CudaDecodedComponent, Error> {
    let input_index = work
        .bands
        .iter()
        .position(|band| band.band_id == work.store.input_band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    let input = work.bands.swap_remove(input_index);
    Ok(CudaDecodedComponent {
        buffer: input.buffer,
        store: work.store,
        dispatches: work.dispatches,
        decode_dispatches: work.decode_dispatches,
        timings: work.timings,
    })
}
