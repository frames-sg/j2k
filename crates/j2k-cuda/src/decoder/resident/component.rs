// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    cuda_error, profile, CudaBufferPool, CudaClassicCodeBlockJob, CudaClassicSegment,
    CudaCoefficientBand, CudaComponentDecodeWork, CudaDecodeStageTimings, CudaDecodedComponent,
    CudaHtj2kCodeBlockJob, CudaHtj2kDecodePlan, CudaHtj2kDecodeResources,
    CudaHtj2kDecodeTableResources, CudaPendingClassicBand, CudaPendingDequantBand, Error,
    CUDA_HTJ2K_KERNELS_NOT_READY, CUDA_HTJ2K_PLAN_INVARIANT_FAILED, CUDA_HTJ2K_STORE_UNSUPPORTED,
};
use super::cleanup_dequant::run_component_cleanup_dequant_batches;
use super::idwt::run_cuda_component_idwt_steps;
use crate::allocation::{checked_cuda_element_count, host_allocation_error, HostPhaseBudget};

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn cuda_code_block_job_from_plan_block(
    block: &crate::CudaHtj2kCodeBlock,
    subband_width: u32,
) -> Result<CudaHtj2kCodeBlockJob, Error> {
    let output_offset = block
        .output_y
        .checked_mul(subband_width)
        .and_then(|base| base.checked_add(block.output_x))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })?;
    Ok(CudaHtj2kCodeBlockJob {
        payload_offset: block.payload_offset,
        width: block.width,
        height: block.height,
        payload_len: block.payload_len,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        missing_bit_planes: block.missing_bit_planes,
        num_bitplanes: block.num_bitplanes,
        number_of_coding_passes: block.number_of_coding_passes,
        output_stride: block.output_stride,
        output_offset,
        dequantization_step: block.dequantization_step,
        stripe_causal: block.stripe_causal != 0,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn decode_cuda_component_plan(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    tables: Option<&CudaHtj2kDecodeTableResources>,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
) -> Result<CudaDecodedComponent, Error> {
    let resource_upload_start = profile::profile_now(collect_stage_timings);
    let decode_resources = match tables {
        Some(tables) => context.upload_htj2k_decode_resources_with_tables(plan.payload(), tables),
        None => context.upload_j2k_decode_payload(plan.payload()),
    }
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
        .checked_add(plan.classic_subbands().len())
        .and_then(|count| count.checked_add(plan.idwt_steps().len()))
        .ok_or_else(|| {
            host_allocation_error::<CudaCoefficientBand>(
                usize::MAX,
                "j2k CUDA decoded coefficient bands",
            )
        })?;
    let mut bands = host_budget.try_vec_with_capacity(band_capacity)?;
    let mut pending_classic_bands =
        host_budget.try_vec_with_capacity(plan.classic_subbands().len())?;
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
        let output_words = checked_cuda_element_count(subband.width, subband.height).ok_or(
            Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            },
        )?;
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

    let classic_allocate_us = append_classic_subbands(
        context,
        plan,
        pool,
        collect_stage_timings,
        host_budget,
        &mut bands,
        &mut pending_classic_bands,
    )?;
    timings.h2d = timings.h2d.saturating_add(classic_allocate_us);

    let [store] = plan.store_steps() else {
        return Err(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_STORE_UNSUPPORTED,
        });
    };

    Ok(CudaComponentDecodeWork {
        bands,
        pending_classic_bands,
        pending_dequant_bands,
        store: *store,
        dispatches,
        decode_dispatches,
        timings,
    })
}

#[cfg(feature = "cuda-runtime")]
fn append_classic_subbands(
    context: &j2k_cuda_runtime::CudaContext,
    plan: &CudaHtj2kDecodePlan,
    pool: &CudaBufferPool,
    collect_stage_timings: bool,
    host_budget: &mut HostPhaseBudget,
    bands: &mut Vec<CudaCoefficientBand>,
    pending_bands: &mut Vec<CudaPendingClassicBand>,
) -> Result<u128, Error> {
    let mut allocate_us = 0u128;
    for subband in plan.classic_subbands() {
        let start = subband.code_block_start as usize;
        let end = start.checked_add(subband.code_block_count as usize).ok_or(
            Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            },
        )?;
        let code_blocks =
            plan.classic_code_blocks()
                .get(start..end)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
                })?;
        let segment_start = code_blocks
            .first()
            .map_or(0, |block| block.segment_start as usize);
        let segment_base =
            u32::try_from(segment_start).map_err(|_| Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
        let segment_end = code_blocks.last().map_or(segment_start, |block| {
            block.segment_start as usize + block.segment_count as usize
        });
        let plan_segments = plan
            .classic_segments()
            .get(segment_start..segment_end)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
        let segments = host_budget.try_collect_results_exact(
            plan_segments
                .iter()
                .map(|segment| Ok::<_, Error>(cuda_classic_segment_from_plan(segment))),
        )?;
        let jobs = host_budget.try_collect_results_exact(
            code_blocks
                .iter()
                .map(|block| cuda_classic_job_from_plan(block, subband.width, segment_base)),
        )?;
        let output_words = checked_cuda_element_count(subband.width, subband.height).ok_or(
            Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            },
        )?;
        let allocate_start = profile::profile_now(collect_stage_timings);
        let buffer = context
            .allocate_classic_coefficients_with_pool(output_words, pool)
            .map_err(cuda_error)?;
        allocate_us = allocate_us.saturating_add(profile::elapsed_us(allocate_start));
        let band_index = bands.len();
        bands.push(CudaCoefficientBand {
            band_id: subband.band_id,
            buffer,
        });
        if !jobs.is_empty() {
            pending_bands.push(CudaPendingClassicBand {
                band_index,
                jobs,
                segments,
                output_words,
            });
        }
    }
    Ok(allocate_us)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_classic_segment_from_plan(
    segment: &crate::direct_plan::CudaClassicSegment,
) -> CudaClassicSegment {
    CudaClassicSegment {
        data_offset: segment.data_offset,
        data_length: segment.data_length,
        start_coding_pass: u32::from(segment.start_coding_pass),
        end_coding_pass: u32::from(segment.end_coding_pass),
        use_arithmetic: segment.use_arithmetic,
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_classic_job_from_plan(
    block: &crate::direct_plan::CudaClassicCodeBlock,
    subband_width: u32,
    segment_base: u32,
) -> Result<CudaClassicCodeBlockJob, Error> {
    let output_offset = block
        .output_y
        .checked_mul(subband_width)
        .and_then(|base| base.checked_add(block.output_x))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
        })?;
    let segment_start =
        block
            .segment_start
            .checked_sub(segment_base)
            .ok_or(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
            })?;
    Ok(CudaClassicCodeBlockJob {
        payload_offset: block.payload_offset,
        payload_len: block.payload_len,
        segment_start,
        segment_count: block.segment_count,
        width: block.width,
        height: block.height,
        output_stride: block.output_stride,
        output_offset,
        missing_bitplanes: u32::from(block.missing_bit_planes),
        total_bitplanes: u32::from(block.total_bitplanes),
        number_of_coding_passes: u32::from(block.number_of_coding_passes),
        sub_band_type: u32::from(block.sub_band_type),
        style_flags: block.style_flags,
        strict: block.strict,
        dequantization_step: block.dequantization_step,
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
