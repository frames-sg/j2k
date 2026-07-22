// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    cuda_error, profile, CudaBufferPool, CudaClassicCodeBlockJob, CudaClassicSegment,
    CudaCoefficientBand, CudaHtj2kDecodePlan, CudaPendingClassicBand, Error,
    CUDA_HTJ2K_KERNELS_NOT_READY, CUDA_HTJ2K_PLAN_INVARIANT_FAILED,
};
use crate::allocation::{checked_cuda_element_count, HostPhaseBudget};

pub(super) fn append_classic_subbands(
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
