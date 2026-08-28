// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodedHtJ2kCodeBlock, EncodedHtJ2kCodeBlockSet, J2kEncodeStageError, J2kHtCodeBlockEncodeJob,
    J2kHtCodeBlockSetEncodeJob, J2kHtSubbandEncodeJob, J2kResidentHtj2kTileEncodeJob,
};
use j2k_cuda_j2k_engine::{
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeResources,
    CudaHtj2kEncodeTables, CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob,
};
use j2k_cuda_runtime::{CudaContext, CudaDeviceBuffer};

use crate::allocation::{try_vec_push, try_vec_with_capacity, HostPhaseBudget};
use crate::encode::stage_error::{arithmetic_overflow, runtime_error, CudaStageResult};

use super::super::{time_cuda_stage, CudaEncodeStageTimings};
use super::htj2k_allocation_error;
use super::types::{
    CudaEncodedHtSubband, CudaEncodedHtj2kSubband, CudaHtj2kEncodeRuntime,
    CudaHtj2kTileEncodeStats, CudaTileSubbandKind, CudaTileSubbandRegion,
};

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_ht_code_block(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> CudaStageResult<j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlocks> {
    let coefficient_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block encode coefficient count"))?;
    if coefficient_len != job.coefficients.len() {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K code-block encode job has invalid coefficient length",
        ));
    }
    let cuda_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
        target_coding_passes: job.target_coding_passes,
    }];
    j2k_cuda_j2k_engine::J2kCudaEngine::new(context)
        .encode_htj2k_codeblocks_with_resources(job.coefficients, &cuda_jobs, resources)
        .map_err(|error| runtime_error("encode CUDA HTJ2K code block", error))
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_ht_code_blocks(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> CudaStageResult<j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlocks> {
    let total_coefficients = jobs.iter().try_fold(0usize, |acc, job| {
        let coefficient_len = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block batch coefficient count"))?;
        if coefficient_len != job.coefficients.len() {
            return Err(J2kEncodeStageError::invalid_request(
                "CUDA HTJ2K code-block encode job has invalid coefficient length",
            ));
        }
        acc.checked_add(coefficient_len)
            .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block batch coefficient count"))
    })?;
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K batch staging");
    let mut coefficients = host_budget
        .try_vec_with_capacity(total_coefficients)
        .map_err(htj2k_allocation_error)?;
    let mut cuda_jobs = host_budget
        .try_vec_with_capacity(jobs.len())
        .map_err(htj2k_allocation_error)?;
    for job in jobs {
        let coefficient_offset = u32::try_from(coefficients.len())
            .map_err(|_| arithmetic_overflow("CUDA HTJ2K code-block batch coefficient offset"))?;
        host_budget
            .try_vec_extend_from_slice(&mut coefficients, job.coefficients)
            .map_err(htj2k_allocation_error)?;
        host_budget
            .try_vec_push(
                &mut cuda_jobs,
                CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset,
                    width: job.width,
                    height: job.height,
                    total_bitplanes: job.total_bitplanes,
                    target_coding_passes: job.target_coding_passes,
                },
            )
            .map_err(htj2k_allocation_error)?;
    }

    j2k_cuda_j2k_engine::J2kCudaEngine::new(context)
        .encode_htj2k_codeblocks_with_resources_and_live_host_bytes(
            &coefficients,
            &cuda_jobs,
            resources,
            host_budget.live_bytes(),
        )
        .map_err(|error| runtime_error("encode CUDA HTJ2K code-block batch", error))
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_ht_code_block_sets(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    jobs: &[J2kHtCodeBlockSetEncodeJob<'_>],
) -> CudaStageResult<Option<j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlocks>> {
    let total_coefficients = jobs.iter().try_fold(0usize, |acc, job| {
        let coefficient_len = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or_else(|| arithmetic_overflow("CUDA HT candidate coefficient count"))?;
        if coefficient_len != job.coefficients.len() {
            return Err(J2kEncodeStageError::invalid_request(
                "CUDA HT candidate job has invalid coefficient length",
            ));
        }
        acc.checked_add(coefficient_len)
            .ok_or_else(|| arithmetic_overflow("CUDA HT candidate coefficient count"))
    })?;
    if jobs.iter().any(|job| {
        !matches!(
            (job.cleanup_bitplane, job.target_coding_passes),
            (0, 1) | (1 | 2, 3)
        )
    }) {
        return Ok(None);
    }
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HT candidate staging");
    let mut coefficients = host_budget
        .try_vec_with_capacity(total_coefficients)
        .map_err(htj2k_allocation_error)?;
    let mut cuda_jobs = host_budget
        .try_vec_with_capacity(jobs.len())
        .map_err(htj2k_allocation_error)?;
    for job in jobs {
        let coefficient_offset = u32::try_from(coefficients.len())
            .map_err(|_| arithmetic_overflow("CUDA HT candidate coefficient offset"))?;
        let shift = job
            .target_coding_passes
            .saturating_sub(1)
            .saturating_sub(job.cleanup_bitplane);
        let kernel_total_bitplanes = job
            .total_bitplanes
            .checked_add(shift)
            .filter(|total| *total <= 31);
        let Some(kernel_total_bitplanes) = kernel_total_bitplanes else {
            return Ok(None);
        };
        for &coefficient in job.coefficients {
            let Some(coefficient) = coefficient.checked_shl(u32::from(shift)) else {
                return Ok(None);
            };
            host_budget
                .try_vec_push(&mut coefficients, coefficient)
                .map_err(htj2k_allocation_error)?;
        }
        host_budget
            .try_vec_push(
                &mut cuda_jobs,
                CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset,
                    width: job.width,
                    height: job.height,
                    total_bitplanes: kernel_total_bitplanes,
                    target_coding_passes: job.target_coding_passes,
                },
            )
            .map_err(htj2k_allocation_error)?;
    }
    j2k_cuda_j2k_engine::J2kCudaEngine::new(context)
        .encode_htj2k_codeblocks_with_resources_and_live_host_bytes(
            &coefficients,
            &cuda_jobs,
            resources,
            host_budget.live_bytes(),
        )
        .map(Some)
        .map_err(|error| runtime_error("encode CUDA HT candidate sets", error))
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_ht_region_jobs(
    width: u32,
    height: u32,
    code_block_width: u32,
    code_block_height: u32,
    total_bitplanes: u8,
) -> CudaStageResult<Vec<CudaHtj2kEncodeCodeBlockRegionJob>> {
    if code_block_width == 0 || code_block_height == 0 {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K encode job has invalid code-block dimensions",
        ));
    }
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    let num_cbs_x = width.div_ceil(code_block_width);
    let num_cbs_y = height.div_ceil(code_block_height);
    let count = (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block count"))?;
    let mut cuda_jobs = try_vec_with_capacity(count, "j2k CUDA HTJ2K region jobs")
        .map_err(htj2k_allocation_error)?;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx
                .checked_mul(code_block_width)
                .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block x offset"))?;
            let y0 = cby
                .checked_mul(code_block_height)
                .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block y offset"))?;
            let block_width = (x0 + code_block_width).min(width) - x0;
            let block_height = (y0 + code_block_height).min(height) - y0;
            let offset = (y0 as usize)
                .checked_mul(width as usize)
                .and_then(|row| row.checked_add(x0 as usize))
                .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K code-block offset"))?;
            try_vec_push(
                &mut cuda_jobs,
                CudaHtj2kEncodeCodeBlockRegionJob {
                    coefficient_offset: u32::try_from(offset).map_err(|_| {
                        arithmetic_overflow("CUDA HTJ2K code-block offset exceeds u32")
                    })?,
                    coefficient_stride: width,
                    width: block_width,
                    height: block_height,
                    total_bitplanes,
                    target_coding_passes: 1,
                },
                "j2k CUDA HTJ2K region jobs",
            )
            .map_err(htj2k_allocation_error)?;
        }
    }
    Ok(cuda_jobs)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_ht_subband(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kHtSubbandEncodeJob<'_>,
    collect_profile: bool,
) -> CudaStageResult<CudaEncodedHtSubband> {
    let expected_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K subband coefficient count"))?;
    if expected_len != job.coefficients.len() {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K subband encode job has invalid coefficient length",
        ));
    }
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K subband encode job has invalid code-block dimensions",
        ));
    }

    let sample_buffer = context
        .upload_f32_pinned(job.coefficients)
        .map_err(|error| runtime_error("upload CUDA HTJ2K subband", error))?;
    let (quantized, quantize_us) = time_cuda_stage(
        "j2k.htj2k.encode.subband.quantize",
        context,
        collect_profile,
        || {
            j2k_cuda_j2k_engine::J2kCudaEngine::new(context).j2k_quantize_subband_resident(
                &sample_buffer,
                job.coefficients.len(),
                CudaJ2kQuantizeJob {
                    step_exponent: job.step_exponent,
                    step_mantissa: job.step_mantissa,
                    range_bits: job.range_bits,
                    reversible: job.reversible,
                },
            )
        },
    )
    .map_err(|error| runtime_error("quantize CUDA HTJ2K subband", error))?;
    let cuda_jobs = cuda_ht_subband_region_jobs(job)?;
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K resident subband jobs");
    host_budget
        .account_vec(&cuda_jobs)
        .map_err(htj2k_allocation_error)?;
    let pool = context.buffer_pool();
    let encoded = j2k_cuda_j2k_engine::J2kCudaEngine::new(context)
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool_and_live_host_bytes(
            quantized.buffer(),
            quantized.coefficient_count(),
            &cuda_jobs,
            encode_resources,
            &pool,
            host_budget.live_bytes(),
        )
        .map_err(|error| runtime_error("encode CUDA HTJ2K resident subband", error))?;

    Ok(CudaEncodedHtSubband {
        quantize_dispatches: quantized.execution().kernel_dispatches(),
        timings: CudaEncodeStageTimings {
            quantize_us,
            ht_encode_us: encoded.stage_timings().ht_encode_us,
            ..CudaEncodeStageTimings::default()
        },
        encode: encoded,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_ht_subband_region_jobs(
    job: J2kHtSubbandEncodeJob<'_>,
) -> CudaStageResult<Vec<CudaHtj2kEncodeCodeBlockRegionJob>> {
    cuda_ht_region_jobs(
        job.width,
        job.height,
        job.code_block_width,
        job.code_block_height,
        job.total_bitplanes,
    )
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_tile_subband_region(
    runtime: CudaHtj2kEncodeRuntime<'_>,
    source: &CudaDeviceBuffer,
    region: CudaTileSubbandRegion,
    quantization_step: (u16, u16),
    job: J2kResidentHtj2kTileEncodeJob<'_>,
    subband_kind: CudaTileSubbandKind,
    stats: &mut CudaHtj2kTileEncodeStats,
) -> CudaStageResult<CudaEncodedHtj2kSubband> {
    if region.width == 0 || region.height == 0 {
        return Ok(CudaEncodedHtj2kSubband {
            code_blocks: Vec::new(),
            num_cbs_x: 0,
            num_cbs_y: 0,
        });
    }

    let (step_exponent, step_mantissa) = quantization_step;
    let step_exponent_u8 = u8::try_from(step_exponent).map_err(|_| {
        J2kEncodeStageError::invalid_request("CUDA HTJ2K tile quantization exponent exceeds u8")
    })?;
    let total_bitplanes = job
        .guard_bits
        .saturating_add(step_exponent_u8)
        .saturating_sub(1);
    let (quantized, quantize_us) = time_cuda_stage(
        "j2k.htj2k.encode.tile.quantize",
        runtime.context,
        stats.collect_profile,
        || {
            j2k_cuda_j2k_engine::J2kCudaEngine::new(runtime.context)
                .j2k_quantize_subband_region_resident(
                    source,
                    CudaJ2kQuantizeSubbandRegionJob {
                        x0: region.x0,
                        y0: region.y0,
                        width: region.width,
                        height: region.height,
                        stride: region.stride,
                        quantization: CudaJ2kQuantizeJob {
                            step_exponent,
                            step_mantissa,
                            range_bits: cuda_tile_subband_range_bits(
                                job.input.bit_depth(),
                                subband_kind,
                            ),
                            reversible: job.reversible,
                        },
                    },
                )
        },
    )
    .map_err(|error| runtime_error("quantize CUDA HTJ2K tile subband", error))?;
    stats.quantize_jobs = stats.quantize_jobs.saturating_add(1);
    stats.quantize_dispatches = stats
        .quantize_dispatches
        .saturating_add(quantized.execution().kernel_dispatches());
    stats.timings.quantize_us = stats.timings.quantize_us.saturating_add(quantize_us);

    let region_jobs = cuda_ht_region_jobs(
        region.width,
        region.height,
        job.code_block_width,
        job.code_block_height,
        total_bitplanes,
    )?;
    stats.ht_code_block_jobs = stats.ht_code_block_jobs.saturating_add(region_jobs.len());
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K tile region jobs");
    host_budget
        .account_vec(&region_jobs)
        .map_err(htj2k_allocation_error)?;
    let encoded = j2k_cuda_j2k_engine::J2kCudaEngine::new(runtime.context)
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool_and_live_host_bytes(
            quantized.buffer(),
            quantized.coefficient_count(),
            &region_jobs,
            runtime.resources,
            runtime.pool,
            host_budget.live_bytes(),
        )
        .map_err(|error| runtime_error("encode CUDA HTJ2K tile code blocks", error))?;
    stats.ht_code_block_dispatches = stats
        .ht_code_block_dispatches
        .saturating_add(encoded.execution().kernel_dispatches());
    stats.timings.ht_encode_us = stats
        .timings
        .ht_encode_us
        .saturating_add(encoded.stage_timings().ht_encode_us);
    let maximum_cleanup_magnitude = encoded
        .code_blocks()
        .iter()
        .map(|block| u64::from(block.status().detail))
        .max()
        .unwrap_or(0);
    let required_ht_magnitude_bound = j2k_native::htj2k_required_magnitude_bound(
        maximum_cleanup_magnitude,
        job.reversible,
        region.decomposition_level,
    );
    stats.required_ht_magnitude_bound = Some(
        stats
            .required_ht_magnitude_bound
            .map_or(required_ht_magnitude_bound, |current| {
                current.max(required_ht_magnitude_bound)
            }),
    );

    Ok(CudaEncodedHtj2kSubband {
        code_blocks: encoded_ht_code_blocks_from_cuda(encoded)?,
        num_cbs_x: region.width.div_ceil(job.code_block_width),
        num_cbs_y: region.height.div_ceil(job.code_block_height),
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_tile_subband_range_bits(bit_depth: u8, subband_kind: CudaTileSubbandKind) -> u8 {
    let log_gain = match subband_kind {
        CudaTileSubbandKind::LowLow => 0,
        CudaTileSubbandKind::HighLow | CudaTileSubbandKind::LowHigh => 1,
        CudaTileSubbandKind::HighHigh => 2,
    };
    bit_depth.saturating_add(log_gain)
}

#[cfg(feature = "cuda-runtime")]
fn encoded_ht_code_block_from_cuda(
    encoded: j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlock,
) -> EncodedHtJ2kCodeBlock {
    let (data, cleanup_length, refinement_length, num_coding_passes, num_zero_bitplanes) =
        encoded.into_parts();
    EncodedHtJ2kCodeBlock {
        data,
        cleanup_length,
        refinement_length,
        num_coding_passes,
        num_zero_bitplanes,
    }
}

#[cfg(feature = "cuda-runtime")]
fn encoded_ht_code_block_set_from_cuda(
    encoded: j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlock,
) -> EncodedHtJ2kCodeBlockSet {
    let (
        data,
        cleanup_length,
        sigprop_length,
        magref_length,
        num_coding_passes,
        num_zero_bitplanes,
    ) = encoded.into_exact_parts();
    EncodedHtJ2kCodeBlockSet {
        data,
        cleanup_length,
        sigprop_length,
        magref_length,
        num_coding_passes,
        num_zero_bitplanes,
    }
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn encoded_ht_code_blocks_from_cuda(
    encoded: j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlocks,
) -> CudaStageResult<Vec<EncodedHtJ2kCodeBlock>> {
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K encoded code-block conversion");
    host_budget
        .account_bytes(encoded.host_capacity_bytes())
        .map_err(htj2k_allocation_error)?;
    let code_blocks = encoded.into_code_blocks();
    let mut outputs = host_budget
        .try_vec_with_capacity(code_blocks.len())
        .map_err(htj2k_allocation_error)?;
    for code_block in code_blocks {
        outputs.push(encoded_ht_code_block_from_cuda(code_block));
    }
    Ok(outputs)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn encoded_ht_code_block_sets_from_cuda(
    encoded: j2k_cuda_j2k_engine::CudaHtj2kEncodedCodeBlocks,
) -> CudaStageResult<Vec<EncodedHtJ2kCodeBlockSet>> {
    let mut host_budget = HostPhaseBudget::new("j2k CUDA encoded HT candidate conversion");
    host_budget
        .account_bytes(encoded.host_capacity_bytes())
        .map_err(htj2k_allocation_error)?;
    let code_blocks = encoded.into_code_blocks();
    let mut outputs = host_budget
        .try_vec_with_capacity(code_blocks.len())
        .map_err(htj2k_allocation_error)?;
    for code_block in code_blocks {
        outputs.push(encoded_ht_code_block_set_from_cuda(code_block));
    }
    Ok(outputs)
}

#[cfg(feature = "cuda-runtime")]
pub(crate) fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: j2k_native::ht_vlc_encode_table0(),
        vlc_table1: j2k_native::ht_vlc_encode_table1(),
        uvlc_table: j2k_native::ht_uvlc_encode_table_bytes(),
    }
}
