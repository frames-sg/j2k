// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodedHtJ2kCodeBlock, J2kEncodeStageError, J2kHtCodeBlockEncodeJob, J2kHtSubbandEncodeJob,
};
use j2k_cuda_runtime::{
    CudaContext, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeTables, CudaJ2kQuantizeJob,
};

use crate::allocation::{try_vec_push, try_vec_with_capacity, HostPhaseBudget};
use crate::encode::stage_error::{arithmetic_overflow, runtime_error, CudaStageResult};

use super::super::{time_cuda_stage, CudaEncodeStageTimings};
use super::htj2k_allocation_error;
use super::types::CudaEncodedHtSubband;

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_ht_code_block(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> CudaStageResult<j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks> {
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
    context
        .encode_htj2k_codeblocks_with_resources(job.coefficients, &cuda_jobs, resources)
        .map_err(|error| runtime_error("encode CUDA HTJ2K code block", error))
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_ht_code_blocks(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> CudaStageResult<j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks> {
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

    context
        .encode_htj2k_codeblocks_with_resources_and_live_host_bytes(
            &coefficients,
            &cuda_jobs,
            resources,
            host_budget.live_bytes(),
        )
        .map_err(|error| runtime_error("encode CUDA HTJ2K code-block batch", error))
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
            context.j2k_quantize_subband_resident(
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
    let encoded = context
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
fn encoded_ht_code_block_from_cuda(
    encoded: j2k_cuda_runtime::CudaHtj2kEncodedCodeBlock,
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
pub(in crate::encode) fn encoded_ht_code_blocks_from_cuda(
    encoded: j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks,
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
pub(crate) fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: j2k_native::ht_vlc_encode_table0(),
        vlc_table1: j2k_native::ht_vlc_encode_table1(),
        uvlc_table: j2k_native::ht_uvlc_encode_table_bytes(),
    }
}
