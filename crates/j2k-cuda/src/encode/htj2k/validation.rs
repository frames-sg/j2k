// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    J2kEncodeStageError, J2kHtj2kTileEncodeJob, J2kResidentEncodeInput,
    J2kResidentEncodeInputError, J2kResidentHtj2kTileEncodeJob,
};

use crate::encode::stage_error::{arithmetic_overflow, CudaStageResult};

use super::super::cuda_component_count_u8;

#[cfg(feature = "cuda-runtime")]
pub(super) fn resident_job_from_host(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> CudaStageResult<J2kResidentHtj2kTileEncodeJob<'_>> {
    let input = J2kResidentEncodeInput::new(
        job.width,
        job.height,
        job.num_components,
        job.bit_depth,
        job.signed,
    )
    .map_err(|error| match error {
        J2kResidentEncodeInputError::AddressSpaceOverflow => arithmetic_overflow(error.reason()),
        _ => J2kEncodeStageError::invalid_request(error.reason()),
    })?;
    Ok(J2kResidentHtj2kTileEncodeJob {
        input,
        num_decomposition_levels: job.num_decomposition_levels,
        reversible: job.reversible,
        use_mct: job.use_mct,
        guard_bits: job.guard_bits,
        code_block_width: job.code_block_width,
        code_block_height: job.code_block_height,
        progression_order: job.progression_order,
        component_sampling: job.component_sampling,
        quantization_steps: job.quantization_steps,
    })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn validate_cuda_htj2k_tile_job(
    job: J2kResidentHtj2kTileEncodeJob<'_>,
) -> CudaStageResult<()> {
    let _ = cuda_component_count_u8(
        job.num_components(),
        "CUDA HTJ2K tile encode supports at most 255 components",
    )?;
    if job
        .component_sampling
        .iter()
        .any(|&sampling| sampling != (1, 1))
    {
        return Err(J2kEncodeStageError::unsupported(
            "CUDA HTJ2K tile encode does not support component subsampling != (1, 1)",
        ));
    }
    // Native treats `use_mct = options.use_mct && num_components >= 3`, applying the
    // color transform to component planes 0,1,2 and passing any 4th plane through
    // unchanged. The resident path mirrors this: RCT/ICT runs on the first three
    // planes (see `j2k_forward_rct_resident`/`j2k_forward_ict_resident`), and every
    // component — including the passthrough 4th — still flows through the per-component
    // DWT → quantize → HT code-block → packetization loop below.
    //
    // Only `{1, 3, 4}` component counts are in scope. Reject any other count with a
    // typed hard error rather than `Ok(None)` (a silent CPU fallback is forbidden for
    // in-scope inputs).
    if !matches!(job.num_components(), 1 | 3 | 4) {
        return Err(J2kEncodeStageError::unsupported(
            "CUDA HTJ2K tile encode supports 1, 3, or 4 components",
        ));
    }
    if job.use_mct && job.num_components() < 3 {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K tile encode requires at least three components for MCT",
        ));
    }
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K tile encode job has invalid code-block dimensions",
        ));
    }
    let expected_quantization_steps = 1usize
        .checked_add(usize::from(job.num_decomposition_levels).saturating_mul(3))
        .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K tile quantization step count"))?;
    if job.quantization_steps.len() != expected_quantization_steps {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K tile quantization step count mismatch",
        ));
    }
    Ok(())
}
