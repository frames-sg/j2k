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
    if job.input.width() > (1 << 15) || job.input.height() > (1 << 15) {
        return Err(J2kEncodeStageError::unsupported(
            "CUDA resident HTJ2K encode supports one implicit precinct per resolution; use staged host encode for larger tiles",
        ));
    }
    let _ = cuda_component_count_u8(
        job.input.num_components(),
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
    if !matches!(job.input.num_components(), 1 | 3 | 4) {
        return Err(J2kEncodeStageError::unsupported(
            "CUDA HTJ2K tile encode supports 1, 3, or 4 components",
        ));
    }
    if job.use_mct && job.input.num_components() < 3 {
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

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use super::*;

    #[test]
    fn resident_tile_requires_one_implicit_precinct_before_device_work() {
        for (width, height, supported) in
            [(32_768, 4, true), (32_769, 4, false), (4, 32_769, false)]
        {
            let job = J2kResidentHtj2kTileEncodeJob {
                input: J2kResidentEncodeInput::new(width, height, 1, 8, false)
                    .expect("logical input"),
                num_decomposition_levels: 0,
                reversible: true,
                use_mct: false,
                guard_bits: 1,
                code_block_width: 64,
                code_block_height: 64,
                progression_order: j2k::J2kPacketizationProgressionOrder::Lrcp,
                component_sampling: &[(1, 1)],
                quantization_steps: &[(8, 0)],
            };
            let result = validate_cuda_htj2k_tile_job(job);
            if supported {
                result.expect("one precinct is supported");
            } else {
                assert_eq!(
                    result
                        .expect_err("multiple precincts need staged encode")
                        .kind(),
                    j2k::J2kEncodeStageErrorKind::Unsupported
                );
            }
        }
    }
}
