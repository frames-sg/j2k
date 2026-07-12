// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed accelerator provenance around shared code-block metadata validation.

use super::super::super::code_block_metadata::{
    validate_accelerated_classic_code_block, validate_accelerated_ht_job_output,
};
use super::super::super::{
    bitplane_encode, J2kTier1CodeBlockEncodeJob, NativeEncodePipelineError,
    NativeEncodePipelineResult,
};
use crate::{EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, J2kHtCodeBlockEncodeJob};

pub(in crate::j2c::encode::tier1_driver) fn validate_ht_batch_outputs(
    encoded: &[EncodedHtJ2kCodeBlock],
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> NativeEncodePipelineResult<()> {
    const OPERATION: &str = "HT Tier-1 code-block batch encode";
    if encoded.len() != jobs.len() {
        return Err(accelerator_error(
            OPERATION,
            "accelerated HT code-block batch length mismatch",
        ));
    }
    for (block, job) in encoded.iter().zip(jobs) {
        validate_accelerated_ht_job_output(
            block,
            job.coefficients,
            job.total_bitplanes,
            job.target_coding_passes,
        )
        .map_err(|detail| accelerator_error(OPERATION, detail))?;
    }
    Ok(())
}

pub(in crate::j2c::encode::tier1_driver) fn validate_classic_batch_outputs(
    encoded: &[EncodedJ2kCodeBlock],
    jobs: &[J2kTier1CodeBlockEncodeJob<'_>],
) -> NativeEncodePipelineResult<()> {
    const OPERATION: &str = "classic Tier-1 code-block batch encode";
    if encoded.len() != jobs.len() {
        return Err(accelerator_error(
            OPERATION,
            "accelerated classic code-block batch length mismatch",
        ));
    }
    for (block, job) in encoded.iter().zip(jobs) {
        validate_accelerated_classic_code_block(
            block,
            job.coefficients,
            job.total_bitplanes,
            job.style,
        )
        .map_err(|detail| accelerator_error(OPERATION, detail))?;
    }
    Ok(())
}

pub(in crate::j2c::encode::tier1_driver) fn validated_ht_output(
    encoded: EncodedHtJ2kCodeBlock,
    job: &J2kHtCodeBlockEncodeJob<'_>,
) -> NativeEncodePipelineResult<bitplane_encode::EncodedCodeBlock> {
    const OPERATION: &str = "HT Tier-1 code-block encode";
    validate_accelerated_ht_job_output(
        &encoded,
        job.coefficients,
        job.total_bitplanes,
        job.target_coding_passes,
    )
    .map_err(|detail| accelerator_error(OPERATION, detail))?;
    Ok(super::ht_encoded_code_block_from_accelerator(encoded))
}

pub(in crate::j2c::encode::tier1_driver) fn validated_classic_output(
    encoded: EncodedJ2kCodeBlock,
    job: &J2kTier1CodeBlockEncodeJob<'_>,
) -> NativeEncodePipelineResult<bitplane_encode::EncodedCodeBlock> {
    const OPERATION: &str = "classic Tier-1 code-block encode";
    validate_accelerated_classic_code_block(
        &encoded,
        job.coefficients,
        job.total_bitplanes,
        job.style,
    )
    .map_err(|detail| accelerator_error(OPERATION, detail))?;
    Ok(super::encoded_code_block_from_accelerator(encoded))
}

fn accelerator_error(operation: &'static str, detail: &'static str) -> NativeEncodePipelineError {
    crate::EncodeError::Accelerator {
        operation,
        source: crate::J2kEncodeStageError::internal_invariant(detail),
    }
    .into()
}
