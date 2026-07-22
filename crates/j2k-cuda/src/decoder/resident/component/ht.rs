// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{CudaHtj2kCodeBlockJob, Error, CUDA_HTJ2K_KERNELS_NOT_READY};

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
