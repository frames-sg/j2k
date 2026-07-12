// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    CudaCoefficientBand, CudaDeviceBuffer, CudaError, CudaHtj2kBandId, CudaHtj2kCodeBlockJob,
    CudaHtj2kIdwtStep, CudaHtj2kStoreStep, CudaHtj2kTransform, CudaJ2kIdwtJob, CudaJ2kRect,
    CudaPooledDeviceBuffer, Error, CUDA_HTJ2K_KERNELS_NOT_READY,
};

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
pub(in crate::decoder) fn validate_color_stores(
    stores: [&CudaHtj2kStoreStep; 3],
    dimensions: (u32, u32),
) -> Result<(), Error> {
    let first = stores[0];
    for store in stores {
        let input_width = store.input_rect.x1.saturating_sub(store.input_rect.x0);
        let input_height = store.input_rect.y1.saturating_sub(store.input_rect.y0);
        let source_end_x =
            store
                .source_x
                .checked_add(store.copy_width)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        let source_end_y =
            store
                .source_y
                .checked_add(store.copy_height)
                .ok_or(Error::UnsupportedCudaRequest {
                    reason: CUDA_HTJ2K_KERNELS_NOT_READY,
                })?;
        if store.output_x != 0
            || store.output_y != 0
            || store.copy_width != dimensions.0
            || store.copy_height != dimensions.1
            || store.output_width != dimensions.0
            || store.output_height != dimensions.1
            || source_end_x > input_width
            || source_end_y > input_height
            || store.source_x != first.source_x
            || store.source_y != first.source_y
        {
            return Err(Error::UnsupportedCudaRequest {
                reason: CUDA_HTJ2K_KERNELS_NOT_READY,
            });
        }
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn bit_depth_addend(bit_depth: u8) -> f32 {
    let shift = bit_depth.saturating_sub(1).min(15);
    f32::from(1_u16 << shift)
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn checked_area(width: u32, height: u32) -> Result<usize, Error> {
    width
        .try_into()
        .ok()
        .and_then(|width: usize| width.checked_mul(height as usize))
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn find_cuda_band(
    bands: &[CudaCoefficientBand],
    band_id: CudaHtj2kBandId,
) -> Result<&CudaCoefficientBand, Error> {
    bands
        .iter()
        .find(|band| band.band_id == band_id)
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn pooled_cuda_buffer(
    buffer: &CudaPooledDeviceBuffer,
) -> Result<&CudaDeviceBuffer, Error> {
    buffer
        .as_device_buffer()
        .ok_or(Error::UnsupportedCudaRequest {
            reason: CUDA_HTJ2K_KERNELS_NOT_READY,
        })
}

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "conversion consumes the owned plan error to preserve its message"
)]
pub(super) fn cuda_invalid_decode_plan(error: Error) -> CudaError {
    CudaError::InvalidArgument {
        message: error.to_string(),
    }
}

#[cfg(feature = "cuda-runtime")]
fn cuda_runtime_rect(rect: crate::CudaHtj2kRect) -> CudaJ2kRect {
    CudaJ2kRect {
        x0: rect.x0,
        y0: rect.y0,
        x1: rect.x1,
        y1: rect.y1,
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_idwt_job_from_step(step: &CudaHtj2kIdwtStep) -> CudaJ2kIdwtJob {
    CudaJ2kIdwtJob {
        rect: cuda_runtime_rect(step.rect),
        ll_rect: cuda_runtime_rect(step.ll_rect),
        hl_rect: cuda_runtime_rect(step.hl_rect),
        lh_rect: cuda_runtime_rect(step.lh_rect),
        hh_rect: cuda_runtime_rect(step.hh_rect),
        irreversible97: u32::from(step.transform == CudaHtj2kTransform::Irreversible97),
    }
}
