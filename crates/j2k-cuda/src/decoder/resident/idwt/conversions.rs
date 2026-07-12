// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    CudaCoefficientBand, CudaHtj2kBandId, CudaHtj2kIdwtStep, CudaHtj2kTransform, CudaJ2kIdwtJob,
    CudaJ2kRect, Error, CUDA_HTJ2K_KERNELS_NOT_READY,
};

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
