// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{CudaDeviceBuffer, CudaPooledDeviceBuffer, Error, CUDA_HTJ2K_KERNELS_NOT_READY};

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
