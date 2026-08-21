// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{CudaDeviceBuffer, CudaPooledDeviceBuffer, Error, CUDA_HTJ2K_KERNELS_NOT_READY};

#[cfg(feature = "cuda-runtime")]
pub(in crate::decoder) fn pooled_cuda_buffer(
    buffer: &CudaPooledDeviceBuffer,
) -> Result<&CudaDeviceBuffer, Error> {
    buffer.as_device_buffer().ok_or(Error::capability_rejected(
        j2k_core::CapabilityRejection::missing_prepared_plan(CUDA_HTJ2K_KERNELS_NOT_READY),
    ))
}
