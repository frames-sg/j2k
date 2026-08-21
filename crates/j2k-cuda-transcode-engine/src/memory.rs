// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::CudaError;
pub(crate) use j2k_cuda_runtime::{CudaBufferPool, CudaDeviceBuffer, CudaPooledDeviceBuffer};

pub(crate) fn pooled_device_buffer(
    buffer: &CudaPooledDeviceBuffer,
) -> Result<&CudaDeviceBuffer, CudaError> {
    buffer
        .as_device_buffer()
        .ok_or_else(|| CudaError::InvalidArgument {
            message: "pooled CUDA buffer checkout is empty".to_string(),
        })
}
