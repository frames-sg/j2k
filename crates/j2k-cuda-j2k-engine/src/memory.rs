// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_cuda_runtime::{
    checked_image_words, CheckedDeviceBufferRanges, CudaBufferPool, CudaBufferPoolReuseGuard,
    CudaDeviceBuffer, CudaDeviceBufferRange, CudaExternalDeviceBufferViewMut,
    CudaPooledDeviceBuffer,
};

use crate::allocation::HostPhaseBudget;
use j2k_cuda_runtime::CudaError;

pub(crate) fn pooled_device_buffer(
    buffer: &CudaPooledDeviceBuffer,
) -> Result<&CudaDeviceBuffer, CudaError> {
    buffer
        .as_device_buffer()
        .ok_or_else(|| CudaError::StatePoisoned {
            message: "pooled CUDA buffer lost its device allocation".to_string(),
        })
}

pub(crate) fn copy_pooled_bytes_to_vec_uninit_with_budget(
    buffer: &CudaPooledDeviceBuffer,
    byte_len: usize,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<u8>, CudaError> {
    let mut out = host_budget.try_vec_with_capacity(byte_len)?;
    pooled_device_buffer(buffer)?
        .copy_range_to_host_uninit(0, &mut out.spare_capacity_mut()[..byte_len])?;
    // SAFETY: the successful device copy initialized exactly `byte_len`
    // elements in the reserved spare capacity.
    unsafe {
        out.set_len(byte_len);
    }
    Ok(out)
}
