// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{allocation::HostPhaseBudget, CudaError};

use super::{pooled_device_buffer, CudaPooledDeviceBuffer};

#[cfg(test)]
pub(crate) fn copy_pooled_bytes_to_vec_uninit(
    buffer: &CudaPooledDeviceBuffer,
    byte_len: usize,
) -> Result<Vec<u8>, CudaError> {
    let mut host_budget = HostPhaseBudget::new("CUDA pooled readback");
    copy_pooled_bytes_to_vec_uninit_with_budget(buffer, byte_len, &mut host_budget)
}

pub(crate) fn copy_pooled_bytes_to_vec_uninit_with_budget(
    buffer: &CudaPooledDeviceBuffer,
    byte_len: usize,
    host_budget: &mut HostPhaseBudget,
) -> Result<Vec<u8>, CudaError> {
    let mut out = host_budget.try_vec_with_capacity(byte_len)?;
    pooled_device_buffer(buffer)?
        .copy_range_to_host_uninit(0, &mut out.spare_capacity_mut()[..byte_len])?;
    // SAFETY: copy_range_to_host_uninit returned success after writing exactly
    // byte_len initialized bytes into the Vec spare capacity.
    unsafe {
        out.set_len(byte_len);
    }
    Ok(out)
}
