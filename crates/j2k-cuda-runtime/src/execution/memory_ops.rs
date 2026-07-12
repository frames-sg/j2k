// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

use crate::{context::CudaContext, error::CudaError, memory::CudaDeviceBuffer};

impl CudaContext {
    fn validate_memset_target(
        &self,
        dst: &CudaDeviceBuffer,
        required: usize,
    ) -> Result<bool, CudaError> {
        if !dst.is_owned_by(self) {
            return Err(CudaError::InvalidArgument {
                message: "CUDA memset target must belong to the launch context".to_string(),
            });
        }
        if required > dst.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required,
                have: dst.byte_len(),
            });
        }
        Ok(required != 0)
    }

    pub(crate) fn memset_d8(
        &self,
        dst: &CudaDeviceBuffer,
        value: u8,
        bytes: usize,
    ) -> Result<(), CudaError> {
        if !self.validate_memset_target(dst, bytes)? {
            return Ok(());
        }
        self.inner.with_current_resource_operation(|| {
            // SAFETY: `dst` is a live CUDA allocation in this context, `bytes`
            // was bounds-checked, and the context lifecycle gate is held.
            self.inner.driver.check("cuMemsetD8_v2", unsafe {
                (self.inner.driver.cu_memset_d8)(dst.device_ptr(), value, bytes)
            })
        })
    }

    pub(crate) fn memset_d32(
        &self,
        dst: &CudaDeviceBuffer,
        value: c_uint,
        words: usize,
    ) -> Result<(), CudaError> {
        let required = words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or(CudaError::LengthTooLarge { len: words })?;
        if !self.validate_memset_target(dst, required)? {
            return Ok(());
        }
        self.inner.with_current_resource_operation(|| {
            // SAFETY: `dst` is a live CUDA allocation in this context, `words`
            // was bounds-checked, and the context lifecycle gate is held.
            self.inner.driver.check("cuMemsetD32_v2", unsafe {
                (self.inner.driver.cu_memset_d32)(dst.device_ptr(), value, words)
            })
        })
    }
}
