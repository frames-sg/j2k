// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    build_flags::CUDA_IDWT_TRACE_ENV_VAR,
    bytes::{f32_slice_as_bytes_mut, i32_slice_as_bytes_mut},
    error::CudaError,
    memory::{CudaDeviceBuffer, CudaPooledDeviceBuffer},
};

use super::CudaContext;

pub(crate) fn cuda_idwt_trace_enabled() -> bool {
    std::env::var_os(CUDA_IDWT_TRACE_ENV_VAR).is_some()
}

impl CudaContext {
    pub(crate) fn download_i32_band(
        buffer: &CudaDeviceBuffer,
        count: usize,
    ) -> Result<Vec<i32>, CudaError> {
        let mut out = vec![0i32; count];
        if count != 0 {
            buffer.copy_to_host(i32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    pub(crate) fn download_f32_band(
        buffer: &CudaDeviceBuffer,
        count: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0f32; count];
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    pub(crate) fn download_pooled_f32_band(
        buffer: &CudaPooledDeviceBuffer,
        count: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = vec![0f32; count];
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }
}
