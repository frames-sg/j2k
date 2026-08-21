// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    allocation::HostPhaseBudget,
    bytes::{f32_slice_as_bytes_mut, i32_slice_as_bytes_mut},
};
use crate::{error::CudaError, CudaTranscodeEngine};

pub(crate) fn ensure_context_ownership(
    matches_context: impl IntoIterator<Item = bool>,
    mismatch_message: &'static str,
) -> Result<(), CudaError> {
    if matches_context.into_iter().all(|matches| matches) {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: mismatch_message.to_string(),
        })
    }
}

impl CudaTranscodeEngine<'_> {
    pub(crate) fn download_i32_band(
        buffer: &j2k_cuda_runtime::CudaDeviceBuffer,
        count: usize,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Vec<i32>, CudaError> {
        let mut out = host_budget.try_vec_filled(count, 0i32)?;
        if count != 0 {
            buffer.copy_to_host(i32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    pub(crate) fn download_f32_band(
        buffer: &j2k_cuda_runtime::CudaDeviceBuffer,
        count: usize,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = host_budget.try_vec_filled(count, 0f32)?;
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    pub(crate) fn download_pooled_f32_band(
        buffer: &j2k_cuda_runtime::CudaPooledDeviceBuffer,
        count: usize,
        host_budget: &mut HostPhaseBudget,
    ) -> Result<Vec<f32>, CudaError> {
        let mut out = host_budget.try_vec_filled(count, 0f32)?;
        if count != 0 {
            buffer.copy_to_host(f32_slice_as_bytes_mut(&mut out))?;
        }
        Ok(out)
    }

    pub(crate) fn upload(
        &self,
        bytes: &[u8],
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context.upload(bytes)
    }

    pub(crate) fn upload_f32(
        &self,
        values: &[f32],
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context.upload_f32(values)
    }

    pub(crate) fn allocate(
        &self,
        len: usize,
    ) -> Result<j2k_cuda_runtime::CudaDeviceBuffer, CudaError> {
        self.context.allocate(len)
    }

    pub(crate) fn time_default_stream_us<T>(
        &self,
        work: impl FnMut() -> Result<T, CudaError>,
    ) -> Result<(T, u128), CudaError> {
        self.context
            .time_default_stream_named_us("j2k-cuda-transcode-engine", work)
    }

    pub(crate) fn launch_kernel(
        &self,
        spec: j2k_cuda_runtime::CudaKernelSpec,
        geometry: j2k_cuda_runtime::CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: engine launch sites bind parameters matching each static
        // kernel specification and retain all referenced allocations.
        unsafe { self.context.launch_compiled_kernel(spec, geometry, params) }
    }

    pub(crate) fn launch_kernel_async(
        &self,
        spec: j2k_cuda_runtime::CudaKernelSpec,
        geometry: j2k_cuda_runtime::CudaLaunchGeometry,
        params: &mut [*mut std::ffi::c_void],
    ) -> Result<(), CudaError> {
        // SAFETY: callers retain all referenced allocations until a later
        // timed completion boundary has completed the default stream.
        unsafe {
            self.context
                .launch_compiled_kernel_async(spec, geometry, params)
        }
    }
}
