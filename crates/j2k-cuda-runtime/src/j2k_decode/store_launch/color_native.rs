// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{j2k_store_batch_launch_geometry, CudaKernel},
    memory::CudaDeviceBuffer,
};

impl CudaContext {
    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_rgb8_native_batch_enqueue(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function =
            self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgb8NativeBatch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_rgb16_native_batch_enqueue(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function =
            self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgb16NativeBatch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_rgbi16_native_batch_enqueue(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function =
            self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgbI16NativeBatch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }
}
