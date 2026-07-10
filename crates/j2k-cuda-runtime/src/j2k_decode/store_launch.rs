// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    context::CudaContext,
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{j2k_forward_rct_launch_geometry, j2k_store_batch_launch_geometry, CudaKernel},
    memory::CudaDeviceBuffer,
};

impl CudaContext {
    pub(in crate::j2k_decode) fn launch_j2k_store_gray8(
        &self,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreGray8)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(input_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) fn launch_j2k_store_gray16(
        &self,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreGray16)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(input_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) fn launch_j2k_inverse_mct(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kInverseMct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params = cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, job_ptr);
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) fn launch_j2k_store_rgb8(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgb8)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params =
            cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) fn launch_j2k_store_rgb16(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgb16)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params =
            cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) fn launch_j2k_store_rgb8_mct_batch(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgb8MctBatch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) fn launch_j2k_store_rgb16_mct(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        job: &CudaDeviceBuffer,
        pixels: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreRgb16Mct)?;
        let mut plane0_ptr = plane0.device_ptr();
        let mut plane1_ptr = plane1.device_ptr();
        let mut plane2_ptr = plane2.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut job_ptr = job.device_ptr();
        let mut params =
            cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, output_ptr, job_ptr);
        let geometry = j2k_forward_rct_launch_geometry(pixels)
            .ok_or(CudaError::LengthTooLarge { len: pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    fn j2k_decode_store_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner
            .cuda_oxide_j2k_decode_store_kernel_function(kernel)
    }
}
