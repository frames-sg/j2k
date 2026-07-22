// SPDX-License-Identifier: MIT OR Apache-2.0

mod color_native;
mod color_native_rgba;

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

    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_gray8_batch_enqueue(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreGray8Batch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_gray16_batch_enqueue(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreGray16Batch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel_async(function, geometry, &mut params)
    }

    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_grayi16_batch_enqueue(
        &self,
        jobs: &CudaDeviceBuffer,
        max_pixels: usize,
        job_count: usize,
    ) -> Result<(), CudaError> {
        let function = self.j2k_decode_store_kernel_function(CudaKernel::J2kStoreGrayI16Batch)?;
        let mut jobs_ptr = jobs.device_ptr();
        let mut params = cuda_kernel_params!(jobs_ptr);
        let geometry = j2k_store_batch_launch_geometry(max_pixels, job_count)
            .ok_or(CudaError::LengthTooLarge { len: max_pixels })?;
        self.launch_kernel_async(function, geometry, &mut params)
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

    pub(in crate::j2k_decode) unsafe fn launch_j2k_store_rgb8_mct_batch_enqueue(
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
        self.launch_kernel_async(function, geometry, &mut params)
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

#[cfg(test)]
mod tests {
    #[test]
    fn grayscale_batch_final_stores_enqueue_without_context_synchronization() {
        let source = include_str!("store_launch.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("production store launch source");
        for name in [
            "launch_j2k_store_gray8_batch_enqueue",
            "launch_j2k_store_gray16_batch_enqueue",
            "launch_j2k_store_grayi16_batch_enqueue",
        ] {
            let function = source
                .split(name)
                .nth(1)
                .unwrap_or_else(|| panic!("missing {name}"))
                .split("\n    }")
                .next()
                .expect("batch store function");
            assert!(function.contains("launch_kernel_async"), "{name}");
            assert!(!function.contains("launch_kernel(function"), "{name}");
            assert!(!function.contains("synchronize"), "{name}");
        }
    }
}
