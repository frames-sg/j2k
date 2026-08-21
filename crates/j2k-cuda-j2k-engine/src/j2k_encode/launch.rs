// SPDX-License-Identifier: MIT OR Apache-2.0

mod input;

use crate::{
    driver::{CuDevicePtr, CuFunction},
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{j2k_dwt53_launch_geometry, j2k_forward_rct_launch_geometry, CudaKernel},
    memory::CudaDeviceBuffer,
};

use super::{CudaDwt53Pass, CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob};

impl crate::J2kCudaEngine<'_> {
    pub(super) fn launch_j2k_forward_rct_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_forward_rct_ptrs(
            plane0.device_ptr(),
            plane1.device_ptr(),
            plane2.device_ptr(),
            len,
        )
    }

    fn j2k_encode_kernel_function(kernel: CudaKernel) -> Result<CuFunction, CudaError> {
        kernel.spec()
    }

    pub(super) fn launch_j2k_forward_rct_ptrs(
        &self,
        plane0: CuDevicePtr,
        plane1: CuDevicePtr,
        plane2: CuDevicePtr,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(CudaKernel::J2kForwardRct)?;
        let mut plane0_ptr = plane0;
        let mut plane1_ptr = plane1;
        let mut plane2_ptr = plane2;
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, len_u64);
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    pub(super) fn launch_j2k_forward_ict_buffers(
        &self,
        plane0: &CudaDeviceBuffer,
        plane1: &CudaDeviceBuffer,
        plane2: &CudaDeviceBuffer,
        len: usize,
    ) -> Result<(), CudaError> {
        self.launch_j2k_forward_ict_ptrs(
            plane0.device_ptr(),
            plane1.device_ptr(),
            plane2.device_ptr(),
            len,
        )
    }

    pub(super) fn launch_j2k_forward_ict_ptrs(
        &self,
        plane0: CuDevicePtr,
        plane1: CuDevicePtr,
        plane2: CuDevicePtr,
        len: usize,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(CudaKernel::J2kForwardIct)?;
        let mut plane0_ptr = plane0;
        let mut plane1_ptr = plane1;
        let mut plane2_ptr = plane2;
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut params = cuda_kernel_params!(plane0_ptr, plane1_ptr, plane2_ptr, len_u64);
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    pub(super) fn launch_j2k_forward_dwt53_pass(
        &self,
        kernel: CudaKernel,
        input: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        pass: CudaDwt53Pass,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(kernel)?;
        let mut input_ptr = input.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut full_width = pass.full_width;
        let mut current_width = pass.current_width;
        let mut current_height = pass.current_height;
        let mut low_extent = pass.low_extent;
        let mut params = cuda_kernel_params!(
            input_ptr,
            output_ptr,
            full_width,
            current_width,
            current_height,
            low_extent
        );
        let geometry = j2k_dwt53_launch_geometry(current_width, current_height).ok_or(
            CudaError::ImageTooLarge {
                width: pass.current_width,
                height: pass.current_height,
                channels: 1,
            },
        )?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(super) fn launch_j2k_quantize_subband(
        &self,
        samples: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        len: usize,
        job: CudaJ2kQuantizeJob,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(CudaKernel::J2kQuantizeSubband)?;
        let mut samples_ptr = samples.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut len_u64 = u64::try_from(len).map_err(|_| CudaError::LengthTooLarge { len })?;
        let mut step_exponent = u32::from(job.step_exponent);
        let mut step_mantissa = u32::from(job.step_mantissa);
        let mut range_bits = u32::from(job.range_bits);
        let mut reversible = u32::from(job.reversible);
        let mut params = cuda_kernel_params!(
            samples_ptr,
            coefficients_ptr,
            len_u64,
            step_exponent,
            step_mantissa,
            range_bits,
            reversible
        );
        let geometry =
            j2k_forward_rct_launch_geometry(len).ok_or(CudaError::LengthTooLarge { len })?;

        self.launch_kernel(function, geometry, &mut params)
    }

    pub(super) fn launch_j2k_quantize_subband_region(
        &self,
        samples: &CudaDeviceBuffer,
        coefficients: &CudaDeviceBuffer,
        job: CudaJ2kQuantizeSubbandRegionJob,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(CudaKernel::J2kQuantizeSubbandStrided)?;
        let mut samples_ptr = samples.device_ptr();
        let mut coefficients_ptr = coefficients.device_ptr();
        let mut x0 = job.x0;
        let mut y0 = job.y0;
        let mut width = job.width;
        let mut height = job.height;
        let mut stride = job.stride;
        let mut step_exponent = u32::from(job.quantization.step_exponent);
        let mut step_mantissa = u32::from(job.quantization.step_mantissa);
        let mut range_bits = u32::from(job.quantization.range_bits);
        let mut reversible = u32::from(job.quantization.reversible);
        let mut params = cuda_kernel_params!(
            samples_ptr,
            coefficients_ptr,
            x0,
            y0,
            width,
            height,
            stride,
            step_exponent,
            step_mantissa,
            range_bits,
            reversible
        );
        let geometry =
            j2k_dwt53_launch_geometry(job.width, job.height).ok_or(CudaError::ImageTooLarge {
                width: job.width,
                height: job.height,
                channels: 1,
            })?;

        self.launch_kernel(function, geometry, &mut params)
    }
}
