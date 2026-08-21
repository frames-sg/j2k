// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{j2k_forward_rct_launch_geometry, CudaKernel},
    memory::CudaDeviceBuffer,
};

use super::super::J2kStridedDeinterleaveLaunch;

impl crate::J2kCudaEngine<'_> {
    pub(in crate::j2k_encode) fn launch_j2k_deinterleave_to_f32(
        &self,
        pixels: &CudaDeviceBuffer,
        output: &CudaDeviceBuffer,
        num_pixels: usize,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(CudaKernel::J2kDeinterleaveToF32)?;
        let mut pixels_ptr = pixels.device_ptr();
        let mut output_ptr = output.device_ptr();
        let mut num_pixels_u64 =
            u64::try_from(num_pixels).map_err(|_| CudaError::LengthTooLarge { len: num_pixels })?;
        let mut num_components_u32 = u32::from(num_components);
        let mut bit_depth_u32 = u32::from(bit_depth);
        let mut signed_u32 = u32::from(signed);
        let mut params = cuda_kernel_params!(
            pixels_ptr,
            output_ptr,
            num_pixels_u64,
            num_components_u32,
            bit_depth_u32,
            signed_u32
        );
        let geometry = j2k_forward_rct_launch_geometry(num_pixels)
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }

    pub(in crate::j2k_encode) fn launch_j2k_deinterleave_strided_to_f32(
        &self,
        request: J2kStridedDeinterleaveLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function = Self::j2k_encode_kernel_function(CudaKernel::J2kDeinterleaveStridedToF32)?;
        let mut pixels_ptr = request.pixels.device_ptr();
        let mut output_ptr = request.output.device_ptr();
        let mut width_u64 = u64::from(request.width);
        let mut height_u64 = u64::from(request.height);
        let mut byte_offset_u64 =
            u64::try_from(request.byte_offset).map_err(|_| CudaError::LengthTooLarge {
                len: request.byte_offset,
            })?;
        let mut pitch_bytes_u64 =
            u64::try_from(request.pitch_bytes).map_err(|_| CudaError::LengthTooLarge {
                len: request.pitch_bytes,
            })?;
        let mut num_components_u32 = u32::from(request.num_components);
        let mut bit_depth_u32 = u32::from(request.bit_depth);
        let mut signed_u32 = u32::from(request.signed);
        let mut params = cuda_kernel_params!(
            pixels_ptr,
            output_ptr,
            width_u64,
            height_u64,
            byte_offset_u64,
            pitch_bytes_u64,
            num_components_u32,
            bit_depth_u32,
            signed_u32
        );
        let num_pixels = strided_pixel_count(request)?;
        let geometry = j2k_forward_rct_launch_geometry(num_pixels)
            .ok_or(CudaError::LengthTooLarge { len: num_pixels })?;
        self.launch_kernel(function, geometry, &mut params)
    }
}

fn strided_pixel_count(request: J2kStridedDeinterleaveLaunch<'_>) -> Result<usize, CudaError> {
    (request.width as usize)
        .checked_mul(request.height as usize)
        .ok_or(CudaError::ImageTooLarge {
            width: request.width,
            height: request.height,
            channels: usize::from(request.num_components),
        })
}
