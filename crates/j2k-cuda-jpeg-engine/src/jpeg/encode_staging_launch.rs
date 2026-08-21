// SPDX-License-Identifier: MIT OR Apache-2.0

//! Typed kernel launches for staged CUDA baseline JPEG encode.

use super::encode_launch::CudaJpegBaselineHuffmanLaunch;
use crate::{
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::CudaDeviceBuffer,
    JpegCudaEngine,
};

pub(super) struct CudaJpegBaselinePrecomputeLaunch<'a> {
    pub(super) input_ptr: u64,
    pub(super) coefficients: &'a CudaDeviceBuffer,
    pub(super) params: &'a CudaDeviceBuffer,
    pub(super) q_luma: &'a CudaDeviceBuffer,
    pub(super) q_chroma: &'a CudaDeviceBuffer,
    pub(super) tile_count: u32,
    pub(super) total_mcus: u32,
    pub(super) geometry: CudaLaunchGeometry,
}

pub(super) struct CudaJpegBaselineEntropyFromCoeffsLaunch<'a> {
    pub(super) coefficients: &'a CudaDeviceBuffer,
    pub(super) entropy: &'a CudaDeviceBuffer,
    pub(super) status: &'a CudaDeviceBuffer,
    pub(super) params: &'a CudaDeviceBuffer,
    pub(super) huffman: CudaJpegBaselineHuffmanLaunch<'a>,
    pub(super) tile_count: u32,
    pub(super) geometry: CudaLaunchGeometry,
}

impl JpegCudaEngine<'_> {
    pub(super) fn launch_jpeg_encode_baseline_precompute_batch(
        self,
        request: &CudaJpegBaselinePrecomputeLaunch<'_>,
    ) -> Result<(), CudaError> {
        let mut input_ptr = request.input_ptr;
        let mut coefficient_ptr = request.coefficients.device_ptr();
        let mut params_ptr = request.params.device_ptr();
        let mut q_luma_ptr = request.q_luma.device_ptr();
        let mut q_chroma_ptr = request.q_chroma.device_ptr();
        let mut tile_count = request.tile_count;
        let mut total_mcus = request.total_mcus;
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            coefficient_ptr,
            params_ptr,
            q_luma_ptr,
            q_chroma_ptr,
            tile_count,
            total_mcus,
        );
        self.launch_kernel(
            CudaKernel::JpegEncodeBaselinePrecomputeBatch,
            request.geometry,
            &mut kernel_params,
        )
    }

    #[expect(
        clippy::similar_names,
        reason = "DC/AC luma/chroma pointer names preserve CUDA parameter order"
    )]
    pub(super) fn launch_jpeg_encode_baseline_entropy_from_coeffs_batch(
        self,
        request: &CudaJpegBaselineEntropyFromCoeffsLaunch<'_>,
    ) -> Result<(), CudaError> {
        let mut coefficient_ptr = request.coefficients.device_ptr();
        let mut entropy_ptr = request.entropy.device_ptr();
        let mut status_ptr = request.status.device_ptr();
        let mut params_ptr = request.params.device_ptr();
        let mut huff_dc_luma_ptr = request.huffman.dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = request.huffman.ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = request.huffman.dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = request.huffman.ac_chroma.device_ptr();
        let mut tile_count = request.tile_count;
        let mut kernel_params = cuda_kernel_params!(
            coefficient_ptr,
            entropy_ptr,
            status_ptr,
            params_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr,
            tile_count,
        );
        self.launch_kernel(
            CudaKernel::JpegEncodeBaselineEntropyFromCoeffsBatch,
            request.geometry,
            &mut kernel_params,
        )
    }
}
