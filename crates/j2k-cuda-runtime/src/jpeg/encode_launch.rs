// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::{
        JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS, JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN,
        JPEG_BASELINE_ENCODE_STATUS_OK, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW,
    },
    CudaJpegBaselineEncodeParams, CudaJpegBaselineEncodeStatus,
};
use crate::{
    context::CudaContext,
    error::CudaError,
    execution::cuda_kernel_param,
    kernels::{CudaKernel, CudaLaunchGeometry},
    memory::CudaDeviceBuffer,
};

pub(super) struct CudaJpegBaselineQuantLaunch<'a> {
    pub(super) luma: &'a CudaDeviceBuffer,
    pub(super) chroma: &'a CudaDeviceBuffer,
}

pub(super) struct CudaJpegBaselineHuffmanLaunch<'a> {
    pub(super) dc_luma: &'a CudaDeviceBuffer,
    pub(super) ac_luma: &'a CudaDeviceBuffer,
    pub(super) dc_chroma: &'a CudaDeviceBuffer,
    pub(super) ac_chroma: &'a CudaDeviceBuffer,
}

pub(super) struct CudaJpegBaselineEntropyLaunch<'a> {
    pub(super) input_ptr: u64,
    pub(super) entropy: &'a CudaDeviceBuffer,
    pub(super) status: &'a CudaDeviceBuffer,
    pub(super) params: CudaJpegBaselineEncodeParams,
    pub(super) quant: CudaJpegBaselineQuantLaunch<'a>,
    pub(super) huffman: CudaJpegBaselineHuffmanLaunch<'a>,
}

pub(super) struct CudaJpegBaselineEntropyBatchLaunch<'a> {
    pub(super) input: &'a CudaDeviceBuffer,
    pub(super) entropy: &'a CudaDeviceBuffer,
    pub(super) status: &'a CudaDeviceBuffer,
    pub(super) params: &'a CudaDeviceBuffer,
    pub(super) quant: CudaJpegBaselineQuantLaunch<'a>,
    pub(super) huffman: CudaJpegBaselineHuffmanLaunch<'a>,
    pub(super) tile_count: u32,
    pub(super) geometry: CudaLaunchGeometry,
}

impl CudaContext {
    #[expect(
        clippy::similar_names,
        reason = "DC/AC luma/chroma pointer names preserve CUDA parameter order"
    )]
    pub(super) fn launch_jpeg_encode_baseline_entropy(
        &self,
        request: &CudaJpegBaselineEntropyLaunch<'_>,
    ) -> Result<(), CudaError> {
        let geometry = CudaLaunchGeometry::new((1, 1, 1), (1, 1, 1)).ok_or_else(|| {
            CudaError::InvalidArgument {
                message: "fixed JPEG encode launch geometry is invalid".to_string(),
            }
        })?;
        let function = self.jpeg_encode_kernel_function(CudaKernel::JpegEncodeBaselineEntropy)?;
        let mut input_ptr = request.input_ptr;
        let mut entropy_ptr = request
            .entropy
            .device_ptr()
            .checked_add(u64::from(request.params.entropy_offset_bytes))
            .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
        let mut status_ptr = request.status.device_ptr();
        let mut params = request.params;
        let mut q_luma_ptr = request.quant.luma.device_ptr();
        let mut q_chroma_ptr = request.quant.chroma.device_ptr();
        let mut huff_dc_luma_ptr = request.huffman.dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = request.huffman.ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = request.huffman.dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = request.huffman.ac_chroma.device_ptr();
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            entropy_ptr,
            status_ptr,
            params,
            q_luma_ptr,
            q_chroma_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr
        );
        self.launch_kernel(function, geometry, &mut kernel_params)
    }

    #[expect(
        clippy::similar_names,
        reason = "DC/AC luma/chroma pointer names preserve CUDA parameter order"
    )]
    pub(super) fn launch_jpeg_encode_baseline_entropy_batch(
        &self,
        request: &CudaJpegBaselineEntropyBatchLaunch<'_>,
    ) -> Result<(), CudaError> {
        let function =
            self.jpeg_encode_kernel_function(CudaKernel::JpegEncodeBaselineEntropyBatch)?;
        let mut input_ptr = request.input.device_ptr();
        let mut entropy_ptr = request.entropy.device_ptr();
        let mut status_ptr = request.status.device_ptr();
        let mut params_ptr = request.params.device_ptr();
        let mut q_luma_ptr = request.quant.luma.device_ptr();
        let mut q_chroma_ptr = request.quant.chroma.device_ptr();
        let mut huff_dc_luma_ptr = request.huffman.dc_luma.device_ptr();
        let mut huff_ac_luma_ptr = request.huffman.ac_luma.device_ptr();
        let mut huff_dc_chroma_ptr = request.huffman.dc_chroma.device_ptr();
        let mut huff_ac_chroma_ptr = request.huffman.ac_chroma.device_ptr();
        let mut tile_count = request.tile_count;
        let mut kernel_params = cuda_kernel_params!(
            input_ptr,
            entropy_ptr,
            status_ptr,
            params_ptr,
            q_luma_ptr,
            q_chroma_ptr,
            huff_dc_luma_ptr,
            huff_ac_luma_ptr,
            huff_dc_chroma_ptr,
            huff_ac_chroma_ptr,
            tile_count
        );
        self.launch_kernel(function, request.geometry, &mut kernel_params)
    }

    fn jpeg_encode_kernel_function(
        &self,
        kernel: CudaKernel,
    ) -> Result<crate::driver::CuFunction, CudaError> {
        self.inner.cuda_oxide_jpeg_encode_kernel_function(kernel)
    }
}

pub(super) fn validate_jpeg_encode_status(
    status: CudaJpegBaselineEncodeStatus,
    kernel: &'static str,
) -> Result<(), CudaError> {
    match status.code {
        JPEG_BASELINE_ENCODE_STATUS_OK => Ok(()),
        JPEG_BASELINE_ENCODE_STATUS_OVERFLOW
        | JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN
        | JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS => Err(CudaError::KernelStatus {
            kernel,
            code: status.code,
            detail: status.detail,
        }),
        code => Err(CudaError::KernelStatus {
            kernel,
            code,
            detail: status.detail,
        }),
    }
}
