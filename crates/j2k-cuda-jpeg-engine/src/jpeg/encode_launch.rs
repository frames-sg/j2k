// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    types::{
        JPEG_BASELINE_ENCODE_STATUS_INVALID_PARAMS, JPEG_BASELINE_ENCODE_STATUS_MISSING_HUFFMAN,
        JPEG_BASELINE_ENCODE_STATUS_OK, JPEG_BASELINE_ENCODE_STATUS_OVERFLOW,
    },
    CudaJpegBaselineEncodeStatus,
};
use crate::{error::CudaError, memory::CudaDeviceBuffer};

pub(super) struct CudaJpegBaselineHuffmanLaunch<'a> {
    pub(super) dc_luma: &'a CudaDeviceBuffer,
    pub(super) ac_luma: &'a CudaDeviceBuffer,
    pub(super) dc_chroma: &'a CudaDeviceBuffer,
    pub(super) ac_chroma: &'a CudaDeviceBuffer,
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
