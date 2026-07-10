// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::kernels::CudaKernel;

/// Bundled CUDA kernel identifiers that can be preloaded by runtime internals.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernelName {
    CopyU8,
    Htj2kDecodeCodeblocks,
    Htj2kDecodeCodeblocksMultiCleanupDequantize,
    J2kDequantizeHtj2kCodeblocks,
    J2kDequantizeHtj2kCodeblocksMulti,
    J2kDequantizeHtj2kCleanupJobsMulti,
    J2kIdwtInterleave,
    J2kIdwtInterleaveHorizontal53Multi,
    J2kIdwtInterleaveHorizontal97Multi,
    J2kIdwtHorizontal53,
    J2kIdwtHorizontal97,
    J2kIdwtVertical53Multi,
    J2kIdwtVertical97Multi,
    J2kIdwtVertical97MultiCols4,
    J2kIdwtVertical53,
    J2kIdwtVertical97,
    J2kInverseMct,
    J2kStoreGray8,
    J2kStoreGray16,
    J2kStoreRgb8,
    J2kStoreRgb8MctBatch,
    J2kStoreRgb16,
    J2kStoreRgb16Mct,
    Htj2kEncodeCodeblocks,
    Htj2kEncodeCodeblocksMultiInput,
    Htj2kEncodeCodeblocksMultiInputCleanup,
    Htj2kEncodeCodeblocksMultiInputCleanup64,
    Htj2kCompactCodeblocks,
    Htj2kPacketizeCleanup,
}

impl CudaKernelName {
    pub(crate) fn kernel(self) -> CudaKernel {
        match self {
            Self::CopyU8 => CudaKernel::CopyU8,
            Self::Htj2kDecodeCodeblocks => CudaKernel::Htj2kDecodeCodeblocks,
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize
            }
            Self::J2kDequantizeHtj2kCodeblocks => CudaKernel::J2kDequantizeHtj2kCodeblocks,
            Self::J2kDequantizeHtj2kCodeblocksMulti => {
                CudaKernel::J2kDequantizeHtj2kCodeblocksMulti
            }
            Self::J2kDequantizeHtj2kCleanupJobsMulti => {
                CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti
            }
            Self::J2kIdwtInterleave => CudaKernel::J2kIdwtInterleave,
            Self::J2kIdwtInterleaveHorizontal53Multi => {
                CudaKernel::J2kIdwtInterleaveHorizontal53Multi
            }
            Self::J2kIdwtInterleaveHorizontal97Multi => {
                CudaKernel::J2kIdwtInterleaveHorizontal97Multi
            }
            Self::J2kIdwtHorizontal53 => CudaKernel::J2kIdwtHorizontal53,
            Self::J2kIdwtHorizontal97 => CudaKernel::J2kIdwtHorizontal97,
            Self::J2kIdwtVertical53Multi => CudaKernel::J2kIdwtVertical53Multi,
            Self::J2kIdwtVertical97Multi => CudaKernel::J2kIdwtVertical97Multi,
            Self::J2kIdwtVertical97MultiCols4 => CudaKernel::J2kIdwtVertical97MultiCols4,
            Self::J2kIdwtVertical53 => CudaKernel::J2kIdwtVertical53,
            Self::J2kIdwtVertical97 => CudaKernel::J2kIdwtVertical97,
            Self::J2kInverseMct => CudaKernel::J2kInverseMct,
            Self::J2kStoreGray8 => CudaKernel::J2kStoreGray8,
            Self::J2kStoreGray16 => CudaKernel::J2kStoreGray16,
            Self::J2kStoreRgb8 => CudaKernel::J2kStoreRgb8,
            Self::J2kStoreRgb8MctBatch => CudaKernel::J2kStoreRgb8MctBatch,
            Self::J2kStoreRgb16 => CudaKernel::J2kStoreRgb16,
            Self::J2kStoreRgb16Mct => CudaKernel::J2kStoreRgb16Mct,
            Self::Htj2kEncodeCodeblocks => CudaKernel::Htj2kEncodeCodeblocks,
            Self::Htj2kEncodeCodeblocksMultiInput => CudaKernel::Htj2kEncodeCodeblocksMultiInput,
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64
            }
            Self::Htj2kCompactCodeblocks => CudaKernel::Htj2kCompactCodeblocks,
            Self::Htj2kPacketizeCleanup => CudaKernel::Htj2kPacketizeCleanup,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static str {
        match self {
            Self::CopyU8 => "j2k_copy_u8",
            Self::Htj2kDecodeCodeblocks => "j2k_htj2k_decode_codeblocks",
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                "j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize"
            }
            Self::J2kDequantizeHtj2kCodeblocks => "j2k_dequantize_htj2k_codeblocks",
            Self::J2kDequantizeHtj2kCodeblocksMulti => "j2k_dequantize_htj2k_codeblocks_multi",
            Self::J2kDequantizeHtj2kCleanupJobsMulti => "j2k_dequantize_htj2k_cleanup_jobs_multi",
            Self::J2kIdwtInterleave => "j2k_idwt_interleave",
            Self::J2kIdwtInterleaveHorizontal53Multi => "j2k_idwt_interleave_horizontal_53_multi",
            Self::J2kIdwtInterleaveHorizontal97Multi => "j2k_idwt_interleave_horizontal_97_multi",
            Self::J2kIdwtHorizontal53 => "j2k_idwt_horizontal_53",
            Self::J2kIdwtHorizontal97 => "j2k_idwt_horizontal_97",
            Self::J2kIdwtVertical53Multi => "j2k_idwt_vertical_53_multi",
            Self::J2kIdwtVertical97Multi => "j2k_idwt_vertical_97_multi",
            Self::J2kIdwtVertical97MultiCols4 => "j2k_idwt_vertical_97_multi_cols4",
            Self::J2kIdwtVertical53 => "j2k_idwt_vertical_53",
            Self::J2kIdwtVertical97 => "j2k_idwt_vertical_97",
            Self::J2kInverseMct => "j2k_inverse_mct",
            Self::J2kStoreGray8 => "j2k_store_gray8",
            Self::J2kStoreGray16 => "j2k_store_gray16",
            Self::J2kStoreRgb8 => "j2k_store_rgb8",
            Self::J2kStoreRgb8MctBatch => "j2k_store_rgb8_mct_batch",
            Self::J2kStoreRgb16 => "j2k_store_rgb16",
            Self::J2kStoreRgb16Mct => "j2k_store_rgb16_mct",
            Self::Htj2kEncodeCodeblocks => "j2k_htj2k_encode_codeblocks",
            Self::Htj2kEncodeCodeblocksMultiInput => "j2k_htj2k_encode_codeblocks_multi_input",
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                "j2k_htj2k_encode_codeblocks_multi_input_cleanup"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                "j2k_htj2k_encode_codeblocks_multi_input_cleanup_64"
            }
            Self::Htj2kCompactCodeblocks => "j2k_htj2k_compact_codeblocks",
            Self::Htj2kPacketizeCleanup => "j2k_htj2k_packetize_cleanup",
        }
    }
}

/// Metadata for a preloaded CUDA kernel module entry point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaKernelModule {
    pub(crate) entrypoint: &'static str,
}

impl CudaKernelModule {
    pub(crate) fn entrypoint(&self) -> &'static str {
        self.entrypoint
    }
}
