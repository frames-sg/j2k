// SPDX-License-Identifier: MIT OR Apache-2.0

mod geometry;
mod j2k;
mod jpeg;
mod shared;
#[cfg(test)]
mod tests;
mod transcode;

pub(crate) use geometry::CudaLaunchGeometry;
#[cfg(test)]
pub(crate) use geometry::{CUDA_MAX_GRID_DIM_X, CUDA_MAX_GRID_DIM_Y_Z};
#[cfg(feature = "cuda-oxide-htj2k-decode")]
pub(crate) use j2k::cuda_oxide_htj2k_decode_ptx;
#[cfg(feature = "cuda-oxide-htj2k-encode")]
pub(crate) use j2k::cuda_oxide_htj2k_encode_ptx;
#[cfg(feature = "cuda-oxide-j2k-classic-decode")]
pub(crate) use j2k::cuda_oxide_j2k_classic_decode_ptx;
#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) use j2k::cuda_oxide_j2k_decode_store_ptx;
#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) use j2k::cuda_oxide_j2k_dequantize_ptx;
#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) use j2k::cuda_oxide_j2k_encode_ptx;
#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) use j2k::cuda_oxide_j2k_idwt_ptx;
#[cfg(feature = "cuda-oxide-j2k-ml")]
pub(crate) use j2k::cuda_oxide_j2k_ml_ptx;
#[cfg(test)]
use j2k::J2K_ENCODE_THREADS_Y;
pub(crate) use j2k::{
    htj2k_codeblock_launch_geometry, htj2k_codeblock_sample_launch_geometry,
    htj2k_encode_codeblock_launch_geometry, htj2k_packetize_launch_geometry,
    j2k_classic_codeblock_launch_geometry, j2k_dwt53_launch_geometry,
    j2k_forward_rct_launch_geometry, j2k_idwt_multi_1d_launch_geometry,
    j2k_idwt_multi_coop_axis_launch_geometry, j2k_idwt_multi_coop_columns_launch_geometry,
    j2k_idwt_multi_coop_launch_geometry, j2k_store_batch_launch_geometry,
};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) use jpeg::cuda_oxide_jpeg_decode_ptx;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(crate) use jpeg::cuda_oxide_jpeg_encode_ptx;
#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) use shared::cuda_oxide_copy_u8_ptx;
pub(crate) use shared::{copy_u8_launch_geometry, with_grid_y, with_grid_z};
#[cfg(test)]
use shared::{x_blocks_launch_geometry, COPY_U8_THREADS, COPY_U8_THREADS_CUDA};
#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) use transcode::cuda_oxide_transcode_ptx;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
    #[cfg_attr(
        all(not(feature = "cuda-oxide-copy-u8"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the CopyU8 kernel feature"
        )
    )]
    CopyU8,
    #[cfg_attr(
        not(feature = "cuda-oxide-j2k-ml"),
        expect(
            dead_code,
            reason = "variant is used only by the j2k-ml kernel feature"
        )
    )]
    J2kMlConvert,
    J2kDeinterleaveToF32,
    J2kDeinterleaveStridedToF32,
    J2kForwardRct,
    J2kForwardIct,
    J2kForwardDwt53Horizontal,
    J2kForwardDwt53Vertical,
    J2kForwardDwt97Horizontal,
    J2kForwardDwt97Vertical,
    J2kQuantizeSubband,
    J2kQuantizeSubbandStrided,
    Htj2kDecodeCodeblocks,
    Htj2kDecodeCodeblocksMulti,
    Htj2kDecodeCodeblocksMultiCleanupOnly,
    Htj2kDecodeCodeblocksMultiCleanupDequantize,
    J2kClassicDecodeCodeblocksMulti,
    J2kDequantizeHtj2kCodeblocks,
    J2kDequantizeHtj2kCodeblocksMulti,
    J2kDequantizeHtj2kCleanupJobsMulti,
    J2kIdwtInterleave,
    J2kIdwtInterleaveHorizontalMulti,
    J2kIdwtInterleaveHorizontal53Multi,
    J2kIdwtInterleaveHorizontal97Multi,
    J2kIdwtHorizontal53,
    J2kIdwtHorizontal97,
    J2kIdwtVerticalMulti,
    J2kIdwtVertical53Multi,
    J2kIdwtVertical97Multi,
    J2kIdwtVertical97MultiCols4,
    J2kIdwtVertical53,
    J2kIdwtVertical97,
    Htj2kEncodeCodeblocks,
    Htj2kEncodeCodeblocksMultiInput,
    Htj2kEncodeCodeblocksMultiInputCleanup,
    Htj2kEncodeCodeblocksMultiInputCleanup64,
    Htj2kCompactCodeblocks,
    Htj2kPacketizeCleanup,
    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG decode kernel feature"
        )
    )]
    JpegDecodeFast420Rgb8,
    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG decode kernel feature"
        )
    )]
    JpegDecodeFast422Rgb8,
    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG decode kernel feature"
        )
    )]
    JpegDecodeFast444Rgb8,
    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG decode kernel feature"
        )
    )]
    JpegSubsampledPlanesToRgb8,
    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG decode kernel feature"
        )
    )]
    JpegEntropySync420,
    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG decode kernel feature"
        )
    )]
    JpegEntropyOverflow420,
    #[cfg_attr(
        not(feature = "cuda-oxide-jpeg-encode"),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG encode kernel feature"
        )
    )]
    JpegEncodeBaselineEntropy,
    #[cfg_attr(
        not(feature = "cuda-oxide-jpeg-encode"),
        expect(
            dead_code,
            reason = "variant is used only by the JPEG encode kernel feature"
        )
    )]
    JpegEncodeBaselineEntropyBatch,
    J2kInverseMct,
    J2kStoreGray16,
    J2kStoreGray16Batch,
    J2kStoreGrayI16Batch,
    J2kStoreGray8,
    J2kStoreGray8Batch,
    J2kStoreRgb16,
    J2kStoreRgb16Mct,
    J2kStoreRgb8,
    J2kStoreRgb8MctBatch,
    J2kStoreRgb8NativeBatch,
    J2kStoreRgb16NativeBatch,
    J2kStoreRgbI16NativeBatch,
    J2kStoreRgba8NativeBatch,
    J2kStoreRgba16NativeBatch,
    J2kStoreRgbaI16NativeBatch,
    // Coefficient-domain JPEG->HTJ2K transcode (j2k-transcode-cuda).
    TranscodeReversible53Idct,
    TranscodeReversible53VerticalLow,
    TranscodeReversible53VerticalHigh,
    TranscodeReversible53HorizontalLow,
    TranscodeReversible53HorizontalHigh,
    TranscodeDwt97Idct,
    TranscodeDwt97RowLift,
    TranscodeDwt97ColumnLift,
    TranscodeDwt97IdctBatch,
    TranscodeDwt97IdctI16Batch,
    TranscodeDwt97RowLiftBatch,
    TranscodeDwt97RowLiftBatchCoop,
    TranscodeDwt97ColumnLiftBatch,
    TranscodeDwt97QuantizeCodeblocks,
    TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch,
}

impl CudaKernel {
    #[cfg_attr(
        all(not(j2k_cuda_oxide_enabled), not(test)),
        expect(
            dead_code,
            reason = "entrypoint lookup is used only when CUDA Oxide modules are built"
        )
    )]
    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => b"j2k_copy_u8\0",
            Self::J2kMlConvert => b"j2k_ml_convert\0",
            Self::J2kDeinterleaveToF32 => b"j2k_deinterleave_to_f32\0",
            Self::J2kDeinterleaveStridedToF32 => b"j2k_deinterleave_strided_to_f32\0",
            Self::J2kForwardRct => b"j2k_forward_rct\0",
            Self::J2kForwardIct => b"j2k_forward_ict\0",
            Self::J2kForwardDwt53Horizontal => b"j2k_forward_dwt53_horizontal\0",
            Self::J2kForwardDwt53Vertical => b"j2k_forward_dwt53_vertical\0",
            Self::J2kForwardDwt97Horizontal => b"j2k_forward_dwt97_horizontal\0",
            Self::J2kForwardDwt97Vertical => b"j2k_forward_dwt97_vertical\0",
            Self::J2kQuantizeSubband => b"j2k_quantize_subband\0",
            Self::J2kQuantizeSubbandStrided => b"j2k_quantize_subband_strided\0",
            Self::Htj2kDecodeCodeblocks => b"j2k_htj2k_decode_codeblocks\0",
            Self::Htj2kDecodeCodeblocksMulti => b"j2k_htj2k_decode_codeblocks_multi\0",
            Self::Htj2kDecodeCodeblocksMultiCleanupOnly => {
                b"j2k_htj2k_decode_codeblocks_multi_cleanup_only\0"
            }
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                b"j2k_htj2k_decode_codeblocks_multi_cleanup_dequantize\0"
            }
            Self::J2kClassicDecodeCodeblocksMulti => b"j2k_decode_classic_codeblocks_multi\0",
            Self::J2kDequantizeHtj2kCodeblocks => b"j2k_dequantize_htj2k_codeblocks\0",
            Self::J2kDequantizeHtj2kCodeblocksMulti => b"j2k_dequantize_htj2k_codeblocks_multi\0",
            Self::J2kDequantizeHtj2kCleanupJobsMulti => {
                b"j2k_dequantize_htj2k_cleanup_jobs_multi\0"
            }
            Self::J2kIdwtInterleave => b"j2k_idwt_interleave\0",
            Self::J2kIdwtInterleaveHorizontalMulti => b"j2k_idwt_interleave_horizontal_multi\0",
            Self::J2kIdwtInterleaveHorizontal53Multi => {
                b"j2k_idwt_interleave_horizontal_53_multi\0"
            }
            Self::J2kIdwtInterleaveHorizontal97Multi => {
                b"j2k_idwt_interleave_horizontal_97_multi\0"
            }
            Self::J2kIdwtHorizontal53 => b"j2k_idwt_horizontal_53\0",
            Self::J2kIdwtHorizontal97 => b"j2k_idwt_horizontal_97\0",
            Self::J2kIdwtVerticalMulti => b"j2k_idwt_vertical_multi\0",
            Self::J2kIdwtVertical53Multi => b"j2k_idwt_vertical_53_multi\0",
            Self::J2kIdwtVertical97Multi => b"j2k_idwt_vertical_97_multi\0",
            Self::J2kIdwtVertical97MultiCols4 => b"j2k_idwt_vertical_97_multi_cols4\0",
            Self::J2kIdwtVertical53 => b"j2k_idwt_vertical_53\0",
            Self::J2kIdwtVertical97 => b"j2k_idwt_vertical_97\0",
            Self::Htj2kEncodeCodeblocks => b"j2k_htj2k_encode_codeblocks\0",
            Self::Htj2kEncodeCodeblocksMultiInput => b"j2k_htj2k_encode_codeblocks_multi_input\0",
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                b"j2k_htj2k_encode_codeblocks_multi_input_cleanup\0"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                b"j2k_htj2k_encode_codeblocks_multi_input_cleanup_64\0"
            }
            Self::Htj2kCompactCodeblocks => b"j2k_htj2k_compact_codeblocks\0",
            Self::Htj2kPacketizeCleanup => b"j2k_htj2k_packetize_cleanup\0",
            Self::JpegDecodeFast420Rgb8 => b"j2k_jpeg_decode_fast420_rgb8\0",
            Self::JpegDecodeFast422Rgb8 => b"j2k_jpeg_decode_fast422_rgb8\0",
            Self::JpegDecodeFast444Rgb8 => b"j2k_jpeg_decode_fast444_rgb8\0",
            Self::JpegSubsampledPlanesToRgb8 => b"j2k_jpeg_subsampled_planes_to_rgb8\0",
            Self::JpegEntropySync420 => b"j2k_jpeg_entropy_sync420\0",
            Self::JpegEntropyOverflow420 => b"j2k_jpeg_entropy_overflow420\0",
            Self::JpegEncodeBaselineEntropy => b"j2k_jpeg_encode_baseline_entropy\0",
            Self::JpegEncodeBaselineEntropyBatch => b"j2k_jpeg_encode_baseline_entropy_batch\0",
            Self::J2kInverseMct => b"j2k_inverse_mct\0",
            Self::J2kStoreGray16 => b"j2k_store_gray16\0",
            Self::J2kStoreGray16Batch => b"j2k_store_gray16_batch\0",
            Self::J2kStoreGrayI16Batch => b"j2k_store_grayi16_batch\0",
            Self::J2kStoreGray8 => b"j2k_store_gray8\0",
            Self::J2kStoreGray8Batch => b"j2k_store_gray8_batch\0",
            Self::J2kStoreRgb16 => b"j2k_store_rgb16\0",
            Self::J2kStoreRgb16Mct => b"j2k_store_rgb16_mct\0",
            Self::J2kStoreRgb8 => b"j2k_store_rgb8\0",
            Self::J2kStoreRgb8MctBatch => b"j2k_store_rgb8_mct_batch\0",
            Self::J2kStoreRgb8NativeBatch => b"j2k_store_rgb8_native_batch\0",
            Self::J2kStoreRgb16NativeBatch => b"j2k_store_rgb16_native_batch\0",
            Self::J2kStoreRgbI16NativeBatch => b"j2k_store_rgbi16_native_batch\0",
            Self::J2kStoreRgba8NativeBatch => b"j2k_store_rgba8_native_batch\0",
            Self::J2kStoreRgba16NativeBatch => b"j2k_store_rgba16_native_batch\0",
            Self::J2kStoreRgbaI16NativeBatch => b"j2k_store_rgbai16_native_batch\0",
            Self::TranscodeReversible53Idct => b"transcode_reversible53_idct\0",
            Self::TranscodeReversible53VerticalLow => b"transcode_reversible53_vertical_low\0",
            Self::TranscodeReversible53VerticalHigh => b"transcode_reversible53_vertical_high\0",
            Self::TranscodeReversible53HorizontalLow => b"transcode_reversible53_horizontal_low\0",
            Self::TranscodeReversible53HorizontalHigh => {
                b"transcode_reversible53_horizontal_high\0"
            }
            Self::TranscodeDwt97Idct => b"transcode_dwt97_idct\0",
            Self::TranscodeDwt97RowLift => b"transcode_dwt97_row_lift\0",
            Self::TranscodeDwt97ColumnLift => b"transcode_dwt97_column_lift\0",
            Self::TranscodeDwt97IdctBatch => b"transcode_dwt97_idct_batch\0",
            Self::TranscodeDwt97IdctI16Batch => b"transcode_dwt97_idct_i16_batch\0",
            Self::TranscodeDwt97RowLiftBatch => b"transcode_dwt97_row_lift_batch\0",
            Self::TranscodeDwt97RowLiftBatchCoop => b"transcode_dwt97_row_lift_batch_coop\0",
            Self::TranscodeDwt97ColumnLiftBatch => b"transcode_dwt97_column_lift_batch\0",
            Self::TranscodeDwt97QuantizeCodeblocks => b"transcode_dwt97_quantize_codeblocks\0",
            Self::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch => {
                b"transcode_dwt97_column_lift_quantize_codeblocks_batch\0"
            }
        }
    }
}
