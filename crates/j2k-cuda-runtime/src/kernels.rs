use std::os::raw::c_uint;

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
    J2kStoreGray8,
    J2kStoreRgb16,
    J2kStoreRgb16Mct,
    J2kStoreRgb8,
    J2kStoreRgb8MctBatch,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaLaunchGeometry {
    pub grid: (c_uint, c_uint, c_uint),
    pub block: (c_uint, c_uint, c_uint),
}

impl CudaKernel {
    pub(crate) fn is_j2k_encode_stage(self) -> bool {
        matches!(
            self,
            Self::J2kDeinterleaveToF32
                | Self::J2kDeinterleaveStridedToF32
                | Self::J2kForwardRct
                | Self::J2kForwardIct
                | Self::J2kForwardDwt53Horizontal
                | Self::J2kForwardDwt53Vertical
                | Self::J2kForwardDwt97Horizontal
                | Self::J2kForwardDwt97Vertical
                | Self::J2kQuantizeSubband
                | Self::J2kQuantizeSubbandStrided
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-j2k-encode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the J2K encode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_j2k_encode_stage(self) -> bool {
        self.is_j2k_encode_stage()
            || matches!(
                self,
                Self::Htj2kCompactCodeblocks | Self::Htj2kPacketizeCleanup
            )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-j2k-decode-store"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the J2K store kernel feature"
        )
    )]
    pub(crate) fn is_j2k_decode_store_stage(self) -> bool {
        matches!(
            self,
            Self::J2kInverseMct
                | Self::J2kStoreGray16
                | Self::J2kStoreGray8
                | Self::J2kStoreRgb16
                | Self::J2kStoreRgb16Mct
                | Self::J2kStoreRgb8
                | Self::J2kStoreRgb8MctBatch
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-j2k-dequantize"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the J2K dequantize kernel feature"
        )
    )]
    pub(crate) fn is_j2k_dequantize_stage(self) -> bool {
        matches!(
            self,
            Self::J2kDequantizeHtj2kCodeblocks
                | Self::J2kDequantizeHtj2kCodeblocksMulti
                | Self::J2kDequantizeHtj2kCleanupJobsMulti
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-htj2k-decode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the HTJ2K decode kernel feature"
        )
    )]
    pub(crate) fn is_htj2k_decode_stage(self) -> bool {
        matches!(
            self,
            Self::Htj2kDecodeCodeblocks
                | Self::Htj2kDecodeCodeblocksMulti
                | Self::Htj2kDecodeCodeblocksMultiCleanupOnly
                | Self::Htj2kDecodeCodeblocksMultiCleanupDequantize
        )
    }

    pub(crate) fn is_htj2k_encode_codeblock_stage(self) -> bool {
        matches!(
            self,
            Self::Htj2kEncodeCodeblocks
                | Self::Htj2kEncodeCodeblocksMultiInput
                | Self::Htj2kEncodeCodeblocksMultiInputCleanup
                | Self::Htj2kEncodeCodeblocksMultiInputCleanup64
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-j2k-idwt"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the J2K IDWT kernel feature"
        )
    )]
    pub(crate) fn is_j2k_idwt_stage(self) -> bool {
        matches!(
            self,
            Self::J2kIdwtInterleave
                | Self::J2kIdwtInterleaveHorizontalMulti
                | Self::J2kIdwtInterleaveHorizontal53Multi
                | Self::J2kIdwtInterleaveHorizontal97Multi
                | Self::J2kIdwtHorizontal53
                | Self::J2kIdwtHorizontal97
                | Self::J2kIdwtVerticalMulti
                | Self::J2kIdwtVertical53Multi
                | Self::J2kIdwtVertical97Multi
                | Self::J2kIdwtVertical97MultiCols4
                | Self::J2kIdwtVertical53
                | Self::J2kIdwtVertical97
        )
    }

    pub(crate) fn is_transcode_reversible53_stage(self) -> bool {
        matches!(
            self,
            Self::TranscodeReversible53Idct
                | Self::TranscodeReversible53VerticalLow
                | Self::TranscodeReversible53VerticalHigh
                | Self::TranscodeReversible53HorizontalLow
                | Self::TranscodeReversible53HorizontalHigh
        )
    }

    pub(crate) fn is_transcode_dwt97_single_stage(self) -> bool {
        matches!(
            self,
            Self::TranscodeDwt97Idct | Self::TranscodeDwt97RowLift | Self::TranscodeDwt97ColumnLift
        )
    }

    pub(crate) fn is_transcode_dwt97_batch_stage(self) -> bool {
        matches!(
            self,
            Self::TranscodeDwt97IdctBatch
                | Self::TranscodeDwt97IdctI16Batch
                | Self::TranscodeDwt97RowLiftBatch
                | Self::TranscodeDwt97RowLiftBatchCoop
                | Self::TranscodeDwt97ColumnLiftBatch
                | Self::TranscodeDwt97QuantizeCodeblocks
                | Self::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-transcode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the transcode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_transcode_stage(self) -> bool {
        self.is_transcode_reversible53_stage()
            || self.is_transcode_dwt97_single_stage()
            || self.is_transcode_dwt97_batch_stage()
    }

    pub(crate) fn is_jpeg_entropy_stage(self) -> bool {
        matches!(
            self,
            Self::JpegEntropySync420 | Self::JpegEntropyOverflow420
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the JPEG decode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_jpeg_decode_stage(self) -> bool {
        self.is_jpeg_entropy_stage()
            || matches!(
                self,
                Self::JpegDecodeFast420Rgb8
                    | Self::JpegDecodeFast422Rgb8
                    | Self::JpegDecodeFast444Rgb8
            )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-encode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the JPEG encode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_jpeg_encode_stage(self) -> bool {
        matches!(
            self,
            Self::JpegEncodeBaselineEntropy | Self::JpegEncodeBaselineEntropyBatch
        )
    }

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
            Self::JpegEntropySync420 => b"j2k_jpeg_entropy_sync420\0",
            Self::JpegEntropyOverflow420 => b"j2k_jpeg_entropy_overflow420\0",
            Self::JpegEncodeBaselineEntropy => b"j2k_jpeg_encode_baseline_entropy\0",
            Self::JpegEncodeBaselineEntropyBatch => b"j2k_jpeg_encode_baseline_entropy_batch\0",
            Self::J2kInverseMct => b"j2k_inverse_mct\0",
            Self::J2kStoreGray16 => b"j2k_store_gray16\0",
            Self::J2kStoreGray8 => b"j2k_store_gray8\0",
            Self::J2kStoreRgb16 => b"j2k_store_rgb16\0",
            Self::J2kStoreRgb16Mct => b"j2k_store_rgb16_mct\0",
            Self::J2kStoreRgb8 => b"j2k_store_rgb8\0",
            Self::J2kStoreRgb8MctBatch => b"j2k_store_rgb8_mct_batch\0",
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

pub(crate) fn copy_u8_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(len, 1, COPY_U8_THREADS)
}

const COPY_U8_THREADS: usize = 256;
const COPY_U8_THREADS_CUDA: c_uint = 256;
const J2K_IDWT_COOP_THREADS_SMALL_CUDA: c_uint = 256;
const J2K_IDWT_COOP_THREADS_LARGE_CUDA: c_uint = 512;
const J2K_ENCODE_THREADS_X: c_uint = 16;
const J2K_ENCODE_THREADS_Y: c_uint = 16;
#[cfg(feature = "cuda-oxide-copy-u8")]
const CUDA_OXIDE_COPY_U8_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_copy_u8.ptx"));
#[cfg(feature = "cuda-oxide-j2k-encode")]
const CUDA_OXIDE_J2K_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_encode.ptx"));
#[cfg(feature = "cuda-oxide-j2k-decode-store")]
const CUDA_OXIDE_J2K_DECODE_STORE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_decode_store.ptx"));
#[cfg(feature = "cuda-oxide-j2k-dequantize")]
const CUDA_OXIDE_J2K_DEQUANTIZE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_dequantize.ptx"));
#[cfg(feature = "cuda-oxide-j2k-idwt")]
const CUDA_OXIDE_J2K_IDWT_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_idwt.ptx"));
#[cfg(feature = "cuda-oxide-htj2k-decode")]
const CUDA_OXIDE_HTJ2K_DECODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_htj2k_decode.ptx"));
#[cfg(feature = "cuda-oxide-htj2k-encode")]
const CUDA_OXIDE_HTJ2K_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_htj2k_encode.ptx"));
#[cfg(feature = "cuda-oxide-transcode")]
const CUDA_OXIDE_TRANSCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_transcode.ptx"));
#[cfg(feature = "cuda-oxide-jpeg-decode")]
const CUDA_OXIDE_JPEG_DECODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_decode.ptx"));
#[cfg(feature = "cuda-oxide-jpeg-encode")]
const CUDA_OXIDE_JPEG_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_encode.ptx"));
const HTJ2K_DECODE_CODEBLOCK_THREADS: usize = 32;
const HTJ2K_DECODE_CODEBLOCK_THREADS_CUDA: c_uint = 32;
const HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS: usize = 2_048;
const HTJ2K_ENCODE_CODEBLOCK_THREADS_CUDA: c_uint = 128;

pub(crate) fn j2k_forward_rct_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(len, 1, COPY_U8_THREADS)
}

pub(crate) fn j2k_dwt53_launch_geometry(width: u32, height: u32) -> Option<CudaLaunchGeometry> {
    let grid_x = c_uint::try_from(width.div_ceil(J2K_ENCODE_THREADS_X)).ok()?;
    let grid_y = c_uint::try_from(height.div_ceil(J2K_ENCODE_THREADS_Y)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (grid_x, grid_y, 1),
        block: (J2K_ENCODE_THREADS_X, J2K_ENCODE_THREADS_Y, 1),
    })
}

pub(crate) fn j2k_idwt_multi_1d_launch_geometry(
    max_len: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(max_len, job_count, COPY_U8_THREADS)
}

pub(crate) fn j2k_idwt_multi_coop_launch_geometry(
    max_len: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let lanes = c_uint::try_from(max_len).ok()?;
    let jobs = c_uint::try_from(job_count).ok()?;
    let threads = if max_len > COPY_U8_THREADS {
        J2K_IDWT_COOP_THREADS_LARGE_CUDA
    } else {
        J2K_IDWT_COOP_THREADS_SMALL_CUDA
    };
    Some(CudaLaunchGeometry {
        grid: (lanes, jobs, 1),
        block: (threads, 1, 1),
    })
}

pub(crate) fn j2k_idwt_multi_coop_axis_launch_geometry(
    work_items: usize,
    lane_count: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let blocks = c_uint::try_from(work_items).ok()?;
    let jobs = c_uint::try_from(job_count).ok()?;
    let threads = if lane_count > COPY_U8_THREADS {
        J2K_IDWT_COOP_THREADS_LARGE_CUDA
    } else {
        J2K_IDWT_COOP_THREADS_SMALL_CUDA
    };
    Some(CudaLaunchGeometry {
        grid: (blocks, jobs, 1),
        block: (threads, 1, 1),
    })
}

pub(crate) fn j2k_idwt_multi_coop_columns_launch_geometry(
    columns: usize,
    rows: usize,
    job_count: usize,
    columns_per_block: usize,
) -> Option<CudaLaunchGeometry> {
    if rows == 0 || columns_per_block == 0 || rows.saturating_mul(columns_per_block) > 1024 {
        return None;
    }
    let blocks = c_uint::try_from(columns.div_ceil(columns_per_block)).ok()?;
    let jobs = c_uint::try_from(job_count).ok()?;
    let block_x = c_uint::try_from(columns_per_block).ok()?;
    let block_y = c_uint::try_from(rows).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, jobs, 1),
        block: (block_x, block_y, 1),
    })
}

pub(crate) fn htj2k_codeblock_launch_geometry(job_count: usize) -> Option<CudaLaunchGeometry> {
    if job_count >= HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS {
        let jobs = c_uint::try_from(job_count.div_ceil(HTJ2K_DECODE_CODEBLOCK_THREADS)).ok()?;
        Some(CudaLaunchGeometry {
            grid: (jobs, 1, 1),
            block: (HTJ2K_DECODE_CODEBLOCK_THREADS_CUDA, 1, 1),
        })
    } else {
        let jobs = c_uint::try_from(job_count).ok()?;
        Some(CudaLaunchGeometry {
            grid: (jobs, 1, 1),
            block: (1, 1, 1),
        })
    }
}

pub(crate) fn htj2k_codeblock_sample_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    Some(CudaLaunchGeometry {
        grid: (jobs, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

pub(crate) fn j2k_store_batch_launch_geometry(
    max_pixels: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(max_pixels, job_count, COPY_U8_THREADS)
}

fn x_blocks_launch_geometry(
    work_items: usize,
    grid_y: usize,
    threads_per_block: usize,
) -> Option<CudaLaunchGeometry> {
    if threads_per_block == 0 {
        return None;
    }
    let blocks = c_uint::try_from(work_items.div_ceil(threads_per_block)).ok()?;
    let grid_y = c_uint::try_from(grid_y).ok()?;
    let block_x = c_uint::try_from(threads_per_block).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, grid_y, 1),
        block: (block_x, 1, 1),
    })
}

pub(crate) fn with_grid_y(base: CudaLaunchGeometry, grid_y: c_uint) -> CudaLaunchGeometry {
    CudaLaunchGeometry {
        grid: (base.grid.0, grid_y, base.grid.2),
        block: base.block,
    }
}

pub(crate) fn with_grid_z(base: CudaLaunchGeometry, grid_z: c_uint) -> CudaLaunchGeometry {
    CudaLaunchGeometry {
        grid: (base.grid.0, base.grid.1, grid_z),
        block: base.block,
    }
}

pub(crate) fn htj2k_encode_codeblock_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    Some(CudaLaunchGeometry {
        grid: (jobs, 1, 1),
        block: (HTJ2K_ENCODE_CODEBLOCK_THREADS_CUDA, 1, 1),
    })
}

pub(crate) fn htj2k_packetize_launch_geometry(packet_count: usize) -> Option<CudaLaunchGeometry> {
    htj2k_codeblock_sample_launch_geometry(packet_count)
}

#[cfg(feature = "cuda-oxide-copy-u8")]
pub(crate) fn cuda_oxide_copy_u8_ptx() -> &'static [u8] {
    CUDA_OXIDE_COPY_U8_PTX
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) fn cuda_oxide_j2k_encode_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_ENCODE_PTX
}

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) fn cuda_oxide_j2k_decode_store_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_DECODE_STORE_PTX
}

#[cfg(feature = "cuda-oxide-j2k-dequantize")]
pub(crate) fn cuda_oxide_j2k_dequantize_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_DEQUANTIZE_PTX
}

#[cfg(feature = "cuda-oxide-j2k-idwt")]
pub(crate) fn cuda_oxide_j2k_idwt_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_IDWT_PTX
}

#[cfg(feature = "cuda-oxide-htj2k-decode")]
pub(crate) fn cuda_oxide_htj2k_decode_ptx() -> &'static [u8] {
    CUDA_OXIDE_HTJ2K_DECODE_PTX
}

#[cfg(feature = "cuda-oxide-htj2k-encode")]
pub(crate) fn cuda_oxide_htj2k_encode_ptx() -> &'static [u8] {
    CUDA_OXIDE_HTJ2K_ENCODE_PTX
}

#[cfg(feature = "cuda-oxide-transcode")]
pub(crate) fn cuda_oxide_transcode_ptx() -> &'static [u8] {
    CUDA_OXIDE_TRANSCODE_PTX
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn cuda_oxide_jpeg_decode_ptx() -> &'static [u8] {
    CUDA_OXIDE_JPEG_DECODE_PTX
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(crate) fn cuda_oxide_jpeg_encode_ptx() -> &'static [u8] {
    CUDA_OXIDE_JPEG_ENCODE_PTX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_inventory_forbids_test_only_orphan_entrypoints() {
        let kernel_source = include_str!("kernels.rs");
        let production_kernel_source = kernel_source
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("production kernel source");
        assert!(
            !production_kernel_source.contains("#[cfg_attr(not(test), allow(dead_code))]"),
            "production CUDA kernels must not use test-only dead-code exemptions"
        );

        let context_source = include_str!("context.rs");
        for variant in [
            "J2kIdwtHorizontal",
            "J2kIdwtVertical",
            "Htj2kEncodeCodeblock",
            "J2kInverseDwtSingle",
            "J2kStoreRgb8Mct",
        ] {
            assert!(
                !production_kernel_source.contains(&format!("{variant},")),
                "orphan CUDA kernel variant returned: {variant}"
            );
            assert!(
                !context_source.contains(&format!("{variant},")),
                "test kernel inventory must not retain orphan variant: {variant}"
            );
        }

        for (source, entrypoint) in [
            (
                include_str!("cuda_oxide_j2k_idwt/simt/src/main.rs"),
                "j2k_idwt_horizontal",
            ),
            (
                include_str!("cuda_oxide_j2k_idwt/simt/src/main.rs"),
                "j2k_idwt_vertical",
            ),
            (
                include_str!("cuda_oxide_j2k_idwt/simt/src/main.rs"),
                "j2k_inverse_dwt_single",
            ),
            (
                include_str!("cuda_oxide_htj2k_encode/simt/src/main.rs"),
                "j2k_htj2k_encode_codeblock",
            ),
            (
                include_str!("cuda_oxide_j2k_decode_store/simt/src/main.rs"),
                "j2k_store_rgb8_mct",
            ),
        ] {
            assert!(
                !source.contains(&format!("fn {entrypoint}(")),
                "orphan CUDA device entrypoint returned: {entrypoint}"
            );
        }
    }

    #[cfg(all(feature = "cuda-oxide-copy-u8", j2k_cuda_oxide_copy_u8_built))]
    #[test]
    fn cuda_oxide_copy_u8_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_copy_u8_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry j2k_copy_u8("));
        assert_eq!(CudaKernel::CopyU8.entrypoint(), b"j2k_copy_u8\0");
    }

    #[test]
    fn jpeg_decode_entrypoints_are_stable() {
        assert_eq!(CudaKernel::CopyU8.entrypoint(), b"j2k_copy_u8\0");
        assert_eq!(
            CudaKernel::JpegDecodeFast420Rgb8.entrypoint(),
            b"j2k_jpeg_decode_fast420_rgb8\0"
        );
        assert_eq!(
            CudaKernel::JpegDecodeFast422Rgb8.entrypoint(),
            b"j2k_jpeg_decode_fast422_rgb8\0"
        );
        assert_eq!(
            CudaKernel::JpegDecodeFast444Rgb8.entrypoint(),
            b"j2k_jpeg_decode_fast444_rgb8\0"
        );
        assert_eq!(
            CudaKernel::JpegEntropySync420.entrypoint(),
            b"j2k_jpeg_entropy_sync420\0"
        );
        assert_eq!(
            CudaKernel::JpegEntropyOverflow420.entrypoint(),
            b"j2k_jpeg_entropy_overflow420\0"
        );
    }

    #[cfg(all(feature = "cuda-oxide-jpeg-decode", j2k_cuda_oxide_jpeg_decode_built))]
    #[test]
    fn cuda_oxide_jpeg_decode_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_jpeg_decode_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::JpegDecodeFast420Rgb8,
            CudaKernel::JpegDecodeFast422Rgb8,
            CudaKernel::JpegDecodeFast444Rgb8,
            CudaKernel::JpegEntropySync420,
            CudaKernel::JpegEntropyOverflow420,
        ];
        for kernel in kernels {
            assert!(kernel.is_cuda_oxide_jpeg_decode_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide JPEG decode entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(feature = "cuda-oxide-jpeg-encode", j2k_cuda_oxide_jpeg_encode_built))]
    #[test]
    fn cuda_oxide_jpeg_encode_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_jpeg_encode_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::JpegEncodeBaselineEntropy,
            CudaKernel::JpegEncodeBaselineEntropyBatch,
        ];
        for kernel in kernels {
            assert!(kernel.is_cuda_oxide_jpeg_encode_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide JPEG encode entrypoint {entrypoint}"
            );
        }
    }

    #[test]
    fn htj2k_sample_geometry_uses_threads_with_one_block_per_codeblock() {
        let geometry = htj2k_codeblock_sample_launch_geometry(3).expect("geometry");
        assert_eq!(geometry.grid, (3, 1, 1));
        assert_eq!(geometry.block, (COPY_U8_THREADS_CUDA, 1, 1));
    }

    #[test]
    fn htj2k_cleanup_decode_geometry_packs_large_batches_into_warps() {
        let small_geometry = htj2k_codeblock_launch_geometry(1_200).expect("small geometry");
        assert_eq!(small_geometry.grid, (1_200, 1, 1));
        assert_eq!(small_geometry.block, (1, 1, 1));

        let large_geometry = htj2k_codeblock_launch_geometry(2_048).expect("large geometry");
        assert_eq!(large_geometry.grid, (64, 1, 1));
        assert_eq!(large_geometry.block, (32, 1, 1));
    }

    #[test]
    fn htj2k_encode_geometry_uses_cooperative_threads_per_codeblock() {
        let geometry = htj2k_encode_codeblock_launch_geometry(327).expect("geometry");
        assert_eq!(geometry.grid, (327, 1, 1));
        assert_eq!(geometry.block, (128, 1, 1));
    }

    #[test]
    fn htj2k_packetize_geometry_uses_cooperative_threads_per_packet() {
        let geometry = htj2k_packetize_launch_geometry(5).expect("geometry");
        assert_eq!(geometry.grid, (5, 1, 1));
        assert_eq!(geometry.block, (COPY_U8_THREADS_CUDA, 1, 1));
    }

    #[test]
    fn j2k_encode_entrypoints_are_stable() {
        assert_eq!(
            CudaKernel::J2kDeinterleaveToF32.entrypoint(),
            b"j2k_deinterleave_to_f32\0"
        );
        assert_eq!(CudaKernel::J2kForwardRct.entrypoint(), b"j2k_forward_rct\0");
        assert_eq!(CudaKernel::J2kForwardIct.entrypoint(), b"j2k_forward_ict\0");
        assert_eq!(
            CudaKernel::J2kForwardDwt53Horizontal.entrypoint(),
            b"j2k_forward_dwt53_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Vertical.entrypoint(),
            b"j2k_forward_dwt53_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt97Horizontal.entrypoint(),
            b"j2k_forward_dwt97_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt97Vertical.entrypoint(),
            b"j2k_forward_dwt97_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kQuantizeSubband.entrypoint(),
            b"j2k_quantize_subband\0"
        );
        assert_eq!(
            CudaKernel::J2kQuantizeSubbandStrided.entrypoint(),
            b"j2k_quantize_subband_strided\0"
        );
    }

    #[cfg(all(feature = "cuda-oxide-j2k-encode", j2k_cuda_oxide_j2k_encode_built))]
    #[test]
    fn cuda_oxide_j2k_encode_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_j2k_encode_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::J2kDeinterleaveToF32,
            CudaKernel::J2kDeinterleaveStridedToF32,
            CudaKernel::J2kForwardRct,
            CudaKernel::J2kForwardIct,
            CudaKernel::J2kForwardDwt53Horizontal,
            CudaKernel::J2kForwardDwt53Vertical,
            CudaKernel::J2kForwardDwt97Horizontal,
            CudaKernel::J2kForwardDwt97Vertical,
            CudaKernel::J2kQuantizeSubband,
            CudaKernel::J2kQuantizeSubbandStrided,
            CudaKernel::Htj2kCompactCodeblocks,
            CudaKernel::Htj2kPacketizeCleanup,
        ];
        for kernel in kernels {
            assert!(kernel.is_cuda_oxide_j2k_encode_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide J2K encode entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(
        feature = "cuda-oxide-j2k-decode-store",
        j2k_cuda_oxide_j2k_decode_store_built
    ))]
    #[test]
    fn cuda_oxide_j2k_decode_store_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_j2k_decode_store_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::J2kInverseMct,
            CudaKernel::J2kStoreGray8,
            CudaKernel::J2kStoreGray16,
            CudaKernel::J2kStoreRgb8,
            CudaKernel::J2kStoreRgb8MctBatch,
            CudaKernel::J2kStoreRgb16,
            CudaKernel::J2kStoreRgb16Mct,
        ];
        for kernel in kernels {
            assert!(kernel.is_j2k_decode_store_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide J2K decode-store entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(
        feature = "cuda-oxide-j2k-dequantize",
        j2k_cuda_oxide_j2k_dequantize_built
    ))]
    #[test]
    fn cuda_oxide_j2k_dequantize_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_j2k_dequantize_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::J2kDequantizeHtj2kCodeblocks,
            CudaKernel::J2kDequantizeHtj2kCodeblocksMulti,
            CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti,
        ];
        for kernel in kernels {
            assert!(kernel.is_j2k_dequantize_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide J2K dequantize entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(feature = "cuda-oxide-j2k-idwt", j2k_cuda_oxide_j2k_idwt_built))]
    #[test]
    fn cuda_oxide_j2k_idwt_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_j2k_idwt_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::J2kIdwtInterleave,
            CudaKernel::J2kIdwtInterleaveHorizontalMulti,
            CudaKernel::J2kIdwtInterleaveHorizontal53Multi,
            CudaKernel::J2kIdwtInterleaveHorizontal97Multi,
            CudaKernel::J2kIdwtHorizontal53,
            CudaKernel::J2kIdwtHorizontal97,
            CudaKernel::J2kIdwtVerticalMulti,
            CudaKernel::J2kIdwtVertical53Multi,
            CudaKernel::J2kIdwtVertical97Multi,
            CudaKernel::J2kIdwtVertical97MultiCols4,
            CudaKernel::J2kIdwtVertical53,
            CudaKernel::J2kIdwtVertical97,
        ];
        for kernel in kernels {
            assert!(kernel.is_j2k_idwt_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide J2K IDWT entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
    #[test]
    fn cuda_oxide_transcode_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_transcode_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::TranscodeReversible53Idct,
            CudaKernel::TranscodeReversible53VerticalLow,
            CudaKernel::TranscodeReversible53VerticalHigh,
            CudaKernel::TranscodeReversible53HorizontalLow,
            CudaKernel::TranscodeReversible53HorizontalHigh,
            CudaKernel::TranscodeDwt97Idct,
            CudaKernel::TranscodeDwt97RowLift,
            CudaKernel::TranscodeDwt97ColumnLift,
            CudaKernel::TranscodeDwt97IdctBatch,
            CudaKernel::TranscodeDwt97IdctI16Batch,
            CudaKernel::TranscodeDwt97RowLiftBatch,
            CudaKernel::TranscodeDwt97RowLiftBatchCoop,
            CudaKernel::TranscodeDwt97ColumnLiftBatch,
            CudaKernel::TranscodeDwt97QuantizeCodeblocks,
            CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch,
        ];
        for kernel in kernels {
            assert!(kernel.is_cuda_oxide_transcode_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide transcode entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(feature = "cuda-oxide-htj2k-decode", j2k_cuda_oxide_htj2k_decode_built))]
    #[test]
    fn cuda_oxide_htj2k_decode_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_htj2k_decode_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::Htj2kDecodeCodeblocks,
            CudaKernel::Htj2kDecodeCodeblocksMulti,
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly,
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize,
        ];
        for kernel in kernels {
            assert!(kernel.is_htj2k_decode_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide HTJ2K decode entrypoint {entrypoint}"
            );
        }
    }

    #[cfg(all(feature = "cuda-oxide-htj2k-encode", j2k_cuda_oxide_htj2k_encode_built))]
    #[test]
    fn cuda_oxide_htj2k_encode_kernel_metadata_matches_generated_ptx() {
        let ptx = cuda_oxide_htj2k_encode_ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        let kernels = [
            CudaKernel::Htj2kEncodeCodeblocks,
            CudaKernel::Htj2kEncodeCodeblocksMultiInput,
            CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup,
            CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64,
        ];
        for kernel in kernels {
            assert!(kernel.is_htj2k_encode_codeblock_stage());
            let entrypoint =
                std::str::from_utf8(&kernel.entrypoint()[..kernel.entrypoint().len() - 1])
                    .expect("entrypoint utf8");
            assert!(
                source.contains(&format!(".visible .entry {entrypoint}(")),
                "missing cuda-oxide HTJ2K encode entrypoint {entrypoint}"
            );
        }
    }

    #[test]
    fn transcode_kernel_entrypoints_match_names() {
        assert_eq!(
            CudaKernel::TranscodeDwt97Idct.entrypoint(),
            b"transcode_dwt97_idct\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97RowLift.entrypoint(),
            b"transcode_dwt97_row_lift\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97ColumnLift.entrypoint(),
            b"transcode_dwt97_column_lift\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97IdctBatch.entrypoint(),
            b"transcode_dwt97_idct_batch\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97IdctI16Batch.entrypoint(),
            b"transcode_dwt97_idct_i16_batch\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97RowLiftBatch.entrypoint(),
            b"transcode_dwt97_row_lift_batch\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97RowLiftBatchCoop.entrypoint(),
            b"transcode_dwt97_row_lift_batch_coop\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97ColumnLiftBatch.entrypoint(),
            b"transcode_dwt97_column_lift_batch\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97QuantizeCodeblocks.entrypoint(),
            b"transcode_dwt97_quantize_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch.entrypoint(),
            b"transcode_dwt97_column_lift_quantize_codeblocks_batch\0"
        );
    }

    #[test]
    fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
        assert_eq!(copy_u8_launch_geometry(1).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(256).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(257).unwrap().grid, (2, 1, 1));
    }

    #[test]
    fn x_blocks_launch_geometry_rounds_work_items_and_preserves_y_grid() {
        let geometry = x_blocks_launch_geometry(513, 7, COPY_U8_THREADS).unwrap();

        assert_eq!(geometry.grid, (3, 7, 1));
        assert_eq!(geometry.block, (COPY_U8_THREADS_CUDA, 1, 1));
    }

    #[test]
    fn x_blocks_launch_geometry_rejects_zero_threads() {
        assert_eq!(x_blocks_launch_geometry(513, 7, 0), None);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn x_blocks_launch_geometry_rejects_grid_dimensions_above_cuda_uint() {
        assert_eq!(x_blocks_launch_geometry(usize::MAX, usize::MAX, 1), None);
    }

    #[test]
    fn with_grid_y_preserves_block_and_other_grid_axes() {
        let base = CudaLaunchGeometry {
            grid: (2, 3, 4),
            block: (16, 8, 1),
        };

        let geometry = with_grid_y(base, 9);

        assert_eq!(geometry.grid, (2, 9, 4));
        assert_eq!(geometry.block, base.block);
    }

    #[test]
    fn with_grid_z_preserves_block_and_other_grid_axes() {
        let base = CudaLaunchGeometry {
            grid: (2, 3, 4),
            block: (16, 8, 1),
        };

        let geometry = with_grid_z(base, 11);

        assert_eq!(geometry.grid, (2, 3, 11));
        assert_eq!(geometry.block, base.block);
    }

    #[test]
    fn j2k_dwt53_launch_geometry_uses_16_by_16_thread_blocks() {
        let geometry = j2k_dwt53_launch_geometry(17, 33).unwrap();
        assert_eq!(geometry.grid, (2, 3, 1));
        assert_eq!(geometry.block, (16, 16, 1));
    }
}
