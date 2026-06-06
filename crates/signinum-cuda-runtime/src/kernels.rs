use std::os::raw::c_uint;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
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
    J2kIdwtHorizontal,
    J2kIdwtHorizontal53,
    J2kIdwtHorizontal97,
    J2kIdwtVertical,
    J2kIdwtVerticalMulti,
    J2kIdwtVertical53Multi,
    J2kIdwtVertical97Multi,
    J2kIdwtVertical97MultiCols4,
    J2kIdwtVertical53,
    J2kIdwtVertical97,
    Htj2kEncodeCodeblock,
    Htj2kEncodeCodeblocks,
    Htj2kEncodeCodeblocksMultiInput,
    Htj2kEncodeCodeblocksMultiInputCleanup,
    Htj2kEncodeCodeblocksMultiInputCleanup64,
    Htj2kCompactCodeblocks,
    Htj2kPacketizeCleanup,
    #[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
    JpegDecodeFast420Rgb8,
    #[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
    JpegDecodeFast422Rgb8,
    #[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
    JpegDecodeFast444Rgb8,
    #[cfg_attr(not(signinum_cuda_jpeg_decode_ptx_built), allow(dead_code))]
    JpegEntropySync420,
    #[allow(dead_code)]
    JpegEntropyOverflow420,
    J2kInverseDwtSingle,
    J2kInverseMct,
    J2kStoreGray16,
    J2kStoreGray8,
    J2kStoreRgb16,
    J2kStoreRgb16Mct,
    J2kStoreRgb8,
    J2kStoreRgb8Mct,
    J2kStoreRgb8MctBatch,
    // Coefficient-domain JPEG->HTJ2K transcode (signinum-transcode-cuda).
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
    pub(crate) fn ptx(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => COPY_U8_PTX,
            Self::J2kDeinterleaveToF32
            | Self::J2kDeinterleaveStridedToF32
            | Self::J2kForwardRct
            | Self::J2kForwardIct
            | Self::J2kForwardDwt53Horizontal
            | Self::J2kForwardDwt53Vertical
            | Self::J2kForwardDwt97Horizontal
            | Self::J2kForwardDwt97Vertical
            | Self::J2kQuantizeSubband
            | Self::J2kQuantizeSubbandStrided => J2K_ENCODE_PTX,
            Self::Htj2kDecodeCodeblocks
            | Self::Htj2kDecodeCodeblocksMulti
            | Self::Htj2kDecodeCodeblocksMultiCleanupOnly
            | Self::Htj2kDecodeCodeblocksMultiCleanupDequantize
            | Self::J2kDequantizeHtj2kCodeblocks
            | Self::J2kDequantizeHtj2kCodeblocksMulti
            | Self::J2kDequantizeHtj2kCleanupJobsMulti
            | Self::J2kIdwtInterleave
            | Self::J2kIdwtInterleaveHorizontalMulti
            | Self::J2kIdwtInterleaveHorizontal53Multi
            | Self::J2kIdwtInterleaveHorizontal97Multi
            | Self::J2kIdwtHorizontal
            | Self::J2kIdwtHorizontal53
            | Self::J2kIdwtHorizontal97
            | Self::J2kIdwtVertical
            | Self::J2kIdwtVerticalMulti
            | Self::J2kIdwtVertical53Multi
            | Self::J2kIdwtVertical97Multi
            | Self::J2kIdwtVertical97MultiCols4
            | Self::J2kIdwtVertical53
            | Self::J2kIdwtVertical97
            | Self::J2kInverseDwtSingle
            | Self::J2kInverseMct
            | Self::J2kStoreGray16
            | Self::J2kStoreGray8
            | Self::J2kStoreRgb16
            | Self::J2kStoreRgb16Mct
            | Self::J2kStoreRgb8
            | Self::J2kStoreRgb8Mct
            | Self::J2kStoreRgb8MctBatch => HTJ2K_DECODE_PTX,
            Self::Htj2kEncodeCodeblock
            | Self::Htj2kEncodeCodeblocks
            | Self::Htj2kEncodeCodeblocksMultiInput
            | Self::Htj2kEncodeCodeblocksMultiInputCleanup
            | Self::Htj2kEncodeCodeblocksMultiInputCleanup64
            | Self::Htj2kCompactCodeblocks
            | Self::Htj2kPacketizeCleanup => HTJ2K_ENCODE_PTX,
            Self::JpegDecodeFast420Rgb8
            | Self::JpegDecodeFast422Rgb8
            | Self::JpegDecodeFast444Rgb8
            | Self::JpegEntropySync420
            | Self::JpegEntropyOverflow420 => JPEG_DECODE_PTX,
            Self::TranscodeReversible53Idct
            | Self::TranscodeReversible53VerticalLow
            | Self::TranscodeReversible53VerticalHigh
            | Self::TranscodeReversible53HorizontalLow
            | Self::TranscodeReversible53HorizontalHigh
            | Self::TranscodeDwt97Idct
            | Self::TranscodeDwt97RowLift
            | Self::TranscodeDwt97ColumnLift
            | Self::TranscodeDwt97IdctBatch
            | Self::TranscodeDwt97IdctI16Batch
            | Self::TranscodeDwt97RowLiftBatch
            | Self::TranscodeDwt97RowLiftBatchCoop
            | Self::TranscodeDwt97ColumnLiftBatch
            | Self::TranscodeDwt97QuantizeCodeblocks
            | Self::TranscodeDwt97ColumnLiftQuantizeCodeblocksBatch => TRANSCODE_PTX,
        }
    }

    pub(crate) fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::CopyU8 => b"signinum_copy_u8\0",
            Self::J2kDeinterleaveToF32 => b"signinum_j2k_deinterleave_to_f32\0",
            Self::J2kDeinterleaveStridedToF32 => b"signinum_j2k_deinterleave_strided_to_f32\0",
            Self::J2kForwardRct => b"signinum_j2k_forward_rct\0",
            Self::J2kForwardIct => b"signinum_j2k_forward_ict\0",
            Self::J2kForwardDwt53Horizontal => b"signinum_j2k_forward_dwt53_horizontal\0",
            Self::J2kForwardDwt53Vertical => b"signinum_j2k_forward_dwt53_vertical\0",
            Self::J2kForwardDwt97Horizontal => b"signinum_j2k_forward_dwt97_horizontal\0",
            Self::J2kForwardDwt97Vertical => b"signinum_j2k_forward_dwt97_vertical\0",
            Self::J2kQuantizeSubband => b"signinum_j2k_quantize_subband\0",
            Self::J2kQuantizeSubbandStrided => b"signinum_j2k_quantize_subband_strided\0",
            Self::Htj2kDecodeCodeblocks => b"signinum_htj2k_decode_codeblocks\0",
            Self::Htj2kDecodeCodeblocksMulti => b"signinum_htj2k_decode_codeblocks_multi\0",
            Self::Htj2kDecodeCodeblocksMultiCleanupOnly => {
                b"signinum_htj2k_decode_codeblocks_multi_cleanup_only\0"
            }
            Self::Htj2kDecodeCodeblocksMultiCleanupDequantize => {
                b"signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize\0"
            }
            Self::J2kDequantizeHtj2kCodeblocks => b"signinum_j2k_dequantize_htj2k_codeblocks\0",
            Self::J2kDequantizeHtj2kCodeblocksMulti => {
                b"signinum_j2k_dequantize_htj2k_codeblocks_multi\0"
            }
            Self::J2kDequantizeHtj2kCleanupJobsMulti => {
                b"signinum_j2k_dequantize_htj2k_cleanup_jobs_multi\0"
            }
            Self::J2kIdwtInterleave => b"signinum_j2k_idwt_interleave\0",
            Self::J2kIdwtInterleaveHorizontalMulti => {
                b"signinum_j2k_idwt_interleave_horizontal_multi\0"
            }
            Self::J2kIdwtInterleaveHorizontal53Multi => {
                b"signinum_j2k_idwt_interleave_horizontal_53_multi\0"
            }
            Self::J2kIdwtInterleaveHorizontal97Multi => {
                b"signinum_j2k_idwt_interleave_horizontal_97_multi\0"
            }
            Self::J2kIdwtHorizontal => b"signinum_j2k_idwt_horizontal\0",
            Self::J2kIdwtHorizontal53 => b"signinum_j2k_idwt_horizontal_53\0",
            Self::J2kIdwtHorizontal97 => b"signinum_j2k_idwt_horizontal_97\0",
            Self::J2kIdwtVertical => b"signinum_j2k_idwt_vertical\0",
            Self::J2kIdwtVerticalMulti => b"signinum_j2k_idwt_vertical_multi\0",
            Self::J2kIdwtVertical53Multi => b"signinum_j2k_idwt_vertical_53_multi\0",
            Self::J2kIdwtVertical97Multi => b"signinum_j2k_idwt_vertical_97_multi\0",
            Self::J2kIdwtVertical97MultiCols4 => b"signinum_j2k_idwt_vertical_97_multi_cols4\0",
            Self::J2kIdwtVertical53 => b"signinum_j2k_idwt_vertical_53\0",
            Self::J2kIdwtVertical97 => b"signinum_j2k_idwt_vertical_97\0",
            Self::Htj2kEncodeCodeblock => b"signinum_htj2k_encode_codeblock\0",
            Self::Htj2kEncodeCodeblocks => b"signinum_htj2k_encode_codeblocks\0",
            Self::Htj2kEncodeCodeblocksMultiInput => {
                b"signinum_htj2k_encode_codeblocks_multi_input\0"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup => {
                b"signinum_htj2k_encode_codeblocks_multi_input_cleanup\0"
            }
            Self::Htj2kEncodeCodeblocksMultiInputCleanup64 => {
                b"signinum_htj2k_encode_codeblocks_multi_input_cleanup_64\0"
            }
            Self::Htj2kCompactCodeblocks => b"signinum_htj2k_compact_codeblocks\0",
            Self::Htj2kPacketizeCleanup => b"signinum_htj2k_packetize_cleanup\0",
            Self::JpegDecodeFast420Rgb8 => b"signinum_jpeg_decode_fast420_rgb8\0",
            Self::JpegDecodeFast422Rgb8 => b"signinum_jpeg_decode_fast422_rgb8\0",
            Self::JpegDecodeFast444Rgb8 => b"signinum_jpeg_decode_fast444_rgb8\0",
            Self::JpegEntropySync420 => b"signinum_jpeg_entropy_sync420\0",
            Self::JpegEntropyOverflow420 => b"signinum_jpeg_entropy_overflow420\0",
            Self::J2kInverseDwtSingle => b"signinum_j2k_inverse_dwt_single\0",
            Self::J2kInverseMct => b"signinum_j2k_inverse_mct\0",
            Self::J2kStoreGray16 => b"signinum_j2k_store_gray16\0",
            Self::J2kStoreGray8 => b"signinum_j2k_store_gray8\0",
            Self::J2kStoreRgb16 => b"signinum_j2k_store_rgb16\0",
            Self::J2kStoreRgb16Mct => b"signinum_j2k_store_rgb16_mct\0",
            Self::J2kStoreRgb8 => b"signinum_j2k_store_rgb8\0",
            Self::J2kStoreRgb8Mct => b"signinum_j2k_store_rgb8_mct\0",
            Self::J2kStoreRgb8MctBatch => b"signinum_j2k_store_rgb8_mct_batch\0",
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
    let blocks = c_uint::try_from(len.div_ceil(COPY_U8_THREADS)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
}

const COPY_U8_THREADS: usize = 256;
const COPY_U8_THREADS_CUDA: c_uint = 256;
const J2K_IDWT_COOP_THREADS_SMALL_CUDA: c_uint = 256;
const J2K_IDWT_COOP_THREADS_LARGE_CUDA: c_uint = 512;
const J2K_ENCODE_THREADS_X: c_uint = 16;
const J2K_ENCODE_THREADS_Y: c_uint = 16;
const J2K_ENCODE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/j2k_encode_kernels.ptx"));
const HTJ2K_DECODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/htj2k_decode_kernels.ptx"));
const HTJ2K_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/htj2k_encode_kernels.ptx"));
const JPEG_DECODE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/jpeg_decode_kernels.ptx"));
// Always resolves: build.rs writes a placeholder empty module when nvcc is
// absent (the dispatch checks `signinum_cuda_transcode_ptx_built` before load).
const TRANSCODE_PTX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/transcode_kernels.ptx"));
const HTJ2K_DECODE_CODEBLOCK_THREADS: usize = 32;
const HTJ2K_DECODE_CODEBLOCK_THREADS_CUDA: c_uint = 32;
const HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS: usize = 2_048;
const HTJ2K_ENCODE_CODEBLOCK_THREADS_CUDA: c_uint = 128;

pub(crate) fn j2k_forward_rct_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    let blocks = c_uint::try_from(len.div_ceil(COPY_U8_THREADS)).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, 1, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
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
    let blocks = c_uint::try_from(max_len.div_ceil(COPY_U8_THREADS)).ok()?;
    let jobs = c_uint::try_from(job_count).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, jobs, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
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
    let blocks = c_uint::try_from(max_pixels.div_ceil(COPY_U8_THREADS)).ok()?;
    let jobs = c_uint::try_from(job_count).ok()?;
    Some(CudaLaunchGeometry {
        grid: (blocks, jobs, 1),
        block: (COPY_U8_THREADS_CUDA, 1, 1),
    })
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

const COPY_U8_PTX: &[u8] = concat!(
    r"
.version 7.0
.target sm_52
.address_size 64

.visible .entry signinum_copy_u8(
    .param .u64 dst,
    .param .u64 src,
    .param .u64 len
)
{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<7>;
    .reg .b16 %u;

    ld.param.u64 %rd1, [dst];
    ld.param.u64 %rd2, [src];
    ld.param.u64 %rd3, [len];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;
    cvt.u64.u32 %rd4, %r4;
    setp.ge.u64 %p, %rd4, %rd3;
    @%p bra DONE;
    add.u64 %rd5, %rd2, %rd4;
    ld.global.u8 %u, [%rd5];
    add.u64 %rd6, %rd1, %rd4;
    st.global.u8 [%rd6], %u;
DONE:
    ret;
}
",
    "\0"
)
.as_bytes();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_u8_kernel_metadata_matches_embedded_ptx() {
        let ptx = CudaKernel::CopyU8.ptx();
        assert_eq!(ptx.last(), Some(&0));
        let source = std::str::from_utf8(&ptx[..ptx.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_copy_u8("));
        assert_eq!(CudaKernel::CopyU8.entrypoint(), b"signinum_copy_u8\0");
    }

    #[test]
    fn jpeg_decode_kernel_metadata_matches_source_entrypoints() {
        assert_eq!(
            CudaKernel::JpegEntropySync420.entrypoint(),
            b"signinum_jpeg_entropy_sync420\0"
        );
        assert_eq!(
            CudaKernel::JpegEntropyOverflow420.entrypoint(),
            b"signinum_jpeg_entropy_overflow420\0"
        );

        let cuda_source = include_str!("jpeg_decode_kernels.cu");
        assert!(cuda_source.contains("extern \"C\" __global__ void signinum_jpeg_entropy_sync420("));
        assert!(
            cuda_source.contains("extern \"C\" __global__ void signinum_jpeg_entropy_overflow420(")
        );
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
    fn j2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(J2K_ENCODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::J2kDeinterleaveToF32.entrypoint(),
            b"signinum_j2k_deinterleave_to_f32\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardRct.entrypoint(),
            b"signinum_j2k_forward_rct\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardIct.entrypoint(),
            b"signinum_j2k_forward_ict\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Horizontal.entrypoint(),
            b"signinum_j2k_forward_dwt53_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt53Vertical.entrypoint(),
            b"signinum_j2k_forward_dwt53_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt97Horizontal.entrypoint(),
            b"signinum_j2k_forward_dwt97_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kForwardDwt97Vertical.entrypoint(),
            b"signinum_j2k_forward_dwt97_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kQuantizeSubband.entrypoint(),
            b"signinum_j2k_quantize_subband\0"
        );
        assert_eq!(
            CudaKernel::J2kQuantizeSubbandStrided.entrypoint(),
            b"signinum_j2k_quantize_subband_strided\0"
        );
        let source =
            std::str::from_utf8(&J2K_ENCODE_PTX[..J2K_ENCODE_PTX.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_j2k_deinterleave_to_f32("));
        assert!(source.contains(".visible .entry signinum_j2k_quantize_subband_strided("));
    }

    #[test]
    fn j2k_encode_kernel_uses_native_irreversible_delta_formula() {
        let cuda_source = include_str!("j2k_encode_kernels.cu");
        assert!(cuda_source.contains("const int exponent = int(range_bits) - int(step_exponent);"));
        assert!(!cuda_source.contains("const int exponent = int(step_exponent) - int(range_bits);"));
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn htj2k_decode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(HTJ2K_DECODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::Htj2kDecodeCodeblocks.entrypoint(),
            b"signinum_htj2k_decode_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::Htj2kDecodeCodeblocksMulti.entrypoint(),
            b"signinum_htj2k_decode_codeblocks_multi\0"
        );
        assert_eq!(
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupOnly.entrypoint(),
            b"signinum_htj2k_decode_codeblocks_multi_cleanup_only\0"
        );
        assert_eq!(
            CudaKernel::Htj2kDecodeCodeblocksMultiCleanupDequantize.entrypoint(),
            b"signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize\0"
        );
        assert_eq!(
            CudaKernel::J2kDequantizeHtj2kCodeblocks.entrypoint(),
            b"signinum_j2k_dequantize_htj2k_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::J2kDequantizeHtj2kCodeblocksMulti.entrypoint(),
            b"signinum_j2k_dequantize_htj2k_codeblocks_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kDequantizeHtj2kCleanupJobsMulti.entrypoint(),
            b"signinum_j2k_dequantize_htj2k_cleanup_jobs_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtInterleave.entrypoint(),
            b"signinum_j2k_idwt_interleave\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtInterleaveHorizontalMulti.entrypoint(),
            b"signinum_j2k_idwt_interleave_horizontal_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtInterleaveHorizontal53Multi.entrypoint(),
            b"signinum_j2k_idwt_interleave_horizontal_53_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtInterleaveHorizontal97Multi.entrypoint(),
            b"signinum_j2k_idwt_interleave_horizontal_97_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtHorizontal.entrypoint(),
            b"signinum_j2k_idwt_horizontal\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtVertical.entrypoint(),
            b"signinum_j2k_idwt_vertical\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtVerticalMulti.entrypoint(),
            b"signinum_j2k_idwt_vertical_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtVertical53Multi.entrypoint(),
            b"signinum_j2k_idwt_vertical_53_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtVertical97Multi.entrypoint(),
            b"signinum_j2k_idwt_vertical_97_multi\0"
        );
        assert_eq!(
            CudaKernel::J2kIdwtVertical97MultiCols4.entrypoint(),
            b"signinum_j2k_idwt_vertical_97_multi_cols4\0"
        );
        assert_eq!(
            CudaKernel::J2kInverseDwtSingle.entrypoint(),
            b"signinum_j2k_inverse_dwt_single\0"
        );
        assert_eq!(
            CudaKernel::J2kInverseMct.entrypoint(),
            b"signinum_j2k_inverse_mct\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreGray8.entrypoint(),
            b"signinum_j2k_store_gray8\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreGray16.entrypoint(),
            b"signinum_j2k_store_gray16\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb8.entrypoint(),
            b"signinum_j2k_store_rgb8\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb8Mct.entrypoint(),
            b"signinum_j2k_store_rgb8_mct\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb8MctBatch.entrypoint(),
            b"signinum_j2k_store_rgb8_mct_batch\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb16.entrypoint(),
            b"signinum_j2k_store_rgb16\0"
        );
        assert_eq!(
            CudaKernel::J2kStoreRgb16Mct.entrypoint(),
            b"signinum_j2k_store_rgb16_mct\0"
        );
        let source =
            std::str::from_utf8(&HTJ2K_DECODE_PTX[..HTJ2K_DECODE_PTX.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_htj2k_decode_codeblocks_multi("));
        assert!(
            source.contains(".visible .entry signinum_htj2k_decode_codeblocks_multi_cleanup_only(")
        );
        assert!(source.contains(
            ".visible .entry signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize("
        ));
        let cuda_source = include_str!("htj2k_decode_kernels.cu");
        assert!(cuda_source.contains(
            "extern \"C\" __global__ void signinum_htj2k_decode_codeblocks_multi_cleanup_dequantize("
        ));
        assert!(source.contains(".visible .entry signinum_j2k_dequantize_htj2k_codeblocks("));
        assert!(source.contains(".visible .entry signinum_j2k_dequantize_htj2k_codeblocks_multi("));
        assert!(
            source.contains(".visible .entry signinum_j2k_dequantize_htj2k_cleanup_jobs_multi(")
        );
        assert!(source.contains(".visible .entry signinum_j2k_idwt_interleave("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_interleave_horizontal_multi("));
        assert!(
            source.contains(".visible .entry signinum_j2k_idwt_interleave_horizontal_53_multi(")
        );
        assert!(
            source.contains(".visible .entry signinum_j2k_idwt_interleave_horizontal_97_multi(")
        );
        assert!(source.contains(".visible .entry signinum_j2k_idwt_horizontal("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_multi("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_53_multi("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_97_multi("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_horizontal_53("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_53("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_horizontal_97("));
        assert!(source.contains(".visible .entry signinum_j2k_idwt_vertical_97("));
        assert!(source.contains(".visible .entry signinum_j2k_store_rgb8_mct("));
        assert!(source.contains(".visible .entry signinum_j2k_store_rgb8_mct_batch("));
        assert!(source.contains(".visible .entry signinum_j2k_store_rgb16_mct("));
    }

    #[test]
    fn htj2k_decode_cleanup_kernels_guard_padded_launch_threads() {
        let cuda_source = include_str!("htj2k_decode_kernels.cu");
        assert!(cuda_source.contains("if (gid >= job_count)"));
    }

    #[test]
    fn htj2k_encode_kernel_metadata_matches_generated_ptx() {
        assert_eq!(HTJ2K_ENCODE_PTX.last(), Some(&0));
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblock.entrypoint(),
            b"signinum_htj2k_encode_codeblock\0"
        );
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblocks.entrypoint(),
            b"signinum_htj2k_encode_codeblocks\0"
        );
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblocksMultiInput.entrypoint(),
            b"signinum_htj2k_encode_codeblocks_multi_input\0"
        );
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup.entrypoint(),
            b"signinum_htj2k_encode_codeblocks_multi_input_cleanup\0"
        );
        assert_eq!(
            CudaKernel::Htj2kEncodeCodeblocksMultiInputCleanup64.entrypoint(),
            b"signinum_htj2k_encode_codeblocks_multi_input_cleanup_64\0"
        );
        assert_eq!(
            CudaKernel::Htj2kPacketizeCleanup.entrypoint(),
            b"signinum_htj2k_packetize_cleanup\0"
        );
        let source =
            std::str::from_utf8(&HTJ2K_ENCODE_PTX[..HTJ2K_ENCODE_PTX.len() - 1]).expect("ptx utf8");
        assert!(source.contains(".visible .entry signinum_htj2k_encode_codeblocks("));
        assert!(source.contains(".visible .entry signinum_htj2k_encode_codeblocks_multi_input("));
        if cfg!(signinum_cuda_htj2k_encode_ptx_built) {
            assert!(source
                .contains(".visible .entry signinum_htj2k_encode_codeblocks_multi_input_cleanup("));
            assert!(source.contains(
                ".visible .entry signinum_htj2k_encode_codeblocks_multi_input_cleanup_64("
            ));
        }
        assert!(source.contains(".visible .entry signinum_htj2k_packetize_cleanup("));
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains("j2k_ht_reduce_max_magnitude_cooperative"));
        assert!(cuda_source.contains("j2k_packet_copy_body_cooperative"));
    }

    #[test]
    fn htj2k_encode_kernel_reports_zero_passes_for_all_zero_codeblocks() {
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains(
            "j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_OK, 0u, 0u, 0u, params.total_bitplanes);"
        ));
        assert!(!cuda_source.contains(
            "j2k_set_ht_encode_status(status, J2K_ENCODE_STATUS_OK, 0u, 0u, 1u, params.total_bitplanes);"
        ));
    }

    #[test]
    fn htj2k_encode_kernel_uses_width_bounded_cleanup_scratch_clear() {
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains("FIXED_64 ? 34u : j2k_ht_cleanup_scratch_entries(width)"));
        assert!(cuda_source.contains(
            "params.width == 64u && params.height == 64u && params.coefficient_stride == 64u"
        ));
        assert!(!cuda_source.contains("idx < 513u; ++idx"));
    }

    #[test]
    fn htj2k_encode_kernel_uses_shared_cleanup_scratch() {
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains("__shared__ uchar cleanup_e_val[J2K_HT_SIGPROP_SCRATCH];"));
        assert!(cuda_source.contains("__shared__ uchar cleanup_cx_val[J2K_HT_SIGPROP_SCRATCH];"));
        assert!(cuda_source.contains("cleanup_e_val,\n        cleanup_cx_val"));
    }

    #[test]
    fn htj2k_encode_kernel_sizes_max_reduction_for_encode_launch_threads() {
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains("J2K_HT_ENCODE_THREADS = 128u"));
        assert!(cuda_source.contains("__shared__ uint block_max[J2K_HT_ENCODE_THREADS];"));
        assert!(!cuda_source.contains("__shared__ uint block_max[256];"));
    }

    #[test]
    fn htj2k_encode_multi_input_kernel_declares_launch_bounds() {
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        let expected = concat!(
            "extern \"C\" __global__ void __",
            "launch_bounds__(J2K_HT_ENCODE_THREADS) ",
            "signinum_htj2k_encode_codeblocks_multi_input"
        );
        assert!(cuda_source.contains(expected));
        assert!(cuda_source.contains("signinum_htj2k_encode_codeblocks_multi_input_cleanup"));
    }

    #[test]
    fn htj2k_encode_kernel_has_contiguous_max_reduction_fast_path() {
        let cuda_source = include_str!("htj2k_encode_kernels.cu");
        assert!(cuda_source.contains("if (coefficient_stride == width)"));
        assert!(cuda_source.contains("j2k_classic_magnitude(coefficients[sample])"));
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
        // All transcode kernels share the one translation unit's PTX.
        assert_eq!(
            CudaKernel::TranscodeDwt97QuantizeCodeblocks.ptx().as_ptr(),
            TRANSCODE_PTX.as_ptr()
        );

        // The placeholder PTX is empty when nvcc is absent; only validate entry
        // points are present once the runner actually compiled the kernels.
        if cfg!(signinum_cuda_transcode_ptx_built) {
            let source = std::str::from_utf8(&TRANSCODE_PTX[..TRANSCODE_PTX.len() - 1])
                .expect("transcode ptx utf8");
            assert!(source.contains(".visible .entry transcode_dwt97_idct_batch("));
            assert!(source.contains(".visible .entry transcode_dwt97_idct_i16_batch("));
            assert!(source.contains(".visible .entry transcode_dwt97_row_lift_batch("));
            assert!(source.contains(".visible .entry transcode_dwt97_row_lift_batch_coop("));
            assert!(source.contains(".visible .entry transcode_dwt97_column_lift_batch("));
            assert!(source.contains(".visible .entry transcode_dwt97_quantize_codeblocks("));
            assert!(source.contains(
                ".visible .entry transcode_dwt97_column_lift_quantize_codeblocks_batch("
            ));
        }
    }

    #[test]
    fn transcode_dwt97_idct_uses_precomputed_basis_table() {
        let cuda_source = include_str!("transcode_kernels.cu");
        assert!(cuda_source.contains("DWT97_IDCT8_BASIS"));
        assert!(cuda_source.contains("DWT97_IDCT8_BASIS[sample_idx * 8 + freq]"));
        assert!(!cuda_source.contains("sqrtf(1.0f / 8.0f)"));
        assert!(!cuda_source.contains("cosf(angle)"));
    }

    #[test]
    fn transcode_dwt97_idct_unrolls_fixed_basis_loops() {
        let cuda_source = include_str!("transcode_kernels.cu");
        assert!(cuda_source.contains("transcode_dwt97_idct_unroll_guard"));
        assert!(
            cuda_source.contains("#pragma unroll\n    for (int freq_y = 0; freq_y < 8; ++freq_y)")
        );
        assert!(cuda_source
            .contains("#pragma unroll\n        for (int freq_x = 0; freq_x < 8; ++freq_x)"));
    }

    #[test]
    fn transcode_dwt97_batch_row_lift_has_cooperative_kernel() {
        let cuda_source = include_str!("transcode_kernels.cu");
        assert!(cuda_source.contains("transcode_dwt97_row_lift_batch_coop("));
        assert!(cuda_source.contains("DWT97_ROW_LIFT_MAX_WIDTH"));
        assert!(cuda_source.contains(
            "__shared__ f32 rows[DWT97_ROW_LIFT_ROWS_PER_BLOCK][DWT97_ROW_LIFT_MAX_WIDTH];"
        ));
    }

    #[test]
    fn copy_u8_launch_geometry_rounds_up_to_256_thread_blocks() {
        assert_eq!(copy_u8_launch_geometry(1).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(256).unwrap().grid, (1, 1, 1));
        assert_eq!(copy_u8_launch_geometry(257).unwrap().grid, (2, 1, 1));
    }

    #[test]
    fn j2k_dwt53_launch_geometry_uses_16_by_16_thread_blocks() {
        let geometry = j2k_dwt53_launch_geometry(17, 33).unwrap();
        assert_eq!(geometry.grid, (2, 3, 1));
        assert_eq!(geometry.block, (16, 16, 1));
    }
}
