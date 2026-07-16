// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

use super::{
    shared::{x_blocks_launch_geometry, COPY_U8_THREADS, COPY_U8_THREADS_CUDA},
    CudaKernel, CudaLaunchGeometry,
};

const J2K_IDWT_COOP_THREADS_SMALL_CUDA: c_uint = 256;
const J2K_IDWT_COOP_THREADS_LARGE_CUDA: c_uint = 512;
const J2K_ENCODE_THREADS_X: c_uint = 16;
pub(super) const J2K_ENCODE_THREADS_Y: c_uint = 16;

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

    #[cfg_attr(
        not(feature = "cuda-oxide-j2k-classic-decode"),
        expect(
            dead_code,
            reason = "classifier is used only by the classic J2K decode feature"
        )
    )]
    pub(crate) fn is_j2k_classic_decode_stage(self) -> bool {
        matches!(self, Self::J2kClassicDecodeCodeblocksMulti)
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
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
const CUDA_OXIDE_J2K_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_encode.ptx"));
#[cfg(feature = "cuda-oxide-j2k-ml")]
const CUDA_OXIDE_J2K_ML_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_ml.ptx"));
#[cfg(feature = "cuda-oxide-j2k-decode-store")]
const CUDA_OXIDE_J2K_DECODE_STORE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_j2k_decode_store.ptx"));
#[cfg(feature = "cuda-oxide-j2k-classic-decode")]
const CUDA_OXIDE_J2K_CLASSIC_DECODE_PTX: &[u8] = include_bytes!(concat!(
    env!("OUT_DIR"),
    "/cuda_oxide_j2k_classic_decode.ptx"
));
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
const HTJ2K_DECODE_CODEBLOCK_THREADS: usize = 32;
const HTJ2K_DECODE_CODEBLOCK_THREADS_CUDA: c_uint = 32;
const CLASSIC_DECODE_CODEBLOCK_THREADS_CUDA: c_uint = 32;
const HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS: usize = 2_048;
const HTJ2K_ENCODE_CODEBLOCK_THREADS_CUDA: c_uint = 128;

pub(crate) fn j2k_forward_rct_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(len, 1, COPY_U8_THREADS)
}

pub(crate) fn j2k_dwt53_launch_geometry(width: u32, height: u32) -> Option<CudaLaunchGeometry> {
    let grid_x = c_uint::try_from(width.div_ceil(J2K_ENCODE_THREADS_X)).ok()?;
    let grid_y = c_uint::try_from(height.div_ceil(J2K_ENCODE_THREADS_Y)).ok()?;
    CudaLaunchGeometry::new(
        (grid_x, grid_y, 1),
        (J2K_ENCODE_THREADS_X, J2K_ENCODE_THREADS_Y, 1),
    )
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
    CudaLaunchGeometry::new((lanes, jobs, 1), (threads, 1, 1))
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
    CudaLaunchGeometry::new((blocks, jobs, 1), (threads, 1, 1))
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
    CudaLaunchGeometry::new((blocks, jobs, 1), (block_x, block_y, 1))
}

pub(crate) fn htj2k_codeblock_launch_geometry(job_count: usize) -> Option<CudaLaunchGeometry> {
    if job_count >= HTJ2K_DECODE_PACKED_BLOCK_MIN_JOBS {
        let jobs = c_uint::try_from(job_count.div_ceil(HTJ2K_DECODE_CODEBLOCK_THREADS)).ok()?;
        CudaLaunchGeometry::new((jobs, 1, 1), (HTJ2K_DECODE_CODEBLOCK_THREADS_CUDA, 1, 1))
    } else {
        let jobs = c_uint::try_from(job_count).ok()?;
        CudaLaunchGeometry::new((jobs, 1, 1), (1, 1, 1))
    }
}

pub(crate) fn htj2k_codeblock_sample_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    CudaLaunchGeometry::new((jobs, 1, 1), (COPY_U8_THREADS_CUDA, 1, 1))
}

pub(crate) fn j2k_classic_codeblock_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    CudaLaunchGeometry::new((jobs, 1, 1), (CLASSIC_DECODE_CODEBLOCK_THREADS_CUDA, 1, 1))
}

pub(crate) fn j2k_store_batch_launch_geometry(
    max_pixels: usize,
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(max_pixels, job_count, COPY_U8_THREADS)
}

pub(crate) fn htj2k_encode_codeblock_launch_geometry(
    job_count: usize,
) -> Option<CudaLaunchGeometry> {
    let jobs = c_uint::try_from(job_count).ok()?;
    CudaLaunchGeometry::new((jobs, 1, 1), (HTJ2K_ENCODE_CODEBLOCK_THREADS_CUDA, 1, 1))
}

pub(crate) fn htj2k_packetize_launch_geometry(packet_count: usize) -> Option<CudaLaunchGeometry> {
    htj2k_codeblock_sample_launch_geometry(packet_count)
}
#[cfg(feature = "cuda-oxide-j2k-ml")]
pub(crate) fn cuda_oxide_j2k_ml_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_ML_PTX
}

#[cfg(feature = "cuda-oxide-j2k-encode")]
pub(crate) fn cuda_oxide_j2k_encode_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_ENCODE_PTX
}

#[cfg(feature = "cuda-oxide-j2k-decode-store")]
pub(crate) fn cuda_oxide_j2k_decode_store_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_DECODE_STORE_PTX
}

#[cfg(feature = "cuda-oxide-j2k-classic-decode")]
pub(crate) fn cuda_oxide_j2k_classic_decode_ptx() -> &'static [u8] {
    CUDA_OXIDE_J2K_CLASSIC_DECODE_PTX
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
