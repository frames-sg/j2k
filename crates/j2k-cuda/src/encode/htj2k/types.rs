// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::EncodedHtJ2kCodeBlock;
use j2k_cuda_runtime::{CudaBufferPool, CudaContext, CudaHtj2kEncodeResources};

use super::super::CudaEncodeStageTimings;

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) struct CudaEncodedHtj2kTile {
    pub(in crate::encode) tile_data: Vec<u8>,
    pub(in crate::encode) deinterleave_dispatches: usize,
    pub(in crate::encode) forward_rct_dispatches: usize,
    pub(in crate::encode) forward_ict_dispatches: usize,
    pub(in crate::encode) forward_dwt53_dispatches: usize,
    pub(in crate::encode) forward_dwt97_dispatches: usize,
    pub(in crate::encode) quantize_jobs: usize,
    pub(in crate::encode) quantize_dispatches: usize,
    pub(in crate::encode) ht_code_block_dispatches: usize,
    pub(in crate::encode) ht_code_block_jobs: usize,
    pub(in crate::encode) packetization_dispatches: usize,
    pub(in crate::encode) timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Default)]
pub(super) struct CudaHtj2kTileEncodeStats {
    pub(super) collect_profile: bool,
    pub(super) deinterleave_dispatches: usize,
    pub(super) forward_rct_dispatches: usize,
    pub(super) forward_ict_dispatches: usize,
    pub(super) forward_dwt53_dispatches: usize,
    pub(super) forward_dwt97_dispatches: usize,
    pub(super) quantize_jobs: usize,
    pub(super) quantize_dispatches: usize,
    pub(super) ht_code_block_dispatches: usize,
    pub(super) ht_code_block_jobs: usize,
    pub(super) timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
pub(super) struct CudaEncodedHtj2kResolution {
    pub(super) subbands: Vec<CudaEncodedHtj2kSubband>,
}

#[cfg(feature = "cuda-runtime")]
pub(super) struct CudaEncodedHtj2kSubband {
    pub(super) code_blocks: Vec<EncodedHtJ2kCodeBlock>,
    pub(super) num_cbs_x: u32,
    pub(super) num_cbs_y: u32,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
pub(super) struct CudaTileSubbandRegion {
    pub(super) x0: u32,
    pub(super) y0: u32,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) stride: u32,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
pub(super) enum CudaTileSubbandKind {
    LowLow,
    HighLow,
    LowHigh,
    HighHigh,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
pub(super) struct CudaHtj2kEncodeRuntime<'a> {
    pub(super) context: &'a CudaContext,
    pub(super) resources: &'a CudaHtj2kEncodeResources,
    pub(super) pool: &'a CudaBufferPool,
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) struct CudaEncodedHtSubband {
    pub(in crate::encode) quantize_dispatches: usize,
    pub(in crate::encode) encode: j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks,
    pub(in crate::encode) timings: CudaEncodeStageTimings,
}
