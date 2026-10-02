// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::build_flags::PINNED_POOLED_I16_UPLOAD_MAX_BYTES;

mod dwt97;
mod htj2k97;
mod launch;
mod readback;
mod reversible53;
mod types;
mod validation;

pub use self::types::{
    CudaDwt97BatchGeometry, CudaDwt97BatchStageTimings, CudaDwt97BatchWithPoolRequest,
    CudaHtj2k97CodeblockBands, CudaHtj2k97CodeblockBatchWithPoolRequest,
    CudaHtj2k97DeviceCodeblockBands, CudaHtj2k97I16CodeblockBatchWithPoolRequest,
    CudaHtj2k97QuantizeParams, CudaTranscodeDwt97Bands, CudaTranscodeReversible53Bands,
};
pub(crate) use self::{
    types::{
        DctBlockGrid, Dwt97BatchDeviceBands, Dwt97BatchInput, Dwt97CodeblockBandBuffers,
        Reversible53Dims,
    },
    validation::{checked_i32, validate_dct_block_grid},
};

pub(crate) fn should_use_pinned_pooled_i16_upload(byte_len: usize) -> bool {
    byte_len <= PINNED_POOLED_I16_UPLOAD_MAX_BYTES
}
