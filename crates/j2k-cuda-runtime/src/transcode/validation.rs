// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{should_use_pinned_pooled_i16_upload, DctBlockGrid, Dwt97BatchInput, Reversible53Dims};
use crate::{
    bytes::i16_slice_as_bytes,
    context::{ensure_context_ownership, CudaContext},
    error::CudaError,
    memory::{CudaBufferPool, CudaPooledDeviceBuffer},
};

pub(super) const TRANSCODE_POOL_CONTEXT_MISMATCH: &str =
    "CUDA transcode buffer pool must belong to the launch context";

pub(super) fn validate_transcode_pool_context_match(
    matches_context: bool,
) -> Result<(), CudaError> {
    ensure_context_ownership([matches_context], TRANSCODE_POOL_CONTEXT_MISMATCH)
}

pub(super) fn validate_transcode_pool_context(
    context: &CudaContext,
    pool: &CudaBufferPool,
) -> Result<(), CudaError> {
    validate_transcode_pool_context_match(pool.is_owned_by(context))
}

fn transcode_runtime_ptx_available() -> bool {
    crate::build_flags::transcode_kernels_built()
}

pub(super) fn ensure_transcode_runtime_ptx_available() -> Result<(), CudaError> {
    if transcode_runtime_ptx_available() {
        Ok(())
    } else {
        Err(CudaError::InvalidArgument {
            message: "CUDA Oxide transcode PTX was not built; enable j2k-cuda-runtime/cuda-oxide-transcode or a crate cuda-runtime feature that implies it, and use J2K_REQUIRE_CUDA_OXIDE_BUILD=1 on CUDA hosts"
                .to_string(),
        })
    }
}

impl Dwt97BatchInput<'_> {
    pub(super) fn len(self) -> usize {
        match self {
            Self::F32(blocks) => blocks.len(),
            Self::I16(blocks) => blocks.len(),
        }
    }

    pub(super) fn upload(self, pool: &CudaBufferPool) -> Result<CudaPooledDeviceBuffer, CudaError> {
        match self {
            Self::F32(blocks) => pool.upload_f32(blocks),
            Self::I16(blocks) => {
                let bytes = i16_slice_as_bytes(blocks);
                if should_use_pinned_pooled_i16_upload(bytes.len()) {
                    pool.upload_pinned(bytes)
                } else {
                    pool.upload(bytes)
                }
            }
        }
    }
}

pub(crate) fn validate_dct_block_grid(
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
    item_count: usize,
    coeff_len: usize,
    invalid_message: &'static str,
) -> Result<DctBlockGrid, CudaError> {
    let block_count = block_cols
        .checked_mul(block_rows)
        .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
    let covered_w = block_cols
        .checked_mul(8)
        .ok_or(CudaError::LengthTooLarge { len: block_cols })?;
    let covered_h = block_rows
        .checked_mul(8)
        .ok_or(CudaError::LengthTooLarge { len: block_rows })?;
    let per_item_coeffs = block_count
        .checked_mul(64)
        .ok_or(CudaError::LengthTooLarge { len: block_count })?;
    let expected_coeffs =
        per_item_coeffs
            .checked_mul(item_count)
            .ok_or(CudaError::LengthTooLarge {
                len: per_item_coeffs,
            })?;
    if item_count == 0
        || width == 0
        || height == 0
        || width > covered_w
        || height > covered_h
        || coeff_len != expected_coeffs
    {
        return Err(CudaError::InvalidArgument {
            message: invalid_message.to_string(),
        });
    }

    let low_width = width.div_ceil(2);
    let low_height = height.div_ceil(2);
    let high_width = width / 2;
    let high_height = height / 2;
    Ok(DctBlockGrid {
        block_count,
        expected_coeffs,
        low_width,
        low_height,
        high_width,
        high_height,
        dims: Reversible53Dims {
            block_cols: checked_i32(block_cols)?,
            width: checked_i32(width)?,
            height: checked_i32(height)?,
            low_width: checked_i32(low_width)?,
            high_width: checked_i32(high_width)?,
        },
    })
}

pub(crate) fn checked_i32(value: usize) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::LengthTooLarge { len: value })
}

#[cfg(test)]
mod tests;
