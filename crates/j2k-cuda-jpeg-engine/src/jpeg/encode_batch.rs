// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded output collection for staged CUDA JPEG baseline batches.

use super::{
    encode_allocation::checked_batch_private_host_bytes,
    encode_launch::validate_jpeg_encode_status, CudaJpegBaselineEncodeStatus,
    CudaJpegBaselineEntropyEncodeBatchJob,
};
use crate::{
    allocation::{try_vec_filled, try_vec_with_capacity},
    error::CudaError,
    memory::CudaDeviceBuffer,
    JpegCudaEngine,
};

pub(super) struct LaunchedBatch {
    pub(super) entropy: CudaDeviceBuffer,
    pub(super) statuses: Vec<CudaJpegBaselineEncodeStatus>,
}

impl JpegCudaEngine<'_> {
    pub(super) fn collect_jpeg_baseline_entropy_batch(
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
        external_live_bytes: usize,
        launched: LaunchedBatch,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        let LaunchedBatch { entropy, statuses } = launched;
        let mut out = try_vec_with_capacity(job.params.len())?;
        checked_batch_private_host_bytes(
            external_live_bytes,
            job.params.capacity(),
            job.params.len(),
            statuses.capacity(),
            out.capacity(),
            job.entropy_capacity,
        )?;
        let mut output_payload_capacity = 0usize;
        for (index, (status, params)) in statuses.iter().copied().zip(&job.params).enumerate() {
            let mut chunk = checked_entropy_chunk(status, params, job.entropy_capacity)?;
            entropy
                .copy_range_to_host(
                    usize::try_from(params.entropy_offset_bytes)
                        .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?,
                    &mut chunk,
                )
                .map_err(|error| map_batch_copy_error(error, index))?;
            output_payload_capacity = output_payload_capacity.saturating_add(chunk.capacity());
            out.push(chunk);
            checked_batch_private_host_bytes(
                external_live_bytes,
                job.params.capacity(),
                job.params.len(),
                statuses.capacity(),
                out.capacity(),
                output_payload_capacity,
            )?;
        }
        Ok(out)
    }
}

fn checked_entropy_chunk(
    status: CudaJpegBaselineEncodeStatus,
    params: &super::CudaJpegBaselineEncodeParams,
    total_entropy_capacity: usize,
) -> Result<Vec<u8>, CudaError> {
    validate_jpeg_encode_status(status, "j2k_jpeg_encode_baseline_entropy_from_coeffs_batch")?;
    let entropy_len = usize::try_from(status.entropy_len)
        .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
    let offset = usize::try_from(params.entropy_offset_bytes)
        .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
    let capacity = usize::try_from(params.entropy_capacity)
        .map_err(|_| CudaError::LengthTooLarge { len: usize::MAX })?;
    if entropy_len > capacity {
        return Err(CudaError::OutputTooSmall {
            required: entropy_len,
            have: capacity,
        });
    }
    let end = offset
        .checked_add(entropy_len)
        .ok_or(CudaError::LengthTooLarge { len: usize::MAX })?;
    if end > total_entropy_capacity {
        return Err(CudaError::OutputTooSmall {
            required: end,
            have: total_entropy_capacity,
        });
    }
    try_vec_filled(entropy_len, 0u8)
}

fn map_batch_copy_error(error: CudaError, index: usize) -> CudaError {
    if matches!(error, CudaError::OutputTooSmall { .. }) {
        CudaError::InvalidArgument {
            message: format!("JPEG CUDA encode batch tile {index} entropy range is out of bounds"),
        }
    } else {
        error
    }
}
