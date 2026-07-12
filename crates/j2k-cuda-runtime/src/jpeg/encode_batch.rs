// SPDX-License-Identifier: MIT OR Apache-2.0

//! CUDA JPEG baseline batch launch and bounded output collection.

use super::{
    encode_allocation::checked_batch_private_host_bytes,
    encode_launch::{
        validate_jpeg_encode_status, CudaJpegBaselineEntropyBatchLaunch,
        CudaJpegBaselineHuffmanLaunch, CudaJpegBaselineQuantLaunch,
    },
    encode_validation::CudaJpegBaselineEncodeValidation,
    CudaJpegBaselineEncodeStatus, CudaJpegBaselineEntropyEncodeBatchJob,
};
use crate::{
    allocation::{try_vec_defaulted, try_vec_filled, try_vec_with_capacity},
    bytes::{
        cuda_jpeg_baseline_encode_huffman_table_as_bytes,
        cuda_jpeg_baseline_encode_params_as_bytes, cuda_jpeg_baseline_encode_statuses_as_bytes,
        cuda_jpeg_baseline_encode_statuses_as_bytes_mut,
    },
    context::CudaContext,
    error::CudaError,
    kernels::CudaLaunchGeometry,
    memory::CudaDeviceBuffer,
};

struct LaunchedBatch {
    entropy: CudaDeviceBuffer,
    statuses: Vec<CudaJpegBaselineEncodeStatus>,
}

impl CudaContext {
    pub(super) fn execute_jpeg_baseline_entropy_batch(
        &self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
        external_live_bytes: usize,
        validated: CudaJpegBaselineEncodeValidation,
        geometry: CudaLaunchGeometry,
    ) -> Result<Vec<Vec<u8>>, CudaError> {
        let launched = self.launch_jpeg_baseline_entropy_batch_job(
            job,
            external_live_bytes,
            validated,
            geometry,
        )?;
        Self::collect_jpeg_baseline_entropy_batch(job, external_live_bytes, launched)
    }

    #[expect(
        clippy::similar_names,
        reason = "DC/AC luma/chroma names mirror the four distinct JPEG Huffman table roles"
    )]
    fn launch_jpeg_baseline_entropy_batch_job(
        &self,
        job: &CudaJpegBaselineEntropyEncodeBatchJob<'_>,
        external_live_bytes: usize,
        validated: CudaJpegBaselineEncodeValidation,
        geometry: CudaLaunchGeometry,
    ) -> Result<LaunchedBatch, CudaError> {
        self.inner.set_current()?;
        let entropy = self.allocate(job.entropy_capacity)?;
        let mut statuses: Vec<CudaJpegBaselineEncodeStatus> = try_vec_defaulted(job.params.len())?;
        checked_batch_private_host_bytes(
            external_live_bytes,
            job.params.capacity(),
            job.params.len(),
            statuses.capacity(),
            job.params.len(),
            job.entropy_capacity,
        )?;
        let status_buffer = self.upload(cuda_jpeg_baseline_encode_statuses_as_bytes(&statuses))?;
        let params_buffer = self.upload(cuda_jpeg_baseline_encode_params_as_bytes(&job.params))?;
        let q_luma = self.upload(&job.q_luma)?;
        let q_chroma = self.upload(&job.q_chroma)?;
        let huff_dc_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
            &job.huff_dc_luma,
        ))?;
        let huff_ac_luma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
            &job.huff_ac_luma,
        ))?;
        let huff_dc_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
            &job.huff_dc_chroma,
        ))?;
        let huff_ac_chroma = self.upload(cuda_jpeg_baseline_encode_huffman_table_as_bytes(
            &job.huff_ac_chroma,
        ))?;
        self.launch_jpeg_encode_baseline_entropy_batch(&CudaJpegBaselineEntropyBatchLaunch {
            input: job.input,
            entropy: &entropy,
            status: &status_buffer,
            params: &params_buffer,
            quant: CudaJpegBaselineQuantLaunch {
                luma: &q_luma,
                chroma: &q_chroma,
            },
            huffman: CudaJpegBaselineHuffmanLaunch {
                dc_luma: &huff_dc_luma,
                ac_luma: &huff_ac_luma,
                dc_chroma: &huff_dc_chroma,
                ac_chroma: &huff_ac_chroma,
            },
            tile_count: validated.tile_count,
            geometry,
        })?;
        status_buffer.copy_to_host(cuda_jpeg_baseline_encode_statuses_as_bytes_mut(
            &mut statuses,
        ))?;
        Ok(LaunchedBatch { entropy, statuses })
    }

    fn collect_jpeg_baseline_entropy_batch(
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
    validate_jpeg_encode_status(status, "j2k_jpeg_encode_baseline_entropy_batch")?;
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
