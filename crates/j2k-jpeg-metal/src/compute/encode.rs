// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal baseline JPEG entropy submission and bounded host readback.

use std::mem::{size_of, size_of_val};

use metal::MTLSize;

use super::{
    commit_and_wait_jpeg, new_command_buffer, new_compute_command_encoder, with_runtime_for_session,
};
use crate::abi::{
    JpegBaselineEncodeHuffmanTable, JpegBaselineEncodeParams, JpegBaselineEncodeStatus,
    JpegBaselineEntropyEncodeBatchJob, JpegBaselineEntropyEncodeJob,
    JPEG_BASELINE_ENCODE_STATUS_OK,
};
use crate::buffers::{
    checked_buffer_read, checked_buffer_slice, checked_buffer_slice_at, new_shared_buffer,
    new_shared_buffer_with_slice,
};
use crate::compute::status::jpeg_baseline_encode_status_error;
use crate::{encode::allocation as encode_allocation, Error};

pub(crate) fn encode_jpeg_baseline_entropy_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    encode_allocation::checked_single_output_bytes(job.entropy_capacity)?;
    with_runtime_for_session(session, |runtime| {
        let entropy_buffer = new_shared_buffer(&runtime.device, job.entropy_capacity)?;
        let status = JpegBaselineEncodeStatus::default();
        let status_buffer =
            new_shared_buffer_with_slice(&runtime.device, std::slice::from_ref(&status))?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.jpeg_baseline_encode_pipeline);
        encoder.set_buffer(0, Some(job.input), job.input_offset as u64);
        encoder.set_buffer(1, Some(&entropy_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<JpegBaselineEncodeParams>() as u64,
            (&raw const job.params).cast(),
        );
        encoder.set_bytes(
            4,
            size_of_val(&job.q_luma) as u64,
            job.q_luma.as_ptr().cast(),
        );
        encoder.set_bytes(
            5,
            size_of_val(&job.q_chroma) as u64,
            job.q_chroma.as_ptr().cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_luma).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_luma).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_chroma).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_chroma).cast(),
        );
        encoder.dispatch_threads(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;

        let status = checked_buffer_read::<JpegBaselineEncodeStatus>(
            &status_buffer,
            "baseline encode status",
        )?;
        if status.code != JPEG_BASELINE_ENCODE_STATUS_OK {
            return Err(jpeg_baseline_encode_status_error(status));
        }
        let entropy_len = usize::try_from(status.entropy_len).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal encode entropy length exceeds usize".to_string(),
        })?;
        if entropy_len > job.entropy_capacity {
            return Err(Error::MetalKernel {
                message: "JPEG Baseline Metal encode reported length exceeds output capacity"
                    .to_string(),
            });
        }
        let entropy =
            checked_buffer_slice::<u8>(&entropy_buffer, entropy_len, "baseline encode entropy")?;
        encode_allocation::checked_single_output_bytes(entropy.capacity())?;
        Ok(entropy)
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "the entropy batch path keeps shared Metal buffers, per-tile descriptors, command submission, and readback in one lifetime scope"
)]
pub(crate) fn encode_jpeg_baseline_entropy_batch_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeBatchJob<'_>,
) -> Result<Vec<Vec<u8>>, Error> {
    if job.params.is_empty() {
        return Ok(Vec::new());
    }
    encode_allocation::checked_batch_runtime_bytes::<
        JpegBaselineEncodeParams,
        JpegBaselineEncodeStatus,
    >(
        job.params.capacity(),
        job.params.len(),
        job.params.len(),
        job.entropy_capacity,
    )?;
    with_runtime_for_session(session, |runtime| {
        let entropy_buffer = new_shared_buffer(&runtime.device, job.entropy_capacity)?;
        let statuses = encode_allocation::try_vec_filled(
            job.params.len(),
            JpegBaselineEncodeStatus::default(),
        )?;
        encode_allocation::checked_batch_runtime_bytes::<
            JpegBaselineEncodeParams,
            JpegBaselineEncodeStatus,
        >(
            job.params.capacity(),
            statuses.capacity(),
            job.params.len(),
            job.entropy_capacity,
        )?;
        let status_buffer = new_shared_buffer_with_slice(&runtime.device, &statuses)?;
        // Metal copied the initialization bytes; do not retain a duplicate
        // caller-length status vector through command submission and readback.
        drop(statuses);
        let params_buffer = new_shared_buffer_with_slice(&runtime.device, &job.params)?;
        let tile_count = u32::try_from(job.params.len()).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal batch tile count exceeds u32".to_string(),
        })?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.jpeg_baseline_encode_batch_pipeline);
        encoder.set_buffer(0, Some(job.input), 0);
        encoder.set_buffer(1, Some(&entropy_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_buffer(3, Some(&params_buffer), 0);
        encoder.set_bytes(
            4,
            size_of_val(&job.q_luma) as u64,
            job.q_luma.as_ptr().cast(),
        );
        encoder.set_bytes(
            5,
            size_of_val(&job.q_chroma) as u64,
            job.q_chroma.as_ptr().cast(),
        );
        encoder.set_bytes(
            6,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_luma).cast(),
        );
        encoder.set_bytes(
            7,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_luma).cast(),
        );
        encoder.set_bytes(
            8,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_dc_chroma).cast(),
        );
        encoder.set_bytes(
            9,
            size_of::<JpegBaselineEncodeHuffmanTable>() as u64,
            (&raw const job.huff_ac_chroma).cast(),
        );
        encoder.set_bytes(10, size_of::<u32>() as u64, (&raw const tile_count).cast());
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(tile_count),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_jpeg(&command_buffer)?;

        let status_slice = checked_buffer_slice::<JpegBaselineEncodeStatus>(
            &status_buffer,
            job.params.len(),
            "baseline batch encode statuses",
        )?;
        let mut out = encode_allocation::try_vec_with_capacity(job.params.len())?;
        encode_allocation::checked_batch_runtime_bytes::<
            JpegBaselineEncodeParams,
            JpegBaselineEncodeStatus,
        >(
            job.params.capacity(),
            status_slice.capacity(),
            out.capacity(),
            job.entropy_capacity,
        )?;
        let mut output_payload_capacity = 0usize;
        for (status, params) in status_slice.iter().copied().zip(job.params.iter()) {
            if status.code != JPEG_BASELINE_ENCODE_STATUS_OK {
                return Err(jpeg_baseline_encode_status_error(status));
            }
            let entropy_len =
                usize::try_from(status.entropy_len).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal encode entropy length exceeds usize".to_string(),
                })?;
            let offset =
                usize::try_from(params.entropy_offset_bytes).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy offset exceeds usize".to_string(),
                })?;
            let capacity =
                usize::try_from(params.entropy_capacity).map_err(|_| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy capacity exceeds usize".to_string(),
                })?;
            if entropy_len > capacity {
                return Err(Error::MetalKernel {
                    message:
                        "JPEG Baseline Metal encode reported length exceeds tile output capacity"
                            .to_string(),
                });
            }
            let end = offset
                .checked_add(entropy_len)
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy range overflow".to_string(),
                })?;
            if end > job.entropy_capacity {
                return Err(Error::MetalKernel {
                    message: "JPEG Baseline Metal batch entropy range exceeds buffer".to_string(),
                });
            }
            // Copy the validated tile range directly. A whole-buffer host
            // readback here would overlap every returned chunk and double the
            // entropy portion of the host peak.
            let chunk = checked_buffer_slice_at::<u8>(
                &entropy_buffer,
                offset,
                entropy_len,
                "baseline batch encode entropy chunk",
            )?;
            output_payload_capacity = output_payload_capacity.saturating_add(chunk.capacity());
            out.push(chunk);
            encode_allocation::checked_batch_runtime_bytes::<
                JpegBaselineEncodeParams,
                JpegBaselineEncodeStatus,
            >(
                job.params.capacity(),
                status_slice.capacity(),
                out.capacity(),
                output_payload_capacity,
            )?;
        }
        Ok(out)
    })
}
