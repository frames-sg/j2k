// SPDX-License-Identifier: MIT OR Apache-2.0

//! Metal baseline JPEG entropy submission and bounded host readback.

use crate::metal_types::prelude::*;
use crate::metal_types::{BufferRef, CommandBufferRef};

use super::{
    commit_and_wait_jpeg, dispatch_1d_pipeline, new_command_buffer, new_compute_command_encoder,
    runtime::MetalRuntime, with_runtime_for_session,
};
use crate::abi::{
    JpegBaselineEncodeHuffmanTable, JpegBaselineEncodeParams, JpegBaselineEncodeStatus,
    JpegBaselineEntropyEncodeBatchJob, JpegBaselineEntropyEncodeJob,
    JPEG_BASELINE_ENCODE_STATUS_OK,
};
use crate::buffers::{checked_buffer_read, checked_buffer_slice, checked_buffer_slice_at};
use crate::compute::status::jpeg_baseline_encode_status_error;
use crate::{encode::allocation as encode_allocation, Error};

// One baseline MCU has at most six 8x8 blocks. A block needs at most 250 raw
// entropy bytes: one 31-bit DC code/value, up to 63 31-bit AC code/values, and
// a 16-bit EOB. Leave the remainder for whole-byte addressing in the shader.
const JPEG_BASELINE_ENCODE_SEGMENT_BYTES: u32 = 1536;

fn staged_coefficient_plan(params: &[JpegBaselineEncodeParams]) -> Result<(usize, u32), Error> {
    let mut coefficient_count = 0usize;
    let mut mcu_count = 0usize;
    for params in params {
        let components = usize::try_from(params.components).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal component count exceeds usize".to_string(),
        })?;
        if !(1..=3).contains(&components) {
            return Err(Error::MetalKernel {
                message: "JPEG Baseline Metal staged encode requires one to three components"
                    .to_string(),
            });
        }
        let component_blocks = [
            params.h0.checked_mul(params.v0),
            params.h1.checked_mul(params.v1),
            params.h2.checked_mul(params.v2),
        ];
        let mut blocks_per_mcu = 0usize;
        for blocks in component_blocks.into_iter().take(components) {
            blocks_per_mcu = blocks_per_mcu
                .checked_add(blocks.ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Baseline Metal block geometry overflowed".to_string(),
                })? as usize)
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Baseline Metal blocks-per-MCU overflowed".to_string(),
                })?;
        }
        let tile_mcus = (params.mcus_per_row as usize)
            .checked_mul(params.mcu_rows as usize)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal MCU count overflowed".to_string(),
            })?;
        mcu_count = mcu_count
            .checked_add(tile_mcus)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal batch MCU count overflowed".to_string(),
            })?;
        coefficient_count = coefficient_count
            .checked_add(
                tile_mcus
                    .checked_mul(blocks_per_mcu)
                    .and_then(|blocks| blocks.checked_mul(64))
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Baseline Metal coefficient count overflowed".to_string(),
                    })?,
            )
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal batch coefficient count overflowed".to_string(),
            })?;
    }
    let mcu_count = u32::try_from(mcu_count).map_err(|_| Error::MetalKernel {
        message: "JPEG Baseline Metal batch MCU count exceeds u32".to_string(),
    })?;
    u32::try_from(coefficient_count).map_err(|_| Error::MetalKernel {
        message: "JPEG Baseline Metal coefficient indexing exceeds u32".to_string(),
    })?;
    Ok((coefficient_count, mcu_count))
}

fn record_baseline_entropy_passes(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    buffers: [&BufferRef; 6],
    huffman_tables: [&JpegBaselineEncodeHuffmanTable; 4],
    tile_count: u32,
    mcu_count: u32,
) -> Result<(), Error> {
    let [coefficient_buffer, segment_buffer, segment_status_buffer, entropy_buffer, status_buffer, params_buffer] =
        buffers;
    let [dc_luma, ac_luma, dc_chroma, ac_chroma] = huffman_tables;

    let pipeline = &runtime
        .pipelines
        .jpeg_baseline_encode_entropy_segments_batch;
    let segments = new_compute_command_encoder(command_buffer)?;
    segments.setComputePipelineState(pipeline);
    segments.bind_buffer(0, Some(coefficient_buffer), 0);
    segments.bind_buffer(1, Some(segment_buffer), 0);
    segments.bind_buffer(2, Some(segment_status_buffer), 0);
    segments.bind_buffer(3, Some(params_buffer), 0);
    segments.bind_bytes::<JpegBaselineEncodeHuffmanTable>(4, dc_luma);
    segments.bind_bytes::<JpegBaselineEncodeHuffmanTable>(5, ac_luma);
    segments.bind_bytes::<JpegBaselineEncodeHuffmanTable>(6, dc_chroma);
    segments.bind_bytes::<JpegBaselineEncodeHuffmanTable>(7, ac_chroma);
    segments.bind_bytes::<u32>(8, &tile_count);
    dispatch_1d_pipeline(&segments, pipeline, mcu_count);
    segments.endEncoding();

    let pipeline = &runtime
        .pipelines
        .jpeg_baseline_encode_prepare_segments_batch;
    let prepare = new_compute_command_encoder(command_buffer)?;
    prepare.setComputePipelineState(pipeline);
    prepare.bind_buffer(0, Some(segment_buffer), 0);
    prepare.bind_buffer(1, Some(segment_status_buffer), 0);
    prepare.bind_buffer(2, Some(entropy_buffer), 0);
    prepare.bind_buffer(3, Some(status_buffer), 0);
    prepare.bind_buffer(4, Some(params_buffer), 0);
    prepare.bind_bytes::<u32>(5, &tile_count);
    dispatch_1d_pipeline(&prepare, pipeline, tile_count);
    prepare.endEncoding();

    let pipeline = &runtime
        .pipelines
        .jpeg_baseline_encode_count_stuffed_segments_batch;
    let count_stuffed = new_compute_command_encoder(command_buffer)?;
    count_stuffed.setComputePipelineState(pipeline);
    count_stuffed.bind_buffer(0, Some(segment_buffer), 0);
    count_stuffed.bind_buffer(1, Some(segment_status_buffer), 0);
    count_stuffed.bind_buffer(2, Some(status_buffer), 0);
    count_stuffed.bind_buffer(3, Some(params_buffer), 0);
    count_stuffed.bind_bytes::<u32>(4, &tile_count);
    dispatch_1d_pipeline(&count_stuffed, pipeline, mcu_count);
    count_stuffed.endEncoding();

    let pipeline = &runtime
        .pipelines
        .jpeg_baseline_encode_offset_stuffed_segments_batch;
    let offset_stuffed = new_compute_command_encoder(command_buffer)?;
    offset_stuffed.setComputePipelineState(pipeline);
    offset_stuffed.bind_buffer(0, Some(segment_status_buffer), 0);
    offset_stuffed.bind_buffer(1, Some(status_buffer), 0);
    offset_stuffed.bind_buffer(2, Some(params_buffer), 0);
    offset_stuffed.bind_bytes::<u32>(3, &tile_count);
    dispatch_1d_pipeline(&offset_stuffed, pipeline, tile_count);
    offset_stuffed.endEncoding();

    let pipeline = &runtime
        .pipelines
        .jpeg_baseline_encode_scatter_stuffed_segments_batch;
    let scatter_stuffed = new_compute_command_encoder(command_buffer)?;
    scatter_stuffed.setComputePipelineState(pipeline);
    scatter_stuffed.bind_buffer(0, Some(segment_buffer), 0);
    scatter_stuffed.bind_buffer(1, Some(segment_status_buffer), 0);
    scatter_stuffed.bind_buffer(2, Some(entropy_buffer), 0);
    scatter_stuffed.bind_buffer(3, Some(status_buffer), 0);
    scatter_stuffed.bind_buffer(4, Some(params_buffer), 0);
    scatter_stuffed.bind_bytes::<u32>(5, &tile_count);
    dispatch_1d_pipeline(&scatter_stuffed, pipeline, mcu_count);
    scatter_stuffed.endEncoding();
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "the entropy path keeps pooled buffers and its ordered Metal passes in one lifetime scope"
)]
pub(crate) fn encode_jpeg_baseline_entropy_with_session(
    session: &crate::MetalBackendSession,
    job: &JpegBaselineEntropyEncodeJob<'_>,
) -> Result<Vec<u8>, Error> {
    encode_allocation::checked_single_output_bytes(job.entropy_capacity)?;
    with_runtime_for_session(session, |runtime| {
        let mut scratch = runtime.batch_scratch()?;
        let entropy_buffer = scratch.shared_buffer(
            &runtime.device,
            "JPEG baseline encode entropy",
            job.entropy_capacity,
        )?;
        let status = JpegBaselineEncodeStatus::default();
        let status_buffer = scratch.shared_buffer_with_slice(
            &runtime.device,
            "JPEG baseline encode status",
            std::slice::from_ref(&status),
        )?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let mut params = job.params;
        params.input_offset_bytes = 0;
        params.entropy_offset_bytes = 0;
        let params_buffer = scratch.shared_buffer_with_slice(
            &runtime.device,
            "JPEG baseline encode params",
            std::slice::from_ref(&params),
        )?;
        let (coefficient_count, mcu_count) =
            staged_coefficient_plan(std::slice::from_ref(&params))?;
        let coefficient_bytes = coefficient_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal coefficient byte count overflowed".to_string(),
            })?;
        let coefficient_buffer = scratch.private_buffer(
            &runtime.device,
            "JPEG baseline encode coefficients",
            coefficient_bytes,
        )?;
        let mcu_count_usize = usize::try_from(mcu_count).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal MCU count exceeds usize".to_string(),
        })?;
        let segment_bytes = mcu_count
            .checked_mul(JPEG_BASELINE_ENCODE_SEGMENT_BYTES)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal entropy segment indexing exceeds u32".to_string(),
            })?;
        let segment_bytes = usize::try_from(segment_bytes).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal entropy segment bytes exceed usize".to_string(),
        })?;
        let segment_status_bytes = mcu_count_usize
            .checked_mul(std::mem::size_of::<JpegBaselineEncodeStatus>())
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal entropy segment status overflowed".to_string(),
            })?;
        let segment_buffer = scratch.private_buffer(
            &runtime.device,
            "JPEG baseline encode segments",
            segment_bytes,
        )?;
        let segment_status_buffer = scratch.private_buffer(
            &runtime.device,
            "JPEG baseline encode segment statuses",
            segment_status_bytes,
        )?;
        let tile_count = 1u32;

        let precompute = new_compute_command_encoder(&command_buffer)?;
        precompute
            .setComputePipelineState(&runtime.pipelines.jpeg_baseline_encode_precompute_batch);
        precompute.bind_buffer(0, Some(job.input), job.input_offset as u64);
        precompute.bind_buffer(1, Some(&coefficient_buffer), 0);
        precompute.bind_buffer(2, Some(&params_buffer), 0);
        precompute.bind_bytes::<[u8; 64]>(3, &job.q_luma);
        precompute.bind_bytes::<[u8; 64]>(4, &job.q_chroma);
        precompute.bind_bytes::<u32>(5, &tile_count);
        dispatch_1d_pipeline(
            &precompute,
            &runtime.pipelines.jpeg_baseline_encode_precompute_batch,
            mcu_count,
        );
        precompute.endEncoding();

        record_baseline_entropy_passes(
            runtime,
            &command_buffer,
            [
                &coefficient_buffer,
                &segment_buffer,
                &segment_status_buffer,
                &entropy_buffer,
                &status_buffer,
                &params_buffer,
            ],
            [
                &job.huff_dc_luma,
                &job.huff_ac_luma,
                &job.huff_dc_chroma,
                &job.huff_ac_chroma,
            ],
            tile_count,
            mcu_count,
        )?;
        commit_and_wait_jpeg(&command_buffer)?;
        drop(coefficient_buffer);
        drop(segment_buffer);
        drop(segment_status_buffer);

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
        let mut scratch = runtime.batch_scratch()?;
        let entropy_buffer = scratch.shared_buffer(
            &runtime.device,
            "JPEG baseline encode entropy",
            job.entropy_capacity,
        )?;
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
        let status_buffer = scratch.shared_buffer_with_slice(
            &runtime.device,
            "JPEG baseline encode status",
            &statuses,
        )?;
        // Metal copied the initialization bytes; do not retain a duplicate
        // caller-length status vector through command submission and readback.
        drop(statuses);
        let params_buffer = scratch.shared_buffer_with_slice(
            &runtime.device,
            "JPEG baseline encode params",
            &job.params,
        )?;
        let tile_count = u32::try_from(job.params.len()).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal batch tile count exceeds u32".to_string(),
        })?;

        let command_buffer = new_command_buffer(&runtime.queue)?;
        let (coefficient_count, mcu_count) = staged_coefficient_plan(&job.params)?;
        let coefficient_bytes = coefficient_count
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal coefficient byte count overflowed".to_string(),
            })?;
        let coefficient_buffer = scratch.private_buffer(
            &runtime.device,
            "JPEG baseline encode coefficients",
            coefficient_bytes,
        )?;
        let mcu_count_usize = usize::try_from(mcu_count).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal batch MCU count exceeds usize".to_string(),
        })?;
        let segment_bytes = mcu_count
            .checked_mul(JPEG_BASELINE_ENCODE_SEGMENT_BYTES)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal batch entropy segment indexing exceeds u32"
                    .to_string(),
            })?;
        let segment_bytes = usize::try_from(segment_bytes).map_err(|_| Error::MetalKernel {
            message: "JPEG Baseline Metal batch entropy segment bytes exceed usize".to_string(),
        })?;
        let segment_status_bytes = mcu_count_usize
            .checked_mul(std::mem::size_of::<JpegBaselineEncodeStatus>())
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Baseline Metal batch entropy segment status overflowed".to_string(),
            })?;
        let segment_buffer = scratch.private_buffer(
            &runtime.device,
            "JPEG baseline encode segments",
            segment_bytes,
        )?;
        let segment_status_buffer = scratch.private_buffer(
            &runtime.device,
            "JPEG baseline encode segment statuses",
            segment_status_bytes,
        )?;

        let precompute = new_compute_command_encoder(&command_buffer)?;
        precompute
            .setComputePipelineState(&runtime.pipelines.jpeg_baseline_encode_precompute_batch);
        precompute.bind_buffer(0, Some(job.input), 0);
        precompute.bind_buffer(1, Some(&coefficient_buffer), 0);
        precompute.bind_buffer(2, Some(&params_buffer), 0);
        precompute.bind_bytes::<[u8; 64]>(3, &job.q_luma);
        precompute.bind_bytes::<[u8; 64]>(4, &job.q_chroma);
        precompute.bind_bytes::<u32>(5, &tile_count);
        dispatch_1d_pipeline(
            &precompute,
            &runtime.pipelines.jpeg_baseline_encode_precompute_batch,
            mcu_count,
        );
        precompute.endEncoding();

        record_baseline_entropy_passes(
            runtime,
            &command_buffer,
            [
                &coefficient_buffer,
                &segment_buffer,
                &segment_status_buffer,
                &entropy_buffer,
                &status_buffer,
                &params_buffer,
            ],
            [
                &job.huff_dc_luma,
                &job.huff_ac_luma,
                &job.huff_dc_chroma,
                &job.huff_ac_chroma,
            ],
            tile_count,
            mcu_count,
        )?;
        commit_and_wait_jpeg(&command_buffer)?;
        drop(coefficient_buffer);
        drop(segment_buffer);
        drop(segment_status_buffer);

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
