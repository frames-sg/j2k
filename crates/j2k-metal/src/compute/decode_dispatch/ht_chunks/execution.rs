// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{HtGpuJobChunkLimits, HtGpuJobPassBucket};

use crate::compute::resident_codestream::{
    dispatch_ht_cleanup_batched_in_encoder_with_status_offset,
    dispatch_ht_cleanup_repeated_batched_in_encoder_with_status_offset, HtCleanupBatchDispatch,
    HtCleanupRepeatedBatchDispatch,
};

use super::super::{
    dispatch_zero_u32_buffer_in_encoder, new_compute_command_encoder, new_shared_buffer,
    size_of as abi_size_of, Buffer, CommandBufferRef, ComputeCommandEncoderRef, DirectStatusCheck,
    Error, J2kHtRepeatedBatchParams, MetalRuntime,
};
use super::{
    metal_ht_pipeline_kind_for_bucket, prepared::PreparedMetalHtExecution,
    prepared_metal_ht_execution, HtBatchInput, PackedMetalHtChunk,
};

pub(in crate::compute) fn encode_metal_ht_batches_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    batches: &[HtBatchInput<'_>],
    decoded: &Buffer,
    output_word_count: usize,
    limits: HtGpuJobChunkLimits,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let output_bytes = output_word_count
        .checked_mul(abi_size_of::<u32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output arena byte length overflow".to_string(),
        })?;
    if u64::try_from(output_bytes).map_or(true, |bytes| bytes > decoded.length()) {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K Metal output arena",
            reason: "logical coefficient arena exceeds the destination buffer",
        });
    }
    let prepared = prepared_metal_ht_execution(runtime, batches, limits)?;
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, output_word_count)?;
    encoder.memory_barrier_with_resources(&[decoded]);
    if prepared.job_count() == 0 {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
                recyclable_status: None,
            },
        ));
    }

    let status_bytes = prepared
        .job_count()
        .checked_mul(abi_size_of::<super::super::J2kHtStatus>())
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal chunk status arena size overflow".to_string(),
        })?;
    let status_owner = runtime.take_shared_buffer(status_bytes)?;
    let status_buffer = status_owner.buffer().clone();
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("HTJ2K Metal chunk submission owners");
    let mut retained_buffers = budget.try_vec(
        prepared.chunks().len().saturating_mul(2),
        "HTJ2K Metal chunk retained buffers",
    )?;
    let mut source_indices = budget.try_vec(
        prepared.job_count(),
        "HTJ2K Metal chunk ordered source indices",
    )?;
    let mut status_offset_bytes = 0usize;

    for chunk in prepared.chunks() {
        let status_offset = u64::try_from(status_offset_bytes).map_err(|_| Error::MetalKernel {
            message: "HTJ2K Metal chunk status offset exceeds u64".to_string(),
        })?;
        dispatch_ht_cleanup_batched_in_encoder_with_status_offset(
            runtime,
            encoder,
            metal_ht_pipeline_kind_for_bucket(chunk.bucket),
            HtCleanupBatchDispatch {
                coded_data: &chunk.coded_buffer,
                jobs: &chunk.jobs_buffer,
                job_count: chunk.job_count(),
                decoded,
                status_buffer: &status_buffer,
                status_offset_bytes: status_offset,
            },
        );
        let chunk_status_bytes = chunk
            .job_count()
            .checked_mul(abi_size_of::<super::super::J2kHtStatus>())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal chunk status offset overflow".to_string(),
            })?;
        status_offset_bytes = status_offset_bytes
            .checked_add(chunk_status_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal aggregate status offset overflow".to_string(),
            })?;
        retained_buffers.push(chunk.coded_buffer.clone());
        retained_buffers.push(chunk.jobs_buffer.clone());
        source_indices.extend_from_slice(&chunk.source_indices);
    }

    if status_offset_bytes != status_bytes || source_indices.len() != prepared.job_count() {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K Metal chunk submission",
            reason: "dispatched chunk status ownership does not match the shared plan",
        });
    }
    Ok((
        retained_buffers,
        DirectStatusCheck::Ht {
            buffer: status_buffer,
            len: prepared.job_count(),
            source_indices: Some(source_indices),
            recyclable_status: Some(status_owner),
        },
    ))
}

pub(in crate::compute) fn encode_repeated_metal_ht_batch_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batch: HtBatchInput<'_>,
    count: usize,
    output_plane_len: usize,
    decoded: &Buffer,
    limits: HtGpuJobChunkLimits,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let prepared = prepared_metal_ht_execution(runtime, &[batch], limits)?;
    let total_job_count =
        prepared
            .job_count()
            .checked_mul(count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal repeated chunk job count overflow".to_string(),
            })?;
    let decoded_word_count =
        output_plane_len
            .checked_mul(count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal repeated chunk output span overflow".to_string(),
            })?;
    let encoder = new_compute_command_encoder(command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(runtime, &encoder, decoded, decoded_word_count)?;
    encoder.memory_barrier_with_resources(&[decoded]);
    if total_job_count == 0 {
        encoder.end_encoding();
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
                recyclable_status: None,
            },
        ));
    }

    let status_bytes = total_job_count
        .checked_mul(abi_size_of::<super::super::J2kHtStatus>())
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal repeated chunk status arena size overflow".to_string(),
        })?;
    let status_owner = runtime.take_shared_buffer(status_bytes)?;
    let status_buffer = status_owner.buffer().clone();
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K Metal repeated chunk submission owners",
    );
    let mut retained_buffers = budget.try_vec(
        prepared.chunks().len().saturating_mul(2),
        "HTJ2K Metal repeated chunk retained buffers",
    )?;
    let mut source_indices = budget.try_vec(
        total_job_count,
        "HTJ2K Metal repeated chunk ordered source indices",
    )?;
    let output_plane_len = u32::try_from(output_plane_len).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal repeated output plane length exceeds u32".to_string(),
    })?;
    let batch_count = u32::try_from(count).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal repeated batch count exceeds u32".to_string(),
    })?;
    let status_offset_bytes = RepeatedHtChunkEncoder {
        runtime,
        encoder: &encoder,
        prepared: &prepared,
        count,
        output_plane_len,
        batch_count,
        decoded,
        status_buffer: &status_buffer,
    }
    .encode(&mut retained_buffers, &mut source_indices)?;
    encoder.end_encoding();

    if status_offset_bytes != status_bytes || source_indices.len() != total_job_count {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K Metal repeated chunk submission",
            reason: "dispatched chunk status ownership does not match the shared plan",
        });
    }
    Ok((
        retained_buffers,
        DirectStatusCheck::Ht {
            buffer: status_buffer,
            len: total_job_count,
            source_indices: Some(source_indices),
            recyclable_status: Some(status_owner),
        },
    ))
}

struct RepeatedHtChunkEncoder<'a, 'plan> {
    runtime: &'a MetalRuntime,
    encoder: &'a ComputeCommandEncoderRef,
    prepared: &'plan PreparedMetalHtExecution,
    count: usize,
    output_plane_len: u32,
    batch_count: u32,
    decoded: &'a Buffer,
    status_buffer: &'a Buffer,
}

impl RepeatedHtChunkEncoder<'_, '_> {
    fn encode(
        &self,
        retained_buffers: &mut Vec<Buffer>,
        source_indices: &mut Vec<usize>,
    ) -> Result<usize, Error> {
        let mut status_offset_bytes = 0usize;
        for chunk in self.prepared.chunks() {
            let status_offset =
                u64::try_from(status_offset_bytes).map_err(|_| Error::MetalKernel {
                    message: "HTJ2K Metal repeated chunk status offset exceeds u64".to_string(),
                })?;
            let job_count = u32::try_from(chunk.job_count()).map_err(|_| Error::MetalKernel {
                message: "HTJ2K Metal repeated chunk job count exceeds u32".to_string(),
            })?;
            dispatch_ht_cleanup_repeated_batched_in_encoder_with_status_offset(
                self.runtime,
                self.encoder,
                metal_ht_pipeline_kind_for_bucket(chunk.bucket),
                HtCleanupRepeatedBatchDispatch {
                    coded_data: &chunk.coded_buffer,
                    jobs: &chunk.jobs_buffer,
                    base_job_count: chunk.job_count(),
                    repeated: J2kHtRepeatedBatchParams {
                        job_count,
                        output_plane_len: self.output_plane_len,
                        batch_count: self.batch_count,
                    },
                    decoded: self.decoded,
                    status_buffer: self.status_buffer,
                    status_offset_bytes: status_offset,
                },
            );
            let chunk_status_count =
                chunk
                    .job_count()
                    .checked_mul(self.count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K Metal repeated chunk status count overflow".to_string(),
                    })?;
            let chunk_status_bytes = chunk_status_count
                .checked_mul(abi_size_of::<super::super::J2kHtStatus>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal repeated chunk status offset overflow".to_string(),
                })?;
            status_offset_bytes = status_offset_bytes
                .checked_add(chunk_status_bytes)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal repeated aggregate status offset overflow".to_string(),
                })?;
            for source_index in 0..self.count {
                source_indices.extend(core::iter::repeat_n(source_index, chunk.job_count()));
            }
            retained_buffers.push(chunk.coded_buffer.clone());
            retained_buffers.push(chunk.jobs_buffer.clone());
        }
        Ok(status_offset_bytes)
    }
}

pub(super) fn validate_pass_homogeneous_chunk(chunk: &PackedMetalHtChunk) -> Result<(), Error> {
    if chunk.jobs.iter().any(|job| job.number_of_coding_passes > 3) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "HTJ2K Metal decoding supports at most three coding passes per code block",
        });
    }
    let matches_bucket = chunk.jobs.iter().all(|job| match chunk.bucket {
        HtGpuJobPassBucket::CleanupOnly => job.number_of_coding_passes == 1,
        HtGpuJobPassBucket::SigProp => job.number_of_coding_passes == 2,
        HtGpuJobPassBucket::MagRef => job.number_of_coding_passes == 3,
    });
    if !matches_bucket {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K Metal chunk submission",
            reason: "shared planner returned a pass-heterogeneous chunk",
        });
    }
    Ok(())
}
