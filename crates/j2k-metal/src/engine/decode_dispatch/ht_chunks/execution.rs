// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use j2k_core::{HtGpuJobChunkLimits, HtGpuJobPassBucket};

use crate::engine::resident_codestream::{
    dispatch_ht_cleanup_batched_in_encoder_with_status_offset,
    dispatch_ht_cleanup_repeated_batched_in_encoder_with_status_offset, HtCleanupBatchDispatch,
    HtCleanupRepeatedBatchDispatch,
};

use super::super::{
    dispatch_zero_u32_buffer_in_encoder, new_compute_command_encoder, new_shared_buffer,
    size_of as abi_size_of, Buffer, CommandBufferRef, ComputeCommandEncoderRef, DirectStatusCheck,
    Error, J2kHtRepeatedBatchParams, MetalRuntime,
};
#[cfg(test)]
use super::PackedMetalHtChunk;
use super::{
    metal_ht_pipeline_kind_for_bucket, prepared::PreparedMetalHtExecution,
    prepared_metal_ht_execution, HtBatchInput, HtZeroLayout,
};

pub(in crate::engine) fn encode_metal_ht_batches_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    batches: &[HtBatchInput<'_>],
    decoded: &Buffer,
    output_word_count: usize,
    limits: HtGpuJobChunkLimits,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    encode_metal_ht_batches_with_zero_layouts_in_encoder(
        runtime,
        encoder,
        batches,
        None,
        decoded,
        output_word_count,
        limits,
    )
}

/// [`encode_metal_ht_batches_in_encoder`], zeroing only what the jobs leave
/// unwritten when `zero_layouts` gives each batch input's arena.
#[expect(
    clippy::too_many_lines,
    reason = "the command-encoder boundary keeps Metal buffer lifetimes and status offsets atomic"
)]
pub(in crate::engine) fn encode_metal_ht_batches_with_zero_layouts_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    batches: &[HtBatchInput<'_>],
    zero_layouts: Option<&[HtZeroLayout<'_>]>,
    decoded: &Buffer,
    output_word_count: usize,
    limits: HtGpuJobChunkLimits,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let output_bytes = output_word_count
        .checked_mul(abi_size_of::<u32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output arena byte length overflow".to_string(),
        })?;
    if output_bytes > decoded.length() {
        return Err(Error::MetalStateInvariant {
            state: "HTJ2K Metal output arena",
            reason: "logical coefficient arena exceeds the destination buffer",
        });
    }
    let prepared = prepared_metal_ht_execution(runtime, batches, limits)?;
    let zero_rects_buffer = encode_ht_output_zero(
        runtime,
        encoder,
        batches,
        zero_layouts,
        decoded,
        output_word_count,
    )?;
    encoder.memory_barrier_with_resources(&[decoded]);
    if prepared.job_count() == 0 {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        let mut retained = crate::batch_allocation::try_vec_from_array(
            [empty.clone()],
            "J2K Metal empty HT retained buffer",
        )?;
        retained.extend(zero_rects_buffer);
        return Ok((
            retained,
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
                dispatches: 0,
                refinement_dispatches: 0,
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
        prepared.chunks().len().saturating_mul(2).saturating_add(1),
        "HTJ2K Metal chunk retained buffers",
    )?;
    retained_buffers.extend(zero_rects_buffer);
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
            runtime.decode()?,
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
        )?;
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
            dispatches: prepared.chunks().len(),
            refinement_dispatches: prepared
                .chunks()
                .iter()
                .filter(|chunk| chunk.bucket != HtGpuJobPassBucket::CleanupOnly)
                .count(),
            source_indices: Some(source_indices),
            recyclable_status: Some(status_owner),
        },
    ))
}

pub(in crate::engine) fn encode_repeated_metal_ht_batch_in_command_buffer(
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
        encoder.endEncoding();
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            crate::batch_allocation::try_vec_from_array(
                [empty.clone()],
                "J2K Metal empty HT retained buffer",
            )?,
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
                dispatches: 0,
                refinement_dispatches: 0,
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
    encoder.endEncoding();

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
            dispatches: prepared.chunks().len(),
            refinement_dispatches: prepared
                .chunks()
                .iter()
                .filter(|chunk| chunk.bucket != HtGpuJobPassBucket::CleanupOnly)
                .count(),
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
                self.runtime.decode()?,
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
            )?;
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

#[cfg(test)]
pub(super) fn validate_pass_homogeneous_chunk(chunk: &PackedMetalHtChunk) -> Result<(), Error> {
    validate_pass_homogeneous_jobs(chunk.bucket, &chunk.jobs)
}

pub(super) fn validate_pass_homogeneous_jobs(
    bucket: HtGpuJobPassBucket,
    jobs: &[super::J2kHtCleanupBatchJob],
) -> Result<(), Error> {
    if jobs.iter().any(|job| job.number_of_coding_passes > 3) {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::resource_limit(
                "HTJ2K Metal decoding supports at most three coding passes per code block",
            ),
        ));
    }
    let matches_bucket = jobs.iter().all(|job| match bucket {
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

/// Zeroes what the jobs of `batches` leave unwritten in the first
/// `output_word_count` words of `decoded`: each input's uncovered rectangles
/// and the words outside every input's arena, or the whole range when an
/// input has no layout or the arenas overlap. Returns the rectangle buffer the
/// submission must retain.
fn encode_ht_output_zero(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    batches: &[HtBatchInput<'_>],
    zero_layouts: Option<&[HtZeroLayout<'_>]>,
    decoded: &Buffer,
    output_word_count: usize,
) -> Result<Option<Buffer>, Error> {
    #[cfg(test)]
    let repeats = crate::engine::test_counters::single_step_repeats_for_test(
        crate::engine::test_counters::SingleStepKind::HtZero,
    );
    #[cfg(not(test))]
    let repeats = 1;
    let rects = match zero_layouts {
        Some(layouts) if layouts.len() == batches.len() => {
            ht_zero_rects(batches, layouts, output_word_count)?
        }
        _ => None,
    };
    let Some(rects) = rects else {
        for _ in 0..repeats {
            dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, output_word_count)?;
        }
        return Ok(None);
    };
    if rects.is_empty() {
        return Ok(None);
    }
    let rects_buffer =
        crate::engine::HtInputAllocator::new(&runtime.device, &runtime.ht_input_heap)
            .copied_slice(&rects)?;
    let pipeline = &runtime.buffers()?.zero_u32_rects;
    encoder.setComputePipelineState(pipeline);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_buffer(1, Some(&rects_buffer), 0);
    for _ in 0..repeats {
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(rects.len() as u64, 1, 1),
            j2k_metal_support::mtl_size(32, 8, 1),
        );
    }
    Ok(Some(rects_buffer))
}

/// The rectangles to zero for `batches` laid out by `layouts`, or `None` when
/// the arenas overlap or extend past `output_word_count`.
fn ht_zero_rects(
    batches: &[HtBatchInput<'_>],
    layouts: &[HtZeroLayout<'_>],
    output_word_count: usize,
) -> Result<Option<Vec<crate::engine::abi::J2kZeroRect>>, Error> {
    let mut spans = Vec::new();
    spans
        .try_reserve_exact(batches.len())
        .map_err(|source| Error::MetalKernel {
            message: format!("HTJ2K Metal zero spans allocation failed: {source}"),
        })?;
    for (batch, layout) in batches.iter().zip(layouts) {
        let Some(end) = batch.output_base.checked_add(layout.extent) else {
            return Ok(None);
        };
        spans.push((batch.output_base, end, layout.uncovered));
    }
    spans.sort_unstable_by_key(|span| span.0);
    let mut rects = Vec::new();
    let mut cursor = 0usize;
    for (start, end, uncovered) in spans {
        if start < cursor || end > output_word_count {
            return Ok(None);
        }
        crate::engine::append_span_rects(cursor, start - cursor, &mut rects)?;
        for rect in uncovered {
            let offset = start
                .checked_add(rect.offset as usize)
                .and_then(|offset| u32::try_from(offset).ok())
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K Metal zero rectangle offset exceeds u32".to_string(),
                })?;
            rects.push(crate::engine::abi::J2kZeroRect { offset, ..*rect });
        }
        cursor = end;
    }
    crate::engine::append_span_rects(cursor, output_word_count - cursor, &mut rects)?;
    Ok(Some(rects))
}
