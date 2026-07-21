// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::ht_subband::dispatch_zero_u32_buffer_in_encoder;
use super::{
    classic_batch_uses_plain_fast_path, dispatch_classic_cleanup_batched_in_encoder,
    distinct_allocation::{allocate_distinct_classic_metadata, DistinctClassicMetadata},
    ClassicCleanupBatchDispatch,
};
use crate::compute::abi::{J2kClassicCleanupBatchJob, J2kClassicSegment};
use crate::compute::{
    copied_slice_buffer, new_compute_command_encoder, new_shared_buffer,
    take_classic_coefficients_scratch_buffer, Buffer, CommandBufferRef, ComputeCommandEncoderRef,
    DirectScratchBuffer, DirectStatusCheck, Error, MetalRuntime, PreparedClassicSubBand,
    PreparedClassicSubBandGroup,
};

pub(in crate::compute) fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result = encode_distinct_classic_sub_bands_to_buffer_in_encoder(
        runtime,
        &encoder,
        sub_bands,
        output,
        scratch_buffers,
    );
    encoder.end_encoding();
    result
}

pub(in crate::compute) fn encode_distinct_classic_sub_bands_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_classic_batches_to_buffer_in_encoder(
        runtime,
        encoder,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctClassicBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                segments: &sub_band.segments,
                output_base: index * per_instance_len,
                output_len: per_instance_len,
                zero_fill: sub_band.zero_fill,
            }),
        output,
        scratch_buffers,
    )
}

pub(in crate::compute) fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result = encode_distinct_classic_sub_band_groups_to_buffer_in_encoder(
        runtime,
        &encoder,
        groups,
        output,
        scratch_buffers,
    );
    encoder.end_encoding();
    result
}

pub(in crate::compute) fn encode_distinct_classic_sub_band_groups_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_classic_batches_to_buffer_in_encoder(
        runtime,
        encoder,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
                output_len: per_instance_len,
                zero_fill: group.zero_fill,
            }),
        output,
        scratch_buffers,
    )
}

#[derive(Clone, Copy)]
struct DistinctClassicBatch<'a> {
    coded_data: &'a [u8],
    jobs: &'a [J2kClassicCleanupBatchJob],
    segments: &'a [J2kClassicSegment],
    output_base: usize,
    output_len: usize,
    zero_fill: bool,
}

fn append_distinct_classic_batch(
    metadata: &mut DistinctClassicMetadata,
    batch: DistinctClassicBatch<'_>,
) -> Result<(), Error> {
    let coded_base = u32::try_from(metadata.coded_data.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect distinct color coded payload exceeds u32".to_string(),
    })?;
    let segment_base = u32::try_from(metadata.segments.len()).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect distinct color segment table exceeds u32".to_string(),
    })?;
    metadata.coded_data.extend_from_slice(batch.coded_data);
    for segment in batch.segments {
        let mut adjusted = *segment;
        adjusted.data_offset = adjusted
            .data_offset
            .checked_add(coded_base)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct color segment offset overflow"
                    .to_string(),
            })?;
        metadata.segments.push(adjusted);
    }
    let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
        message: "classic J2K MetalDirect distinct color output offset exceeds u32".to_string(),
    })?;
    for job in batch.jobs {
        let mut adjusted = *job;
        adjusted.coded_offset = adjusted
            .coded_offset
            .checked_add(coded_base)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct color job coded offset overflow"
                    .to_string(),
            })?;
        adjusted.segment_offset = adjusted
            .segment_offset
            .checked_add(segment_base)
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct color job segment offset overflow"
                    .to_string(),
            })?;
        adjusted.output_offset =
            adjusted
                .output_offset
                .checked_add(output_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect distinct color job output offset overflow"
                        .to_string(),
                })?;
        metadata.jobs.push(adjusted);
    }
    Ok(())
}

fn encode_distinct_classic_batches_to_buffer_in_encoder<'a>(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    batches: impl Iterator<Item = DistinctClassicBatch<'a>> + Clone,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let zero_fill_word_count = batches.clone().try_fold(0usize, |word_count, batch| {
        if !batch.zero_fill && !batch.jobs.is_empty() {
            return Ok(word_count);
        }
        batch
            .output_base
            .checked_add(batch.output_len)
            .map(|end| word_count.max(end))
            .ok_or_else(|| Error::MetalKernel {
                message: "classic J2K MetalDirect distinct output span overflow".to_string(),
            })
    })?;
    let coded_len = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.coded_data.len()),
        "classic J2K MetalDirect distinct color coded payload",
    )?;
    let job_count = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.jobs.len()),
        "classic J2K MetalDirect distinct color jobs",
    )?;
    let segment_count = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.segments.len()),
        "classic J2K MetalDirect distinct color segments",
    )?;
    let mut metadata = allocate_distinct_classic_metadata(
        coded_len,
        job_count,
        segment_count,
        crate::batch_allocation::BatchMetadataBudget::new(
            "classic J2K MetalDirect distinct color submission",
        ),
    )?;

    for batch in batches {
        append_distinct_classic_batch(&mut metadata, batch)?;
    }
    let DistinctClassicMetadata {
        coded_data,
        jobs,
        segments,
    } = metadata;

    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, zero_fill_word_count)?;

    if jobs.is_empty() {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }
    if zero_fill_word_count != 0 {
        encoder.memory_barrier_with_resources(&[output]);
    }

    let coded_buffer = copied_slice_buffer(&runtime.device, &coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, &jobs)?;
    let segments_buffer = copied_slice_buffer(&runtime.device, &segments)?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        encoder,
        ClassicCleanupBatchDispatch {
            runtime,
            coded_data: &coded_buffer,
            jobs: &jobs_buffer,
            job_count: jobs.len(),
            use_plain_fast_path,
            segments: &segments_buffer,
            decoded: output,
            coefficients_scratch: &coefficients_scratch.buffer,
        },
    )?;
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(test)]
#[path = "distinct_metadata_tests.rs"]
mod tests;
