// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    default_metal_ht_chunk_limits, encode_metal_ht_batches_in_encoder, new_compute_command_encoder,
    new_shared_buffer, Buffer, CommandBufferRef, DirectStatusCheck, Error, HtBatchInput,
    J2kHtCleanupBatchJob, MetalRuntime, PreparedHtSubBand, PreparedHtSubBandGroup,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedHtSubBand],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result =
        encode_distinct_ht_sub_bands_to_buffer_in_encoder(runtime, &encoder, sub_bands, output);
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_sub_bands_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &metal::ComputeCommandEncoderRef,
    sub_bands: &[&PreparedHtSubBand],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
                source_indices: None,
                recyclable_status: None,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    let output_word_count = per_instance_len
        .checked_mul(sub_bands.len())
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct sub-band output length overflow".to_string(),
        })?;
    encode_distinct_ht_batches_to_buffer_in_encoder(
        runtime,
        encoder,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctHtBatch {
                payload: sub_band.payload_source.as_ht_payload_source(),
                jobs: &sub_band.jobs,
                output_base: index * per_instance_len,
                execution_owner: &sub_band.execution_owner,
            }),
        output,
        output_word_count,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedHtSubBandGroup],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let encoder = new_compute_command_encoder(command_buffer)?;
    let result =
        encode_distinct_ht_sub_band_groups_to_buffer_in_encoder(runtime, &encoder, groups, output);
    encoder.end_encoding();
    result
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_sub_band_groups_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &metal::ComputeCommandEncoderRef,
    groups: &[&PreparedHtSubBandGroup],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
                source_indices: None,
                recyclable_status: None,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    let output_word_count =
        per_instance_len
            .checked_mul(groups.len())
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect distinct group output length overflow".to_string(),
            })?;
    encode_distinct_ht_batches_to_buffer_in_encoder(
        runtime,
        encoder,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctHtBatch {
                payload: group.payload_source.as_ht_payload_source(),
                jobs: &group.jobs,
                output_base: index * per_instance_len,
                execution_owner: &group.execution_owner,
            }),
        output,
        output_word_count,
    )
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct DistinctHtBatch<'a> {
    pub(in crate::compute) payload: super::HtPayloadSource<'a>,
    pub(in crate::compute) jobs: &'a [J2kHtCleanupBatchJob],
    pub(in crate::compute) output_base: usize,
    pub(in crate::compute) execution_owner:
        &'a std::sync::Arc<crate::compute::PreparedHtExecutionOwner>,
}

#[cfg(target_os = "macos")]
fn encode_distinct_ht_batches_to_buffer_in_encoder<'a>(
    runtime: &MetalRuntime,
    encoder: &metal::ComputeCommandEncoderRef,
    batches: impl Iterator<Item = DistinctHtBatch<'a>> + Clone,
    output: &Buffer,
    output_word_count: usize,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let batch_count = batches.clone().count();
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
        "HTJ2K MetalDirect distinct chunk submission",
    );
    let mut inputs = budget.try_vec(
        batch_count,
        "HTJ2K MetalDirect distinct chunk input descriptors",
    )?;
    inputs.extend(
        batches
            .enumerate()
            .map(|(source_index, batch)| HtBatchInput {
                source_index,
                payload: batch.payload,
                jobs: batch.jobs,
                output_base: batch.output_base,
                execution_owner: batch.execution_owner,
            }),
    );
    for input in &inputs {
        for job in input.jobs {
            let output_offset = (job.output_offset as usize)
                .checked_add(input.output_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "HTJ2K MetalDirect distinct output span overflow".to_string(),
                })?;
            let output_offset = u32::try_from(output_offset).map_err(|_| Error::MetalKernel {
                message: "HTJ2K MetalDirect distinct output span exceeds u32".to_string(),
            })?;
            let job_output_end = super::ht_output_word_count(
                output_offset,
                job.output_stride,
                job.width,
                job.height,
            )?;
            if job_output_end > output_word_count {
                return Err(Error::MetalStateInvariant {
                    state: "HTJ2K MetalDirect distinct output",
                    reason: "code-block output span exceeds the logical coefficient arena",
                });
            }
        }
    }

    encode_metal_ht_batches_in_encoder(
        runtime,
        encoder,
        &inputs,
        output,
        output_word_count,
        default_metal_ht_chunk_limits(),
    )
}
