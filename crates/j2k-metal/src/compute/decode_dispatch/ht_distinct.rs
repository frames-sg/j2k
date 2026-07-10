// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dispatch_ht_cleanup_batched_in_command_buffer, ht_batch_output_word_count, owned_slice_buffer,
    Buffer, CommandBufferRef, DirectStatusCheck, Error, J2kHtCleanupBatchJob, MTLResourceOptions,
    MetalRuntime, PreparedHtSubBand, PreparedHtSubBandGroup,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedHtSubBand],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctHtBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedHtSubBandGroup],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctHtBatch {
                coded_data: &group.coded_arena.data,
                jobs: &group.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct DistinctHtBatch<'a> {
    pub(in crate::compute) coded_data: &'a [u8],
    pub(in crate::compute) jobs: &'a [J2kHtCleanupBatchJob],
    pub(in crate::compute) output_base: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctHtBatch<'a>>,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let status_check = dispatch_ht_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        output,
        ht_batch_output_word_count(&jobs)?,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}
