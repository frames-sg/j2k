// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    copied_slice_buffer, dispatch_ht_cleanup_batched_in_command_buffer, ht_batch_output_word_count,
    new_shared_buffer, Buffer, CommandBufferRef, DirectStatusCheck, Error, J2kHtCleanupBatchJob,
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
        let empty = new_shared_buffer(&runtime.device, 1)?;
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
        let empty = new_shared_buffer(&runtime.device, 1)?;
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
struct DistinctHtMetadata {
    coded_data: Vec<u8>,
    jobs: Vec<J2kHtCleanupBatchJob>,
}

#[cfg(target_os = "macos")]
fn allocate_distinct_ht_metadata(
    coded_len: usize,
    job_count: usize,
    mut budget: crate::batch_allocation::BatchMetadataBudget,
) -> Result<DistinctHtMetadata, Error> {
    let requests = [
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(coded_len),
        crate::batch_allocation::BatchMetadataRequest::of::<J2kHtCleanupBatchJob>(job_count),
    ];
    budget.preflight(&requests)?;
    Ok(DistinctHtMetadata {
        coded_data: budget.try_vec(
            coded_len,
            "HTJ2K MetalDirect distinct grayscale coded payload",
        )?,
        jobs: budget.try_vec(job_count, "HTJ2K MetalDirect distinct grayscale jobs")?,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_ht_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl Iterator<Item = DistinctHtBatch<'a>> + Clone,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let coded_len = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.coded_data.len()),
        "HTJ2K MetalDirect distinct grayscale coded payload",
    )?;
    let job_count = crate::batch_allocation::checked_count_sum(
        batches.clone().map(|batch| batch.jobs.len()),
        "HTJ2K MetalDirect distinct grayscale jobs",
    )?;
    let DistinctHtMetadata {
        mut coded_data,
        mut jobs,
    } = allocate_distinct_ht_metadata(
        coded_len,
        job_count,
        crate::batch_allocation::BatchMetadataBudget::new(
            "HTJ2K MetalDirect distinct grayscale submission",
        ),
    )?;

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
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = copied_slice_buffer(&runtime.device, &coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, &jobs)?;
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use core::mem::size_of;

    use j2k_core::BatchInfrastructureError;

    use super::{allocate_distinct_ht_metadata, Error, J2kHtCleanupBatchJob};
    use crate::batch_allocation::BatchMetadataBudget;

    #[test]
    fn distinct_ht_metadata_honors_exact_cap_and_one_byte_over() {
        let coded_len = 7;
        let job_count = 3;
        let exact_cap = coded_len + job_count * size_of::<J2kHtCleanupBatchJob>();
        let owners = allocate_distinct_ht_metadata(
            coded_len,
            job_count,
            BatchMetadataBudget::with_cap(
                "HTJ2K MetalDirect distinct grayscale submission",
                exact_cap,
            ),
        )
        .expect("exact distinct HT metadata cap");
        assert_eq!(owners.coded_data.capacity(), coded_len);
        assert_eq!(owners.jobs.capacity(), job_count);

        assert!(matches!(
            allocate_distinct_ht_metadata(
                coded_len,
                job_count,
                BatchMetadataBudget::with_cap(
                    "HTJ2K MetalDirect distinct grayscale submission",
                    exact_cap - 1,
                ),
            ),
            Err(Error::BatchInfrastructure(
                BatchInfrastructureError::AllocationTooLarge {
                    requested,
                    cap,
                    ..
                }
            )) if requested == exact_cap && cap == exact_cap - 1
        ));
    }
}
