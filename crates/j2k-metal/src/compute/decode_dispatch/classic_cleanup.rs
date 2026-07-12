// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, decode_classic_status_error,
    j2k_u32_param, new_command_buffer, new_compute_command_encoder, new_shared_buffer, size_of,
    take_classic_coefficients_scratch_buffer, zeroed_shared_buffer, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, DirectScratchBuffer, DirectStatusCheck, Error,
    J2kClassicCleanupBatchJob, J2kClassicRepeatedBatchParams, J2kClassicSegment, J2kClassicStatus,
    MTLSize, MetalRuntime, PreparedClassicSubBand, PreparedClassicSubBandGroup,
    J2K_CLASSIC_MAX_HEIGHT, J2K_CLASSIC_MAX_WIDTH, J2K_CLASSIC_STATUS_OK,
};

#[cfg(target_os = "macos")]
mod distinct_allocation;

#[cfg(target_os = "macos")]
use self::distinct_allocation::{allocate_distinct_classic_metadata, DistinctClassicMetadata};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn classic_batch_uses_plain_fast_path(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        if job.style_flags != 0
            || job.width > J2K_CLASSIC_MAX_WIDTH
            || job.height > J2K_CLASSIC_MAX_HEIGHT
        {
            return false;
        }
        let start = job.segment_offset as usize;
        let Some(end) = start.checked_add(job.segment_count as usize) else {
            return false;
        };
        segments.get(start..end).is_some_and(|job_segments| {
            job_segments
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
        })
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn classic_repeated_uses_plain_fast_path(
    count: usize,
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    let _ = (count, jobs, segments);
    // Batch-16 WSI benches are faster with device-state cleanup plus the separate parallel store.
    false
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn classic_batch_is_plain_arithmetic(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        job.style_flags == 0
            && segments[job.segment_offset as usize
                ..job.segment_offset as usize + job.segment_count as usize]
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = copied_slice_buffer(&runtime.device, coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, jobs)?;
    let segments_buffer = copied_slice_buffer(&runtime.device, segments)?;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(jobs, segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        jobs.len().max(1) * size_of::<J2kClassicStatus>(),
    )?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&segments_buffer), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(&coefficients_scratch.buffer), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(jobs.len() as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();
    commit_and_wait_metal(&command_buffer)?;

    let statuses =
        checked_buffer_slice::<J2kClassicStatus>(&status_buffer, jobs.len(), "classic status")?;
    let status = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_CLASSIC_STATUS_OK);
    runtime.recycle_private_buffer(coefficients_scratch.bytes, coefficients_scratch.buffer)?;
    if let Some(status) = status {
        return Err(decode_classic_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct ClassicCleanupBatchDispatch<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) coded_data: &'a Buffer,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) job_count: usize,
    pub(in crate::compute) use_plain_fast_path: bool,
    pub(in crate::compute) segments: &'a Buffer,
    pub(in crate::compute) decoded: &'a Buffer,
    pub(in crate::compute) coefficients_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct ClassicRepeatedCleanupDispatch<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) coded_data: &'a Buffer,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) job_count: usize,
    pub(in crate::compute) total_job_count: usize,
    pub(in crate::compute) output_plane_len: usize,
    pub(in crate::compute) use_plain_fast_path: bool,
    pub(in crate::compute) segments: &'a Buffer,
    pub(in crate::compute) decoded: &'a Buffer,
    pub(in crate::compute) coefficients_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct ClassicPlainDevRepeatedCleanupDispatch<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) coded_data: &'a Buffer,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) job_count: usize,
    pub(in crate::compute) total_job_count: usize,
    pub(in crate::compute) output_plane_len: usize,
    pub(in crate::compute) segments: &'a Buffer,
    pub(in crate::compute) decoded: &'a Buffer,
    pub(in crate::compute) coefficients_scratch: &'a Buffer,
    pub(in crate::compute) states_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct ClassicRepeatedStoreDispatch<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) job_count: usize,
    pub(in crate::compute) total_job_count: usize,
    pub(in crate::compute) output_plane_len: usize,
    pub(in crate::compute) decoded: &'a Buffer,
    pub(in crate::compute) coefficients_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_cleanup_batched_in_command_buffer(
    command_buffer: &CommandBufferRef,
    dispatch: ClassicCleanupBatchDispatch<'_>,
) -> Result<(DirectStatusCheck, Option<Buffer>), Error> {
    let status_buffer = zeroed_shared_buffer(
        &dispatch.runtime.device,
        dispatch.job_count.max(1) * size_of::<J2kClassicStatus>(),
    )?;

    let encoder = new_compute_command_encoder(command_buffer)?;
    dispatch_classic_cleanup_batched_in_encoder_with_status(&encoder, dispatch, &status_buffer);
    encoder.end_encoding();

    Ok((
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: dispatch.job_count,
        },
        None,
    ))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_cleanup_batched_in_encoder(
    encoder: &ComputeCommandEncoderRef,
    dispatch: ClassicCleanupBatchDispatch<'_>,
) -> Result<(DirectStatusCheck, Option<Buffer>), Error> {
    let status_buffer = zeroed_shared_buffer(
        &dispatch.runtime.device,
        dispatch.job_count.max(1) * size_of::<J2kClassicStatus>(),
    )?;
    dispatch_classic_cleanup_batched_in_encoder_with_status(encoder, dispatch, &status_buffer);

    Ok((
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: dispatch.job_count,
        },
        None,
    ))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_cleanup_batched_in_encoder_with_status(
    encoder: &ComputeCommandEncoderRef,
    dispatch: ClassicCleanupBatchDispatch<'_>,
    status_buffer: &Buffer,
) {
    let ClassicCleanupBatchDispatch {
        runtime,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
    } = dispatch;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
}

#[cfg(target_os = "macos")]
fn classic_repeated_batch_params(
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
) -> Result<J2kClassicRepeatedBatchParams, Error> {
    Ok(J2kClassicRepeatedBatchParams {
        job_count: j2k_u32_param(job_count, "classic repeated base job count exceeds u32")?,
        output_plane_len: j2k_u32_param(
            output_plane_len,
            "classic repeated output plane len exceeds u32",
        )?,
        batch_count: j2k_u32_param(
            total_job_count / job_count.max(1),
            "classic repeated batch count exceeds u32",
        )?,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_cleanup_repeated_batched_in_command_buffer(
    dispatch: ClassicRepeatedCleanupDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let ClassicRepeatedCleanupDispatch {
        runtime,
        command_buffer,
        coded_data,
        jobs,
        job_count,
        total_job_count,
        output_plane_len,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
    } = dispatch;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_repeated_batched
    } else {
        &runtime.classic_cleanup_repeated_batched
    };
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        total_job_count.max(1) * size_of::<J2kClassicStatus>(),
    )?;
    let repeated = classic_repeated_batch_params(job_count, total_job_count, output_plane_len)?;

    let encoder = new_compute_command_encoder(command_buffer)?;
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();

    Ok(DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
    dispatch: ClassicPlainDevRepeatedCleanupDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let ClassicPlainDevRepeatedCleanupDispatch {
        runtime,
        command_buffer,
        coded_data,
        jobs,
        job_count,
        total_job_count,
        output_plane_len,
        segments,
        decoded,
        coefficients_scratch,
        states_scratch,
    } = dispatch;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        total_job_count.max(1) * size_of::<J2kClassicStatus>(),
    )?;
    let repeated = classic_repeated_batch_params(job_count, total_job_count, output_plane_len)?;

    let encoder = new_compute_command_encoder(command_buffer)?;
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_plain_dev_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_buffer(6, Some(states_scratch), 0);
    encoder.set_bytes(
        7,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    let width = runtime
        .classic_cleanup_plain_dev_repeated_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_classic_store_repeated_batched_in_command_buffer(
    dispatch: ClassicRepeatedStoreDispatch<'_>,
) -> Result<(), Error> {
    let ClassicRepeatedStoreDispatch {
        runtime,
        command_buffer,
        jobs,
        job_count,
        total_job_count,
        output_plane_len,
        decoded,
        coefficients_scratch,
    } = dispatch;
    let repeated = classic_repeated_batch_params(job_count, total_job_count, output_plane_len)?;

    let encoder = new_compute_command_encoder(command_buffer)?;
    encoder.set_compute_pipeline_state(&runtime.classic_store_repeated_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_buffer(1, Some(jobs), 0);
    encoder.set_buffer(2, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
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
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctClassicBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                segments: &sub_band.segments,
                output_base: index * per_instance_len,
            }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
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
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
            }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct DistinctClassicBatch<'a> {
    pub(in crate::compute) coded_data: &'a [u8],
    pub(in crate::compute) jobs: &'a [J2kClassicCleanupBatchJob],
    pub(in crate::compute) segments: &'a [J2kClassicSegment],
    pub(in crate::compute) output_base: usize,
}

#[cfg(target_os = "macos")]
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

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_distinct_classic_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl Iterator<Item = DistinctClassicBatch<'a>> + Clone,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
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

    let coded_buffer = copied_slice_buffer(&runtime.device, &coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, &jobs)?;
    let segments_buffer = copied_slice_buffer(&runtime.device, &segments)?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_command_buffer(
        command_buffer,
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

#[cfg(all(test, target_os = "macos"))]
mod distinct_metadata_tests;
