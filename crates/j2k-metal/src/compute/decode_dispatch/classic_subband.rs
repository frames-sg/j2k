// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    classic_batch_is_plain_arithmetic, classic_batch_uses_plain_fast_path,
    classic_repeated_uses_plain_fast_path, dispatch_classic_cleanup_batched_in_encoder,
    dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer,
    dispatch_classic_cleanup_repeated_batched_in_command_buffer,
    dispatch_classic_store_repeated_batched_in_command_buffer, dispatch_zero_u32_buffer_in_encoder,
    take_classic_coefficients_scratch_buffer, take_classic_states_scratch_buffer, Buffer,
    ClassicCleanupBatchDispatch, ClassicPlainDevRepeatedCleanupDispatch,
    ClassicRepeatedCleanupDispatch, ClassicRepeatedStoreDispatch, CommandBufferRef,
    ComputeCommandEncoderRef, DirectScratchBuffer, DirectStatusCheck, Error, MTLResourceOptions,
    MetalRuntime, PreparedClassicSubBand, PreparedClassicSubBandGroup,
};

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "classic subband dispatch mirrors fixed Metal argument and resource order"
)]
pub(in crate::compute) fn encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedClassicSubBand,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    if job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = job
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &job.jobs, &job.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&job.jobs, &job.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            ClassicRepeatedCleanupDispatch {
                runtime,
                command_buffer,
                coded_data: &coded_buffer,
                jobs: &jobs_buffer,
                job_count: job.jobs.len(),
                total_job_count: total_jobs,
                output_plane_len: job.width as usize * job.height as usize,
                use_plain_fast_path: true,
                segments: &segments_buffer,
                decoded: output,
                coefficients_scratch: &coefficients_scratch.buffer,
            },
        )?
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            ClassicPlainDevRepeatedCleanupDispatch {
                runtime,
                command_buffer,
                coded_data: &coded_buffer,
                jobs: &jobs_buffer,
                job_count: job.jobs.len(),
                total_job_count: total_jobs,
                output_plane_len: job.width as usize * job.height as usize,
                segments: &segments_buffer,
                decoded: output,
                coefficients_scratch: &coefficients_scratch.buffer,
                states_scratch: &states_scratch.buffer,
            },
        )?
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            ClassicRepeatedCleanupDispatch {
                runtime,
                command_buffer,
                coded_data: &coded_buffer,
                jobs: &jobs_buffer,
                job_count: job.jobs.len(),
                total_job_count: total_jobs,
                output_plane_len: job.width as usize * job.height as usize,
                use_plain_fast_path,
                segments: &segments_buffer,
                decoded: output,
                coefficients_scratch: &coefficients_scratch.buffer,
            },
        )?
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(ClassicRepeatedStoreDispatch {
            runtime,
            command_buffer,
            jobs: &jobs_buffer,
            job_count: job.jobs.len(),
            total_job_count: total_jobs,
            output_plane_len: job.width as usize * job.height as usize,
            decoded: output,
            coefficients_scratch: &coefficients_scratch.buffer,
        })?;
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "grouped subband dispatch mirrors fixed Metal argument and resource order"
)]
pub(in crate::compute) fn encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedClassicSubBandGroup,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let total_jobs = group
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "classic J2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path =
        classic_repeated_uses_plain_fast_path(count, &group.jobs, &group.segments)
            && runtime
                .classic_cleanup_plain_repeated_batched
                .max_total_threads_per_threadgroup()
                >= 32;
    let use_plain_dev_path = !use_plain_fast_path
        && count <= 16
        && classic_batch_is_plain_arithmetic(&group.jobs, &group.segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if use_plain_dev_path {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = if use_plain_fast_path {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            ClassicRepeatedCleanupDispatch {
                runtime,
                command_buffer,
                coded_data: &coded_buffer,
                jobs: &jobs_buffer,
                job_count: group.jobs.len(),
                total_job_count: total_jobs,
                output_plane_len: group.total_coefficients,
                use_plain_fast_path: true,
                segments: &segments_buffer,
                decoded: output,
                coefficients_scratch: &coefficients_scratch.buffer,
            },
        )?
    } else if let Some(states_scratch) = states_scratch.as_ref() {
        dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
            ClassicPlainDevRepeatedCleanupDispatch {
                runtime,
                command_buffer,
                coded_data: &coded_buffer,
                jobs: &jobs_buffer,
                job_count: group.jobs.len(),
                total_job_count: total_jobs,
                output_plane_len: group.total_coefficients,
                segments: &segments_buffer,
                decoded: output,
                coefficients_scratch: &coefficients_scratch.buffer,
                states_scratch: &states_scratch.buffer,
            },
        )?
    } else {
        dispatch_classic_cleanup_repeated_batched_in_command_buffer(
            ClassicRepeatedCleanupDispatch {
                runtime,
                command_buffer,
                coded_data: &coded_buffer,
                jobs: &jobs_buffer,
                job_count: group.jobs.len(),
                total_job_count: total_jobs,
                output_plane_len: group.total_coefficients,
                use_plain_fast_path,
                segments: &segments_buffer,
                decoded: output,
                coefficients_scratch: &coefficients_scratch.buffer,
            },
        )?
    };
    if !use_plain_fast_path {
        dispatch_classic_store_repeated_batched_in_command_buffer(ClassicRepeatedStoreDispatch {
            runtime,
            command_buffer,
            jobs: &jobs_buffer,
            job_count: group.jobs.len(),
            total_job_count: total_jobs,
            output_plane_len: group.total_coefficients,
            decoded: output,
            coefficients_scratch: &coefficients_scratch.buffer,
        })?;
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    let retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_classic_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedClassicSubBand,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if job.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = job.coded_buffer.clone();
    let jobs_buffer = job.jobs_buffer.clone();
    let segments_buffer = job.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&job.jobs, &job.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, job.jobs.len())?;
    if job.zero_fill {
        dispatch_zero_u32_buffer_in_encoder(
            runtime,
            encoder,
            output,
            job.width as usize * job.height as usize,
        )?;
    }
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        encoder,
        ClassicCleanupBatchDispatch {
            runtime,
            coded_data: &coded_buffer,
            jobs: &jobs_buffer,
            job_count: job.jobs.len(),
            use_plain_fast_path,
            segments: &segments_buffer,
            decoded: output,
            coefficients_scratch: &coefficients_scratch.buffer,
        },
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedClassicSubBandGroup,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let segments_buffer = group.segments_buffer.clone();
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&group.jobs, &group.segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, group.jobs.len())?;
    if group.zero_fill {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
    }
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_encoder(
        encoder,
        ClassicCleanupBatchDispatch {
            runtime,
            coded_data: &coded_buffer,
            jobs: &jobs_buffer,
            job_count: group.jobs.len(),
            use_plain_fast_path,
            segments: &segments_buffer,
            decoded: output,
            coefficients_scratch: &coefficients_scratch.buffer,
        },
    );
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}
