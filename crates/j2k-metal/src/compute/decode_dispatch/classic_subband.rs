// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    classic_batch_is_plain_arithmetic, classic_batch_uses_plain_fast_path,
    classic_repeated_uses_plain_fast_path, dispatch_classic_cleanup_batched_in_encoder,
    dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer,
    dispatch_classic_cleanup_repeated_batched_in_command_buffer,
    dispatch_classic_store_repeated_batched_in_command_buffer, dispatch_zero_u32_buffer_in_encoder,
    new_shared_buffer, take_classic_coefficients_scratch_buffer,
    take_classic_states_scratch_buffer, Buffer, ClassicCleanupBatchDispatch,
    ClassicPlainDevRepeatedCleanupDispatch, ClassicRepeatedCleanupDispatch,
    ClassicRepeatedStoreDispatch, CommandBufferRef, ComputeCommandEncoderRef, DirectScratchBuffer,
    DirectStatusCheck, Error, J2kClassicCleanupBatchJob, J2kClassicSegment, MetalRuntime,
    PreparedClassicSubBand, PreparedClassicSubBandGroup,
};

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct ClassicBatchView<'a> {
    coded_buffer: &'a Buffer,
    jobs_buffer: &'a Buffer,
    segments_buffer: &'a Buffer,
    jobs: &'a [J2kClassicCleanupBatchJob],
    segments: &'a [J2kClassicSegment],
    output_plane_len: usize,
    repeated_overflow_message: &'static str,
}

#[cfg(target_os = "macos")]
impl<'a> ClassicBatchView<'a> {
    fn from_sub_band(job: &'a PreparedClassicSubBand) -> Self {
        Self {
            coded_buffer: &job.coded_buffer,
            jobs_buffer: &job.jobs_buffer,
            segments_buffer: &job.segments_buffer,
            jobs: &job.jobs,
            segments: &job.segments,
            output_plane_len: job.width as usize * job.height as usize,
            repeated_overflow_message: "classic J2K MetalDirect repeated job count overflow",
        }
    }

    fn from_group(group: &'a PreparedClassicSubBandGroup) -> Self {
        Self {
            coded_buffer: &group.coded_buffer,
            jobs_buffer: &group.jobs_buffer,
            segments_buffer: &group.segments_buffer,
            jobs: &group.jobs,
            segments: &group.segments,
            output_plane_len: group.total_coefficients,
            repeated_overflow_message:
                "classic J2K MetalDirect repeated grouped job count overflow",
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum RepeatedClassicKernel {
    PlainFast,
    PlainDeviceState,
    General,
}

#[cfg(target_os = "macos")]
fn select_repeated_classic_kernel(
    runtime: &MetalRuntime,
    count: usize,
    batch: ClassicBatchView<'_>,
) -> RepeatedClassicKernel {
    let plain_fast = classic_repeated_uses_plain_fast_path(count, batch.jobs, batch.segments)
        && runtime
            .classic_cleanup_plain_repeated_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    if plain_fast {
        RepeatedClassicKernel::PlainFast
    } else if count <= 16 && classic_batch_is_plain_arithmetic(batch.jobs, batch.segments) {
        RepeatedClassicKernel::PlainDeviceState
    } else {
        RepeatedClassicKernel::General
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct RepeatedClassicCleanupRequest<'r, 'p> {
    runtime: &'r MetalRuntime,
    command_buffer: &'r CommandBufferRef,
    batch: ClassicBatchView<'p>,
    total_jobs: usize,
    output: &'r Buffer,
    coefficients_scratch: &'r Buffer,
    states_scratch: Option<&'r Buffer>,
}

#[cfg(target_os = "macos")]
fn dispatch_repeated_classic_cleanup(
    request: RepeatedClassicCleanupRequest<'_, '_>,
    kernel: RepeatedClassicKernel,
) -> Result<DirectStatusCheck, Error> {
    let RepeatedClassicCleanupRequest {
        runtime,
        command_buffer,
        batch,
        total_jobs,
        output,
        coefficients_scratch,
        states_scratch,
    } = request;
    match kernel {
        RepeatedClassicKernel::PlainFast | RepeatedClassicKernel::General => {
            dispatch_classic_cleanup_repeated_batched_in_command_buffer(
                ClassicRepeatedCleanupDispatch {
                    runtime,
                    command_buffer,
                    coded_data: batch.coded_buffer,
                    jobs: batch.jobs_buffer,
                    job_count: batch.jobs.len(),
                    total_job_count: total_jobs,
                    output_plane_len: batch.output_plane_len,
                    use_plain_fast_path: kernel == RepeatedClassicKernel::PlainFast,
                    segments: batch.segments_buffer,
                    decoded: output,
                    coefficients_scratch,
                },
            )
        }
        RepeatedClassicKernel::PlainDeviceState => {
            let states_scratch = states_scratch.ok_or(Error::MetalStateInvariant {
                state: "classic repeated cleanup dispatch",
                reason: "plain device-state dispatch has no state scratch buffer",
            })?;
            dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
                ClassicPlainDevRepeatedCleanupDispatch {
                    runtime,
                    command_buffer,
                    coded_data: batch.coded_buffer,
                    jobs: batch.jobs_buffer,
                    job_count: batch.jobs.len(),
                    total_job_count: total_jobs,
                    output_plane_len: batch.output_plane_len,
                    segments: batch.segments_buffer,
                    decoded: output,
                    coefficients_scratch,
                    states_scratch,
                },
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn empty_repeated_classic_execution(
    runtime: &MetalRuntime,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let empty = new_shared_buffer(&runtime.device, 1)?;
    Ok((
        vec![empty.clone()],
        DirectStatusCheck::Classic {
            buffer: empty,
            len: 0,
            source_indices: Some(Vec::new()),
        },
    ))
}

#[cfg(target_os = "macos")]
fn encode_repeated_classic_batch_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batch: ClassicBatchView<'_>,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || batch.jobs.is_empty() {
        return empty_repeated_classic_execution(runtime);
    }
    let total_jobs = batch
        .jobs
        .len()
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: batch.repeated_overflow_message.to_string(),
        })?;
    let kernel = select_repeated_classic_kernel(runtime, count, batch);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, total_jobs)?;
    let states_scratch = if kernel == RepeatedClassicKernel::PlainDeviceState {
        Some(take_classic_states_scratch_buffer(runtime, total_jobs)?)
    } else {
        None
    };
    let status_check = dispatch_repeated_classic_cleanup(
        RepeatedClassicCleanupRequest {
            runtime,
            command_buffer,
            batch,
            total_jobs,
            output,
            coefficients_scratch: &coefficients_scratch.buffer,
            states_scratch: states_scratch
                .as_ref()
                .map(|scratch| scratch.buffer.buffer()),
        },
        kernel,
    )?;
    if kernel != RepeatedClassicKernel::PlainFast {
        dispatch_classic_store_repeated_batched_in_command_buffer(ClassicRepeatedStoreDispatch {
            runtime,
            command_buffer,
            jobs: batch.jobs_buffer,
            job_count: batch.jobs.len(),
            total_job_count: total_jobs,
            output_plane_len: batch.output_plane_len,
            decoded: output,
            coefficients_scratch: &coefficients_scratch.buffer,
        })?;
    }
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        scratch_buffers.push(states_scratch);
    }
    Ok((
        vec![
            batch.coded_buffer.clone(),
            batch.jobs_buffer.clone(),
            batch.segments_buffer.clone(),
        ],
        status_check,
    ))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedClassicSubBand,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    encode_repeated_classic_batch_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        ClassicBatchView::from_sub_band(job),
        count,
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedClassicSubBandGroup,
    count: usize,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    encode_repeated_classic_batch_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        ClassicBatchView::from_group(group),
        count,
        output,
        scratch_buffers,
    )
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
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
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
        encoder.memory_barrier_with_resources(&[output]);
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
        None,
    )?;
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
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Classic {
                buffer: empty,
                len: 0,
                source_indices: Some(Vec::new()),
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
        encoder.memory_barrier_with_resources(&[output]);
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
        None,
    )?;
    let mut retained_buffers = vec![coded_buffer, jobs_buffer, segments_buffer];
    scratch_buffers.push(coefficients_scratch);
    if let Some(states_scratch) = states_scratch {
        retained_buffers.push(states_scratch);
    }
    Ok((retained_buffers, status_check))
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
