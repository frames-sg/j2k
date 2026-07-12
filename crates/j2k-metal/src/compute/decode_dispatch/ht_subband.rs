// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    dispatch_1d_pipeline, dispatch_ht_cleanup_batched_in_encoder,
    dispatch_ht_cleanup_repeated_batched_in_command_buffer, new_shared_buffer, prepared_ht_buffer,
    size_of, Buffer, CommandBufferRef, ComputeCommandEncoderRef, DirectStatusCheck, Error,
    HtCodeBlockDecodeJob, HtRepeatedCleanupDispatch, J2kHtCleanupBatchJob, MetalRuntime,
    PreparedHtSubBand, PreparedHtSubBandGroup,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn required_ht_output_len(
    job: HtCodeBlockDecodeJob<'_>,
) -> Result<usize, Error> {
    if job.height == 0 {
        return Ok(0);
    }

    job.output_stride
        .checked_mul(job.height as usize - 1)
        .and_then(|prefix| prefix.checked_add(job.width as usize))
        .ok_or_else(|| Error::MetalKernel {
            message: "HTJ2K Metal output size overflow".to_string(),
        })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedHtSubBand,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || job.jobs.is_empty() {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
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
            message: "HTJ2K MetalDirect repeated job count overflow".to_string(),
        })?;
    let coded_buffer = prepared_ht_buffer(job.coded_buffer.as_ref(), "coded")?.clone();
    let jobs_buffer = prepared_ht_buffer(job.jobs_buffer.as_ref(), "jobs")?.clone();
    let status_check =
        dispatch_ht_cleanup_repeated_batched_in_command_buffer(HtRepeatedCleanupDispatch {
            runtime,
            command_buffer,
            coded_data: &coded_buffer,
            jobs: &jobs_buffer,
            base_job_count: job.jobs.len(),
            total_job_count: total_jobs,
            output_plane_len: job.width as usize * job.height as usize,
            decoded: output,
        })?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
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
            message: "HTJ2K MetalDirect repeated grouped job count overflow".to_string(),
        })?;
    let coded_buffer = group.coded_arena.buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check =
        dispatch_ht_cleanup_repeated_batched_in_command_buffer(HtRepeatedCleanupDispatch {
            runtime,
            command_buffer,
            coded_data: &coded_buffer,
            jobs: &jobs_buffer,
            base_job_count: group.jobs.len(),
            total_job_count: total_jobs,
            output_plane_len: group.total_coefficients,
            decoded: output,
        })?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_ht_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedHtSubBand,
    output: &Buffer,
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
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = prepared_ht_buffer(job.coded_buffer.as_ref(), "coded")?.clone();
    let jobs_buffer = prepared_ht_buffer(job.jobs_buffer.as_ref(), "jobs")?.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        job.jobs.len(),
        output,
        job.width as usize * job.height as usize,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = new_shared_buffer(&runtime.device, 1)?;
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = group.coded_arena.buffer.clone();
    let jobs_buffer = group.jobs_buffer.clone();
    let status_check = dispatch_ht_cleanup_batched_in_encoder(
        runtime,
        encoder,
        &coded_buffer,
        &jobs_buffer,
        group.jobs.len(),
        output,
        group.total_coefficients,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn ht_output_word_count(
    output_offset: u32,
    output_stride: u32,
    width: u32,
    height: u32,
) -> Result<usize, Error> {
    let end = if width == 0 || height == 0 {
        u64::from(output_offset)
    } else {
        u64::from(output_offset)
            .checked_add(u64::from(height - 1) * u64::from(output_stride))
            .and_then(|offset| offset.checked_add(u64::from(width)))
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K Metal output span overflow".to_string(),
            })?
    };
    usize::try_from(end).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal output span exceeds usize".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn ht_batch_output_word_count(
    jobs: &[J2kHtCleanupBatchJob],
) -> Result<usize, Error> {
    let mut word_count = 0usize;
    for job in jobs {
        let job_word_count =
            ht_output_word_count(job.output_offset, job.output_stride, job.width, job.height)?;
        word_count = word_count.max(job_word_count);
    }
    Ok(word_count)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_zero_u32_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    buffer: &Buffer,
    word_count: usize,
) -> Result<(), Error> {
    let word_count = u32::try_from(word_count).map_err(|_| Error::MetalKernel {
        message: "HTJ2K Metal zero-fill word count exceeds u32".to_string(),
    })?;
    if word_count == 0 {
        return Ok(());
    }

    encoder.set_compute_pipeline_state(&runtime.zero_u32_buffer);
    encoder.set_buffer(0, Some(buffer), 0);
    encoder.set_bytes(1, size_of::<u32>() as u64, (&raw const word_count).cast());
    dispatch_1d_pipeline(encoder, &runtime.zero_u32_buffer, u64::from(word_count));
    Ok(())
}
