// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_ht_status_error, dispatch_single_thread, dispatch_zero_u32_buffer_in_encoder,
    ht_batch_output_word_count, ht_output_word_count, new_command_buffer,
    new_compute_command_encoder, size_of, zeroed_shared_buffer, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, DirectStatusCheck, Error, J2kHtCleanupBatchJob, J2kHtCleanupParams,
    J2kHtRepeatedBatchParams, J2kHtStatus, MTLSize, MetalRuntime, J2K_HT_STATUS_OK,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    params: J2kHtCleanupParams,
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = copied_slice_buffer(&runtime.device, coded_data)?;
    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kHtStatus>())?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        &encoder,
        decoded,
        ht_output_word_count(
            params.output_offset,
            params.output_stride,
            params.width,
            params.height,
        )?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kHtCleanupParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    dispatch_single_thread(&encoder);
    encoder.end_encoding();
    commit_and_wait_metal(&command_buffer)?;

    let status = checked_buffer_read::<J2kHtStatus>(&status_buffer, "HT cleanup status")?;
    if status.code != J2K_HT_STATUS_OK {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kHtCleanupBatchJob],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = copied_slice_buffer(&runtime.device, coded_data)?;
    let jobs_buffer = copied_slice_buffer(&runtime.device, jobs)?;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        jobs.len().max(1) * size_of::<J2kHtStatus>(),
    )?;

    let command_buffer = new_command_buffer(&runtime.queue)?;
    let encoder = new_compute_command_encoder(&command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(
        runtime,
        &encoder,
        decoded,
        ht_batch_output_word_count(jobs)?,
    )?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
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
    encoder.end_encoding();
    commit_and_wait_metal(&command_buffer)?;

    let statuses =
        checked_buffer_slice::<J2kHtStatus>(&status_buffer, jobs.len(), "HT cleanup statuses")?;
    if let Some(status) = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        return Err(decode_ht_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup_batched_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer =
        zeroed_shared_buffer(&runtime.device, job_count.max(1) * size_of::<J2kHtStatus>())?;

    let encoder = new_compute_command_encoder(command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(runtime, &encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        &encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup_batched_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    decoded_word_count: usize,
) -> Result<DirectStatusCheck, Error> {
    let status_buffer =
        zeroed_shared_buffer(&runtime.device, job_count.max(1) * size_of::<J2kHtStatus>())?;
    dispatch_zero_u32_buffer_in_encoder(runtime, encoder, decoded, decoded_word_count)?;
    dispatch_ht_cleanup_batched_in_encoder_with_status(
        runtime,
        encoder,
        coded_data,
        jobs,
        job_count,
        decoded,
        &status_buffer,
    );

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: job_count,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup_batched_in_encoder_with_status(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    coded_data: &Buffer,
    jobs: &Buffer,
    job_count: usize,
    decoded: &Buffer,
    status_buffer: &Buffer,
) {
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(7, Some(status_buffer), 0);
    let width = runtime
        .ht_cleanup_batched
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

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct HtRepeatedCleanupDispatch<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) coded_data: &'a Buffer,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) base_job_count: usize,
    pub(in crate::compute) total_job_count: usize,
    pub(in crate::compute) output_plane_len: usize,
    pub(in crate::compute) decoded: &'a Buffer,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup_repeated_batched_in_command_buffer(
    dispatch: HtRepeatedCleanupDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let HtRepeatedCleanupDispatch {
        runtime,
        command_buffer,
        coded_data,
        jobs,
        base_job_count,
        total_job_count,
        output_plane_len,
        decoded,
    } = dispatch;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        total_job_count.max(1) * size_of::<J2kHtStatus>(),
    )?;
    let batch_count =
        total_job_count
            .checked_div(base_job_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated base job count is zero".to_string(),
            })?;
    let decoded_word_count =
        output_plane_len
            .checked_mul(batch_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "HTJ2K MetalDirect repeated output span overflow".to_string(),
            })?;
    let repeated = J2kHtRepeatedBatchParams {
        job_count: u32::try_from(base_job_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated base job count exceeds u32".to_string(),
        })?,
        output_plane_len: u32::try_from(output_plane_len).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated output plane length exceeds u32".to_string(),
        })?,
        batch_count: u32::try_from(batch_count).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect repeated batch count exceeds u32".to_string(),
        })?,
    };

    let encoder = new_compute_command_encoder(command_buffer)?;
    dispatch_zero_u32_buffer_in_encoder(runtime, &encoder, decoded, decoded_word_count)?;
    encoder.set_compute_pipeline_state(&runtime.ht_cleanup_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kHtRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(5, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(7, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(8, Some(&status_buffer), 0);
    let width = runtime
        .ht_cleanup_repeated_batched
        .thread_execution_width()
        .max(1)
        .min(base_job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: base_job_count as u64,
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

    Ok(DirectStatusCheck::Ht {
        buffer: status_buffer,
        len: total_job_count,
    })
}
