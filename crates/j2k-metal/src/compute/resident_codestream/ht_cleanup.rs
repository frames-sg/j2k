// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_ht_status_error, dispatch_single_thread, dispatch_zero_u32_buffer_in_encoder,
    ht_batch_output_word_count, ht_output_word_count, new_command_buffer,
    new_compute_command_encoder, size_of, zeroed_shared_buffer, Buffer, ComputeCommandEncoderRef,
    Error, J2kHtCleanupBatchJob, J2kHtCleanupParams, J2kHtRepeatedBatchParams, J2kHtStatus,
    MTLSize, MetalRuntime, J2K_HT_STATUS_OK,
};
#[cfg(target_os = "macos")]
use crate::compute::decode_dispatch::MetalHtPipelineKind;

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct HtCleanupBatchDispatch<'a> {
    pub(in crate::compute) coded_data: &'a Buffer,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) job_count: usize,
    pub(in crate::compute) decoded: &'a Buffer,
    pub(in crate::compute) status_buffer: &'a Buffer,
    pub(in crate::compute) status_offset_bytes: u64,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct HtCleanupRepeatedBatchDispatch<'a> {
    pub(in crate::compute) coded_data: &'a Buffer,
    pub(in crate::compute) jobs: &'a Buffer,
    pub(in crate::compute) base_job_count: usize,
    pub(in crate::compute) repeated: J2kHtRepeatedBatchParams,
    pub(in crate::compute) decoded: &'a Buffer,
    pub(in crate::compute) status_buffer: &'a Buffer,
    pub(in crate::compute) status_offset_bytes: u64,
}

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
    encoder.memory_barrier_with_resources(&[decoded]);
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
    encoder.memory_barrier_with_resources(&[decoded]);
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

#[cfg(all(test, target_os = "macos"))]
mod tests;

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_ht_cleanup_batched_in_encoder_with_status_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    pipeline_kind: MetalHtPipelineKind,
    dispatch: HtCleanupBatchDispatch<'_>,
) {
    let pipeline = match pipeline_kind {
        MetalHtPipelineKind::CleanupOnly => &runtime.ht_cleanup_batched_cleanup_only,
        MetalHtPipelineKind::SigProp => &runtime.ht_cleanup_batched_sigprop,
        MetalHtPipelineKind::MagRef => &runtime.ht_cleanup_batched_magref,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(dispatch.coded_data), 0);
    encoder.set_buffer(1, Some(dispatch.decoded), 0);
    encoder.set_buffer(2, Some(dispatch.jobs), 0);
    encoder.set_buffer(3, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(5, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(
        7,
        Some(dispatch.status_buffer),
        dispatch.status_offset_bytes,
    );
    let width = pipeline
        .thread_execution_width()
        .max(1)
        .min(dispatch.job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: dispatch.job_count as u64,
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
pub(in crate::compute) fn dispatch_ht_cleanup_repeated_batched_in_encoder_with_status_offset(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    pipeline_kind: MetalHtPipelineKind,
    dispatch: HtCleanupRepeatedBatchDispatch<'_>,
) {
    let pipeline = match pipeline_kind {
        MetalHtPipelineKind::CleanupOnly => &runtime.ht_cleanup_repeated_batched_cleanup_only,
        MetalHtPipelineKind::SigProp => &runtime.ht_cleanup_repeated_batched_sigprop,
        MetalHtPipelineKind::MagRef => &runtime.ht_cleanup_repeated_batched_magref,
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(dispatch.coded_data), 0);
    encoder.set_buffer(1, Some(dispatch.decoded), 0);
    encoder.set_buffer(2, Some(dispatch.jobs), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kHtRepeatedBatchParams>() as u64,
        (&raw const dispatch.repeated).cast(),
    );
    encoder.set_buffer(4, Some(&runtime.ht_vlc_table0), 0);
    encoder.set_buffer(5, Some(&runtime.ht_vlc_table1), 0);
    encoder.set_buffer(6, Some(&runtime.ht_uvlc_table0), 0);
    encoder.set_buffer(7, Some(&runtime.ht_uvlc_table1), 0);
    encoder.set_buffer(
        8,
        Some(dispatch.status_buffer),
        dispatch.status_offset_bytes,
    );
    let width = pipeline
        .thread_execution_width()
        .max(1)
        .min(dispatch.base_job_count as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: dispatch.base_job_count as u64,
            height: u64::from(dispatch.repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}
