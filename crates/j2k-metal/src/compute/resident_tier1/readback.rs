// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_read, completed_command_buffers_gpu_duration, encode_status_error,
    finish_completed_resident_lossless_codestream_batch, size_of,
    wait_resident_codestream_command_buffer, Buffer, CommandBufferRef, Error,
    J2kClassicEncodeBatchJob, J2kClassicEncodeStatus, J2kCodestreamAssemblyStatus,
    J2kHtEncodeStatus, J2kPendingResidentLosslessCodestream,
    J2kPendingResidentLosslessCodestreamBatch, J2kResidentLosslessCodestream,
    J2kResidentLosslessCodestreamBatchResult, J2kResidentTier1StatusKind,
    J2kResidentTier1StatusReadback, MTLResourceOptions, MetalRuntime, J2K_ENCODE_STATUS_OK,
};

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream(
    pending: J2kPendingResidentLosslessCodestream,
) -> Result<J2kResidentLosslessCodestream, Error> {
    wait_resident_codestream_command_buffer(&pending.command_buffer)?;
    let gpu_duration = completed_command_buffers_gpu_duration(
        &pending.retained_command_buffers,
        &pending.command_buffer,
    );
    let status = checked_buffer_read::<J2kCodestreamAssemblyStatus>(
        &pending.status_buffer,
        "resident codestream assembly status",
    )?;
    if status.code != J2K_ENCODE_STATUS_OK {
        return Err(encode_status_error(
            pending.status_stage,
            status.code,
            status.detail,
        ));
    }
    let data_len = usize::try_from(status.data_len).map_err(|_| Error::MetalKernel {
        message: pending.length_error.to_string(),
    })?;
    if data_len > pending.capacity {
        return Err(Error::MetalKernel {
            message: pending.capacity_error.to_string(),
        });
    }
    Ok(J2kResidentLosslessCodestream {
        buffer: pending.buffer,
        byte_offset: 0,
        byte_len: data_len,
        capacity: pending.capacity,
        gpu_duration,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream_batch(
    pending: J2kPendingResidentLosslessCodestreamBatch,
) -> Result<J2kResidentLosslessCodestreamBatchResult, Error> {
    wait_resident_codestream_command_buffer(&pending.command_buffer)?;
    finish_completed_resident_lossless_codestream_batch(pending)
}

#[cfg(target_os = "macos")]
pub(crate) fn wait_resident_lossless_codestream_batches(
    pending_batches: Vec<J2kPendingResidentLosslessCodestreamBatch>,
) -> Result<Vec<J2kResidentLosslessCodestreamBatchResult>, Error> {
    if let Some(last) = pending_batches.last() {
        // These command buffers are submitted on the same Metal queue before
        // harvest, so completing the final one implies earlier chunks are done.
        wait_resident_codestream_command_buffer(&last.command_buffer)?;
    }
    pending_batches
        .into_iter()
        .map(finish_completed_resident_lossless_codestream_batch)
        .collect()
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct ResidentTier1StatusReadbackRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) status_buffer: &'a Buffer,
    pub(in crate::compute) kind: J2kResidentTier1StatusKind,
    pub(in crate::compute) classic_style_flags: u32,
    pub(in crate::compute) classic_jobs: Option<&'a [J2kClassicEncodeBatchJob]>,
    pub(in crate::compute) count: usize,
    pub(in crate::compute) status_size: usize,
    pub(in crate::compute) profile_stages: bool,
}

#[cfg(target_os = "macos")]
impl<'a> ResidentTier1StatusReadbackRequest<'a> {
    pub(in crate::compute) fn high_throughput(
        runtime: &'a MetalRuntime,
        command_buffer: &'a CommandBufferRef,
        status_buffer: &'a Buffer,
        count: usize,
        profile_stages: bool,
    ) -> Self {
        Self {
            runtime,
            command_buffer,
            status_buffer,
            kind: J2kResidentTier1StatusKind::HighThroughput,
            classic_style_flags: 0,
            classic_jobs: None,
            count,
            status_size: size_of::<J2kHtEncodeStatus>(),
            profile_stages,
        }
    }

    pub(in crate::compute) fn classic(
        runtime: &'a MetalRuntime,
        command_buffer: &'a CommandBufferRef,
        status_buffer: &'a Buffer,
        classic_style_flags: u32,
        classic_jobs: &'a [J2kClassicEncodeBatchJob],
        profile_stages: bool,
    ) -> Self {
        Self {
            runtime,
            command_buffer,
            status_buffer,
            kind: J2kResidentTier1StatusKind::Classic,
            classic_style_flags,
            classic_jobs: Some(classic_jobs),
            count: classic_jobs.len(),
            status_size: size_of::<J2kClassicEncodeStatus>(),
            profile_stages,
        }
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn schedule_resident_tier1_status_readback(
    request: ResidentTier1StatusReadbackRequest<'_>,
) -> Result<Option<J2kResidentTier1StatusReadback>, Error> {
    let ResidentTier1StatusReadbackRequest {
        runtime,
        command_buffer,
        status_buffer,
        kind,
        classic_style_flags,
        classic_jobs,
        count,
        status_size,
        profile_stages,
    } = request;
    if !profile_stages || count == 0 {
        return Ok(None);
    }
    let byte_len = count
        .checked_mul(status_size)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident Tier-1 status readback size overflow".to_string(),
        })?;
    let readback = runtime.device.new_buffer(
        byte_len.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let blit = command_buffer.new_blit_command_encoder();
    blit.copy_from_buffer(status_buffer, 0, &readback, 0, byte_len as u64);
    blit.end_encoding();
    Ok(Some(J2kResidentTier1StatusReadback {
        buffer: readback,
        kind,
        classic_style_flags,
        classic_jobs: classic_jobs.map(<[J2kClassicEncodeBatchJob]>::to_vec),
        count,
    }))
}
