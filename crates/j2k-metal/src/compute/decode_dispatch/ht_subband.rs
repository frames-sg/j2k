// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    default_metal_ht_chunk_limits, dispatch_1d_pipeline, encode_metal_ht_batches_in_encoder,
    encode_repeated_metal_ht_batch_in_command_buffer, size_of, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, DirectStatusCheck, Error, HtBatchInput, HtCodeBlockDecodeJob,
    J2kHtCleanupBatchJob, MetalRuntime, PreparedHtSubBand, PreparedHtSubBandGroup,
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
    encode_repeated_metal_ht_batch_in_command_buffer(
        runtime,
        command_buffer,
        HtBatchInput {
            source_index: 0,
            payload: job.payload_source.as_ht_payload_source(),
            jobs: &job.jobs,
            output_base: 0,
            execution_owner: &job.execution_owner,
        },
        count,
        job.width as usize * job.height as usize,
        output,
        default_metal_ht_chunk_limits(),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    encode_repeated_metal_ht_batch_in_command_buffer(
        runtime,
        command_buffer,
        HtBatchInput {
            source_index: 0,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &group.jobs,
            output_base: 0,
            execution_owner: &group.execution_owner,
        },
        count,
        group.total_coefficients,
        output,
        default_metal_ht_chunk_limits(),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_ht_sub_band_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    job: &PreparedHtSubBand,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    encode_metal_ht_batches_in_encoder(
        runtime,
        encoder,
        &[HtBatchInput {
            source_index: 0,
            payload: job.payload_source.as_ht_payload_source(),
            jobs: &job.jobs,
            output_base: 0,
            execution_owner: &job.execution_owner,
        }],
        output,
        job.width as usize * job.height as usize,
        default_metal_ht_chunk_limits(),
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    encode_metal_ht_batches_in_encoder(
        runtime,
        encoder,
        &[HtBatchInput {
            source_index: 0,
            payload: group.payload_source.as_ht_payload_source(),
            jobs: &group.jobs,
            output_base: 0,
            execution_owner: &group.execution_owner,
        }],
        output,
        group.total_coefficients,
        default_metal_ht_chunk_limits(),
    )
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
