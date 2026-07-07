// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::test_counters;
use super::{
    active_forward_dwt53_buffers, borrow_mut_slice_buffer, checked_buffer_read,
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, decode_mct_status_error,
    dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline,
    dispatch_forward_dwt53_batched_pass, dispatch_forward_dwt53_pass, hybrid_stage_signpost,
    label_command_buffer, label_compute_encoder,
    metal_profile_coefficient_prep_split_commands_enabled, new_resident_encode_command_buffer,
    size_of, take_recyclable_private_buffer, with_runtime, with_runtime_for_session,
    zeroed_shared_buffer, Buffer, CommandBuffer, CommandBufferRef, Error,
    J2kBatchedPacketPayloadCopyDispatch, J2kForwardDwt53BatchedParams, J2kForwardDwt53Params,
    J2kForwardIctParams, J2kForwardRctParams, J2kLosslessCoefficientJob,
    J2kLosslessDeinterleaveParams, J2kLosslessDeviceBatchPrepareItem, J2kLosslessDeviceCodeBlock,
    J2kLosslessDevicePrepareJob, J2kMctStatus, J2kPacketPayloadCopyParams,
    J2kPreparedLosslessDeviceCodeBlocks, J2kQuantizeSubbandJob, J2kQuantizeSubbandParams,
    MTLResourceOptions, MTLSize, MetalRuntime, J2K_MCT_STATUS_OK,
    PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE, PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
};

#[cfg(target_os = "macos")]
pub(super) fn dispatch_batched_packet_payload_copy(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    dispatch: J2kBatchedPacketPayloadCopyDispatch<'_>,
) -> bool {
    if dispatch.tile_count == 0 || dispatch.max_payload_copy_jobs_per_tile == 0 {
        return false;
    }

    let signpost = hybrid_stage_signpost(dispatch.signpost_name);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, dispatch.label);
    encoder.set_compute_pipeline_state(&runtime.packet_payload_copy_batched);
    encoder.set_buffer(0, Some(dispatch.payload_buffer), 0);
    encoder.set_buffer(1, Some(dispatch.packet_output_buffer), 0);
    encoder.set_buffer(2, Some(dispatch.packet_job_buffer), 0);
    encoder.set_buffer(3, Some(dispatch.packet_status_buffer), 0);
    encoder.set_buffer(4, Some(dispatch.packet_payload_copy_job_buffer), 0);
    let params = J2kPacketPayloadCopyParams {
        bytes_per_thread: PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE,
        stripes_per_job: PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
    };
    encoder.set_bytes(
        5,
        size_of::<J2kPacketPayloadCopyParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.dispatch_threads(
        MTLSize {
            width: dispatch.max_payload_copy_jobs_per_tile,
            height: dispatch.tile_count,
            depth: u64::from(PACKET_PAYLOAD_COPY_STRIPES_PER_JOB),
        },
        MTLSize {
            width: runtime
                .packet_payload_copy_batched
                .thread_execution_width()
                .max(1),
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    drop(signpost);
    true
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_lossless_deinterleave(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
) -> Result<(), Error> {
    let input_byte_offset =
        u64::try_from(job.input_byte_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode input offset exceeds u64".to_string(),
        })?;
    let src_stride = u32::try_from(job.input_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode input pitch exceeds u32".to_string(),
    })?;
    let sample_offset = if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident encode bit depth must be 1-16".to_string(),
        });
    } else {
        1u32 << (u32::from(job.bit_depth) - 1)
    };
    let params = J2kLosslessDeinterleaveParams {
        src_width: job.input_width,
        src_height: job.input_height,
        src_stride,
        dst_width: job.output_width,
        dst_height: job.output_height,
        components: u32::from(job.component_count),
        bytes_per_sample: u32::from(job.bytes_per_sample),
        bit_depth: u32::from(job.bit_depth),
        sample_offset,
        signed_samples: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep deinterleave");
    encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_to_planes);
    encoder.set_buffer(0, Some(job.input), input_byte_offset);
    encoder.set_buffer(1, Some(plane0), 0);
    encoder.set_buffer(2, Some(plane1), 0);
    encoder.set_buffer(3, Some(plane2), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kLosslessDeinterleaveParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(5, Some(plane2), 0);
    dispatch_2d_pipeline(
        encoder,
        &runtime.lossless_deinterleave_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_lossless_deinterleave_rct_rgb8(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: J2kLosslessDevicePrepareJob<'_>,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
    status_buffer: &Buffer,
) -> Result<(), Error> {
    let input_byte_offset =
        u64::try_from(job.input_byte_offset).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode input offset exceeds u64".to_string(),
        })?;
    let src_stride = u32::try_from(job.input_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode input pitch exceeds u32".to_string(),
    })?;
    let sample_offset = if job.bit_depth == 0 || job.bit_depth > 16 {
        return Err(Error::MetalKernel {
            message: "J2K Metal resident encode bit depth must be 1-16".to_string(),
        });
    } else {
        1u32 << (u32::from(job.bit_depth) - 1)
    };
    let params = J2kLosslessDeinterleaveParams {
        src_width: job.input_width,
        src_height: job.input_height,
        src_stride,
        dst_width: job.output_width,
        dst_height: job.output_height,
        components: u32::from(job.component_count),
        bytes_per_sample: u32::from(job.bytes_per_sample),
        bit_depth: u32::from(job.bit_depth),
        sample_offset,
        signed_samples: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep deinterleave + RCT");
    encoder.set_compute_pipeline_state(&runtime.lossless_deinterleave_rct_rgb8_to_planes);
    encoder.set_buffer(0, Some(job.input), input_byte_offset);
    encoder.set_buffer(1, Some(plane0), 0);
    encoder.set_buffer(2, Some(plane1), 0);
    encoder.set_buffer(3, Some(plane2), 0);
    encoder.set_bytes(
        4,
        size_of::<J2kLosslessDeinterleaveParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(5, Some(status_buffer), 0);
    dispatch_2d_pipeline(
        encoder,
        &runtime.lossless_deinterleave_rct_rgb8_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    #[cfg(test)]
    test_counters::record_lossless_deinterleave_rct_fused_dispatch();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn lossless_deinterleave_rct_rgb8_supported(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> bool {
    job.component_count == 3 && job.bytes_per_sample == 1
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_rct_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane0: &Buffer,
    plane1: &Buffer,
    plane2: &Buffer,
    len: usize,
    status_buffer: &Buffer,
) -> Result<(), Error> {
    if len == 0 {
        return Ok(());
    }
    let params = J2kForwardRctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K Metal resident encode RCT length exceeds u32".to_string(),
        })?,
        _reserved0: 0,
        _reserved1: 0,
        _reserved2: 0,
    };
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep RCT");
    encoder.set_compute_pipeline_state(&runtime.forward_rct);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, Some(plane1), 0);
    encoder.set_buffer(2, Some(plane2), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kForwardRctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(status_buffer), 0);
    let width = runtime
        .forward_rct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
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
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt53_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> Buffer {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
        };

        if current_height >= 2 {
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    if active_is_input {
        input.to_owned()
    } else {
        scratch.to_owned()
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct ForwardDwt53ComponentsDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) plane_buffers: &'a [Buffer],
    pub(super) scratch_buffers: &'a [Buffer],
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_levels: u8,
    pub(super) component_count: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt53_components_on_buffers(
    dispatch: ForwardDwt53ComponentsDispatch<'_>,
) -> Vec<Buffer> {
    let ForwardDwt53ComponentsDispatch {
        runtime,
        command_buffer,
        plane_buffers,
        scratch_buffers,
        width,
        height,
        num_levels,
        component_count,
    } = dispatch;
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let component_count_u32 = component_count as u32;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53BatchedParams {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            component_count: component_count_u32,
        };

        if current_height >= 2 {
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_vertical_batched,
                command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_horizontal_batched,
                command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active_buffers = if active_is_input {
        plane_buffers
    } else {
        scratch_buffers
    };
    active_buffers[..component_count].to_vec()
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt53_on_buffers_split_profile(
    runtime: &MetalRuntime,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> (Buffer, Vec<CommandBuffer>, Vec<CommandBuffer>) {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53Params {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
        };

        if current_height >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 vertical",
            );
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 horizontal",
            );
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            command_buffer.commit();
            horizontal_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active = if active_is_input {
        input.to_owned()
    } else {
        scratch.to_owned()
    };
    (active, vertical_command_buffers, horizontal_command_buffers)
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_forward_dwt53_components_split_profile(
    runtime: &MetalRuntime,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
    width: u32,
    height: u32,
    num_levels: u8,
    component_count: usize,
) -> (Vec<Buffer>, Vec<CommandBuffer>, Vec<CommandBuffer>) {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();
    let component_count_u32 = component_count as u32;

    while levels_run < num_levels && (current_width >= 2 || current_height >= 2) {
        let low_width = current_width.div_ceil(2);
        let low_height = current_height.div_ceil(2);
        let params = J2kForwardDwt53BatchedParams {
            full_width: width,
            current_width,
            current_height,
            low_width,
            low_height,
            component_count: component_count_u32,
        };

        if current_height >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 vertical",
            );
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_vertical_batched,
                &command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            );
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 horizontal",
            );
            let (inputs, outputs) = if active_is_input {
                (plane_buffers, scratch_buffers)
            } else {
                (scratch_buffers, plane_buffers)
            };
            dispatch_forward_dwt53_batched_pass(
                &runtime.fdwt53_horizontal_batched,
                &command_buffer,
                inputs,
                outputs,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            );
            command_buffer.commit();
            horizontal_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    let active_buffers = if active_is_input {
        plane_buffers
    } else {
        scratch_buffers
    };
    (
        active_buffers[..component_count].to_vec(),
        vertical_command_buffers,
        horizontal_command_buffers,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_lossless_extract_coefficients(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: &[Buffer],
    coefficient_buffer: &Buffer,
    coefficient_jobs: &[J2kLosslessCoefficientJob],
    output_width: u32,
) -> Result<Buffer, Error> {
    let coefficient_job_buffer = copied_slice_buffer(&runtime.device, coefficient_jobs);
    let job_count = u32::try_from(coefficient_jobs.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal resident encode coefficient job count exceeds u32".to_string(),
    })?;
    let max_block_width = coefficient_jobs
        .iter()
        .map(|job| job.block_width)
        .max()
        .unwrap_or(1);
    let max_block_height = coefficient_jobs
        .iter()
        .map(|job| job.block_height)
        .max()
        .unwrap_or(1);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K coefficient prep extract");
    encoder.set_compute_pipeline_state(&runtime.lossless_extract_coefficients);
    encoder.set_buffer(0, planes.first().map(|buffer| &**buffer), 0);
    encoder.set_buffer(
        1,
        planes
            .get(1)
            .or_else(|| planes.first())
            .map(|buffer| &**buffer),
        0,
    );
    encoder.set_buffer(
        2,
        planes
            .get(2)
            .or_else(|| planes.first())
            .map(|buffer| &**buffer),
        0,
    );
    encoder.set_buffer(3, Some(coefficient_buffer), 0);
    encoder.set_buffer(4, Some(&coefficient_job_buffer), 0);
    encoder.set_bytes(5, size_of::<u32>() as u64, (&raw const job_count).cast());
    dispatch_3d_pipeline(
        encoder,
        &runtime.lossless_extract_coefficients,
        (max_block_width, max_block_height, job_count),
    );
    encoder.end_encoding();
    let _ = output_width;
    Ok(coefficient_job_buffer)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct J2kLosslessPrepareSizes {
    pub(super) plane_len: usize,
    pub(super) plane_bytes: usize,
    pub(super) coefficient_bytes: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn lossless_prepare_sizes(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> Result<J2kLosslessPrepareSizes, Error> {
    if job.component_count != 1 && job.component_count != 3 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports grayscale or RGB input",
        });
    }
    if job.bytes_per_sample != 1 && job.bytes_per_sample != 2 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal resident encode supports 8-bit or 16-bit samples",
        });
    }
    let plane_len = (job.output_width as usize)
        .checked_mul(job.output_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode plane size overflow".to_string(),
        })?;
    let plane_bytes =
        plane_len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident encode plane byte size overflow".to_string(),
            })?;
    let coefficient_bytes = job
        .coefficient_count
        .max(1)
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal resident encode coefficient size overflow".to_string(),
        })?;
    Ok(J2kLosslessPrepareSizes {
        plane_len,
        plane_bytes,
        coefficient_bytes,
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks(
    session: &crate::MetalBackendSession,
    job: J2kLosslessDevicePrepareJob<'_>,
    code_blocks: Vec<J2kLosslessDeviceCodeBlock>,
) -> Result<J2kPreparedLosslessDeviceCodeBlocks, Error> {
    let sizes = lossless_prepare_sizes(job)?;

    with_runtime_for_session(session, |runtime| {
        let mut plane_buffers = Vec::with_capacity(3);
        let mut scratch_buffers = Vec::with_capacity(usize::from(job.component_count));
        for _ in 0..3 {
            plane_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        for _ in 0..job.component_count {
            scratch_buffers.push(runtime.device.new_buffer(
                sizes.plane_bytes as u64,
                MTLResourceOptions::StorageModePrivate,
            ));
        }
        let coefficient_buffer = runtime.device.new_buffer(
            sizes.coefficient_bytes as u64,
            MTLResourceOptions::StorageModePrivate,
        );
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());
        let command_buffer = runtime.queue.new_command_buffer();

        if lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_lossless_deinterleave_rct_rgb8(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                &status_buffer,
            )?;
        } else {
            dispatch_lossless_deinterleave(
                runtime,
                command_buffer,
                job,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
            )?;
        }
        if job.component_count == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
            dispatch_forward_rct_on_buffers(
                runtime,
                command_buffer,
                &plane_buffers[0],
                &plane_buffers[1],
                &plane_buffers[2],
                sizes.plane_len,
                &status_buffer,
            )?;
        }

        let mut active_planes = Vec::with_capacity(usize::from(job.component_count));
        for component in 0..usize::from(job.component_count) {
            if job.num_decomposition_levels == 0 {
                active_planes.push(plane_buffers[component].clone());
            } else {
                active_planes.push(dispatch_forward_dwt53_on_buffers(
                    runtime,
                    command_buffer,
                    &plane_buffers[component],
                    &scratch_buffers[component],
                    job.output_width,
                    job.output_height,
                    job.num_decomposition_levels,
                ));
            }
        }
        while active_planes.len() < 3 {
            active_planes.push(active_planes[0].clone());
        }

        let coefficient_jobs = code_blocks
            .iter()
            .map(|block| J2kLosslessCoefficientJob {
                coefficient_offset: block.coefficient_offset,
                component: block.component,
                subband_x: block.subband_x,
                subband_y: block.subband_y,
                block_x: block.block_x,
                block_y: block.block_y,
                block_width: block.width,
                block_height: block.height,
                full_width: job.output_width,
            })
            .collect::<Vec<_>>();
        let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
            runtime,
            command_buffer,
            &active_planes,
            &coefficient_buffer,
            &coefficient_jobs,
            job.output_width,
        )?;

        command_buffer.commit();
        Ok(J2kPreparedLosslessDeviceCodeBlocks {
            coefficient_buffer,
            coefficient_byte_offset: 0,
            coefficient_byte_len: sizes.coefficient_bytes,
            coefficient_buffer_is_batch_shared: false,
            code_blocks,
            recyclable_private_buffers: Vec::new(),
            _prepare_command_buffer: command_buffer.to_owned(),
            _prepare_deinterleave_rct_command_buffer: None,
            _prepare_dwt53_command_buffer: None,
            _prepare_dwt53_vertical_command_buffers: Vec::new(),
            _prepare_dwt53_horizontal_command_buffers: Vec::new(),
            _prepare_coefficient_extract_command_buffer: None,
            _deinterleave_status_buffer: status_buffer,
            _plane_buffers: plane_buffers,
            _scratch_buffers: scratch_buffers,
            _coefficient_job_buffer: coefficient_job_buffer,
        })
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kLosslessDeviceBatchPrepareItem<'_>>,
) -> Result<Vec<J2kPreparedLosslessDeviceCodeBlocks>, Error> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut sizes = Vec::with_capacity(items.len());
    let mut coefficient_byte_offsets = Vec::with_capacity(items.len());
    let mut total_coefficient_bytes = 0usize;
    for item in &items {
        let item_sizes = lossless_prepare_sizes(item.job).map_err(|err| Error::MetalKernel {
            message: format!(
                "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                item.tile_index
            ),
        })?;
        coefficient_byte_offsets.push(total_coefficient_bytes);
        total_coefficient_bytes = total_coefficient_bytes
            .checked_add(item_sizes.coefficient_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident batch coefficient size overflow".to_string(),
            })?;
        sizes.push(item_sizes);
    }

    with_runtime_for_session(session, |runtime| {
        let mut shared_recyclable_private_buffers = Vec::new();
        let coefficient_buffer = take_recyclable_private_buffer(
            runtime,
            total_coefficient_bytes.max(1),
            &mut shared_recyclable_private_buffers,
        )?;
        let split_prepare_command_buffers = metal_profile_coefficient_prep_split_commands_enabled();
        let shared_command_buffer = if split_prepare_command_buffers {
            None
        } else {
            Some(runtime.queue.new_command_buffer().to_owned())
        };
        let mut prepared = Vec::with_capacity(items.len());

        for ((item, item_sizes), coefficient_byte_offset) in
            items.into_iter().zip(sizes).zip(coefficient_byte_offsets)
        {
            let job = item.job;
            let mut recyclable_private_buffers = Vec::new();
            if !shared_recyclable_private_buffers.is_empty() {
                recyclable_private_buffers.append(&mut shared_recyclable_private_buffers);
            }
            let mut plane_buffers = Vec::with_capacity(3);
            let mut scratch_buffers = Vec::with_capacity(usize::from(job.component_count));
            for _ in 0..3 {
                plane_buffers.push(take_recyclable_private_buffer(
                    runtime,
                    item_sizes.plane_bytes,
                    &mut recyclable_private_buffers,
                )?);
            }
            for _ in 0..job.component_count {
                scratch_buffers.push(take_recyclable_private_buffer(
                    runtime,
                    item_sizes.plane_bytes,
                    &mut recyclable_private_buffers,
                )?);
            }

            let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

            let mut prepare_deinterleave_rct_command_buffer = None;
            let prepare_dwt53_command_buffer = None;
            let mut prepare_dwt53_vertical_command_buffers = Vec::new();
            let mut prepare_dwt53_horizontal_command_buffers = Vec::new();
            let mut prepare_coefficient_extract_command_buffer = None;
            let deinterleave_command_buffer = if split_prepare_command_buffers {
                new_resident_encode_command_buffer(runtime, "j2k coefficient prep deinterleave rct")
            } else {
                shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists")
                    .clone()
            };
            if lossless_deinterleave_rct_rgb8_supported(job) {
                dispatch_lossless_deinterleave_rct_rgb8(
                    runtime,
                    &deinterleave_command_buffer,
                    job,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                    &status_buffer,
                )
            } else {
                dispatch_lossless_deinterleave(
                    runtime,
                    &deinterleave_command_buffer,
                    job,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                )
            }
            .map_err(|err| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                    item.tile_index
                ),
            })?;
            if job.component_count == 3 && !lossless_deinterleave_rct_rgb8_supported(job) {
                dispatch_forward_rct_on_buffers(
                    runtime,
                    &deinterleave_command_buffer,
                    &plane_buffers[0],
                    &plane_buffers[1],
                    &plane_buffers[2],
                    item_sizes.plane_len,
                    &status_buffer,
                )
                .map_err(|err| Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                        item.tile_index
                    ),
                })?;
            }
            if split_prepare_command_buffers {
                deinterleave_command_buffer.commit();
                prepare_deinterleave_rct_command_buffer = Some(deinterleave_command_buffer);
            }

            let mut active_planes = Vec::with_capacity(usize::from(job.component_count));
            if job.num_decomposition_levels == 0 {
                active_planes.extend(
                    plane_buffers
                        .iter()
                        .take(usize::from(job.component_count))
                        .cloned(),
                );
            } else if split_prepare_command_buffers {
                let component_count = usize::from(job.component_count);
                if component_count > 1 {
                    let (
                        mut component_active_planes,
                        mut vertical_command_buffers,
                        mut horizontal_command_buffers,
                    ) = dispatch_forward_dwt53_components_split_profile(
                        runtime,
                        &plane_buffers,
                        &scratch_buffers,
                        job.output_width,
                        job.output_height,
                        job.num_decomposition_levels,
                        component_count,
                    );
                    active_planes.append(&mut component_active_planes);
                    prepare_dwt53_vertical_command_buffers.append(&mut vertical_command_buffers);
                    prepare_dwt53_horizontal_command_buffers
                        .append(&mut horizontal_command_buffers);
                } else {
                    for component in 0..component_count {
                        let (
                            active_plane,
                            mut vertical_command_buffers,
                            mut horizontal_command_buffers,
                        ) = dispatch_forward_dwt53_on_buffers_split_profile(
                            runtime,
                            &plane_buffers[component],
                            &scratch_buffers[component],
                            job.output_width,
                            job.output_height,
                            job.num_decomposition_levels,
                        );
                        active_planes.push(active_plane);
                        prepare_dwt53_vertical_command_buffers
                            .append(&mut vertical_command_buffers);
                        prepare_dwt53_horizontal_command_buffers
                            .append(&mut horizontal_command_buffers);
                    }
                }
            } else {
                let dwt_command_buffer_ref = shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists");
                let component_count = usize::from(job.component_count);
                if component_count > 1 {
                    active_planes = dispatch_forward_dwt53_components_on_buffers(
                        ForwardDwt53ComponentsDispatch {
                            runtime,
                            command_buffer: dwt_command_buffer_ref,
                            plane_buffers: &plane_buffers,
                            scratch_buffers: &scratch_buffers,
                            width: job.output_width,
                            height: job.output_height,
                            num_levels: job.num_decomposition_levels,
                            component_count,
                        },
                    );
                } else {
                    for component in 0..component_count {
                        active_planes.push(dispatch_forward_dwt53_on_buffers(
                            runtime,
                            dwt_command_buffer_ref,
                            &plane_buffers[component],
                            &scratch_buffers[component],
                            job.output_width,
                            job.output_height,
                            job.num_decomposition_levels,
                        ));
                    }
                }
            }
            while active_planes.len() < 3 {
                active_planes.push(active_planes[0].clone());
            }

            let coefficient_word_offset = coefficient_byte_offset
                .checked_div(size_of::<i32>())
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal resident batch coefficient offset division failed"
                        .to_string(),
                })?;
            let coefficient_word_offset_u32 =
                u32::try_from(coefficient_word_offset).map_err(|_| Error::MetalKernel {
                    message: format!(
                        "J2K Metal resident batch coefficient offset exceeds u32 at tile {}",
                        item.tile_index
                    ),
                })?;
            let coefficient_jobs = item
                .code_blocks
                .iter()
                .map(|block| {
                    let coefficient_offset = block
                        .coefficient_offset
                        .checked_add(coefficient_word_offset_u32)
                        .ok_or_else(|| Error::MetalKernel {
                            message: format!(
                                "J2K Metal resident batch coefficient offset overflow at tile {}",
                                item.tile_index
                            ),
                        })?;
                    Ok(J2kLosslessCoefficientJob {
                        coefficient_offset,
                        component: block.component,
                        subband_x: block.subband_x,
                        subband_y: block.subband_y,
                        block_x: block.block_x,
                        block_y: block.block_y,
                        block_width: block.width,
                        block_height: block.height,
                        full_width: job.output_width,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let extract_command_buffer = if split_prepare_command_buffers {
                new_resident_encode_command_buffer(runtime, "j2k coefficient prep extract")
            } else {
                shared_command_buffer
                    .as_ref()
                    .expect("shared coefficient prep command buffer exists")
                    .clone()
            };
            let coefficient_job_buffer = dispatch_lossless_extract_coefficients(
                runtime,
                &extract_command_buffer,
                &active_planes,
                &coefficient_buffer,
                &coefficient_jobs,
                job.output_width,
            )
            .map_err(|err| Error::MetalKernel {
                message: format!(
                    "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                    item.tile_index
                ),
            })?;
            let prepare_command_buffer = extract_command_buffer.clone();
            if split_prepare_command_buffers {
                extract_command_buffer.commit();
                prepare_coefficient_extract_command_buffer = Some(extract_command_buffer);
            }

            prepared.push(J2kPreparedLosslessDeviceCodeBlocks {
                coefficient_buffer: coefficient_buffer.clone(),
                coefficient_byte_offset,
                coefficient_byte_len: item_sizes.coefficient_bytes,
                coefficient_buffer_is_batch_shared: true,
                code_blocks: item.code_blocks,
                recyclable_private_buffers,
                _prepare_command_buffer: prepare_command_buffer,
                _prepare_deinterleave_rct_command_buffer: prepare_deinterleave_rct_command_buffer,
                _prepare_dwt53_command_buffer: prepare_dwt53_command_buffer,
                _prepare_dwt53_vertical_command_buffers: prepare_dwt53_vertical_command_buffers,
                _prepare_dwt53_horizontal_command_buffers: prepare_dwt53_horizontal_command_buffers,
                _prepare_coefficient_extract_command_buffer:
                    prepare_coefficient_extract_command_buffer,
                _deinterleave_status_buffer: status_buffer,
                _plane_buffers: plane_buffers,
                _scratch_buffers: scratch_buffers,
                _coefficient_job_buffer: coefficient_job_buffer,
            });
        }

        if let Some(command_buffer) = shared_command_buffer {
            command_buffer.commit();
        }
        Ok(prepared)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_rct(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal forward RCT plane lengths must match".to_string(),
            });
        }

        let params = J2kForwardRctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal forward RCT plane length exceeds u32".to_string(),
            })?,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_rct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardRctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_rct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
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
        commit_and_wait_metal(command_buffer)?;

        let status = checked_buffer_read::<J2kMctStatus>(&status_buffer, "forward RCT status")?;
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_ict(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::UnsupportedMetalRequest {
                reason: "J2K Metal forward ICT plane lengths must match",
            });
        }

        let params = J2kForwardIctParams {
            _len: u32::try_from(len).map_err(|_| Error::UnsupportedMetalRequest {
                reason: "J2K Metal forward ICT plane length exceeds u32",
            })?,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_ict);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardIctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_ict
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
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
        commit_and_wait_metal(command_buffer)?;

        let status = checked_buffer_read::<J2kMctStatus>(&status_buffer, "forward ICT status")?;
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(super) fn validate_encode_quantize_subband_job(
    job: J2kQuantizeSubbandJob<'_>,
) -> Result<(), Error> {
    if job.step_exponent > 31 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step exponents <= 31",
        });
    }
    if job.step_mantissa > 2047 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step mantissas <= 2047",
        });
    }
    if job.range_bits == 0 || job.range_bits > 31 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports range bits 1-31",
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_quantize_subband(job: J2kQuantizeSubbandJob<'_>) -> Result<Vec<i32>, Error> {
    validate_encode_quantize_subband_job(job)?;
    let len = job.coefficients.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let len_u32 = u32::try_from(len).map_err(|_| Error::UnsupportedMetalRequest {
        reason: "J2K Metal encode quantize_subband coefficient count exceeds u32",
    })?;
    let output_bytes = len
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode quantize_subband output length overflow".to_string(),
        })?;

    with_runtime(|runtime| {
        let input_buffer = copied_slice_buffer(&runtime.device, job.coefficients);
        let output_buffer = runtime
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params = J2kQuantizeSubbandParams {
            _len: len_u32,
            _step_exponent: u32::from(job.step_exponent),
            _step_mantissa: u32::from(job.step_mantissa),
            _range_bits: u32::from(job.range_bits),
            _reversible: u32::from(job.reversible),
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k encode-stage quantize_subband");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K encode-stage quantize_subband");
        encoder.set_compute_pipeline_state(&runtime.quantize_subband);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kQuantizeSubbandParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_1d_pipeline(encoder, &runtime.quantize_subband, u64::from(len_u32));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        let coefficients = checked_buffer_slice::<i32>(&output_buffer, len, "quantized subband")?;
        Ok(coefficients.to_vec())
    })
}
