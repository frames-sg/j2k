// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use super::test_counters;
use super::{
    active_forward_dwt53_buffers, copied_slice_buffer, dispatch_2d_pipeline, dispatch_3d_pipeline,
    dispatch_forward_dwt53_batched_pass, dispatch_forward_dwt53_pass, hybrid_stage_signpost,
    label_compute_encoder, new_compute_command_encoder, new_resident_encode_command_buffer,
    size_of, Buffer, CommandBuffer, CommandBufferRef, Error, J2kBatchedPacketPayloadCopyDispatch,
    J2kForwardDwt53BatchedParams, J2kForwardDwt53Params, J2kForwardRctParams,
    J2kLosslessCoefficientJob, J2kLosslessDeinterleaveParams, J2kLosslessDevicePrepareJob,
    J2kPacketPayloadCopyParams, MTLSize, MetalRuntime, PACKET_PAYLOAD_COPY_BYTES_PER_STRIPE,
    PACKET_PAYLOAD_COPY_STRIPES_PER_JOB,
};

pub(in crate::compute) struct ForwardDwt53SplitProfile<T> {
    pub(in crate::compute) active: T,
    pub(in crate::compute) vertical_command_buffers: Vec<CommandBuffer>,
    pub(in crate::compute) horizontal_command_buffers: Vec<CommandBuffer>,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_batched_packet_payload_copy(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    dispatch: J2kBatchedPacketPayloadCopyDispatch<'_>,
) -> Result<bool, Error> {
    if dispatch.tile_count == 0 || dispatch.max_payload_copy_jobs_per_tile == 0 {
        return Ok(false);
    }

    let signpost = hybrid_stage_signpost(dispatch.signpost_name);
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, dispatch.label);
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
    Ok(true)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_lossless_deinterleave(
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K coefficient prep deinterleave");
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
        &encoder,
        &runtime.lossless_deinterleave_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_lossless_deinterleave_rct_rgb8(
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K coefficient prep deinterleave + RCT");
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
        &encoder,
        &runtime.lossless_deinterleave_rct_rgb8_to_planes,
        (job.output_width, job.output_height),
    );
    encoder.end_encoding();
    #[cfg(test)]
    test_counters::record_lossless_deinterleave_rct_fused_dispatch();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn lossless_deinterleave_rct_rgb8_supported(
    job: J2kLosslessDevicePrepareJob<'_>,
) -> bool {
    job.component_count == 3 && job.bytes_per_sample == 1
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_forward_rct_on_buffers(
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K coefficient prep RCT");
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
pub(in crate::compute) fn dispatch_forward_dwt53_on_buffers(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<Buffer, Error> {
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
            )?;
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
            )?;
            active_is_input = !active_is_input;
        }

        current_width = low_width;
        current_height = low_height;
        levels_run = levels_run.saturating_add(1);
    }

    if active_is_input {
        Ok(input.to_owned())
    } else {
        Ok(scratch.to_owned())
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct ForwardDwt53ComponentsDispatch<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) plane_buffers: &'a [Buffer],
    pub(in crate::compute) scratch_buffers: &'a [Buffer],
    pub(in crate::compute) width: u32,
    pub(in crate::compute) height: u32,
    pub(in crate::compute) num_levels: u8,
    pub(in crate::compute) component_count: usize,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_forward_dwt53_components_on_buffers(
    dispatch: ForwardDwt53ComponentsDispatch<'_>,
) -> Result<Vec<Buffer>, Error> {
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
    let component_count_u32 = u32::try_from(component_count).map_err(|_| Error::MetalKernel {
        message: "JPEG 2000 component count exceeds u32".to_string(),
    })?;

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
            )?;
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
            )?;
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
    Ok(active_buffers[..component_count].to_vec())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_forward_dwt53_on_buffers_split_profile(
    runtime: &MetalRuntime,
    input: &Buffer,
    scratch: &Buffer,
    width: u32,
    height: u32,
    num_levels: u8,
) -> Result<ForwardDwt53SplitProfile<Buffer>, Error> {
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
            )?;
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_vertical,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 vertical",
            )?;
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 horizontal",
            )?;
            let (src, dst) = active_forward_dwt53_buffers(input, scratch, active_is_input);
            dispatch_forward_dwt53_pass(
                &runtime.fdwt53_horizontal,
                &command_buffer,
                src,
                dst,
                params,
                "J2K coefficient prep DWT 5/3 horizontal",
            )?;
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
    Ok(ForwardDwt53SplitProfile {
        active,
        vertical_command_buffers,
        horizontal_command_buffers,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_forward_dwt53_components_split_profile(
    runtime: &MetalRuntime,
    plane_buffers: &[Buffer],
    scratch_buffers: &[Buffer],
    width: u32,
    height: u32,
    num_levels: u8,
    component_count: usize,
) -> Result<ForwardDwt53SplitProfile<Vec<Buffer>>, Error> {
    let mut current_width = width;
    let mut current_height = height;
    let mut levels_run = 0u8;
    let mut active_is_input = true;
    let mut vertical_command_buffers = Vec::new();
    let mut horizontal_command_buffers = Vec::new();
    let component_count_u32 = u32::try_from(component_count).map_err(|_| Error::MetalKernel {
        message: "JPEG 2000 component count exceeds u32".to_string(),
    })?;

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
            )?;
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
            )?;
            command_buffer.commit();
            vertical_command_buffers.push(command_buffer);
            active_is_input = !active_is_input;
        }
        if current_width >= 2 {
            let command_buffer = new_resident_encode_command_buffer(
                runtime,
                "j2k coefficient prep DWT 5/3 horizontal",
            )?;
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
            )?;
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
    Ok(ForwardDwt53SplitProfile {
        active: active_buffers[..component_count].to_vec(),
        vertical_command_buffers,
        horizontal_command_buffers,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_lossless_extract_coefficients(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: &[Buffer],
    coefficient_buffer: &Buffer,
    coefficient_jobs: &[J2kLosslessCoefficientJob],
    output_width: u32,
) -> Result<Buffer, Error> {
    let coefficient_job_buffer = copied_slice_buffer(&runtime.device, coefficient_jobs)?;
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
    let encoder = new_compute_command_encoder(command_buffer)?;
    label_compute_encoder(&encoder, "J2K coefficient prep extract");
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
        &encoder,
        &runtime.lossless_extract_coefficients,
        (max_block_width, max_block_height, job_count),
    );
    encoder.end_encoding();
    let _ = output_width;
    Ok(coefficient_job_buffer)
}
