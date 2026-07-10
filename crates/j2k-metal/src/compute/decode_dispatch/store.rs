// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    borrow_slice_buffer, checked_metal_buffer_len_u64, checked_metal_surface_len,
    commit_and_wait_metal, dispatch_2d_pipeline, dispatch_3d_pipeline, hybrid_stage_signpost,
    label_compute_encoder, size_of, with_runtime, wrap_f32_output_buffer, Buffer, CommandBufferRef,
    ComputeCommandEncoderRef, Error, J2kGrayStoreParams, J2kRepeatedGrayStoreParams,
    J2kRepeatedStoreParams, J2kStoreComponentJob, J2kStoreParams, MTLResourceOptions, MTLSize,
    MetalRuntime, PixelFormat, Surface, SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
};

#[cfg(target_os = "macos")]
pub(crate) fn decode_store_component_and_capture(
    job: J2kStoreComponentJob<'_>,
) -> Result<Buffer, Error> {
    let J2kStoreComponentJob {
        input,
        input_width,
        source_x,
        source_y,
        copy_width,
        copy_height,
        output,
        output_width,
        output_x,
        output_y,
        addend,
    } = job;
    with_runtime(|runtime| {
        if copy_width == 0 || copy_height == 0 {
            return Ok(wrap_f32_output_buffer(&runtime.device, output));
        }

        let required_input_height =
            source_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store source height overflow".to_string(),
                })?;
        let required_output_height =
            output_y
                .checked_add(copy_height)
                .ok_or_else(|| Error::MetalKernel {
                    message: "J2K Metal store destination height overflow".to_string(),
                })?;
        if source_x
            .checked_add(copy_width)
            .is_none_or(|end| end > input_width)
            || output_x
                .checked_add(copy_width)
                .is_none_or(|end| end > output_width)
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store copy rectangle exceeds row bounds".to_string(),
            });
        }
        if input.len()
            < input_width as usize
                * usize::try_from(required_input_height).map_err(|_| Error::MetalKernel {
                    message: "J2K Metal store source height exceeds usize".to_string(),
                })?
            || output.len()
                < output_width as usize
                    * usize::try_from(required_output_height).map_err(|_| Error::MetalKernel {
                        message: "J2K Metal store destination height exceeds usize".to_string(),
                    })?
        {
            return Err(Error::MetalKernel {
                message: "J2K Metal store buffers are smaller than required".to_string(),
            });
        }

        let params = J2kStoreParams {
            input_width,
            source_x,
            source_y,
            copy_width,
            copy_height,
            output_width,
            output_x,
            output_y,
            addend,
        };
        let input_buffer = borrow_slice_buffer(&runtime.device, input);
        let output_buffer = wrap_f32_output_buffer(&runtime.device, output);
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.store_component);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kStoreParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(encoder, &runtime.store_component, (copy_width, copy_height));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;
        Ok(output_buffer)
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_store_component_buffer_in_command_buffer_with_offsets(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid component store");
    dispatch_store_component_buffer_in_encoder_with_offsets(
        runtime,
        encoder,
        input,
        input_offset_bytes,
        output,
        output_offset_bytes,
        params,
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_store_component_buffer_in_encoder_with_offsets(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    output_offset_bytes: usize,
    params: J2kStoreParams,
) {
    encoder.set_compute_pipeline_state(&runtime.store_component);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), output_offset_bytes as u64);
    encoder.set_bytes(
        2,
        size_of::<J2kStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.store_component,
        (params.copy_width, params.copy_height),
    );
}

pub(in crate::compute) fn dispatch_store_component_repeated_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    input_offset_bytes: usize,
    output: &Buffer,
    params: J2kRepeatedStoreParams,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated component store");
    encoder.set_compute_pipeline_state(&runtime.store_component_repeated);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(output), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.store_component_repeated,
        (params.copy_width, params.copy_height, params.batch_count),
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn repeated_gray_store_is_contiguous_full_surface(
    params: J2kRepeatedGrayStoreParams,
) -> bool {
    params.source_x == 0
        && params.source_y == 0
        && params.output_x == 0
        && params.output_y == 0
        && params.copy_width == params.input_width
        && params.copy_height == params.input_height
        && params.copy_width == params.output_width
        && params.copy_height == params.output_height
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_repeated_gray_store_to_surfaces_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    input: &Buffer,
    params: J2kRepeatedGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
    count: usize,
) -> Result<Vec<Surface>, Error> {
    let (_pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        "J2K Metal repeated grayscale fused store size overflow",
    )?;
    let total_bytes = surface_bytes
        .checked_mul(count)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal repeated grayscale fused store total size overflow".to_string(),
        })?;
    let output_len = checked_metal_buffer_len_u64(
        total_bytes,
        "J2K Metal repeated grayscale fused store output size exceeds u64",
    )?;
    let out_buffer = runtime
        .device
        .new_buffer(output_len, MTLResourceOptions::StorageModeShared);
    let contiguous_full_surface = repeated_gray_store_is_contiguous_full_surface(params);
    let pipeline = match (fmt, contiguous_full_surface) {
        (PixelFormat::Gray8, true) => &runtime.store_component_repeated_gray_u8_contiguous,
        (PixelFormat::Gray8, false) => &runtime.store_component_repeated_gray_u8,
        (PixelFormat::Gray16, true) => &runtime.store_component_repeated_gray_u16_contiguous,
        (PixelFormat::Gray16, false) => &runtime.store_component_repeated_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!(
                    "J2K Metal repeated grayscale fused store does not support {fmt:?}"
                ),
            })
        }
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), 0);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kRepeatedGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    if contiguous_full_surface {
        let total_samples = u64::from(params.input_width)
            * u64::from(params.input_height)
            * u64::from(params.batch_count);
        encoder.dispatch_threads(
            MTLSize {
                width: total_samples,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: max_threads,
                height: 1,
                depth: 1,
            },
        );
    } else {
        dispatch_3d_pipeline(
            encoder,
            pipeline,
            (params.copy_width, params.copy_height, params.batch_count),
        );
    }
    encoder.end_encoding();

    let mut surfaces = Vec::with_capacity(count);
    for instance_idx in 0..count {
        surfaces.push(Surface::from_metal_buffer_with_offset(
            out_buffer.clone(),
            dims,
            fmt,
            instance_idx * surface_bytes,
        ));
    }
    Ok(surfaces)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_gray_store_to_surface_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    input: &Buffer,
    input_offset_bytes: usize,
    params: J2kGrayStoreParams,
    dims: (u32, u32),
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    let (_pitch_bytes, surface_bytes) = checked_metal_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        "J2K Metal grayscale fused store size overflow",
    )?;
    let out_buffer = runtime.device.new_buffer(
        checked_metal_buffer_len_u64(
            surface_bytes,
            "J2K Metal grayscale fused store output size exceeds u64",
        )?,
        MTLResourceOptions::StorageModeShared,
    );
    let pipeline = match fmt {
        PixelFormat::Gray8 => &runtime.store_component_gray_u8,
        PixelFormat::Gray16 => &runtime.store_component_gray_u16,
        _ => {
            return Err(Error::MetalKernel {
                message: format!("J2K Metal grayscale fused store does not support {fmt:?}"),
            })
        }
    };

    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input), input_offset_bytes as u64);
    encoder.set_buffer(1, Some(&out_buffer), 0);
    encoder.set_bytes(
        2,
        size_of::<J2kGrayStoreParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, pipeline, (params.copy_width, params.copy_height));

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}
