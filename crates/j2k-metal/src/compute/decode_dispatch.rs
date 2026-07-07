// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    borrow_slice_buffer, checked_buffer_read, checked_buffer_slice, checked_metal_buffer_len_u64,
    checked_metal_surface_len, commit_and_wait_metal, copied_slice_buffer,
    decode_classic_status_error, decode_idwt_status_error, decode_mct_status_error,
    dispatch_1d_pipeline, dispatch_2d_pipeline, dispatch_3d_pipeline,
    dispatch_ht_cleanup_batched_in_command_buffer, dispatch_ht_cleanup_batched_in_encoder,
    dispatch_ht_cleanup_repeated_batched_in_command_buffer, dispatch_single_thread,
    hybrid_stage_signpost, j2k_u32_param, label_compute_encoder, owned_slice_buffer,
    prepared_ht_buffer, size_of, take_classic_coefficients_scratch_buffer,
    take_classic_states_scratch_buffer, with_runtime, wrap_f32_output_buffer, zeroed_shared_buffer,
    Buffer, CommandBufferRef, ComputeCommandEncoderRef, DirectIdwtCommandBuffers,
    DirectScratchBuffer, DirectStatusCheck, Error, HtCodeBlockDecodeJob, HtRepeatedCleanupDispatch,
    J2kClassicCleanupBatchJob, J2kClassicRepeatedBatchParams, J2kClassicSegment, J2kClassicStatus,
    J2kGrayStoreParams, J2kHtCleanupBatchJob, J2kIdwtSingleDecompositionParams, J2kIdwtStatus,
    J2kInverseMctJob, J2kInverseMctParams, J2kMctStatus, J2kRepeatedGrayStoreParams,
    J2kRepeatedIdwtSingleDecompositionParams, J2kRepeatedStoreParams,
    J2kSingleDecompositionIdwtJob, J2kStoreComponentJob, J2kStoreParams, J2kWaveletTransform,
    MTLResourceOptions, MTLSize, MetalRuntime, PixelFormat, PreparedClassicSubBand,
    PreparedClassicSubBandGroup, PreparedHtSubBand, PreparedHtSubBandGroup, Surface,
    J2K_CLASSIC_MAX_HEIGHT, J2K_CLASSIC_MAX_WIDTH, J2K_CLASSIC_STATUS_OK, J2K_IDWT_STATUS_OK,
    J2K_MCT_STATUS_OK, SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE,
    SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE, SIGNPOST_DECODE_HYBRID_STORE_COMMAND_ENCODE,
};

#[cfg(target_os = "macos")]
pub(crate) fn decode_inverse_mct(job: J2kInverseMctJob<'_>) -> Result<Vec<Buffer>, Error> {
    let J2kInverseMctJob {
        transform,
        plane0,
        plane1,
        plane2,
        addend0,
        addend1,
        addend2,
    } = job;
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal inverse MCT plane lengths must match".to_string(),
            });
        }

        let transform = match transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        };
        let params = J2kInverseMctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal inverse MCT plane length exceeds u32".to_string(),
            })?,
            _transform: transform,
            _addend0: addend0,
            _addend1: addend1,
            _addend2: addend2,
        };
        let plane0_buffer = copied_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = copied_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = copied_slice_buffer(&runtime.device, plane2);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kInverseMctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .inverse_mct
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

        let status = checked_buffer_read::<J2kMctStatus>(&status_buffer, "inverse MCT status")?;
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        let plane0_host = checked_buffer_slice::<f32>(&plane0_buffer, len, "inverse MCT plane 0")?;
        let plane1_host = checked_buffer_slice::<f32>(&plane1_buffer, len, "inverse MCT plane 1")?;
        let plane2_host = checked_buffer_slice::<f32>(&plane2_buffer, len, "inverse MCT plane 2")?;
        for (dst, sample) in plane0.iter_mut().zip(plane0_host.iter().copied()) {
            *dst = sample - addend0;
        }
        for (dst, sample) in plane1.iter_mut().zip(plane1_host.iter().copied()) {
            *dst = sample - addend1;
        }
        for (dst, sample) in plane2.iter_mut().zip(plane2_host.iter().copied()) {
            *dst = sample - addend2;
        }
        Ok(vec![plane0_buffer, plane1_buffer, plane2_buffer])
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_inverse_mct_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    len: usize,
    transform: J2kWaveletTransform,
    addends: [f32; 3],
) -> Result<DirectStatusCheck, Error> {
    if len == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color MCT cannot run on an empty plane".to_string(),
        });
    }

    let transform = match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    };
    let params = J2kInverseMctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color MCT plane length exceeds u32".to_string(),
        })?,
        _transform: transform,
        _addend0: addends[0],
        _addend1: addends[1],
        _addend2: addends[2],
    };
    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.inverse_mct);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kInverseMctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .inverse_mct
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

    Ok(DirectStatusCheck::Mct(status_buffer))
}

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
pub(super) fn dispatch_store_component_buffer_in_command_buffer_with_offsets(
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
pub(super) fn dispatch_store_component_buffer_in_encoder_with_offsets(
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

pub(super) fn dispatch_store_component_repeated_in_command_buffer(
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
pub(super) fn repeated_gray_store_is_contiguous_full_surface(
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
pub(super) fn encode_repeated_gray_store_to_surfaces_in_command_buffer(
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
pub(super) fn encode_gray_store_to_surface_in_encoder(
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

#[cfg(target_os = "macos")]
pub(crate) fn decode_reversible53_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);

        let command_buffer = runtime.queue.new_command_buffer();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.idwt_interleave,
            (params.width, params.height),
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let horizontal_width = runtime
            .idwt_reversible53_horizontal
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.height),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: horizontal_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
        encoder.set_buffer(0, Some(&decoded), 0);
        encoder.set_bytes(
            1,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        let vertical_width = runtime
            .idwt_reversible53_vertical
            .thread_execution_width()
            .max(1);
        encoder.dispatch_threads(
            MTLSize {
                width: u64::from(params.width),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: vertical_width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;
        Ok(())
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct IdwtSubBandBuffers<'a> {
    pub(super) ll: &'a Buffer,
    pub(super) ll_offset: usize,
    pub(super) hl: &'a Buffer,
    pub(super) hl_offset: usize,
    pub(super) lh: &'a Buffer,
    pub(super) lh_offset: usize,
    pub(super) hh: &'a Buffer,
    pub(super) hh_offset: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct SingleIdwtDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) sub_bands: IdwtSubBandBuffers<'a>,
    pub(super) params: J2kIdwtSingleDecompositionParams,
    pub(super) decoded: &'a Buffer,
    pub(super) decoded_offset: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct RepeatedIdwtDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) sub_bands: IdwtSubBandBuffers<'a>,
    pub(super) params: J2kRepeatedIdwtSingleDecompositionParams,
    pub(super) decoded: &'a Buffer,
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_reversible53_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid reversible53 IDWT");
    dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(encoder, dispatch);
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_reversible53_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) {
    let SingleIdwtDispatch {
        runtime,
        sub_bands,
        params,
        decoded,
        decoded_offset,
    } = dispatch;
    let IdwtSubBandBuffers {
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
    } = sub_bands;
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(
        encoder,
        &runtime.idwt_interleave,
        (params.width, params.height),
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );

    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical);
    encoder.set_buffer(0, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        1,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_reversible53_repeated_buffers_in_command_buffer_with_offsets(
    command_buffers: DirectIdwtCommandBuffers<'_>,
    dispatch: RepeatedIdwtDispatch<'_>,
) {
    let RepeatedIdwtDispatch {
        runtime,
        sub_bands,
        params,
        decoded,
    } = dispatch;
    let IdwtSubBandBuffers {
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
    } = sub_bands;
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let encoder = command_buffers.interleave.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT interleave");
    encoder.set_compute_pipeline_state(&runtime.idwt_interleave_batched);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), 0);
    encoder.set_bytes(
        5,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_3d_pipeline(
        encoder,
        &runtime.idwt_interleave_batched,
        (params.width, params.height, params.batch_count),
    );
    encoder.end_encoding();

    let encoder = command_buffers.horizontal.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT horizontal");
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_horizontal_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let horizontal_width = runtime
        .idwt_reversible53_horizontal_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.height),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: horizontal_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    let encoder = command_buffers.vertical.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid repeated IDWT vertical");
    encoder.set_compute_pipeline_state(&runtime.idwt_reversible53_vertical_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_bytes(
        1,
        size_of::<J2kRepeatedIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    let vertical_width = runtime
        .idwt_reversible53_vertical_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(params.width),
            height: u64::from(params.batch_count),
            depth: 1,
        },
        MTLSize {
            width: vertical_width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_irreversible97_single_decomposition_idwt(
    job: J2kSingleDecompositionIdwtJob<'_>,
    output: &mut [f32],
) -> Result<(), Error> {
    with_runtime(|runtime| {
        let required_len = job.rect.width() as usize * job.rect.height() as usize;
        if output.len() < required_len {
            return Err(Error::MetalKernel {
                message: "J2K Metal IDWT output slice is too small".to_string(),
            });
        }

        let params = J2kIdwtSingleDecompositionParams {
            x0: job.rect.x0,
            y0: job.rect.y0,
            output_x: 0,
            output_y: 0,
            width: job.rect.width(),
            height: job.rect.height(),
            ll_x: 0,
            ll_y: 0,
            ll_width: job.ll.rect.width(),
            ll_height: job.ll.rect.height(),
            hl_x: 0,
            hl_y: 0,
            hl_width: job.hl.rect.width(),
            hl_height: job.hl.rect.height(),
            lh_x: 0,
            lh_y: 0,
            lh_width: job.lh.rect.width(),
            lh_height: job.lh.rect.height(),
            hh_x: 0,
            hh_y: 0,
            hh_width: job.hh.rect.width(),
            hh_height: job.hh.rect.height(),
        };

        let ll = borrow_slice_buffer(&runtime.device, job.ll.coefficients);
        let hl = borrow_slice_buffer(&runtime.device, job.hl.coefficients);
        let lh = borrow_slice_buffer(&runtime.device, job.lh.coefficients);
        let hh = borrow_slice_buffer(&runtime.device, job.hh.coefficients);
        let decoded = wrap_f32_output_buffer(&runtime.device, output);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kIdwtStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
        encoder.set_buffer(0, Some(&ll), 0);
        encoder.set_buffer(1, Some(&hl), 0);
        encoder.set_buffer(2, Some(&lh), 0);
        encoder.set_buffer(3, Some(&hh), 0);
        encoder.set_buffer(4, Some(&decoded), 0);
        encoder.set_bytes(
            5,
            size_of::<J2kIdwtSingleDecompositionParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(6, Some(&status_buffer), 0);
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        let status = checked_buffer_read::<J2kIdwtStatus>(&status_buffer, "IDWT status")?;
        if status.code != J2K_IDWT_STATUS_OK {
            return Err(decode_idwt_status_error(status));
        }
        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_irreversible97_single_decomposition_buffers_in_command_buffer_with_offsets(
    command_buffer: &CommandBufferRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> DirectStatusCheck {
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_IDWT_COMMAND_ENCODE);
    let status_buffer = zeroed_shared_buffer(&dispatch.runtime.device, size_of::<J2kIdwtStatus>());

    let encoder = command_buffer.new_compute_command_encoder();
    label_compute_encoder(encoder, "J2K decode hybrid irreversible97 IDWT");
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        encoder,
        dispatch,
        &status_buffer,
    );
    encoder.end_encoding();

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_offsets(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
) -> DirectStatusCheck {
    let status_buffer = zeroed_shared_buffer(&dispatch.runtime.device, size_of::<J2kIdwtStatus>());
    dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
        encoder,
        dispatch,
        &status_buffer,
    );

    DirectStatusCheck::Idwt(status_buffer)
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_irreversible97_single_decomposition_buffers_in_encoder_with_status(
    encoder: &ComputeCommandEncoderRef,
    dispatch: SingleIdwtDispatch<'_>,
    status_buffer: &Buffer,
) {
    let SingleIdwtDispatch {
        runtime,
        sub_bands,
        params,
        decoded,
        decoded_offset,
    } = dispatch;
    let IdwtSubBandBuffers {
        ll,
        ll_offset,
        hl,
        hl_offset,
        lh,
        lh_offset,
        hh,
        hh_offset,
    } = sub_bands;
    encoder.set_compute_pipeline_state(&runtime.idwt_irreversible97_single_decomposition);
    encoder.set_buffer(0, Some(ll), ll_offset as u64);
    encoder.set_buffer(1, Some(hl), hl_offset as u64);
    encoder.set_buffer(2, Some(lh), lh_offset as u64);
    encoder.set_buffer(3, Some(hh), hh_offset as u64);
    encoder.set_buffer(4, Some(decoded), decoded_offset as u64);
    encoder.set_bytes(
        5,
        size_of::<J2kIdwtSingleDecompositionParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(6, Some(status_buffer), 0);
    dispatch_single_thread(encoder);
}

#[cfg(target_os = "macos")]
pub(super) fn classic_batch_uses_plain_fast_path(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        if job.style_flags != 0
            || job.width > J2K_CLASSIC_MAX_WIDTH
            || job.height > J2K_CLASSIC_MAX_HEIGHT
        {
            return false;
        }
        let start = job.segment_offset as usize;
        let Some(end) = start.checked_add(job.segment_count as usize) else {
            return false;
        };
        segments.get(start..end).is_some_and(|job_segments| {
            job_segments
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
        })
    })
}

#[cfg(target_os = "macos")]
pub(super) fn classic_repeated_uses_plain_fast_path(
    count: usize,
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    let _ = (count, jobs, segments);
    // Batch-16 WSI benches are faster with device-state cleanup plus the separate parallel store.
    false
}

#[cfg(target_os = "macos")]
pub(super) fn classic_batch_is_plain_arithmetic(
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
) -> bool {
    jobs.iter().all(|job| {
        job.style_flags == 0
            && segments[job.segment_offset as usize
                ..job.segment_offset as usize + job.segment_count as usize]
                .iter()
                .all(|segment| segment.use_arithmetic != 0)
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_cleanup_batched(
    runtime: &MetalRuntime,
    coded_data: &[u8],
    jobs: &[J2kClassicCleanupBatchJob],
    segments: &[J2kClassicSegment],
    decoded: &Buffer,
) -> Result<(), Error> {
    let input = borrow_slice_buffer(&runtime.device, coded_data);
    let jobs_buffer = borrow_slice_buffer(&runtime.device, jobs);
    let segments_buffer = borrow_slice_buffer(&runtime.device, segments);
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(jobs, segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        jobs.len().max(1) * size_of::<J2kClassicStatus>(),
    );

    let command_buffer = runtime.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(&input), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(&jobs_buffer), 0);
    encoder.set_buffer(3, Some(&segments_buffer), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(&coefficients_scratch.buffer), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: jobs.len() as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
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
    }
    encoder.end_encoding();
    commit_and_wait_metal(command_buffer)?;

    let statuses =
        checked_buffer_slice::<J2kClassicStatus>(&status_buffer, jobs.len(), "classic status")?;
    let status = statuses
        .iter()
        .copied()
        .find(|status| status.code != J2K_CLASSIC_STATUS_OK);
    runtime.recycle_private_buffer(coefficients_scratch.bytes, coefficients_scratch.buffer)?;
    if let Some(status) = status {
        return Err(decode_classic_status_error(status));
    }

    Ok(())
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct ClassicCleanupBatchDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) coded_data: &'a Buffer,
    pub(super) jobs: &'a Buffer,
    pub(super) job_count: usize,
    pub(super) use_plain_fast_path: bool,
    pub(super) segments: &'a Buffer,
    pub(super) decoded: &'a Buffer,
    pub(super) coefficients_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct ClassicRepeatedCleanupDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) coded_data: &'a Buffer,
    pub(super) jobs: &'a Buffer,
    pub(super) job_count: usize,
    pub(super) total_job_count: usize,
    pub(super) output_plane_len: usize,
    pub(super) use_plain_fast_path: bool,
    pub(super) segments: &'a Buffer,
    pub(super) decoded: &'a Buffer,
    pub(super) coefficients_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct ClassicPlainDevRepeatedCleanupDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) coded_data: &'a Buffer,
    pub(super) jobs: &'a Buffer,
    pub(super) job_count: usize,
    pub(super) total_job_count: usize,
    pub(super) output_plane_len: usize,
    pub(super) segments: &'a Buffer,
    pub(super) decoded: &'a Buffer,
    pub(super) coefficients_scratch: &'a Buffer,
    pub(super) states_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) struct ClassicRepeatedStoreDispatch<'a> {
    pub(super) runtime: &'a MetalRuntime,
    pub(super) command_buffer: &'a CommandBufferRef,
    pub(super) jobs: &'a Buffer,
    pub(super) job_count: usize,
    pub(super) total_job_count: usize,
    pub(super) output_plane_len: usize,
    pub(super) decoded: &'a Buffer,
    pub(super) coefficients_scratch: &'a Buffer,
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_cleanup_batched_in_command_buffer(
    command_buffer: &CommandBufferRef,
    dispatch: ClassicCleanupBatchDispatch<'_>,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = zeroed_shared_buffer(
        &dispatch.runtime.device,
        dispatch.job_count.max(1) * size_of::<J2kClassicStatus>(),
    );

    let encoder = command_buffer.new_compute_command_encoder();
    dispatch_classic_cleanup_batched_in_encoder_with_status(encoder, dispatch, &status_buffer);
    encoder.end_encoding();

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: dispatch.job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_cleanup_batched_in_encoder(
    encoder: &ComputeCommandEncoderRef,
    dispatch: ClassicCleanupBatchDispatch<'_>,
) -> (DirectStatusCheck, Option<Buffer>) {
    let status_buffer = zeroed_shared_buffer(
        &dispatch.runtime.device,
        dispatch.job_count.max(1) * size_of::<J2kClassicStatus>(),
    );
    dispatch_classic_cleanup_batched_in_encoder_with_status(encoder, dispatch, &status_buffer);

    (
        DirectStatusCheck::Classic {
            buffer: status_buffer,
            len: dispatch.job_count,
        },
        None,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_cleanup_batched_in_encoder_with_status(
    encoder: &ComputeCommandEncoderRef,
    dispatch: ClassicCleanupBatchDispatch<'_>,
    status_buffer: &Buffer,
) {
    let ClassicCleanupBatchDispatch {
        runtime,
        coded_data,
        jobs,
        job_count,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
    } = dispatch;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_batched
    } else {
        &runtime.classic_cleanup_batched
    };
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
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
}

#[cfg(target_os = "macos")]
fn classic_repeated_batch_params(
    job_count: usize,
    total_job_count: usize,
    output_plane_len: usize,
) -> Result<J2kClassicRepeatedBatchParams, Error> {
    Ok(J2kClassicRepeatedBatchParams {
        job_count: j2k_u32_param(job_count, "classic repeated base job count exceeds u32")?,
        output_plane_len: j2k_u32_param(
            output_plane_len,
            "classic repeated output plane len exceeds u32",
        )?,
        batch_count: j2k_u32_param(
            total_job_count / job_count.max(1),
            "classic repeated batch count exceeds u32",
        )?,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_cleanup_repeated_batched_in_command_buffer(
    dispatch: ClassicRepeatedCleanupDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let ClassicRepeatedCleanupDispatch {
        runtime,
        command_buffer,
        coded_data,
        jobs,
        job_count,
        total_job_count,
        output_plane_len,
        use_plain_fast_path,
        segments,
        decoded,
        coefficients_scratch,
    } = dispatch;
    let pipeline = if use_plain_fast_path {
        &runtime.classic_cleanup_plain_repeated_batched
    } else {
        &runtime.classic_cleanup_repeated_batched
    };
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        total_job_count.max(1) * size_of::<J2kClassicStatus>(),
    );
    let repeated = classic_repeated_batch_params(job_count, total_job_count, output_plane_len)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        6,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    if use_plain_fast_path {
        encoder.dispatch_thread_groups(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
    } else {
        let width = pipeline
            .thread_execution_width()
            .max(1)
            .min(job_count as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: job_count as u64,
                height: u64::from(repeated.batch_count),
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end_encoding();

    Ok(DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_cleanup_plain_dev_repeated_batched_in_command_buffer(
    dispatch: ClassicPlainDevRepeatedCleanupDispatch<'_>,
) -> Result<DirectStatusCheck, Error> {
    let ClassicPlainDevRepeatedCleanupDispatch {
        runtime,
        command_buffer,
        coded_data,
        jobs,
        job_count,
        total_job_count,
        output_plane_len,
        segments,
        decoded,
        coefficients_scratch,
        states_scratch,
    } = dispatch;
    let status_buffer = zeroed_shared_buffer(
        &runtime.device,
        total_job_count.max(1) * size_of::<J2kClassicStatus>(),
    );
    let repeated = classic_repeated_batch_params(job_count, total_job_count, output_plane_len)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_cleanup_plain_dev_repeated_batched);
    encoder.set_buffer(0, Some(coded_data), 0);
    encoder.set_buffer(1, Some(decoded), 0);
    encoder.set_buffer(2, Some(jobs), 0);
    encoder.set_buffer(3, Some(segments), 0);
    encoder.set_buffer(4, Some(&status_buffer), 0);
    encoder.set_buffer(5, Some(coefficients_scratch), 0);
    encoder.set_buffer(6, Some(states_scratch), 0);
    encoder.set_bytes(
        7,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    let width = runtime
        .classic_cleanup_plain_dev_repeated_batched
        .thread_execution_width()
        .max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: job_count as u64,
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

    Ok(DirectStatusCheck::Classic {
        buffer: status_buffer,
        len: total_job_count,
    })
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_classic_store_repeated_batched_in_command_buffer(
    dispatch: ClassicRepeatedStoreDispatch<'_>,
) -> Result<(), Error> {
    let ClassicRepeatedStoreDispatch {
        runtime,
        command_buffer,
        jobs,
        job_count,
        total_job_count,
        output_plane_len,
        decoded,
        coefficients_scratch,
    } = dispatch;
    let repeated = classic_repeated_batch_params(job_count, total_job_count, output_plane_len)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.classic_store_repeated_batched);
    encoder.set_buffer(0, Some(decoded), 0);
    encoder.set_buffer(1, Some(jobs), 0);
    encoder.set_buffer(2, Some(coefficients_scratch), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kClassicRepeatedBatchParams>() as u64,
        (&raw const repeated).cast(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: job_count as u64,
            height: u64::from(repeated.batch_count),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(super) fn encode_distinct_classic_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedClassicSubBand],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
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
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands.iter().map(|sub_band| DistinctClassicBatch {
            coded_data: &sub_band.coded_data,
            jobs: &sub_band.jobs,
            segments: &sub_band.segments,
            output_base: sub_bands
                .iter()
                .position(|candidate| core::ptr::eq(*candidate, *sub_band))
                .expect("sub-band exists")
                * per_instance_len,
        }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn encode_distinct_classic_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedClassicSubBandGroup],
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
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
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_classic_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctClassicBatch {
                coded_data: &group.coded_data,
                jobs: &group.jobs,
                segments: &group.segments,
                output_base: index * per_instance_len,
            }),
        output,
        scratch_buffers,
    )
}

#[cfg(target_os = "macos")]
pub(super) struct DistinctClassicBatch<'a> {
    pub(super) coded_data: &'a [u8],
    pub(super) jobs: &'a [J2kClassicCleanupBatchJob],
    pub(super) segments: &'a [J2kClassicSegment],
    pub(super) output_base: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_distinct_classic_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctClassicBatch<'a>>,
    output: &Buffer,
    scratch_buffers: &mut Vec<DirectScratchBuffer>,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();
    let mut segments = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color coded payload exceeds u32".to_string(),
        })?;
        let segment_base = u32::try_from(segments.len()).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color segment table exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        for segment in batch.segments {
            let mut adjusted = *segment;
            adjusted.data_offset =
                adjusted
                    .data_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color segment offset overflow"
                            .to_string(),
                    })?;
            segments.push(adjusted);
        }
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "classic J2K MetalDirect distinct color output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "classic J2K MetalDirect distinct color job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.segment_offset = adjusted
                .segment_offset
                .checked_add(segment_base)
                .ok_or_else(|| Error::MetalKernel {
                    message: "classic J2K MetalDirect distinct color job segment offset overflow"
                        .to_string(),
                })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message:
                            "classic J2K MetalDirect distinct color job output offset overflow"
                                .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
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

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let segments_buffer = owned_slice_buffer(&runtime.device, &segments);
    let use_plain_fast_path = classic_batch_uses_plain_fast_path(&jobs, &segments)
        && runtime
            .classic_cleanup_plain_batched
            .max_total_threads_per_threadgroup()
            >= 32;
    let coefficients_scratch = take_classic_coefficients_scratch_buffer(runtime, jobs.len())?;
    let (status_check, states_scratch) = dispatch_classic_cleanup_batched_in_command_buffer(
        command_buffer,
        ClassicCleanupBatchDispatch {
            runtime,
            coded_data: &coded_buffer,
            jobs: &jobs_buffer,
            job_count: jobs.len(),
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
pub(super) fn encode_distinct_ht_sub_bands_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    sub_bands: &[&PreparedHtSubBand],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = sub_bands.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.width as usize * first.height as usize;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        sub_bands
            .iter()
            .enumerate()
            .map(|(index, sub_band)| DistinctHtBatch {
                coded_data: &sub_band.coded_data,
                jobs: &sub_band.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
pub(super) fn encode_distinct_ht_sub_band_groups_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    groups: &[&PreparedHtSubBandGroup],
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let Some(first) = groups.first() else {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    };
    let per_instance_len = first.total_coefficients;
    encode_distinct_ht_batches_to_buffer_in_command_buffer(
        runtime,
        command_buffer,
        groups
            .iter()
            .enumerate()
            .map(|(index, group)| DistinctHtBatch {
                coded_data: &group.coded_arena.data,
                jobs: &group.jobs,
                output_base: index * per_instance_len,
            }),
        output,
    )
}

#[cfg(target_os = "macos")]
pub(super) struct DistinctHtBatch<'a> {
    pub(super) coded_data: &'a [u8],
    pub(super) jobs: &'a [J2kHtCleanupBatchJob],
    pub(super) output_base: usize,
}

#[cfg(target_os = "macos")]
pub(super) fn encode_distinct_ht_batches_to_buffer_in_command_buffer<'a>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    batches: impl IntoIterator<Item = DistinctHtBatch<'a>>,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    let mut coded_data = Vec::new();
    let mut jobs = Vec::new();

    for batch in batches {
        let coded_base = u32::try_from(coded_data.len()).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale coded payload exceeds u32".to_string(),
        })?;
        coded_data.extend_from_slice(batch.coded_data);
        let output_base = u32::try_from(batch.output_base).map_err(|_| Error::MetalKernel {
            message: "HTJ2K MetalDirect distinct grayscale output offset exceeds u32".to_string(),
        })?;
        for job in batch.jobs {
            let mut adjusted = *job;
            adjusted.coded_offset =
                adjusted
                    .coded_offset
                    .checked_add(coded_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job coded offset overflow"
                            .to_string(),
                    })?;
            adjusted.output_offset =
                adjusted
                    .output_offset
                    .checked_add(output_base)
                    .ok_or_else(|| Error::MetalKernel {
                        message: "HTJ2K MetalDirect distinct grayscale job output offset overflow"
                            .to_string(),
                    })?;
            jobs.push(adjusted);
        }
    }

    if jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok((
            vec![empty.clone()],
            DirectStatusCheck::Ht {
                buffer: empty,
                len: 0,
            },
        ));
    }

    let coded_buffer = owned_slice_buffer(&runtime.device, &coded_data);
    let jobs_buffer = owned_slice_buffer(&runtime.device, &jobs);
    let status_check = dispatch_ht_cleanup_batched_in_command_buffer(
        runtime,
        command_buffer,
        &coded_buffer,
        &jobs_buffer,
        jobs.len(),
        output,
        ht_batch_output_word_count(&jobs)?,
    )?;
    Ok((vec![coded_buffer, jobs_buffer], status_check))
}

#[cfg(target_os = "macos")]
pub(super) fn encode_repeated_classic_sub_band_to_buffer_in_command_buffer(
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
pub(super) fn encode_repeated_classic_sub_band_group_to_buffer_in_command_buffer(
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
pub(super) fn encode_prepared_classic_sub_band_to_buffer_in_encoder(
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
pub(super) fn encode_prepared_classic_sub_band_group_to_buffer_in_encoder(
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

#[cfg(target_os = "macos")]
pub(super) fn required_ht_output_len(job: HtCodeBlockDecodeJob<'_>) -> Result<usize, Error> {
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
pub(super) fn encode_repeated_ht_sub_band_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    job: &PreparedHtSubBand,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || job.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
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
pub(super) fn encode_repeated_ht_sub_band_group_to_buffer_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    group: &PreparedHtSubBandGroup,
    count: usize,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if count == 0 || group.jobs.is_empty() {
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
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
pub(super) fn encode_prepared_ht_sub_band_to_buffer_in_encoder(
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
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
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
pub(super) fn encode_prepared_ht_sub_band_group_to_buffer_in_encoder(
    runtime: &MetalRuntime,
    encoder: &ComputeCommandEncoderRef,
    group: &PreparedHtSubBandGroup,
    output: &Buffer,
) -> Result<(Vec<Buffer>, DirectStatusCheck), Error> {
    if group.jobs.is_empty() {
        dispatch_zero_u32_buffer_in_encoder(runtime, encoder, output, group.total_coefficients)?;
        let empty = runtime
            .device
            .new_buffer(1, MTLResourceOptions::StorageModeShared);
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
pub(super) fn ht_output_word_count(
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
pub(super) fn ht_batch_output_word_count(jobs: &[J2kHtCleanupBatchJob]) -> Result<usize, Error> {
    let mut word_count = 0usize;
    for job in jobs {
        let job_word_count =
            ht_output_word_count(job.output_offset, job.output_stride, job.width, job.height)?;
        word_count = word_count.max(job_word_count);
    }
    Ok(word_count)
}

#[cfg(target_os = "macos")]
pub(super) fn dispatch_zero_u32_buffer_in_encoder(
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
