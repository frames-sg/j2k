// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_jpeg_pack_to_surface_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    plane0: &Buffer,
    plane1: Option<&Buffer>,
    plane2: Option<&Buffer>,
    dims: (u32, u32),
    mode: PlaneMode,
    fmt: PixelFormat,
) -> Result<Surface, Error> {
    match (mode, fmt) {
        (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => {
            return Ok(Surface::from_metal_buffer(plane0.clone(), dims, fmt));
        }
        (
            PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
            PixelFormat::Rgb8 | PixelFormat::Rgba8,
        )
        | (PlaneMode::Rgb, PixelFormat::Gray8) => {}
        _ => {
            return Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            });
        }
    }

    let pitch_bytes = dims.0 as usize * fmt.bytes_per_pixel();
    let out_buffer = runtime.device.new_buffer(
        (pitch_bytes * dims.1 as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let params = JpegPackParams {
        width: dims.0,
        height: dims.1,
        out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        mode: match mode {
            PlaneMode::Gray => MODE_GRAY,
            PlaneMode::YCbCr => MODE_YCBCR,
            PlaneMode::Rgb => MODE_RGB,
        },
        out_format: match fmt {
            PixelFormat::Gray8 => OUT_GRAY,
            PixelFormat::Rgb8 => OUT_RGB,
            PixelFormat::Rgba8 => OUT_RGBA,
            _ => unreachable!("validated by caller"),
        },
    };

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
    encoder.set_buffer(0, Some(plane0), 0);
    encoder.set_buffer(1, plane1.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(2, plane2.map(std::convert::AsRef::as_ref), 0);
    encoder.set_buffer(3, Some(&out_buffer), 0);
    encoder.set_bytes(
        4,
        size_of::<JpegPackParams>() as u64,
        (&raw const params).cast(),
    );
    dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, dims);
    encoder.end_encoding();

    Ok(Surface::from_metal_buffer(out_buffer, dims, fmt))
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_region_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let source_window = fast_subsampled_full_mcu_window::<P>(packet.dimensions(), roi);
    let mut params = fast_subsampled_region_params(packet, fmt, source_window)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        P::MCU_WIDTH,
        P::MCU_HEIGHT,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;

    let local_roi = j2k_jpeg::Rect {
        x: roi.x - source_window.x,
        y: roi.y - source_window.y,
        w: roi.w,
        h: roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, false);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::region_decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (roi.w, roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (roi.w, roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_scaled_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast_subsampled_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal {} scale {scale:?}", P::FAMILY_NAME),
        });
    };

    let y_len = params.scaled_width as usize * params.scaled_height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, fmt == PixelFormat::Gray8);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, packet.restart_offsets())?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::scaled_decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.scaled_width as usize * fmt.bytes_per_pixel() * params.scaled_height as usize)
                as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_params = JpegFast420Params {
            width: params.scaled_width,
            height: params.scaled_height,
            chroma_width: params.chroma_width,
            chroma_height: params.chroma_height,
            mcus_per_row: params.mcus_per_row,
            mcu_rows: params.mcu_rows,
            restart_interval_mcus: params.restart_interval_mcus,
            restart_offset_count: params.restart_offset_count,
            restart_start_mcu: params.restart_start_mcu,
            entropy_len: params.entropy_len,
            out_stride: checked_u32(
                params.scaled_width as usize * fmt.bytes_per_pixel(),
                "scaled output stride",
            )?,
            alpha: u32::from(u8::MAX),
            out_format: pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            })?,
            origin_x: 0,
            origin_y: 0,
        };
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported JPEG Metal {} pixel format {fmt:?}",
                    P::FAMILY_NAME
                ),
            });
        };
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&cb_plane), 0);
        pack_encoder.set_buffer(2, Some(&cr_plane), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const pack_params).cast(),
        );
        dispatch_2d_pipeline(
            pack_encoder,
            pack_pipeline,
            (params.scaled_width, params.scaled_height),
        );
        pack_encoder.end_encoding();
    }

    let surface = match out_buffer {
        Some(out_buffer) => {
            Surface::from_metal_buffer(out_buffer, (params.scaled_width, params.scaled_height), fmt)
        }
        None => Surface::from_metal_buffer(
            y_plane.clone(),
            (params.scaled_width, params.scaled_height),
            fmt,
        ),
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast_subsampled_scaled_region_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    roi: Rect,
    scale: j2k_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(full_params) = fast_subsampled_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal {} scale {scale:?}", P::FAMILY_NAME),
        });
    };
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let Some(mut decode_params) =
        fast_subsampled_scaled_region_params(packet, scale, source_window)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "unsupported JPEG Metal {} scaled region {scale:?}",
                P::FAMILY_NAME
            ),
        });
    };
    let mcu_width = P::MCU_WIDTH >> decode_params.scale_shift;
    let mcu_height = P::MCU_HEIGHT >> decode_params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        source_window,
        packet.mcus_per_row(),
        packet.mcu_rows(),
        mcu_width,
        mcu_height,
    );
    let total_mcus = packet.mcus_per_row() * packet.mcu_rows();
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        packet.restart_offsets(),
        packet.restart_interval_mcus(),
        total_mcus,
        first_mcu,
        end_mcu,
    );
    decode_params.restart_start_mcu = restart_start_mcu;
    decode_params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    )?;
    let local_roi = j2k_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        fmt,
        local_roi,
    )?;
    let y_len = source_window.w as usize * source_window.h as usize;
    let chroma_len =
        source_window.w.div_ceil(2) as usize * P::chroma_height(source_window.h) as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, false);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        restart_offsets.len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        packet.entropy_bytes(),
        packet.entropy_checkpoints(),
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::scaled_region_decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let out_buffer = runtime.device.new_buffer(
        (pack_params.out_stride as usize * scaled_roi.h as usize) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let pack_encoder = command_buffer.new_compute_command_encoder();
    let pack_pipeline = P::pack_windowed_pipeline_for_format(runtime, fmt);
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420WindowedPackParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_2d_pipeline(pack_encoder, pack_pipeline, (scaled_roi.w, scaled_roi.h));
    pack_encoder.end_encoding();

    Ok(BatchedDecodeItem {
        request_index,
        surface: Surface::from_metal_buffer(out_buffer, (scaled_roi.w, scaled_roi.h), fmt),
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast_subsampled_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = new_decode_plane_buffer(&runtime.device, y_len, fmt == PixelFormat::Gray8);
    let cb_plane = new_private_buffer(&runtime.device, chroma_len);
    let cr_plane = new_private_buffer(&runtime.device, chroma_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus(),
        packet.restart_offsets().len(),
        packet.entropy_checkpoints().len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes().as_ptr().cast(),
        packet.entropy_bytes().len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, packet.restart_offsets())?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, packet.entropy_checkpoints())?;

    let dc_tables = [
        PreparedHuffmanHost::from(packet.y_dc_table()),
        PreparedHuffmanHost::from(packet.cb_dc_table()),
        PreparedHuffmanHost::from(packet.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(packet.y_ac_table()),
        PreparedHuffmanHost::from(packet.cb_ac_table()),
        PreparedHuffmanHost::from(packet.cr_ac_table()),
    ];

    let decode_pipeline = P::decode_pipeline(runtime);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast420Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [packet.y_quant(), packet.cb_quant(), packet.cr_quant()],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(decoder_encoder, decode_pipeline, decode_threads);
    decoder_encoder.end_encoding();

    let surface = if fmt == PixelFormat::Gray8 {
        Surface::from_metal_buffer(y_plane.clone(), packet.dimensions(), fmt)
    } else {
        let Some(pack_pipeline) = P::pack_pipeline_for_format(runtime, fmt) else {
            return Err(Error::MetalKernel {
                message: format!(
                    "unsupported JPEG Metal {} pixel format {fmt:?}",
                    P::FAMILY_NAME
                ),
            });
        };
        let out_buffer = runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(pack_pipeline);
        bind_three_plane_pack::<JpegFast420Params>(
            pack_encoder,
            [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
            &out_buffer,
            &params,
        );
        dispatch_2d_pipeline(pack_encoder, pack_pipeline, packet.dimensions());
        pack_encoder.end_encoding();
        Surface::from_metal_buffer(out_buffer, packet.dimensions(), fmt)
    };

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

/// Route one batch request to the family's encode item for its op.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast_subsampled_op_batch_item<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &P,
    fmt: PixelFormat,
    op: batch::BatchOp,
) -> Result<BatchedDecodeItem, Error> {
    match op {
        batch::BatchOp::Full => {
            encode_fast_subsampled_batch_item(runtime, command_buffer, request_index, packet, fmt)
        }
        batch::BatchOp::Region(roi) => encode_fast_subsampled_region_batch_item(
            runtime,
            command_buffer,
            request_index,
            packet,
            fmt,
            roi,
        ),
        batch::BatchOp::Scaled(scale) => encode_fast_subsampled_scaled_batch_item(
            runtime,
            command_buffer,
            request_index,
            packet,
            fmt,
            scale,
        ),
        batch::BatchOp::RegionScaled { roi, scale } => {
            encode_fast_subsampled_scaled_region_batch_item(
                runtime,
                command_buffer,
                device_buffer_cache,
                request_index,
                packet,
                fmt,
                roi,
                scale,
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn encode_fast444_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    roi: Rect,
) -> Result<BatchedDecodeItem, Error> {
    let roi = core_rect_to_jpeg(roi);
    let mut params = fast444_region_params(packet, roi)?;
    let (first_mcu, end_mcu) = mcu_range_for_rect(roi, packet.mcus_per_row, packet.mcu_rows, 8, 8);
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;

    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (roi.w, roi.h),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_scaled_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let Some(params) = fast444_scaled_params(packet, scale) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast444 scale {scale:?}"),
        });
    };

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (params.scaled_width, params.scaled_height),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn encode_fast444_scaled_region_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    device_buffer_cache: &mut BatchDeviceBufferCache,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
    roi: Rect,
    scale: j2k_core::Downscale,
) -> Result<BatchedDecodeItem, Error> {
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let Some(mut params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
        return Err(Error::MetalKernel {
            message: format!("unsupported JPEG Metal fast444 scaled region {scale:?}"),
        });
    };
    let mcu_size = 8u32 >> params.scale_shift;
    let (first_mcu, end_mcu) = mcu_range_for_rect(
        scaled_roi,
        packet.mcus_per_row,
        packet.mcu_rows,
        mcu_size,
        mcu_size,
    );
    let total_mcus = packet.mcus_per_row * packet.mcu_rows;
    let (restart_start_mcu, restart_offsets) = restart_work_for_mcu_range(
        &packet.restart_offsets,
        packet.restart_interval_mcus,
        total_mcus,
        first_mcu,
        end_mcu,
    );
    params.restart_start_mcu = restart_start_mcu;
    params.restart_offset_count = checked_entropy_segment_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    )?;

    let plane_len = params.scaled_width as usize * params.scaled_height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, restart_offsets)?;
    let (entropy_buffer, entropy_checkpoints_buffer) = device_buffer_cache.packet_buffers(
        runtime,
        &packet.entropy_bytes,
        &packet.entropy_checkpoints,
    )?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        (scaled_roi.w, scaled_roi.h),
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn encode_fast444_batch_item(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    request_index: usize,
    packet: &JpegFast444PacketV1,
    mode: PlaneMode,
    fmt: PixelFormat,
) -> Result<BatchedDecodeItem, Error> {
    let params = fast444_params(packet)?;
    let plane_len = params.width as usize * params.height as usize;
    let y_plane = new_decode_plane_buffer(
        &runtime.device,
        plane_len,
        fmt == PixelFormat::Gray8 && mode != PlaneMode::Rgb,
    );
    let cb_plane = new_private_buffer(&runtime.device, plane_len);
    let cr_plane = new_private_buffer(&runtime.device, plane_len);
    let decode_threads = entropy_decode_thread_count(
        packet.restart_interval_mcus,
        packet.restart_offsets.len(),
        packet.entropy_checkpoints.len(),
    );
    let status_buffer = decode_status_buffer(&runtime.device, decode_threads);
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let restart_offsets_buffer = restart_offsets_buffer(&runtime.device, &packet.restart_offsets)?;
    let entropy_checkpoints_buffer =
        entropy_checkpoints_buffer(&runtime.device, &packet.entropy_checkpoints)?;

    let dc_tables = [
        PreparedHuffmanHost::from(&packet.y_dc_table),
        PreparedHuffmanHost::from(&packet.cb_dc_table),
        PreparedHuffmanHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&packet.y_ac_table),
        PreparedHuffmanHost::from(&packet.cb_ac_table),
        PreparedHuffmanHost::from(&packet.cr_ac_table),
    ];

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &entropy_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        &params,
        [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
        &dc_tables,
        &ac_tables,
        &restart_offsets_buffer,
        &status_buffer,
        &entropy_checkpoints_buffer,
    );
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_decode_pipeline,
        decode_threads,
    );
    decoder_encoder.end_encoding();

    let surface = encode_jpeg_pack_to_surface_in_command_buffer(
        runtime,
        command_buffer,
        &y_plane,
        Some(&cb_plane),
        Some(&cr_plane),
        packet.dimensions,
        mode,
        fmt,
    )?;

    Ok(BatchedDecodeItem {
        request_index,
        surface,
        status_buffer: status_buffer.clone(),
        decode_threads,
        _decode_resources: vec![
            y_plane,
            cb_plane,
            cr_plane,
            entropy_buffer,
            restart_offsets_buffer,
            entropy_checkpoints_buffer,
            status_buffer,
        ],
    })
}

#[cfg(target_os = "macos")]
fn checked_u32(value: usize, label: &str) -> Result<u32, Error> {
    u32::try_from(value).map_err(|_| Error::MetalKernel {
        message: format!("JPEG Metal {label} does not fit in u32"),
    })
}

#[cfg(target_os = "macos")]
fn batch_output_buffer_or_new(
    runtime: &MetalRuntime,
    output: Option<&crate::MetalBatchOutputBuffer>,
    dimensions: (u32, u32),
    tile_count: usize,
    out_stride: usize,
    out_tile_len: usize,
) -> Result<Buffer, Error> {
    let Some(output) = output else {
        let byte_len = out_tile_len
            .checked_mul(tile_count)
            .ok_or(BufferError::SizeOverflow {
                what: "JPEG Metal batch output bytes",
            })?;
        let byte_len_u64 = u64::try_from(byte_len).map_err(|_| BufferError::SizeOverflow {
            what: "JPEG Metal batch output bytes",
        })?;
        return Ok(runtime
            .device
            .new_buffer(byte_len_u64, MTLResourceOptions::StorageModeShared));
    };

    if output.dimensions() != dimensions
        || output.pixel_format() != PixelFormat::Rgb8
        || output.pitch_bytes() != out_stride
        || output.tile_stride_bytes() < out_tile_len
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal batch output buffer shape does not match requested RGB8 tiles",
        });
    }
    if output.tile_capacity() < tile_count {
        return Err(BufferError::OutputTooSmall {
            required: output.tile_stride_bytes().checked_mul(tile_count).ok_or(
                BufferError::SizeOverflow {
                    what: "JPEG Metal batch output bytes",
                },
            )?,
            have: output.byte_len(),
        }
        .into());
    }

    Ok(output.clone_buffer())
}

#[cfg(target_os = "macos")]
type GroupedSurfaceResult = (usize, Result<Surface, Error>);

#[cfg(target_os = "macos")]
type GroupedTextureResult = (usize, Result<crate::MetalTextureTile, Error>);

#[cfg(target_os = "macos")]
fn copy_grouped_surfaces_to_output(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchOutputBuffer,
    dimensions: (u32, u32),
    out_tile_len: usize,
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
) -> Result<Vec<GroupedSurfaceResult>, Error> {
    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal grouped buffer result count mismatch".to_string(),
        });
    }

    let output_buffer = output.clone_buffer();
    let mut copies = Vec::<(Buffer, usize, usize)>::new();
    let mut mapped_results = Vec::with_capacity(group_indices.len());
    for (original_index, result) in group_indices.iter().copied().zip(group_results) {
        match result {
            Ok(surface) => {
                let (source, source_offset) =
                    surface.metal_buffer().ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal grouped buffer source was not Metal-backed"
                            .to_string(),
                    })?;
                let destination_offset = original_index
                    .checked_mul(output.tile_stride_bytes())
                    .ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal grouped buffer destination offset overflowed"
                            .to_string(),
                    })?;
                copies.push((source.clone(), source_offset, destination_offset));
                mapped_results.push((
                    original_index,
                    Ok(Surface::from_metal_buffer_offset(
                        output_buffer.clone(),
                        dimensions,
                        PixelFormat::Rgb8,
                        destination_offset,
                    )),
                ));
            }
            Err(error) => mapped_results.push((original_index, Err(error))),
        }
    }

    if !copies.is_empty() {
        let command_buffer = runtime.queue.new_command_buffer();
        let blit = command_buffer.new_blit_command_encoder();
        for (source, source_offset, destination_offset) in copies {
            blit.copy_from_buffer(
                &source,
                u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer source offset exceeds u64".to_string(),
                })?,
                &output_buffer,
                u64::try_from(destination_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer destination offset exceeds u64".to_string(),
                })?,
                u64::try_from(out_tile_len).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal grouped buffer copy size exceeds u64".to_string(),
                })?,
            );
        }
        blit.end_encoding();
        commit_and_wait_jpeg(command_buffer)?;
    }

    Ok(mapped_results)
}

#[cfg(target_os = "macos")]
fn validate_rgba_texture_batch_output(
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
    out_tile_len: usize,
) -> Result<(), Error> {
    if output.dimensions() != dimensions
        || output.pixel_format() != PixelFormat::Rgba8
        || output.metal_pixel_format() != MTLPixelFormat::RGBA8Unorm
    {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal batch texture output shape does not match requested RGBA8 tiles",
        });
    }
    if output.tile_capacity() < tile_count {
        return Err(BufferError::OutputTooSmall {
            required: out_tile_len
                .checked_mul(tile_count)
                .ok_or(BufferError::SizeOverflow {
                    what: "JPEG Metal batch texture output bytes",
                })?,
            have: out_tile_len.checked_mul(output.tile_capacity()).ok_or(
                BufferError::SizeOverflow {
                    what: "JPEG Metal batch texture output bytes",
                },
            )?,
        }
        .into());
    }

    for index in 0..tile_count {
        let Some(texture) = output.texture(index) else {
            return Err(Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            });
        };
        if texture.width() != u64::from(dimensions.0)
            || texture.height() != u64::from(dimensions.1)
            || texture.pixel_format() != MTLPixelFormat::RGBA8Unorm
        {
            return Err(Error::UnsupportedMetalRequest {
                reason:
                    "JPEG Metal batch texture output texture does not match requested RGBA8 tiles",
            });
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn texture_batch_success_results(
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let mut results = Vec::with_capacity(tile_count);
    for index in 0..tile_count {
        let texture = output
            .clone_texture(index)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
        results.push(Ok(crate::MetalTextureTile::new(
            texture,
            dimensions,
            PixelFormat::Rgba8,
        )));
    }
    Ok(results)
}

#[cfg(target_os = "macos")]
fn copy_rgb8_surfaces_to_rgba_textures(
    runtime: &MetalRuntime,
    output: &crate::MetalBatchTextureOutput,
    dimensions: (u32, u32),
    tile_count: usize,
    group_indices: &[usize],
    group_results: Vec<Result<Surface, Error>>,
) -> Result<Vec<GroupedTextureResult>, Error> {
    if group_results.len() != group_indices.len() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal grouped texture result count mismatch".to_string(),
        });
    }
    let out_tile_len = dimensions
        .0
        .checked_mul(dimensions.1)
        .and_then(|pixels| {
            pixels.checked_mul(u32::try_from(PixelFormat::Rgba8.bytes_per_pixel()).ok()?)
        })
        .ok_or(BufferError::SizeOverflow {
            what: "JPEG Metal batch texture output bytes",
        })? as usize;
    validate_rgba_texture_batch_output(output, dimensions, tile_count, out_tile_len)?;

    let in_stride = dimensions
        .0
        .checked_mul(
            u32::try_from(PixelFormat::Rgb8.bytes_per_pixel()).map_err(|_| {
                BufferError::SizeOverflow {
                    what: "JPEG Metal RGB texture copy input stride",
                }
            })?,
        )
        .ok_or(BufferError::SizeOverflow {
            what: "JPEG Metal RGB texture copy input stride",
        })?;
    let params = JpegRgb8ToRgbaTextureParams {
        width: dimensions.0,
        height: dimensions.1,
        in_stride,
        alpha: u32::from(u8::MAX),
    };
    let mut copies = Vec::<(usize, Buffer, usize)>::new();
    let mut mapped_results = Vec::with_capacity(group_indices.len());
    for (original_index, result) in group_indices.iter().copied().zip(group_results) {
        match result {
            Ok(surface) => {
                if surface.dimensions != dimensions || surface.fmt != PixelFormat::Rgb8 {
                    return Err(Error::MetalKernel {
                        message: "JPEG Metal texture copy source shape mismatch".to_string(),
                    });
                }
                let (source, source_offset) =
                    surface.metal_buffer().ok_or_else(|| Error::MetalKernel {
                        message: "JPEG Metal texture copy source was not Metal-backed".to_string(),
                    })?;
                let texture =
                    output
                        .clone_texture(original_index)
                        .ok_or_else(|| Error::MetalKernel {
                            message: "JPEG Metal batch texture output slot was missing".to_string(),
                        })?;
                copies.push((original_index, source.clone(), source_offset));
                mapped_results.push((
                    original_index,
                    Ok(crate::MetalTextureTile::new(
                        texture,
                        dimensions,
                        PixelFormat::Rgba8,
                    )),
                ));
            }
            Err(error) => mapped_results.push((original_index, Err(error))),
        }
    }

    if !copies.is_empty() {
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.rgb8_to_rgba_texture_pipeline);
        for (original_index, source, source_offset) in copies {
            let texture = output
                .texture(original_index)
                .ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Metal batch texture output slot was missing".to_string(),
                })?;
            encoder.set_buffer(
                0,
                Some(&source),
                u64::try_from(source_offset).map_err(|_| Error::MetalKernel {
                    message: "JPEG Metal texture copy source offset exceeds u64".to_string(),
                })?,
            );
            encoder.set_bytes(
                1,
                size_of::<JpegRgb8ToRgbaTextureParams>() as u64,
                (&raw const params).cast(),
            );
            encoder.set_texture(0, Some(texture));
            dispatch_2d_pipeline(encoder, &runtime.rgb8_to_rgba_texture_pipeline, dimensions);
        }
        encoder.end_encoding();
        commit_and_wait_jpeg(command_buffer)?;
    }

    Ok(mapped_results)
}

#[cfg(target_os = "macos")]
fn dispatch_rgba_texture_pack(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    planes: (&Buffer, &Buffer, &Buffer),
    output: &crate::MetalBatchTextureOutput,
    params: JpegTexturePackBatchParams,
    tile_count: usize,
    dispatch_dims: (u32, u32),
) -> Result<(), Error> {
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pipeline);
    pack_encoder.set_buffer(0, Some(planes.0), 0);
    pack_encoder.set_buffer(1, Some(planes.1), 0);
    pack_encoder.set_buffer(2, Some(planes.2), 0);
    for index in 0..tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let mut params = params;
        params.tile_index = checked_u32(index, "texture batch tile index")?;
        pack_encoder.set_texture(0, Some(texture));
        pack_encoder.set_bytes(
            3,
            size_of::<JpegTexturePackBatchParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pipeline, dispatch_dims);
    }
    pack_encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
fn dispatch_windowed_rgba_texture_pack(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    planes: (&Buffer, &Buffer, &Buffer),
    output: &crate::MetalBatchTextureOutput,
    params: JpegWindowedTexturePackBatchParams,
    tile_count: usize,
    dispatch_dims: (u32, u32),
) -> Result<(), Error> {
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pipeline);
    pack_encoder.set_buffer(0, Some(planes.0), 0);
    pack_encoder.set_buffer(1, Some(planes.1), 0);
    pack_encoder.set_buffer(2, Some(planes.2), 0);
    for index in 0..tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let mut params = params;
        params.tile_index = checked_u32(index, "windowed texture batch tile index")?;
        pack_encoder.set_texture(0, Some(texture));
        pack_encoder.set_bytes(
            3,
            size_of::<JpegWindowedTexturePackBatchParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, pipeline, dispatch_dims);
    }
    pack_encoder.end_encoding();
    Ok(())
}

/// Encode the split coeff-decode + IDCT-deposit passes shared by the surfaces
/// and texture drivers' `SplitCoeffIdct` debug mode.
#[cfg(all(target_os = "macos", test))]
#[allow(clippy::too_many_arguments)]
fn encode_split_coeff_idct_passes(
    command_buffer: &CommandBufferRef,
    pipelines: (&ComputePipelineState, &ComputePipelineState),
    params: &JpegFast420BatchParams,
    quants: [&[u16; 64]; 3],
    dc_tables: &[PreparedHuffmanHost; 3],
    ac_tables: &[PreparedHuffmanHost; 3],
    entropy: (&Buffer, &Buffer, &Buffer, &Buffer),
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    scratch: (&Buffer, &Buffer),
    total_decode_threads: u32,
    idct_grid: (u32, u32, u32),
) {
    let (coeffs_pipeline, idct_pipeline) = pipelines;
    let (entropy_payload, entropy_offsets, entropy_lens, entropy_checkpoints) = entropy;
    let (coeff_blocks, dc_only_flags) = scratch;

    let coeff_encoder = command_buffer.new_compute_command_encoder();
    coeff_encoder.set_compute_pipeline_state(coeffs_pipeline);
    coeff_encoder.set_buffer(0, Some(entropy_payload), 0);
    coeff_encoder.set_buffer(1, Some(coeff_blocks), 0);
    coeff_encoder.set_buffer(2, Some(dc_only_flags), 0);
    coeff_encoder.set_bytes(
        4,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    coeff_encoder.set_bytes(5, size_of::<[u16; 64]>() as u64, quants[0].as_ptr().cast());
    coeff_encoder.set_bytes(6, size_of::<[u16; 64]>() as u64, quants[1].as_ptr().cast());
    coeff_encoder.set_bytes(7, size_of::<[u16; 64]>() as u64, quants[2].as_ptr().cast());
    coeff_encoder.set_bytes(
        8,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        9,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    coeff_encoder.set_bytes(
        10,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        11,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    coeff_encoder.set_bytes(
        12,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    coeff_encoder.set_bytes(
        13,
        size_of::<PreparedHuffmanHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    coeff_encoder.set_buffer(14, Some(entropy_offsets), 0);
    coeff_encoder.set_buffer(15, Some(entropy_lens), 0);
    coeff_encoder.set_buffer(16, Some(status_buffer), 0);
    coeff_encoder.set_buffer(17, Some(entropy_checkpoints), 0);
    dispatch_1d_pipeline(coeff_encoder, coeffs_pipeline, total_decode_threads);
    coeff_encoder.end_encoding();

    let idct_encoder = command_buffer.new_compute_command_encoder();
    idct_encoder.set_compute_pipeline_state(idct_pipeline);
    idct_encoder.set_buffer(0, Some(coeff_blocks), 0);
    idct_encoder.set_buffer(1, Some(dc_only_flags), 0);
    idct_encoder.set_buffer(2, Some(planes[0]), 0);
    idct_encoder.set_buffer(3, Some(planes[1]), 0);
    idct_encoder.set_buffer(4, Some(planes[2]), 0);
    idct_encoder.set_bytes(
        5,
        size_of::<JpegFast420BatchParams>() as u64,
        (&raw const *params).cast(),
    );
    dispatch_3d_pipeline(idct_encoder, idct_pipeline, idct_grid);
    idct_encoder.end_encoding();
}

