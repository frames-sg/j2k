// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    bind_fast_decode_entropy_inputs, checked_entropy_segment_count, core_rect_to_jpeg,
    decode_status_buffer, dispatch_1d_pipeline, entropy_checkpoints_buffer,
    entropy_decode_thread_count, fast444_params, fast444_region_params, fast444_scaled_params,
    fast444_scaled_region_params, fast_packet_huffman_tables, mcu_range_for_rect,
    new_decode_plane_buffer, new_private_buffer, restart_offsets_buffer,
    restart_work_for_mcu_range, BatchedDecodeItem, CommandBufferRef, Error,
    FastDecodeEntropyInputs, JpegFast444PacketV1, JpegFast444Params, JpegFast444ScaledParams,
    MTLResourceOptions, MetalRuntime, PixelFormat, PlaneMode, Rect,
};
use super::common::{
    encode_jpeg_pack_to_surface_in_command_buffer, Fast444ScaledRegionBatchItemRequest,
    JpegPackSurfaceRequest,
};

#[cfg(target_os = "macos")]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast444_region_batch_item(
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

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &cb_plane, &cr_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
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
        JpegPackSurfaceRequest {
            plane0: &y_plane,
            plane1: Some(&cb_plane),
            plane2: Some(&cr_plane),
            dims: (roi.w, roi.h),
            mode,
            fmt,
        },
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

#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast444_scaled_batch_item(
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

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &cb_plane, &cr_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
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
        JpegPackSurfaceRequest {
            plane0: &y_plane,
            plane1: Some(&cb_plane),
            plane2: Some(&cr_plane),
            dims: (params.scaled_width, params.scaled_height),
            mode,
            fmt,
        },
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
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast444_scaled_region_batch_item(
    request: Fast444ScaledRegionBatchItemRequest<'_>,
) -> Result<BatchedDecodeItem, Error> {
    let Fast444ScaledRegionBatchItemRequest {
        runtime,
        command_buffer,
        device_buffer_cache,
        request_index,
        packet,
        mode,
        fmt,
        roi,
        scale,
    } = request;
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

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444ScaledParams>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &cb_plane, &cr_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
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
        JpegPackSurfaceRequest {
            plane0: &y_plane,
            plane1: Some(&cb_plane),
            plane2: Some(&cr_plane),
            dims: (scaled_roi.w, scaled_roi.h),
            mode,
            fmt,
        },
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
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
pub(in crate::compute) fn encode_fast444_batch_item(
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

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(packet);

    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFast444Params>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffer,
            planes: [&y_plane, &cb_plane, &cr_plane],
            params: &params,
            quants: [&packet.y_quant, &packet.cb_quant, &packet.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &restart_offsets_buffer,
            slot15_buffer: &status_buffer,
            slot16_buffer: &entropy_checkpoints_buffer,
        },
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
        JpegPackSurfaceRequest {
            plane0: &y_plane,
            plane1: Some(&cb_plane),
            plane2: Some(&cr_plane),
            dims: packet.dimensions,
            mode,
            fmt,
        },
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
