#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn fast444_region_scaled_rgb_output_shape(
    packet: &JpegFast444PacketV1,
    roi: Rect,
    scale: j2k_core::Downscale,
) -> Option<((u32, u32), usize, usize)> {
    let scaled = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    };
    let params = fast444_scaled_region_params(packet, scale, scaled_roi)?;
    let out_dims = (params.scaled_width, params.scaled_height);
    let out_stride = out_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * out_dims.1 as usize;
    Some((out_dims, out_stride, out_tile_len))
}

#[cfg(target_os = "macos")]
fn first_region_scaled_op(requests: &[batch::QueuedRequest]) -> Option<(Rect, j2k_core::Downscale)> {
    let batch::BatchOp::RegionScaled { roi, scale } = requests.first()?.op else {
        return None;
    };
    Some((roi, scale))
}

#[cfg(target_os = "macos")]
fn fast444_region_packets<'a>(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'a>],
) -> Option<Vec<(&'a JpegFast444PacketV1, PlaneMode)>> {
    if requests.is_empty() || requests.iter().any(|request| request.fmt != PixelFormat::Rgb8) {
        return None;
    }
    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return None;
        };
        fast444_packets.push((*packet, *mode));
    }
    Some(fast444_packets)
}

#[cfg(target_os = "macos")]
fn subsampled_region_rgb_packets<'a, P: FastRegionScaledMetal>(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'a>],
) -> Option<Vec<(&'a P, PlaneMode)>> {
    if requests.is_empty() || requests.iter().any(|request| request.fmt != PixelFormat::Rgb8) {
        return None;
    }
    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let packet = P::from_region_scaled_batched(packet)?;
        family_packets.push(packet);
    }
    Some(family_packets)
}

#[cfg(target_os = "macos")]
fn subsampled_region_texture_packets<'a, P: FastSubsampledMetal>(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'a>],
) -> Option<Vec<&'a P>> {
    if requests.is_empty() || requests.iter().any(|request| request.fmt != PixelFormat::Rgb8) {
        return None;
    }
    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let packet = P::from_batched(packet)?;
        family_packets.push(packet);
    }
    Some(family_packets)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_restart_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if !fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return Ok(None);
    }
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.entropy_bytes.is_empty() || packet.entropy_checkpoints.is_empty())
    {
        return Ok(None);
    }

    let mut first_shape = None;
    for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let Some((out_dims, _, _)) = fast444_region_scaled_rgb_output_shape(packet, roi, scale)
        else {
            return Ok(None);
        };
        let out_tile_len =
            out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, out_dims, requests.len(), out_tile_len)?;
        first_shape.get_or_insert(out_dims);
    }

    let Some(out_dims) = first_shape else {
        return Ok(Some(Vec::new()));
    };
    let mut surfaces = Vec::with_capacity(requests.len());
    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = BatchedFastPacket::Fast444(packet, mode);
        surfaces.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        out_dims,
        requests.len(),
        &group_indices,
        surfaces,
    )?;
    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart fast444 region scaled texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct Fast444RegionTextureShape {
    tile_count: usize,
    segment_count: usize,
    total_decode_threads: u32,
    out_dims: (u32, u32),
    out_tile_len: usize,
    plane_len: usize,
    decode_params: JpegFastRegionScaledBatchParams,
    pack_params: JpegTexturePackBatchParams,
}

#[cfg(target_os = "macos")]
fn fast444_region_texture_shape(
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    first: &JpegFast444PacketV1,
    first_mode: PlaneMode,
    first_roi: Rect,
    first_scale: j2k_core::Downscale,
) -> Result<Option<Fast444RegionTextureShape>, Error> {
    let first_scaled = first_roi.scaled_covering(first_scale);
    let first_scaled_roi = j2k_jpeg::Rect {
        x: first_scaled.x,
        y: first_scaled.y,
        w: first_scaled.w,
        h: first_scaled.h,
    };
    let Some(first_decode_params) =
        fast444_scaled_region_params(first, first_scale, first_scaled_roi)
    else {
        return Ok(None);
    };
    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled texture batch tile count")?;
    let segment_count_u32 =
        checked_u32(segment_count, "region scaled texture batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal region scaled texture batch decode thread count overflowed"
                    .to_string(),
            })?,
        "region scaled texture batch decode thread count",
    )?;

    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || mode != first_mode
            || !fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
        {
            return Ok(None);
        }
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = j2k_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        if fast444_scaled_region_params(packet, scale, scaled_roi) != Some(first_decode_params) {
            return Ok(None);
        }
    }

    let out_dims = (
        first_decode_params.scaled_width,
        first_decode_params.scaled_height,
    );
    let out_tile_len =
        out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
    Ok(Some(Fast444RegionTextureShape {
        tile_count,
        segment_count,
        total_decode_threads,
        out_dims,
        out_tile_len,
        plane_len: first_decode_params.scaled_width as usize
            * first_decode_params.scaled_height as usize,
        decode_params: JpegFastRegionScaledBatchParams {
            scaled_width: first_decode_params.scaled_width,
            scaled_height: first_decode_params.scaled_height,
            chroma_width: first_decode_params.scaled_width,
            chroma_height: first_decode_params.scaled_height,
            mcus_per_row: first_decode_params.mcus_per_row,
            mcu_rows: first_decode_params.mcu_rows,
            segment_count: segment_count_u32,
            tile_count: tile_count_u32,
            scale_shift: first_decode_params.scale_shift,
            origin_x: first_decode_params.origin_x,
            origin_y: first_decode_params.origin_y,
        },
        pack_params: JpegTexturePackBatchParams {
            width: out_dims.0,
            height: out_dims.1,
            chroma_width: out_dims.0,
            chroma_height: out_dims.1,
            tile_index: 0,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(first_mode),
        },
    }))
}

#[cfg(target_os = "macos")]
fn encode_fast444_region_texture_decode(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &JpegFast444PacketV1,
    entropy_buffers: &BatchEntropyBuffers,
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    shape: Fast444RegionTextureShape,
) {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffers.payload,
            planes,
            params: &shape.decode_params,
            quants: [&first.y_quant, &first.cb_quant, &first.cr_quant],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &entropy_buffers.offsets,
            slot15_buffer: &entropy_buffers.lens,
            slot16_buffer: status_buffer,
        },
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        shape.total_decode_threads,
    );
    decoder_encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(fast444_packets) = fast444_region_packets(requests, packets) else {
        return Ok(None);
    };

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let Some((first_roi, first_scale)) = first_region_scaled_op(requests) else {
        return Ok(None);
    };
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return try_decode_fast444_restart_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
        );
    }
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_region_scaled_batch_groups(requests, &fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let Some(shape) = fast444_region_texture_shape(
        requests,
        &fast444_packets,
        first,
        first_mode,
        first_roi,
        first_scale,
    )?
    else {
        return Ok(None);
    };
    validate_rgba_texture_batch_output(output, shape.out_dims, shape.tile_count, shape.out_tile_len)?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_region_scaled_texture_entropy",
            offsets: "fast444_region_scaled_texture_entropy_offsets",
            lens: "fast444_region_scaled_texture_entropy_lens",
            checkpoints: "fast444_region_scaled_texture_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        shape.tile_count,
        shape.segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_y",
        shape.plane_len * shape.tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cb",
        shape.plane_len * shape.tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cr",
        shape.plane_len * shape.tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); shape.total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_region_scaled_texture_status",
        &statuses,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    encode_fast444_region_texture_decode(
        runtime,
        command_buffer,
        first,
        &entropy_buffers,
        &status_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        shape,
    );
    dispatch_rgba_texture_pack(
        command_buffer,
        &runtime.pack_444_rgba_texture_pipeline,
        (&y_plane, &cb_plane, &cr_plane),
        output,
        shape.pack_params,
        shape.tile_count,
        shape.out_dims,
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        shape.out_dims,
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = j2k_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
            return Ok(None);
        };
        let out_dims = (params.scaled_width, params.scaled_height);
        let out_tile_len =
            out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, out_dims, requests.len(), out_tile_len)?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast444_region_scaled_rgba_batch_to_textures(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: "JPEG Metal grouped fast444 region scaled texture result count mismatch"
                    .to_string(),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped fast444 region scaled texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_restart_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[(&P, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if !family_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus() != 0)
    {
        return Ok(None);
    }
    if family_packets.iter().any(|(packet, _)| {
        packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty()
    })
    {
        return Ok(None);
    }

    let mut first_plan = None;
    if output.is_some() {
        for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} restart region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(
                    packet,
                    roi,
                    scale,
                    1,
                    segment_count_u32,
                    mode,
                )
            else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                output,
                plan.out_dims,
                requests.len(),
                plan.pack_params.out_stride as usize,
                plan.out_tile_len,
            )?;
            first_plan.get_or_insert(plan);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = packet.to_region_scaled_batched(mode);
        results.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let Some(output) = output else {
        return Ok(Some(results));
    };
    let Some(plan) = first_plan else {
        return Ok(Some(results));
    };
    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_grouped_surfaces_to_output(
        runtime,
        output,
        plan.out_dims,
        plan.out_tile_len,
        &group_indices,
        results,
    )?;
    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart {} region scaled buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct SubsampledRegionBatchShape {
    tile_count: usize,
    tile_count_u32: u32,
    segment_count: usize,
    total_decode_threads: u32,
    plan: RegionScaledBatchPlan,
}

#[cfg(target_os = "macos")]
fn subsampled_region_rgb_batch_shape<P: FastRegionScaledMetal>(
    requests: &[batch::QueuedRequest],
    family_packets: &[(&P, PlaneMode)],
    first: &P,
    first_mode: PlaneMode,
    first_roi: Rect,
    first_scale: j2k_core::Downscale,
) -> Result<Option<SubsampledRegionBatchShape>, Error> {
    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let Some(plan) = fast_subsampled_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
        first_mode,
    ) else {
        return Ok(None);
    };
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} region scaled batch decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!("{} region scaled batch decode thread count", P::FAMILY_NAME),
    )?;
    for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if mode != first_mode
            || scale != first_scale
            || !fast_subsampled_packets_share_full_rgb_batch_shape(first, packet, segment_count)
            || fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                tile_count_u32,
                segment_count_u32,
                mode,
            ) != Some(plan)
        {
            return Ok(None);
        }
    }
    Ok(Some(SubsampledRegionBatchShape {
        tile_count,
        tile_count_u32,
        segment_count,
        total_decode_threads,
        plan,
    }))
}

#[cfg(target_os = "macos")]
fn subsampled_region_texture_batch_shape<P: FastSubsampledMetal + FastRegionScaledMetal>(
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    first: &P,
    first_roi: Rect,
    first_scale: j2k_core::Downscale,
) -> Result<Option<SubsampledRegionBatchShape>, Error> {
    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled texture batch tile count")?;
    let segment_count_u32 =
        checked_u32(segment_count, "region scaled texture batch segment count")?;
    let Some(plan) = fast_subsampled_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
        PlaneMode::YCbCr,
    ) else {
        return Ok(None);
    };
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} region scaled texture decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!(
            "{} region scaled texture decode thread count",
            P::FAMILY_NAME
        ),
    )?;
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        if scale != first_scale
            || !fast_subsampled_packets_share_full_rgb_batch_shape(first, packet, segment_count)
            || fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                tile_count_u32,
                segment_count_u32,
                PlaneMode::YCbCr,
            ) != Some(plan)
        {
            return Ok(None);
        }
    }
    Ok(Some(SubsampledRegionBatchShape {
        tile_count,
        tile_count_u32,
        segment_count,
        total_decode_threads,
        plan,
    }))
}

#[cfg(target_os = "macos")]
fn encode_subsampled_region_rgb_decode<P: FastRegionScaledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &P,
    entropy_buffers: &BatchEntropyBuffers,
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    shape: SubsampledRegionBatchShape,
) {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(
        <P as FastRegionScaledMetal>::scaled_region_batch_decode_pipeline(runtime),
    );
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffers.payload,
            planes,
            params: &shape.plan.decode_params,
            quants: [first.y_quant(), first.cb_quant(), first.cr_quant()],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &entropy_buffers.offsets,
            slot15_buffer: &entropy_buffers.lens,
            slot16_buffer: status_buffer,
        },
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        <P as FastRegionScaledMetal>::scaled_region_batch_decode_pipeline(runtime),
        shape.total_decode_threads,
    );
    decoder_encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn encode_subsampled_region_texture_decode<P: FastSubsampledMetal + FastRegionScaledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &P,
    entropy_buffers: &BatchEntropyBuffers,
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    shape: SubsampledRegionBatchShape,
) {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(
        <P as FastSubsampledMetal>::scaled_region_batch_decode_pipeline(runtime),
    );
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &FastDecodeEntropyInputs {
            entropy_buffer: &entropy_buffers.payload,
            planes,
            params: &shape.plan.decode_params,
            quants: [first.y_quant(), first.cb_quant(), first.cr_quant()],
            dc_tables: &dc_tables,
            ac_tables: &ac_tables,
            slot14_buffer: &entropy_buffers.offsets,
            slot15_buffer: &entropy_buffers.lens,
            slot16_buffer: status_buffer,
        },
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        <P as FastSubsampledMetal>::scaled_region_batch_decode_pipeline(runtime),
        shape.total_decode_threads,
    );
    decoder_encoder.end_encoding();
}

#[cfg(target_os = "macos")]
fn region_plane_buffers(
    batch_scratch: &mut MetalBatchScratch,
    device: &Device,
    keys: &FastScratchKeys,
    plan: RegionScaledBatchPlan,
    tile_count: usize,
) -> (Buffer, Buffer, Buffer) {
    (
        batch_scratch.private_buffer(device, keys.y, plan.y_len * tile_count),
        batch_scratch.private_buffer(device, keys.cb, plan.chroma_len * tile_count),
        batch_scratch.private_buffer(device, keys.cr, plan.chroma_len * tile_count),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(family_packets) = subsampled_region_rgb_packets::<P>(requests, packets) else {
        return Ok(None);
    };

    let Some((first, first_mode)) = family_packets.first().copied() else {
        return Ok(None);
    };
    let Some((first_roi, first_scale)) = first_region_scaled_op(requests) else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus() != 0)
    {
        return try_decode_fast_subsampled_restart_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &family_packets,
            output,
        );
    }
    if first.restart_interval_mcus() != 0 || first.entropy_checkpoints().is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_region_scaled_batch_groups(requests, &family_packets)?
    else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &family_packets,
            output,
            groups,
        );
    }

    let Some(shape) = subsampled_region_rgb_batch_shape::<P>(
        requests,
        &family_packets,
        first,
        first_mode,
        first_roi,
        first_scale,
    )?
    else {
        return Ok(None);
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: P::REGION_SCALED_KEYS.entropy,
            offsets: P::REGION_SCALED_KEYS.entropy_offsets,
            lens: P::REGION_SCALED_KEYS.entropy_lens,
            checkpoints: P::REGION_SCALED_KEYS.entropy_checkpoints,
        },
        family_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints()),
        shape.tile_count,
        shape.segment_count,
    )?
    else {
        return Ok(None);
    };

    let (y_plane, cb_plane, cr_plane) = region_plane_buffers(
        &mut batch_scratch,
        &runtime.device,
        &P::REGION_SCALED_KEYS,
        shape.plan,
        shape.tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        shape.plan.out_dims,
        shape.tile_count,
        shape.plan.pack_params.out_stride as usize,
        shape.plan.out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); shape.total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_KEYS.status,
        &statuses,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    encode_subsampled_region_rgb_decode::<P>(
        runtime,
        command_buffer,
        first,
        &entropy_buffers,
        &status_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        shape,
    );

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(P::pack_windowed_rgb_batch_pipeline(runtime));
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &shape.plan.pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        P::pack_windowed_rgb_batch_pipeline(runtime),
        (shape.plan.out_dims.0, shape.plan.out_dims.1, shape.tile_count_u32),
    );
    pack_encoder.end_encoding();

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(surface_batch_success_results(
        &out_buffer,
        shape.plan.out_dims,
        PixelFormat::Rgb8,
        requests.len(),
        shape.plan.out_tile_len,
    )))
}

fn try_decode_fast420_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast420PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[(&P, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (request, (packet, mode)) in requests.iter().zip(family_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(
                    packet,
                    roi,
                    scale,
                    1,
                    segment_count_u32,
                    mode,
                )
            else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                plan.out_dims,
                requests.len(),
                plan.pack_params.out_stride as usize,
                plan.out_tile_len,
            )?;
        }
    }

    let mut merged_results: Vec<Option<Result<Surface, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| {
                let (packet, mode) = family_packets[index];
                packet.to_region_scaled_batched(mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) =
            try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<P>(
                runtime,
                &group_requests,
                &group_packets,
                None,
            )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let batch::BatchOp::RegionScaled { roi, scale } = requests[first_group_index].op else {
                return Ok(None);
            };
            let (packet, mode) = family_packets[first_group_index];
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(
                    packet,
                    roi,
                    scale,
                    1,
                    segment_count_u32,
                    mode,
                )
            else {
                return Ok(None);
            };
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                plan.out_dims,
                plan.out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal grouped {} region scaled buffer result count mismatch",
                        P::FAMILY_NAME
                    ),
                });
            }
            for (original_index, result) in group_indices.into_iter().zip(group_results) {
                merged_results[original_index] = Some(result);
            }
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} region scaled buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_restart_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if !family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return Ok(None);
    }
    if family_packets
        .iter()
        .any(|packet| packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty())
    {
        return Ok(None);
    }

    let mut first_plan = None;
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count_u32 = checked_u32(
            packet.entropy_checkpoints().len(),
            &format!(
                "{} restart region scaled texture segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) =
            fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                1,
                segment_count_u32,
                PlaneMode::YCbCr,
            )
        else {
            return Ok(None);
        };
        let out_tile_len = plan.out_dims.0 as usize
            * plan.out_dims.1 as usize
            * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, plan.out_dims, requests.len(), out_tile_len)?;
        first_plan.get_or_insert(plan);
    }

    let Some(plan) = first_plan else {
        return Ok(Some(Vec::new()));
    };
    let mut surfaces = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = packet.to_batched();
        surfaces.push(decode_region_scaled_packet_surface(
            runtime,
            &decoder,
            request,
            &batched_packet,
        ));
    }

    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        plan.out_dims,
        requests.len(),
        &group_indices,
        surfaces,
    )?;
    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for (index, result) in copied {
        merged_results[index] = Some(result);
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal restart {} region scaled texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal + FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(family_packets) = subsampled_region_texture_packets::<P>(requests, packets) else {
        return Ok(None);
    };

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    let Some((first_roi, first_scale)) = first_region_scaled_op(requests) else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
    {
        return try_decode_fast_subsampled_restart_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &family_packets,
            output,
        );
    }
    if first.restart_interval_mcus() != 0 || first.entropy_checkpoints().is_empty() {
        return Ok(None);
    }

    let grouped_family_packets = family_packets
        .iter()
        .copied()
        .map(|packet| (packet, PlaneMode::YCbCr))
        .collect::<Vec<_>>();
    let Some(groups) =
        fast_subsampled_region_scaled_batch_groups(requests, &grouped_family_packets)?
    else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_region_scaled_rgba_batch_to_textures(
            runtime,
            requests,
            &family_packets,
            output,
            groups,
        );
    }

    let Some(shape) = subsampled_region_texture_batch_shape::<P>(
        requests,
        &family_packets,
        first,
        first_roi,
        first_scale,
    )?
    else {
        return Ok(None);
    };
    let out_tile_len = shape.plan.out_dims.0 as usize
        * shape.plan.out_dims.1 as usize
        * PixelFormat::Rgba8.bytes_per_pixel();
    validate_rgba_texture_batch_output(
        output,
        shape.plan.out_dims,
        shape.tile_count,
        out_tile_len,
    )?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: P::REGION_SCALED_TEXTURE_KEYS.entropy,
            offsets: P::REGION_SCALED_TEXTURE_KEYS.entropy_offsets,
            lens: P::REGION_SCALED_TEXTURE_KEYS.entropy_lens,
            checkpoints: P::REGION_SCALED_TEXTURE_KEYS.entropy_checkpoints,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
        shape.tile_count,
        shape.segment_count,
    )?
    else {
        return Ok(None);
    };

    let (y_plane, cb_plane, cr_plane) = region_plane_buffers(
        &mut batch_scratch,
        &runtime.device,
        &P::REGION_SCALED_TEXTURE_KEYS,
        shape.plan,
        shape.tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); shape.total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.status,
        &statuses,
    );

    let command_buffer = runtime.queue.new_command_buffer();
    encode_subsampled_region_texture_decode::<P>(
        runtime,
        command_buffer,
        first,
        &entropy_buffers,
        &status_buffer,
        [&y_plane, &cb_plane, &cr_plane],
        shape,
    );

    dispatch_windowed_rgba_texture_pack(
        command_buffer,
        P::pack_windowed_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        windowed_texture_pack_params(shape.plan),
        shape.tile_count,
        shape.plan.out_dims,
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        shape.plan.out_dims,
        requests.len(),
    )?))
}

fn try_decode_fast420_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<JpegFast420PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_region_scaled_rgba_batch_to_textures<
    P: FastSubsampledMetal + FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count_u32 = checked_u32(
            packet.entropy_checkpoints().len(),
            &format!(
                "{} grouped region scaled texture batch segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) =
            fast_subsampled_region_scaled_batch_plan(
                packet,
                roi,
                scale,
                1,
                segment_count_u32,
                PlaneMode::YCbCr,
            )
        else {
            return Ok(None);
        };
        let out_tile_len = plan.out_dims.0 as usize
            * plan.out_dims.1 as usize
            * PixelFormat::Rgba8.bytes_per_pixel();
        validate_rgba_texture_batch_output(output, plan.out_dims, requests.len(), out_tile_len)?;
    }

    let mut merged_results: Vec<Option<Result<crate::MetalTextureTile, Error>>> =
        (0..requests.len()).map(|_| None).collect();
    for group_indices in groups {
        let group_output = output.clone_slots(&group_indices)?;
        let group_requests = group_indices
            .iter()
            .map(|&index| requests[index].clone())
            .collect::<Vec<_>>();
        let group_packets = group_indices
            .iter()
            .map(|&index| family_packets[index].to_batched())
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<
            P,
        >(
            runtime, &group_requests, &group_packets, &group_output
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} region scaled texture result count mismatch",
                    P::FAMILY_NAME
                ),
            });
        }
        for (original_index, result) in group_indices.into_iter().zip(group_results) {
            merged_results[original_index] = Some(result);
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (index, result) in merged_results.into_iter().enumerate() {
        results.push(result.ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal grouped {} region scaled texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime, requests, packets, None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

fn try_decode_fast422_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast422PacketV1>(
        runtime, requests, packets, output,
    )
}

fn try_decode_fast422_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures::<JpegFast422PacketV1>(
        runtime, requests, packets, output,
    )
}

#[cfg(target_os = "macos")]
fn requests_share_one_input(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests.iter().all(|request| {
        request.input.as_ptr() == first.input.as_ptr() && request.input.len() == first.input.len()
    })
}

#[cfg(target_os = "macos")]
fn requests_share_one_region_scaled_work(requests: &[batch::QueuedRequest]) -> bool {
    let Some(first) = requests.first() else {
        return false;
    };
    requests_share_one_input(requests)
        && requests.iter().all(|request| {
            request.fmt == first.fmt && request.backend == first.backend && request.op == first.op
        })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_packet_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    request: &batch::QueuedRequest,
    packet: &BatchedFastPacket<'_>,
) -> Result<Surface, Error> {
    let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
        return Err(Error::MetalKernel {
            message: "JPEG Metal expected a region scaled batch request".to_string(),
        });
    };
    let scaled = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled.x,
        y: scaled.y,
        w: scaled.w,
        h: scaled.h,
    };
    match packet {
        BatchedFastPacket::Fast420(packet) => try_decode_fast420_scaled_region_to_surface(
            runtime,
            decoder,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast422(packet) => try_decode_fast422_scaled_region_to_surface(
            runtime,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast444(packet, _) => try_decode_fast444_scaled_region_to_surface(
            runtime,
            decoder,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
    }
    .and_then(|surface| {
        surface.ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal repeated region scaled batch was not packet-decodable".to_string(),
        })
    })
}

#[cfg(target_os = "macos")]
fn try_decode_repeated_region_scaled_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.len() <= REGION_SCALED_BATCH_CHUNK
        || !requests_share_one_input(requests)
        || !requests
            .iter()
            .all(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }))
    {
        return Ok(None);
    }

    let decoder = CpuDecoder::new(requests[0].input.as_ref())?;
    if requests_share_one_region_scaled_work(requests) {
        let surface =
            decode_region_scaled_packet_surface(runtime, &decoder, &requests[0], &packets[0])?;
        return Ok(Some(
            (0..requests.len())
                .map(|_| Ok(surface.clone()))
                .collect::<Vec<_>>(),
        ));
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(packets.iter()) {
        results.push(decode_region_scaled_packet_surface(
            runtime, &decoder, request, packet,
        ));
    }

    Ok(Some(results))
}
