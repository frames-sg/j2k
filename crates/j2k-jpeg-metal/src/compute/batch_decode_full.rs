// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
        runtime,
        requests,
        packets,
        fast_batch_decode_mode(),
        None,
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
        runtime,
        requests,
        packets,
        fast_batch_decode_mode(),
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    decode_mode: FastBatchDecodeMode,
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let timing_enabled =
        decode_mode == FastBatchDecodeMode::Fused && P::full_rgb_batch_timing_enabled();
    let timing_total_start = timing_enabled.then(Instant::now);
    let mut timing = FastBatchTiming::default();

    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut family_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    if (!P::FULL_RGB_BATCH_SUPPORTS_RESTART && first.restart_interval_mcus() != 0)
        || first.entropy_checkpoints().is_empty()
    {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_full_rgb_batch_groups(&family_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_full_rgb_batch_to_surfaces_with_output::<P>(
            runtime,
            requests,
            &family_packets,
            decode_mode,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    if !family_packets.iter().all(|packet| {
        fast_subsampled_packets_share_full_rgb_batch_shape(first, packet, segment_count)
    }) {
        return Ok(None);
    }

    let tile_count = family_packets.len();
    let shape = full_rgb_surface_batch_shape::<P>(first, tile_count, segment_count)?;
    if timing_enabled {
        timing.accepted = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_entropy_start = timing_enabled.then(Instant::now);
    let Some(entropy_data) = batch_entropy_host_data(
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
        tile_count,
        segment_count,
        BatchEntropyLabels {
            total_len_overflow: "JPEG Metal batch entropy length overflowed",
            offset: "batch entropy offset",
            len: "batch entropy length",
        },
    )?
    else {
        return Ok(None);
    };
    if timing_enabled {
        timing.entropy_concat = timing_entropy_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_buffer_start = timing_enabled.then(Instant::now);
    let mut batch_scratch = runtime.batch_scratch()?;
    let buffers = full_rgb_surface_batch_buffers::<P>(
        runtime,
        &mut batch_scratch,
        output,
        first,
        shape,
        &entropy_data,
    )?;
    if timing_enabled {
        timing.buffer_alloc = timing_buffer_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let mut command_buffer = runtime.queue.new_command_buffer();
    let decode_pass = FullRgbDecodePass {
        runtime,
        first,
        buffers: &buffers,
        shape,
        decode_mode,
        huffman_tables: (&dc_tables, &ac_tables),
    };
    let split_scratch = encode_fast_subsampled_full_rgb_decode::<P>(
        &decode_pass,
        &mut command_buffer,
        timing_enabled,
        &mut timing,
    )?;
    Ok(Some(finish_fast_subsampled_full_rgb_batch::<P>(
        FullRgbFinishState {
            runtime,
            requests,
            first,
            command_buffer,
            batch_scratch,
            buffers: &buffers,
            shape,
            split_scratch,
        },
        FullRgbFinishTiming {
            enabled: timing_enabled,
            total_start: timing_total_start,
            timing,
            segment_count,
        },
    )?))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FullRgbSurfaceBatchShape {
    width: u32,
    height: u32,
    y_len: usize,
    chroma_len: usize,
    out_stride: usize,
    out_tile_len: usize,
    tile_count: usize,
    tile_count_u32: u32,
    total_decode_threads: u32,
    #[cfg(test)]
    total_blocks: Option<usize>,
    params: JpegFast420BatchParams,
}

#[cfg(target_os = "macos")]
struct FullRgbSurfaceBatchBuffers {
    y_plane: Buffer,
    cb_plane: Buffer,
    cr_plane: Buffer,
    out_buffer: Buffer,
    status_buffer: Buffer,
    entropy_buffer: Buffer,
    entropy_offsets_buffer: Buffer,
    entropy_lens_buffer: Buffer,
    entropy_checkpoints_buffer: Buffer,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FullRgbDecodePass<'a, P> {
    runtime: &'a MetalRuntime,
    first: &'a P,
    buffers: &'a FullRgbSurfaceBatchBuffers,
    shape: FullRgbSurfaceBatchShape,
    decode_mode: FastBatchDecodeMode,
    huffman_tables: (
        &'a [PreparedHuffmanHost; 3],
        &'a [PreparedHuffmanHost; 3],
    ),
}

#[cfg(target_os = "macos")]
struct FullRgbFinishState<'a, 'scratch, P> {
    runtime: &'a MetalRuntime,
    requests: &'a [batch::QueuedRequest],
    first: &'a P,
    command_buffer: &'a CommandBufferRef,
    batch_scratch: MutexGuard<'scratch, MetalBatchScratch>,
    buffers: &'a FullRgbSurfaceBatchBuffers,
    shape: FullRgbSurfaceBatchShape,
    split_scratch: Option<(Buffer, Buffer)>,
}

#[cfg(target_os = "macos")]
struct FullRgbFinishTiming {
    enabled: bool,
    total_start: Option<Instant>,
    timing: FastBatchTiming,
    segment_count: usize,
}

#[cfg(target_os = "macos")]
fn full_rgb_surface_batch_shape<P: FastSubsampledMetal>(
    first: &P,
    tile_count: usize,
    segment_count: usize,
) -> Result<FullRgbSurfaceBatchShape, Error> {
    let tile_count_u32 = checked_u32(tile_count, "batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch decode thread count overflowed".to_string(),
            })?,
        "batch decode thread count",
    )?;
    let width = first.dimensions().0;
    let height = first.dimensions().1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = P::chroma_height(height);
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    #[cfg(test)]
    let total_blocks = full_rgb_surface_total_blocks::<P>(first, tile_count)?;
    Ok(FullRgbSurfaceBatchShape {
        width,
        height,
        y_len,
        chroma_len,
        out_stride,
        out_tile_len,
        tile_count,
        tile_count_u32,
        total_decode_threads,
        #[cfg(test)]
        total_blocks,
        params: JpegFast420BatchParams {
            width,
            height,
            chroma_width,
            chroma_height,
            mcus_per_row: first.mcus_per_row(),
            mcu_rows: first.mcu_rows(),
            segment_count: segment_count_u32,
            tile_count: tile_count_u32,
            out_stride: checked_u32(out_stride, "batch output stride")?,
            alpha: u32::from(u8::MAX),
        },
    })
}

#[cfg(all(target_os = "macos", test))]
fn full_rgb_surface_total_blocks<P: FastSubsampledMetal>(
    first: &P,
    tile_count: usize,
) -> Result<Option<usize>, Error> {
    let Some(blocks_per_mcu) = P::FULL_RGB_BATCH_BLOCKS_PER_MCU else {
        return Ok(None);
    };
    let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
    let blocks_per_tile = total_mcus
        .checked_mul(blocks_per_mcu)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("JPEG Metal {} batch block count overflowed", P::FAMILY_NAME),
        })?;
    let total_blocks =
        blocks_per_tile
            .checked_mul(tile_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} batch total block count overflowed",
                    P::FAMILY_NAME
                ),
            })?;
    let _total_blocks_u32 =
        checked_u32(total_blocks, &format!("{} batch block count", P::FAMILY_NAME))?;
    Ok(Some(total_blocks))
}

#[cfg(target_os = "macos")]
fn full_rgb_surface_batch_buffers<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    batch_scratch: &mut MetalBatchScratch,
    output: Option<&crate::MetalBatchOutputBuffer>,
    first: &P,
    shape: FullRgbSurfaceBatchShape,
    entropy_data: &BatchEntropyHostData,
) -> Result<FullRgbSurfaceBatchBuffers, Error> {
    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.y,
        shape.y_len * shape.tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.cb,
        shape.chroma_len * shape.tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.cr,
        shape.chroma_len * shape.tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first.dimensions(),
        shape.tile_count,
        shape.out_stride,
        shape.out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); shape.total_decode_threads as usize];
    let checkpoint_hosts = entropy_checkpoint_hosts(&entropy_data.checkpoints)?;
    Ok(FullRgbSurfaceBatchBuffers {
        y_plane,
        cb_plane,
        cr_plane,
        out_buffer,
        status_buffer: batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::FULL_BATCH_KEYS.status,
            &statuses,
        ),
        entropy_buffer: batch_scratch.shared_buffer_with_bytes(
            &runtime.device,
            P::FULL_BATCH_KEYS.entropy,
            &entropy_data.bytes,
        ),
        entropy_offsets_buffer: batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::FULL_BATCH_KEYS.entropy_offsets,
            &entropy_data.offsets,
        ),
        entropy_lens_buffer: batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::FULL_BATCH_KEYS.entropy_lens,
            &entropy_data.lens,
        ),
        entropy_checkpoints_buffer: batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::FULL_BATCH_KEYS.entropy_checkpoints,
            &checkpoint_hosts,
        ),
    })
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_full_rgb_decode<'a, P: FastSubsampledMetal>(
    pass: &FullRgbDecodePass<'a, P>,
    command_buffer: &mut &'a CommandBufferRef,
    timing_enabled: bool,
    timing: &mut FastBatchTiming,
) -> Result<Option<(Buffer, Buffer)>, Error> {
    match pass.decode_mode {
        FastBatchDecodeMode::Fused => {
            let timing_encode_start = timing_enabled.then(Instant::now);
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(pass.runtime);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                decoder_encoder,
                &FastDecodeEntropyInputs {
                    entropy_buffer: &pass.buffers.entropy_buffer,
                    planes: [
                        &pass.buffers.y_plane,
                        &pass.buffers.cb_plane,
                        &pass.buffers.cr_plane,
                    ],
                    params: &pass.shape.params,
                    quants: [
                        pass.first.y_quant(),
                        pass.first.cb_quant(),
                        pass.first.cr_quant(),
                    ],
                    dc_tables: pass.huffman_tables.0,
                    ac_tables: pass.huffman_tables.1,
                    slot14_buffer: &pass.buffers.entropy_offsets_buffer,
                    slot15_buffer: &pass.buffers.entropy_lens_buffer,
                    slot16_buffer: &pass.buffers.status_buffer,
                },
            );
            decoder_encoder.set_buffer(17, Some(&pass.buffers.entropy_checkpoints_buffer), 0);
            dispatch_1d_pipeline(
                decoder_encoder,
                decode_pipeline,
                pass.shape.total_decode_threads,
            );
            decoder_encoder.end_encoding();
            if timing_enabled {
                timing.encode_decode = timing_encode_start
                    .expect("timing start is set when timing is enabled")
                    .elapsed();
                command_buffer.commit();
                let timing_wait_start = Instant::now();
                let completed =
                    std::mem::replace(command_buffer, pass.runtime.queue.new_command_buffer());
                wait_for_completion_jpeg(completed)?;
                timing.wait_decode = timing_wait_start.elapsed();
            }
            Ok(None)
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => encode_fast_subsampled_full_rgb_split_decode::<P>(
            pass.runtime,
            command_buffer,
            pass.first,
            pass.buffers,
            pass.shape,
            pass.huffman_tables,
        ),
    }
}

#[cfg(all(target_os = "macos", test))]
fn encode_fast_subsampled_full_rgb_split_decode<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &P,
    buffers: &FullRgbSurfaceBatchBuffers,
    shape: FullRgbSurfaceBatchShape,
    huffman_tables: (&[PreparedHuffmanHost; 3], &[PreparedHuffmanHost; 3]),
) -> Result<Option<(Buffer, Buffer)>, Error> {
    let Some((split, total_blocks)) =
        P::split_coeff_idct_pipelines(runtime).zip(shape.total_blocks)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "JPEG Metal {} batch split coeff/IDCT decode mode is unsupported",
                P::FAMILY_NAME
            ),
        });
    };
    let coeff_bytes = total_blocks
        .checked_mul(64)
        .and_then(|bytes| bytes.checked_mul(size_of::<i16>()))
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal {} batch coefficient scratch overflowed",
                P::FAMILY_NAME
            ),
        })?;
    let idct_component_depth = shape.tile_count_u32.checked_mul(6).ok_or_else(|| {
        Error::MetalKernel {
            message: format!(
                "JPEG Metal {} batch IDCT dispatch overflowed",
                P::FAMILY_NAME
            ),
        }
    })?;
    let coeff_blocks = runtime
        .device
        .new_buffer(coeff_bytes as u64, MTLResourceOptions::StorageModePrivate);
    let dc_only_flags = runtime
        .device
        .new_buffer(total_blocks as u64, MTLResourceOptions::StorageModePrivate);

    encode_split_coeff_idct_passes(SplitCoeffIdctPasses {
        command_buffer,
        pipelines: split,
        params: &shape.params,
        quants: [first.y_quant(), first.cb_quant(), first.cr_quant()],
        dc_tables: huffman_tables.0,
        ac_tables: huffman_tables.1,
        entropy: (
            &buffers.entropy_buffer,
            &buffers.entropy_offsets_buffer,
            &buffers.entropy_lens_buffer,
            &buffers.entropy_checkpoints_buffer,
        ),
        status_buffer: &buffers.status_buffer,
        planes: [&buffers.y_plane, &buffers.cb_plane, &buffers.cr_plane],
        scratch: (&coeff_blocks, &dc_only_flags),
        total_decode_threads: shape.total_decode_threads,
        idct_grid: (
            first.mcus_per_row(),
            first.mcu_rows(),
            idct_component_depth,
        ),
    });
    Ok(Some((coeff_blocks, dc_only_flags)))
}

#[cfg(target_os = "macos")]
fn finish_fast_subsampled_full_rgb_batch<P: FastSubsampledMetal>(
    state: FullRgbFinishState<'_, '_, P>,
    mut timing: FullRgbFinishTiming,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    let FullRgbFinishState {
        runtime,
        requests,
        first,
        command_buffer,
        batch_scratch,
        buffers,
        shape,
        split_scratch,
    } = state;
    let timing_pack_encode_start = timing.enabled.then(Instant::now);
    let pack_pipeline = P::pack_full_rgb_batch_pipeline(runtime);
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420BatchParams>(
        pack_encoder,
        [
            Some(&buffers.y_plane),
            Some(&buffers.cb_plane),
            Some(&buffers.cr_plane),
        ],
        &buffers.out_buffer,
        &shape.params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        pack_pipeline,
        (
            packed_pair_extent(shape.width),
            P::packed_height_extent(shape.height),
            shape.tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();
    if timing.enabled {
        timing.timing.encode_pack = timing_pack_encode_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    command_buffer.commit();
    if timing.enabled {
        let timing_wait_start = Instant::now();
        wait_for_completion_jpeg(command_buffer)?;
        timing.timing.wait_pack = timing_wait_start.elapsed();
        timing.timing.total = timing
            .total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
        timing.timing.log(
            P::FULL_RGB_BATCH_TIMING_TAG,
            "fused-stages",
            shape.tile_count,
            first.dimensions(),
            timing.segment_count,
        );
    } else {
        wait_for_completion_jpeg(command_buffer)?;
    }
    drop(split_scratch);
    drop(batch_scratch);

    if let Some(results) =
        surface_batch_error_results(requests, &buffers.status_buffer, shape.total_decode_threads)?
    {
        return Ok(results);
    }
    Ok(surface_batch_success_results(
        &buffers.out_buffer,
        first.dimensions(),
        PixelFormat::Rgb8,
        requests.len(),
        shape.out_tile_len,
    ))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_full_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    decode_mode: FastBatchDecodeMode,
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for packet in family_packets {
            let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions().1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                packet.dimensions(),
                requests.len(),
                out_stride,
                out_tile_len,
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
            .map(|&index| family_packets[index].to_batched())
            .collect::<Vec<_>>();

        let Some(group_results) =
            try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output::<P>(
                runtime,
                &group_requests,
                &group_packets,
                decode_mode,
                None,
            )?
        else {
            return Ok(None);
        };

        if let Some(output) = output {
            let Some(&first_group_index) = group_indices.first() else {
                continue;
            };
            let packet = family_packets[first_group_index];
            let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions().1 as usize;
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                packet.dimensions(),
                out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: format!(
                        "JPEG Metal grouped {} buffer result count mismatch",
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
                "JPEG Metal grouped {} buffer result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast_subsampled_full_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
    decode_mode: FastBatchDecodeMode,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut family_packets = Vec::with_capacity(packets.len());
    let mut family_modes = Vec::with_capacity(packets.len());
    let mut family_mode = None;
    for packet in packets {
        let Some(packet_mode) = P::texture_plane_mode_from_batched(packet) else {
            return Ok(None);
        };
        if let Some(previous_mode) = family_mode.replace(packet_mode) {
            if previous_mode != packet_mode {
                return Ok(None);
            }
        }
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
        family_modes.push(packet_mode);
    }

    let Some(first) = family_packets.first().copied() else {
        return Ok(None);
    };
    if (!P::FULL_RGB_BATCH_SUPPORTS_RESTART && first.restart_interval_mcus() != 0)
        || first.entropy_checkpoints().is_empty()
    {
        return Ok(None);
    }

    let Some(groups) = fast_subsampled_full_rgb_batch_groups(&family_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            requests,
            &family_packets,
            &family_modes,
            output,
            decode_mode,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let shape = full_rgba_texture_batch_shape::<P>(
        first,
        tile_count,
        segment_count,
        family_mode.unwrap_or(PlaneMode::YCbCr),
    )?;
    validate_rgba_texture_batch_output(output, first.dimensions(), tile_count, shape.out_tile_len)?;

    #[cfg(test)]
    let total_blocks = full_rgba_texture_total_blocks::<P>(shape.total_mcus, shape.tile_count)?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: P::TEXTURE_KEYS.entropy,
            offsets: P::TEXTURE_KEYS.entropy_offsets,
            lens: P::TEXTURE_KEYS.entropy_lens,
            checkpoints: P::TEXTURE_KEYS.entropy_checkpoints,
        },
        family_packets.iter().map(|packet| packet.entropy_bytes()),
        family_packets
            .iter()
            .map(|packet| packet.entropy_checkpoints()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    if decode_mode == FastBatchDecodeMode::Fused {
        return Ok(Some(
            decode_fast_subsampled_full_rgba_fused_texture_batch::<P>(
                FullRgbaTextureBatchCtx {
                    runtime,
                    requests,
                    first,
                    output,
                    batch_scratch,
                    entropy_buffers: &entropy_buffers,
                    shape,
                },
            )?,
        ));
    }

    Ok(Some(
        decode_fast_subsampled_full_rgba_staged_texture_batch::<P>(
            FullRgbaTextureBatchCtx {
                runtime,
                requests,
                first,
                output,
                batch_scratch,
                entropy_buffers: &entropy_buffers,
                shape,
            },
            decode_mode,
            #[cfg(test)]
            total_blocks,
        )?,
    ))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FullRgbaTextureBatchShape {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    y_len: usize,
    chroma_len: usize,
    out_tile_len: usize,
    total_mcus: usize,
    mcu_threads: Option<u32>,
    tile_count: usize,
    #[cfg(test)]
    tile_count_u32: u32,
    segment_count_u32: u32,
    total_decode_threads: u32,
    params: JpegFast420BatchParams,
    mode: PlaneMode,
}

#[cfg(target_os = "macos")]
struct FullRgbaTextureBatchCtx<'a, 'scratch, P> {
    runtime: &'a MetalRuntime,
    requests: &'a [batch::QueuedRequest],
    first: &'a P,
    output: &'a crate::MetalBatchTextureOutput,
    batch_scratch: MutexGuard<'scratch, MetalBatchScratch>,
    entropy_buffers: &'a BatchEntropyBuffers,
    shape: FullRgbaTextureBatchShape,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct FullRgbaTextureDecodeTiles<'a, P> {
    runtime: &'a MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    output: &'a crate::MetalBatchTextureOutput,
    first: &'a P,
    entropy_buffers: &'a BatchEntropyBuffers,
    status_buffer: &'a Buffer,
    boundary_buffers: (&'a Buffer, &'a Buffer),
    vertical_buffers: Option<&'a (Buffer, Buffer)>,
    shape: FullRgbaTextureBatchShape,
    tile_index_ctx: &'a str,
}

#[cfg(all(target_os = "macos", test))]
struct FullRgbaSplitDecodePass<'a, P> {
    runtime: &'a MetalRuntime,
    command_buffer: &'a CommandBufferRef,
    first: &'a P,
    batch_scratch: &'a mut MetalBatchScratch,
    entropy_buffers: &'a BatchEntropyBuffers,
    status_buffer: &'a Buffer,
    planes: [&'a Buffer; 3],
    shape: FullRgbaTextureBatchShape,
    total_blocks: Option<usize>,
    huffman_tables: (
        &'a [PreparedHuffmanHost; 3],
        &'a [PreparedHuffmanHost; 3],
    ),
}

#[cfg(target_os = "macos")]
fn full_rgba_texture_batch_shape<P: FastSubsampledMetal>(
    first: &P,
    tile_count: usize,
    segment_count: usize,
    mode: PlaneMode,
) -> Result<FullRgbaTextureBatchShape, Error> {
    let tile_count_u32 = checked_u32(
        tile_count,
        &format!("{} texture batch tile count", P::FAMILY_NAME),
    )?;
    let segment_count_u32 = checked_u32(
        segment_count,
        &format!("{} texture batch segment count", P::FAMILY_NAME),
    )?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch decode thread count overflowed",
                    P::FAMILY_NAME
                ),
            })?,
        &format!("{} texture batch decode thread count", P::FAMILY_NAME),
    )?;
    let width = first.dimensions().0;
    let height = first.dimensions().1;
    let chroma_width = width.div_ceil(2);
    let chroma_height = P::chroma_height(height);
    let y_len = width as usize * height as usize;
    let chroma_len = chroma_width as usize * chroma_height as usize;
    let out_stride = width as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
    let mcu_threads = P::texture_mcu_dispatch_threads(total_mcus)?;
    Ok(FullRgbaTextureBatchShape {
        width,
        height,
        chroma_width,
        chroma_height,
        y_len,
        chroma_len,
        out_tile_len,
        total_mcus,
        mcu_threads,
        tile_count,
        #[cfg(test)]
        tile_count_u32,
        segment_count_u32,
        total_decode_threads,
        params: JpegFast420BatchParams {
            width,
            height,
            chroma_width,
            chroma_height,
            mcus_per_row: first.mcus_per_row(),
            mcu_rows: first.mcu_rows(),
            segment_count: segment_count_u32,
            tile_count: tile_count_u32,
            out_stride: checked_u32(
                out_stride,
                &format!("{} texture batch output stride", P::FAMILY_NAME),
            )?,
            alpha: u32::from(u8::MAX),
        },
        mode,
    })
}

#[cfg(all(target_os = "macos", test))]
fn full_rgba_texture_total_blocks<P: FastSubsampledMetal>(
    total_mcus: usize,
    tile_count: usize,
) -> Result<Option<usize>, Error> {
    let Some(blocks_per_mcu) = P::FULL_RGB_BATCH_BLOCKS_PER_MCU else {
        return Ok(None);
    };
    let blocks_per_tile = total_mcus
        .checked_mul(blocks_per_mcu)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("JPEG Metal {} texture batch block count overflowed", P::FAMILY_NAME),
        })?;
    blocks_per_tile
        .checked_mul(tile_count)
        .map(Some)
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch total block count overflowed",
                P::FAMILY_NAME
            ),
        })
}

#[cfg(target_os = "macos")]
fn full_rgba_texture_params_for_tile<P: FastSubsampledMetal>(
    first: &P,
    shape: FullRgbaTextureBatchShape,
    tile_index: usize,
    tile_index_ctx: &str,
) -> Result<JpegFast420TextureBatchParams, Error> {
    Ok(JpegFast420TextureBatchParams {
        width: shape.width,
        height: shape.height,
        chroma_width: shape.chroma_width,
        chroma_height: shape.chroma_height,
        mcus_per_row: first.mcus_per_row(),
        mcu_rows: first.mcu_rows(),
        segment_count: shape.segment_count_u32,
        tile_index: checked_u32(tile_index, tile_index_ctx)?,
        alpha: u32::from(u8::MAX),
    })
}

#[cfg(target_os = "macos")]
fn bind_full_rgba_texture_params_for_tile<P: FastSubsampledMetal>(
    encoder: &metal::ComputeCommandEncoderRef,
    buffer_index: u64,
    first: &P,
    shape: FullRgbaTextureBatchShape,
    tile_index: usize,
    tile_index_ctx: &str,
) -> Result<(), Error> {
    if P::USE_FAST444_TEXTURE_PARAMS {
        let decode_params = JpegFast444TextureBatchParams {
            width: shape.width,
            height: shape.height,
            mcus_per_row: first.mcus_per_row(),
            mcu_rows: first.mcu_rows(),
            segment_count: shape.segment_count_u32,
            tile_index: checked_u32(tile_index, tile_index_ctx)?,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(shape.mode),
        };
        encoder.set_bytes(
            buffer_index,
            size_of::<JpegFast444TextureBatchParams>() as u64,
            (&raw const decode_params).cast(),
        );
    } else {
        let decode_params =
            full_rgba_texture_params_for_tile::<P>(first, shape, tile_index, tile_index_ctx)?;
        encoder.set_bytes(
            buffer_index,
            size_of::<JpegFast420TextureBatchParams>() as u64,
            (&raw const decode_params).cast(),
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn fast_subsampled_full_texture_vertical_buffers<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    batch_scratch: &mut MetalBatchScratch,
    total_repair_records: usize,
) -> Option<(Buffer, Buffer)> {
    P::TEXTURE_VERTICAL_REPAIR.map(|spec| {
        let vertical_meta = vec![0u32; total_repair_records * spec.meta_words];
        let vertical_samples = vec![0u8; total_repair_records * spec.sample_bytes];
        (
            batch_scratch.shared_buffer_with_slice(&runtime.device, spec.meta_key, &vertical_meta),
            batch_scratch.shared_buffer_with_bytes(
                &runtime.device,
                spec.samples_key,
                &vertical_samples,
            ),
        )
    })
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_full_rgba_texture_decode_tiles<P: FastSubsampledMetal>(
    pass: &FullRgbaTextureDecodeTiles<'_, P>,
) -> Result<(), Error> {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(pass.first);
    let texture_decode_pipeline = P::rgba_texture_batch_decode_pipeline(pass.runtime);
    for index in 0..pass.shape.tile_count {
        let texture = pass.output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let decoder_encoder = pass.command_buffer.new_compute_command_encoder();
        decoder_encoder.set_compute_pipeline_state(texture_decode_pipeline);
        decoder_encoder.set_buffer(0, Some(&pass.entropy_buffers.payload), 0);
        bind_full_rgba_texture_params_for_tile::<P>(
            decoder_encoder,
            4,
            pass.first,
            pass.shape,
            index,
            pass.tile_index_ctx,
        )?;
        decoder_encoder.set_bytes(
            5,
            size_of::<[u16; 64]>() as u64,
            pass.first.y_quant().as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            6,
            size_of::<[u16; 64]>() as u64,
            pass.first.cb_quant().as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            7,
            size_of::<[u16; 64]>() as u64,
            pass.first.cr_quant().as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            8,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const dc_tables[0]).cast(),
        );
        decoder_encoder.set_bytes(
            9,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const ac_tables[0]).cast(),
        );
        decoder_encoder.set_bytes(
            10,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const dc_tables[1]).cast(),
        );
        decoder_encoder.set_bytes(
            11,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const ac_tables[1]).cast(),
        );
        decoder_encoder.set_bytes(
            12,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const dc_tables[2]).cast(),
        );
        decoder_encoder.set_bytes(
            13,
            size_of::<PreparedHuffmanHost>() as u64,
            (&raw const ac_tables[2]).cast(),
        );
        decoder_encoder.set_buffer(14, Some(&pass.entropy_buffers.offsets), 0);
        decoder_encoder.set_buffer(15, Some(&pass.entropy_buffers.lens), 0);
        decoder_encoder.set_buffer(16, Some(pass.status_buffer), 0);
        decoder_encoder.set_buffer(17, Some(&pass.entropy_buffers.checkpoints), 0);
        decoder_encoder.set_buffer(18, Some(pass.boundary_buffers.0), 0);
        decoder_encoder.set_buffer(19, Some(pass.boundary_buffers.1), 0);
        if let Some((vertical_meta_buffer, vertical_samples_buffer)) = pass.vertical_buffers {
            decoder_encoder.set_buffer(20, Some(vertical_meta_buffer), 0);
            decoder_encoder.set_buffer(21, Some(vertical_samples_buffer), 0);
        }
        decoder_encoder.set_texture(0, Some(texture));
        dispatch_1d_pipeline(
            decoder_encoder,
            texture_decode_pipeline,
            pass.shape.segment_count_u32,
        );
        decoder_encoder.end_encoding();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn encode_fast_subsampled_full_rgba_texture_boundary_passes<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    output: &crate::MetalBatchTextureOutput,
    first: &P,
    boundary_buffers: (&Buffer, &Buffer),
    shape: FullRgbaTextureBatchShape,
    tile_index_ctx: &str,
) -> Result<(), Error> {
    let Some(repair_threads) =
        P::horizontal_repair_threads(first, shape.segment_count_u32, shape.mcu_threads)
    else {
        return Ok(());
    };
    let boundary_pipeline = P::rgba_texture_boundary_pipeline(runtime);
    for index in 0..shape.tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let decode_params =
            full_rgba_texture_params_for_tile::<P>(first, shape, index, tile_index_ctx)?;
        let boundary_encoder = command_buffer.new_compute_command_encoder();
        boundary_encoder.set_compute_pipeline_state(boundary_pipeline);
        boundary_encoder.set_buffer(0, Some(boundary_buffers.0), 0);
        boundary_encoder.set_buffer(1, Some(boundary_buffers.1), 0);
        boundary_encoder.set_bytes(
            2,
            size_of::<JpegFast420TextureBatchParams>() as u64,
            (&raw const decode_params).cast(),
        );
        boundary_encoder.set_texture(0, Some(texture));
        dispatch_1d_pipeline(boundary_encoder, boundary_pipeline, repair_threads);
        boundary_encoder.end_encoding();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn decode_fast_subsampled_full_rgba_fused_texture_batch<P: FastSubsampledMetal>(
    ctx: FullRgbaTextureBatchCtx<'_, '_, P>,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let FullRgbaTextureBatchCtx {
        runtime,
        requests,
        first,
        output,
        mut batch_scratch,
        entropy_buffers,
        shape,
    } = ctx;
    // Chroma reconstruction needs neighboring samples at MCU boundaries. The
    // fused path carries same-segment boundaries in-thread, then resolves
    // cross-segment boundaries from compact shared records.
    let statuses = vec![JpegDecodeStatus::default(); shape.total_decode_threads as usize];
    let status_buffer =
        batch_scratch.shared_buffer_with_slice(&runtime.device, P::TEXTURE_KEYS.status, &statuses);
    let total_repair_records = P::texture_repair_record_count(
        shape.tile_count,
        shape.total_mcus,
        shape.total_decode_threads,
    )?;
    let boundary_meta = vec![0u32; total_repair_records * P::TEXTURE_BOUNDARY_META_WORDS];
    let boundary_samples = vec![0u8; total_repair_records * P::TEXTURE_BOUNDARY_SAMPLE_BYTES];
    let boundary_meta_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::TEXTURE_BOUNDARY_META_KEY,
        &boundary_meta,
    );
    let boundary_samples_buffer = batch_scratch.shared_buffer_with_bytes(
        &runtime.device,
        P::TEXTURE_BOUNDARY_SAMPLES_KEY,
        &boundary_samples,
    );
    let vertical_buffers =
        fast_subsampled_full_texture_vertical_buffers::<P>(runtime, &mut batch_scratch, total_repair_records);
    let tile_index_ctx = format!("{} texture batch tile index", P::FAMILY_NAME);
    let command_buffer = runtime.queue.new_command_buffer();
    let decode_tiles = FullRgbaTextureDecodeTiles {
        runtime,
        command_buffer,
        output,
        first,
        entropy_buffers,
        status_buffer: &status_buffer,
        boundary_buffers: (&boundary_meta_buffer, &boundary_samples_buffer),
        vertical_buffers: vertical_buffers.as_ref(),
        shape,
        tile_index_ctx: &tile_index_ctx,
    };
    encode_fast_subsampled_full_rgba_texture_decode_tiles::<P>(&decode_tiles)?;
    encode_fast_subsampled_full_rgba_texture_boundary_passes::<P>(
        runtime,
        command_buffer,
        output,
        first,
        (&boundary_meta_buffer, &boundary_samples_buffer),
        shape,
        &tile_index_ctx,
    )?;
    P::encode_extra_texture_repair_passes(
        runtime,
        &FastTextureRepairCtx {
            command_buffer,
            output,
            boundary_meta_buffer: &boundary_meta_buffer,
            vertical_buffers: vertical_buffers.as_ref(),
            decode_params: JpegFast420TextureBatchParams {
                width: shape.width,
                height: shape.height,
                chroma_width: shape.chroma_width,
                chroma_height: shape.chroma_height,
                mcus_per_row: first.mcus_per_row(),
                mcu_rows: first.mcu_rows(),
                segment_count: shape.segment_count_u32,
                tile_index: 0,
                alpha: u32::from(u8::MAX),
            },
            tile_count: shape.tile_count,
            mcu_threads: shape.mcu_threads,
            tile_index_ctx: &tile_index_ctx,
        },
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);
    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(results);
    }
    texture_batch_success_results(output, first.dimensions(), requests.len())
}

#[cfg(target_os = "macos")]
fn decode_fast_subsampled_full_rgba_staged_texture_batch<P: FastSubsampledMetal>(
    ctx: FullRgbaTextureBatchCtx<'_, '_, P>,
    decode_mode: FastBatchDecodeMode,
    #[cfg(test)] total_blocks: Option<usize>,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let FullRgbaTextureBatchCtx {
        runtime,
        requests,
        first,
        output,
        mut batch_scratch,
        entropy_buffers,
        shape,
    } = ctx;
    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::TEXTURE_KEYS.y,
        shape.y_len * shape.tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::TEXTURE_KEYS.cb,
        shape.chroma_len * shape.tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::TEXTURE_KEYS.cr,
        shape.chroma_len * shape.tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); shape.total_decode_threads as usize];
    let status_buffer =
        batch_scratch.shared_buffer_with_slice(&runtime.device, P::TEXTURE_KEYS.status, &statuses);
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let command_buffer = runtime.queue.new_command_buffer();
    match decode_mode {
        FastBatchDecodeMode::Fused => {
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(runtime);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                decoder_encoder,
                &FastDecodeEntropyInputs {
                    entropy_buffer: &entropy_buffers.payload,
                    planes: [&y_plane, &cb_plane, &cr_plane],
                    params: &shape.params,
                    quants: [first.y_quant(), first.cb_quant(), first.cr_quant()],
                    dc_tables: &dc_tables,
                    ac_tables: &ac_tables,
                    slot14_buffer: &entropy_buffers.offsets,
                    slot15_buffer: &entropy_buffers.lens,
                    slot16_buffer: &status_buffer,
                },
            );
            decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
            dispatch_1d_pipeline(decoder_encoder, decode_pipeline, shape.total_decode_threads);
            decoder_encoder.end_encoding();
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => {
            let mut split_pass = FullRgbaSplitDecodePass {
                runtime,
                command_buffer,
                first,
                batch_scratch: &mut batch_scratch,
                entropy_buffers,
                status_buffer: &status_buffer,
                planes: [&y_plane, &cb_plane, &cr_plane],
                shape,
                total_blocks,
                huffman_tables: (&dc_tables, &ac_tables),
            };
            encode_fast_subsampled_full_rgba_split_decode::<P>(&mut split_pass)?;
        }
    }

    let pack_params = JpegTexturePackBatchParams {
        width: shape.width,
        height: shape.height,
        chroma_width: shape.chroma_width,
        chroma_height: shape.chroma_height,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
        mode: MODE_YCBCR,
    };
    dispatch_rgba_texture_pack(
        command_buffer,
        P::pack_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        pack_params,
        shape.tile_count,
        (
            packed_pair_extent(shape.width),
            P::packed_height_extent(shape.height),
        ),
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);
    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, shape.total_decode_threads)?
    {
        return Ok(results);
    }
    texture_batch_success_results(output, first.dimensions(), requests.len())
}

#[cfg(all(target_os = "macos", test))]
fn encode_fast_subsampled_full_rgba_split_decode<P: FastSubsampledMetal>(
    pass: &mut FullRgbaSplitDecodePass<'_, P>,
) -> Result<(), Error> {
    let Some((split, total_blocks)) =
        P::split_coeff_idct_pipelines(pass.runtime).zip(pass.total_blocks)
    else {
        return Err(Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch split coeff/IDCT decode mode is unsupported",
                P::FAMILY_NAME
            ),
        });
    };
    let coeff_bytes = total_blocks
        .checked_mul(64)
        .and_then(|bytes| bytes.checked_mul(size_of::<i16>()))
        .ok_or_else(|| Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch coefficient scratch overflowed",
                P::FAMILY_NAME
            ),
        })?;
    let idct_component_depth = pass.shape.tile_count_u32.checked_mul(6).ok_or_else(|| {
        Error::MetalKernel {
            message: format!(
                "JPEG Metal {} texture batch IDCT dispatch overflowed",
                P::FAMILY_NAME
            ),
        }
    })?;
    let coeff_blocks = pass.batch_scratch.private_buffer(
        &pass.runtime.device,
        P::SPLIT_TEXTURE_SCRATCH_KEYS.0,
        coeff_bytes,
    );
    let dc_only_flags = pass.batch_scratch.private_buffer(
        &pass.runtime.device,
        P::SPLIT_TEXTURE_SCRATCH_KEYS.1,
        total_blocks,
    );

    encode_split_coeff_idct_passes(SplitCoeffIdctPasses {
        command_buffer: pass.command_buffer,
        pipelines: split,
        params: &pass.shape.params,
        quants: [
            pass.first.y_quant(),
            pass.first.cb_quant(),
            pass.first.cr_quant(),
        ],
        dc_tables: pass.huffman_tables.0,
        ac_tables: pass.huffman_tables.1,
        entropy: (
            &pass.entropy_buffers.payload,
            &pass.entropy_buffers.offsets,
            &pass.entropy_buffers.lens,
            &pass.entropy_buffers.checkpoints,
        ),
        status_buffer: pass.status_buffer,
        planes: pass.planes,
        scratch: (&coeff_blocks, &dc_only_flags),
        total_decode_threads: pass.shape.total_decode_threads,
        idct_grid: (
            pass.first.mcus_per_row(),
            pass.first.mcu_rows(),
            idct_component_depth,
        ),
    });
    Ok(())
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    family_modes: &[PlaneMode],
    output: &crate::MetalBatchTextureOutput,
    decode_mode: FastBatchDecodeMode,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for packet in family_packets {
        let out_stride = packet.dimensions().0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions().1 as usize;
        validate_rgba_texture_batch_output(
            output,
            packet.dimensions(),
            requests.len(),
            out_tile_len,
        )?;
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
            .map(|&index| family_packets[index].to_batched_with_texture_mode(family_modes[index]))
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            &group_requests,
            &group_packets,
            &group_output,
            decode_mode,
        )?
        else {
            return Ok(None);
        };
        if group_results.len() != group_indices.len() {
            return Err(Error::MetalKernel {
                message: format!(
                    "JPEG Metal grouped {} texture result count mismatch",
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
                "JPEG Metal grouped {} texture result for tile {index} was missing",
                P::FAMILY_NAME
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_full_rgb_batch_to_surfaces_with_output(runtime, requests, packets, None)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces_into_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_full_rgb_batch_to_surfaces_with_output(runtime, requests, packets, Some(output))
}

#[cfg(target_os = "macos")]
fn fast444_full_region_scaled_requests(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Option<Vec<batch::QueuedRequest>> {
    if requests.is_empty() || requests.len() != packets.len() {
        return None;
    }

    let mut region_requests = Vec::with_capacity(requests.len());
    for (request, packet) in requests.iter().zip(packets) {
        if request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8 {
            return None;
        }
        let BatchedFastPacket::Fast444(packet, _) = packet else {
            return None;
        };
        let mut request = request.clone();
        request.op = batch::BatchOp::RegionScaled {
            roi: Rect::full(packet.dimensions),
            scale: j2k_core::Downscale::None,
        };
        region_requests.push(request);
    }
    Some(region_requests)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(region_requests) = fast444_full_region_scaled_requests(requests, packets) else {
        return Ok(None);
    };
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<
        JpegFast444PacketV1,
    >(runtime, &region_requests, packets, output)
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let decode_mode = fast_batch_decode_mode();
    if decode_mode == FastBatchDecodeMode::Fused {
        if let Some(results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<
            JpegFast444PacketV1,
        >(
            runtime,
            requests,
            packets,
            output,
            decode_mode,
        )? {
            return Ok(Some(results));
        }
    }

    let Some(region_requests) = fast444_full_region_scaled_requests(requests, packets) else {
        return Ok(None);
    };
    try_decode_fast444_region_scaled_rgba_batch_to_textures(
        runtime,
        &region_requests,
        packets,
        output,
    )
}
