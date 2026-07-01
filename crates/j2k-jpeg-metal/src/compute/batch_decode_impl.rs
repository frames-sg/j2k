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
    #[cfg_attr(not(test), allow(unused_variables))]
    let total_blocks = match P::FULL_RGB_BATCH_BLOCKS_PER_MCU {
        Some(blocks_per_mcu) => {
            let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
            let blocks_per_tile =
                total_mcus
                    .checked_mul(blocks_per_mcu)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} batch block count overflowed",
                            P::FAMILY_NAME
                        ),
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
            let _total_blocks_u32 = checked_u32(
                total_blocks,
                &format!("{} batch block count", P::FAMILY_NAME),
            )?;
            Some(total_blocks)
        }
        None => None,
    };

    let params = JpegFast420BatchParams {
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
    };
    if timing_enabled {
        timing.accepted = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_entropy_start = timing_enabled.then(Instant::now);
    let total_entropy_len = family_packets
        .iter()
        .map(|packet| packet.entropy_bytes().len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch entropy length overflowed".to_string(),
        })?;
    if total_entropy_len == 0 {
        return Ok(None);
    }

    let mut entropy_bytes = Vec::with_capacity(total_entropy_len);
    let mut entropy_offsets = Vec::with_capacity(tile_count);
    let mut entropy_lens = Vec::with_capacity(tile_count);
    let mut entropy_checkpoints = Vec::with_capacity(tile_count * segment_count);
    for packet in &family_packets {
        entropy_offsets.push(checked_u32(entropy_bytes.len(), "batch entropy offset")?);
        entropy_lens.push(checked_u32(
            packet.entropy_bytes().len(),
            "batch entropy length",
        )?);
        entropy_bytes.extend_from_slice(packet.entropy_bytes());
        entropy_checkpoints.extend(packet.entropy_checkpoints().iter().copied());
    }
    if timing_enabled {
        timing.entropy_concat = timing_entropy_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let timing_buffer_start = timing_enabled.then(Instant::now);
    let mut batch_scratch = runtime.batch_scratch()?;
    let y_plane =
        batch_scratch.private_buffer(&runtime.device, P::FULL_BATCH_KEYS.y, y_len * tile_count);
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.cb,
        chroma_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::FULL_BATCH_KEYS.cr,
        chroma_len * tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first.dimensions(),
        tile_count,
        out_stride,
        out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let checkpoint_hosts = entropy_checkpoint_hosts(&entropy_checkpoints)?;
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.status,
        &statuses,
    );
    let entropy_buffer = batch_scratch.shared_buffer_with_bytes(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy,
        &entropy_bytes,
    );
    let entropy_offsets_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy_offsets,
        &entropy_offsets,
    );
    let entropy_lens_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy_lens,
        &entropy_lens,
    );
    let entropy_checkpoints_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::FULL_BATCH_KEYS.entropy_checkpoints,
        &checkpoint_hosts,
    );
    if timing_enabled {
        timing.buffer_alloc = timing_buffer_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let mut command_buffer = runtime.queue.new_command_buffer();
    #[cfg(test)]
    let mut split_scratch: Option<(Buffer, Buffer)> = None;
    match decode_mode {
        FastBatchDecodeMode::Fused => {
            let timing_encode_start = timing_enabled.then(Instant::now);
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(runtime);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                decoder_encoder,
                &entropy_buffer,
                [&y_plane, &cb_plane, &cr_plane],
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                &entropy_offsets_buffer,
                &entropy_lens_buffer,
                &status_buffer,
            );
            decoder_encoder.set_buffer(17, Some(&entropy_checkpoints_buffer), 0);
            dispatch_1d_pipeline(decoder_encoder, decode_pipeline, total_decode_threads);
            decoder_encoder.end_encoding();
            if timing_enabled {
                timing.encode_decode = timing_encode_start
                    .expect("timing start is set when timing is enabled")
                    .elapsed();
                command_buffer.commit();
                let timing_wait_start = Instant::now();
                wait_for_completion_jpeg(command_buffer)?;
                timing.wait_decode = timing_wait_start.elapsed();
                command_buffer = runtime.queue.new_command_buffer();
            }
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => {
            let Some((split, total_blocks)) =
                P::split_coeff_idct_pipelines(runtime).zip(total_blocks)
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
            let idct_component_depth =
                tile_count_u32
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} batch IDCT dispatch overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            let coeff_blocks = runtime
                .device
                .new_buffer(coeff_bytes as u64, MTLResourceOptions::StorageModePrivate);
            let dc_only_flags = runtime
                .device
                .new_buffer(total_blocks as u64, MTLResourceOptions::StorageModePrivate);

            encode_split_coeff_idct_passes(
                command_buffer,
                split,
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                (
                    &entropy_buffer,
                    &entropy_offsets_buffer,
                    &entropy_lens_buffer,
                    &entropy_checkpoints_buffer,
                ),
                &status_buffer,
                [&y_plane, &cb_plane, &cr_plane],
                (&coeff_blocks, &dc_only_flags),
                total_decode_threads,
                (first.mcus_per_row(), first.mcu_rows(), idct_component_depth),
            );

            split_scratch = Some((coeff_blocks, dc_only_flags));
        }
    }

    let timing_pack_encode_start = timing_enabled.then(Instant::now);
    let pack_pipeline = P::pack_full_rgb_batch_pipeline(runtime);
    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(pack_pipeline);
    bind_three_plane_pack::<JpegFast420BatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        pack_pipeline,
        (
            packed_pair_extent(width),
            P::packed_height_extent(height),
            tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();
    if timing_enabled {
        timing.encode_pack = timing_pack_encode_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
    }

    command_buffer.commit();
    if timing_enabled {
        let timing_wait_start = Instant::now();
        wait_for_completion_jpeg(command_buffer)?;
        timing.wait_pack = timing_wait_start.elapsed();
        timing.total = timing_total_start
            .expect("timing start is set when timing is enabled")
            .elapsed();
        timing.log(
            P::FULL_RGB_BATCH_TIMING_TAG,
            "fused-stages",
            tile_count,
            first.dimensions(),
            segment_count,
        );
    } else {
        wait_for_completion_jpeg(command_buffer)?;
    }
    #[cfg(test)]
    drop(split_scratch);
    drop(batch_scratch);

    if let Some(status) = first_decode_error_status(&status_buffer, total_decode_threads) {
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            let decoder = CpuDecoder::new(request.input.as_ref())?;
            results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
        }
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first.dimensions(),
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
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
        return try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures::<P>(
            runtime,
            requests,
            &family_packets,
            output,
            decode_mode,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
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
    validate_rgba_texture_batch_output(output, first.dimensions(), tile_count, out_tile_len)?;

    let total_mcus = first.mcus_per_row() as usize * first.mcu_rows() as usize;
    let mcu_threads = P::texture_mcu_dispatch_threads(total_mcus)?;
    #[cfg(test)]
    let total_blocks = match P::FULL_RGB_BATCH_BLOCKS_PER_MCU {
        Some(blocks_per_mcu) => {
            let blocks_per_tile =
                total_mcus
                    .checked_mul(blocks_per_mcu)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} texture batch block count overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            Some(
                blocks_per_tile
                    .checked_mul(tile_count)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} texture batch total block count overflowed",
                            P::FAMILY_NAME
                        ),
                    })?,
            )
        }
        None => None,
    };

    let params = JpegFast420BatchParams {
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
    };

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

    // Chroma reconstruction needs neighboring samples at MCU boundaries (4:2:0
    // repairs both axes with per-MCU records, 4:2:2 repairs horizontal
    // boundaries per entropy segment). The fused path carries same-segment
    // boundaries in-thread and resolves cross-segment boundaries from compact
    // shared records before returning the caller-owned texture.
    if decode_mode == FastBatchDecodeMode::Fused {
        let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
        let status_buffer = batch_scratch.shared_buffer_with_slice(
            &runtime.device,
            P::TEXTURE_KEYS.status,
            &statuses,
        );
        let total_repair_records =
            P::texture_repair_record_count(tile_count, total_mcus, total_decode_threads)?;
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
        let vertical_buffers = match &P::TEXTURE_VERTICAL_REPAIR {
            Some(spec) => {
                let vertical_meta = vec![0u32; total_repair_records * spec.meta_words];
                let vertical_samples = vec![0u8; total_repair_records * spec.sample_bytes];
                let vertical_meta_buffer = batch_scratch.shared_buffer_with_slice(
                    &runtime.device,
                    spec.meta_key,
                    &vertical_meta,
                );
                let vertical_samples_buffer = batch_scratch.shared_buffer_with_bytes(
                    &runtime.device,
                    spec.samples_key,
                    &vertical_samples,
                );
                Some((vertical_meta_buffer, vertical_samples_buffer))
            }
            None => None,
        };
        let dc_tables = [
            PreparedHuffmanHost::from(first.y_dc_table()),
            PreparedHuffmanHost::from(first.cb_dc_table()),
            PreparedHuffmanHost::from(first.cr_dc_table()),
        ];
        let ac_tables = [
            PreparedHuffmanHost::from(first.y_ac_table()),
            PreparedHuffmanHost::from(first.cb_ac_table()),
            PreparedHuffmanHost::from(first.cr_ac_table()),
        ];

        let tile_index_ctx = format!("{} texture batch tile index", P::FAMILY_NAME);
        let texture_decode_pipeline = P::rgba_texture_batch_decode_pipeline(runtime);
        let command_buffer = runtime.queue.new_command_buffer();
        for index in 0..tile_count {
            let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal batch texture output slot was missing".to_string(),
            })?;
            let decode_params = JpegFast420TextureBatchParams {
                width,
                height,
                chroma_width,
                chroma_height,
                mcus_per_row: first.mcus_per_row(),
                mcu_rows: first.mcu_rows(),
                segment_count: segment_count_u32,
                tile_index: checked_u32(index, &tile_index_ctx)?,
                alpha: u32::from(u8::MAX),
            };
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(texture_decode_pipeline);
            decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
            decoder_encoder.set_bytes(
                4,
                size_of::<JpegFast420TextureBatchParams>() as u64,
                (&raw const decode_params).cast(),
            );
            decoder_encoder.set_bytes(
                5,
                size_of::<[u16; 64]>() as u64,
                first.y_quant().as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                6,
                size_of::<[u16; 64]>() as u64,
                first.cb_quant().as_ptr().cast(),
            );
            decoder_encoder.set_bytes(
                7,
                size_of::<[u16; 64]>() as u64,
                first.cr_quant().as_ptr().cast(),
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
            decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
            decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
            decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
            decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
            decoder_encoder.set_buffer(18, Some(&boundary_meta_buffer), 0);
            decoder_encoder.set_buffer(19, Some(&boundary_samples_buffer), 0);
            if let Some((vertical_meta_buffer, vertical_samples_buffer)) = &vertical_buffers {
                decoder_encoder.set_buffer(20, Some(vertical_meta_buffer), 0);
                decoder_encoder.set_buffer(21, Some(vertical_samples_buffer), 0);
            }
            decoder_encoder.set_texture(0, Some(texture));
            dispatch_1d_pipeline(decoder_encoder, texture_decode_pipeline, segment_count_u32);
            decoder_encoder.end_encoding();
        }
        if let Some(repair_threads) =
            P::horizontal_repair_threads(first, segment_count_u32, mcu_threads)
        {
            let boundary_pipeline = P::rgba_texture_boundary_pipeline(runtime);
            for index in 0..tile_count {
                let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
                    message: "JPEG Metal batch texture output slot was missing".to_string(),
                })?;
                let decode_params = JpegFast420TextureBatchParams {
                    width,
                    height,
                    chroma_width,
                    chroma_height,
                    mcus_per_row: first.mcus_per_row(),
                    mcu_rows: first.mcu_rows(),
                    segment_count: segment_count_u32,
                    tile_index: checked_u32(index, &tile_index_ctx)?,
                    alpha: u32::from(u8::MAX),
                };
                let boundary_encoder = command_buffer.new_compute_command_encoder();
                boundary_encoder.set_compute_pipeline_state(boundary_pipeline);
                boundary_encoder.set_buffer(0, Some(&boundary_meta_buffer), 0);
                boundary_encoder.set_buffer(1, Some(&boundary_samples_buffer), 0);
                boundary_encoder.set_bytes(
                    2,
                    size_of::<JpegFast420TextureBatchParams>() as u64,
                    (&raw const decode_params).cast(),
                );
                boundary_encoder.set_texture(0, Some(texture));
                dispatch_1d_pipeline(boundary_encoder, boundary_pipeline, repair_threads);
                boundary_encoder.end_encoding();
            }
        }
        P::encode_extra_texture_repair_passes(
            runtime,
            &FastTextureRepairCtx {
                command_buffer,
                output,
                boundary_meta_buffer: &boundary_meta_buffer,
                vertical_buffers: vertical_buffers.as_ref(),
                decode_params: JpegFast420TextureBatchParams {
                    width,
                    height,
                    chroma_width,
                    chroma_height,
                    mcus_per_row: first.mcus_per_row(),
                    mcu_rows: first.mcu_rows(),
                    segment_count: segment_count_u32,
                    tile_index: 0,
                    alpha: u32::from(u8::MAX),
                },
                tile_count,
                mcu_threads,
                tile_index_ctx: &tile_index_ctx,
            },
        )?;

        commit_and_wait_jpeg(command_buffer)?;
        drop(batch_scratch);

        if let Some(results) =
            texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
        {
            return Ok(Some(results));
        }

        return Ok(Some(texture_batch_success_results(
            output,
            first.dimensions(),
            requests.len(),
        )?));
    }

    let y_plane =
        batch_scratch.private_buffer(&runtime.device, P::TEXTURE_KEYS.y, y_len * tile_count);
    let cb_plane =
        batch_scratch.private_buffer(&runtime.device, P::TEXTURE_KEYS.cb, chroma_len * tile_count);
    let cr_plane =
        batch_scratch.private_buffer(&runtime.device, P::TEXTURE_KEYS.cr, chroma_len * tile_count);
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer =
        batch_scratch.shared_buffer_with_slice(&runtime.device, P::TEXTURE_KEYS.status, &statuses);
    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    match decode_mode {
        FastBatchDecodeMode::Fused => {
            let decode_pipeline = P::full_rgb_batch_decode_pipeline(runtime);
            let decoder_encoder = command_buffer.new_compute_command_encoder();
            decoder_encoder.set_compute_pipeline_state(decode_pipeline);
            bind_fast_decode_entropy_inputs::<JpegFast420BatchParams>(
                decoder_encoder,
                &entropy_buffers.payload,
                [&y_plane, &cb_plane, &cr_plane],
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                &entropy_buffers.offsets,
                &entropy_buffers.lens,
                &status_buffer,
            );
            decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
            dispatch_1d_pipeline(decoder_encoder, decode_pipeline, total_decode_threads);
            decoder_encoder.end_encoding();
        }
        #[cfg(test)]
        FastBatchDecodeMode::SplitCoeffIdct => {
            let Some((split, total_blocks)) =
                P::split_coeff_idct_pipelines(runtime).zip(total_blocks)
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
            let idct_component_depth =
                tile_count_u32
                    .checked_mul(6)
                    .ok_or_else(|| Error::MetalKernel {
                        message: format!(
                            "JPEG Metal {} texture batch IDCT dispatch overflowed",
                            P::FAMILY_NAME
                        ),
                    })?;
            let coeff_blocks = batch_scratch.private_buffer(
                &runtime.device,
                P::SPLIT_TEXTURE_SCRATCH_KEYS.0,
                coeff_bytes,
            );
            let dc_only_flags = batch_scratch.private_buffer(
                &runtime.device,
                P::SPLIT_TEXTURE_SCRATCH_KEYS.1,
                total_blocks,
            );

            encode_split_coeff_idct_passes(
                command_buffer,
                split,
                &params,
                [first.y_quant(), first.cb_quant(), first.cr_quant()],
                &dc_tables,
                &ac_tables,
                (
                    &entropy_buffers.payload,
                    &entropy_buffers.offsets,
                    &entropy_buffers.lens,
                    &entropy_buffers.checkpoints,
                ),
                &status_buffer,
                [&y_plane, &cb_plane, &cr_plane],
                (&coeff_blocks, &dc_only_flags),
                total_decode_threads,
                (first.mcus_per_row(), first.mcu_rows(), idct_component_depth),
            );
        }
    }

    let pack_params = JpegTexturePackBatchParams {
        width,
        height,
        chroma_width,
        chroma_height,
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
        tile_count,
        (packed_pair_extent(width), P::packed_height_extent(height)),
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        first.dimensions(),
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
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
            .map(|&index| family_packets[index].to_batched())
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
    try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
        runtime,
        requests,
        packets,
        Some(output),
    )
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_full_rgb_batch_groups(&fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints.len();
    if !fast444_packets.iter().all(|(packet, mode)| {
        *mode == first_mode
            && fast444_packets_share_region_scaled_batch_shape(first, packet, segment_count)
    }) {
        return Ok(None);
    }

    let tile_count = fast444_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "fast444 batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "fast444 batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast444 batch decode thread count overflowed".to_string(),
            })?,
        "fast444 batch decode thread count",
    )?;

    let width = first.dimensions.0;
    let height = first.dimensions.1;
    let out_stride = width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    let plane_len = width as usize * height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
        scaled_width: width,
        scaled_height: height,
        chroma_width: width,
        chroma_height: height,
        mcus_per_row: first.mcus_per_row,
        mcu_rows: first.mcu_rows,
        segment_count: segment_count_u32,
        tile_count: tile_count_u32,
        scale_shift: 0,
        origin_x: 0,
        origin_y: 0,
    };
    let pack_params = JpegWindowedPackBatchParams {
        src_width: width,
        src_height: height,
        chroma_width: width,
        chroma_height: height,
        src_x: 0,
        src_y: 0,
        width,
        height,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "fast444 batch output stride")?,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
        out_format: OUT_RGB,
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_full_entropy",
            offsets: "fast444_full_entropy_offsets",
            lens: "fast444_full_entropy_lens",
            checkpoints: "fast444_full_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane =
        batch_scratch.private_buffer(&runtime.device, "fast444_full_y", plane_len * tile_count);
    let cb_plane =
        batch_scratch.private_buffer(&runtime.device, "fast444_full_cb", plane_len * tile_count);
    let cr_plane =
        batch_scratch.private_buffer(&runtime.device, "fast444_full_cr", plane_len * tile_count);
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first.dimensions,
        tile_count,
        out_stride,
        out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer =
        batch_scratch.shared_buffer_with_slice(&runtime.device, "fast444_full_status", &statuses);
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [&first.y_quant, &first.cb_quant, &first.cr_quant],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_444_rgb_batch_pipeline);
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_444_rgb_batch_pipeline,
        (width, height, tile_count_u32),
    );
    pack_encoder.end_encoding();

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first.dimensions,
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_full_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (packet, _) in fast444_packets {
            let out_stride = packet.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions.1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                packet.dimensions,
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
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
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
            let (packet, _) = fast444_packets[first_group_index];
            let out_stride = packet.dimensions.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * packet.dimensions.1 as usize;
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                packet.dimensions,
                out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message: "JPEG Metal grouped fast444 buffer result count mismatch".to_string(),
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
                "JPEG Metal grouped fast444 buffer result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    if first.restart_interval_mcus != 0 || first.entropy_checkpoints.is_empty() {
        return Ok(None);
    }

    let Some(groups) = fast444_full_rgb_batch_groups(&fast444_packets) else {
        return Ok(None);
    };
    if groups.len() > 1 {
        return try_decode_grouped_fast444_full_rgba_batch_to_textures(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

    let segment_count = first.entropy_checkpoints.len();
    let tile_count = fast444_packets.len();
    let width = first.dimensions.0;
    let height = first.dimensions.1;
    let out_stride = width as usize * PixelFormat::Rgba8.bytes_per_pixel();
    let out_tile_len = out_stride * height as usize;
    validate_rgba_texture_batch_output(output, first.dimensions, tile_count, out_tile_len)?;

    let segment_count_u32 = checked_u32(segment_count, "fast444 batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal fast444 texture batch decode thread count overflowed"
                    .to_string(),
            })?,
        "fast444 texture batch decode thread count",
    )?;

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_texture_entropy",
            offsets: "fast444_texture_entropy_offsets",
            lens: "fast444_texture_entropy_lens",
            checkpoints: "fast444_texture_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_texture_status",
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    for index in 0..tile_count {
        let texture = output.texture(index).ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal batch texture output slot was missing".to_string(),
        })?;
        let decode_params = JpegFast444TextureBatchParams {
            width,
            height,
            mcus_per_row: first.mcus_per_row,
            mcu_rows: first.mcu_rows,
            segment_count: segment_count_u32,
            tile_index: checked_u32(index, "fast444 texture batch tile index")?,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(first_mode),
        };
        let decoder_encoder = command_buffer.new_compute_command_encoder();
        decoder_encoder
            .set_compute_pipeline_state(&runtime.fast444_rgba_texture_batch_decode_pipeline);
        decoder_encoder.set_buffer(0, Some(&entropy_buffers.payload), 0);
        decoder_encoder.set_bytes(
            4,
            size_of::<JpegFast444TextureBatchParams>() as u64,
            (&raw const decode_params).cast(),
        );
        decoder_encoder.set_bytes(
            5,
            size_of::<[u16; 64]>() as u64,
            first.y_quant.as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            6,
            size_of::<[u16; 64]>() as u64,
            first.cb_quant.as_ptr().cast(),
        );
        decoder_encoder.set_bytes(
            7,
            size_of::<[u16; 64]>() as u64,
            first.cr_quant.as_ptr().cast(),
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
        decoder_encoder.set_buffer(14, Some(&entropy_buffers.offsets), 0);
        decoder_encoder.set_buffer(15, Some(&entropy_buffers.lens), 0);
        decoder_encoder.set_buffer(16, Some(&status_buffer), 0);
        decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
        decoder_encoder.set_texture(0, Some(texture));
        dispatch_1d_pipeline(
            decoder_encoder,
            &runtime.fast444_rgba_texture_batch_decode_pipeline,
            segment_count_u32,
        );
        decoder_encoder.end_encoding();
    }

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        first.dimensions,
        requests.len(),
    )?))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_full_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: &crate::MetalBatchTextureOutput,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    for (packet, _) in fast444_packets {
        let out_stride = packet.dimensions.0 as usize * PixelFormat::Rgba8.bytes_per_pixel();
        let out_tile_len = out_stride * packet.dimensions.1 as usize;
        validate_rgba_texture_batch_output(
            output,
            packet.dimensions,
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
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) = try_decode_fast444_full_rgba_batch_to_textures(
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
                message: "JPEG Metal grouped fast444 texture result count mismatch".to_string(),
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
                "JPEG Metal grouped fast444 texture result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
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
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
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
fn try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
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
    if output.is_some() {
        for (request, (packet, _)) in requests.iter().zip(fast444_packets.iter().copied()) {
            let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
                return Ok(None);
            };
            let Some((out_dims, out_stride, out_tile_len)) =
                fast444_region_scaled_rgb_output_shape(packet, roi, scale)
            else {
                return Ok(None);
            };
            batch_output_buffer_or_new(
                runtime,
                output,
                out_dims,
                requests.len(),
                out_stride,
                out_tile_len,
            )?;
            first_shape.get_or_insert((out_dims, out_tile_len));
        }
    }

    let mut results = Vec::with_capacity(requests.len());
    for (request, (packet, mode)) in requests.iter().zip(fast444_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = BatchedFastPacket::Fast444(packet, mode);
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
    let Some((out_dims, out_tile_len)) = first_shape else {
        return Ok(Some(results));
    };
    let group_indices = (0..requests.len()).collect::<Vec<_>>();
    let copied = copy_grouped_surfaces_to_output(
        runtime,
        output,
        out_dims,
        out_tile_len,
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
                "JPEG Metal restart fast444 region scaled buffer result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if fast444_packets
        .iter()
        .any(|(packet, _)| packet.restart_interval_mcus != 0)
    {
        return try_decode_fast444_restart_region_scaled_rgb_batch_to_surfaces_with_output(
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
        return try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
            runtime,
            requests,
            &fast444_packets,
            output,
            groups,
        );
    }

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
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let total_decode_threads = checked_u32(
        tile_count
            .checked_mul(segment_count)
            .ok_or_else(|| Error::MetalKernel {
                message: "JPEG Metal region scaled batch decode thread count overflowed"
                    .to_string(),
            })?,
        "region scaled batch decode thread count",
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

    let out_stride =
        first_decode_params.scaled_width as usize * PixelFormat::Rgb8.bytes_per_pixel();
    let out_tile_len = out_stride * first_decode_params.scaled_height as usize;

    let plane_len =
        first_decode_params.scaled_width as usize * first_decode_params.scaled_height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
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
    };
    let pack_params = JpegWindowedPackBatchParams {
        src_width: first_decode_params.scaled_width,
        src_height: first_decode_params.scaled_height,
        chroma_width: first_decode_params.scaled_width,
        chroma_height: first_decode_params.scaled_height,
        src_x: 0,
        src_y: 0,
        width: first_decode_params.scaled_width,
        height: first_decode_params.scaled_height,
        tile_count: tile_count_u32,
        out_stride: checked_u32(out_stride, "region scaled batch output stride")?,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
        out_format: OUT_RGB,
    };

    let mut batch_scratch = runtime.batch_scratch()?;
    let Some(entropy_buffers) = batch_entropy_buffers(
        runtime,
        &mut batch_scratch,
        BatchEntropyBufferKeys {
            payload: "fast444_region_scaled_entropy",
            offsets: "fast444_region_scaled_entropy_offsets",
            lens: "fast444_region_scaled_entropy_lens",
            checkpoints: "fast444_region_scaled_entropy_checkpoints",
        },
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_bytes.as_slice()),
        fast444_packets
            .iter()
            .map(|(packet, _)| packet.entropy_checkpoints.as_slice()),
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_y",
        plane_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_cb",
        plane_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_cr",
        plane_len * tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        (
            first_decode_params.scaled_width,
            first_decode_params.scaled_height,
        ),
        tile_count,
        out_stride,
        out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_region_scaled_status",
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [&first.y_quant, &first.cb_quant, &first.cr_quant],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(&runtime.pack_444_rgb_batch_pipeline);
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        &runtime.pack_444_rgb_batch_pipeline,
        (
            first_decode_params.scaled_width,
            first_decode_params.scaled_height,
            tile_count_u32,
        ),
    );
    pack_encoder.end_encoding();

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            (
                first_decode_params.scaled_width,
                first_decode_params.scaled_height,
            ),
            PixelFormat::Rgb8,
            index * out_tile_len,
        )));
    }
    Ok(Some(results))
}

#[cfg(target_os = "macos")]
fn try_decode_grouped_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    fast444_packets: &[(&JpegFast444PacketV1, PlaneMode)],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
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
            let out_stride = out_dims.0 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            let out_tile_len = out_stride * out_dims.1 as usize;
            batch_output_buffer_or_new(
                runtime,
                Some(output),
                out_dims,
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
            .map(|&index| {
                let (packet, mode) = fast444_packets[index];
                BatchedFastPacket::Fast444(packet, mode)
            })
            .collect::<Vec<_>>();

        let Some(group_results) =
            try_decode_fast444_region_scaled_rgb_batch_to_surfaces_with_output(
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
            let (packet, _) = fast444_packets[first_group_index];
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
                out_dims.0 as usize * out_dims.1 as usize * PixelFormat::Rgb8.bytes_per_pixel();
            for (original_index, result) in copy_grouped_surfaces_to_output(
                runtime,
                output,
                out_dims,
                out_tile_len,
                &group_indices,
                group_results,
            )? {
                merged_results[original_index] = Some(result);
            }
        } else {
            if group_results.len() != group_indices.len() {
                return Err(Error::MetalKernel {
                    message:
                        "JPEG Metal grouped fast444 region scaled buffer result count mismatch"
                            .to_string(),
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
                "JPEG Metal grouped fast444 region scaled buffer result for tile {index} was missing"
            ),
        })?);
    }
    Ok(Some(results))
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
fn try_decode_fast444_region_scaled_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }

    let mut fast444_packets = Vec::with_capacity(packets.len());
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }

    let Some((first, first_mode)) = fast444_packets.first().copied() else {
        return Ok(None);
    };
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
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
    validate_rgba_texture_batch_output(output, out_dims, tile_count, out_tile_len)?;

    let plane_len =
        first_decode_params.scaled_width as usize * first_decode_params.scaled_height as usize;
    let decode_params = JpegFastRegionScaledBatchParams {
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
    };

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
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_y",
        plane_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cb",
        plane_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        "fast444_region_scaled_texture_cr",
        plane_len * tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        "fast444_region_scaled_texture_status",
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(&first.y_dc_table),
        PreparedHuffmanHost::from(&first.cb_dc_table),
        PreparedHuffmanHost::from(&first.cr_dc_table),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(&first.y_ac_table),
        PreparedHuffmanHost::from(&first.cb_ac_table),
        PreparedHuffmanHost::from(&first.cr_ac_table),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder
        .set_compute_pipeline_state(&runtime.fast444_scaled_region_batch_decode_pipeline);
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &decode_params,
        [&first.y_quant, &first.cb_quant, &first.cr_quant],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        &runtime.fast444_scaled_region_batch_decode_pipeline,
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_params = JpegTexturePackBatchParams {
        width: out_dims.0,
        height: out_dims.1,
        chroma_width: out_dims.0,
        chroma_height: out_dims.1,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
        mode: plane_mode_to_u32(first_mode),
    };
    dispatch_rgba_texture_pack(
        command_buffer,
        &runtime.pack_444_rgba_texture_pipeline,
        (&y_plane, &cb_plane, &cr_plane),
        output,
        pack_params,
        tile_count,
        out_dims,
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        out_dims,
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
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
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
    if output.is_some() {
        for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
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
                fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
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
    for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
        let decoder = CpuDecoder::new(request.input.as_ref())?;
        let batched_packet = packet.to_batched();
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
fn try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output<
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
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
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
        return Ok(None);
    };
    if family_packets
        .iter()
        .any(|packet| packet.restart_interval_mcus() != 0)
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

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled batch tile count")?;
    let segment_count_u32 = checked_u32(segment_count, "region scaled batch segment count")?;
    let Some(first_plan) = fast_subsampled_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
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
            ) != Some(first_plan)
        {
            return Ok(None);
        }
    }

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

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_KEYS.y,
        first_plan.y_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_KEYS.cb,
        first_plan.chroma_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_KEYS.cr,
        first_plan.chroma_len * tile_count,
    );
    let out_buffer = batch_output_buffer_or_new(
        runtime,
        output,
        first_plan.out_dims,
        tile_count,
        first_plan.pack_params.out_stride as usize,
        first_plan.out_tile_len,
    )?;
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_KEYS.status,
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(P::scaled_region_batch_decode_pipeline(runtime));
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &first_plan.decode_params,
        [first.y_quant(), first.cb_quant(), first.cr_quant()],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        P::scaled_region_batch_decode_pipeline(runtime),
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    let pack_encoder = command_buffer.new_compute_command_encoder();
    pack_encoder.set_compute_pipeline_state(P::pack_windowed_rgb_batch_pipeline(runtime));
    bind_three_plane_pack::<JpegWindowedPackBatchParams>(
        pack_encoder,
        [Some(&y_plane), Some(&cb_plane), Some(&cr_plane)],
        &out_buffer,
        &first_plan.pack_params,
    );
    dispatch_3d_pipeline(
        pack_encoder,
        P::pack_windowed_rgb_batch_pipeline(runtime),
        (first_plan.out_dims.0, first_plan.out_dims.1, tile_count_u32),
    );
    pack_encoder.end_encoding();

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        region_scaled_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    for index in 0..requests.len() {
        results.push(Ok(Surface::from_metal_buffer_offset(
            out_buffer.clone(),
            first_plan.out_dims,
            PixelFormat::Rgb8,
            index * first_plan.out_tile_len,
        )));
    }
    Ok(Some(results))
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
    P: FastSubsampledMetal,
>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    family_packets: &[&P],
    output: Option<&crate::MetalBatchOutputBuffer>,
    groups: Vec<Vec<usize>>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(output) = output {
        for (request, packet) in requests.iter().zip(family_packets.iter().copied()) {
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
                fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
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
            .map(|&index| family_packets[index].to_batched())
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
            let packet = family_packets[first_group_index];
            let segment_count_u32 = checked_u32(
                packet.entropy_checkpoints().len(),
                &format!(
                    "{} grouped region scaled buffer segment count",
                    P::FAMILY_NAME
                ),
            )?;
            let Some(plan) =
                fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
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
            fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
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
fn try_decode_fast_subsampled_region_scaled_rgba_batch_to_textures<P: FastSubsampledMetal>(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
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
    let batch::BatchOp::RegionScaled {
        roi: first_roi,
        scale: first_scale,
    } = requests[0].op
    else {
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

    let Some(groups) = fast_subsampled_region_scaled_batch_groups(requests, &family_packets)?
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

    let segment_count = first.entropy_checkpoints().len();
    let tile_count = family_packets.len();
    let tile_count_u32 = checked_u32(tile_count, "region scaled texture batch tile count")?;
    let segment_count_u32 =
        checked_u32(segment_count, "region scaled texture batch segment count")?;
    let Some(first_plan) = fast_subsampled_region_scaled_batch_plan(
        first,
        first_roi,
        first_scale,
        tile_count_u32,
        segment_count_u32,
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
            ) != Some(first_plan)
        {
            return Ok(None);
        }
    }

    let out_tile_len = first_plan.out_dims.0 as usize
        * first_plan.out_dims.1 as usize
        * PixelFormat::Rgba8.bytes_per_pixel();
    validate_rgba_texture_batch_output(output, first_plan.out_dims, tile_count, out_tile_len)?;

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
        tile_count,
        segment_count,
    )?
    else {
        return Ok(None);
    };

    let y_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.y,
        first_plan.y_len * tile_count,
    );
    let cb_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.cb,
        first_plan.chroma_len * tile_count,
    );
    let cr_plane = batch_scratch.private_buffer(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.cr,
        first_plan.chroma_len * tile_count,
    );
    let statuses = vec![JpegDecodeStatus::default(); total_decode_threads as usize];
    let status_buffer = batch_scratch.shared_buffer_with_slice(
        &runtime.device,
        P::REGION_SCALED_TEXTURE_KEYS.status,
        &statuses,
    );
    let dc_tables = [
        PreparedHuffmanHost::from(first.y_dc_table()),
        PreparedHuffmanHost::from(first.cb_dc_table()),
        PreparedHuffmanHost::from(first.cr_dc_table()),
    ];
    let ac_tables = [
        PreparedHuffmanHost::from(first.y_ac_table()),
        PreparedHuffmanHost::from(first.cb_ac_table()),
        PreparedHuffmanHost::from(first.cr_ac_table()),
    ];

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(P::scaled_region_batch_decode_pipeline(runtime));
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        decoder_encoder,
        &entropy_buffers.payload,
        [&y_plane, &cb_plane, &cr_plane],
        &first_plan.decode_params,
        [first.y_quant(), first.cb_quant(), first.cr_quant()],
        &dc_tables,
        &ac_tables,
        &entropy_buffers.offsets,
        &entropy_buffers.lens,
        &status_buffer,
    );
    decoder_encoder.set_buffer(17, Some(&entropy_buffers.checkpoints), 0);
    dispatch_1d_pipeline(
        decoder_encoder,
        P::scaled_region_batch_decode_pipeline(runtime),
        total_decode_threads,
    );
    decoder_encoder.end_encoding();

    dispatch_windowed_rgba_texture_pack(
        command_buffer,
        P::pack_windowed_rgba_texture_pipeline(runtime),
        (&y_plane, &cb_plane, &cr_plane),
        output,
        windowed_texture_pack_params(first_plan),
        tile_count,
        first_plan.out_dims,
    )?;

    commit_and_wait_jpeg(command_buffer)?;
    drop(batch_scratch);

    if let Some(results) =
        texture_batch_error_results(requests, &status_buffer, total_decode_threads)?
    {
        return Ok(Some(results));
    }

    Ok(Some(texture_batch_success_results(
        output,
        first_plan.out_dims,
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
    P: FastSubsampledMetal,
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
            fast_subsampled_region_scaled_batch_plan(packet, roi, scale, 1, segment_count_u32)
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

#[cfg(target_os = "macos")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn decode_full_batch_to_surfaces(
    requests: &[batch::QueuedRequest],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime(|runtime| decode_full_batch_to_surfaces_with_runtime(runtime, requests, &packets))
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_batch_to_surfaces_with_session(
    requests: &[batch::QueuedRequest],
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_full_batch_to_surfaces_with_runtime(runtime, requests, &packets)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_batch_to_surfaces_with_session_state(
    requests: &[batch::QueuedRequest],
    session: &mut crate::session::SessionState,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    let backend_session = session.backend_session()?;
    with_runtime_for_session(backend_session, |runtime| {
        decode_full_batch_to_surfaces_with_runtime(runtime, requests, &packets)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_rgb8_batch_into_output_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchOutputBuffer,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_full_rgb8_batch_into_output_with_runtime(runtime, requests, &packets, output)
    })
}

#[cfg(target_os = "macos")]
fn decode_full_rgb8_batch_into_output_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output::<
        JpegFast420PacketV1,
    >(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output::<
        JpegFast422PacketV1,
    >(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast444_full_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_full_rgb8_batch_into_textures_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchTextureOutput,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_full_rgb8_batch_into_textures_with_runtime(runtime, requests, &packets, output)
    })
}

#[cfg(target_os = "macos")]
fn decode_full_rgb8_batch_into_textures_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if let Some(results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<
        JpegFast420PacketV1,
    >(runtime, requests, packets, output, fast_batch_decode_mode())?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<
        JpegFast422PacketV1,
    >(runtime, requests, packets, output, fast_batch_decode_mode())?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast444_full_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_rgb8_batch_into_output_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchOutputBuffer,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_region_scaled_rgb8_batch_into_output_with_runtime(
            runtime, requests, &packets, output,
        )
    })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_rgb8_batch_into_output_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(results) = try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output(
        runtime, requests, packets, output,
    )? {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_rgb8_batch_into_textures_with_session(
    requests: &[batch::QueuedRequest],
    output: &crate::MetalBatchTextureOutput,
    session: &crate::MetalBackendSession,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let Some(packets) = batched_fast_packets(requests)? else {
        return Ok(None);
    };

    with_runtime_for_session(session, |runtime| {
        decode_region_scaled_rgb8_batch_into_textures_with_runtime(
            runtime, requests, &packets, output,
        )
    })
}

#[cfg(target_os = "macos")]
fn decode_region_scaled_rgb8_batch_into_textures_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    if let Some(results) =
        try_decode_fast444_region_scaled_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast420_region_scaled_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast422_region_scaled_rgba_batch_to_textures(runtime, requests, packets, output)?
    {
        return Ok(Some(results));
    }

    Ok(None)
}

#[cfg(target_os = "macos")]
fn decode_full_batch_to_surfaces_with_runtime(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<
        JpegFast420PacketV1,
    >(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) = try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<
        JpegFast422PacketV1,
    >(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast444_full_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_repeated_region_scaled_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast444_region_scaled_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast420_region_scaled_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }
    if let Some(results) =
        try_decode_fast422_region_scaled_rgb_batch_to_surfaces(runtime, requests, packets)?
    {
        return Ok(Some(results));
    }

    let mut results = Vec::with_capacity(requests.len());
    let has_region_scaled = requests
        .iter()
        .any(|request| matches!(request.op, batch::BatchOp::RegionScaled { .. }));
    let chunk_size = if has_region_scaled {
        REGION_SCALED_BATCH_CHUNK
    } else {
        requests.len().max(1)
    };
    for chunk_start in (0..requests.len()).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(requests.len());
        let command_buffer = runtime.queue.new_command_buffer();
        let mut encoded = Vec::with_capacity(chunk_end - chunk_start);
        let mut device_buffer_cache = BatchDeviceBufferCache::default();
        for index in chunk_start..chunk_end {
            let request = &requests[index];
            let packet = &packets[index];
            let item = match packet {
                BatchedFastPacket::Fast420(packet) => encode_fast_subsampled_op_batch_item(
                    runtime,
                    command_buffer,
                    &mut device_buffer_cache,
                    index,
                    *packet,
                    request.fmt,
                    request.op,
                )?,
                BatchedFastPacket::Fast422(packet) => encode_fast_subsampled_op_batch_item(
                    runtime,
                    command_buffer,
                    &mut device_buffer_cache,
                    index,
                    *packet,
                    request.fmt,
                    request.op,
                )?,
                BatchedFastPacket::Fast444(packet, mode) => match request.op {
                    batch::BatchOp::Full => encode_fast444_batch_item(
                        runtime,
                        command_buffer,
                        index,
                        packet,
                        *mode,
                        request.fmt,
                    )?,
                    batch::BatchOp::Region(roi) => encode_fast444_region_batch_item(
                        runtime,
                        command_buffer,
                        index,
                        packet,
                        *mode,
                        request.fmt,
                        roi,
                    )?,
                    batch::BatchOp::Scaled(scale) => encode_fast444_scaled_batch_item(
                        runtime,
                        command_buffer,
                        index,
                        packet,
                        *mode,
                        request.fmt,
                        scale,
                    )?,
                    batch::BatchOp::RegionScaled { roi, scale } => {
                        encode_fast444_scaled_region_batch_item(
                            runtime,
                            command_buffer,
                            &mut device_buffer_cache,
                            index,
                            packet,
                            *mode,
                            request.fmt,
                            roi,
                            scale,
                        )?
                    }
                },
            };
            encoded.push(item);
        }

        commit_and_wait_jpeg(command_buffer)?;

        for item in encoded {
            if let Some(status) =
                first_decode_error_status(&item.status_buffer, item.decode_threads)
            {
                let request = &requests[item.request_index];
                let decoder = CpuDecoder::new(request.input.as_ref())?;
                results.push(Err(decode_error_from_cpu(&decoder, request.fmt, status)));
            } else {
                results.push(Ok(item.surface));
            }
        }
    }
    Ok(Some(results))
}

