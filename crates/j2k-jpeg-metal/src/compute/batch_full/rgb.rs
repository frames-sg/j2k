// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
use std::mem::size_of;
use std::{sync::MutexGuard, time::Instant};

use super::super::{
    batch, batch_entropy_host_data, batch_output_buffer_or_new, bind_fast_decode_entropy_inputs,
    bind_three_plane_pack, checked_u32, copy_grouped_surfaces_to_output, dispatch_1d_pipeline,
    dispatch_3d_pipeline, entropy_checkpoint_hosts, fast_batch_decode_mode,
    fast_packet_huffman_tables, fast_subsampled_full_rgb_batch_groups,
    fast_subsampled_packets_share_full_rgb_batch_shape, packed_pair_extent,
    surface_batch_error_results, surface_batch_success_results, wait_for_completion_jpeg,
    BatchEntropyHostData, BatchEntropyLabels, BatchedFastPacket, Buffer, CommandBufferRef, Error,
    FastBatchDecodeMode, FastBatchTiming, FastDecodeEntropyInputs, FastSubsampledMetal,
    JpegDecodeStatus, JpegFast420BatchParams, MetalBatchScratch, MetalRuntime, PixelFormat,
    PreparedHuffmanHost, Surface,
};
#[cfg(test)]
use super::super::{encode_split_coeff_idct_passes, MTLResourceOptions, SplitCoeffIdctPasses};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces<
    P: FastSubsampledMetal,
>(
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
pub(in crate::compute) fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output<
    P: FastSubsampledMetal,
>(
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
pub(in crate::compute) fn try_decode_fast_subsampled_full_rgb_batch_to_surfaces_with_mode_and_output<
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
        output,
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
    huffman_tables: (&'a [PreparedHuffmanHost; 3], &'a [PreparedHuffmanHost; 3]),
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
    let blocks_per_tile =
        total_mcus
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
    let _total_blocks_u32 = checked_u32(
        total_blocks,
        &format!("{} batch block count", P::FAMILY_NAME),
    )?;
    Ok(Some(total_blocks))
}

#[cfg(target_os = "macos")]
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
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
    let idct_component_depth =
        shape
            .tile_count_u32
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
        idct_grid: (first.mcus_per_row(), first.mcu_rows(), idct_component_depth),
    });
    Ok(Some((coeff_blocks, dc_only_flags)))
}

#[cfg(target_os = "macos")]
fn finish_fast_subsampled_full_rgb_batch<P: FastSubsampledMetal>(
    state: FullRgbFinishState<'_, '_, P>,
    mut timing: FullRgbFinishTiming,
    output: Option<&crate::MetalBatchOutputBuffer>,
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
        output,
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
