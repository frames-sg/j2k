// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{mem::size_of, sync::MutexGuard};

use super::super::{
    batch, batch_entropy_buffers, bind_fast_decode_entropy_inputs, checked_u32,
    commit_and_wait_jpeg, dispatch_1d_pipeline, dispatch_rgba_texture_pack,
    fast_packet_huffman_tables, fast_subsampled_full_rgb_batch_groups, packed_pair_extent,
    plane_mode_to_u32, texture_batch_error_results, texture_batch_success_results,
    validate_rgba_texture_batch_output, BatchEntropyBufferKeys, BatchEntropyBuffers,
    BatchedFastPacket, Buffer, CommandBufferRef, Error, FastBatchDecodeMode,
    FastDecodeEntropyInputs, FastSubsampledMetal, FastTextureRepairCtx, JpegDecodeStatus,
    JpegFast420BatchParams, JpegFast420TextureBatchParams, JpegFast444TextureBatchParams,
    JpegTexturePackBatchParams, MetalBatchScratch, MetalRuntime, PixelFormat, PlaneMode,
    PreparedHuffmanHost, MODE_YCBCR,
};
#[cfg(test)]
use super::super::{encode_split_coeff_idct_passes, SplitCoeffIdctPasses};
use super::texture_grouped::try_decode_grouped_fast_subsampled_full_rgba_batch_to_textures;

#[cfg(target_os = "macos")]
#[expect(
    clippy::too_many_lines,
    reason = "ordered Metal texture command and resource lifetime"
)]
pub(in crate::compute) fn try_decode_fast_subsampled_full_rgba_batch_to_textures<
    P: FastSubsampledMetal,
>(
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
            decode_fast_subsampled_full_rgba_fused_texture_batch::<P>(FullRgbaTextureBatchCtx {
                runtime,
                requests,
                first,
                output,
                batch_scratch,
                entropy_buffers: &entropy_buffers,
                shape,
            })?,
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
    huffman_tables: (&'a [PreparedHuffmanHost; 3], &'a [PreparedHuffmanHost; 3]),
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
    let blocks_per_tile =
        total_mcus
            .checked_mul(blocks_per_mcu)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch block count overflowed",
                    P::FAMILY_NAME
                ),
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
        let texture = pass
            .output
            .texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
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
        let texture = output
            .texture_trusted(index)
            .ok_or_else(|| Error::MetalKernel {
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
    let vertical_buffers = fast_subsampled_full_texture_vertical_buffers::<P>(
        runtime,
        &mut batch_scratch,
        total_repair_records,
    );
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
#[expect(
    clippy::similar_names,
    reason = "Cb and Cr are normative JPEG component names"
)]
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
    let idct_component_depth =
        pass.shape
            .tile_count_u32
            .checked_mul(6)
            .ok_or_else(|| Error::MetalKernel {
                message: format!(
                    "JPEG Metal {} texture batch IDCT dispatch overflowed",
                    P::FAMILY_NAME
                ),
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
