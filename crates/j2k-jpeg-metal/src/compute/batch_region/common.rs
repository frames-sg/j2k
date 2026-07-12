// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, bind_fast_decode_entropy_inputs, checked_u32, dispatch_1d_pipeline,
    fast_decode_status_error, fast_packet_huffman_tables,
    fast_subsampled_packets_share_full_rgb_batch_shape, fast_subsampled_region_scaled_batch_plan,
    new_compute_command_encoder, try_decode_fast420_scaled_region_to_surface_with_status,
    try_decode_fast422_scaled_region_to_surface,
    try_decode_fast444_scaled_region_to_surface_with_mode_and_status, BatchEntropyBuffers,
    BatchedFastPacket, Buffer, CommandBufferRef, Device, Error, FastDecodeEntropyInputs,
    FastRegionScaledMetal, FastScratchKeys, FastSubsampledMetal, JpegFast444PacketV1,
    JpegFastRegionScaledBatchParams, MetalBatchScratch, MetalRuntime, PixelFormat, PlaneMode, Rect,
    RegionScaledBatchPlan, Surface,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) fn first_region_scaled_op(
    requests: &[batch::QueuedRequest],
) -> Option<(Rect, j2k_core::Downscale)> {
    let batch::BatchOp::RegionScaled { roi, scale } = requests.first()?.op else {
        return None;
    };
    Some((roi, scale))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_region_packets<'a>(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'a>],
) -> Result<Option<Vec<(&'a JpegFast444PacketV1, PlaneMode)>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal fast444 region packet plan",
        requests,
    )?;
    let mut fast444_packets =
        budget.try_vec(packets.len(), "JPEG Metal fast444 region packet references")?;
    for packet in packets {
        let BatchedFastPacket::Fast444(packet, mode) = packet else {
            return Ok(None);
        };
        fast444_packets.push((*packet, *mode));
    }
    Ok(Some(fast444_packets))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn subsampled_region_rgb_packets<'a, P: FastRegionScaledMetal>(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'a>],
) -> Result<Option<Vec<(&'a P, PlaneMode)>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal subsampled region RGB packet plan",
        requests,
    )?;
    let mut family_packets = budget.try_vec(
        packets.len(),
        "JPEG Metal subsampled region RGB packet references",
    )?;
    for packet in packets {
        let Some(packet) = P::from_region_scaled_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }
    Ok(Some(family_packets))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn subsampled_region_texture_packets<'a, P: FastSubsampledMetal>(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'a>],
) -> Result<Option<Vec<&'a P>>, Error> {
    if requests.is_empty()
        || requests
            .iter()
            .any(|request| request.fmt != PixelFormat::Rgb8)
    {
        return Ok(None);
    }
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal subsampled region texture packet plan",
        requests,
    )?;
    let mut family_packets = budget.try_vec(
        packets.len(),
        "JPEG Metal subsampled region texture packet references",
    )?;
    for packet in packets {
        let Some(packet) = P::from_batched(packet) else {
            return Ok(None);
        };
        family_packets.push(packet);
    }
    Ok(Some(family_packets))
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(in crate::compute) struct SubsampledRegionBatchShape {
    pub(in crate::compute) tile_count: usize,
    pub(in crate::compute) tile_count_u32: u32,
    pub(in crate::compute) segment_count: usize,
    pub(in crate::compute) total_decode_threads: u32,
    pub(in crate::compute) plan: RegionScaledBatchPlan,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn subsampled_region_rgb_batch_shape<P: FastRegionScaledMetal>(
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
pub(in crate::compute) fn subsampled_region_texture_batch_shape<
    P: FastSubsampledMetal + FastRegionScaledMetal,
>(
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
pub(in crate::compute) fn encode_subsampled_region_rgb_decode<P: FastRegionScaledMetal>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &P,
    entropy_buffers: &BatchEntropyBuffers,
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    shape: SubsampledRegionBatchShape,
) -> Result<(), Error> {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
    decoder_encoder.set_compute_pipeline_state(
        <P as FastRegionScaledMetal>::scaled_region_batch_decode_pipeline(runtime),
    );
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        &decoder_encoder,
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
        &decoder_encoder,
        <P as FastRegionScaledMetal>::scaled_region_batch_decode_pipeline(runtime),
        shape.total_decode_threads,
    );
    decoder_encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn encode_subsampled_region_texture_decode<
    P: FastSubsampledMetal + FastRegionScaledMetal,
>(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    first: &P,
    entropy_buffers: &BatchEntropyBuffers,
    status_buffer: &Buffer,
    planes: [&Buffer; 3],
    shape: SubsampledRegionBatchShape,
) -> Result<(), Error> {
    let (dc_tables, ac_tables) = fast_packet_huffman_tables(first);
    let decoder_encoder = new_compute_command_encoder(command_buffer)?;
    decoder_encoder.set_compute_pipeline_state(
        <P as FastSubsampledMetal>::scaled_region_batch_decode_pipeline(runtime),
    );
    bind_fast_decode_entropy_inputs::<JpegFastRegionScaledBatchParams>(
        &decoder_encoder,
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
        &decoder_encoder,
        <P as FastSubsampledMetal>::scaled_region_batch_decode_pipeline(runtime),
        shape.total_decode_threads,
    );
    decoder_encoder.end_encoding();
    Ok(())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn region_plane_buffers(
    batch_scratch: &mut MetalBatchScratch,
    device: &Device,
    keys: &FastScratchKeys,
    plan: RegionScaledBatchPlan,
    tile_count: usize,
) -> Result<(Buffer, Buffer, Buffer), Error> {
    Ok((
        batch_scratch.private_buffer(device, keys.y, plan.y_len * tile_count)?,
        batch_scratch.private_buffer(device, keys.cb, plan.chroma_len * tile_count)?,
        batch_scratch.private_buffer(device, keys.cr, plan.chroma_len * tile_count)?,
    ))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn decode_region_scaled_packet_surface(
    runtime: &MetalRuntime,
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
        BatchedFastPacket::Fast420(packet) => {
            try_decode_fast420_scaled_region_to_surface_with_status(
                runtime,
                Some(packet),
                request.fmt,
                scaled_roi,
                scale,
                |status| Ok(fast_decode_status_error(status)),
            )
        }
        BatchedFastPacket::Fast422(packet) => try_decode_fast422_scaled_region_to_surface(
            runtime,
            Some(packet),
            request.fmt,
            scaled_roi,
            scale,
        ),
        BatchedFastPacket::Fast444(packet, mode) => {
            try_decode_fast444_scaled_region_to_surface_with_mode_and_status(
                runtime,
                Some(packet),
                request.fmt,
                scaled_roi,
                scale,
                *mode,
                |status| Ok(fast_decode_status_error(status)),
            )
        }
    }
    .and_then(|surface| {
        surface.ok_or_else(|| Error::MetalKernel {
            message: "JPEG Metal repeated region scaled batch was not packet-decodable".to_string(),
        })
    })
}
