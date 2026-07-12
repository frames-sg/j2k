// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, fast_batch_decode_mode, try_decode_fast444_region_scaled_rgba_batch_to_textures,
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output, BatchedFastPacket,
    Error, FastBatchDecodeMode, JpegFast444PacketV1, MetalRuntime, PixelFormat, Rect, Surface,
};
use super::texture::try_decode_fast_subsampled_full_rgba_batch_to_textures;

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_full_rgb_batch_to_surfaces(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    try_decode_fast444_full_rgb_batch_to_surfaces_with_output(runtime, requests, packets, None)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_full_rgb_batch_to_surfaces_into_output(
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
fn fast444_full_region_scaled_requests(
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
) -> Result<Option<Vec<batch::QueuedRequest>>, Error> {
    if requests.is_empty() || requests.len() != packets.len() {
        return Ok(None);
    }

    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal fast444 full request conversion",
        requests,
    )?;
    let mut region_requests =
        budget.try_vec(requests.len(), "JPEG Metal fast444 region-scaled requests")?;
    for (request, packet) in requests.iter().zip(packets) {
        if request.op != batch::BatchOp::Full || request.fmt != PixelFormat::Rgb8 {
            return Ok(None);
        }
        let BatchedFastPacket::Fast444(packet, _) = packet else {
            return Ok(None);
        };
        let mut request = request.clone();
        request.op = batch::BatchOp::RegionScaled {
            roi: Rect::full(packet.dimensions),
            scale: j2k_core::Downscale::None,
        };
        region_requests.push(request);
    }
    Ok(Some(region_requests))
}

#[cfg(target_os = "macos")]
fn try_decode_fast444_full_rgb_batch_to_surfaces_with_output(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: Option<&crate::MetalBatchOutputBuffer>,
) -> Result<Option<Vec<Result<Surface, Error>>>, Error> {
    let Some(region_requests) = fast444_full_region_scaled_requests(requests, packets)? else {
        return Ok(None);
    };
    try_decode_fast_subsampled_region_scaled_rgb_batch_to_surfaces_with_output::<JpegFast444PacketV1>(
        runtime,
        &region_requests,
        packets,
        output,
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn try_decode_fast444_full_rgba_batch_to_textures(
    runtime: &MetalRuntime,
    requests: &[batch::QueuedRequest],
    packets: &[BatchedFastPacket<'_>],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Option<Vec<Result<crate::MetalTextureTile, Error>>>, Error> {
    let decode_mode = fast_batch_decode_mode();
    if decode_mode == FastBatchDecodeMode::Fused {
        if let Some(results) = try_decode_fast_subsampled_full_rgba_batch_to_textures::<
            JpegFast444PacketV1,
        >(runtime, requests, packets, output, decode_mode)?
        {
            return Ok(Some(results));
        }
    }

    let Some(region_requests) = fast444_full_region_scaled_requests(requests, packets)? else {
        return Ok(None);
    };
    try_decode_fast444_region_scaled_rgba_batch_to_textures(
        runtime,
        &region_requests,
        packets,
        output,
    )
}
