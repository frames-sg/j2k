// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    batch, batched_fast_packets, commit_and_wait_jpeg, decode_error_from_cpu,
    encode_fast444_batch_item, encode_fast444_region_batch_item, encode_fast444_scaled_batch_item,
    encode_fast444_scaled_region_batch_item, encode_fast_subsampled_op_batch_item,
    fast_batch_decode_mode, first_decode_error_status,
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast420_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast420_region_scaled_rgba_batch_to_textures,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast422_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast422_region_scaled_rgba_batch_to_textures,
    try_decode_fast444_full_rgb_batch_to_surfaces,
    try_decode_fast444_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_full_rgba_batch_to_textures,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces,
    try_decode_fast444_region_scaled_rgb_batch_to_surfaces_into_output,
    try_decode_fast444_region_scaled_rgba_batch_to_textures,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces,
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces_into_output,
    try_decode_fast_subsampled_full_rgba_batch_to_textures,
    try_decode_repeated_region_scaled_batch_to_surfaces, with_runtime, with_runtime_for_session,
    BatchDeviceBufferCache, BatchedFastPacket, CpuDecoder, Error,
    Fast444ScaledRegionBatchItemRequest, FastSubsampledOpBatchItemRequest, JpegFast420PacketV1,
    JpegFast422PacketV1, MetalRuntime, Surface, REGION_SCALED_BATCH_CHUNK,
};

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
    let _output_access = output.lock_for_safe_access()?;

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
    let _texture_output_access = output.lock_for_safe_access()?;

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
    let _output_access = output.lock_for_safe_access()?;

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
    let _texture_output_access = output.lock_for_safe_access()?;

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
#[expect(
    clippy::too_many_lines,
    reason = "the fallback batch encoder keeps command submission, retained resources, and result ordering in one lifetime scope"
)]
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
                BatchedFastPacket::Fast420(packet) => {
                    encode_fast_subsampled_op_batch_item(FastSubsampledOpBatchItemRequest {
                        runtime,
                        command_buffer,
                        device_buffer_cache: &mut device_buffer_cache,
                        request_index: index,
                        packet: *packet,
                        fmt: request.fmt,
                        op: request.op,
                    })?
                }
                BatchedFastPacket::Fast422(packet) => {
                    encode_fast_subsampled_op_batch_item(FastSubsampledOpBatchItemRequest {
                        runtime,
                        command_buffer,
                        device_buffer_cache: &mut device_buffer_cache,
                        request_index: index,
                        packet: *packet,
                        fmt: request.fmt,
                        op: request.op,
                    })?
                }
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
                            Fast444ScaledRegionBatchItemRequest {
                                runtime,
                                command_buffer,
                                device_buffer_cache: &mut device_buffer_cache,
                                request_index: index,
                                packet,
                                mode: *mode,
                                fmt: request.fmt,
                                roi,
                                scale,
                            },
                        )?
                    }
                },
            };
            encoded.push(item);
        }

        commit_and_wait_jpeg(command_buffer)?;

        for item in encoded {
            if let Some(status) =
                first_decode_error_status(&item.status_buffer, item.decode_threads)?
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
