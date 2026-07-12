// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    cached_plane_stage, private_jpeg_tile_from_fast_rgb_buffer, with_runtime,
    with_runtime_for_session, CpuDecoder, Error, JpegFast420PacketV1, JpegFast422PacketV1,
    JpegFast444PacketV1, JpegFastPackets, MTLResourceOptions, MetalRuntime, PixelFormat,
    PlaneStage, Rect, Surface,
};
use super::fast444::{
    try_decode_fast444_region_to_surface, try_decode_fast444_scaled_region_to_surface,
    try_decode_fast444_scaled_to_surface, try_decode_fast444_to_private_rgb8_tile,
    try_decode_fast444_to_surface,
};
use super::subsampled::{
    decode_fast420_to_rgb_buffer, decode_fast422_to_rgb_buffer,
    try_decode_fast420_region_to_surface, try_decode_fast420_scaled_region_to_surface,
    try_decode_fast420_scaled_to_surface, try_decode_fast420_to_surface,
    try_decode_fast422_region_to_surface, try_decode_fast422_scaled_region_to_surface,
    try_decode_fast422_scaled_to_surface, try_decode_fast422_to_surface,
};

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    fmt: PixelFormat,
    packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        decode_to_surface_with_runtime(runtime, decoder, pool, fmt, packets, external_live_bytes)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    fmt: PixelFormat,
    packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
    session: &crate::MetalBackendSession,
) -> Result<Surface, Error> {
    with_runtime_for_session(session, |runtime| {
        decode_to_surface_with_runtime(runtime, decoder, pool, fmt, packets, external_live_bytes)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_private_rgb8_tile_with_session(
    decoder: &CpuDecoder<'_>,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
    session: &crate::MetalBackendSession,
) -> Result<crate::ResidentPrivateJpegTile, Error> {
    with_runtime_for_session(session, |runtime| {
        if let Some(tile) =
            try_decode_fast444_to_private_rgb8_tile(runtime, decoder, fast444_packet)?
        {
            return Ok(tile);
        }
        if let Some(decoded) = decode_fast422_to_rgb_buffer(
            runtime,
            fast422_packet,
            PixelFormat::Rgb8,
            MTLResourceOptions::StorageModePrivate,
        )? {
            return Ok(private_jpeg_tile_from_fast_rgb_buffer(decoded));
        }
        if let Some(decoded) = decode_fast420_to_rgb_buffer(
            runtime,
            decoder,
            fast420_packet,
            PixelFormat::Rgb8,
            MTLResourceOptions::StorageModePrivate,
        )? {
            return Ok(private_jpeg_tile_from_fast_rgb_buffer(decoded));
        }
        Err(Error::UnsupportedMetalRequest {
            reason:
                "private JPEG Metal output supports only fast baseline 4:4:4, 4:2:2, or 4:2:0 RGB8 full-tile decode",
        })
    })
}

#[cfg(target_os = "macos")]
fn decode_to_surface_with_runtime(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    fmt: PixelFormat,
    packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    if let Some(surface) = try_decode_fast444_to_surface(runtime, decoder, packets.fast444, fmt)? {
        return Ok(surface);
    }
    if let Some(surface) = try_decode_fast422_to_surface(runtime, packets.fast422, fmt)? {
        return Ok(surface);
    }
    if let Some(surface) = try_decode_fast420_to_surface(runtime, decoder, packets.fast420, fmt)? {
        return Ok(surface);
    }
    let mut stage = PlaneStage::new(
        &runtime.device,
        decoder.info().color_space,
        decoder.info().dimensions,
        external_live_bytes,
    )?;
    decoder.decode_component_rows_with_scratch(pool, &mut stage)?;
    stage.finish_with_runtime(runtime, fmt)
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
    packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_region_to_surface(runtime, decoder, packets.fast444, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast422_region_to_surface(runtime, packets.fast422, fmt, roi)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_region_to_surface(runtime, decoder, packets.fast420, fmt, roi)?
        {
            return Ok(surface);
        }
        let dims = (roi.w, roi.h);
        let mut stage = cached_plane_stage(
            runtime,
            decoder.info().color_space,
            dims,
            external_live_bytes,
        )?;
        decoder.decode_region_component_rows_with_scratch(
            pool,
            &mut stage,
            roi,
            j2k_core::Downscale::None,
        )?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    fmt: PixelFormat,
    scale: j2k_core::Downscale,
    packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) =
            try_decode_fast444_scaled_to_surface(runtime, decoder, packets.fast444, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast422_scaled_to_surface(runtime, packets.fast422, fmt, scale)?
        {
            return Ok(surface);
        }
        if let Some(surface) =
            try_decode_fast420_scaled_to_surface(runtime, decoder, packets.fast420, fmt, scale)?
        {
            return Ok(surface);
        }
        let full = decoder.info().dimensions;
        let roi = j2k_jpeg::Rect {
            x: 0,
            y: 0,
            w: full.0,
            h: full.1,
        };
        let scaled = (Rect {
            x: 0,
            y: 0,
            w: full.0,
            h: full.1,
        })
        .scaled_covering(scale);
        let mut stage = cached_plane_stage(
            runtime,
            decoder.info().color_space,
            (scaled.w, scaled.h),
            external_live_bytes,
        )?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut j2k_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
    scale: j2k_core::Downscale,
    packets: JpegFastPackets<'_>,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let scaled_roi = (Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        })
        .scaled_covering(scale);
        if let Some(surface) = try_decode_fast444_scaled_region_to_surface(
            runtime,
            decoder,
            packets.fast444,
            fmt,
            j2k_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast422_scaled_region_to_surface(
            runtime,
            packets.fast422,
            fmt,
            j2k_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        if let Some(surface) = try_decode_fast420_scaled_region_to_surface(
            runtime,
            decoder,
            packets.fast420,
            fmt,
            j2k_jpeg::Rect {
                x: scaled_roi.x,
                y: scaled_roi.y,
                w: scaled_roi.w,
                h: scaled_roi.h,
            },
            scale,
        )? {
            return Ok(surface);
        }
        let scaled = (Rect {
            x: roi.x,
            y: roi.y,
            w: roi.w,
            h: roi.h,
        })
        .scaled_covering(scale);
        let mut stage = cached_plane_stage(
            runtime,
            decoder.info().color_space,
            (scaled.w, scaled.h),
            external_live_bytes,
        )?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}
