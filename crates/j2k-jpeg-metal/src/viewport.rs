// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
#[cfg(target_os = "macos")]
use j2k_jpeg::adapter::decoder_bytes;
use j2k_jpeg::adapter::{
    build_fast420_packet_for_decoder, build_fast422_packet_for_decoder,
    build_fast444_packet_for_decoder, JpegFast420PacketV1, JpegFast422PacketV1,
    JpegFast444PacketV1,
};
#[cfg(target_os = "macos")]
use j2k_jpeg::ColorSpace as JpegColorSpace;
use j2k_jpeg::{Decoder as CpuDecoder, Rect as JpegRect, ScratchPool};

use crate::{batch, routing, Error, Surface};
#[cfg(target_os = "macos")]
use crate::{
    Codec, MetalBackendSession, MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalTextureTile,
};

const VIEWPORT_TILE_EDGE: u32 = 96;
const VIEWPORT_TILE_COLS: u32 = 6;
const VIEWPORT_TILE_ROWS: u32 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// One source-to-destination region in a composed viewport.
pub struct ViewportTile {
    /// Source region in the JPEG image before downscaling.
    pub source_roi: Rect,
    /// Destination rectangle in the viewport after downscaling.
    pub dest: Rect,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Planned viewport decode made of one or more source tiles.
pub struct ViewportWorkload {
    /// Downscale factor applied to every source tile.
    pub scale: Downscale,
    /// Output viewport dimensions in pixels.
    pub viewport_dims: (u32, u32),
    /// Tiles to decode and place into the viewport.
    pub tiles: Vec<ViewportTile>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Execution strategy selected for a viewport decode.
pub enum ViewportSurfaceStrategy {
    /// Decode each tile on CPU and composite into a host viewport.
    CpuComposite,
    /// Decode one contiguous source region on CPU.
    CpuContiguous,
    /// Decode or upload through Metal while compositing multiple source tiles.
    HybridComposite,
    /// Decode one contiguous source region through the Metal path.
    HybridContiguous,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Resident Metal output strategy selected for a reusable viewport decode.
pub enum ViewportResidentOutputStrategy {
    /// Decode the contiguous source bounds through the direct resident batch path.
    DirectContiguous,
    /// Decode component rows into resident planes and pack the composed viewport.
    Composite,
}

/// Compute the bounding source rectangle covering all tiles in a workload.
pub fn viewport_source_bounds(workload: &ViewportWorkload) -> Rect {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for tile in &workload.tiles {
        min_x = min_x.min(tile.source_roi.x);
        min_y = min_y.min(tile.source_roi.y);
        max_x = max_x.max(tile.source_roi.x.saturating_add(tile.source_roi.w));
        max_y = max_y.max(tile.source_roi.y.saturating_add(tile.source_roi.h));
    }

    Rect {
        x: min_x,
        y: min_y,
        w: max_x.saturating_sub(min_x),
        h: max_y.saturating_sub(min_y),
    }
}

/// Return whether the workload covers a contiguous viewport without overlaps.
pub fn is_contiguous_viewport_workload(workload: &ViewportWorkload) -> bool {
    if workload.tiles.is_empty() {
        return false;
    }

    let source = viewport_source_bounds(workload);
    let scaled_source = source.scaled_covering(workload.scale);
    if (scaled_source.w, scaled_source.h) != workload.viewport_dims {
        return false;
    }

    let viewport_area = u64::from(workload.viewport_dims.0) * u64::from(workload.viewport_dims.1);
    let mut area_sum = 0u64;

    for tile in &workload.tiles {
        let scaled_tile = tile.source_roi.scaled_covering(workload.scale);
        let expected = Rect {
            x: scaled_tile.x.saturating_sub(scaled_source.x),
            y: scaled_tile.y.saturating_sub(scaled_source.y),
            w: scaled_tile.w,
            h: scaled_tile.h,
        };
        if tile.dest != expected {
            return false;
        }
        if tile.dest.x.saturating_add(tile.dest.w) > workload.viewport_dims.0
            || tile.dest.y.saturating_add(tile.dest.h) > workload.viewport_dims.1
        {
            return false;
        }

        area_sum = area_sum.saturating_add(u64::from(tile.dest.w) * u64::from(tile.dest.h));
    }

    for (idx, tile) in workload.tiles.iter().enumerate() {
        let tile_right = tile.dest.x.saturating_add(tile.dest.w);
        let tile_bottom = tile.dest.y.saturating_add(tile.dest.h);
        for other in &workload.tiles[idx + 1..] {
            let other_right = other.dest.x.saturating_add(other.dest.w);
            let other_bottom = other.dest.y.saturating_add(other.dest.h);
            let separated = tile_right <= other.dest.x
                || other_right <= tile.dest.x
                || tile_bottom <= other.dest.y
                || other_bottom <= tile.dest.y;
            if !separated {
                return false;
            }
        }
    }

    area_sum == viewport_area
}

/// Choose the backend strategy for a workload without inspecting JPEG capabilities.
pub fn choose_viewport_surface_strategy(
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ViewportSurfaceStrategy, Error> {
    let contiguous = is_contiguous_viewport_workload(workload);
    match backend {
        BackendRequest::Cpu => Ok(if contiguous {
            ViewportSurfaceStrategy::CpuContiguous
        } else {
            ViewportSurfaceStrategy::CpuComposite
        }),
        BackendRequest::Auto | BackendRequest::Metal => {
            #[cfg(target_os = "macos")]
            {
                Ok(if contiguous {
                    ViewportSurfaceStrategy::HybridContiguous
                } else {
                    ViewportSurfaceStrategy::HybridComposite
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                if matches!(backend, BackendRequest::Metal) {
                    Err(Error::MetalUnavailable)
                } else if contiguous {
                    Ok(ViewportSurfaceStrategy::CpuContiguous)
                } else {
                    Ok(ViewportSurfaceStrategy::CpuComposite)
                }
            }
        }
        BackendRequest::Cuda => Err(Error::UnsupportedBackend { request: backend }),
    }
}

fn choose_viewport_surface_strategy_for_decoder(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<ViewportSurfaceStrategy, Error> {
    if matches!(backend, BackendRequest::Metal) {
        validate_explicit_metal_viewport_request(decoder, workload)?;
        return choose_viewport_surface_strategy(workload, backend);
    }

    if !matches!(backend, BackendRequest::Auto) {
        return choose_viewport_surface_strategy(workload, backend);
    }

    #[cfg(not(target_os = "macos"))]
    let _ = decoder;

    #[cfg(target_os = "macos")]
    {
        let contiguous = is_contiguous_viewport_workload(workload);
        if !contiguous {
            return Ok(ViewportSurfaceStrategy::CpuComposite);
        }

        let restart_coded = decoder.info().restart_interval.is_some();
        if !restart_coded {
            return Ok(ViewportSurfaceStrategy::CpuContiguous);
        }

        let has_direct_packet = has_direct_viewport_packet(decoder);
        Ok(if has_direct_packet {
            ViewportSurfaceStrategy::HybridContiguous
        } else {
            ViewportSurfaceStrategy::CpuContiguous
        })
    }

    #[cfg(not(target_os = "macos"))]
    {
        choose_viewport_surface_strategy(workload, backend)
    }
}

#[cfg(target_os = "macos")]
fn has_direct_viewport_packet(decoder: &CpuDecoder<'_>) -> bool {
    build_fast444_packet_for_decoder(decoder).is_ok()
        || build_fast422_packet_for_decoder(decoder).is_ok()
        || build_fast420_packet_for_decoder(decoder).is_ok()
}

fn validate_explicit_metal_viewport_request(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
) -> Result<(), Error> {
    let fast444_packet = build_fast444_packet_for_decoder(decoder).ok();
    let fast422_packet = build_fast422_packet_for_decoder(decoder).ok();
    let fast420_packet = build_fast420_packet_for_decoder(decoder).ok();
    validate_explicit_metal_viewport_request_with_packets(
        decoder,
        workload,
        fast444_packet.as_ref(),
        fast422_packet.as_ref(),
        fast420_packet.as_ref(),
    )
}

fn validate_explicit_metal_viewport_request_with_packets(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    fast444_packet: Option<&JpegFast444PacketV1>,
    fast422_packet: Option<&JpegFast422PacketV1>,
    fast420_packet: Option<&JpegFast420PacketV1>,
) -> Result<(), Error> {
    let source = viewport_source_bounds(workload);
    let capabilities = routing::JpegMetalCapabilities::for_request(
        decoder,
        PixelFormat::Rgb8,
        batch::BatchOp::RegionScaled {
            roi: source,
            scale: workload.scale,
        },
        fast444_packet,
        fast422_packet,
        fast420_packet,
    );
    let decision = routing::decide_route(BackendRequest::Metal, capabilities);
    if let Some(err) = routing::decision_error(decision) {
        return Err(err);
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_resident_viewport_composition_request(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
) -> Result<(), Error> {
    if workload.tiles.is_empty() {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal resident viewport output requires at least one viewport tile",
        });
    }
    if matches!(
        decoder.info().color_space,
        JpegColorSpace::Cmyk | JpegColorSpace::Ycck
    ) {
        return Err(Error::UnsupportedMetalRequest {
            reason:
                "JPEG Metal resident viewport composition does not support CMYK/YCCK JPEG output",
        });
    }

    for tile in &workload.tiles {
        let dims = tile.source_roi.scaled_covering(workload.scale);
        if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
            return Err(Error::UnsupportedMetalRequest {
                reason:
                    "JPEG Metal resident viewport tile dimensions do not match destination rect",
            });
        }
        if tile.dest.x.saturating_add(tile.dest.w) > workload.viewport_dims.0
            || tile.dest.y.saturating_add(tile.dest.h) > workload.viewport_dims.1
        {
            return Err(Error::UnsupportedMetalRequest {
                reason: "JPEG Metal resident viewport destination exceeds viewport dimensions",
            });
        }
    }

    Ok(())
}

#[cfg(target_os = "macos")]
/// Choose the resident Metal strategy for a reusable viewport output request.
pub fn choose_resizable_metal_viewport_strategy(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
) -> Result<ViewportResidentOutputStrategy, Error> {
    if is_contiguous_viewport_workload(workload)
        && validate_explicit_metal_viewport_request(decoder, workload).is_ok()
    {
        return Ok(ViewportResidentOutputStrategy::DirectContiguous);
    }

    validate_resident_viewport_composition_request(decoder, workload)?;
    Ok(ViewportResidentOutputStrategy::Composite)
}

#[cfg(target_os = "macos")]
fn choose_resizable_metal_viewport_strategy_for_decoder(
    decoder: &crate::Decoder<'_>,
    workload: &ViewportWorkload,
) -> Result<ViewportResidentOutputStrategy, Error> {
    if is_contiguous_viewport_workload(workload)
        && validate_explicit_metal_viewport_request_with_packets(
            decoder.inner(),
            workload,
            decoder.fast444_packet(),
            decoder.fast422_packet(),
            decoder.fast420_packet(),
        )
        .is_ok()
    {
        return Ok(ViewportResidentOutputStrategy::DirectContiguous);
    }

    validate_resident_viewport_composition_request(decoder.inner(), workload)?;
    Ok(ViewportResidentOutputStrategy::Composite)
}

/// Suggest a fixed-size centered viewport workload for an image.
pub fn suggest_viewport_workload(dimensions: (u32, u32)) -> Option<ViewportWorkload> {
    let scales = [
        Downscale::Eighth,
        Downscale::Quarter,
        Downscale::Half,
        Downscale::None,
    ];
    let viewport_dims = (
        VIEWPORT_TILE_EDGE * VIEWPORT_TILE_COLS,
        VIEWPORT_TILE_EDGE * VIEWPORT_TILE_ROWS,
    );
    for scale in scales {
        let denom = scale.denominator();
        let Some(x) = viewport_origin(dimensions.0, viewport_dims.0.saturating_mul(denom), denom)
        else {
            continue;
        };
        let Some(y) = viewport_origin(dimensions.1, viewport_dims.1.saturating_mul(denom), denom)
        else {
            continue;
        };
        let source_viewport = Rect {
            x,
            y,
            w: viewport_dims.0.saturating_mul(denom),
            h: viewport_dims.1.saturating_mul(denom),
        };
        let scaled_source = source_viewport.scaled_covering(scale);
        if (scaled_source.w, scaled_source.h) != viewport_dims {
            continue;
        }
        let source_tile = VIEWPORT_TILE_EDGE.saturating_mul(denom);
        let mut tiles = Vec::with_capacity((VIEWPORT_TILE_COLS * VIEWPORT_TILE_ROWS) as usize);
        for row in 0..VIEWPORT_TILE_ROWS {
            for col in 0..VIEWPORT_TILE_COLS {
                tiles.push(ViewportTile {
                    source_roi: Rect {
                        x: source_viewport.x + col * source_tile,
                        y: source_viewport.y + row * source_tile,
                        w: source_tile,
                        h: source_tile,
                    },
                    dest: Rect {
                        x: col * VIEWPORT_TILE_EDGE,
                        y: row * VIEWPORT_TILE_EDGE,
                        w: VIEWPORT_TILE_EDGE,
                        h: VIEWPORT_TILE_EDGE,
                    },
                });
            }
        }

        return Some(ViewportWorkload {
            scale,
            viewport_dims,
            tiles,
        });
    }

    None
}

/// Decode each viewport tile on CPU and composite the result into host bytes.
pub fn compose_viewport_cpu(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Vec<u8>, Error> {
    let bpp = fmt.bytes_per_pixel();
    let viewport_stride = viewport_dims.0 as usize * bpp;
    let mut viewport = vec![0u8; viewport_stride * viewport_dims.1 as usize];

    for tile in tiles {
        let scaled = tile.source_roi.scaled_covering(scale);
        let tile_dims = (scaled.w, scaled.h);
        if tile_dims != (tile.dest.w, tile.dest.h) {
            return Err(Error::MetalKernel {
                message: format!(
                    "viewport tile dims {:?} do not match destination rect {:?}",
                    tile_dims, tile.dest
                ),
            });
        }
        let tile_stride = tile_dims.0 as usize * bpp;
        let mut tile_bytes = vec![0u8; tile_stride * tile_dims.1 as usize];
        decoder.decode_region_scaled_into_with_scratch(
            pool,
            &mut tile_bytes,
            tile_stride,
            fmt,
            to_jpeg_rect(tile.source_roi),
            scale,
        )?;
        blit_into_viewport(
            &tile_bytes,
            tile_dims,
            fmt,
            &mut viewport,
            viewport_dims,
            tile.dest,
        )?;
    }

    Ok(viewport)
}

/// Decode the contiguous source region for a workload into host bytes.
pub fn decode_viewport_region_cpu(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    fmt: PixelFormat,
    workload: &ViewportWorkload,
) -> Result<Vec<u8>, Error> {
    let source = viewport_source_bounds(workload);
    let stride = workload.viewport_dims.0 as usize * fmt.bytes_per_pixel();
    let mut viewport = vec![0u8; stride * workload.viewport_dims.1 as usize];
    decoder.decode_region_scaled_into_with_scratch(
        pool,
        &mut viewport,
        stride,
        fmt,
        to_jpeg_rect(source),
        workload.scale,
    )?;
    Ok(viewport)
}

/// Decode a viewport workload into a surface using the requested backend policy.
pub fn decode_viewport_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    match choose_viewport_surface_strategy_for_decoder(decoder, workload, backend)? {
        ViewportSurfaceStrategy::CpuComposite => compose_viewport_cpu_to_surface(
            decoder,
            pool,
            workload.scale,
            workload.viewport_dims,
            &workload.tiles,
        ),
        ViewportSurfaceStrategy::CpuContiguous => {
            decode_viewport_region_cpu_to_surface(decoder, pool, workload)
        }
        ViewportSurfaceStrategy::HybridComposite => compose_viewport_hybrid(
            decoder,
            pool,
            workload.scale,
            workload.viewport_dims,
            &workload.tiles,
        ),
        ViewportSurfaceStrategy::HybridContiguous => {
            decode_viewport_region_hybrid(decoder, pool, workload)
        }
    }
}

#[cfg(target_os = "macos")]
/// Decode the contiguous source region on CPU and upload it to a surface.
pub fn decode_viewport_region_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    let bytes = decode_viewport_region_cpu(decoder, pool, PixelFormat::Rgb8, workload)?;
    crate::upload_surface(
        bytes,
        workload.viewport_dims,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cpu,
    )
}

#[cfg(not(target_os = "macos"))]
/// Decode the contiguous source region on CPU and return a host-backed surface.
pub fn decode_viewport_region_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    let bytes = decode_viewport_region_cpu(decoder, pool, PixelFormat::Rgb8, workload)?;
    crate::upload_surface(
        bytes,
        workload.viewport_dims,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cpu,
    )
}

#[cfg(target_os = "macos")]
/// Decode and composite viewport tiles on CPU, then upload to a surface.
pub fn compose_viewport_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    let bytes = compose_viewport_cpu(
        decoder,
        pool,
        PixelFormat::Rgb8,
        scale,
        viewport_dims,
        tiles,
    )?;
    crate::upload_surface(
        bytes,
        viewport_dims,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cpu,
    )
}

#[cfg(not(target_os = "macos"))]
/// Decode and composite viewport tiles on CPU into a host-backed surface.
pub fn compose_viewport_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    let bytes = compose_viewport_cpu(
        decoder,
        pool,
        PixelFormat::Rgb8,
        scale,
        viewport_dims,
        tiles,
    )?;
    crate::upload_surface(
        bytes,
        viewport_dims,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cpu,
    )
}

#[cfg(target_os = "macos")]
/// Compose a multi-tile viewport through the Metal hybrid path.
pub fn compose_viewport_hybrid(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    crate::compute::compose_rgb_viewport_from_regions(decoder, pool, scale, viewport_dims, tiles)
}

#[cfg(target_os = "macos")]
/// Compose a viewport workload into a reusable caller-owned Metal buffer.
///
/// This path supports sparse and non-contiguous workloads. It decodes component
/// rows into reusable Metal plane buffers, resizes `output` to one RGB8 viewport
/// slot, and packs the composed viewport directly into that caller-owned buffer.
pub fn compose_viewport_to_resizable_metal_buffer_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    validate_resident_viewport_composition_request(decoder, workload)?;
    output.ensure_rgb8_tiles(session, workload.viewport_dims, 1)?;
    crate::compute::compose_rgb_viewport_from_regions_into_output_with_session(
        decoder,
        pool,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
        output,
        session,
    )
}

#[cfg(target_os = "macos")]
/// Compose a viewport workload into a reusable caller-owned Metal texture.
///
/// This path supports sparse and non-contiguous workloads. It decodes component
/// rows into reusable Metal plane buffers, resizes `output` to one RGBA8
/// viewport slot, and packs the composed viewport directly into that
/// caller-owned texture.
pub fn compose_viewport_to_resizable_metal_textures_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    validate_resident_viewport_composition_request(decoder, workload)?;
    output.ensure_rgba8_tiles(session, workload.viewport_dims, 1)?;
    crate::compute::compose_rgb_viewport_from_regions_into_textures_with_session(
        decoder,
        pool,
        workload.scale,
        workload.viewport_dims,
        &workload.tiles,
        output,
        session,
    )
}

#[cfg(target_os = "macos")]
/// Decode any viewport workload into a reusable caller-owned Metal buffer.
///
/// Contiguous workloads use the direct resident region-scaled batch path when
/// eligible. Sparse or unsupported direct shapes use resident component-row
/// composition into the same caller-owned RGB8 buffer.
pub fn decode_viewport_to_resizable_metal_buffer_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    match choose_resizable_metal_viewport_strategy(decoder, workload)? {
        ViewportResidentOutputStrategy::DirectContiguous => {
            decode_viewport_region_to_resizable_metal_buffer_with_session(
                decoder_bytes(decoder),
                workload,
                output,
                session,
            )
        }
        ViewportResidentOutputStrategy::Composite => {
            compose_viewport_to_resizable_metal_buffer_with_session(
                decoder, pool, workload, output, session,
            )
        }
    }
}

#[cfg(target_os = "macos")]
/// Decode any viewport workload into reusable caller-owned Metal textures.
///
/// Contiguous workloads use the direct resident region-scaled texture batch path
/// when eligible. Sparse or unsupported direct shapes use resident component-row
/// composition into the same caller-owned RGBA8 texture.
pub fn decode_viewport_to_resizable_metal_textures_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    match choose_resizable_metal_viewport_strategy(decoder, workload)? {
        ViewportResidentOutputStrategy::DirectContiguous => {
            decode_viewport_region_to_resizable_metal_textures_with_session(
                decoder_bytes(decoder),
                workload,
                output,
                session,
            )
        }
        ViewportResidentOutputStrategy::Composite => {
            compose_viewport_to_resizable_metal_textures_with_session(
                decoder, pool, workload, output, session,
            )
        }
    }
}

#[cfg(target_os = "macos")]
/// Decode any viewport workload into a reusable caller-owned Metal buffer using
/// an already parsed Metal decoder wrapper.
///
/// Contiguous workloads use the wrapper's cached fast-packet state for direct
/// resident region-scaled decode. Sparse or unsupported direct shapes use
/// resident component-row composition through the wrapper's CPU decoder.
pub fn decode_viewport_to_resizable_metal_buffer_with_decoder_session(
    decoder: &crate::Decoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    match choose_resizable_metal_viewport_strategy_for_decoder(decoder, workload)? {
        ViewportResidentOutputStrategy::DirectContiguous => {
            decode_viewport_region_to_resizable_metal_buffer_with_decoder_session(
                decoder, workload, output, session,
            )
        }
        ViewportResidentOutputStrategy::Composite => {
            compose_viewport_to_resizable_metal_buffer_with_session(
                decoder.inner(),
                pool,
                workload,
                output,
                session,
            )
        }
    }
}

#[cfg(target_os = "macos")]
/// Decode any viewport workload into reusable caller-owned Metal textures using
/// an already parsed Metal decoder wrapper.
///
/// Contiguous workloads use the wrapper's cached fast-packet state for direct
/// resident region-scaled decode. Sparse or unsupported direct shapes use
/// resident component-row composition through the wrapper's CPU decoder.
pub fn decode_viewport_to_resizable_metal_textures_with_decoder_session(
    decoder: &crate::Decoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    match choose_resizable_metal_viewport_strategy_for_decoder(decoder, workload)? {
        ViewportResidentOutputStrategy::DirectContiguous => {
            decode_viewport_region_to_resizable_metal_textures_with_decoder_session(
                decoder, workload, output, session,
            )
        }
        ViewportResidentOutputStrategy::Composite => {
            compose_viewport_to_resizable_metal_textures_with_session(
                decoder.inner(),
                pool,
                workload,
                output,
                session,
            )
        }
    }
}

#[cfg(target_os = "macos")]
fn decode_viewport_region_to_resizable_metal_buffer_with_decoder_session(
    decoder: &crate::Decoder<'_>,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    validate_explicit_metal_viewport_request_with_packets(
        decoder.inner(),
        workload,
        decoder.fast444_packet(),
        decoder.fast422_packet(),
        decoder.fast420_packet(),
    )?;
    if !is_contiguous_viewport_workload(workload) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport output currently requires a contiguous viewport workload",
        });
    }

    let source = viewport_source_bounds(workload);
    let scaled = source.scaled_covering(workload.scale);
    output.ensure_rgb8_tiles(session, (scaled.w, scaled.h), 1)?;
    let request = decoder
        .rgb8_region_scaled_metal_request(source, workload.scale)
        .with_output_slot(0);
    let requests = [request];
    let mut surfaces = crate::compute::decode_region_scaled_rgb8_batch_into_output_with_session(
        &requests, output, session,
    )?
    .ok_or(Error::UnsupportedMetalRequest {
        reason: "JPEG Metal reusable viewport output currently supports RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs",
    })?;
    let Some(surface) = surfaces.pop() else {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport output did not produce a surface",
        });
    };
    debug_assert!(surfaces.is_empty());
    surface
}

#[cfg(target_os = "macos")]
fn decode_viewport_region_to_resizable_metal_textures_with_decoder_session(
    decoder: &crate::Decoder<'_>,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    validate_explicit_metal_viewport_request_with_packets(
        decoder.inner(),
        workload,
        decoder.fast444_packet(),
        decoder.fast422_packet(),
        decoder.fast420_packet(),
    )?;
    if !is_contiguous_viewport_workload(workload) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport texture output currently requires a contiguous viewport workload",
        });
    }

    let source = viewport_source_bounds(workload);
    let scaled = source.scaled_covering(workload.scale);
    output.ensure_rgba8_tiles(session, (scaled.w, scaled.h), 1)?;
    let request = decoder
        .rgb8_region_scaled_metal_request(source, workload.scale)
        .with_output_slot(0);
    let requests = [request];
    let mut tiles = crate::compute::decode_region_scaled_rgb8_batch_into_textures_with_session(
        &requests, output, session,
    )?
    .ok_or(Error::UnsupportedMetalRequest {
        reason: "JPEG Metal reusable viewport texture output currently supports RGB8 fast 4:2:0, 4:2:2, or 4:4:4 inputs",
    })?;
    let Some(tile) = tiles.pop() else {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport texture output did not produce a tile",
        });
    };
    debug_assert!(tiles.is_empty());
    tile
}

#[cfg(target_os = "macos")]
/// Decode a contiguous viewport region through the Metal hybrid path.
pub fn decode_viewport_region_hybrid(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    let use_direct_kernel = decoder.info().restart_interval.is_some();
    let fast444_packet = use_direct_kernel
        .then(|| build_fast444_packet_for_decoder(decoder).ok())
        .flatten();
    let fast422_packet = use_direct_kernel
        .then(|| build_fast422_packet_for_decoder(decoder).ok())
        .flatten();
    let fast420_packet = use_direct_kernel
        .then(|| build_fast420_packet_for_decoder(decoder).ok())
        .flatten();
    crate::compute::decode_region_scaled_to_surface(
        decoder,
        pool,
        PixelFormat::Rgb8,
        to_jpeg_rect(viewport_source_bounds(workload)),
        workload.scale,
        fast444_packet.as_ref(),
        fast422_packet.as_ref(),
        fast420_packet.as_ref(),
    )
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for hybrid viewport decode requests.
pub fn decode_viewport_region_hybrid(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _workload: &ViewportWorkload,
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
/// Decode a contiguous viewport workload into a reusable caller-owned Metal buffer.
///
/// This is the resident-output counterpart to `decode_viewport_region_hybrid`:
/// it rejects non-contiguous workloads instead of compositing, resizes `output`
/// to one RGB8 viewport slot, and returns a `MetalResidentDecode` surface
/// backed by that caller-owned allocation.
pub fn decode_viewport_region_to_resizable_metal_buffer_with_session(
    input: &[u8],
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    let decoder = CpuDecoder::new(input)?;
    validate_explicit_metal_viewport_request(&decoder, workload)?;
    if !is_contiguous_viewport_workload(workload) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport output currently requires a contiguous viewport workload",
        });
    }

    let mut surfaces =
        Codec::decode_rgb8_region_scaled_batch_into_resizable_metal_buffer_with_session(
            &[input],
            viewport_source_bounds(workload),
            workload.scale,
            output,
            session,
        )?;
    let Some(surface) = surfaces.pop() else {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport output did not produce a surface",
        });
    };
    debug_assert!(surfaces.is_empty());
    surface
}

#[cfg(target_os = "macos")]
/// Decode a contiguous viewport workload into reusable caller-owned Metal textures.
///
/// This is the texture-output counterpart to
/// `decode_viewport_region_to_resizable_metal_buffer_with_session`: it rejects
/// non-contiguous workloads, resizes `output` to one RGBA8 viewport slot, and
/// returns a tile backed by that caller-owned texture.
pub fn decode_viewport_region_to_resizable_metal_textures_with_session(
    input: &[u8],
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    let decoder = CpuDecoder::new(input)?;
    validate_explicit_metal_viewport_request(&decoder, workload)?;
    if !is_contiguous_viewport_workload(workload) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport texture output currently requires a contiguous viewport workload",
        });
    }

    let mut tiles =
        Codec::decode_rgb8_region_scaled_batch_into_resizable_metal_textures_with_session(
            &[input],
            viewport_source_bounds(workload),
            workload.scale,
            output,
            session,
        )?;
    let Some(tile) = tiles.pop() else {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport texture output did not produce a tile",
        });
    };
    debug_assert!(tiles.is_empty());
    tile
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for hybrid viewport composition requests.
pub fn compose_viewport_hybrid(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _scale: Downscale,
    _viewport_dims: (u32, u32),
    _tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
}

fn viewport_origin(full_extent: u32, viewport_extent: u32, align: u32) -> Option<u32> {
    if viewport_extent > full_extent || align == 0 {
        return None;
    }

    let centered = (full_extent - viewport_extent) / 2;
    Some(centered - centered % align)
}

fn to_jpeg_rect(rect: Rect) -> JpegRect {
    JpegRect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

fn blit_into_viewport(
    tile: &[u8],
    tile_dims: (u32, u32),
    fmt: PixelFormat,
    viewport: &mut [u8],
    viewport_dims: (u32, u32),
    dest: Rect,
) -> Result<(), Error> {
    if dest.x.saturating_add(dest.w) > viewport_dims.0
        || dest.y.saturating_add(dest.h) > viewport_dims.1
    {
        return Err(Error::MetalKernel {
            message: format!("viewport destination {dest:?} exceeds viewport {viewport_dims:?}"),
        });
    }

    let bpp = fmt.bytes_per_pixel();
    let tile_stride = tile_dims.0 as usize * bpp;
    let viewport_stride = viewport_dims.0 as usize * bpp;
    for row in 0..tile_dims.1 as usize {
        let src_start = row * tile_stride;
        let src_end = src_start + tile_stride;
        let dst_start = (dest.y as usize + row) * viewport_stride + dest.x as usize * bpp;
        let dst_end = dst_start + tile_stride;
        viewport[dst_start..dst_end].copy_from_slice(&tile[src_start..src_end]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASELINE_420: &[u8] = include_bytes!("../fixtures/jpeg/baseline_420_16x16.jpg");
    #[cfg(target_os = "macos")]
    const BASELINE_422: &[u8] = include_bytes!("../fixtures/jpeg/baseline_422_16x8.jpg");
    const BASELINE_420_RESTART: &[u8] =
        include_bytes!("../fixtures/jpeg/baseline_420_restart_32x16.jpg");

    fn sparse_workload_from(workload: &ViewportWorkload) -> ViewportWorkload {
        ViewportWorkload {
            scale: workload.scale,
            viewport_dims: workload.viewport_dims,
            tiles: vec![
                *workload.tiles.first().expect("viewport tile"),
                *workload.tiles.last().expect("viewport tile"),
            ],
        }
    }

    #[test]
    fn auto_strategy_keeps_large_contiguous_nonrestart_workloads_on_cpu_contiguous() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let workload = suggest_viewport_workload((2_048, 1_024)).expect("contiguous workload");

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::CpuContiguous
        );
    }

    #[test]
    fn auto_strategy_prefers_hybrid_for_restart_coded_contiguous_workloads() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
        let workload = ViewportWorkload {
            scale: Downscale::None,
            viewport_dims: (32, 16),
            tiles: vec![ViewportTile {
                source_roi: Rect {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 16,
                },
                dest: Rect {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 16,
                },
            }],
        };

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            if cfg!(target_os = "macos") {
                ViewportSurfaceStrategy::HybridContiguous
            } else {
                ViewportSurfaceStrategy::CpuContiguous
            }
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn viewport_direct_packet_detection_includes_fast422() {
        let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");

        assert!(has_direct_viewport_packet(&decoder));
    }

    #[test]
    fn auto_strategy_keeps_large_sparse_nonrestart_workloads_on_cpu_composite() {
        let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
        let contiguous = suggest_viewport_workload((2_048, 1_024)).expect("contiguous workload");
        let workload = sparse_workload_from(&contiguous);

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::CpuComposite
        );
    }

    #[test]
    fn auto_strategy_keeps_restart_coded_sparse_workloads_on_cpu_composite() {
        let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
        let contiguous = suggest_viewport_workload((8_192, 2_048)).expect("contiguous workload");
        let workload = sparse_workload_from(&contiguous);

        assert_eq!(
            choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
                .expect("strategy"),
            ViewportSurfaceStrategy::CpuComposite
        );
    }
}
