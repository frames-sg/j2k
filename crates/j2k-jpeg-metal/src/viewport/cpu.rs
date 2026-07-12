// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{Downscale, PixelFormat, Rect};
use j2k_jpeg::{Decoder as CpuDecoder, ScratchPool};

use super::{
    model::CpuViewportComposeRequest, to_jpeg_rect, validate_viewport_workload_budget,
    viewport_source_bounds, ViewportTile, ViewportWorkload,
};
use crate::{Error, Surface};

pub(crate) fn validate_viewport_tile_count(
    tiles: &[ViewportTile],
    external_live_bytes: usize,
) -> Result<(), Error> {
    crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal viewport tile metadata",
        external_live_bytes,
    )
    .preflight(&[crate::batch_allocation::BatchMetadataRequest::of::<
        ViewportTile,
    >(tiles.len())])?;
    Ok(())
}

#[cfg(test)]
pub(super) fn cpu_viewport_allocation_budget_with_cap(
    tile_capacity: usize,
    viewport_len: usize,
    tile_scratch_len: usize,
    cap: usize,
) -> Result<crate::batch_allocation::BatchMetadataBudget, j2k_core::BatchInfrastructureError> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::with_cap(
        "JPEG Metal CPU viewport live allocation",
        cap,
    );
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<ViewportTile>(tile_capacity),
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(viewport_len),
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(tile_scratch_len),
    ])?;
    budget.account_capacity::<ViewportTile>(tile_capacity)?;
    Ok(budget)
}

fn cpu_viewport_allocation_budget(
    tile_capacity: usize,
    viewport_len: usize,
    tile_scratch_len: usize,
    external_live_bytes: usize,
) -> Result<crate::batch_allocation::BatchMetadataBudget, j2k_core::BatchInfrastructureError> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal CPU viewport live allocation",
        external_live_bytes,
    );
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<ViewportTile>(tile_capacity),
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(viewport_len),
        crate::batch_allocation::BatchMetadataRequest::of::<u8>(tile_scratch_len),
    ])?;
    budget.account_capacity::<ViewportTile>(tile_capacity)?;
    Ok(budget)
}

/// Decode each viewport tile on CPU and composite the result into host bytes.
#[cfg(test)]
pub(crate) fn compose_viewport_cpu(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    fmt: PixelFormat,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Vec<u8>, Error> {
    compose_viewport_cpu_with_metadata_capacity(
        decoder,
        pool,
        &CpuViewportComposeRequest {
            fmt,
            scale,
            viewport_dims,
            tiles,
            tile_metadata_capacity: tiles.len(),
            external_live_bytes: 0,
        },
    )
}

fn compose_viewport_cpu_with_metadata_capacity(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    request: &CpuViewportComposeRequest<'_>,
) -> Result<Vec<u8>, Error> {
    let CpuViewportComposeRequest {
        fmt,
        scale,
        viewport_dims,
        tiles,
        tile_metadata_capacity,
        external_live_bytes,
    } = *request;
    validate_viewport_tile_count(tiles, external_live_bytes)?;
    let bpp = fmt.bytes_per_pixel();
    let viewport_stride = crate::batch_allocation::checked_count_product(
        viewport_dims.0 as usize,
        bpp,
        "JPEG Metal viewport row bytes",
    )?;
    let viewport_len = crate::batch_allocation::checked_count_product(
        viewport_stride,
        viewport_dims.1 as usize,
        "JPEG Metal viewport output bytes",
    )?;
    let max_tile_len = tiles.iter().try_fold(0usize, |largest, tile| {
        let scaled = tile.source_roi.scaled_covering(scale);
        let pixels = crate::batch_allocation::checked_count_product(
            scaled.w as usize,
            scaled.h as usize,
            "JPEG Metal viewport tile pixels",
        )?;
        let bytes = crate::batch_allocation::checked_count_product(
            pixels,
            bpp,
            "JPEG Metal viewport tile bytes",
        )?;
        Ok::<_, j2k_core::BatchInfrastructureError>(largest.max(bytes))
    })?;
    let mut budget = cpu_viewport_allocation_budget(
        tile_metadata_capacity,
        viewport_len,
        max_tile_len,
        external_live_bytes,
    )?;
    let mut viewport = budget.try_filled(viewport_len, 0u8, "JPEG Metal viewport output")?;
    let mut tile_bytes =
        budget.try_filled(max_tile_len, 0u8, "JPEG Metal viewport tile scratch")?;

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
        let tile_stride = crate::batch_allocation::checked_count_product(
            tile_dims.0 as usize,
            bpp,
            "JPEG Metal viewport tile row bytes",
        )?;
        let tile_len = crate::batch_allocation::checked_count_product(
            tile_stride,
            tile_dims.1 as usize,
            "JPEG Metal viewport tile bytes",
        )?;
        tile_bytes[..tile_len].fill(0);
        decoder.decode_region_scaled_into_with_scratch(
            pool,
            &mut tile_bytes[..tile_len],
            tile_stride,
            fmt,
            to_jpeg_rect(tile.source_roi),
            scale,
        )?;
        blit_into_viewport(
            &tile_bytes[..tile_len],
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
pub(crate) fn decode_viewport_region_cpu(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    fmt: PixelFormat,
    workload: &ViewportWorkload,
    external_live_bytes: usize,
) -> Result<Vec<u8>, Error> {
    validate_viewport_workload_budget(workload, external_live_bytes)?;
    let source = viewport_source_bounds(workload);
    let stride = crate::batch_allocation::checked_count_product(
        workload.viewport_dims.0 as usize,
        fmt.bytes_per_pixel(),
        "JPEG Metal contiguous viewport row bytes",
    )?;
    let output_len = crate::batch_allocation::checked_count_product(
        stride,
        workload.viewport_dims.1 as usize,
        "JPEG Metal contiguous viewport bytes",
    )?;
    let mut budget = cpu_viewport_allocation_budget(
        workload.tiles.capacity(),
        output_len,
        0,
        external_live_bytes,
    )?;
    let mut viewport = budget.try_filled(output_len, 0u8, "JPEG Metal contiguous viewport")?;
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

/// Decode the contiguous source region on CPU and return a surface.
pub(crate) fn decode_viewport_region_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let bytes = decode_viewport_region_cpu(
        decoder,
        pool,
        PixelFormat::Rgb8,
        workload,
        external_live_bytes,
    )?;
    crate::upload_surface(
        bytes,
        workload.viewport_dims,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cpu,
    )
}

/// Decode and composite viewport tiles on CPU, then return a surface.
pub(super) fn compose_viewport_cpu_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
    tile_metadata_capacity: usize,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let bytes = compose_viewport_cpu_with_metadata_capacity(
        decoder,
        pool,
        &CpuViewportComposeRequest {
            fmt: PixelFormat::Rgb8,
            scale,
            viewport_dims,
            tiles,
            tile_metadata_capacity,
            external_live_bytes,
        },
    )?;
    crate::upload_surface(
        bytes,
        viewport_dims,
        PixelFormat::Rgb8,
        j2k_core::BackendRequest::Cpu,
    )
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
