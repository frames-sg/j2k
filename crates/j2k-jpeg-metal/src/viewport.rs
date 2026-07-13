// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
use j2k_core::{BackendRequest, Downscale, Rect};
use j2k_jpeg::{Decoder as CpuDecoder, Rect as JpegRect, ScratchPool};

#[cfg(target_os = "macos")]
use crate::fast_packets::JpegFastPackets;
use crate::{Error, Surface};

mod cpu;
mod model;
mod policy;
#[cfg(all(target_os = "macos", test))]
mod resident;

#[cfg(test)]
pub(crate) use self::cpu::compose_viewport_cpu;
use self::cpu::compose_viewport_cpu_to_surface;
#[cfg(test)]
use self::cpu::cpu_viewport_allocation_budget_with_cap;
#[cfg(test)]
pub(crate) use self::cpu::decode_viewport_region_cpu;
pub(crate) use self::cpu::decode_viewport_region_cpu_to_surface;
#[cfg(target_os = "macos")]
pub(crate) use self::cpu::validate_viewport_tile_count;
pub use self::model::{
    is_contiguous_viewport_workload, suggest_viewport_workload, viewport_source_bounds,
    ViewportTile, ViewportWorkload,
};
#[cfg(test)]
use self::policy::choose_viewport_surface_strategy_for_decoder;
#[cfg(all(target_os = "macos", test))]
use self::policy::has_direct_viewport_packet;
use self::policy::resolve_viewport_surface_plan;
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::policy::{
    choose_resizable_metal_viewport_strategy, ViewportResidentOutputStrategy,
};
pub use self::policy::{choose_viewport_surface_strategy, ViewportSurfaceStrategy};
#[cfg(all(target_os = "macos", test))]
pub(crate) use self::resident::{
    compose_viewport_to_resizable_metal_buffer_with_session,
    compose_viewport_to_resizable_metal_textures_with_session,
    decode_viewport_region_to_resizable_metal_buffer_with_session,
    decode_viewport_region_to_resizable_metal_textures_with_session,
    decode_viewport_to_resizable_metal_buffer_with_decoder_session,
    decode_viewport_to_resizable_metal_buffer_with_session,
    decode_viewport_to_resizable_metal_textures_with_decoder_session,
    decode_viewport_to_resizable_metal_textures_with_session,
};

fn validate_viewport_workload_budget(
    workload: &ViewportWorkload,
    external_live_bytes: usize,
) -> Result<(), Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::with_external_live(
        "JPEG Metal viewport workload",
        external_live_bytes,
    );
    budget.account_capacity::<ViewportTile>(workload.tiles.capacity())?;
    Ok(())
}

/// Decode a viewport workload into a surface using the requested backend policy.
#[doc(hidden)]
pub fn decode_viewport_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    backend: BackendRequest,
) -> Result<Surface, Error> {
    let plan = resolve_viewport_surface_plan(decoder, workload, backend)?;
    validate_viewport_workload_budget(workload, plan.external_live_bytes)?;
    match plan.strategy {
        ViewportSurfaceStrategy::CpuComposite => compose_viewport_cpu_to_surface(
            decoder,
            pool,
            workload.scale,
            workload.viewport_dims,
            &workload.tiles,
            workload.tiles.capacity(),
            plan.external_live_bytes,
        ),
        ViewportSurfaceStrategy::CpuContiguous => {
            decode_viewport_region_cpu_to_surface(decoder, pool, workload, plan.external_live_bytes)
        }
        ViewportSurfaceStrategy::HybridComposite => compose_viewport_hybrid(
            decoder,
            pool,
            workload.scale,
            workload.viewport_dims,
            &workload.tiles,
            plan.external_live_bytes,
        ),
        ViewportSurfaceStrategy::HybridContiguous => decode_viewport_region_hybrid(
            decoder,
            pool,
            workload,
            plan.fast_packet,
            plan.external_live_bytes,
        ),
    }
}

#[cfg(target_os = "macos")]
/// Compose a multi-tile viewport through the Metal hybrid path.
pub(crate) fn compose_viewport_hybrid(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    scale: Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    crate::compute::compose_rgb_viewport_from_regions(
        decoder,
        pool,
        scale,
        viewport_dims,
        tiles,
        external_live_bytes,
    )
}

#[cfg(target_os = "macos")]
/// Decode a contiguous viewport region through the Metal hybrid path.
pub(crate) fn decode_viewport_region_hybrid(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    fast_packet: Option<crate::SharedJpegFastPacket>,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    let use_direct_kernel = decoder.info().restart_interval.is_some();
    let fast_packet = if use_direct_kernel { fast_packet } else { None };
    crate::compute::decode_region_scaled_to_surface(
        decoder,
        pool,
        PixelFormat::Rgb8,
        to_jpeg_rect(viewport_source_bounds(workload)),
        workload.scale,
        JpegFastPackets::from_shared(fast_packet.as_ref()),
        external_live_bytes,
    )
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for hybrid viewport decode requests.
pub(crate) fn decode_viewport_region_hybrid(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _workload: &ViewportWorkload,
    _fast_packet: Option<crate::SharedJpegFastPacket>,
    _external_live_bytes: usize,
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
}

#[cfg(not(target_os = "macos"))]
/// Return `Error::MetalUnavailable` for hybrid viewport composition requests.
pub(crate) fn compose_viewport_hybrid(
    _decoder: &CpuDecoder<'_>,
    _pool: &mut ScratchPool,
    _scale: Downscale,
    _viewport_dims: (u32, u32),
    _tiles: &[ViewportTile],
    _external_live_bytes: usize,
) -> Result<Surface, Error> {
    Err(Error::MetalUnavailable)
}

fn to_jpeg_rect(rect: Rect) -> JpegRect {
    JpegRect {
        x: rect.x,
        y: rect.y,
        w: rect.w,
        h: rect.h,
    }
}

#[cfg(test)]
mod tests;
