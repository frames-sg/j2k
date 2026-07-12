// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_jpeg::{adapter::decoder_bytes, Decoder as CpuDecoder, ScratchPool};

use super::policy::{
    choose_resizable_metal_viewport_strategy_for_decoder,
    validate_explicit_metal_viewport_request_with_packets,
    validate_resident_viewport_composition_request,
};
use super::{
    choose_resizable_metal_viewport_strategy, is_contiguous_viewport_workload,
    viewport_source_bounds, ViewportResidentOutputStrategy, ViewportWorkload,
};
use crate::{
    Error, MetalBackendSession, MetalBatchOutputBuffer, MetalBatchTextureOutput, MetalTextureTile,
    Surface,
};

/// Compose a viewport workload into a reusable caller-owned Metal buffer.
///
/// This path supports sparse and non-contiguous workloads. It decodes component
/// rows into reusable Metal plane buffers, resizes `output` to one RGB8 viewport
/// slot, and packs the composed viewport directly into that caller-owned buffer.
pub(crate) fn compose_viewport_to_resizable_metal_buffer_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    let external_live_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?;
    compose_viewport_to_resizable_metal_buffer_with_external_live(
        decoder,
        pool,
        workload,
        output,
        session,
        external_live_bytes,
    )
}

fn compose_viewport_to_resizable_metal_buffer_with_external_live(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
    external_live_bytes: usize,
) -> Result<Surface, Error> {
    validate_resident_viewport_composition_request(decoder, workload, external_live_bytes)?;
    output.ensure_rgb8_tiles(session, workload.viewport_dims, 1)?;
    crate::compute::compose_rgb_viewport_from_regions_into_output_with_session(
        decoder,
        pool,
        workload,
        output,
        session,
        external_live_bytes,
    )
}

/// Compose a viewport workload into a reusable caller-owned Metal texture.
///
/// This path supports sparse and non-contiguous workloads. It decodes component
/// rows into reusable Metal plane buffers, resizes `output` to one RGBA8
/// viewport slot, and packs the composed viewport directly into that
/// caller-owned texture.
pub(crate) fn compose_viewport_to_resizable_metal_textures_with_session(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    let external_live_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?;
    compose_viewport_to_resizable_metal_textures_with_external_live(
        decoder,
        pool,
        workload,
        output,
        session,
        external_live_bytes,
    )
}

fn compose_viewport_to_resizable_metal_textures_with_external_live(
    decoder: &CpuDecoder<'_>,
    pool: &mut ScratchPool,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
    external_live_bytes: usize,
) -> Result<MetalTextureTile, Error> {
    validate_resident_viewport_composition_request(decoder, workload, external_live_bytes)?;
    output.ensure_rgba8_tiles(session, workload.viewport_dims, 1)?;
    crate::compute::compose_rgb_viewport_from_regions_into_textures_with_session(
        decoder,
        pool,
        workload,
        output,
        session,
        external_live_bytes,
    )
}

/// Decode any viewport workload into a reusable caller-owned Metal buffer.
///
/// Contiguous workloads use the direct resident region-scaled batch path when
/// eligible. Sparse or unsupported direct shapes use resident component-row
/// composition into the same caller-owned RGB8 buffer.
pub(crate) fn decode_viewport_to_resizable_metal_buffer_with_session(
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

/// Decode any viewport workload into reusable caller-owned Metal textures.
///
/// Contiguous workloads use the direct resident region-scaled texture batch path
/// when eligible. Sparse or unsupported direct shapes use resident component-row
/// composition into the same caller-owned RGBA8 texture.
pub(crate) fn decode_viewport_to_resizable_metal_textures_with_session(
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

/// Decode any viewport workload into a reusable caller-owned Metal buffer using
/// an already parsed Metal decoder wrapper.
///
/// Contiguous workloads use the wrapper's cached fast-packet state for direct
/// resident region-scaled decode. Sparse or unsupported direct shapes use
/// resident component-row composition through the wrapper's CPU decoder.
pub(crate) fn decode_viewport_to_resizable_metal_buffer_with_decoder_session(
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
            compose_viewport_to_resizable_metal_buffer_with_external_live(
                decoder.inner(),
                pool,
                workload,
                output,
                session,
                decoder.retained_host_bytes()?,
            )
        }
    }
}

/// Decode any viewport workload into reusable caller-owned Metal textures using
/// an already parsed Metal decoder wrapper.
///
/// Contiguous workloads use the wrapper's cached fast-packet state for direct
/// resident region-scaled decode. Sparse or unsupported direct shapes use
/// resident component-row composition through the wrapper's CPU decoder.
pub(crate) fn decode_viewport_to_resizable_metal_textures_with_decoder_session(
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
            compose_viewport_to_resizable_metal_textures_with_external_live(
                decoder.inner(),
                pool,
                workload,
                output,
                session,
                decoder.retained_host_bytes()?,
            )
        }
    }
}

fn decode_viewport_region_to_resizable_metal_buffer_with_decoder_session(
    decoder: &crate::Decoder<'_>,
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    validate_explicit_metal_viewport_request_with_packets(
        decoder.inner(),
        workload,
        decoder.fast_packets(),
        decoder.retained_host_bytes()?,
    )?;
    if !is_contiguous_viewport_workload(workload) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport output currently requires a contiguous viewport workload",
        });
    }

    let source = viewport_source_bounds(workload);
    let scaled = source.scaled_covering(workload.scale);
    output.ensure_rgb8_tiles(session, (scaled.w, scaled.h), 1)?;
    let mut request = decoder
        .rgb8_region_scaled_metal_request(source, workload.scale)
        .with_output_slot(0);
    request.set_execution_owner_baseline(0, decoder.retained_host_bytes()?);
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

fn decode_viewport_region_to_resizable_metal_textures_with_decoder_session(
    decoder: &crate::Decoder<'_>,
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    validate_explicit_metal_viewport_request_with_packets(
        decoder.inner(),
        workload,
        decoder.fast_packets(),
        decoder.retained_host_bytes()?,
    )?;
    if !is_contiguous_viewport_workload(workload) {
        return Err(Error::UnsupportedMetalRequest {
            reason: "JPEG Metal reusable viewport texture output currently requires a contiguous viewport workload",
        });
    }

    let source = viewport_source_bounds(workload);
    let scaled = source.scaled_covering(workload.scale);
    output.ensure_rgba8_tiles(session, (scaled.w, scaled.h), 1)?;
    let mut request = decoder
        .rgb8_region_scaled_metal_request(source, workload.scale)
        .with_output_slot(0);
    request.set_execution_owner_baseline(0, decoder.retained_host_bytes()?);
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

/// Decode a contiguous viewport workload into a reusable caller-owned Metal buffer.
///
/// This is the resident-output counterpart to `decode_viewport_region_hybrid`:
/// it rejects non-contiguous workloads instead of compositing, resizes `output`
/// to one RGB8 viewport slot, and returns a `MetalResidentDecode` surface
/// backed by that caller-owned allocation.
pub(crate) fn decode_viewport_region_to_resizable_metal_buffer_with_session(
    input: &[u8],
    workload: &ViewportWorkload,
    output: &mut MetalBatchOutputBuffer,
    session: &MetalBackendSession,
) -> Result<Surface, Error> {
    let decoder = crate::Decoder::new(input)?;
    decode_viewport_region_to_resizable_metal_buffer_with_decoder_session(
        &decoder, workload, output, session,
    )
}

/// Decode a contiguous viewport workload into reusable caller-owned Metal textures.
///
/// This is the texture-output counterpart to
/// `decode_viewport_region_to_resizable_metal_buffer_with_session`: it rejects
/// non-contiguous workloads, resizes `output` to one RGBA8 viewport slot, and
/// returns a tile backed by that caller-owned texture.
pub(crate) fn decode_viewport_region_to_resizable_metal_textures_with_session(
    input: &[u8],
    workload: &ViewportWorkload,
    output: &mut MetalBatchTextureOutput,
    session: &MetalBackendSession,
) -> Result<MetalTextureTile, Error> {
    let decoder = crate::Decoder::new(input)?;
    decode_viewport_region_to_resizable_metal_textures_with_decoder_session(
        &decoder, workload, output, session,
    )
}
