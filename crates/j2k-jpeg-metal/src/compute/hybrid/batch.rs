// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reusable RGB8 batch outputs for batches the fast-packet batch kernels
//! cannot share: mixed sampling families, restart-coded 4:2:2/4:4:4 full
//! tiles, and every JPEG class without a fast packet. Each tile is
//! hybrid-decoded on the session runtime, then copied on the GPU into its
//! caller-owned output slot.

use j2k_core::PixelFormat;

use super::super::{
    copy_grouped_surfaces_to_output, copy_rgb8_surfaces_to_rgba_textures, CpuDecoder, Error,
    MetalRuntime, Surface,
};
use super::{decode_hybrid_to_surface, op_geometry};
use crate::batch::QueuedRequest;
use crate::batch_allocation::BatchMetadataBudget;

/// Hybrid-decode every request into `output` tiles, in request order.
pub(crate) fn decode_hybrid_rgb8_batch_into_output(
    runtime: &MetalRuntime,
    requests: &[QueuedRequest],
    output: &crate::MetalBatchOutputBuffer,
) -> Result<Vec<Result<Surface, Error>>, Error> {
    let dimensions = output.dimensions();
    let (indices, surfaces) = decode_tiles(runtime, requests, dimensions)?;
    let tile_len = (dimensions.0 as usize)
        .checked_mul(dimensions.1 as usize)
        .and_then(|pixels| pixels.checked_mul(PixelFormat::Rgb8.bytes_per_pixel()))
        .ok_or_else(|| super::plan::invariant("batch tile size overflowed"))?;
    let copied = copy_grouped_surfaces_to_output(
        runtime,
        output,
        dimensions,
        tile_len,
        &indices,
        surfaces,
        batch_external_live_bytes(requests),
    )?;
    in_request_order(copied, requests.len())
}

/// Hybrid-decode every request into RGBA8 `output` textures, in request
/// order.
pub(crate) fn decode_hybrid_rgb8_batch_into_textures(
    runtime: &MetalRuntime,
    requests: &[QueuedRequest],
    output: &crate::MetalBatchTextureOutput,
) -> Result<Vec<Result<crate::MetalTextureTile, Error>>, Error> {
    let dimensions = output.dimensions();
    let (indices, surfaces) = decode_tiles(runtime, requests, dimensions)?;
    let copied = copy_rgb8_surfaces_to_rgba_textures(
        runtime,
        output,
        dimensions,
        requests.len(),
        &indices,
        surfaces,
        batch_external_live_bytes(requests),
    )?;
    in_request_order(copied, requests.len())
}

type DecodedTiles = (Vec<usize>, Vec<Result<Surface, Error>>);

fn decode_tiles(
    runtime: &MetalRuntime,
    requests: &[QueuedRequest],
    dimensions: (u32, u32),
) -> Result<DecodedTiles, Error> {
    let mut budget = BatchMetadataBudget::new("JPEG Metal hybrid batch tiles");
    let mut indices = budget.try_vec(requests.len(), "JPEG Metal hybrid batch tile indices")?;
    let mut surfaces = budget.try_vec(requests.len(), "JPEG Metal hybrid batch tile surfaces")?;
    for (index, request) in requests.iter().enumerate() {
        indices.push(index);
        surfaces.push(decode_tile(runtime, request, dimensions));
    }
    Ok((indices, surfaces))
}

fn decode_tile(
    runtime: &MetalRuntime,
    request: &QueuedRequest,
    dimensions: (u32, u32),
) -> Result<Surface, Error> {
    let decoder: CpuDecoder<'_> = request
        .input
        .decoder_with_external_live(request.execution_external_live_bytes())?;
    let (roi, scale) = op_geometry(&decoder, request.op);
    let surface = decode_hybrid_to_surface(runtime, &decoder, PixelFormat::Rgb8, roi, scale)?;
    // The GPU copy moves whole tiles, so every tile must fill its slot.
    if j2k_core::DeviceSurface::dimensions(&surface) != dimensions {
        return Err(Error::capability_rejected(
            j2k_core::CapabilityRejection::unsupported_format(
                "JPEG Metal reusable RGB8 batch output requires matching output dimensions",
            ),
        ));
    }
    Ok(surface)
}

fn batch_external_live_bytes(requests: &[QueuedRequest]) -> usize {
    requests
        .iter()
        .map(QueuedRequest::execution_external_live_bytes)
        .max()
        .unwrap_or(0)
}

fn in_request_order<T>(
    results: Vec<(usize, Result<T, Error>)>,
    count: usize,
) -> Result<Vec<Result<T, Error>>, Error> {
    let mut ordered: Vec<Option<Result<T, Error>>> =
        BatchMetadataBudget::new("JPEG Metal hybrid batch results")
            .try_vec(count, "JPEG Metal hybrid batch result slots")?;
    ordered.resize_with(count, || None);
    for (index, result) in results {
        let slot = ordered
            .get_mut(index)
            .ok_or_else(|| super::plan::invariant("batch result index out of range"))?;
        *slot = Some(result);
    }
    let mut results = BatchMetadataBudget::new("JPEG Metal hybrid batch results")
        .try_vec(count, "JPEG Metal hybrid batch ordered results")?;
    for slot in ordered {
        results.push(slot.ok_or_else(|| super::plan::invariant("batch result missing"))?);
    }
    Ok(results)
}
