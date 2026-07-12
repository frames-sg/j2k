// SPDX-License-Identifier: MIT OR Apache-2.0

//! Resident-output validation and strategy selection.

use j2k_jpeg::{ColorSpace as JpegColorSpace, Decoder as CpuDecoder};

use super::{
    build_resolved_viewport_packet, validate_explicit_metal_viewport_request_with_packets,
    ViewportResidentOutputStrategy,
};
use crate::fast_packets::JpegFastPackets;
use crate::viewport::{
    is_contiguous_viewport_workload, validate_viewport_workload_budget, ViewportWorkload,
};
use crate::Error;

pub(in crate::viewport) fn validate_resident_viewport_composition_request(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
    external_live_bytes: usize,
) -> Result<(), Error> {
    validate_viewport_workload_budget(workload, external_live_bytes)?;
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

/// Choose the resident Metal strategy for a reusable viewport output request.
pub(crate) fn choose_resizable_metal_viewport_strategy(
    decoder: &CpuDecoder<'_>,
    workload: &ViewportWorkload,
) -> Result<ViewportResidentOutputStrategy, Error> {
    if is_contiguous_viewport_workload(workload) {
        let decoder_live_bytes = j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?;
        let (fast_packet, external_live_bytes) =
            build_resolved_viewport_packet(decoder, decoder_live_bytes)?;
        if validate_explicit_metal_viewport_request_with_packets(
            decoder,
            workload,
            JpegFastPackets::from_shared(fast_packet.as_ref()),
            external_live_bytes,
        )
        .is_ok()
        {
            return Ok(ViewportResidentOutputStrategy::DirectContiguous);
        }
    }

    validate_resident_viewport_composition_request(
        decoder,
        workload,
        j2k_jpeg::adapter::decoder_retained_allocation_bytes(decoder)?,
    )?;
    Ok(ViewportResidentOutputStrategy::Composite)
}

pub(in crate::viewport) fn choose_resizable_metal_viewport_strategy_for_decoder(
    decoder: &crate::Decoder<'_>,
    workload: &ViewportWorkload,
) -> Result<ViewportResidentOutputStrategy, Error> {
    if is_contiguous_viewport_workload(workload)
        && validate_explicit_metal_viewport_request_with_packets(
            decoder.inner(),
            workload,
            decoder.fast_packets(),
            decoder.retained_host_bytes()?,
        )
        .is_ok()
    {
        return Ok(ViewportResidentOutputStrategy::DirectContiguous);
    }

    validate_resident_viewport_composition_request(
        decoder.inner(),
        workload,
        decoder.retained_host_bytes()?,
    )?;
    Ok(ViewportResidentOutputStrategy::Composite)
}
