// SPDX-License-Identifier: MIT OR Apache-2.0

//! Final borrowed tile-part views and codestream output high-water accounting.

use alloc::vec::Vec;

use crate::j2c::codestream_write;

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::single_tile::ownership::encode_params_retained_bytes;
use super::super::tile_parts::{encoded_tile_parts_retained_bytes, EncodedTilePart};
use super::super::{
    validate_packet_header_marker_payloads, NativeEncodePipelineError, NativeEncodePipelineResult,
    NativeEncodeSession,
};

pub(in crate::j2c::encode) fn finalize_multitile_codestream(
    params: &codestream_write::EncodeParams,
    tile_bodies: &Vec<EncodedTilePart>,
    quant_params: &[(u16, u16)],
    planning_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let base_bytes = finalization_base_bytes(params, tile_bodies, planning_bytes)?;
    validate_multitile_packet_headers(params, tile_bodies, base_bytes, session)?;
    let (tile_parts, final_retained_bytes) =
        build_tile_part_views(tile_bodies, base_bytes, session)?;
    let accounted = codestream_write::write_codestream_tiles_accounted_with_peak_check(
        params,
        &tile_parts,
        quant_params,
        |writer_peak_bytes| {
            session
                .checked_phase(
                    checked_add_bytes(
                        final_retained_bytes,
                        writer_peak_bytes,
                        "multi-tile codestream writer peak",
                    )?,
                    "multi-tile codestream writer peak",
                )
                .map(|_| ())
        },
    )?;
    if accounted.writer_peak_bytes != accounted.codestream.capacity() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "multi-tile codestream writer peak disagrees with output capacity",
        ));
    }
    Ok(accounted.codestream)
}

fn finalization_base_bytes(
    params: &codestream_write::EncodeParams,
    tile_bodies: &Vec<EncodedTilePart>,
    planning_bytes: usize,
) -> NativeEncodePipelineResult<usize> {
    let retained_tile_bytes =
        encoded_tile_parts_retained_bytes(tile_bodies, tile_bodies.capacity())?;
    let planning_owners = checked_add_bytes(
        encode_params_retained_bytes(params)?,
        planning_bytes,
        "multi-tile final planning owners",
    )?;
    let base_bytes = checked_add_bytes(
        retained_tile_bytes,
        planning_owners,
        "multi-tile final retained owners",
    )?;
    Ok(base_bytes)
}

fn validate_multitile_packet_headers(
    params: &codestream_write::EncodeParams,
    tile_bodies: &[EncodedTilePart],
    base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<()> {
    if !params.write_ppm && !params.write_ppt {
        return Ok(());
    }
    let requested_bytes =
        checked_element_bytes::<&[Vec<u8>]>(tile_bodies.len(), "multi-tile packet-header views")?;
    session.checked_phase(
        checked_add_bytes(
            base_bytes,
            requested_bytes,
            "multi-tile packet-header views",
        )?,
        "multi-tile packet-header views",
    )?;
    let mut header_views = Vec::new();
    header_views
        .try_reserve_exact(tile_bodies.len())
        .map_err(|_| host_allocation_failed("multi-tile packet-header views", requested_bytes))?;
    header_views.extend(
        tile_bodies
            .iter()
            .map(|tile| tile.packet_headers.as_slice()),
    );
    let actual_bytes = checked_element_bytes::<&[Vec<u8>]>(
        header_views.capacity(),
        "multi-tile packet-header views",
    )?;
    session.checked_phase(
        checked_add_bytes(base_bytes, actual_bytes, "multi-tile packet-header views")?,
        "multi-tile packet-header views",
    )?;
    validate_packet_header_marker_payloads(params.write_ppm, params.write_ppt, &header_views)?;
    Ok(())
}

fn build_tile_part_views<'a>(
    tile_bodies: &'a [EncodedTilePart],
    base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<(Vec<codestream_write::TilePartData<'a>>, usize)> {
    let requested_bytes = checked_element_bytes::<codestream_write::TilePartData<'_>>(
        tile_bodies.len(),
        "multi-tile codestream part views",
    )?;
    session.checked_phase(
        checked_add_bytes(
            base_bytes,
            requested_bytes,
            "multi-tile codestream part views",
        )?,
        "multi-tile codestream part views",
    )?;
    let mut tile_parts = Vec::new();
    tile_parts
        .try_reserve_exact(tile_bodies.len())
        .map_err(|_| host_allocation_failed("multi-tile codestream part views", requested_bytes))?;
    tile_parts.extend(
        tile_bodies
            .iter()
            .map(|tile| codestream_write::TilePartData {
                tile_index: tile.tile_index,
                tile_part_index: tile.tile_part_index,
                num_tile_parts: tile.num_tile_parts,
                data: &tile.data,
                packet_lengths: &tile.packet_lengths,
                packet_headers: &tile.packet_headers,
            }),
    );
    let actual_bytes = checked_element_bytes::<codestream_write::TilePartData<'_>>(
        tile_parts.capacity(),
        "multi-tile codestream part views",
    )?;
    let retained_bytes = checked_add_bytes(
        base_bytes,
        actual_bytes,
        "multi-tile codestream finalization",
    )?;
    session.checked_phase(retained_bytes, "multi-tile codestream finalization")?;
    Ok((tile_parts, retained_bytes))
}

#[cfg(test)]
#[path = "finalize/tests.rs"]
mod tests;
