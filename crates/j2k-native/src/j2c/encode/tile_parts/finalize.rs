// SPDX-License-Identifier: MIT OR Apache-2.0

//! Borrowed single-tile part planning and phase-bounded codestream handoff.

use alloc::vec::Vec;
use core::ops::Range;

use crate::j2c::{codestream_write, packet_encode};

use super::super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::super::{NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession};
use super::validate_packet_header_marker_payload;

#[derive(Debug)]
struct BorrowedTilePartRange {
    tile_part_index: u8,
    num_tile_parts: u8,
    data: Range<usize>,
    packets: Range<usize>,
}

/// Finalize one packetized tile without cloning its payload, packet lengths,
/// or separated headers into temporary owned tile parts.
pub(in crate::j2c::encode) fn write_single_tile_packetized_codestream_for_session(
    params: &codestream_write::EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
    quant_params: &[(u16, u16)],
    tile_part_packet_limit: Option<u16>,
    retained_phase_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    validate_packet_header_marker_payload(params, packetized_tile)?;
    let packetized_bytes = packet_encode::packetized_tile_retained_bytes(packetized_tile)?;
    let base_bytes = checked_add_bytes(
        retained_phase_bytes,
        packetized_bytes,
        "single-tile codestream retained owners",
    )?;
    if tile_part_packet_limit.is_none()
        && !(params.write_plt || params.write_plm || params.write_ppm || params.write_ppt)
    {
        let accounted = codestream_write::write_codestream_accounted_with_peak_check(
            params,
            &packetized_tile.data,
            quant_params,
            |writer_peak_bytes| {
                session
                    .checked_phase(
                        checked_add_bytes(
                            base_bytes,
                            writer_peak_bytes,
                            "single-tile codestream writer peak",
                        )?,
                        "single-tile codestream writer peak",
                    )
                    .map(|_| ())
            },
        )?;
        if accounted.writer_peak_bytes != accounted.codestream.capacity() {
            return Err(NativeEncodePipelineError::internal_invariant(
                "single-tile codestream writer peak disagrees with output capacity",
            ));
        }
        return Ok(accounted.codestream);
    }
    write_split_tile_codestream(
        params,
        packetized_tile,
        quant_params,
        tile_part_packet_limit,
        base_bytes,
        session,
    )
}

fn write_split_tile_codestream(
    params: &codestream_write::EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
    quant_params: &[(u16, u16)],
    tile_part_packet_limit: Option<u16>,
    base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<u8>> {
    let ranges = borrowed_tile_part_ranges(
        packetized_tile.data.len(),
        &packetized_tile.packet_lengths,
        packetized_tile.packet_headers.len(),
        tile_part_packet_limit,
        base_bytes,
        session,
    )?;
    let range_bytes = checked_element_bytes::<BorrowedTilePartRange>(
        ranges.capacity(),
        "single-tile borrowed part ranges",
    )?;
    let base_with_ranges =
        checked_add_bytes(base_bytes, range_bytes, "single-tile borrowed part ranges")?;
    let (tile_parts, actual_view_bytes) =
        build_tile_part_views(packetized_tile, &ranges, base_with_ranges, session)?;
    drop(ranges);
    let live_before_output = checked_add_bytes(
        base_bytes,
        actual_view_bytes,
        "single-tile codestream finalization",
    )?;
    session.checked_phase(live_before_output, "single-tile codestream finalization")?;

    let accounted = codestream_write::write_codestream_tiles_accounted_with_peak_check(
        params,
        &tile_parts,
        quant_params,
        |writer_peak_bytes| {
            session
                .checked_phase(
                    checked_add_bytes(
                        live_before_output,
                        writer_peak_bytes,
                        "single-tile codestream writer peak",
                    )?,
                    "single-tile codestream writer peak",
                )
                .map(|_| ())
        },
    )?;
    if accounted.writer_peak_bytes != accounted.codestream.capacity() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "split-tile codestream writer peak disagrees with output capacity",
        ));
    }
    Ok(accounted.codestream)
}

fn build_tile_part_views<'a>(
    packetized_tile: &'a packet_encode::PacketizedTileData,
    ranges: &[BorrowedTilePartRange],
    base_with_ranges: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<(Vec<codestream_write::TilePartData<'a>>, usize)> {
    let requested_bytes = checked_element_bytes::<codestream_write::TilePartData<'_>>(
        ranges.len(),
        "single-tile codestream part views",
    )?;
    session.checked_phase(
        checked_add_bytes(
            base_with_ranges,
            requested_bytes,
            "single-tile codestream part views",
        )?,
        "single-tile codestream part views",
    )?;
    let mut tile_parts = Vec::new();
    tile_parts.try_reserve_exact(ranges.len()).map_err(|_| {
        host_allocation_failed("single-tile codestream part views", requested_bytes)
    })?;
    for range in ranges {
        let packet_headers = if packetized_tile.packet_headers.is_empty() {
            &[]
        } else {
            &packetized_tile.packet_headers[range.packets.clone()]
        };
        tile_parts.push(codestream_write::TilePartData {
            tile_index: 0,
            tile_part_index: range.tile_part_index,
            num_tile_parts: range.num_tile_parts,
            data: &packetized_tile.data[range.data.clone()],
            packet_lengths: &packetized_tile.packet_lengths[range.packets.clone()],
            packet_headers,
        });
    }
    let actual_bytes = checked_element_bytes::<codestream_write::TilePartData<'_>>(
        tile_parts.capacity(),
        "single-tile codestream part views",
    )?;
    session.checked_phase(
        checked_add_bytes(
            base_with_ranges,
            actual_bytes,
            "single-tile codestream part views",
        )?,
        "single-tile codestream part views",
    )?;
    Ok((tile_parts, actual_bytes))
}

fn borrowed_tile_part_ranges(
    data_len: usize,
    packet_lengths: &[u32],
    packet_header_count: usize,
    packet_limit: Option<u16>,
    retained_phase_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<BorrowedTilePartRange>> {
    if packet_header_count != 0 && packet_header_count != packet_lengths.len() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "packet header count does not match packet length count",
        ));
    }
    let Some(packet_limit) = packet_limit else {
        return single_borrowed_tile_part(
            data_len,
            packet_lengths.len(),
            retained_phase_bytes,
            session,
        );
    };
    if packet_limit == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "tile-part packet limit must be non-zero",
        ));
    }
    if packet_lengths.is_empty() {
        return single_borrowed_tile_part(data_len, 0, retained_phase_bytes, session);
    }

    let expected_len = packet_lengths.iter().try_fold(0usize, |acc, &len| {
        acc.checked_add(usize::try_from(len).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("packet length exceeds host usize")
        })?)
        .ok_or(NativeEncodePipelineError::arithmetic_overflow(
            "packet length sum",
        ))
    })?;
    if expected_len != data_len {
        return Err(NativeEncodePipelineError::internal_invariant(
            "packet lengths do not match tile data length",
        ));
    }

    let packet_limit = usize::from(packet_limit);
    let part_count = packet_lengths.len().div_ceil(packet_limit);
    if part_count > usize::from(u8::MAX) {
        return Err(NativeEncodePipelineError::unsupported(
            "tile-part packet limit would emit more than 255 tile-parts",
        ));
    }
    let num_tile_parts = u8::try_from(part_count)
        .map_err(|_| NativeEncodePipelineError::internal_invariant("tile-part count exceeds u8"))?;
    let requested_bytes = checked_element_bytes::<BorrowedTilePartRange>(
        part_count,
        "single-tile borrowed part ranges",
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_phase_bytes,
            requested_bytes,
            "single-tile borrowed part ranges",
        )?,
        "single-tile borrowed part ranges",
    )?;
    let mut parts = Vec::new();
    parts
        .try_reserve_exact(part_count)
        .map_err(|_| host_allocation_failed("single-tile borrowed part ranges", requested_bytes))?;
    let mut data_offset = 0usize;
    for (tile_part_index, packet_chunk) in packet_lengths.chunks(packet_limit).enumerate() {
        let data_start = data_offset;
        let chunk_len = packet_chunk.iter().try_fold(0usize, |acc, &len| {
            acc.checked_add(usize::try_from(len).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow("packet length exceeds host usize")
            })?)
            .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                "packet length sum",
            ))
        })?;
        data_offset = data_offset.checked_add(chunk_len).ok_or(
            NativeEncodePipelineError::arithmetic_overflow("packet length sum"),
        )?;
        let packet_start = tile_part_index.checked_mul(packet_limit).ok_or(
            NativeEncodePipelineError::arithmetic_overflow("tile-part packet range"),
        )?;
        let packet_end = packet_start.checked_add(packet_chunk.len()).ok_or(
            NativeEncodePipelineError::arithmetic_overflow("tile-part packet range"),
        )?;
        parts.push(BorrowedTilePartRange {
            tile_part_index: u8::try_from(tile_part_index).map_err(|_| {
                NativeEncodePipelineError::internal_invariant("tile-part index exceeds u8")
            })?,
            num_tile_parts,
            data: data_start..data_offset,
            packets: packet_start..packet_end,
        });
    }
    Ok(parts)
}

fn single_borrowed_tile_part(
    data_len: usize,
    packet_count: usize,
    retained_phase_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<BorrowedTilePartRange>> {
    let requested_bytes = core::mem::size_of::<BorrowedTilePartRange>();
    session.checked_phase(
        checked_add_bytes(
            retained_phase_bytes,
            requested_bytes,
            "single-tile borrowed part range",
        )?,
        "single-tile borrowed part range",
    )?;
    let mut parts = Vec::new();
    parts
        .try_reserve_exact(1)
        .map_err(|_| host_allocation_failed("single-tile borrowed part range", requested_bytes))?;
    parts.push(BorrowedTilePartRange {
        tile_part_index: 0,
        num_tile_parts: 1,
        data: 0..data_len,
        packets: 0..packet_count,
    });
    Ok(parts)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
