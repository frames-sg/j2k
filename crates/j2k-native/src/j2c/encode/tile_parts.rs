// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::codestream_write::EncodeParams;
use super::super::packet_encode;
use super::allocation::{checked_add_bytes, checked_element_bytes};
use super::{NativeEncodePipelineError, NativeEncodePipelineResult};

mod finalize;
pub(super) use finalize::write_single_tile_packetized_codestream_for_session;
mod consume;
pub(in crate::j2c::encode) use consume::consume_packetized_tile_into_tile_parts;

pub(super) struct EncodedTilePart {
    pub(super) tile_index: u16,
    pub(super) tile_part_index: u8,
    pub(super) num_tile_parts: u8,
    pub(super) data: Vec<u8>,
    pub(super) packet_lengths: Vec<u32>,
    pub(super) packet_headers: Vec<Vec<u8>>,
}

pub(super) fn encoded_tile_parts_retained_bytes(
    parts: &[EncodedTilePart],
    outer_capacity: usize,
) -> crate::EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<EncodedTilePart>(
        outer_capacity,
        "retained multi-tile part owners",
    )?;
    for part in parts {
        bytes = checked_add_bytes(bytes, part.data.capacity(), "retained multi-tile payloads")?;
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<u32>(
                part.packet_lengths.capacity(),
                "retained multi-tile packet lengths",
            )?,
            "retained multi-tile packet lengths",
        )?;
        bytes = checked_add_bytes(
            bytes,
            packet_headers_retained_bytes(&part.packet_headers)?,
            "retained multi-tile packet headers",
        )?;
    }
    Ok(bytes)
}

pub(super) fn packet_headers_retained_bytes(headers: &Vec<Vec<u8>>) -> crate::EncodeResult<usize> {
    let mut bytes =
        checked_element_bytes::<Vec<u8>>(headers.capacity(), "retained packet-header owners")?;
    for header in headers {
        bytes = checked_add_bytes(bytes, header.capacity(), "retained packet-header payloads")?;
    }
    Ok(bytes)
}

fn validate_packet_header_marker_payload(
    params: &EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
) -> NativeEncodePipelineResult<()> {
    if !params.write_ppm && !params.write_ppt {
        return Ok(());
    }
    if params.write_ppm && params.write_ppt {
        return Err(NativeEncodePipelineError::invalid_input(
            "PPM and PPT packet header markers are mutually exclusive",
        ));
    }
    validate_packet_header_marker_payloads(
        params.write_ppm,
        params.write_ppt,
        &[&packetized_tile.packet_headers],
    )?;
    Ok(())
}

#[expect(
    clippy::similar_names,
    reason = "paired axis, subband, and marker names follow JPEG 2000 specification notation"
)]
pub(super) fn validate_packet_header_marker_payloads(
    write_ppm: bool,
    write_ppt: bool,
    tile_packet_headers: &[&[Vec<u8>]],
) -> NativeEncodePipelineResult<()> {
    const PACKET_HEADER_MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
    const PPM_PACKET_HEADER_LIMIT: usize = PACKET_HEADER_MARKER_PAYLOAD_LIMIT - 2;
    const MAX_PACKET_HEADER_MARKERS: usize = u8::MAX as usize + 1;

    if !write_ppm && !write_ppt {
        return Ok(());
    }
    if write_ppm && write_ppt {
        return Err(NativeEncodePipelineError::invalid_input(
            "PPM and PPT packet header markers are mutually exclusive",
        ));
    }
    if tile_packet_headers.iter().any(|headers| headers.is_empty()) {
        return Err(NativeEncodePipelineError::internal_invariant(
            "PPM/PPT encode requires separated packet headers",
        ));
    }
    if write_ppm {
        let mut marker_count = 0usize;
        let mut payload_len = 0usize;
        for header in tile_packet_headers
            .iter()
            .flat_map(|headers| headers.iter())
        {
            if header.len() > PPM_PACKET_HEADER_LIMIT {
                return Err(NativeEncodePipelineError::unsupported(
                    "PPM packet header exceeds marker payload limit",
                ));
            }
            let entry_len = 2usize.checked_add(header.len()).ok_or(
                NativeEncodePipelineError::arithmetic_overflow("PPM marker payload length"),
            )?;
            if payload_len == 0 {
                marker_count = marker_count.checked_add(1).ok_or(
                    NativeEncodePipelineError::arithmetic_overflow("PPM marker count"),
                )?;
            } else if payload_len
                .checked_add(entry_len)
                .is_none_or(|len| len > PACKET_HEADER_MARKER_PAYLOAD_LIMIT)
            {
                marker_count = marker_count.checked_add(1).ok_or(
                    NativeEncodePipelineError::arithmetic_overflow("PPM marker count"),
                )?;
                payload_len = 0;
            }
            payload_len = payload_len.checked_add(entry_len).ok_or(
                NativeEncodePipelineError::arithmetic_overflow("PPM marker payload length"),
            )?;
            if marker_count > MAX_PACKET_HEADER_MARKERS {
                return Err(NativeEncodePipelineError::unsupported(
                    "PPM packet headers require more than 256 marker segments",
                ));
            }
        }
    }
    if write_ppt {
        for headers in tile_packet_headers {
            let payload_len = headers.iter().try_fold(0usize, |acc, header| {
                acc.checked_add(header.len())
                    .ok_or(NativeEncodePipelineError::arithmetic_overflow(
                        "PPT marker payload length",
                    ))
            })?;
            let marker_count = payload_len.div_ceil(PACKET_HEADER_MARKER_PAYLOAD_LIMIT);
            if marker_count > MAX_PACKET_HEADER_MARKERS {
                return Err(NativeEncodePipelineError::unsupported(
                    "PPT packet headers require more than 256 marker segments",
                ));
            }
        }
    }
    Ok(())
}
