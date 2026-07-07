// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::super::codestream_write::{self, EncodeParams};
use super::super::packet_encode;

pub(super) struct EncodedTilePart {
    pub(super) tile_index: u16,
    pub(super) tile_part_index: u8,
    pub(super) num_tile_parts: u8,
    pub(super) data: Vec<u8>,
    pub(super) packet_lengths: Vec<u32>,
    pub(super) packet_headers: Vec<Vec<u8>>,
}

pub(super) fn split_packetized_tile_into_tile_parts(
    tile_index: u16,
    data: &[u8],
    packet_lengths: &[u32],
    packet_headers: &[Vec<u8>],
    packet_limit: Option<u16>,
) -> Result<Vec<EncodedTilePart>, &'static str> {
    if !packet_headers.is_empty() && packet_headers.len() != packet_lengths.len() {
        return Err("packet header count does not match packet length count");
    }
    let Some(packet_limit) = packet_limit else {
        return Ok(vec![EncodedTilePart {
            tile_index,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: data.to_vec(),
            packet_lengths: packet_lengths.to_vec(),
            packet_headers: packet_headers.to_vec(),
        }]);
    };
    if packet_limit == 0 {
        return Err("tile-part packet limit must be non-zero");
    }
    if packet_lengths.is_empty() {
        return Ok(vec![EncodedTilePart {
            tile_index,
            tile_part_index: 0,
            num_tile_parts: 1,
            data: data.to_vec(),
            packet_lengths: Vec::new(),
            packet_headers: Vec::new(),
        }]);
    }

    let expected_len = packet_lengths.iter().try_fold(0usize, |acc, &len| {
        acc.checked_add(usize::try_from(len).map_err(|_| "packet length exceeds usize")?)
            .ok_or("packet length sum overflow")
    })?;
    if expected_len != data.len() {
        return Err("packet lengths do not match tile data length");
    }

    let packet_limit = usize::from(packet_limit);
    let num_tile_parts = packet_lengths.len().div_ceil(packet_limit);
    if num_tile_parts > usize::from(u8::MAX) {
        return Err("tile-part packet limit would emit more than 255 tile-parts");
    }
    let num_tile_parts = u8::try_from(num_tile_parts).map_err(|_| "tile-part count exceeds u8")?;

    let mut parts = Vec::with_capacity(usize::from(num_tile_parts));
    let mut data_offset = 0usize;
    for (tile_part_index, packet_chunk) in packet_lengths.chunks(packet_limit).enumerate() {
        let chunk_len = packet_chunk.iter().try_fold(0usize, |acc, &len| {
            acc.checked_add(usize::try_from(len).map_err(|_| "packet length exceeds usize")?)
                .ok_or("packet length sum overflow")
        })?;
        let end = data_offset
            .checked_add(chunk_len)
            .ok_or("packet length sum overflow")?;
        let tile_part_index =
            u8::try_from(tile_part_index).map_err(|_| "tile-part index exceeds u8")?;
        parts.push(EncodedTilePart {
            tile_index,
            tile_part_index,
            num_tile_parts,
            data: data[data_offset..end].to_vec(),
            packet_lengths: packet_chunk.to_vec(),
            packet_headers: if packet_headers.is_empty() {
                Vec::new()
            } else {
                let packet_start = tile_part_index as usize * packet_limit;
                let packet_end = packet_start + packet_chunk.len();
                packet_headers[packet_start..packet_end].to_vec()
            },
        });
        data_offset = end;
    }
    Ok(parts)
}

pub(super) fn write_single_tile_packetized_codestream(
    params: &EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
    quant_params: &[(u16, u16)],
    tile_part_packet_limit: Option<u16>,
) -> Result<Vec<u8>, &'static str> {
    validate_packet_header_marker_payload(params, packetized_tile)?;
    let tile_parts = split_packetized_tile_into_tile_parts(
        0,
        &packetized_tile.data,
        &packetized_tile.packet_lengths,
        &packetized_tile.packet_headers,
        tile_part_packet_limit,
    )?;
    let codestream_tile_parts = tile_parts
        .iter()
        .map(|part| codestream_write::TilePartData {
            tile_index: part.tile_index,
            tile_part_index: part.tile_part_index,
            num_tile_parts: part.num_tile_parts,
            data: &part.data,
            packet_lengths: &part.packet_lengths,
            packet_headers: &part.packet_headers,
        })
        .collect::<Vec<_>>();
    Ok(codestream_write::write_codestream_tiles(
        params,
        &codestream_tile_parts,
        quant_params,
    ))
}

fn validate_packet_header_marker_payload(
    params: &EncodeParams,
    packetized_tile: &packet_encode::PacketizedTileData,
) -> Result<(), &'static str> {
    if !params.write_ppm && !params.write_ppt {
        return Ok(());
    }
    if params.write_ppm && params.write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    validate_packet_header_marker_payloads(
        params.write_ppm,
        params.write_ppt,
        &[&packetized_tile.packet_headers],
    )?;
    Ok(())
}

pub(super) fn validate_packet_header_marker_payloads(
    write_ppm: bool,
    write_ppt: bool,
    tile_packet_headers: &[&[Vec<u8>]],
) -> Result<(), &'static str> {
    const PACKET_HEADER_MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
    const PPM_PACKET_HEADER_LIMIT: usize = PACKET_HEADER_MARKER_PAYLOAD_LIMIT - 2;
    const MAX_PACKET_HEADER_MARKERS: usize = u8::MAX as usize + 1;

    if !write_ppm && !write_ppt {
        return Ok(());
    }
    if write_ppm && write_ppt {
        return Err("PPM and PPT packet header markers are mutually exclusive");
    }
    if tile_packet_headers.iter().any(|headers| headers.is_empty()) {
        return Err("PPM/PPT encode requires separated packet headers");
    }
    if write_ppm {
        let mut marker_count = 0usize;
        let mut payload_len = 0usize;
        for header in tile_packet_headers
            .iter()
            .flat_map(|headers| headers.iter())
        {
            if header.len() > PPM_PACKET_HEADER_LIMIT {
                return Err("PPM packet header exceeds marker payload limit");
            }
            let entry_len = 2usize
                .checked_add(header.len())
                .ok_or("PPM marker payload length overflow")?;
            if payload_len == 0 {
                marker_count = marker_count
                    .checked_add(1)
                    .ok_or("PPM marker count overflow")?;
            } else if payload_len
                .checked_add(entry_len)
                .is_none_or(|len| len > PACKET_HEADER_MARKER_PAYLOAD_LIMIT)
            {
                marker_count = marker_count
                    .checked_add(1)
                    .ok_or("PPM marker count overflow")?;
                payload_len = 0;
            }
            payload_len = payload_len
                .checked_add(entry_len)
                .ok_or("PPM marker payload length overflow")?;
            if marker_count > MAX_PACKET_HEADER_MARKERS {
                return Err("PPM packet headers require more than 256 marker segments");
            }
        }
    }
    if write_ppt {
        for headers in tile_packet_headers {
            let payload_len = headers.iter().try_fold(0usize, |acc, header| {
                acc.checked_add(header.len())
                    .ok_or("PPT marker payload length overflow")
            })?;
            let marker_count = payload_len.div_ceil(PACKET_HEADER_MARKER_PAYLOAD_LIMIT);
            if marker_count > MAX_PACKET_HEADER_MARKERS {
                return Err("PPT packet headers require more than 256 marker segments");
            }
        }
    }
    Ok(())
}
