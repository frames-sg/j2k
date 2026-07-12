// SPDX-License-Identifier: MIT OR Apache-2.0

//! Allocation-free packet-length and separated-header marker serialization.

use alloc::vec::Vec;

use super::{markers, write_marker, TilePartData};
use crate::j2c::encode::allocation::checked_add_bytes;
use crate::{EncodeError, EncodeResult};

pub(super) const PACKET_HEADER_MARKER_PAYLOAD_LIMIT: usize = u16::MAX as usize - 3;
pub(super) const PPM_PACKET_HEADER_LIMIT: usize = PACKET_HEADER_MARKER_PAYLOAD_LIMIT - 2;
const MAX_PACKET_MARKERS: usize = u8::MAX as usize + 1;
const PLT_CHUNK_SIZE: usize = u16::MAX as usize - 3;
const PLM_CHUNK_SIZE: usize = u16::MAX as usize - 7;

pub(super) fn plt_marker_bytes(packet_lengths: &[u32]) -> EncodeResult<usize> {
    let payload = packet_length_payload_len(packet_lengths)?;
    chunked_marker_bytes(
        payload,
        PLT_CHUNK_SIZE,
        5,
        "PLT packet lengths require more than 256 marker segments",
        "PLT marker byte length overflow",
    )
}

pub(super) fn plm_marker_bytes(tiles: &[TilePartData<'_>]) -> EncodeResult<usize> {
    let payload = tiles.iter().try_fold(0usize, |bytes, tile| {
        checked_add_bytes(
            bytes,
            packet_length_payload_len(tile.packet_lengths)?,
            "PLM packet length payload",
        )
    })?;
    chunked_marker_bytes(
        payload,
        PLM_CHUNK_SIZE,
        9,
        "PLM packet lengths require more than 256 marker segments",
        "PLM marker byte length overflow",
    )
}

pub(super) fn ppm_marker_bytes(tiles: &[TilePartData<'_>]) -> EncodeResult<usize> {
    let mut total = 0usize;
    let mut payload = 0usize;
    let mut marker_count = 0usize;
    for header in tiles.iter().flat_map(|tile| tile.packet_headers.iter()) {
        if header.len() > PPM_PACKET_HEADER_LIMIT {
            return invalid("PPM packet header exceeds marker payload limit");
        }
        let entry = header
            .len()
            .checked_add(2)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "PPM marker payload length",
            })?;
        if payload != 0
            && payload
                .checked_add(entry)
                .is_none_or(|bytes| bytes > PACKET_HEADER_MARKER_PAYLOAD_LIMIT)
        {
            total = add_ppm_marker(total, payload, &mut marker_count)?;
            payload = 0;
        }
        payload = checked_add_bytes(payload, entry, "PPM marker payload length")?;
    }
    if payload != 0 {
        total = add_ppm_marker(total, payload, &mut marker_count)?;
    }
    Ok(total)
}

pub(super) fn ppt_marker_bytes(packet_headers: &[Vec<u8>]) -> EncodeResult<usize> {
    let payload = packet_headers.iter().try_fold(0usize, |bytes, header| {
        checked_add_bytes(bytes, header.len(), "PPT marker payload length")
    })?;
    chunked_marker_bytes(
        payload,
        PACKET_HEADER_MARKER_PAYLOAD_LIMIT,
        5,
        "PPT packet headers require more than 256 marker segments",
        "PPT marker byte length overflow",
    )
}

pub(super) fn write_plt_markers(out: &mut Vec<u8>, packet_lengths: &[u32]) -> EncodeResult<()> {
    let payload = packet_length_payload_len(packet_lengths)?;
    plt_marker_bytes(packet_lengths)?;
    write_packet_length_chunks(out, [packet_lengths], payload, PacketLengthMarker::Plt)
}

pub(super) fn write_plm_markers(out: &mut Vec<u8>, tiles: &[TilePartData<'_>]) -> EncodeResult<()> {
    let payload = tiles.iter().try_fold(0usize, |bytes, tile| {
        checked_add_bytes(
            bytes,
            packet_length_payload_len(tile.packet_lengths)?,
            "PLM packet length payload",
        )
    })?;
    plm_marker_bytes(tiles)?;
    write_packet_length_chunks(
        out,
        tiles.iter().map(|tile| tile.packet_lengths),
        payload,
        PacketLengthMarker::Plm,
    )
}

pub(super) fn write_ppm_markers(out: &mut Vec<u8>, tiles: &[TilePartData<'_>]) -> EncodeResult<()> {
    ppm_marker_bytes(tiles)?;
    let mut cursor = HeaderCursor::default();
    let mut sequence = 0usize;
    loop {
        let mut end = cursor;
        let mut payload = 0usize;
        loop {
            let before = end;
            let Some(header) = next_header(tiles, &mut end) else {
                break;
            };
            let entry = header
                .len()
                .checked_add(2)
                .ok_or(EncodeError::ArithmeticOverflow {
                    what: "PPM marker payload length",
                })?;
            if payload != 0
                && payload
                    .checked_add(entry)
                    .is_none_or(|bytes| bytes > PACKET_HEADER_MARKER_PAYLOAD_LIMIT)
            {
                end = before;
                break;
            }
            payload = checked_add_bytes(payload, entry, "PPM marker payload length")?;
        }
        if payload == 0 {
            break;
        }
        write_marker(out, markers::PPM);
        let marker_len =
            u16::try_from(payload + 3).map_err(|_| EncodeError::InternalInvariant {
                what: "validated PPM marker length exceeds u16",
            })?;
        out.extend_from_slice(&marker_len.to_be_bytes());
        out.push(
            u8::try_from(sequence).map_err(|_| EncodeError::InternalInvariant {
                what: "validated PPM marker sequence exceeds u8",
            })?,
        );
        while cursor != end {
            let header = next_header(tiles, &mut cursor).ok_or(EncodeError::InternalInvariant {
                what: "validated PPM header cursor ended early",
            })?;
            let header_len =
                u16::try_from(header.len()).map_err(|_| EncodeError::InternalInvariant {
                    what: "validated PPM packet header length exceeds u16",
                })?;
            out.extend_from_slice(&header_len.to_be_bytes());
            out.extend_from_slice(header);
        }
        sequence += 1;
    }
    Ok(())
}

pub(super) fn write_ppt_markers(out: &mut Vec<u8>, packet_headers: &[Vec<u8>]) -> EncodeResult<()> {
    let total_payload = packet_headers.iter().try_fold(0usize, |bytes, header| {
        checked_add_bytes(bytes, header.len(), "PPT marker payload length")
    })?;
    ppt_marker_bytes(packet_headers)?;
    let mut payload_remaining = total_payload;
    let mut chunk_remaining = 0usize;
    let mut sequence = 0usize;
    for header in packet_headers {
        let mut remaining = header.as_slice();
        while !remaining.is_empty() {
            if chunk_remaining == 0 {
                let chunk_len = payload_remaining.min(PACKET_HEADER_MARKER_PAYLOAD_LIMIT);
                write_marker(out, markers::PPT);
                let marker_len =
                    u16::try_from(chunk_len + 3).map_err(|_| EncodeError::InternalInvariant {
                        what: "validated PPT marker length exceeds u16",
                    })?;
                out.extend_from_slice(&marker_len.to_be_bytes());
                out.push(
                    u8::try_from(sequence).map_err(|_| EncodeError::InternalInvariant {
                        what: "validated PPT marker sequence exceeds u8",
                    })?,
                );
                sequence += 1;
                chunk_remaining = chunk_len;
                payload_remaining -= chunk_len;
            }
            let take = chunk_remaining.min(remaining.len());
            out.extend_from_slice(&remaining[..take]);
            remaining = &remaining[take..];
            chunk_remaining -= take;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum PacketLengthMarker {
    Plt,
    Plm,
}

fn write_packet_length_chunks<'a>(
    out: &mut Vec<u8>,
    packet_length_sets: impl IntoIterator<Item = &'a [u32]>,
    total_payload: usize,
    marker: PacketLengthMarker,
) -> EncodeResult<()> {
    let chunk_size = match marker {
        PacketLengthMarker::Plt => PLT_CHUNK_SIZE,
        PacketLengthMarker::Plm => PLM_CHUNK_SIZE,
    };
    let mut payload_remaining = total_payload;
    let mut chunk_remaining = 0usize;
    let mut sequence = 0usize;
    for packet_lengths in packet_length_sets {
        for &packet_length in packet_lengths {
            let (encoded, start) = encode_packet_length(packet_length);
            for &byte in &encoded[start..] {
                if chunk_remaining == 0 {
                    let chunk_len = payload_remaining.min(chunk_size);
                    begin_packet_length_marker(out, marker, sequence, chunk_len)?;
                    sequence += 1;
                    chunk_remaining = chunk_len;
                    payload_remaining -= chunk_len;
                }
                out.push(byte);
                chunk_remaining -= 1;
            }
        }
    }
    Ok(())
}

fn begin_packet_length_marker(
    out: &mut Vec<u8>,
    marker: PacketLengthMarker,
    sequence: usize,
    chunk_len: usize,
) -> EncodeResult<()> {
    let (marker_code, length_overhead) = match marker {
        PacketLengthMarker::Plt => (markers::PLT, 3usize),
        PacketLengthMarker::Plm => (markers::PLM, 7usize),
    };
    write_marker(out, marker_code);
    let marker_len =
        u16::try_from(chunk_len + length_overhead).map_err(|_| EncodeError::InternalInvariant {
            what: "validated packet-length marker length exceeds u16",
        })?;
    out.extend_from_slice(&marker_len.to_be_bytes());
    out.push(
        u8::try_from(sequence).map_err(|_| EncodeError::InternalInvariant {
            what: "validated packet-length marker sequence exceeds u8",
        })?,
    );
    if matches!(marker, PacketLengthMarker::Plm) {
        out.extend_from_slice(
            &u32::try_from(chunk_len)
                .map_err(|_| EncodeError::InternalInvariant {
                    what: "validated PLM chunk length exceeds u32",
                })?
                .to_be_bytes(),
        );
    }
    Ok(())
}

fn packet_length_payload_len(packet_lengths: &[u32]) -> EncodeResult<usize> {
    packet_lengths.iter().try_fold(0usize, |bytes, &length| {
        checked_add_bytes(
            bytes,
            encoded_packet_length_len(length),
            "packet length marker payload",
        )
    })
}

fn encoded_packet_length_len(value: u32) -> usize {
    let bits = u32::BITS - value.leading_zeros();
    if bits == 0 {
        1
    } else {
        bits.div_ceil(7) as usize
    }
}

fn encode_packet_length(mut value: u32) -> ([u8; 5], usize) {
    let mut encoded = [0_u8; 5];
    let mut start = encoded.len();
    loop {
        start -= 1;
        encoded[start] = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    let final_index = encoded.len() - 1;
    for byte in &mut encoded[start..final_index] {
        *byte |= 0x80;
    }
    (encoded, start)
}

fn chunked_marker_bytes(
    payload: usize,
    chunk_size: usize,
    marker_overhead: usize,
    count_error: &'static str,
    overflow_what: &'static str,
) -> EncodeResult<usize> {
    if payload == 0 {
        return Ok(0);
    }
    let marker_count = payload.div_ceil(chunk_size);
    if marker_count > MAX_PACKET_MARKERS {
        return invalid(count_error);
    }
    let overhead =
        marker_count
            .checked_mul(marker_overhead)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: overflow_what,
            })?;
    checked_add_bytes(payload, overhead, overflow_what)
}

fn add_ppm_marker(total: usize, payload: usize, marker_count: &mut usize) -> EncodeResult<usize> {
    *marker_count = marker_count
        .checked_add(1)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "packet-header marker count",
        })?;
    if *marker_count > MAX_PACKET_MARKERS {
        return invalid("PPM packet headers require more than 256 marker segments");
    }
    checked_add_bytes(
        total,
        payload
            .checked_add(5)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "packet-header marker byte length",
            })?,
        "packet-header marker byte length",
    )
}

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct HeaderCursor {
    tile: usize,
    header: usize,
}

fn next_header<'a>(tiles: &[TilePartData<'a>], cursor: &mut HeaderCursor) -> Option<&'a [u8]> {
    loop {
        let tile = tiles.get(cursor.tile)?;
        if let Some(header) = tile.packet_headers.get(cursor.header) {
            cursor.header += 1;
            if cursor.header == tile.packet_headers.len() {
                cursor.tile += 1;
                cursor.header = 0;
            }
            return Some(header.as_slice());
        }
        cursor.tile += 1;
        cursor.header = 0;
    }
}

fn invalid<T>(what: &'static str) -> EncodeResult<T> {
    Err(EncodeError::InvalidInput { what })
}
