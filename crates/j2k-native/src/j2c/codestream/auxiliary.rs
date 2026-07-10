// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec;
use alloc::vec::Vec;

use super::progression::read_component_index;
use super::{PacketLengthMarker, PpmMarkerData, PpmPacket, RgnMarkerData};
use crate::reader::BitReader;

/// COM Marker (A.9.2).
pub(super) fn com_marker(reader: &mut BitReader<'_>) -> Option<()> {
    skip_marker_segment(reader)
}

/// TLM marker (A.7.1).
pub(super) fn tlm_marker(reader: &mut BitReader<'_>) -> Option<()> {
    skip_marker_segment(reader)
}

/// PLM marker (A.7.2).
pub(super) fn plm_marker(reader: &mut BitReader<'_>) -> Option<PacketLengthMarker> {
    let segment_len = reader.read_u16()?.checked_sub(2)? as usize;
    let segment = reader.read_bytes(segment_len)?;
    let mut reader = BitReader::new(segment);

    let sequence_idx = reader.read_byte()?;
    let mut packet_lengths = vec![];

    while !reader.at_end() {
        let length_data_len = reader.read_u32()? as usize;
        let length_data = reader.read_bytes(length_data_len)?;
        packet_lengths.extend(decode_packet_lengths(length_data)?);
    }

    Some(PacketLengthMarker {
        sequence_idx,
        packet_lengths,
    })
}

/// PLT marker (A.7.3).
pub(crate) fn plt_marker(reader: &mut BitReader<'_>) -> Option<PacketLengthMarker> {
    let segment_len = reader.read_u16()?.checked_sub(2)? as usize;
    let segment = reader.read_bytes(segment_len)?;
    let mut reader = BitReader::new(segment);

    let sequence_idx = reader.read_byte()?;
    let packet_lengths = decode_packet_lengths(reader.tail()?)?;

    Some(PacketLengthMarker {
        sequence_idx,
        packet_lengths,
    })
}

pub(crate) fn decode_packet_lengths(data: &[u8]) -> Option<Vec<u32>> {
    let mut packet_lengths = vec![];
    let mut value = 0_u32;
    let mut in_progress = false;

    for byte in data {
        value = value.checked_shl(7)?.checked_add(u32::from(byte & 0x7F))?;
        in_progress = true;

        if byte & 0x80 == 0 {
            packet_lengths.push(value);
            value = 0;
            in_progress = false;
        }
    }

    if in_progress {
        return None;
    }

    Some(packet_lengths)
}

/// PPM marker (A.7.4).
pub(super) fn ppm_marker<'a>(reader: &mut BitReader<'a>) -> Option<PpmMarkerData<'a>> {
    let segment_len = reader.read_u16()?.checked_sub(2)? as usize;
    let ppm_data = reader.read_bytes(segment_len)?;
    let mut packets = vec![];

    let mut reader = BitReader::new(ppm_data);
    let sequence_idx = reader.read_byte()?;

    // This parser handles complete packet payloads carried by the current PPM
    // marker. Continuations across multiple PPM markers are rejected by normal
    // length parsing until a multi-marker accumulator is added.
    while !reader.at_end() {
        let packet_len = reader.read_u16()? as usize;
        let data = reader.read_bytes(packet_len)?;

        packets.push(PpmPacket { data });
    }

    Some(PpmMarkerData {
        sequence_idx,
        packets,
    })
}

/// RGN marker (A.6.3).
pub(crate) fn rgn_marker(reader: &mut BitReader<'_>, csiz: u16) -> Option<RgnMarkerData> {
    let length = reader.read_u16()?;
    let component_index_bytes = if csiz < 257 { 1 } else { 2 };
    if length != 4 + component_index_bytes {
        return None;
    }

    let component_index = read_component_index(reader, csiz)?;
    if component_index >= csiz {
        return None;
    }

    let style = reader.read_byte()?;
    let shift = reader.read_byte()?;

    Some(RgnMarkerData {
        component_index,
        style,
        shift,
    })
}

pub(crate) fn skip_marker_segment(reader: &mut BitReader<'_>) -> Option<()> {
    let length = reader.read_u16()?.checked_sub(2)?;
    reader.skip_bytes(length as usize)?;

    Some(())
}
