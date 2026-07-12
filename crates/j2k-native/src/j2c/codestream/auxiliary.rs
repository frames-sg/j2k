// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;
use core::mem::size_of;

use super::progression::read_component_index;
use super::{PpmMarkerData, PpmPacket, RgnMarkerData};
use crate::error::{MarkerError, Result, ValidationError};
use crate::reader::BitReader;
use crate::try_reserve_decode_elements;

mod packet_lengths;

#[cfg(test)]
pub(crate) use packet_lengths::decode_packet_lengths;
pub(super) use packet_lengths::plm_marker;
pub(crate) use packet_lengths::plt_marker;

/// COM Marker (A.9.2).
pub(super) fn com_marker(reader: &mut BitReader<'_>) -> Option<()> {
    skip_marker_segment(reader)
}

/// TLM marker (A.7.1).
pub(super) fn tlm_marker(reader: &mut BitReader<'_>) -> Option<()> {
    skip_marker_segment(reader)
}

/// PPM marker (A.7.4).
pub(super) fn ppm_marker<'a>(
    reader: &mut BitReader<'a>,
    max_owned_bytes: usize,
) -> Result<PpmMarkerData<'a>> {
    let segment_len = reader
        .read_u16()
        .and_then(|length| length.checked_sub(2))
        .ok_or(MarkerError::ParseFailure("PPM"))? as usize;
    let ppm_data = reader
        .read_bytes(segment_len)
        .ok_or(MarkerError::ParseFailure("PPM"))?;
    let sequence_idx = ppm_data
        .first()
        .copied()
        .ok_or(MarkerError::ParseFailure("PPM"))?;
    let payload = &ppm_data[1..];

    let packet_count = visit_ppm_packets(payload, |_| {})?;
    let packet_bytes = packet_count
        .checked_mul(size_of::<PpmPacket<'_>>())
        .ok_or(ValidationError::ImageTooLarge)?;
    if packet_bytes > max_owned_bytes {
        return Err(ValidationError::ImageTooLarge.into());
    }

    let mut packets = Vec::new();
    try_reserve_decode_elements(&mut packets, packet_count)?;
    visit_ppm_packets(payload, |data| packets.push(PpmPacket { data }))?;

    Ok(PpmMarkerData {
        sequence_idx,
        packets,
    })
}

fn visit_ppm_packets<'a>(payload: &'a [u8], mut visit: impl FnMut(&'a [u8])) -> Result<usize> {
    let mut packet_count = 0_usize;
    let mut reader = BitReader::new(payload);
    // This parser handles complete packet payloads carried by the current PPM
    // marker. Continuations across multiple PPM markers are rejected by normal
    // length parsing until a multi-marker accumulator is added.
    while !reader.at_end() {
        let packet_len = reader.read_u16().ok_or(MarkerError::ParseFailure("PPM"))? as usize;
        let data = reader
            .read_bytes(packet_len)
            .ok_or(MarkerError::ParseFailure("PPM"))?;
        visit(data);
        packet_count = packet_count
            .checked_add(1)
            .ok_or(ValidationError::ImageTooLarge)?;
    }
    Ok(packet_count)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DecodeError;

    #[test]
    fn ppm_output_limit_is_checked_before_reservation() {
        // Lppm=9: one sequence byte followed by three empty packet records.
        let data = [0, 9, 0, 0, 0, 0, 0, 0, 0];
        let mut reader = BitReader::new(&data);

        assert_eq!(
            ppm_marker(&mut reader, 2 * size_of::<PpmPacket<'_>>()).unwrap_err(),
            DecodeError::Validation(ValidationError::ImageTooLarge)
        );
        assert_eq!(reader.offset(), data.len());
    }
}
