// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded parsing of PLM/PLT packet-length metadata.

use alloc::vec::Vec;
use core::mem::size_of;

use super::super::PacketLengthMarker;
use crate::error::{MarkerError, Result, ValidationError};
use crate::reader::BitReader;
use crate::try_reserve_decode_elements;
#[cfg(test)]
use crate::DEFAULT_MAX_DECODE_BYTES;

pub(crate) fn plm_marker(
    reader: &mut BitReader<'_>,
    max_owned_bytes: usize,
) -> Result<PacketLengthMarker> {
    let segment_len = reader
        .read_u16()
        .and_then(|length| length.checked_sub(2))
        .ok_or(MarkerError::ParseFailure("PLM"))? as usize;
    let segment = reader
        .read_bytes(segment_len)
        .ok_or(MarkerError::ParseFailure("PLM"))?;
    let sequence_idx = segment
        .first()
        .copied()
        .ok_or(MarkerError::ParseFailure("PLM"))?;
    let payload = &segment[1..];

    let mut packet_count = 0_usize;
    visit_plm_chunks(payload, |chunk| {
        packet_count = packet_count
            .checked_add(visit_packet_lengths(chunk, "PLM", |_| {})?)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    })?;
    validate_output_bytes(packet_count, max_owned_bytes)?;

    let mut packet_lengths = Vec::new();
    try_reserve_decode_elements(&mut packet_lengths, packet_count)?;
    visit_plm_chunks(payload, |chunk| {
        visit_packet_lengths(chunk, "PLM", |length| packet_lengths.push(length))?;
        Ok(())
    })?;

    Ok(PacketLengthMarker {
        sequence_idx,
        packet_lengths,
    })
}

pub(crate) fn plt_marker(
    reader: &mut BitReader<'_>,
    max_owned_bytes: usize,
) -> Result<PacketLengthMarker> {
    let segment_len = reader
        .read_u16()
        .and_then(|length| length.checked_sub(2))
        .ok_or(MarkerError::ParseFailure("PLT"))? as usize;
    let segment = reader
        .read_bytes(segment_len)
        .ok_or(MarkerError::ParseFailure("PLT"))?;
    let (&sequence_idx, payload) = segment
        .split_first()
        .ok_or(MarkerError::ParseFailure("PLT"))?;
    let packet_lengths = decode_packet_lengths_with_limit(payload, max_owned_bytes, "PLT")?;

    Ok(PacketLengthMarker {
        sequence_idx,
        packet_lengths,
    })
}

#[cfg(test)]
pub(crate) fn decode_packet_lengths(data: &[u8]) -> Result<Vec<u32>> {
    decode_packet_lengths_with_limit(data, DEFAULT_MAX_DECODE_BYTES, "packet lengths")
}

fn decode_packet_lengths_with_limit(
    data: &[u8],
    max_owned_bytes: usize,
    marker: &'static str,
) -> Result<Vec<u32>> {
    let packet_count = visit_packet_lengths(data, marker, |_| {})?;
    validate_output_bytes(packet_count, max_owned_bytes)?;

    let mut packet_lengths = Vec::new();
    try_reserve_decode_elements(&mut packet_lengths, packet_count)?;
    visit_packet_lengths(data, marker, |length| packet_lengths.push(length))?;
    Ok(packet_lengths)
}

fn validate_output_bytes(packet_count: usize, max_owned_bytes: usize) -> Result<()> {
    let output_bytes = packet_count
        .checked_mul(size_of::<u32>())
        .ok_or(ValidationError::ImageTooLarge)?;
    if output_bytes > max_owned_bytes {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

fn visit_plm_chunks(payload: &[u8], mut visit: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let mut reader = BitReader::new(payload);
    while !reader.at_end() {
        let length_data_len = reader.read_u32().ok_or(MarkerError::ParseFailure("PLM"))? as usize;
        let length_data = reader
            .read_bytes(length_data_len)
            .ok_or(MarkerError::ParseFailure("PLM"))?;
        visit(length_data)?;
    }
    Ok(())
}

fn visit_packet_lengths(
    data: &[u8],
    marker: &'static str,
    mut visit: impl FnMut(u32),
) -> Result<usize> {
    let mut packet_count = 0_usize;
    let mut value = 0_u32;
    let mut in_progress = false;

    for byte in data {
        value = value
            .checked_shl(7)
            .and_then(|value| value.checked_add(u32::from(byte & 0x7F)))
            .ok_or(MarkerError::ParseFailure(marker))?;
        in_progress = true;

        if byte & 0x80 == 0 {
            visit(value);
            packet_count = packet_count
                .checked_add(1)
                .ok_or(ValidationError::ImageTooLarge)?;
            value = 0;
            in_progress = false;
        }
    }

    if in_progress {
        return Err(MarkerError::ParseFailure(marker).into());
    }
    Ok(packet_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::DecodeError;

    #[test]
    fn plt_output_limit_is_checked_before_reservation() {
        // Lplt=6: one sequence byte followed by three one-byte packet lengths.
        let data = [0, 6, 0, 1, 2, 3];
        let mut reader = BitReader::new(&data);

        assert_eq!(
            plt_marker(&mut reader, 2 * size_of::<u32>()).unwrap_err(),
            DecodeError::Validation(ValidationError::ImageTooLarge)
        );
        assert_eq!(reader.offset(), data.len());
    }
}
