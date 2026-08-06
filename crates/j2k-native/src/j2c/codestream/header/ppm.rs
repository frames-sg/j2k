// SPDX-License-Identifier: MIT OR Apache-2.0

//! PPM tile-part packet-header stream assembly.

use alloc::vec::Vec;

use super::super::{PpmMarkerData, PpmPacket};
use super::allocation::HeaderMarkerBudget;
use crate::error::{MarkerError, Result, ValidationError};

pub(super) fn try_flatten_ppm_packets<'a>(
    markers: Vec<PpmMarkerData<'a>>,
    budget: &mut HeaderMarkerBudget,
) -> Result<Vec<PpmPacket<'a>>> {
    let marker_capacity = markers.capacity();
    for (expected, marker) in markers.iter().enumerate() {
        if u8::try_from(expected).ok() != Some(marker.sequence_idx) {
            return Err(MarkerError::ParseFailure("PPM").into());
        }
    }

    let mut packets = Vec::new();
    {
        let mut cursor = PpmPayloadCursor::new(&markers);
        while let Some(packet_len) = cursor.read_u32()? {
            let mut remaining =
                usize::try_from(packet_len).map_err(|_| ValidationError::ImageTooLarge)?;
            if remaining == 0 {
                budget.try_reserve_next(&mut packets)?;
                packets.push(PpmPacket {
                    data: &[],
                    ends_tile_part: true,
                });
                continue;
            }
            while remaining != 0 {
                let data = cursor
                    .take_fragment(remaining)
                    .ok_or(MarkerError::ParseFailure("PPM"))?;
                remaining -= data.len();
                budget.try_reserve_next(&mut packets)?;
                packets.push(PpmPacket {
                    data,
                    ends_tile_part: remaining == 0,
                });
            }
        }
    }
    drop(markers);
    budget.release_capacity::<PpmMarkerData<'_>>(marker_capacity)?;
    Ok(packets)
}

struct PpmPayloadCursor<'markers, 'data> {
    markers: &'markers [PpmMarkerData<'data>],
    marker: usize,
    offset: usize,
}

impl<'markers, 'data> PpmPayloadCursor<'markers, 'data> {
    const fn new(markers: &'markers [PpmMarkerData<'data>]) -> Self {
        Self {
            markers,
            marker: 0,
            offset: 0,
        }
    }

    fn read_u32(&mut self) -> Result<Option<u32>> {
        let Some(first) = self.next_byte() else {
            return Ok(None);
        };
        let mut bytes = [first, 0, 0, 0];
        for byte in &mut bytes[1..] {
            *byte = self.next_byte().ok_or(MarkerError::ParseFailure("PPM"))?;
        }
        Ok(Some(u32::from_be_bytes(bytes)))
    }

    fn next_byte(&mut self) -> Option<u8> {
        loop {
            let data = self.markers.get(self.marker)?.data;
            if let Some(byte) = data.get(self.offset).copied() {
                self.offset += 1;
                return Some(byte);
            }
            self.marker += 1;
            self.offset = 0;
        }
    }

    fn take_fragment(&mut self, max_len: usize) -> Option<&'data [u8]> {
        loop {
            let data = self.markers.get(self.marker)?.data;
            if self.offset == data.len() {
                self.marker += 1;
                self.offset = 0;
                continue;
            }
            let end = self.offset.saturating_add(max_len).min(data.len());
            let fragment = data.get(self.offset..end)?;
            self.offset = end;
            return Some(fragment);
        }
    }
}

#[cfg(test)]
mod tests;
