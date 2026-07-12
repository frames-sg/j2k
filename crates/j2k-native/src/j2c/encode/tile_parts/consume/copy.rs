// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible packet-range copies for split tile-parts.

use alloc::vec::Vec;

use super::super::{encoded_tile_parts_retained_bytes, EncodedTilePart};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
};
use crate::j2c::packet_encode::PacketizedTileData;

#[expect(
    clippy::too_many_arguments,
    reason = "the split transition keeps source ranges and every simultaneously live owner explicit"
)]
pub(super) fn try_copy_part(
    tile_index: u16,
    tile_part_index: u8,
    num_tile_parts: u8,
    data_offset: usize,
    packet_lengths: &[u32],
    packet_offset: usize,
    packetized: &PacketizedTileData,
    retained_base_bytes: usize,
    source_bytes: usize,
    prior_parts: &Vec<EncodedTilePart>,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<EncodedTilePart> {
    let data_len = packet_lengths.iter().try_fold(0usize, |acc, &len| {
        let len = usize::try_from(len).map_err(|_| {
            NativeEncodePipelineError::arithmetic_overflow("packet length exceeds usize")
        })?;
        acc.checked_add(len)
            .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("packet length sum"))
    })?;
    let data_end = data_offset
        .checked_add(data_len)
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("packet data end"))?;
    let packet_end = packet_offset
        .checked_add(packet_lengths.len())
        .ok_or_else(|| NativeEncodePipelineError::arithmetic_overflow("tile-part packet range"))?;
    let source_headers = if packetized.packet_headers.is_empty() {
        &[]
    } else {
        packetized
            .packet_headers
            .get(packet_offset..packet_end)
            .ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant(
                    "tile-part packet header range is out of bounds",
                )
            })?
    };
    let prior_bytes = encoded_tile_parts_retained_bytes(prior_parts, prior_parts.capacity())?;
    let live_before_part = checked_add_bytes(
        retained_base_bytes,
        checked_add_bytes(source_bytes, prior_bytes, "multi-tile split peak")?,
        "multi-tile split peak",
    )?;
    let mut tracker = PartCopyTracker::new(session, live_before_part);
    tracker.before(
        requested_part_bytes(data_len, packet_lengths.len(), source_headers)?,
        "multi-tile split part preflight",
    )?;

    let data = tracker.try_copy_slice(
        packetized.data.get(data_offset..data_end).ok_or_else(|| {
            NativeEncodePipelineError::internal_invariant(
                "tile-part packet data range is out of bounds",
            )
        })?,
        "multi-tile split payload",
    )?;
    let packet_lengths =
        tracker.try_copy_slice(packet_lengths, "multi-tile split packet lengths")?;
    let packet_headers = try_copy_headers(source_headers, &mut tracker)?;
    let part = EncodedTilePart {
        tile_index,
        tile_part_index,
        num_tile_parts,
        data,
        packet_lengths,
        packet_headers,
    };
    let part_bytes = encoded_tile_parts_retained_bytes(core::slice::from_ref(&part), 0)?;
    tracker.check_actual(part_bytes, "multi-tile split part")?;
    Ok(part)
}

fn requested_part_bytes(
    data_len: usize,
    packet_count: usize,
    headers: &[Vec<u8>],
) -> NativeEncodePipelineResult<usize> {
    let mut bytes = data_len;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<u32>(packet_count, "multi-tile split packet lengths")?,
        "multi-tile split part",
    )?;
    bytes = checked_add_bytes(
        bytes,
        checked_element_bytes::<Vec<u8>>(headers.len(), "multi-tile split header owners")?,
        "multi-tile split part",
    )?;
    for header in headers {
        bytes = checked_add_bytes(bytes, header.len(), "multi-tile split headers")?;
    }
    Ok(bytes)
}

fn try_copy_headers(
    headers: &[Vec<u8>],
    tracker: &mut PartCopyTracker<'_, '_>,
) -> NativeEncodePipelineResult<Vec<Vec<u8>>> {
    let mut copies = tracker.try_vec::<Vec<u8>>(headers.len(), "multi-tile split header owners")?;
    for header in headers {
        copies.push(tracker.try_copy_slice(header, "multi-tile split packet header")?);
    }
    Ok(copies)
}

struct PartCopyTracker<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    retained_base_bytes: usize,
    live_part_bytes: usize,
}

impl<'session, 'input> PartCopyTracker<'session, 'input> {
    const fn new(
        session: &'session NativeEncodeSession<'input>,
        retained_base_bytes: usize,
    ) -> Self {
        Self {
            session,
            retained_base_bytes,
            live_part_bytes: 0,
        }
    }

    fn before(&self, requested_bytes: usize, what: &'static str) -> NativeEncodePipelineResult<()> {
        self.session.checked_phase(
            checked_add_bytes(
                self.retained_base_bytes,
                checked_add_bytes(self.live_part_bytes, requested_bytes, what)?,
                what,
            )?,
            what,
        )?;
        Ok(())
    }

    fn retain(
        &mut self,
        actual_bytes: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<()> {
        self.live_part_bytes = checked_add_bytes(self.live_part_bytes, actual_bytes, what)?;
        self.check_actual(self.live_part_bytes, what)
    }

    fn check_actual(
        &self,
        actual_bytes: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<()> {
        self.session.checked_phase(
            checked_add_bytes(self.retained_base_bytes, actual_bytes, what)?,
            what,
        )?;
        Ok(())
    }

    fn try_vec<T>(
        &mut self,
        count: usize,
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let requested = checked_element_bytes::<T>(count, what)?;
        self.before(requested, what)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, requested))?;
        self.retain(checked_element_bytes::<T>(values.capacity(), what)?, what)?;
        Ok(values)
    }

    fn try_copy_slice<T: Copy>(
        &mut self,
        values: &[T],
        what: &'static str,
    ) -> NativeEncodePipelineResult<Vec<T>> {
        let mut copy = self.try_vec(values.len(), what)?;
        copy.extend_from_slice(values);
        Ok(copy)
    }
}
