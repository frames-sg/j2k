// SPDX-License-Identifier: MIT OR Apache-2.0

//! Consuming, fallible packetized-tile to tile-part ownership transition.

use alloc::vec::Vec;

use super::{encoded_tile_parts_retained_bytes, EncodedTilePart};
use crate::j2c::encode::allocation::{
    checked_add_bytes, checked_element_bytes, host_allocation_failed,
};
use crate::j2c::encode::{
    NativeEncodePipelineError, NativeEncodePipelineResult, NativeEncodeSession,
};
use crate::j2c::packet_encode::{self, PacketizedTileData};

mod copy;
use copy::try_copy_part;

pub(in crate::j2c::encode) fn consume_packetized_tile_into_tile_parts(
    tile_index: u16,
    packetized: PacketizedTileData,
    packet_limit: Option<u16>,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<EncodedTilePart>> {
    validate_packetized_tile(&packetized)?;
    if matches!(packet_limit, Some(0)) {
        return Err(NativeEncodePipelineError::invalid_input(
            "tile-part packet limit must be non-zero",
        ));
    }
    if packet_limit.is_some() && packetized.packet_lengths.is_empty() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "tile-part splitting requires packet-length metadata",
        }
        .into());
    }
    if packet_limit.is_none() {
        return consume_single_part(tile_index, packetized, retained_base_bytes, session);
    }
    let packet_limit = usize::from(packet_limit.ok_or(
        NativeEncodePipelineError::internal_invariant("missing tile-part packet limit"),
    )?);
    if packet_limit == 0 {
        return Err(NativeEncodePipelineError::invalid_input(
            "tile-part packet limit must be non-zero",
        ));
    }
    let part_count = packetized.packet_lengths.len().div_ceil(packet_limit);
    if part_count > usize::from(u8::MAX) {
        return Err(NativeEncodePipelineError::unsupported(
            "tile-part packet limit would emit more than 255 tile-parts",
        ));
    }
    let num_tile_parts = u8::try_from(part_count)
        .map_err(|_| NativeEncodePipelineError::internal_invariant("tile-part count exceeds u8"))?;
    let source_bytes = packet_encode::packetized_tile_retained_bytes(&packetized)?;
    let requested_outer =
        checked_element_bytes::<EncodedTilePart>(part_count, "multi-tile split part owners")?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(source_bytes, requested_outer, "multi-tile split owners")?,
            "multi-tile split owners",
        )?,
        "multi-tile split owners",
    )?;
    let mut parts = Vec::new();
    parts
        .try_reserve_exact(part_count)
        .map_err(|_| host_allocation_failed("multi-tile split part owners", requested_outer))?;
    reconcile_source_and_outer(
        session,
        retained_base_bytes,
        source_bytes,
        parts.capacity(),
        "multi-tile split part owners",
    )?;
    let mut data_offset = 0usize;
    for (part_index, packet_range) in packetized.packet_lengths.chunks(packet_limit).enumerate() {
        let part = try_copy_part(
            tile_index,
            u8::try_from(part_index).map_err(|_| {
                NativeEncodePipelineError::internal_invariant("tile-part index exceeds u8")
            })?,
            num_tile_parts,
            data_offset,
            packet_range,
            part_index.checked_mul(packet_limit).ok_or(
                NativeEncodePipelineError::arithmetic_overflow("tile-part packet range"),
            )?,
            &packetized,
            retained_base_bytes,
            source_bytes,
            &parts,
            session,
        )?;
        data_offset = data_offset.checked_add(part.data.len()).ok_or(
            NativeEncodePipelineError::arithmetic_overflow("packet length sum"),
        )?;
        parts.push(part);
    }
    drop(packetized);
    check_retained_parts(session, retained_base_bytes, &parts)?;
    Ok(parts)
}

fn consume_single_part(
    tile_index: u16,
    packetized: PacketizedTileData,
    retained_base_bytes: usize,
    session: &NativeEncodeSession<'_>,
) -> NativeEncodePipelineResult<Vec<EncodedTilePart>> {
    let source_bytes = packet_encode::packetized_tile_retained_bytes(&packetized)?;
    let requested = core::mem::size_of::<EncodedTilePart>();
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(source_bytes, requested, "multi-tile single part")?,
            "multi-tile single part",
        )?,
        "multi-tile single part",
    )?;
    let mut parts = Vec::new();
    parts
        .try_reserve_exact(1)
        .map_err(|_| host_allocation_failed("multi-tile single part", requested))?;
    reconcile_source_and_outer(
        session,
        retained_base_bytes,
        source_bytes,
        parts.capacity(),
        "multi-tile single part",
    )?;
    parts.push(EncodedTilePart {
        tile_index,
        tile_part_index: 0,
        num_tile_parts: 1,
        data: packetized.data,
        packet_lengths: packetized.packet_lengths,
        packet_headers: packetized.packet_headers,
    });
    check_retained_parts(session, retained_base_bytes, &parts)?;
    Ok(parts)
}

fn reconcile_source_and_outer(
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    source_bytes: usize,
    outer_capacity: usize,
    what: &'static str,
) -> NativeEncodePipelineResult<()> {
    let outer_bytes = checked_element_bytes::<EncodedTilePart>(outer_capacity, what)?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            checked_add_bytes(source_bytes, outer_bytes, what)?,
            what,
        )?,
        what,
    )?;
    Ok(())
}

fn validate_packetized_tile(packetized: &PacketizedTileData) -> NativeEncodePipelineResult<()> {
    if !packetized.packet_headers.is_empty()
        && packetized.packet_headers.len() != packetized.packet_lengths.len()
    {
        return Err(crate::EncodeError::InternalInvariant {
            what: "packet header count does not match packet length count",
        }
        .into());
    }
    if !packetized.packet_lengths.is_empty() {
        let expected_len = packetized
            .packet_lengths
            .iter()
            .try_fold(0usize, |acc, &len| {
                let len =
                    usize::try_from(len).map_err(|_| crate::EncodeError::ArithmeticOverflow {
                        what: "packet length exceeds host usize",
                    })?;
                acc.checked_add(len)
                    .ok_or(crate::EncodeError::ArithmeticOverflow {
                        what: "packet length sum",
                    })
            })?;
        if expected_len != packetized.data.len() {
            return Err(crate::EncodeError::InternalInvariant {
                what: "packet lengths do not match tile data length",
            }
            .into());
        }
    }
    Ok(())
}

fn check_retained_parts(
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    parts: &Vec<EncodedTilePart>,
) -> NativeEncodePipelineResult<()> {
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            encoded_tile_parts_retained_bytes(parts, parts.capacity())?,
            "multi-tile retained parts",
        )?,
        "multi-tile retained parts",
    )?;
    Ok(())
}

#[cfg(test)]
#[path = "consume/tests.rs"]
mod tests;
