// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact retained-byte accounting for layered packet owners.

use super::super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::super::tier1_allocation::segmented_block_ownership;
use super::super::super::{LayeredPreparedBlock, LayeredPreparedPacket, LayeredPreparedSubband};

pub(super) fn layered_block_build_owner_bytes(
    source_bytes: usize,
    layered_packets: &[LayeredPreparedPacket],
    layered_packet_capacity: usize,
    layered_packet: &LayeredPreparedPacket,
    layered_subband: &LayeredPreparedSubband,
) -> Result<usize, crate::EncodeError> {
    checked_sum(
        [
            source_bytes,
            layered_packets_ownership(layered_packets, layered_packet_capacity)?,
            layered_packet_ownership(layered_packet)?,
            layered_subband_ownership(layered_subband)?,
        ],
        "layered Tier-1 construction owners",
    )
}

pub(super) fn layered_packets_ownership(
    packets: &[LayeredPreparedPacket],
    capacity: usize,
) -> Result<usize, crate::EncodeError> {
    let mut bytes =
        checked_element_bytes::<LayeredPreparedPacket>(capacity, "layered packet owners")?;
    for packet in packets {
        bytes = checked_add_bytes(bytes, layered_packet_ownership(packet)?, "layered packets")?;
    }
    Ok(bytes)
}

pub(super) fn layered_packet_ownership(
    packet: &LayeredPreparedPacket,
) -> Result<usize, crate::EncodeError> {
    let mut bytes = checked_element_bytes::<LayeredPreparedSubband>(
        packet.subbands.capacity(),
        "layered subband owners",
    )?;
    for subband in &packet.subbands {
        bytes = checked_add_bytes(
            bytes,
            layered_subband_ownership(subband)?,
            "layered subband graph",
        )?;
    }
    Ok(bytes)
}

pub(super) fn layered_subband_ownership(
    subband: &LayeredPreparedSubband,
) -> Result<usize, crate::EncodeError> {
    let mut bytes = checked_element_bytes::<LayeredPreparedBlock>(
        subband.blocks.capacity(),
        "layered code-block owners",
    )?;
    for block in &subband.blocks {
        let block_bytes = match block {
            LayeredPreparedBlock::Classic {
                encoded,
                segment_layers,
            } => checked_add_bytes(
                segmented_block_ownership(encoded)?,
                checked_element_bytes::<usize>(
                    segment_layers.capacity(),
                    "classic segment-layer metadata",
                )?,
                "classic layered block",
            )?,
            LayeredPreparedBlock::HighThroughput {
                encoded,
                segment_layers,
            } => checked_add_bytes(
                encoded.data.capacity(),
                checked_element_bytes::<usize>(
                    segment_layers.capacity(),
                    "HT segment-layer metadata",
                )?,
                "HT layered block",
            )?,
        };
        bytes = checked_add_bytes(bytes, block_bytes, "layered code-block graph")?;
    }
    Ok(bytes)
}

pub(super) fn checked_sum(
    values: impl IntoIterator<Item = usize>,
    what: &'static str,
) -> Result<usize, crate::EncodeError> {
    values
        .into_iter()
        .try_fold(0usize, |total, value| checked_add_bytes(total, value, what))
}
