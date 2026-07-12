// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capacity accounting at packetization accelerator ownership boundaries.

use super::PacketizedTileData;
use crate::j2c::encode::allocation::{checked_add_bytes, checked_element_bytes};
use crate::{EncodeResult, J2kPacketizationResolution, J2kPacketizationSubband};

pub(crate) fn packet_metadata_retained_bytes(
    resolutions: &[J2kPacketizationResolution<'_>],
    resolution_capacity: usize,
    additional_retained_bytes: usize,
) -> EncodeResult<usize> {
    let mut bytes = add_capacity::<J2kPacketizationResolution<'_>>(
        additional_retained_bytes,
        resolution_capacity,
        "accelerator packet resolution metadata",
    )?;
    for resolution in resolutions {
        bytes = add_capacity::<J2kPacketizationSubband<'_>>(
            bytes,
            resolution.subbands.capacity(),
            "accelerator packet subband metadata",
        )?;
        for subband in &resolution.subbands {
            bytes = add_capacity::<crate::J2kPacketizationCodeBlock<'_>>(
                bytes,
                subband.code_blocks.capacity(),
                "accelerator packet code-block metadata",
            )?;
        }
    }
    Ok(bytes)
}

pub(crate) fn packetized_tile_retained_bytes(tile: &PacketizedTileData) -> EncodeResult<usize> {
    let mut bytes = tile.data.capacity();
    bytes = add_capacity::<u32>(
        bytes,
        tile.packet_lengths.capacity(),
        "packet length output",
    )?;
    bytes = add_capacity::<alloc::vec::Vec<u8>>(
        bytes,
        tile.packet_headers.capacity(),
        "packet header output vectors",
    )?;
    for header in &tile.packet_headers {
        bytes = checked_add_bytes(bytes, header.capacity(), "packet header output payloads")?;
    }
    Ok(bytes)
}

fn add_capacity<T>(bytes: usize, capacity: usize, what: &'static str) -> EncodeResult<usize> {
    checked_add_bytes(bytes, checked_element_bytes::<T>(capacity, what)?, what)
}
