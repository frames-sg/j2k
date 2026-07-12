// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use super::{PacketDescriptor, ResolutionPacket};
use crate::j2c::encode::allocation::{checked_add_bytes, checked_mul_bytes};
use crate::{EncodeResult, J2kPacketizationPacketDescriptor, J2kPacketizationResolution};

pub(crate) fn owned_packet_retained_bytes(
    resolutions: &[ResolutionPacket],
    resolution_capacity: usize,
    descriptor_capacity: usize,
    additional_retained_bytes: usize,
) -> EncodeResult<usize> {
    let mut bytes = additional_retained_bytes;
    bytes = add_capacity::<ResolutionPacket>(
        bytes,
        resolution_capacity,
        "retained packet resolution capacity",
    )?;
    bytes = add_capacity::<PacketDescriptor>(
        bytes,
        descriptor_capacity,
        "retained packet descriptor capacity",
    )?;
    for resolution in resolutions {
        bytes = add_capacity::<super::SubbandPrecinct>(
            bytes,
            resolution.subbands.capacity(),
            "retained packet subband capacity",
        )?;
        for subband in &resolution.subbands {
            bytes = add_capacity::<super::CodeBlockPacketData>(
                bytes,
                subband.code_blocks.capacity(),
                "retained packet code-block capacity",
            )?;
            for code_block in &subband.code_blocks {
                bytes = checked_add_bytes(
                    bytes,
                    code_block.data.capacity(),
                    "retained packet payload capacity",
                )?;
                bytes = add_capacity::<u32>(
                    bytes,
                    code_block.classic_segment_lengths.capacity(),
                    "retained classic segment capacity",
                )?;
            }
        }
    }
    Ok(bytes)
}

pub(crate) fn owned_packet_retained_bytes_for_public_descriptors(
    resolutions: &[ResolutionPacket],
    resolution_capacity: usize,
    descriptor_capacity: usize,
    additional_retained_bytes: usize,
) -> EncodeResult<usize> {
    let bytes = owned_packet_retained_bytes(
        resolutions,
        resolution_capacity,
        0,
        additional_retained_bytes,
    )?;
    add_capacity::<J2kPacketizationPacketDescriptor>(
        bytes,
        descriptor_capacity,
        "retained public packet descriptor capacity",
    )
}

/// Count scalar adapter metadata while excluding genuinely caller-owned
/// borrowed code-block payload bytes.
pub(crate) fn borrowed_scalar_retained_bytes(
    resolutions: &[J2kPacketizationResolution<'_>],
    descriptors: &[J2kPacketizationPacketDescriptor],
    additional_retained_bytes: usize,
) -> EncodeResult<usize> {
    let mut bytes = additional_retained_bytes;
    bytes = add_capacity::<J2kPacketizationResolution<'_>>(
        bytes,
        resolutions.len(),
        "borrowed packet resolution metadata",
    )?;
    bytes = add_capacity::<J2kPacketizationPacketDescriptor>(
        bytes,
        descriptors.len(),
        "borrowed packet descriptor metadata",
    )?;
    for resolution in resolutions {
        bytes = add_capacity::<crate::J2kPacketizationSubband<'_>>(
            bytes,
            resolution.subbands.capacity(),
            "borrowed packet subband metadata",
        )?;
        for subband in &resolution.subbands {
            bytes = add_capacity::<crate::J2kPacketizationCodeBlock<'_>>(
                bytes,
                subband.code_blocks.capacity(),
                "borrowed packet code-block metadata",
            )?;
        }
    }
    Ok(bytes)
}

fn add_capacity<T>(bytes: usize, capacity: usize, what: &'static str) -> EncodeResult<usize> {
    checked_add_bytes(
        bytes,
        checked_mul_bytes(capacity, size_of::<T>(), what)?,
        what,
    )
}
