// SPDX-License-Identifier: MIT OR Apache-2.0

//! Exact ownership accounting and fallible reservation for Tier-1 handoff.

use alloc::vec::Vec;

use super::allocation::{checked_add_bytes, checked_element_bytes, host_allocation_failed};
use super::{
    NativeEncodePipelineResult, NativeEncodeSession, PreparedCodeBlockCoefficients,
    PreparedEncodeCodeBlock, PreparedEncodeSubband, PreparedResolutionPacket,
};
use crate::j2c::bitplane_encode::EncodedCodeBlockWithSegments;
use crate::j2c::packet_encode::{CodeBlockPacketData, ResolutionPacket, SubbandPrecinct};
use crate::{EncodeResult, EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct PreparedOwnership {
    pub(super) structural: usize,
    pub(super) coefficients: usize,
    pub(super) preencoded: usize,
}

impl PreparedOwnership {
    pub(super) fn total(self) -> EncodeResult<usize> {
        checked_sum(
            [self.structural, self.coefficients, self.preencoded],
            "prepared Tier-1 ownership",
        )
    }

    fn add_structural<T>(&mut self, capacity: usize, what: &'static str) -> EncodeResult<()> {
        self.structural = checked_add_bytes(
            self.structural,
            checked_element_bytes::<T>(capacity, what)?,
            what,
        )?;
        Ok(())
    }

    fn add_coefficients(&mut self, block: &PreparedEncodeCodeBlock) -> EncodeResult<()> {
        let bytes = match &block.coefficients {
            PreparedCodeBlockCoefficients::I32(values) => checked_element_bytes::<i32>(
                values.capacity(),
                "prepared i32 code-block coefficients",
            )?,
            PreparedCodeBlockCoefficients::I64(values) => checked_element_bytes::<i64>(
                values.capacity(),
                "prepared i64 code-block coefficients",
            )?,
            PreparedCodeBlockCoefficients::Empty => 0,
        };
        self.coefficients =
            checked_add_bytes(self.coefficients, bytes, "prepared code-block coefficients")?;
        Ok(())
    }

    fn add_preencoded(&mut self, block: &EncodedHtJ2kCodeBlock) -> EncodeResult<()> {
        self.preencoded = checked_add_bytes(
            self.preencoded,
            block.data.capacity(),
            "preencoded HT code-block payload",
        )?;
        Ok(())
    }
}

pub(super) fn prepared_subbands_ownership(
    subbands: &[PreparedEncodeSubband],
    capacity: usize,
) -> EncodeResult<PreparedOwnership> {
    let mut ownership = PreparedOwnership::default();
    add_subbands(&mut ownership, subbands, capacity)?;
    Ok(ownership)
}

pub(super) fn prepared_packets_ownership(
    packets: &[PreparedResolutionPacket],
    capacity: usize,
) -> EncodeResult<PreparedOwnership> {
    let mut ownership = PreparedOwnership::default();
    ownership.add_structural::<PreparedResolutionPacket>(
        capacity,
        "prepared resolution packet owners",
    )?;
    for packet in packets {
        add_subbands(&mut ownership, &packet.subbands, packet.subbands.capacity())?;
    }
    Ok(ownership)
}

pub(super) fn prepared_packet_tree_ownership(
    packets: &[Vec<PreparedResolutionPacket>],
    capacity: usize,
) -> EncodeResult<PreparedOwnership> {
    let mut ownership = PreparedOwnership::default();
    ownership.add_structural::<Vec<PreparedResolutionPacket>>(
        capacity,
        "prepared component packet owners",
    )?;
    for component in packets {
        ownership.add_structural::<PreparedResolutionPacket>(
            component.capacity(),
            "prepared resolution packet owners",
        )?;
        for packet in component {
            add_subbands(&mut ownership, &packet.subbands, packet.subbands.capacity())?;
        }
    }
    Ok(ownership)
}

fn add_subbands(
    ownership: &mut PreparedOwnership,
    subbands: &[PreparedEncodeSubband],
    capacity: usize,
) -> EncodeResult<()> {
    ownership.add_structural::<PreparedEncodeSubband>(capacity, "prepared subband owners")?;
    for subband in subbands {
        ownership.add_structural::<PreparedEncodeCodeBlock>(
            subband.code_blocks.capacity(),
            "prepared code-block owners",
        )?;
        for block in &subband.code_blocks {
            ownership.add_coefficients(block)?;
        }
        if let Some(blocks) = &subband.preencoded_ht_code_blocks {
            ownership.add_structural::<EncodedHtJ2kCodeBlock>(
                blocks.capacity(),
                "preencoded HT code-block owners",
            )?;
            for block in blocks {
                ownership.add_preencoded(block)?;
            }
        }
    }
    Ok(())
}

pub(super) fn resolution_packet_ownership(
    packets: &[ResolutionPacket],
    capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes =
        checked_element_bytes::<ResolutionPacket>(capacity, "encoded resolution packet owners")?;
    for packet in packets {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<SubbandPrecinct>(
                packet.subbands.capacity(),
                "encoded subband owners",
            )?,
            "encoded subband owners",
        )?;
        for subband in &packet.subbands {
            bytes = checked_add_bytes(
                bytes,
                checked_element_bytes::<CodeBlockPacketData>(
                    subband.code_blocks.capacity(),
                    "encoded packet code-block owners",
                )?,
                "encoded packet code-block owners",
            )?;
            for block in &subband.code_blocks {
                bytes = checked_add_bytes(bytes, block.data.capacity(), "Tier-1 packet payload")?;
                bytes = checked_add_bytes(
                    bytes,
                    checked_element_bytes::<u32>(
                        block.classic_segment_lengths.capacity(),
                        "classic segment-length metadata",
                    )?,
                    "classic segment-length metadata",
                )?;
            }
        }
    }
    Ok(bytes)
}

pub(super) fn subband_precincts_ownership(
    subbands: &[SubbandPrecinct],
    capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<SubbandPrecinct>(capacity, "encoded subband owners")?;
    for subband in subbands {
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<CodeBlockPacketData>(
                subband.code_blocks.capacity(),
                "encoded packet code-block owners",
            )?,
            "encoded packet code-block owners",
        )?;
        for block in &subband.code_blocks {
            bytes = checked_add_bytes(bytes, block.data.capacity(), "Tier-1 packet payload")?;
            bytes = checked_add_bytes(
                bytes,
                checked_element_bytes::<u32>(
                    block.classic_segment_lengths.capacity(),
                    "classic segment-length metadata",
                )?,
                "classic segment-length metadata",
            )?;
        }
    }
    Ok(bytes)
}

pub(super) fn segmented_block_ownership(
    block: &EncodedCodeBlockWithSegments,
) -> EncodeResult<usize> {
    checked_sum(
        [
            block.data.capacity(),
            checked_element_bytes::<crate::j2c::bitplane_encode::EncodedCodeBlockSegment>(
                block.segments.capacity(),
                "classic Tier-1 segment metadata",
            )?,
        ],
        "segmented classic Tier-1 output",
    )
}

pub(super) fn public_ht_blocks_ownership(
    blocks: &[EncodedHtJ2kCodeBlock],
    capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<EncodedHtJ2kCodeBlock>(
        capacity,
        "accelerated HT Tier-1 result owners",
    )?;
    for block in blocks {
        bytes = checked_add_bytes(bytes, block.data.capacity(), "accelerated HT payload")?;
    }
    Ok(bytes)
}

pub(super) fn public_classic_blocks_ownership(
    blocks: &[EncodedJ2kCodeBlock],
    capacity: usize,
) -> EncodeResult<usize> {
    let mut bytes = checked_element_bytes::<EncodedJ2kCodeBlock>(
        capacity,
        "accelerated classic Tier-1 result owners",
    )?;
    for block in blocks {
        bytes = checked_add_bytes(bytes, block.data.capacity(), "accelerated classic payload")?;
        bytes = checked_add_bytes(
            bytes,
            checked_element_bytes::<crate::J2kCodeBlockSegment>(
                block.segments.capacity(),
                "accelerated classic segment metadata",
            )?,
            "accelerated classic segment metadata",
        )?;
    }
    Ok(bytes)
}

pub(super) struct Tier1PhaseTracker<'session, 'input> {
    session: &'session NativeEncodeSession<'input>,
    retained_base_bytes: usize,
    peak_phase_bytes: usize,
}

impl<'session, 'input> Tier1PhaseTracker<'session, 'input> {
    pub(super) fn new(
        session: &'session NativeEncodeSession<'input>,
        retained_base_bytes: usize,
    ) -> Self {
        Self {
            session,
            retained_base_bytes,
            peak_phase_bytes: retained_base_bytes,
        }
    }

    pub(super) fn check(
        &mut self,
        live_owner_bytes: impl IntoIterator<Item = usize>,
        what: &'static str,
    ) -> EncodeResult<usize> {
        let live = checked_add_bytes(
            self.retained_base_bytes,
            checked_sum(live_owner_bytes, what)?,
            what,
        )?;
        self.session.checked_phase(live, what)?;
        self.peak_phase_bytes = self.peak_phase_bytes.max(live);
        Ok(live)
    }

    pub(super) fn try_vec<T>(
        &mut self,
        count: usize,
        other_live_bytes: impl IntoIterator<Item = usize> + Clone,
        what: &'static str,
    ) -> NativeEncodePipelineResult<(Vec<T>, usize)> {
        let requested = checked_element_bytes::<T>(count, what)?;
        self.check(
            other_live_bytes.clone().into_iter().chain([requested]),
            what,
        )?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| host_allocation_failed(what, requested))?;
        let actual = checked_element_bytes::<T>(values.capacity(), what)?;
        self.check(other_live_bytes.into_iter().chain([actual]), what)?;
        Ok((values, actual))
    }

    pub(super) fn try_reserve_additional<T>(
        &mut self,
        values: &mut Vec<T>,
        additional: usize,
        other_live_bytes: impl IntoIterator<Item = usize> + Clone,
        what: &'static str,
    ) -> NativeEncodePipelineResult<usize> {
        let requested_capacity = values
            .len()
            .checked_add(additional)
            .ok_or(crate::EncodeError::ArithmeticOverflow { what })?;
        if requested_capacity <= values.capacity() {
            let actual = checked_element_bytes::<T>(values.capacity(), what)?;
            self.check(other_live_bytes.into_iter().chain([actual]), what)?;
            return Ok(actual);
        }
        let requested = checked_element_bytes::<T>(requested_capacity, what)?;
        self.check(
            other_live_bytes.clone().into_iter().chain([requested]),
            what,
        )?;
        values
            .try_reserve_exact(additional)
            .map_err(|_| host_allocation_failed(what, requested))?;
        let actual = checked_element_bytes::<T>(values.capacity(), what)?;
        self.check(other_live_bytes.into_iter().chain([actual]), what)?;
        Ok(actual)
    }

    #[cfg(test)]
    pub(super) const fn peak_phase_bytes(&self) -> usize {
        self.peak_phase_bytes
    }
}

pub(super) fn checked_sum(
    values: impl IntoIterator<Item = usize>,
    what: &'static str,
) -> EncodeResult<usize> {
    values
        .into_iter()
        .try_fold(0usize, |total, value| checked_add_bytes(total, value, what))
}
