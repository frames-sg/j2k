// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared packet orchestration for single- and multi-layer Tier-1 output.

use super::allocation::checked_add_bytes;
use super::tier1_allocation::{
    prepared_packets_ownership, subband_precincts_ownership, Tier1PhaseTracker,
};
use super::{
    encode_prepared_subbands_for_session, J2kEncodeStageAccelerator, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedResolutionPacket, ResolutionPacket,
    SubbandPrecinct, Vec,
};

mod layered;

pub(super) use layered::encode_prepared_resolution_packets_layered_for_session;

pub(super) fn encode_prepared_resolution_packets_for_session(
    prepared_packets: Vec<PreparedResolutionPacket>,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Vec<ResolutionPacket>> {
    let source = prepared_packets_ownership(&prepared_packets, prepared_packets.capacity())?;
    let source_bytes = source.total()?;
    let packet_count = prepared_packets.len();
    let subband_count = prepared_packets.iter().try_fold(0usize, |count, packet| {
        count
            .checked_add(packet.subbands.len())
            .ok_or(crate::EncodeError::ArithmeticOverflow {
                what: "prepared Tier-1 subband count",
            })
    })?;
    let mut tracker = Tier1PhaseTracker::new(session, retained_base_bytes);
    let (mut subband_counts, subband_count_bytes) = tracker.try_vec::<usize>(
        packet_count,
        [source_bytes],
        "prepared packet subband counts",
    )?;
    let (mut prepared_subbands, _) = tracker.try_vec::<super::PreparedEncodeSubband>(
        subband_count,
        [source_bytes, subband_count_bytes],
        "flattened prepared Tier-1 subband owners",
    )?;
    for packet in prepared_packets {
        subband_counts.push(packet.subbands.len());
        prepared_subbands.extend(packet.subbands);
    }

    let tier1_retained = checked_add_bytes(
        retained_base_bytes,
        subband_count_bytes,
        "Tier-1 packet shape baseline",
    )?;
    let encoded_subbands = encode_prepared_subbands_for_session(
        prepared_subbands,
        session,
        tier1_retained,
        accelerator,
    )?;
    let encoded_bytes =
        subband_precincts_ownership(&encoded_subbands, encoded_subbands.capacity())?;
    let (mut resolution_packets, resolution_owner_bytes) = tracker.try_vec::<ResolutionPacket>(
        packet_count,
        [subband_count_bytes, encoded_bytes],
        "encoded resolution packet owners",
    )?;
    let mut rebuilt_structural_bytes = resolution_owner_bytes;
    let mut encoded_subbands = encoded_subbands.into_iter();
    for subband_count in subband_counts {
        let (mut subbands, subband_owner_bytes) = tracker.try_vec::<SubbandPrecinct>(
            subband_count,
            [subband_count_bytes, encoded_bytes, rebuilt_structural_bytes],
            "rebuilt encoded subband owners",
        )?;
        rebuilt_structural_bytes = checked_add_bytes(
            rebuilt_structural_bytes,
            subband_owner_bytes,
            "rebuilt encoded packet structure",
        )?;
        for _ in 0..subband_count {
            subbands.push(encoded_subbands.next().ok_or_else(|| {
                NativeEncodePipelineError::internal_invariant("encoded subband count mismatch")
            })?);
        }
        resolution_packets.push(ResolutionPacket { subbands });
    }
    if encoded_subbands.next().is_some() {
        return Err(NativeEncodePipelineError::internal_invariant(
            "encoded subband count mismatch",
        ));
    }
    Ok(resolution_packets)
}
