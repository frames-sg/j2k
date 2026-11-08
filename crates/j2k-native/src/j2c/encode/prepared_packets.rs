// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared packet orchestration through Tier-1 encoding and tile packetization.

use super::allocation::{checked_add_bytes, checked_element_bytes};
use super::magnitude::required_ht_magnitude_bound;
use super::tier1_allocation::{
    prepared_packet_tree_ownership, prepared_packets_ownership, subband_precincts_ownership,
    Tier1PhaseTracker,
};
use super::{
    encode_prepared_subbands_for_session, ordered_prepared_resolution_packets_for_session,
    packet_descriptors_for_session, packet_encode, packetization_requires_scalar,
    packetize_resolution_packets_with_options_for_session, profile,
    split_component_resolution_packets_by_precinct_for_session, EncodeOptions, EncodeParams,
    J2kEncodeStageAccelerator, J2kPacketizationPacketDescriptor, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedResolutionPacket, ResolutionPacket,
    SubbandPrecinct, Vec,
};

mod layered;

pub(super) use layered::encode_prepared_resolution_packets_layered_for_session;

/// Geometry and live owners accompanying a component packet tree. The owner
/// witness borrows everything counted by `retained_base_bytes`; the session
/// separately retains the original input graph.
pub(super) struct TilePacketRequest<'a, 'input, A: J2kEncodeStageAccelerator, O: ?Sized> {
    pub(super) tile_bounds: j2k_types::J2kRect,
    pub(super) num_levels: u8,
    pub(super) params: &'a EncodeParams,
    pub(super) options: &'a EncodeOptions,
    pub(super) retained_base_bytes: usize,
    pub(super) phase_owners: &'a O,
    pub(super) retained_what: &'static str,
    pub(super) profile_enabled: bool,
    pub(super) session: &'a NativeEncodeSession<'input>,
    pub(super) accelerator: &'a mut A,
}

pub(super) struct EncodedTilePacketization {
    pub(super) packetized_tile: packet_encode::PacketizedTileData,
    pub(super) block_encode_us: u128,
    pub(super) packetize_us: u128,
}

#[expect(
    clippy::too_many_lines,
    reason = "the shared packet state machine keeps ordering, phase ownership, layer selection, and final packetization together"
)]
pub(super) fn encode_component_resolution_packets<A: J2kEncodeStageAccelerator, O: ?Sized>(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    request: TilePacketRequest<'_, '_, A, O>,
) -> NativeEncodePipelineResult<EncodedTilePacketization> {
    let TilePacketRequest {
        tile_bounds,
        num_levels,
        params,
        options,
        retained_base_bytes,
        phase_owners,
        retained_what,
        profile_enabled,
        session,
        accelerator,
    } = request;
    let required_ht_magnitude_bound = required_ht_magnitude_bound(
        component_resolution_packets.iter().flatten(),
        num_levels,
        options.reversible,
        accelerator.prefer_parallel_cpu_code_block_fallback(),
    )?;
    let component_resolution_packets = split_component_resolution_packets_by_precinct_for_session(
        component_resolution_packets,
        tile_bounds,
        &params.component_sampling,
        num_levels,
        &params.precinct_exponents,
        session,
        retained_base_bytes,
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            prepared_packet_tree_ownership(
                &component_resolution_packets,
                component_resolution_packets.capacity(),
            )?
            .total()?,
            "precinct-split prepared packet tree",
        )?,
        "precinct-split prepared packet tree",
    )?;
    let prepared_resolution_packets = ordered_prepared_resolution_packets_for_session(
        component_resolution_packets,
        options,
        session,
        retained_base_bytes,
    )?;
    session.checked_phase(
        checked_add_bytes(
            retained_base_bytes,
            prepared_packets_ownership(
                &prepared_resolution_packets,
                prepared_resolution_packets.capacity(),
            )?
            .total()?,
            "ordered prepared packet tree",
        )?,
        "ordered prepared packet tree",
    )?;
    let stage_start = profile::profile_now(profile_enabled);
    let (resolution_packets, mut packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (packets, descriptors) = encode_prepared_resolution_packets_layered_for_session(
                prepared_resolution_packets,
                options.num_layers,
                &options.quality_layer_byte_targets,
                session,
                retained_base_bytes,
                accelerator,
            )?;
            (packets, descriptors, false)
        } else {
            let descriptors = packet_descriptors_for_session(
                &prepared_resolution_packets,
                prepared_resolution_packets.capacity(),
                session,
                retained_base_bytes,
            )?;
            let descriptor_bytes = checked_element_bytes::<J2kPacketizationPacketDescriptor>(
                descriptors.capacity(),
                "packet descriptor owners",
            )?;
            let packets = encode_prepared_resolution_packets_for_session(
                prepared_resolution_packets,
                session,
                checked_add_bytes(
                    retained_base_bytes,
                    descriptor_bytes,
                    "Tier-1 packet descriptor baseline",
                )?,
                accelerator,
            )?;
            (packets, descriptors, true)
        };
    let block_encode_us = profile::elapsed_us(stage_start);
    let stage_start = profile::profile_now(profile_enabled);
    let packet_session =
        session.checked_child_session(phase_owners, retained_base_bytes, retained_what)?;
    let descriptor_capacity = packet_descriptors.capacity();
    let mut packetized_tile = packetize_resolution_packets_with_options_for_session(
        &resolution_packets,
        resolution_packets.capacity(),
        &mut packet_descriptors,
        descriptor_capacity,
        params,
        tile_bounds,
        packet_encode::PacketMarkerOptions {
            write_sop: params.write_sop,
            write_eph: params.write_eph,
            separate_packet_headers: params.write_ppm || params.write_ppt,
        },
        allow_packetization_accelerator,
        packetization_requires_scalar(params, options.tile_part_packet_limit),
        &packet_session,
        accelerator,
    )?;
    packetized_tile.required_ht_magnitude_bound = required_ht_magnitude_bound;
    let packetize_us = profile::elapsed_us(stage_start);
    Ok(EncodedTilePacketization {
        packetized_tile,
        block_encode_us,
        packetize_us,
    })
}

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
