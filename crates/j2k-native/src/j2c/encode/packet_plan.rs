// SPDX-License-Identifier: MIT OR Apache-2.0

use super::tier1_allocation::{
    prepared_packet_tree_ownership, prepared_packets_ownership, Tier1PhaseTracker,
};
use super::{
    packet_encode, BlockCodingMode, EncodeOptions, EncodeParams, EncodeProgressionOrder,
    J2kEncodeStageAccelerator, J2kPacketizationBlockCodingMode, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, NativeEncodePhase, NativeEncodePipelineError,
    NativeEncodePipelineResult, NativeEncodeSession, PreparedCompactResolutionPacket,
    PreparedResolutionPacket, ResolutionPacket, Vec,
};

mod accelerator_metadata;
use accelerator_metadata::try_public_packetization_resolutions;
mod precinct;
pub(super) use precinct::split_component_resolution_packets_by_precinct_for_session;

pub(super) fn count_code_blocks(
    resolution_packets: &[ResolutionPacket],
) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

pub(super) fn count_compact_code_blocks(
    resolution_packets: &[PreparedCompactResolutionPacket<'_>],
) -> Result<u32, &'static str> {
    let count = resolution_packets
        .iter()
        .flat_map(|resolution| resolution.subbands.iter())
        .try_fold(0usize, |acc, subband| {
            acc.checked_add(subband.code_blocks.len())
                .ok_or("packetization code-block count overflow")
        })?;
    u32::try_from(count).map_err(|_| "packetization code-block count exceeds u32")
}

pub(super) fn packet_descriptors_for_order_for_session(
    packets: &[PreparedResolutionPacket],
    packet_capacity: usize,
    num_layers: u8,
    progression_order: EncodeProgressionOrder,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Vec<J2kPacketizationPacketDescriptor>> {
    if num_layers != 1 {
        return Err(NativeEncodePipelineError::invalid_input(
            "encode currently prepares one packet contribution layer",
        ));
    }
    if packet_capacity < packets.len() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "prepared packet capacity is smaller than its length",
        }
        .into());
    }
    let prepared_bytes = prepared_packets_ownership(packets, packet_capacity)?.total()?;
    let mut tracker = Tier1PhaseTracker::new(session, retained_base_bytes);
    let (mut descriptors, _) = tracker.try_vec::<J2kPacketizationPacketDescriptor>(
        packets.len(),
        [prepared_bytes],
        "packet descriptor owners",
    )?;
    for (packet_index, packet) in packets.iter().enumerate() {
        descriptors.push(J2kPacketizationPacketDescriptor {
            packet_index: u32::try_from(packet_index).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "packet descriptor index exceeds u32",
                )
            })?,
            state_index: u32::try_from(packet_index).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "packet descriptor state index exceeds u32",
                )
            })?,
            layer: 0,
            resolution: packet.resolution,
            component: packet.component,
            precinct: packet.precinct,
        });
    }
    crate::sort_packet_descriptors_for_progression(
        &mut descriptors,
        progression_order.packetization_order(),
    );
    Ok(descriptors)
}

pub(super) fn ordered_prepared_resolution_packets_for_session(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    options: &EncodeOptions,
    session: &NativeEncodeSession<'_>,
    retained_base_bytes: usize,
) -> NativeEncodePipelineResult<Vec<PreparedResolutionPacket>> {
    let source = prepared_packet_tree_ownership(
        &component_resolution_packets,
        component_resolution_packets.capacity(),
    )?;
    let source_bytes = source.total()?;
    let packet_count =
        component_resolution_packets
            .iter()
            .try_fold(0usize, |count, component| {
                count
                    .checked_add(component.len())
                    .ok_or(crate::EncodeError::ArithmeticOverflow {
                        what: "ordered prepared packet count",
                    })
            })?;
    let mut tracker = Tier1PhaseTracker::new(session, retained_base_bytes);
    let (mut packets, _) = tracker.try_vec::<PreparedResolutionPacket>(
        packet_count,
        [source_bytes],
        "ordered prepared packet owners",
    )?;
    for component in component_resolution_packets {
        packets.extend(component);
    }
    match options.progression_order {
        EncodeProgressionOrder::Lrcp
        | EncodeProgressionOrder::Rlcp
        | EncodeProgressionOrder::Rpcl => {
            packets.sort_by_key(|packet| (packet.resolution, packet.component, packet.precinct));
        }
        EncodeProgressionOrder::Pcrl | EncodeProgressionOrder::Cprl => {
            packets.sort_by_key(|packet| (packet.component, packet.resolution, packet.precinct));
        }
    }
    Ok(packets)
}

pub(super) fn public_packetization_progression_order(
    progression_order: EncodeProgressionOrder,
) -> crate::J2kPacketizationProgressionOrder {
    progression_order.packetization_order()
}

#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(super) fn packetize_resolution_packets_with_options_for_session(
    resolution_packets: &[ResolutionPacket],
    resolution_packet_capacity: usize,
    packet_descriptors: &[J2kPacketizationPacketDescriptor],
    packet_descriptor_capacity: usize,
    num_layers: u8,
    num_components: u16,
    progression_order: EncodeProgressionOrder,
    marker_options: packet_encode::PacketMarkerOptions,
    allow_packetization_accelerator: bool,
    force_scalar_packetization: bool,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<packet_encode::PacketizedTileData> {
    if resolution_packet_capacity < resolution_packets.len() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "resolution packet capacity is smaller than its length",
        }
        .into());
    }
    if packet_descriptor_capacity < packet_descriptors.len() {
        return Err(crate::EncodeError::InternalInvariant {
            what: "packet descriptor capacity is smaller than its length",
        }
        .into());
    }
    let owned_packet_bytes = packet_encode::owned_packet_retained_bytes_for_public_descriptors(
        resolution_packets,
        resolution_packet_capacity,
        packet_descriptor_capacity,
        0,
    )?;
    if allow_packetization_accelerator && !force_scalar_packetization {
        let packetization_resolutions =
            try_public_packetization_resolutions(resolution_packets, session, owned_packet_bytes)?;
        let accelerator_phase_bytes = packet_encode::packet_metadata_retained_bytes(
            &packetization_resolutions,
            packetization_resolutions.capacity(),
            owned_packet_bytes,
        )?;
        let accelerator_phase = session.checked_phase(
            accelerator_phase_bytes,
            "retained packet owners and accelerator metadata",
        )?;
        let packetization_job = J2kPacketizationEncodeJob {
            resolution_count: u32::try_from(resolution_packets.len()).map_err(|_| {
                NativeEncodePipelineError::arithmetic_overflow(
                    "packetization resolution count exceeds u32",
                )
            })?,
            num_layers,
            num_components,
            code_block_count: count_code_blocks(resolution_packets)
                .map_err(NativeEncodePipelineError::arithmetic_overflow)?,
            progression_order: public_packetization_progression_order(progression_order),
            packet_descriptors,
            resolutions: &packetization_resolutions,
        };
        if let Some(packetized) =
            try_packetization_accelerator(packetization_job, &accelerator_phase, accelerator)?
        {
            return Ok(packetized);
        }
    }

    let retained_packet_bytes = session.checked_phase_retained_bytes(
        owned_packet_bytes,
        "retained native encode inputs and packet ownership",
    )?;
    Ok(
        packet_encode::form_tile_bitstream_with_public_descriptors_and_retained_baseline(
            resolution_packets,
            packet_descriptors,
            marker_options,
            retained_packet_bytes,
        )?,
    )
}

fn try_packetization_accelerator(
    job: J2kPacketizationEncodeJob<'_>,
    phase: &NativeEncodePhase<'_, '_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<Option<packet_encode::PacketizedTileData>> {
    let Some(data) = accelerator.encode_packetization(job).map_err(|source| {
        crate::EncodeError::Accelerator {
            operation: "packetization",
            source,
        }
    })?
    else {
        return Ok(None);
    };
    let packetized = packet_encode::PacketizedTileData {
        data,
        packet_lengths: Vec::new(),
        packet_headers: Vec::new(),
    };
    phase.reconcile_accelerator_output_bytes(
        packet_encode::packetized_tile_retained_bytes(&packetized)?,
        "accelerator packetization output",
    )?;
    Ok(Some(packetized))
}

pub(super) fn packetization_requires_scalar(
    params: &EncodeParams,
    tile_part_packet_limit: Option<u16>,
) -> bool {
    params.write_plt
        || params.write_plm
        || params.write_ppm
        || params.write_ppt
        || params.write_sop
        || params.write_eph
        || tile_part_packet_limit.is_some()
}

pub(super) fn public_packetization_block_coding_mode(
    block_coding_mode: BlockCodingMode,
) -> J2kPacketizationBlockCodingMode {
    match block_coding_mode {
        BlockCodingMode::Classic => J2kPacketizationBlockCodingMode::Classic,
        BlockCodingMode::HighThroughput => J2kPacketizationBlockCodingMode::HighThroughput,
    }
}

#[cfg(test)]
mod tests;
