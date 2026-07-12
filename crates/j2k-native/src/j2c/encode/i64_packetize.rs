// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::allocation::{checked_add_bytes, checked_element_bytes};
use super::options::EncodeOptions;
use super::{
    encode_prepared_resolution_packets_for_session,
    encode_prepared_resolution_packets_layered_for_session,
    ordered_prepared_resolution_packets_for_session, packet_descriptors_for_order_for_session,
    packetization_requires_scalar, packetize_resolution_packets_with_options_for_session,
    split_component_resolution_packets_by_precinct_for_session, NativeEncodePipelineResult,
    NativeEncodeSession, PreparedResolutionPacket,
};
use crate::j2c::codestream_write::EncodeParams;
use crate::j2c::packet_encode::{self, PacketizedTileData};
use crate::J2kEncodeStageAccelerator;

pub(super) struct I64PacketizeRequest<'a, 'input, A: J2kEncodeStageAccelerator> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_components: u16,
    pub(super) num_levels: u8,
    pub(super) params: &'a EncodeParams,
    pub(super) options: &'a EncodeOptions,
    pub(super) session: &'a NativeEncodeSession<'input>,
    pub(super) retained_base_bytes: usize,
    pub(super) accelerator: &'a mut A,
}

pub(super) fn packetize_i64_component_resolution_packets<A: J2kEncodeStageAccelerator>(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    request: I64PacketizeRequest<'_, '_, A>,
) -> NativeEncodePipelineResult<PacketizedTileData> {
    let I64PacketizeRequest {
        width,
        height,
        num_components,
        num_levels,
        params,
        options,
        session,
        retained_base_bytes,
        accelerator,
    } = request;
    let component_resolution_packets = split_component_resolution_packets_by_precinct_for_session(
        component_resolution_packets,
        width,
        height,
        num_levels,
        &params.precinct_exponents,
        session,
        retained_base_bytes,
    )?;
    let prepared_resolution_packets = ordered_prepared_resolution_packets_for_session(
        component_resolution_packets,
        options,
        session,
        retained_base_bytes,
    )?;
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered_for_session(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    session,
                    retained_base_bytes,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order_for_session(
                &prepared_resolution_packets,
                prepared_resolution_packets.capacity(),
                1,
                options.progression_order,
                session,
                retained_base_bytes,
            )?;
            let descriptor_bytes = checked_element_bytes::<super::J2kPacketizationPacketDescriptor>(
                packet_descriptors.capacity(),
                "packet descriptor owners",
            )?;
            let resolution_packets = encode_prepared_resolution_packets_for_session(
                prepared_resolution_packets,
                session,
                checked_add_bytes(
                    retained_base_bytes,
                    descriptor_bytes,
                    "Tier-1 packet descriptor baseline",
                )?,
                accelerator,
            )?;
            (resolution_packets, packet_descriptors, true)
        };

    let packet_phase_owners = (params, options);
    let packet_session = session.checked_child_session(
        &packet_phase_owners,
        retained_base_bytes,
        "retained typed-i64 packetization owners",
    )?;
    packetize_resolution_packets_with_options_for_session(
        &resolution_packets,
        resolution_packets.capacity(),
        &packet_descriptors,
        packet_descriptors.capacity(),
        options.num_layers,
        num_components,
        options.progression_order,
        packet_encode::PacketMarkerOptions {
            write_sop: params.write_sop,
            write_eph: params.write_eph,
            separate_packet_headers: params.write_ppm || params.write_ppt,
        },
        allow_packetization_accelerator,
        packetization_requires_scalar(params, options.tile_part_packet_limit),
        &packet_session,
        accelerator,
    )
}
