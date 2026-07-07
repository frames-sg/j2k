// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::options::EncodeOptions;
use super::tile_parts::write_single_tile_packetized_codestream;
use super::{
    encode_prepared_resolution_packets, encode_prepared_resolution_packets_layered,
    ordered_prepared_resolution_packets, packet_descriptors_for_order,
    packetization_requires_scalar, packetize_resolution_packets_with_options,
    split_component_resolution_packets_by_precinct, PreparedResolutionPacket,
};
use crate::j2c::codestream_write::EncodeParams;
use crate::j2c::packet_encode::{self, PacketizedTileData};
use crate::J2kEncodeStageAccelerator;

pub(super) struct I64PacketizeRequest<'a, A: J2kEncodeStageAccelerator> {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) num_components: u16,
    pub(super) num_levels: u8,
    pub(super) params: &'a EncodeParams,
    pub(super) options: &'a EncodeOptions,
    pub(super) accelerator: &'a mut A,
}

pub(super) struct I64CodestreamPacketRequest<'a, A: J2kEncodeStageAccelerator> {
    pub(super) packetize: I64PacketizeRequest<'a, A>,
    pub(super) quant_params: &'a [(u16, u16)],
}

pub(super) fn encode_i64_component_resolution_packets<A: J2kEncodeStageAccelerator>(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    request: I64CodestreamPacketRequest<'_, A>,
) -> Result<Vec<u8>, &'static str> {
    let params = request.packetize.params;
    let tile_part_packet_limit = request.packetize.options.tile_part_packet_limit;
    let packetized_tile = packetize_i64_component_resolution_packets(
        component_resolution_packets,
        request.packetize,
    )?;

    write_single_tile_packetized_codestream(
        params,
        &packetized_tile,
        request.quant_params,
        tile_part_packet_limit,
    )
}

pub(super) fn packetize_i64_component_resolution_packets<A: J2kEncodeStageAccelerator>(
    component_resolution_packets: Vec<Vec<PreparedResolutionPacket>>,
    request: I64PacketizeRequest<'_, A>,
) -> Result<PacketizedTileData, &'static str> {
    let I64PacketizeRequest {
        width,
        height,
        num_components,
        num_levels,
        params,
        options,
        accelerator,
    } = request;
    let component_resolution_packets = split_component_resolution_packets_by_precinct(
        component_resolution_packets,
        width,
        height,
        num_levels,
        &params.precinct_exponents,
    )?;
    let prepared_resolution_packets =
        ordered_prepared_resolution_packets(component_resolution_packets, options)?;
    let (resolution_packets, packet_descriptors, allow_packetization_accelerator) =
        if options.num_layers > 1 {
            let (resolution_packets, packet_descriptors) =
                encode_prepared_resolution_packets_layered(
                    prepared_resolution_packets,
                    options.num_layers,
                    options.progression_order,
                    &options.quality_layer_byte_targets,
                    accelerator,
                )?;
            (resolution_packets, packet_descriptors, false)
        } else {
            let packet_descriptors = packet_descriptors_for_order(
                &prepared_resolution_packets,
                1,
                options.progression_order,
            )?;
            let resolution_packets =
                encode_prepared_resolution_packets(prepared_resolution_packets, accelerator)?;
            (resolution_packets, packet_descriptors, true)
        };

    let mut resolution_packets = resolution_packets;
    packetize_resolution_packets_with_options(
        &mut resolution_packets,
        &packet_descriptors,
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
        accelerator,
    )
}
